from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepuplift.models.BaseModel import representation_balance_loss  # noqa: E402
from deepuplift.models.CFRNet import CFRNet  # noqa: E402
from deepuplift.models.DESCN import ESX, esx_loss  # noqa: E402
from deepuplift.models.DragonNet import DragonNet  # noqa: E402
from deepuplift.models.EFIN import EFIN  # noqa: E402


CONTRACT_IDS = {
    "CFRNet": ["CFRNet.phi_x_representation"],
    "DragonNet": ["DragonNet.e_hat_propensity", "DragonNet.targeted_regularization_components"],
    "EFIN": ["EFIN.interaction_attention", "EFIN.uplift_path_contribution"],
    "DESCN": ["DESCN.propensity_and_exposure", "DESCN.entire_space_heads", "DESCN.cross_head_constraints"],
}


def clean_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().float().cpu()
    return {
        "shape": list(value.shape),
        "mean": clean_float(value.mean().item()),
        "std": clean_float(value.std(unbiased=False).item()) if value.numel() > 1 else 0.0,
        "min": clean_float(value.min().item()),
        "max": clean_float(value.max().item()),
        "finite": bool(torch.isfinite(value).all().item()),
    }


def write_frame(path: Path, frame: pd.DataFrame) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return str(path)


def write_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def synthetic_batch(batch: int = 48, features: int = 8) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(20260521)
    x = torch.randn(batch, features)
    t = torch.cat([torch.zeros(batch // 2, 1), torch.ones(batch - batch // 2, 1)], dim=0)
    logits = 0.4 * x[:, :1] - 0.2 * x[:, 1:2] + 0.35 * t
    y = torch.bernoulli(torch.sigmoid(logits)).float()
    return x, t, y


def row_status(required_artifacts: list[str], checks: dict[str, Any]) -> str:
    finite_checks = [value for key, value in checks.items() if key.endswith("_finite") or key.endswith("_pass")]
    if required_artifacts and finite_checks and all(bool(value) for value in finite_checks):
        return "ok"
    return "review"


def cfrnet_rows(root: Path, out_dir: Path) -> list[dict[str, Any]]:
    x, t, y = synthetic_batch()
    model = CFRNet(input_dim=x.shape[1], share_dim=6, task="classification")
    model.eval()
    with torch.no_grad():
        _t_pred, y_preds, phi_x, _eps = model(x, t)
        ipm = representation_balance_loss(phi_x, t, mode="mmd_rbf")
        mean_l2 = representation_balance_loss(phi_x, t, mode="mean_l2")
    artifact_dir = out_dir / "CFRNet"
    sample = pd.DataFrame(phi_x.detach().cpu().numpy()).add_prefix("phi_")
    sample.insert(0, "treatment", t.squeeze(1).detach().cpu().numpy())
    artifacts = [
        write_frame(artifact_dir / "representation_samples.csv", sample),
        write_json(
            artifact_dir / "representation_balance_curve.json",
            {
                "mmd_rbf": clean_float(ipm.item()),
                "mean_l2": clean_float(mean_l2.item()),
                "phi_stats": tensor_stats(phi_x),
                "treated_rows": int(t.sum().item()),
                "control_rows": int((1 - t).sum().item()),
            },
        ),
    ]
    checks = {
        "phi_finite": bool(torch.isfinite(phi_x).all().item()),
        "mmd_finite": clean_float(ipm.item()) is not None,
    }
    return [
        {
            "contract_id": "CFRNet.phi_x_representation",
            "model": "CFRNet",
            "hook_name": "phi_x representation export",
            "status": row_status(artifacts, checks),
            "tensor_keys_exported": ["phi_x", "t_true", "y0_pred", "y1_pred"],
            "artifact_files": artifacts,
            "metric_summary": {
                "mmd_rbf": clean_float(ipm.item()),
                "mean_l2": clean_float(mean_l2.item()),
                "phi_std": tensor_stats(phi_x).get("std"),
            },
            "checks": checks,
            "pass_gate": "phi_x export exists and representation distance metrics are finite.",
            "remaining_gap": "Needs per-epoch curve and PEHE/QINI linkage before paper-level balance claim.",
            "safe_claim": "CFRNet phi_x is now smoke-exported without changing the default forward API.",
            "interview_line": "CFRNet forward hook smoke exports phi_x and finite MMD/mean-L2 balance metrics; next step is per-epoch and benchmark linkage.",
        }
    ]


def dragonnet_rows(root: Path, out_dir: Path) -> list[dict[str, Any]]:
    x, t, y = synthetic_batch()
    model = DragonNet(input_dim=x.shape[1], share_dim=6, task="classification")
    model.eval()
    with torch.no_grad():
        e_hat, y_preds, _phi_x, eps = model(x, t)
        outcome_loss = F.binary_cross_entropy(y_preds[0], y, reduction="mean") + F.binary_cross_entropy(
            y_preds[1], y, reduction="mean"
        )
        treatment_loss = F.binary_cross_entropy(e_hat, t, reduction="mean")
        y_pred = t * y_preds[1] + (1 - t) * y_preds[0]
        e_smooth = (e_hat + 0.01) / 1.02
        clever_covariate = (t / e_smooth) - ((1 - t) / (1 - e_smooth))
        y_pert = y_pred + eps * clever_covariate
        tarreg = torch.mean((y - y_pert).pow(2))
    artifact_dir = out_dir / "DragonNet"
    artifacts_prop = [
        write_frame(
            artifact_dir / "propensity_predictions.csv",
            pd.DataFrame(
                {
                    "row_id": np.arange(len(x)),
                    "t_true": t.squeeze(1).detach().cpu().numpy(),
                    "e_hat": e_hat.squeeze(1).detach().cpu().numpy(),
                }
            ),
        ),
        write_json(
            artifact_dir / "propensity_calibration_bins.json",
            {
                "e_hat_stats": tensor_stats(e_hat),
                "treatment_rate": clean_float(t.mean().item()),
                "e_hat_mean": clean_float(e_hat.mean().item()),
            },
        ),
    ]
    artifacts_tarreg = [
        write_frame(
            artifact_dir / "dragonnet_loss_components.csv",
            pd.DataFrame(
                [
                    {"component": "outcome_loss", "value": clean_float(outcome_loss.item())},
                    {"component": "treatment_loss", "value": clean_float(treatment_loss.item())},
                    {"component": "targeted_regularization", "value": clean_float(tarreg.item())},
                    {"component": "epsilon_mean", "value": clean_float(eps.mean().item())},
                ]
            ),
        ),
        write_json(
            artifact_dir / "tarreg_curve.json",
            {
                "targeted_regularization": clean_float(tarreg.item()),
                "epsilon_stats": tensor_stats(eps),
                "clever_covariate_stats": tensor_stats(clever_covariate),
            },
        ),
    ]
    prop_checks = {
        "e_hat_finite": bool(torch.isfinite(e_hat).all().item()),
        "e_hat_range_pass": bool(((e_hat > 0) & (e_hat < 1)).all().item()),
    }
    tarreg_checks = {
        "tarreg_finite": clean_float(tarreg.item()) is not None,
        "epsilon_finite": bool(torch.isfinite(eps).all().item()),
    }
    return [
        {
            "contract_id": "DragonNet.e_hat_propensity",
            "model": "DragonNet",
            "hook_name": "e_hat propensity export",
            "status": row_status(artifacts_prop, prop_checks),
            "tensor_keys_exported": ["e_hat", "t_true"],
            "artifact_files": artifacts_prop,
            "metric_summary": {
                "e_hat_mean": clean_float(e_hat.mean().item()),
                "e_hat_std": tensor_stats(e_hat).get("std"),
                "treatment_rate": clean_float(t.mean().item()),
            },
            "checks": prop_checks,
            "pass_gate": "e_hat is finite and in (0, 1); calibration/ESS linkage remains next.",
            "remaining_gap": "Needs propensity calibration bins, ESS clipping sensitivity and PEHE/QINI tail attribution.",
            "safe_claim": "DragonNet e_hat can be smoke-exported from the inherited TarNet forward path.",
            "interview_line": "DragonNet propensity smoke exports e_hat and treatment labels; next step is overlap/ESS diagnostics.",
        },
        {
            "contract_id": "DragonNet.targeted_regularization_components",
            "model": "DragonNet",
            "hook_name": "targeted regularization loss split",
            "status": row_status(artifacts_tarreg, tarreg_checks),
            "tensor_keys_exported": ["outcome_loss", "treatment_loss", "targeted_regularization", "epsilon"],
            "artifact_files": artifacts_tarreg,
            "metric_summary": {
                "outcome_loss": clean_float(outcome_loss.item()),
                "treatment_loss": clean_float(treatment_loss.item()),
                "targeted_regularization": clean_float(tarreg.item()),
            },
            "checks": tarreg_checks,
            "pass_gate": "tarreg and epsilon are finite in a guarded smoke batch.",
            "remaining_gap": "Needs optional loss-component return path inside BaseLoss plus beta ablation.",
            "safe_claim": "Targeted regularization can be separated in smoke evidence; production trainer still needs guarded component logging.",
            "interview_line": "DragonNet tarreg smoke separates outcome, treatment and targeted-regularization components without changing BaseLoss yet.",
        },
    ]


def efin_forward_artifacts(model: EFIN, x: torch.Tensor, t: torch.Tensor) -> dict[str, torch.Tensor]:
    t_true = t.reshape(-1, 1)
    x_rep = x.unsqueeze(2) * model.x_rep.weight.unsqueeze(0)
    dims = x_rep.size()
    x_norm = x_rep / torch.linalg.norm(x_rep, dim=1, keepdim=True).clamp_min(1e-8)
    if model.is_self:
        xx, _xx_weight = model.self_attention(x_norm)
        control_input = torch.reshape(xx, (dims[0], dims[1] * dims[2]))
    else:
        control_input = torch.reshape(x_norm, (dims[0], dims[1] * dims[2]))
    c_last = model.control_net(control_input)
    c_logit = model.c_logit(c_last)
    t_rep = model.t_rep(torch.ones_like(t_true))
    xt, attention = model.interaction_attn(t_rep, x_rep)
    u_last = model.uplift_net(xt)
    t_logit = model.t_logit(u_last)
    t_pred = torch.sigmoid(t_logit)
    u_tau = model.u_tau(u_last)
    c_prob = torch.sigmoid(c_logit)
    y0_pred = c_prob
    y1_pred = torch.sigmoid(c_logit.detach() + u_tau)
    return {
        "attention_weights": attention,
        "c_logit": c_logit,
        "u_tau": u_tau,
        "t_pred": t_pred,
        "y0_pred": y0_pred,
        "y1_pred": y1_pred,
    }


def efin_rows(root: Path, out_dir: Path) -> list[dict[str, Any]]:
    x, t, _y = synthetic_batch(features=8)
    x = torch.sigmoid(x)
    model = EFIN(input_dim=x.shape[1], hc_dim=16, hu_dim=8, is_self=False, task="classification")
    model.eval()
    with torch.no_grad():
        artifacts_tensors = efin_forward_artifacts(model, x, t)
    attention = artifacts_tensors["attention_weights"]
    row_sums = attention.sum(dim=1)
    contribution_ratio = artifacts_tensors["u_tau"].abs().mean() / (
        artifacts_tensors["c_logit"].abs().mean() + 1e-8
    )
    artifact_dir = out_dir / "EFIN"
    artifacts_attention = [
        write_frame(
            artifact_dir / "efin_attention_weights.csv",
            pd.DataFrame(attention.detach().cpu().numpy()).add_prefix("feature_"),
        ),
        write_json(
            artifact_dir / "efin_top_interactions.json",
            {
                "mean_attention_by_feature": attention.mean(dim=0).detach().cpu().tolist(),
                "attention_stats": tensor_stats(attention),
            },
        ),
    ]
    artifacts_path = [
        write_frame(
            artifact_dir / "efin_path_contribution.csv",
            pd.DataFrame(
                {
                    "row_id": np.arange(len(x)),
                    "c_logit": artifacts_tensors["c_logit"].squeeze(1).detach().cpu().numpy(),
                    "u_tau": artifacts_tensors["u_tau"].squeeze(1).detach().cpu().numpy(),
                    "y0_pred": artifacts_tensors["y0_pred"].squeeze(1).detach().cpu().numpy(),
                    "y1_pred": artifacts_tensors["y1_pred"].squeeze(1).detach().cpu().numpy(),
                }
            ),
        ),
        write_json(
            artifact_dir / "uplift_path_variance_by_seed.json",
            {
                "u_tau_stats": tensor_stats(artifacts_tensors["u_tau"]),
                "c_logit_stats": tensor_stats(artifacts_tensors["c_logit"]),
                "uplift_contribution_ratio": clean_float(contribution_ratio.item()),
            },
        ),
    ]
    attention_checks = {
        "attention_finite": bool(torch.isfinite(attention).all().item()),
        "attention_row_sum_pass": bool(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)),
    }
    path_checks = {
        "u_tau_finite": bool(torch.isfinite(artifacts_tensors["u_tau"]).all().item()),
        "contribution_pass": clean_float(contribution_ratio.item()) is not None,
    }
    return [
        {
            "contract_id": "EFIN.interaction_attention",
            "model": "EFIN",
            "hook_name": "treatment-aware interaction attention export",
            "status": row_status(artifacts_attention, attention_checks),
            "tensor_keys_exported": ["attention_weights", "treatment_representation", "uplift_logit"],
            "artifact_files": artifacts_attention,
            "metric_summary": {
                "attention_entropy_proxy": clean_float((-(attention * (attention + 1e-8).log()).sum(dim=1).mean()).item()),
                "max_mean_attention": clean_float(attention.mean(dim=0).max().item()),
            },
            "checks": attention_checks,
            "pass_gate": "attention weights are finite and row-normalized.",
            "remaining_gap": "Needs seed stability and no-interaction ablation before interpretability claim.",
            "safe_claim": "EFIN attention weights can be smoke-exported through an evaluator-side reconstruction of the existing forward path.",
            "interview_line": "EFIN attention smoke exports feature-level treatment interaction weights; next step is stability and ablation.",
        },
        {
            "contract_id": "EFIN.uplift_path_contribution",
            "model": "EFIN",
            "hook_name": "base-response vs uplift-path contribution split",
            "status": row_status(artifacts_path, path_checks),
            "tensor_keys_exported": ["c_logit", "u_tau", "y0_pred", "y1_pred"],
            "artifact_files": artifacts_path,
            "metric_summary": {
                "uplift_contribution_ratio": clean_float(contribution_ratio.item()),
                "u_tau_std": tensor_stats(artifacts_tensors["u_tau"]).get("std"),
            },
            "checks": path_checks,
            "pass_gate": "base-response and uplift-path tensors are finite with a non-null contribution ratio.",
            "remaining_gap": "Needs trained validation batches and policy-value linkage.",
            "safe_claim": "EFIN path contribution can now be smoke-exported; it is not yet a trained attribution result.",
            "interview_line": "EFIN path smoke separates base response c_logit from uplift u_tau, preventing response-fit overclaim.",
        },
    ]


def descn_rows(root: Path, out_dir: Path) -> list[dict[str, Any]]:
    x, t, y = synthetic_batch()
    model = ESX(input_dim=x.shape[1], share_dim=12, base_dim=8, task="classification")
    model.eval()
    with torch.no_grad():
        p_prpsy, y_preds, p_estr, p_escr, tau_logit, mu1_logit, mu0_logit, p_prpsy_logit, shared_h = model(x, t)
        loss, outcome_loss, treatment_loss = esx_loss(
            p_prpsy,
            y_preds,
            t,
            y,
            p_estr,
            p_escr,
            tau_logit,
            mu1_logit,
            mu0_logit,
            p_prpsy_logit,
            shared_h,
        )
        cross_tr = torch.sigmoid(mu0_logit + tau_logit)
        cross_cr = torch.sigmoid(mu1_logit - tau_logit)
        cross_residual = torch.mean((cross_tr - y_preds[1]).abs() + (cross_cr - y_preds[0]).abs())
    artifact_dir = out_dir / "DESCN"
    artifacts_prop = [
        write_frame(
            artifact_dir / "descn_propensity_predictions.csv",
            pd.DataFrame(
                {
                    "row_id": np.arange(len(x)),
                    "t_true": t.squeeze(1).detach().cpu().numpy(),
                    "p_prpsy": p_prpsy.squeeze(1).detach().cpu().numpy(),
                    "p_prpsy_logit": p_prpsy_logit.squeeze(1).detach().cpu().numpy(),
                }
            ),
        ),
        write_json(
            artifact_dir / "descn_overlap_histogram.json",
            {"p_prpsy_stats": tensor_stats(p_prpsy), "treatment_rate": clean_float(t.mean().item())},
        ),
    ]
    artifacts_heads = [
        write_frame(
            artifact_dir / "descn_head_outputs.csv",
            pd.DataFrame(
                {
                    "row_id": np.arange(len(x)),
                    "p_h0": y_preds[0].squeeze(1).detach().cpu().numpy(),
                    "p_h1": y_preds[1].squeeze(1).detach().cpu().numpy(),
                    "p_estr": p_estr.squeeze(1).detach().cpu().numpy(),
                    "p_escr": p_escr.squeeze(1).detach().cpu().numpy(),
                    "tau_logit": tau_logit.squeeze(1).detach().cpu().numpy(),
                    "mu1_logit": mu1_logit.squeeze(1).detach().cpu().numpy(),
                    "mu0_logit": mu0_logit.squeeze(1).detach().cpu().numpy(),
                }
            ),
        ),
        write_json(
            artifact_dir / "descn_head_distribution_summary.json",
            {
                "tau_stats": tensor_stats(tau_logit),
                "mu1_stats": tensor_stats(mu1_logit),
                "mu0_stats": tensor_stats(mu0_logit),
            },
        ),
    ]
    artifacts_constraints = [
        write_frame(
            artifact_dir / "descn_loss_components.csv",
            pd.DataFrame(
                [
                    {"component": "total_loss", "value": clean_float(loss.item())},
                    {"component": "outcome_loss", "value": clean_float(outcome_loss.item())},
                    {"component": "treatment_loss", "value": clean_float(treatment_loss.item())},
                    {"component": "cross_head_residual", "value": clean_float(cross_residual.item())},
                ]
            ),
        ),
        write_json(
            artifact_dir / "descn_cross_constraint_curve.json",
            {
                "cross_head_residual": clean_float(cross_residual.item()),
                "cross_tr_stats": tensor_stats(cross_tr),
                "cross_cr_stats": tensor_stats(cross_cr),
            },
        ),
    ]
    prop_checks = {
        "p_prpsy_finite": bool(torch.isfinite(p_prpsy).all().item()),
        "p_prpsy_range_pass": bool(((p_prpsy > 0) & (p_prpsy < 1)).all().item()),
    }
    head_checks = {
        "heads_finite": all(
            bool(torch.isfinite(value).all().item())
            for value in [p_estr, p_escr, tau_logit, mu1_logit, mu0_logit, y_preds[0], y_preds[1]]
        ),
        "head_shape_pass": bool(tau_logit.shape == mu1_logit.shape == mu0_logit.shape),
    }
    constraint_checks = {
        "loss_components_finite": all(clean_float(value.item()) is not None for value in [loss, outcome_loss, treatment_loss]),
        "cross_residual_finite": clean_float(cross_residual.item()) is not None,
    }
    return [
        {
            "contract_id": "DESCN.propensity_and_exposure",
            "model": "DESCN",
            "hook_name": "propensity / exposure head export",
            "status": row_status(artifacts_prop, prop_checks),
            "tensor_keys_exported": ["p_prpsy", "p_prpsy_logit", "t_true"],
            "artifact_files": artifacts_prop,
            "metric_summary": {
                "p_prpsy_mean": clean_float(p_prpsy.mean().item()),
                "p_prpsy_std": tensor_stats(p_prpsy).get("std"),
            },
            "checks": prop_checks,
            "pass_gate": "p_prpsy export is finite and bounded in (0, 1).",
            "remaining_gap": "Needs calibrated overlap/ESS and exposure drift by seed.",
            "safe_claim": "DESCN propensity/exposure output can be smoke-exported from ESX.forward.",
            "interview_line": "DESCN exposure smoke exports p_prpsy and logits; next step is calibration, ESS and drift.",
        },
        {
            "contract_id": "DESCN.entire_space_heads",
            "model": "DESCN",
            "hook_name": "mu0/mu1/tau/estr/escr head export",
            "status": row_status(artifacts_heads, head_checks),
            "tensor_keys_exported": ["mu0_logit", "mu1_logit", "tau_logit", "p_estr", "p_escr", "p_h0", "p_h1"],
            "artifact_files": artifacts_heads,
            "metric_summary": {
                "tau_std": tensor_stats(tau_logit).get("std"),
                "mu1_mean": tensor_stats(mu1_logit).get("mean"),
                "mu0_mean": tensor_stats(mu0_logit).get("mean"),
            },
            "checks": head_checks,
            "pass_gate": "all entire-space head tensors are finite and shape-consistent.",
            "remaining_gap": "Needs arm-level calibration and validation residuals across seeds.",
            "safe_claim": "DESCN head outputs can be smoke-exported; clean CATE claim still requires benchmark linkage.",
            "interview_line": "DESCN head smoke exports mu0/mu1/tau/estr/escr outputs, making head-level review possible.",
        },
        {
            "contract_id": "DESCN.cross_head_constraints",
            "model": "DESCN",
            "hook_name": "cross-head constraint loss split",
            "status": row_status(artifacts_constraints, constraint_checks),
            "tensor_keys_exported": ["total_loss", "outcome_loss", "treatment_loss", "cross_head_residual"],
            "artifact_files": artifacts_constraints,
            "metric_summary": {
                "total_loss": clean_float(loss.item()),
                "outcome_loss": clean_float(outcome_loss.item()),
                "treatment_loss": clean_float(treatment_loss.item()),
                "cross_head_residual": clean_float(cross_residual.item()),
            },
            "checks": constraint_checks,
            "pass_gate": "loss components and cross-head residual are finite in guarded smoke.",
            "remaining_gap": "Needs optional esx_loss component dictionary and no-cross-constraint ablation.",
            "safe_claim": "DESCN cross-head residual can be smoke-computed; promotion needs ablation evidence.",
            "interview_line": "DESCN constraint smoke separates total/outcome/treatment loss and cross-head residual for future ablation.",
        },
    ]


def build_rows(root: Path, out_dir: Path, selected_model: str | None = None) -> list[dict[str, Any]]:
    builders = {
        "CFRNet": cfrnet_rows,
        "DragonNet": dragonnet_rows,
        "EFIN": efin_rows,
        "DESCN": descn_rows,
    }
    rows: list[dict[str, Any]] = []
    for model, builder in builders.items():
        if selected_model and model != selected_model:
            continue
        rows.extend(builder(root, out_dir))
    return rows


def build_payload(root: Path, rows: list[dict[str, Any]], artifact_dir: Path) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    model_counts: dict[str, int] = {}
    for row in rows:
        status_counts[str(row.get("status"))] = status_counts.get(str(row.get("status")), 0) + 1
        model_counts[str(row.get("model"))] = model_counts.get(str(row.get("model")), 0) + 1
    return {
        "schema_version": 1,
        "status": "ok" if rows and all(row.get("status") == "ok" for row in rows) else "review",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "artifact_dir": str(artifact_dir),
        "summary": {
            "models": len(model_counts),
            "contracts": len(rows),
            "ok_contracts": sum(1 for row in rows if row.get("status") == "ok"),
            "artifact_files": sum(len(row.get("artifact_files") or []) for row in rows),
            "status_counts": status_counts,
            "model_counts": model_counts,
        },
        "rows": rows,
        "interview_tracks": {
            "30s": "Forward hook smoke proves the contract is executable on real model forward outputs without changing default APIs.",
            "5min": "It exports CFRNet phi_x, DragonNet e_hat/tarreg, EFIN attention/path split and DESCN exposure/heads/constraints as guarded sidecar artifacts.",
            "15min": "Use this before paper-level claims: smoke-export tensors, add per-epoch trainer hooks, link them to PEHE/QINI/policy movement, then promote only passing claims.",
        },
    }


def flatten(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return "" if value is None else str(value)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "contract_id",
        "model",
        "hook_name",
        "status",
        "tensor_keys_exported",
        "artifact_files",
        "metric_summary",
        "checks",
        "pass_gate",
        "remaining_gap",
        "safe_claim",
        "interview_line",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: flatten(row.get(field)) for field in fields})


def build_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary") or {}
    rows = payload.get("rows") or []
    tracks = payload.get("interview_tracks") or {}
    table = "\n".join(
        "| {model} | {hook} | {status} | {keys} | {artifacts} | {gate} |".format(
            model=str(row.get("model", "")).replace("|", "/"),
            hook=str(row.get("hook_name", "")).replace("|", "/"),
            status=str(row.get("status", "")).replace("|", "/"),
            keys=", ".join(row.get("tensor_keys_exported") or []).replace("|", "/"),
            artifacts=str(len(row.get("artifact_files") or [])).replace("|", "/"),
            gate=str(row.get("pass_gate", "")).replace("|", "/"),
        )
        for row in rows
    )
    sections = []
    for row in rows:
        sections.append(
            f"""## {row.get("contract_id")}

Model: `{row.get("model")}`
Hook: `{row.get("hook_name")}`
Status: `{row.get("status")}`
Tensor keys exported: `{", ".join(row.get("tensor_keys_exported") or [])}`

Artifact files:
{chr(10).join(f"- `{item}`" for item in row.get("artifact_files") or [])}

Metric summary: `{json.dumps(row.get("metric_summary") or {}, ensure_ascii=False, sort_keys=True)}`

Checks: `{json.dumps(row.get("checks") or {}, ensure_ascii=False, sort_keys=True)}`

Pass gate: {row.get("pass_gate")}

Remaining gap: {row.get("remaining_gap")}

Safe claim: {row.get("safe_claim")}

Interview line: {row.get("interview_line")}
"""
        )
    return f"""# DeepUplift Deep Model Forward Hook Smoke

Generated at: `{payload.get("generated_at")}`

This smoke executes guarded model-forward instrumentation on tiny synthetic tensors. It writes sidecar artifacts for advanced hooks without changing the default model APIs, then explains which parts are smoke evidence and which still need trainer-level, per-epoch or benchmark-linked implementation.

Artifact dir: `{payload.get("artifact_dir")}`

## Summary

| Metric | Value |
| --- | ---: |
| Models | {summary.get("models", 0)} |
| Contracts | {summary.get("contracts", 0)} |
| OK contracts | {summary.get("ok_contracts", 0)} |
| Artifact files | {summary.get("artifact_files", 0)} |

Status counts: `{json.dumps(summary.get("status_counts") or {}, ensure_ascii=False, sort_keys=True)}`

## Interview Tracks

- 30s: {tracks.get("30s", "")}
- 5min: {tracks.get("5min", "")}
- 15min: {tracks.get("15min", "")}

## Hook Overview

| Model | Hook | Status | Tensor keys | Artifact count | Pass gate |
| --- | --- | --- | --- | ---: | --- |
{table}

{chr(10).join(sections)}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke guarded forward-hook exports for deep uplift models.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--model", choices=["CFRNet", "DragonNet", "EFIN", "DESCN"], default=None)
    parser.add_argument("--json-output", default="reports/deep_model_forward_hook_smoke_latest.json")
    parser.add_argument("--csv-output", default="reports/deep_model_forward_hook_smoke_latest.csv")
    parser.add_argument("--doc-output", default="docs/DEEPUplift_DEEP_MODEL_FORWARD_HOOK_SMOKE.md")
    args = parser.parse_args()

    root = Path(args.root)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    artifact_dir = root / "reports" / "deep_model_forward_hook_artifacts" / timestamp
    rows = build_rows(root, artifact_dir, selected_model=args.model)
    payload = build_payload(root, rows, artifact_dir)

    json_path = Path(args.json_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(Path(args.csv_output), rows)
    doc_path = Path(args.doc_output)
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(build_markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload.get("status"),
                "json": str(json_path),
                "csv": args.csv_output,
                "markdown": str(doc_path),
                "contracts": (payload.get("summary") or {}).get("contracts"),
                "ok_contracts": (payload.get("summary") or {}).get("ok_contracts"),
                "artifact_dir": str(artifact_dir),
            },
            ensure_ascii=False,
        )
    )
    if payload.get("status") != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
