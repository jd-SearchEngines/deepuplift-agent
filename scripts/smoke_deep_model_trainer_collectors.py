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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepuplift.models.BaseModel import representation_balance_loss  # noqa: E402
from deepuplift.models.CFRNet import CFRNet, cfrnet_loss  # noqa: E402
from deepuplift.models.DESCN import ESX, esx_loss  # noqa: E402
from deepuplift.models.DragonNet import DragonNet, dragonnet_loss  # noqa: E402
from deepuplift.models.EFIN import EFIN, efin_loss  # noqa: E402


CONTRACTS_BY_MODEL = {
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


def tensor_std(tensor: torch.Tensor) -> float | None:
    value = tensor.detach().float()
    if value.numel() <= 1:
        return 0.0
    return clean_float(value.std(unbiased=False).item())


def write_json(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def write_frame(path: Path, frame: pd.DataFrame) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return str(path)


def synthetic_batch(seed: int, rows: int = 64, features: int = 8) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, features, generator=generator)
    logits_t = 0.35 * x[:, :1] - 0.2 * x[:, 1:2]
    propensity = torch.sigmoid(logits_t)
    t = torch.bernoulli(propensity, generator=generator)
    logits_y = -0.15 + 0.5 * x[:, :1] - 0.15 * x[:, 2:3] + 0.45 * t + 0.25 * t * x[:, 3:4]
    y = torch.bernoulli(torch.sigmoid(logits_y), generator=generator).float()
    return x.float(), t.float(), y.float()


def ess_from_propensity(e_hat: torch.Tensor, t: torch.Tensor) -> float | None:
    e = e_hat.detach().float().clamp(0.01, 0.99)
    treatment = t.detach().float()
    weights = treatment / e + (1.0 - treatment) / (1.0 - e)
    numerator = weights.sum().pow(2)
    denominator = weights.pow(2).sum().clamp_min(1e-8)
    return clean_float((numerator / denominator).item())


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
    u_tau = model.u_tau(u_last)
    c_prob = torch.sigmoid(c_logit)
    return {
        "attention_weights": attention,
        "c_logit": c_logit,
        "u_tau": u_tau,
        "y0_pred": c_prob,
        "y1_pred": torch.sigmoid(c_logit.detach() + u_tau),
    }


def finite_row(row: dict[str, Any]) -> bool:
    for value in row.values():
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return False
    return True


def train_cfrnet(epochs: int, artifact_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train_x, train_t, train_y = synthetic_batch(2026052101)
    valid_x, valid_t, valid_y = synthetic_batch(2026052102)
    model = CFRNet(input_dim=train_x.shape[1], share_dim=6, task="classification")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    rows: list[dict[str, Any]] = []
    for epoch in range(epochs):
        model.train()
        t_pred, y_preds, phi_x, eps = model(train_x, train_t)
        loss, outcome_loss, ipm_loss = cfrnet_loss(
            t_pred,
            y_preds,
            train_t,
            train_y,
            phi_x,
            eps,
            alpha=0.2,
            task="classification",
            ipm_mode="mmd_rbf",
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            _t_valid, valid_preds, valid_phi, _eps = model(valid_x, valid_t)
            _valid_loss, valid_outcome_loss, valid_ipm = cfrnet_loss(
                None,
                valid_preds,
                valid_t,
                valid_y,
                valid_phi,
                None,
                alpha=0.2,
                task="classification",
                ipm_mode="mmd_rbf",
            )
            rows.append(
                {
                    "model": "CFRNet",
                    "epoch": epoch,
                    "train_loss": clean_float(loss.item()),
                    "train_outcome_loss": clean_float(outcome_loss.item()),
                    "train_representation_mmd": clean_float(ipm_loss.item()),
                    "valid_outcome_loss": clean_float(valid_outcome_loss.item()),
                    "valid_representation_mmd": clean_float(valid_ipm.item()),
                    "valid_representation_mean_l2": clean_float(
                        representation_balance_loss(valid_phi, valid_t, mode="mean_l2").item()
                    ),
                    "valid_phi_std": tensor_std(valid_phi),
                }
            )
    artifacts = [
        write_frame(artifact_dir / "CFRNet" / "representation_balance_by_epoch.csv", pd.DataFrame(rows)),
        write_json(
            artifact_dir / "CFRNet" / "representation_balance_summary.json",
            {
                "contracts": CONTRACTS_BY_MODEL["CFRNet"],
                "last_valid_representation_mmd": rows[-1]["valid_representation_mmd"],
                "last_valid_phi_std": rows[-1]["valid_phi_std"],
            },
        ),
    ]
    return rows, {"artifact_files": artifacts, "columns": sorted(rows[0])}


def train_dragonnet(epochs: int, artifact_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train_x, train_t, train_y = synthetic_batch(2026052111)
    valid_x, valid_t, valid_y = synthetic_batch(2026052112)
    model = DragonNet(input_dim=train_x.shape[1], share_dim=6, task="classification")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    rows: list[dict[str, Any]] = []
    for epoch in range(epochs):
        model.train()
        t_pred, y_preds, phi_x, eps = model(train_x, train_t)
        loss, outcome_loss, treatment_loss = dragonnet_loss(
            t_pred,
            y_preds,
            train_t,
            train_y,
            phi_x,
            eps,
            alpha=0.05,
            beta=0.02,
            tarreg=True,
            task="classification",
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            e_hat, valid_preds, valid_phi, valid_eps = model(valid_x, valid_t)
            valid_loss, valid_outcome_loss, valid_treatment_loss = dragonnet_loss(
                e_hat,
                valid_preds,
                valid_t,
                valid_y,
                valid_phi,
                valid_eps,
                alpha=0.05,
                beta=0.02,
                tarreg=True,
                task="classification",
            )
            y_pred = valid_t * valid_preds[1] + (1 - valid_t) * valid_preds[0]
            e_smooth = (e_hat + 0.01) / 1.02
            clever_covariate = (valid_t / e_smooth) - ((1 - valid_t) / (1 - e_smooth))
            tarreg = torch.mean((valid_y - (y_pred + valid_eps * clever_covariate)).pow(2))
            rows.append(
                {
                    "model": "DragonNet",
                    "epoch": epoch,
                    "train_loss": clean_float(loss.item()),
                    "train_outcome_loss": clean_float(outcome_loss.item()),
                    "train_propensity_bce": clean_float(treatment_loss.item()),
                    "valid_loss": clean_float(valid_loss.item()),
                    "valid_outcome_loss": clean_float(valid_outcome_loss.item()),
                    "valid_propensity_bce": clean_float(valid_treatment_loss.item()),
                    "valid_propensity_ess": ess_from_propensity(e_hat, valid_t),
                    "valid_e_hat_mean": clean_float(e_hat.mean().item()),
                    "valid_e_hat_std": tensor_std(e_hat),
                    "valid_targeted_regularization": clean_float(tarreg.item()),
                    "epsilon_mean": clean_float(valid_eps.mean().item()),
                    "epsilon_std": tensor_std(valid_eps),
                }
            )
    artifacts = [
        write_frame(artifact_dir / "DragonNet" / "dragonnet_loss_components_by_epoch.csv", pd.DataFrame(rows)),
        write_json(
            artifact_dir / "DragonNet" / "propensity_tarreg_summary.json",
            {
                "contracts": CONTRACTS_BY_MODEL["DragonNet"],
                "last_valid_propensity_ess": rows[-1]["valid_propensity_ess"],
                "last_valid_targeted_regularization": rows[-1]["valid_targeted_regularization"],
            },
        ),
    ]
    return rows, {"artifact_files": artifacts, "columns": sorted(rows[0])}


def train_efin(epochs: int, artifact_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train_x, train_t, train_y = synthetic_batch(2026052121)
    valid_x, valid_t, valid_y = synthetic_batch(2026052122)
    train_x = torch.sigmoid(train_x)
    valid_x = torch.sigmoid(valid_x)
    model = EFIN(input_dim=train_x.shape[1], hc_dim=16, hu_dim=8, is_self=False, task="classification")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    rows: list[dict[str, Any]] = []
    for epoch in range(epochs):
        model.train()
        t_pred, y_preds, u_tau = model(train_x, train_t)
        loss, outcome_loss, treatment_loss = efin_loss(t_pred, y_preds, train_t, train_y, u_tau, task="classification")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            valid_t_pred, valid_preds, valid_u_tau = model(valid_x, valid_t)
            valid_loss, valid_outcome_loss, valid_treatment_loss = efin_loss(
                valid_t_pred,
                valid_preds,
                valid_t,
                valid_y,
                valid_u_tau,
                task="classification",
            )
            details = efin_forward_artifacts(model, valid_x, valid_t)
            attention = details["attention_weights"]
            entropy = -(attention * (attention + 1e-8).log()).sum(dim=1).mean()
            path_ratio = details["u_tau"].abs().mean() / (details["c_logit"].abs().mean() + 1e-8)
            rows.append(
                {
                    "model": "EFIN",
                    "epoch": epoch,
                    "train_loss": clean_float(loss.item()),
                    "train_outcome_loss": clean_float(outcome_loss.item()),
                    "train_treatment_loss": clean_float(treatment_loss.item()),
                    "valid_loss": clean_float(valid_loss.item()),
                    "valid_outcome_loss": clean_float(valid_outcome_loss.item()),
                    "valid_treatment_loss": clean_float(valid_treatment_loss.item()),
                    "valid_attention_entropy": clean_float(entropy.item()),
                    "attention_max_mean_weight": clean_float(attention.mean(dim=0).max().item()),
                    "valid_base_logit_norm": clean_float(details["c_logit"].abs().mean().item()),
                    "valid_uplift_logit_norm": clean_float(details["u_tau"].abs().mean().item()),
                    "valid_uplift_path_ratio": clean_float(path_ratio.item()),
                }
            )
    artifacts = [
        write_frame(artifact_dir / "EFIN" / "efin_attention_path_by_epoch.csv", pd.DataFrame(rows)),
        write_json(
            artifact_dir / "EFIN" / "efin_attention_path_summary.json",
            {
                "contracts": CONTRACTS_BY_MODEL["EFIN"],
                "last_attention_entropy": rows[-1]["valid_attention_entropy"],
                "last_uplift_path_ratio": rows[-1]["valid_uplift_path_ratio"],
            },
        ),
    ]
    return rows, {"artifact_files": artifacts, "columns": sorted(rows[0])}


def train_descn(epochs: int, artifact_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train_x, train_t, train_y = synthetic_batch(2026052131)
    valid_x, valid_t, valid_y = synthetic_batch(2026052132)
    model = ESX(input_dim=train_x.shape[1], share_dim=12, base_dim=8, task="classification")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    rows: list[dict[str, Any]] = []
    for epoch in range(epochs):
        model.train()
        outputs = model(train_x, train_t)
        loss, outcome_loss, treatment_loss = esx_loss(*outputs[:2], train_t, train_y, *outputs[2:])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            p_prpsy, y_preds, p_estr, p_escr, tau_logit, mu1_logit, mu0_logit, p_prpsy_logit, shared_h = model(valid_x, valid_t)
            valid_loss, valid_outcome_loss, valid_treatment_loss = esx_loss(
                p_prpsy,
                y_preds,
                valid_t,
                valid_y,
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
            rows.append(
                {
                    "model": "DESCN",
                    "epoch": epoch,
                    "train_loss": clean_float(loss.item()),
                    "train_outcome_loss": clean_float(outcome_loss.item()),
                    "train_propensity_loss": clean_float(treatment_loss.item()),
                    "valid_loss": clean_float(valid_loss.item()),
                    "valid_outcome_loss": clean_float(valid_outcome_loss.item()),
                    "valid_propensity_loss": clean_float(valid_treatment_loss.item()),
                    "valid_p_prpsy_ess": ess_from_propensity(p_prpsy, valid_t),
                    "valid_p_prpsy_mean": clean_float(p_prpsy.mean().item()),
                    "valid_p_prpsy_std": tensor_std(p_prpsy),
                    "valid_tau_std": tensor_std(tau_logit),
                    "valid_mu0_logit_mean": clean_float(mu0_logit.mean().item()),
                    "valid_mu1_logit_mean": clean_float(mu1_logit.mean().item()),
                    "valid_head_spread": clean_float((y_preds[1] - y_preds[0]).abs().mean().item()),
                    "valid_cross_head_residual": clean_float(cross_residual.item()),
                }
            )
    artifacts = [
        write_frame(artifact_dir / "DESCN" / "descn_heads_constraints_by_epoch.csv", pd.DataFrame(rows)),
        write_json(
            artifact_dir / "DESCN" / "descn_heads_constraints_summary.json",
            {
                "contracts": CONTRACTS_BY_MODEL["DESCN"],
                "last_p_prpsy_ess": rows[-1]["valid_p_prpsy_ess"],
                "last_cross_head_residual": rows[-1]["valid_cross_head_residual"],
            },
        ),
    ]
    return rows, {"artifact_files": artifacts, "columns": sorted(rows[0])}


def build_model_summary(model: str, rows: list[dict[str, Any]], details: dict[str, Any]) -> dict[str, Any]:
    metric_columns = [col for col in details.get("columns", []) if col not in {"model", "epoch"}]
    first = rows[0] if rows else {}
    last = rows[-1] if rows else {}
    metric_delta = {}
    for col in metric_columns:
        if isinstance(first.get(col), (int, float)) and isinstance(last.get(col), (int, float)):
            metric_delta[col] = clean_float(float(last[col]) - float(first[col]))
    return {
        "model": model,
        "status": "ok" if rows and all(finite_row(row) for row in rows) else "review",
        "epochs": len(rows),
        "contracts_covered": CONTRACTS_BY_MODEL[model],
        "collected_epoch_columns": metric_columns,
        "artifact_files": details.get("artifact_files") or [],
        "metric_first": {col: first.get(col) for col in metric_columns[:8]},
        "metric_last": {col: last.get(col) for col in metric_columns[:8]},
        "metric_delta": metric_delta,
        "pass_gate": "Per-epoch collector rows are finite, artifact files exist, and covered bridge contracts are explicit.",
        "fail_action": "Keep this model at hook-smoke/bridge level and inspect the collector or model loss before promoting internal telemetry.",
        "safe_claim": "A guarded tiny trainer collector can emit per-epoch internal telemetry without changing the default model API.",
        "blocked_claim": "This is not a paper-level ablation or production telemetry claim; it is a trainer collector smoke.",
        "interview_line": f"{model} now has executable per-epoch advanced telemetry smoke linked to its forward-hook contracts.",
    }


def build_payload(root: Path, model_rows: list[dict[str, Any]], epoch_rows: list[dict[str, Any]], artifact_dir: Path) -> dict[str, Any]:
    ok_models = sum(1 for row in model_rows if row.get("status") == "ok")
    artifact_files = [item for row in model_rows for item in row.get("artifact_files") or []]
    return {
        "schema_version": 1,
        "status": "ok" if ok_models == 4 and all(Path(item).exists() for item in artifact_files) else "review",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "artifact_dir": str(artifact_dir),
        "summary": {
            "models": len(model_rows),
            "ok_models": ok_models,
            "epochs_per_model": max((row.get("epochs") or 0 for row in model_rows), default=0),
            "epoch_rows": len(epoch_rows),
            "artifact_files": len(artifact_files),
            "contracts_covered": sum(len(row.get("contracts_covered") or []) for row in model_rows),
            "collected_epoch_columns": len({col for row in model_rows for col in row.get("collected_epoch_columns") or []}),
        },
        "model_rows": model_rows,
        "epoch_rows": epoch_rows,
        "interview_tracks": {
            "30s": "Trainer collector smoke proves advanced hook telemetry can be emitted per epoch without changing default model APIs.",
            "5min": "CFRNet logs representation balance, DragonNet logs propensity/tarreg, EFIN logs attention/path and DESCN logs head/constraint telemetry.",
            "15min": "This is the implementation step after hook training bridge; the next promotion step is benchmark-linked metric movement and ablation evidence.",
        },
    }


def flatten(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return "" if value is None else str(value)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: flatten(row.get(field)) for field in fields})


def build_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary") or {}
    model_rows = payload.get("model_rows") or []
    tracks = payload.get("interview_tracks") or {}
    table = "\n".join(
        "| {model} | {status} | {epochs} | {contracts} | {columns} | {artifacts} |".format(
            model=str(row.get("model", "")).replace("|", "/"),
            status=str(row.get("status", "")).replace("|", "/"),
            epochs=str(row.get("epochs", "")).replace("|", "/"),
            contracts=", ".join(row.get("contracts_covered") or []).replace("|", "/"),
            columns=", ".join(row.get("collected_epoch_columns") or [])[:180].replace("|", "/"),
            artifacts=str(len(row.get("artifact_files") or [])).replace("|", "/"),
        )
        for row in model_rows
    )
    details = []
    for row in model_rows:
        details.append(
            f"""## {row.get("model")}

Status: `{row.get("status")}`
Contracts covered: `{", ".join(row.get("contracts_covered") or [])}`
Artifacts:
{chr(10).join(f"- `{item}`" for item in row.get("artifact_files") or [])}

Metric first: `{json.dumps(row.get("metric_first") or {}, ensure_ascii=False, sort_keys=True)}`

Metric last: `{json.dumps(row.get("metric_last") or {}, ensure_ascii=False, sort_keys=True)}`

Pass gate: {row.get("pass_gate")}

Fail action: {row.get("fail_action")}

Safe claim: {row.get("safe_claim")}

Blocked claim: {row.get("blocked_claim")}

Interview line: {row.get("interview_line")}
"""
        )
    return f"""# DeepUplift Deep Model Trainer Collector Smoke

Generated at: `{payload.get("generated_at")}`

This smoke runs tiny guarded training loops for CFRNet, DragonNet, EFIN and DESCN, then writes per-epoch advanced telemetry sidecars. It is the executable step after the hook training bridge: the default model forward/loss APIs remain unchanged, while evaluator-side collectors prove the telemetry can be gathered during training.

Artifact dir: `{payload.get("artifact_dir")}`

## Summary

| Metric | Value |
| --- | ---: |
| Models | {summary.get("models", 0)} |
| OK models | {summary.get("ok_models", 0)} |
| Epoch rows | {summary.get("epoch_rows", 0)} |
| Artifact files | {summary.get("artifact_files", 0)} |
| Contracts covered | {summary.get("contracts_covered", 0)} |
| Collected epoch columns | {summary.get("collected_epoch_columns", 0)} |

## Interview Tracks

- 30s: {tracks.get("30s", "")}
- 5min: {tracks.get("5min", "")}
- 15min: {tracks.get("15min", "")}

## Model Overview

| Model | Status | Epochs | Contracts | Collected epoch columns | Artifact files |
| --- | --- | ---: | --- | --- | ---: |
{table}

{chr(10).join(details)}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tiny trainer collectors for deep model internal telemetry.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--json-output", default="reports/deep_model_trainer_collector_smoke_latest.json")
    parser.add_argument("--model-csv-output", default="reports/deep_model_trainer_collector_models_latest.csv")
    parser.add_argument("--epoch-csv-output", default="reports/deep_model_trainer_collector_epochs_latest.csv")
    parser.add_argument("--doc-output", default="docs/DEEPUplift_DEEP_MODEL_TRAINER_COLLECTOR_SMOKE.md")
    args = parser.parse_args()

    root = Path(args.root)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    artifact_dir = root / "reports" / "deep_model_trainer_collector_artifacts" / timestamp
    builders = {
        "CFRNet": train_cfrnet,
        "DragonNet": train_dragonnet,
        "EFIN": train_efin,
        "DESCN": train_descn,
    }
    model_rows: list[dict[str, Any]] = []
    epoch_rows: list[dict[str, Any]] = []
    for model, builder in builders.items():
        rows, details = builder(args.epochs, artifact_dir)
        epoch_rows.extend(rows)
        model_rows.append(build_model_summary(model, rows, details))
    payload = build_payload(root, model_rows, epoch_rows, artifact_dir)

    json_path = Path(args.json_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    model_fields = [
        "model",
        "status",
        "epochs",
        "contracts_covered",
        "collected_epoch_columns",
        "artifact_files",
        "metric_first",
        "metric_last",
        "metric_delta",
        "pass_gate",
        "fail_action",
        "safe_claim",
        "blocked_claim",
        "interview_line",
    ]
    epoch_fields = sorted({key for row in epoch_rows for key in row})
    write_csv(Path(args.model_csv_output), model_rows, model_fields)
    write_csv(Path(args.epoch_csv_output), epoch_rows, epoch_fields)
    doc_path = Path(args.doc_output)
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(build_markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload.get("status"),
                "json": str(json_path),
                "model_csv": args.model_csv_output,
                "epoch_csv": args.epoch_csv_output,
                "markdown": str(doc_path),
                "ok_models": (payload.get("summary") or {}).get("ok_models"),
                "epoch_rows": (payload.get("summary") or {}).get("epoch_rows"),
                "artifact_dir": str(artifact_dir),
            },
            ensure_ascii=False,
        )
    )
    if payload.get("status") != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
