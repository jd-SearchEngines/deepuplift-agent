from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEEP_MODELS = ("CFRNet", "DragonNet", "EFIN", "DESCN")


CONTRACT_SPECS: list[dict[str, Any]] = [
    {
        "contract_id": "CFRNet.phi_x_representation",
        "model": "CFRNet",
        "hook_name": "phi_x representation export",
        "priority": "P0",
        "contract_type": "forward_output",
        "source_code_entry": "deepuplift/models/TarNet.py::TarNet.forward; deepuplift/models/CFRNet.py::cfrnet_loss",
        "current_return_contract": "TarNet.forward already returns `t_pred, y_preds, phi_x, eps`; CFRNet loss receives phi_x, but training artifacts do not persist per-epoch representation telemetry.",
        "expected_tensor_keys": ["phi_x", "t_true", "y_true", "y0_pred", "y1_pred"],
        "tensor_shape_contract": "`phi_x`: [batch, share_dim]; `t_true`/`y_true`: [batch, 1]; predictions: [batch, 1]",
        "artifact_files": [
            "representation_samples.csv",
            "representation_balance_curve.json",
            "representation_distance_histogram.csv",
            "alpha_ablation_balance_report.json",
        ],
        "producer": "guarded trainer/evaluator hook around CFRNet forward pass",
        "consumer": "telemetry smoke, tuned benchmark battle cards, Model Deconstruction UI",
        "smoke_test": "python3 scripts/smoke_cfrnet_training_balance.py && python3 scripts/smoke_deep_model_forward_hooks.py --model CFRNet",
        "pass_gate": "phi_x export exists, MMD/IPM curve is finite, and balance movement is linked to PEHE/QINI/policy delta.",
        "fail_action": "keep CFRNet at architecture/tuned-candidate evidence and avoid paper-level balance claims.",
        "fallback_behavior": "If hook export is disabled, continue writing existing train_history/predictions/metrics artifacts and mark representation telemetry as missing_guarded_hook.",
        "backward_compat": "Do not change TarNet.forward tuple order; wrap or post-process the existing phi_x output.",
        "migration_step": "Add optional `collect_forward_artifacts=True` to the training smoke or evaluator, then attach artifact paths to evidence_manifest.json.",
        "linked_gap_ids": ["CFRNet.representation_balance_ipm"],
        "expected_metric_link": "IPM/MMD down should not erase effect modifiers; PEHE/QINI movement must be explained.",
        "safe_claim": "可以说 CFRNet forward 已有 phi_x，可实现 representation telemetry；只有落盘并通过 gate 后才能声称 balance 起作用。",
        "blocked_claim": "不能只因为 CFRNet 名字或 IPM loss 存在就声称已经复现 paper-level representation balancing。",
    },
    {
        "contract_id": "DragonNet.e_hat_propensity",
        "model": "DragonNet",
        "hook_name": "e_hat propensity export",
        "priority": "P0",
        "contract_type": "forward_output",
        "source_code_entry": "deepuplift/models/TarNet.py::TarNet.forward; deepuplift/models/DragonNet.py::dragonnet_loss",
        "current_return_contract": "DragonNet inherits TarNet.forward and returns treatment prediction `t_pred`; training artifacts only log aggregate treatment loss.",
        "expected_tensor_keys": ["e_hat", "t_true", "propensity_logit", "propensity_bucket"],
        "tensor_shape_contract": "`e_hat`/`t_true`: [batch, 1]; bucket ids are evaluator-side categorical bins.",
        "artifact_files": [
            "propensity_predictions.csv",
            "propensity_calibration_bins.csv",
            "overlap_histogram.csv",
            "ess_clipping_sensitivity.json",
        ],
        "producer": "DragonNet forward hook plus evaluator propensity diagnostics",
        "consumer": "OPE/overlap diagnostics, telemetry smoke, Evidence Dashboard",
        "smoke_test": "python3 scripts/smoke_deep_model_forward_hooks.py --model DragonNet --hook propensity",
        "pass_gate": "propensity calibration is finite, clipped ESS remains usable, and tail buckets do not dominate PEHE/QINI.",
        "fail_action": "fall back to DR/R/DML nuisance diagnostics or require randomized/holdout data before promotion.",
        "fallback_behavior": "If e_hat artifact is unavailable, keep aggregate treatment_loss and mark overlap evidence incomplete.",
        "backward_compat": "Reuse existing `t_pred` without changing DragonNet.forward tuple order.",
        "migration_step": "Persist e_hat on validation batches, then join with predictions.csv by row_id for bucketed PEHE/QINI analysis.",
        "linked_gap_ids": ["DragonNet.propensity_auxiliary_head"],
        "expected_metric_link": "Better calibration/ESS should reduce PEHE/QINI instability and make DR policy value less tail-driven.",
        "safe_claim": "可以说 DragonNet 的 propensity head 有源码路径和导出契约；上线前还必须看 overlap/ESS。",
        "blocked_claim": "不能说 propensity head 自动修复 selection bias 或缺失 support。",
    },
    {
        "contract_id": "DragonNet.targeted_regularization_components",
        "model": "DragonNet",
        "hook_name": "targeted regularization loss split",
        "priority": "P0",
        "contract_type": "loss_component",
        "source_code_entry": "deepuplift/models/BaseModel.py::BaseLoss; deepuplift/models/DragonNet.py::dragonnet_loss",
        "current_return_contract": "BaseLoss returns total, outcome and treatment losses; targeted regularization is added into total loss but not returned separately.",
        "expected_tensor_keys": ["outcome_loss", "treatment_loss", "targeted_regularization", "epsilon", "beta"],
        "tensor_shape_contract": "loss terms are scalar tensors per batch; epsilon is [batch, 1] or broadcastable scalar.",
        "artifact_files": [
            "dragonnet_loss_components.csv",
            "tarreg_curve.json",
            "beta_ablation_report.json",
            "tarreg_instability_flags.json",
        ],
        "producer": "guarded BaseLoss component dictionary or trainer-side loss callback",
        "consumer": "objective diagnostics, model battle report, interview evidence pack",
        "smoke_test": "python3 scripts/smoke_deep_model_forward_hooks.py --model DragonNet --hook tarreg",
        "pass_gate": "tarreg term is finite, beta sweep has an explained stable region, and PEHE/QINI/policy movement is not an artifact of exploding epsilon.",
        "fail_action": "label tarreg as guarded ablation evidence and do not claim targeted regularization helped.",
        "fallback_behavior": "If component split is disabled, preserve current three-value loss return and log `tarreg_component_status=missing_guarded_hook`.",
        "backward_compat": "Keep existing loss function return shape unless an optional `return_components=True` flag is set.",
        "migration_step": "Add an optional component dict path, then update deep-model trainer to serialize per-epoch components when present.",
        "linked_gap_ids": ["DragonNet.targeted_regularization"],
        "expected_metric_link": "Beta sweep should explain whether targeted regularization improves PEHE/QINI or destabilizes under clipped propensity.",
        "safe_claim": "可以说 targeted regularization 已有公式和实现入口，下一步是拆分 loss component 做可解释 ablation。",
        "blocked_claim": "不能把 DragonNet 的总 loss 下降解释成 tarreg 生效。",
    },
    {
        "contract_id": "EFIN.interaction_attention",
        "model": "EFIN",
        "hook_name": "treatment-aware interaction attention export",
        "priority": "P0",
        "contract_type": "forward_output",
        "source_code_entry": "deepuplift/models/EFIN.py::EFIN.interaction_attn; deepuplift/models/EFIN.py::EFIN.forward",
        "current_return_contract": "EFIN.forward returns `t_pred, y_preds, u_tau`; interaction attention `xt_weight` is computed but not returned or persisted.",
        "expected_tensor_keys": ["attention_weights", "feature_names", "treatment_representation", "uplift_logit"],
        "tensor_shape_contract": "`attention_weights`: [batch, n_features]; `uplift_logit`: [batch, 1].",
        "artifact_files": [
            "efin_attention_weights.csv",
            "efin_top_interactions.csv",
            "interaction_stability_by_seed.csv",
            "no_interaction_ablation_report.json",
        ],
        "producer": "optional EFIN.forward return_artifacts flag or evaluator-side forward hook",
        "consumer": "Model Deconstruction UI, failure attribution, benchmark interpretation",
        "smoke_test": "python3 scripts/smoke_deep_model_forward_hooks.py --model EFIN --hook attention",
        "pass_gate": "attention weights are finite, stable across seeds, and no-interaction ablation changes ranking/policy metrics in the expected direction.",
        "fail_action": "describe EFIN as an interaction architecture candidate, not as interpretable attention evidence.",
        "fallback_behavior": "If attention export is disabled, keep current EFIN predictions and mark attention telemetry as missing_guarded_hook.",
        "backward_compat": "Do not change default EFIN.forward return tuple; use guarded flag or external hook.",
        "migration_step": "Return a sidecar artifact dictionary only when requested, then serialize top-k feature interactions per seed.",
        "linked_gap_ids": ["EFIN.treatment-aware_feature_interaction"],
        "expected_metric_link": "Stable treatment-aware attention should align with uplift@K, QINI and policy value.",
        "safe_claim": "可以说 EFIN 的 attention 计算点明确，能做 clean-room 可解释性导出。",
        "blocked_claim": "不能在未导出 attention 的情况下声称模型已经具备可解释 feature interaction 证据。",
    },
    {
        "contract_id": "EFIN.uplift_path_contribution",
        "model": "EFIN",
        "hook_name": "base-response vs uplift-path contribution split",
        "priority": "P1",
        "contract_type": "forward_output",
        "source_code_entry": "deepuplift/models/EFIN.py::EFIN.forward",
        "current_return_contract": "EFIN returns y0/y1 predictions and u_tau, but does not persist base path and uplift path contribution statistics.",
        "expected_tensor_keys": ["c_logit", "u_tau", "y0_pred", "y1_pred", "uplift_contribution_ratio"],
        "tensor_shape_contract": "`c_logit`, `u_tau`, `y0_pred`, `y1_pred`: [batch, 1]; contribution ratio is scalar per batch/epoch.",
        "artifact_files": [
            "efin_path_contribution.csv",
            "uplift_path_variance_by_seed.csv",
            "topk_policy_variance.json",
        ],
        "producer": "EFIN evaluator hook around control and uplift tower outputs",
        "consumer": "failure attribution, promotion matrix, interview pack",
        "smoke_test": "python3 scripts/smoke_deep_model_forward_hooks.py --model EFIN --hook uplift_path",
        "pass_gate": "uplift path contributes non-trivial, stable signal beyond base response and supports top-K policy value.",
        "fail_action": "state that EFIN ranking is dominated by base response until path telemetry says otherwise.",
        "fallback_behavior": "Keep default predictions; mark path split unavailable and rely on aggregate benchmark metrics.",
        "backward_compat": "Keep default forward output unchanged; derive contribution in optional artifact path.",
        "migration_step": "Expose c_logit/u_tau under guarded flag and add contribution summary to run card.",
        "linked_gap_ids": ["EFIN.base_response_path", "EFIN.uplift_path"],
        "expected_metric_link": "Base calibration plus uplift contribution should explain QINI and policy value movement.",
        "safe_claim": "可以说下一步能区分 response fit 和 uplift path 贡献，避免把响应概率误当增益。",
        "blocked_claim": "不能把 EFIN 的好排名直接归因于 uplift interaction，除非 path contribution artifact 支持。",
    },
    {
        "contract_id": "DESCN.propensity_and_exposure",
        "model": "DESCN",
        "hook_name": "propensity / exposure head export",
        "priority": "P0",
        "contract_type": "forward_output",
        "source_code_entry": "deepuplift/models/DESCN.py::ESX.forward; deepuplift/models/DESCN.py::esx_loss",
        "current_return_contract": "ESX.forward returns `p_prpsy` and `p_prpsy_logit`; artifacts do not persist exposure-head calibration or drift.",
        "expected_tensor_keys": ["p_prpsy", "p_prpsy_logit", "t_true", "exposure_bucket"],
        "tensor_shape_contract": "`p_prpsy`/`p_prpsy_logit`/`t_true`: [batch, 1]; exposure_bucket is evaluator-side categorical bin.",
        "artifact_files": [
            "descn_propensity_predictions.csv",
            "descn_overlap_histogram.csv",
            "descn_exposure_drift_by_seed.csv",
            "descn_ess_clipping_sensitivity.json",
        ],
        "producer": "DESCN/ESX forward hook plus overlap evaluator",
        "consumer": "readiness gate, OPE diagnostics, Evidence Dashboard",
        "smoke_test": "python3 scripts/smoke_deep_model_forward_hooks.py --model DESCN --hook propensity",
        "pass_gate": "propensity calibration and overlap are finite before DESCN is compared as an uplift candidate.",
        "fail_action": "keep DESCN as classification/entire-space architecture evidence only.",
        "fallback_behavior": "If exposure export is disabled, keep existing ESX training evidence and block causal-readiness claim.",
        "backward_compat": "Reuse existing ESX.forward tuple positions; do not rename `p_prpsy` outputs.",
        "migration_step": "Persist exposure predictions on validation rows and attach calibration/ESS report to run manifest.",
        "linked_gap_ids": ["DESCN.propensity_head"],
        "expected_metric_link": "Propensity telemetry should explain when DESCN gains are credible versus exposure leakage.",
        "safe_claim": "可以说 DESCN 已有 propensity/exposure head 输出，下一步要把它变成 overlap/ESS 证据。",
        "blocked_claim": "不能把 exposure modeling 等同于已解决选择偏差。",
    },
    {
        "contract_id": "DESCN.entire_space_heads",
        "model": "DESCN",
        "hook_name": "mu0/mu1/tau/estr/escr head export",
        "priority": "P0",
        "contract_type": "forward_output",
        "source_code_entry": "deepuplift/models/DESCN.py::ESX.forward",
        "current_return_contract": "ESX.forward returns p_estr, p_escr, tau_logit, mu1_logit and mu0_logit; artifacts do not persist head distributions or residuals.",
        "expected_tensor_keys": ["mu0_logit", "mu1_logit", "tau_logit", "p_estr", "p_escr", "p_h0", "p_h1"],
        "tensor_shape_contract": "all head outputs are [batch, 1].",
        "artifact_files": [
            "descn_head_outputs.csv",
            "descn_head_distribution_summary.json",
            "descn_arm_calibration.csv",
            "descn_head_consistency_residuals.csv",
        ],
        "producer": "DESCN/ESX evaluator hook on validation batches",
        "consumer": "Model Deconstruction UI, objective diagnostics, promotion matrix",
        "smoke_test": "python3 scripts/smoke_deep_model_forward_hooks.py --model DESCN --hook heads",
        "pass_gate": "mu0/mu1/tau heads are finite, arm-level calibration is bounded, and consistency residuals are stable across seeds.",
        "fail_action": "keep DESCN as ranking candidate and avoid clean-CATE estimation claims.",
        "fallback_behavior": "If head export is disabled, keep predictions.csv and label head telemetry missing.",
        "backward_compat": "Use existing ESX.forward tuple; add sidecar serialization without changing default API.",
        "migration_step": "Add a guarded head-output serializer and compare residuals against PEHE/ATE/QINI movement.",
        "linked_gap_ids": ["DESCN.entire-space_response_heads"],
        "expected_metric_link": "Head consistency should explain PEHE/ATE error versus ranking-only gains.",
        "safe_claim": "可以说 DESCN 的 entire-space heads 已有源码输出点，后续能落到 head-level evidence。",
        "blocked_claim": "不能只凭 ranking metric 说 DESCN 已经给出可靠 CATE。",
    },
    {
        "contract_id": "DESCN.cross_head_constraints",
        "model": "DESCN",
        "hook_name": "cross-head constraint loss split",
        "priority": "P0",
        "contract_type": "loss_component",
        "source_code_entry": "deepuplift/models/DESCN.py::esx_loss",
        "current_return_contract": "esx_loss computes prpsy/estr/escr/tr/cr/cross_tr/cross_cr/imb terms but returns only total, outcome and treatment losses.",
        "expected_tensor_keys": ["prpsy_loss", "estr_loss", "escr_loss", "tr_loss", "cr_loss", "cross_tr_loss", "cross_cr_loss", "imb_dist_loss"],
        "tensor_shape_contract": "all loss terms are scalar tensors per batch.",
        "artifact_files": [
            "descn_loss_components.csv",
            "descn_cross_constraint_curve.json",
            "descn_no_cross_constraint_ablation.json",
            "descn_task_boundary_guard.json",
        ],
        "producer": "guarded esx_loss component dictionary or trainer-side loss callback",
        "consumer": "objective diagnostics, battle report, launch readiness cards",
        "smoke_test": "python3 scripts/smoke_deep_model_forward_hooks.py --model DESCN --hook cross_constraints",
        "pass_gate": "cross-head residual/constraint term improves a decision metric without worsening PEHE beyond an explained boundary.",
        "fail_action": "state DESCN is classification/entire-space guarded until regression CATE evidence is stronger.",
        "fallback_behavior": "If component split is disabled, preserve current three-value esx_loss return and mark constraint telemetry missing.",
        "backward_compat": "Keep current esx_loss signature and return shape unless optional `return_components=True` is requested.",
        "migration_step": "Add optional component dict to esx_loss and serialize per-epoch loss components in training smoke.",
        "linked_gap_ids": ["DESCN.cross-head_treatment_effect_constraints"],
        "expected_metric_link": "Constraint residual should explain PEHE-vs-QINI tradeoffs in battle cards.",
        "safe_claim": "可以说 cross-head 约束有明确 loss 入口，下一步要拆成组件曲线和 ablation。",
        "blocked_claim": "不能把多 head 结构包装成已验证的 causal constraint，除非组件和 ablation 证据通过。",
    },
]


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def normalize_gap_id(value: str) -> str:
    return value.lower().replace(" / ", "_").replace(" ", "_")


def gap_lookup(root: Path) -> dict[str, dict[str, Any]]:
    payload = read_json(root / "reports" / "deep_model_telemetry_gap_matrix_latest.json")
    rows = payload.get("rows") or []
    lookup: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        telemetry_id = str(row.get("telemetry_id") or "")
        if telemetry_id:
            lookup[telemetry_id] = row
        model = str(row.get("model") or "")
        term = str(row.get("loss_term") or "")
        if model and term:
            lookup[f"{model}.{normalize_gap_id(term)}"] = row
    return lookup


def smoke_lookup(root: Path) -> dict[str, dict[str, Any]]:
    payload = read_json(root / "reports" / "deep_model_telemetry_smoke_latest.json")
    return {
        str(row.get("model")): row
        for row in payload.get("rows") or []
        if isinstance(row, dict) and row.get("model")
    }


def build_rows(root: Path) -> list[dict[str, Any]]:
    gaps = gap_lookup(root)
    smoke = smoke_lookup(root)
    rows: list[dict[str, Any]] = []
    for spec in CONTRACT_SPECS:
        model = str(spec["model"])
        linked = [gaps.get(gap_id) for gap_id in spec.get("linked_gap_ids", [])]
        linked = [row for row in linked if row]
        smoke_row = smoke.get(model, {})
        gap_priorities = sorted({str(row.get("priority")) for row in linked if row.get("priority")})
        gap_families = sorted({str(row.get("telemetry_family")) for row in linked if row.get("telemetry_family")})
        evidence_refs = [
            "reports/deep_model_telemetry_smoke_latest.json",
            "reports/deep_model_telemetry_gap_matrix_latest.json",
            "reports/deep_model_objective_diagnostics_latest.json",
            f"docs/model_deconstruction/{model}.md",
        ]
        rows.append(
            {
                **spec,
                "status": "contract_ready",
                "guarded": True,
                "linked_gap_priorities": gap_priorities,
                "linked_gap_families": gap_families,
                "telemetry_smoke_status": smoke_row.get("advanced_hook_status", "missing_guarded_hook"),
                "basic_telemetry_pass": bool(smoke_row.get("basic_telemetry_pass")),
                "current_run_dir": smoke_row.get("run_dir"),
                "evidence_refs": evidence_refs,
                "ui_route": "Model Deconstruction -> Forward Hook Contracts -> Evidence Dashboard",
                "interview_line": (
                    f"{model} hook contract `{spec['hook_name']}` defines keys={', '.join(spec['expected_tensor_keys'])}, "
                    f"artifacts={', '.join(spec['artifact_files'][:2])}, pass_gate={spec['pass_gate']}"
                ),
            }
        )
    return rows


def build_payload(root: Path) -> dict[str, Any]:
    rows = build_rows(root)
    by_model: dict[str, int] = {}
    by_priority: dict[str, int] = {}
    by_type: dict[str, int] = {}
    for row in rows:
        by_model[str(row.get("model"))] = by_model.get(str(row.get("model")), 0) + 1
        by_priority[str(row.get("priority"))] = by_priority.get(str(row.get("priority")), 0) + 1
        by_type[str(row.get("contract_type"))] = by_type.get(str(row.get("contract_type")), 0) + 1
    return {
        "schema_version": 1,
        "status": "ok",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "summary": {
            "models": len(by_model),
            "contracts": len(rows),
            "p0_contracts": by_priority.get("P0", 0),
            "guarded_contracts": sum(1 for row in rows if row.get("guarded")),
            "basic_telemetry_pass_models": len({row["model"] for row in rows if row.get("basic_telemetry_pass")}),
            "model_counts": by_model,
            "priority_counts": by_priority,
            "contract_type_counts": by_type,
        },
        "rows": rows,
        "interview_tracks": {
            "30s": "Forward hook contracts turn missing deep-model observability into exact tensor keys, artifact files and pass gates.",
            "5min": "CFRNet exports phi_x, DragonNet exports e_hat and tarreg components, EFIN exports attention/path contribution, DESCN exports heads and cross-head losses, all guarded for backward compatibility.",
            "15min": "This is the bridge between benchmark failure attribution and source refactor: first freeze the contract, then implement optional hooks, then rerun telemetry smoke, benchmark and promotion gates.",
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
        "priority",
        "contract_type",
        "status",
        "guarded",
        "source_code_entry",
        "current_return_contract",
        "expected_tensor_keys",
        "tensor_shape_contract",
        "artifact_files",
        "producer",
        "consumer",
        "smoke_test",
        "pass_gate",
        "fail_action",
        "fallback_behavior",
        "backward_compat",
        "migration_step",
        "linked_gap_ids",
        "linked_gap_priorities",
        "linked_gap_families",
        "telemetry_smoke_status",
        "basic_telemetry_pass",
        "ui_route",
        "interview_line",
        "safe_claim",
        "blocked_claim",
        "evidence_refs",
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
        "| {model} | {hook} | {type} | {priority} | {keys} | {gate} |".format(
            model=str(row.get("model", "")).replace("|", "/"),
            hook=str(row.get("hook_name", "")).replace("|", "/"),
            type=str(row.get("contract_type", "")).replace("|", "/"),
            priority=str(row.get("priority", "")).replace("|", "/"),
            keys=", ".join(row.get("expected_tensor_keys") or []).replace("|", "/"),
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
Type: `{row.get("contract_type")}`
Priority: `{row.get("priority")}`
Source entry: `{row.get("source_code_entry")}`

Current return contract: {row.get("current_return_contract")}

Expected tensor keys: `{", ".join(row.get("expected_tensor_keys") or [])}`

Tensor shape contract: {row.get("tensor_shape_contract")}

Artifact files: `{", ".join(row.get("artifact_files") or [])}`

Producer: {row.get("producer")}

Consumer: {row.get("consumer")}

Smoke test: `{row.get("smoke_test")}`

Pass gate: {row.get("pass_gate")}

Fail action: {row.get("fail_action")}

Fallback behavior: {row.get("fallback_behavior")}

Backward compatibility: {row.get("backward_compat")}

Migration step: {row.get("migration_step")}

Safe claim: {row.get("safe_claim")}

Blocked claim: {row.get("blocked_claim")}

Interview line: {row.get("interview_line")}
"""
        )
    return f"""# DeepUplift Deep Model Forward Hook Contracts

Generated at: `{payload.get("generated_at")}`

This artifact is the implementation contract between deep-model source code and evidence governance. It does not claim the advanced hooks are already implemented; it freezes the tensor keys, artifact files, smoke tests, pass gates, fallback behavior and interview-safe boundaries required before the next refactor.

## Summary

| Metric | Value |
| --- | ---: |
| Models | {summary.get("models", 0)} |
| Contracts | {summary.get("contracts", 0)} |
| P0 contracts | {summary.get("p0_contracts", 0)} |
| Guarded contracts | {summary.get("guarded_contracts", 0)} |
| Basic telemetry pass models | {summary.get("basic_telemetry_pass_models", 0)} |

Priority counts: `{json.dumps(summary.get("priority_counts") or {}, ensure_ascii=False, sort_keys=True)}`

Contract types: `{json.dumps(summary.get("contract_type_counts") or {}, ensure_ascii=False, sort_keys=True)}`

## Interview Tracks

- 30s: {tracks.get("30s", "")}
- 5min: {tracks.get("5min", "")}
- 15min: {tracks.get("15min", "")}

## Contract Overview

| Model | Hook | Type | Priority | Tensor keys | Pass gate |
| --- | --- | --- | --- | --- | --- |
{table}

{chr(10).join(sections)}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate guarded forward-hook contracts for deep uplift models.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--json-output", default="reports/deep_model_forward_hook_contracts_latest.json")
    parser.add_argument("--csv-output", default="reports/deep_model_forward_hook_contracts_latest.csv")
    parser.add_argument("--doc-output", default="docs/DEEPUplift_DEEP_MODEL_FORWARD_HOOK_CONTRACTS.md")
    args = parser.parse_args()

    root = Path(args.root)
    payload = build_payload(root)
    json_path = Path(args.json_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(Path(args.csv_output), payload.get("rows") or [])
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
                "models": (payload.get("summary") or {}).get("models"),
            },
            ensure_ascii=False,
        )
    )
    if payload.get("status") != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
