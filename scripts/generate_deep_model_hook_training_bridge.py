from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any


BRIDGE_SPECS: dict[str, dict[str, Any]] = {
    "CFRNet.phi_x_representation": {
        "priority": "P0",
        "epoch_columns": ["train_representation_mmd", "valid_representation_mmd", "train_representation_mean_l2"],
        "epoch_artifacts": ["representation_balance_by_epoch.csv", "representation_balance_curve.json"],
        "validation_metrics": ["PEHE delta", "QINI delta", "top-K known-CATE recall", "treated/control representation distance"],
        "producer_change": "Add an optional representation collector around CFRNet.forward, logging phi_x on train/valid batches.",
        "pass_gate": "MMD/mean-L2 curves are finite for every epoch and can be joined to PEHE/QINI/policy movement.",
        "fail_action": "Keep the current smoke-only claim and run balance-weight ablation before promoting CFRNet.",
        "interview_line": "CFRNet should graduate from one-batch phi_x export to per-epoch representation-balance curves tied to PEHE and QINI.",
    },
    "DragonNet.e_hat_propensity": {
        "priority": "P0",
        "epoch_columns": ["train_propensity_bce", "valid_propensity_bce", "valid_propensity_ess", "valid_e_hat_auc"],
        "epoch_artifacts": ["propensity_calibration_by_epoch.csv", "propensity_clipping_sensitivity.json"],
        "validation_metrics": ["effective sample size", "calibration error", "clipped DR policy value", "overlap tail share"],
        "producer_change": "Add an optional propensity collector for e_hat and treatment labels on train/valid batches.",
        "pass_gate": "e_hat stays finite/bounded and ESS/calibration are reported with clipping sensitivity.",
        "fail_action": "Do not claim DragonNet overlap robustness; route to DR/R learner baseline or stronger propensity regularization.",
        "interview_line": "DragonNet's e_hat is only useful if it improves overlap diagnostics, not just because the head exists.",
    },
    "DragonNet.targeted_regularization_components": {
        "priority": "P0",
        "epoch_columns": ["train_targeted_regularization", "valid_targeted_regularization", "epsilon_mean", "epsilon_std"],
        "epoch_artifacts": ["dragonnet_loss_components_by_epoch.csv", "tarreg_beta_ablation.json"],
        "validation_metrics": ["PEHE delta", "ATE error delta", "targeted regularization share", "beta ablation lift"],
        "producer_change": "Add optional component logging in the DragonNet loss path while preserving the default scalar loss return.",
        "pass_gate": "Outcome, treatment and targeted-regularization components are finite and beta ablation improves or explains PEHE/ATE.",
        "fail_action": "Keep DragonNet at source-deconstruction/tuned-benchmark level and avoid tarreg superiority claims.",
        "interview_line": "The tarreg bridge separates the clever-covariate loss from outcome/treatment losses so the claim is measurable.",
    },
    "EFIN.interaction_attention": {
        "priority": "P1",
        "epoch_columns": ["valid_attention_entropy", "attention_top_feature_jaccard", "attention_max_mean_weight"],
        "epoch_artifacts": ["efin_attention_by_epoch.csv", "efin_attention_seed_stability.json"],
        "validation_metrics": ["attention stability", "no-attention ablation", "QINI delta", "segment-level policy value"],
        "producer_change": "Add a guarded EFIN attention collector that stores aggregate attention, not full user-level tensors by default.",
        "pass_gate": "Attention distributions are finite, row-normalized and stable enough across seeds or ablations to discuss.",
        "fail_action": "Treat attention as debugging telemetry, not model explanation, until ablation/stability passes.",
        "interview_line": "EFIN attention needs stability and ablation evidence before it becomes an interpretability claim.",
    },
    "EFIN.uplift_path_contribution": {
        "priority": "P1",
        "epoch_columns": ["valid_base_logit_norm", "valid_uplift_logit_norm", "valid_uplift_path_ratio"],
        "epoch_artifacts": ["efin_path_contribution_by_epoch.csv", "efin_path_policy_linkage.json"],
        "validation_metrics": ["uplift-path ratio", "policy value delta", "calibration MAE", "response-fit leakage check"],
        "producer_change": "Log aggregate c_logit/u_tau norms and their relation to validation policy value.",
        "pass_gate": "Path contribution is finite and explains whether EFIN is learning uplift signal or only response propensity.",
        "fail_action": "Block uplift-path interpretation and compare against T/DR learner calibration.",
        "interview_line": "The uplift-path bridge prevents overclaiming response fit as CATE learning.",
    },
    "DESCN.propensity_and_exposure": {
        "priority": "P0",
        "epoch_columns": ["valid_p_prpsy_ess", "valid_p_prpsy_calibration_mae", "valid_overlap_tail_share"],
        "epoch_artifacts": ["descn_propensity_overlap_by_epoch.csv", "descn_exposure_drift.json"],
        "validation_metrics": ["ESS", "overlap tail share", "exposure drift", "DR policy sensitivity"],
        "producer_change": "Add DESCN propensity/exposure aggregation for train/valid batches.",
        "pass_gate": "p_prpsy is finite/bounded, calibrated enough, and does not collapse in overlap tails.",
        "fail_action": "Keep DESCN as research-only for biased assignment settings until exposure diagnostics pass.",
        "interview_line": "DESCN entire-space modeling must prove exposure overlap, otherwise the extra heads can hide selection bias.",
    },
    "DESCN.entire_space_heads": {
        "priority": "P0",
        "epoch_columns": ["valid_tau_std", "valid_mu0_calibration_mae", "valid_mu1_calibration_mae", "valid_head_spread"],
        "epoch_artifacts": ["descn_head_outputs_by_epoch.csv", "descn_head_calibration.json"],
        "validation_metrics": ["mu0/mu1 calibration", "tau variance", "PEHE", "ATE error"],
        "producer_change": "Store aggregate mu0/mu1/tau/estr/escr head statistics and calibration summaries.",
        "pass_gate": "All heads are finite, shape-consistent and calibrated enough to support CATE accuracy claims.",
        "fail_action": "Use DESCN only as architecture-deconstruction evidence until head calibration is stable.",
        "interview_line": "DESCN's many heads become useful only when each head has calibration and consistency evidence.",
    },
    "DESCN.cross_head_constraints": {
        "priority": "P0",
        "epoch_columns": ["train_cross_head_residual", "valid_cross_head_residual", "cross_constraint_loss_share"],
        "epoch_artifacts": ["descn_cross_constraint_by_epoch.csv", "descn_no_constraint_ablation.json"],
        "validation_metrics": ["constraint residual", "constraint ablation PEHE", "QINI delta", "policy value delta"],
        "producer_change": "Expose cross-head residual/loss components through an optional ESX loss component dictionary.",
        "pass_gate": "Constraint residual is finite and an ablation shows whether it helps CATE or policy ranking.",
        "fail_action": "Avoid claiming cross-head constraints improve uplift until ablation evidence exists.",
        "interview_line": "DESCN constraint telemetry turns a complex loss into a falsifiable ablation claim.",
    },
}


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def latest(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def flatten(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return "" if value is None else str(value)


def history_columns(row: dict[str, Any]) -> list[str]:
    history = row.get("history") or []
    if history and isinstance(history[0], dict):
        return list(history[0].keys())
    run_dir = Path(str(row.get("run_dir") or ""))
    history_path = run_dir / "train_history.csv"
    if history_path.is_file():
        with history_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            return next(reader, [])
    return []


def artifact_exists(path_value: str) -> bool:
    path = Path(path_value)
    return path.exists()


def build_rows(root: Path, hook_smoke: dict[str, Any], training_evidence: dict[str, Any]) -> list[dict[str, Any]]:
    hook_rows = hook_smoke.get("rows") or []
    training_rows = {
        str(row.get("model")): row
        for row in training_evidence.get("models", [])
        if isinstance(row, dict) and row.get("model")
    }
    rows: list[dict[str, Any]] = []
    for hook in hook_rows:
        contract_id = str(hook.get("contract_id") or "")
        spec = BRIDGE_SPECS.get(contract_id)
        if spec is None:
            continue
        model = str(hook.get("model") or "")
        training = training_rows.get(model, {})
        current_artifacts = [str(item) for item in hook.get("artifact_files") or []]
        current_artifact_count = sum(1 for item in current_artifacts if artifact_exists(item))
        train_run_dir = str(training.get("run_dir") or "")
        manifest = str(training.get("evidence_manifest") or "")
        current_columns = history_columns(training)
        history = training.get("history") or []
        sidecar_ok = hook.get("status") == "ok" and current_artifact_count == len(current_artifacts)
        training_ok = (
            training.get("status") == "ok"
            and bool(train_run_dir)
            and Path(train_run_dir).exists()
            and bool(manifest)
            and Path(manifest).exists()
            and len(current_columns) >= 4
        )
        bridge_status = "bridge_ready" if sidecar_ok and training_ok else "review"
        rows.append(
            {
                "bridge_id": f"{contract_id}.trainer_bridge",
                "model": model,
                "contract_id": contract_id,
                "hook_name": hook.get("hook_name"),
                "priority": spec["priority"],
                "bridge_status": bridge_status,
                "guarded": True,
                "current_forward_status": hook.get("status"),
                "current_training_status": training.get("status") or "missing",
                "train_run_id": training.get("run_id") or "",
                "train_run_dir": train_run_dir,
                "evidence_manifest": manifest,
                "epochs_observed": len(history) if history else 0,
                "current_training_columns": current_columns,
                "current_forward_artifacts": current_artifacts,
                "current_forward_artifacts_present": current_artifact_count,
                "proposed_epoch_columns": spec["epoch_columns"],
                "proposed_epoch_artifacts": spec["epoch_artifacts"],
                "validation_metrics": spec["validation_metrics"],
                "producer_change": spec["producer_change"],
                "consumer_surfaces": ["Evidence Dashboard", "Model Deconstruction UI", "Agent 2.0", "interview evidence pack"],
                "compatibility_rule": "Default forward/loss APIs remain unchanged; trainer collectors are optional and guarded.",
                "pass_gate": spec["pass_gate"],
                "fail_action": spec["fail_action"],
                "next_migration_step": "Add opt-in trainer collector, write per-epoch sidecar, then link metric movement to benchmark rows.",
                "current_metric_link": {
                    "qini_score": training.get("qini_score"),
                    "auuc_score": training.get("auuc_score"),
                    "calibration_mae": training.get("calibration_mae"),
                    "train_loss_delta": (training.get("loss_summary") or {}).get("train_loss_delta"),
                    "aux_loss_last": training.get("aux_loss_last"),
                },
                "safe_claim": "Forward sidecar artifacts and training run evidence can be joined by contract; per-epoch trainer telemetry is the next implementation step.",
                "blocked_claim": "Do not claim paper-level internal telemetry or causal interpretability until proposed epoch artifacts and ablations pass.",
                "interview_line": spec["interview_line"],
            }
        )
    return rows


def build_payload(root: Path, rows: list[dict[str, Any]], hook_path: Path | None, training_path: Path | None) -> dict[str, Any]:
    model_counts: dict[str, int] = {}
    priority_counts: dict[str, int] = {}
    columns: set[str] = set()
    artifacts: set[str] = set()
    for row in rows:
        model_counts[str(row.get("model"))] = model_counts.get(str(row.get("model")), 0) + 1
        priority_counts[str(row.get("priority"))] = priority_counts.get(str(row.get("priority")), 0) + 1
        columns.update(str(item) for item in row.get("proposed_epoch_columns") or [])
        artifacts.update(str(item) for item in row.get("proposed_epoch_artifacts") or [])
    bridge_ready = sum(1 for row in rows if row.get("bridge_status") == "bridge_ready")
    return {
        "schema_version": 1,
        "status": "ok" if len(rows) >= 8 and bridge_ready == len(rows) else "review",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "source_artifacts": {
            "forward_hook_smoke": str(hook_path) if hook_path else "",
            "deep_model_training_evidence": str(training_path) if training_path else "",
        },
        "summary": {
            "models": len(model_counts),
            "bridges": len(rows),
            "bridge_ready": bridge_ready,
            "p0_bridges": priority_counts.get("P0", 0),
            "priority_counts": priority_counts,
            "proposed_epoch_columns": len(columns),
            "proposed_epoch_artifacts": len(artifacts),
            "current_forward_artifacts_present": sum(int(row.get("current_forward_artifacts_present") or 0) for row in rows),
            "model_counts": model_counts,
        },
        "rows": rows,
        "interview_tracks": {
            "30s": "Hook smoke proves tensors can be exported; the training bridge shows exactly how to promote those tensors into per-epoch evidence.",
            "5min": "For CFRNet/DragonNet/EFIN/DESCN, each bridge row names current sidecar proof, current training run proof, proposed train_history columns, pass gates and fail actions.",
            "15min": "This is the safe migration layer before model-source refactors: keep default APIs stable, add opt-in collectors, write sidecars, then require PEHE/QINI/policy or ablation movement before stronger claims.",
        },
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "bridge_id",
        "model",
        "contract_id",
        "hook_name",
        "priority",
        "bridge_status",
        "guarded",
        "current_forward_status",
        "current_training_status",
        "train_run_id",
        "train_run_dir",
        "evidence_manifest",
        "epochs_observed",
        "current_training_columns",
        "current_forward_artifacts",
        "current_forward_artifacts_present",
        "proposed_epoch_columns",
        "proposed_epoch_artifacts",
        "validation_metrics",
        "producer_change",
        "consumer_surfaces",
        "compatibility_rule",
        "pass_gate",
        "fail_action",
        "next_migration_step",
        "current_metric_link",
        "safe_claim",
        "blocked_claim",
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
    overview = "\n".join(
        "| {model} | {contract} | {priority} | {status} | {columns} | {gate} |".format(
            model=str(row.get("model", "")).replace("|", "/"),
            contract=str(row.get("contract_id", "")).replace("|", "/"),
            priority=str(row.get("priority", "")).replace("|", "/"),
            status=str(row.get("bridge_status", "")).replace("|", "/"),
            columns=", ".join(row.get("proposed_epoch_columns") or []).replace("|", "/"),
            gate=str(row.get("pass_gate", "")).replace("|", "/"),
        )
        for row in rows
    )
    details = []
    for row in rows:
        details.append(
            f"""## {row.get("bridge_id")}

Model: `{row.get("model")}`
Contract: `{row.get("contract_id")}`
Status: `{row.get("bridge_status")}`
Training run: `{row.get("train_run_dir")}`
Evidence manifest: `{row.get("evidence_manifest")}`

Current proof:
- Forward artifacts present: `{row.get("current_forward_artifacts_present")}` / `{len(row.get("current_forward_artifacts") or [])}`
- Current train history columns: `{", ".join(row.get("current_training_columns") or [])}`
- Current metric link: `{json.dumps(row.get("current_metric_link") or {}, ensure_ascii=False, sort_keys=True)}`

Proposed trainer telemetry:
- Epoch columns: `{", ".join(row.get("proposed_epoch_columns") or [])}`
- Epoch artifacts: `{", ".join(row.get("proposed_epoch_artifacts") or [])}`
- Validation metrics: `{", ".join(row.get("validation_metrics") or [])}`

Producer change: {row.get("producer_change")}

Compatibility rule: {row.get("compatibility_rule")}

Pass gate: {row.get("pass_gate")}

Fail action: {row.get("fail_action")}

Safe claim: {row.get("safe_claim")}

Blocked claim: {row.get("blocked_claim")}

Interview line: {row.get("interview_line")}
"""
        )
    return f"""# DeepUplift Deep Model Hook Training Bridge

Generated at: `{payload.get("generated_at")}`

This bridge connects two evidence layers that already exist:

1. `reports/deep_model_forward_hook_smoke_latest.json` proves selected internal tensors can be exported as guarded sidecar artifacts.
2. `reports/deep_model_training_evidence_latest.json` proves the same model families have runnable training runs, history, metrics and evidence manifests.

The bridge does not change model APIs yet. It defines the next safe trainer-level contract: optional collectors, per-epoch columns, sidecar artifact names, pass gates and failure actions. This keeps the demo stable while making the next deep-model refactor reviewable.

## Summary

| Metric | Value |
| --- | ---: |
| Models | {summary.get("models", 0)} |
| Bridges | {summary.get("bridges", 0)} |
| Bridge-ready rows | {summary.get("bridge_ready", 0)} |
| P0 bridges | {summary.get("p0_bridges", 0)} |
| Proposed epoch columns | {summary.get("proposed_epoch_columns", 0)} |
| Proposed epoch artifacts | {summary.get("proposed_epoch_artifacts", 0)} |
| Current forward artifacts present | {summary.get("current_forward_artifacts_present", 0)} |

Source artifacts: `{json.dumps(payload.get("source_artifacts") or {}, ensure_ascii=False, sort_keys=True)}`

## Interview Tracks

- 30s: {tracks.get("30s", "")}
- 5min: {tracks.get("5min", "")}
- 15min: {tracks.get("15min", "")}

## Bridge Overview

| Model | Contract | Priority | Status | Proposed epoch columns | Pass gate |
| --- | --- | --- | --- | --- | --- |
{overview}

{chr(10).join(details)}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate trainer-level bridge contracts from deep model hook smoke artifacts.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--json-output", default="reports/deep_model_hook_training_bridge_latest.json")
    parser.add_argument("--csv-output", default="reports/deep_model_hook_training_bridge_latest.csv")
    parser.add_argument("--doc-output", default="docs/DEEPUplift_DEEP_MODEL_HOOK_TRAINING_BRIDGE.md")
    args = parser.parse_args()

    root = Path(args.root)
    reports_dir = root / "reports"
    hook_path = latest(reports_dir, "deep_model_forward_hook_smoke_latest.json")
    training_path = latest(reports_dir, "deep_model_training_evidence_latest.json")
    hook_smoke = read_json(hook_path) if hook_path else {}
    training_evidence = read_json(training_path) if training_path else {}
    rows = build_rows(root, hook_smoke, training_evidence)
    payload = build_payload(root, rows, hook_path, training_path)

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
                "bridges": (payload.get("summary") or {}).get("bridges"),
                "bridge_ready": (payload.get("summary") or {}).get("bridge_ready"),
            },
            ensure_ascii=False,
        )
    )
    if payload.get("status") != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
