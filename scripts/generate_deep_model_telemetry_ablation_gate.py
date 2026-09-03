from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any


GATE_SPECS: dict[str, dict[str, Any]] = {
    "CFRNet.phi_x_representation": {
        "ablation_family": "balance_weight_sweep",
        "required_variants": ["CFRNet_low_balance", "CFRNet_high_balance"],
        "command": "python3 scripts/run_deep_model_tuned_benchmark.py --preset tuned-v1 --rows 400 --known-effect-bootstrap-samples 16",
        "metrics_to_watch": ["valid_representation_mmd", "PEHE", "QINI", "policy value", "delta vs TLearner/DRLearner"],
        "pass_gate": "At least one balance variant improves PEHE or QINI versus baseline without widening stability/confidence risk, and MMD movement is finite across seeds.",
        "fail_action": "Keep CFRNet as architecture/telemetry evidence; explain that small-tabular baselines still win and tune representation weight before claim promotion.",
    },
    "DragonNet.e_hat_propensity": {
        "ablation_family": "propensity_overlap_clipping",
        "required_variants": ["DragonNet_no_tarreg", "DragonNet_stronger_tarreg"],
        "command": "python3 scripts/run_deep_model_tuned_benchmark.py --preset tuned-v1 --rows 400 --known-effect-bootstrap-samples 16",
        "metrics_to_watch": ["valid_propensity_ess", "valid_propensity_bce", "PEHE", "ATE error", "DR policy value"],
        "pass_gate": "Propensity ESS/calibration improves or remains stable while PEHE/ATE/QINI improves over the strongest T/DR baseline.",
        "fail_action": "Do not claim overlap robustness; route production candidate to DR/R learner and keep DragonNet as source/loss depth evidence.",
    },
    "DragonNet.targeted_regularization_components": {
        "ablation_family": "targeted_regularization_beta_sweep",
        "required_variants": ["DragonNet_no_tarreg", "DragonNet_stronger_tarreg"],
        "command": "python3 scripts/run_deep_model_tuned_benchmark.py --preset tuned-v1 --rows 400 --known-effect-bootstrap-samples 16",
        "metrics_to_watch": ["valid_targeted_regularization", "epsilon_mean", "PEHE", "ATE error", "QINI"],
        "pass_gate": "No-tarreg/stronger-tarreg comparison shows targeted regularization reduces PEHE or ATE error without ranking collapse.",
        "fail_action": "Describe tarreg as implemented and observable, not as empirically superior; keep beta sweep in P0 backlog.",
    },
    "EFIN.interaction_attention": {
        "ablation_family": "attention_removal_or_stability",
        "required_variants": ["EFIN_wide", "EFIN_no_attention"],
        "command": "python3 scripts/run_deep_model_tuned_benchmark.py --preset tuned-v1 --rows 400 --known-effect-bootstrap-samples 16",
        "metrics_to_watch": ["valid_attention_entropy", "attention_top_feature_stability", "QINI", "policy value", "segment policy value"],
        "pass_gate": "Attention variant improves ranking/policy or no-attention ablation degrades it; attention remains stable enough across seeds.",
        "fail_action": "Treat EFIN attention as debugging telemetry, not causal explanation; keep interpretation claim guarded.",
    },
    "EFIN.uplift_path_contribution": {
        "ablation_family": "uplift_path_ratio_stability",
        "required_variants": ["EFIN_wide", "EFIN_path_regularized"],
        "command": "python3 scripts/run_deep_model_tuned_benchmark.py --preset tuned-v1 --rows 400 --known-effect-bootstrap-samples 16",
        "metrics_to_watch": ["valid_uplift_path_ratio", "valid_base_logit_norm", "QINI", "policy value", "calibration"],
        "pass_gate": "Uplift-path ratio changes align with QINI/policy improvement and do not simply mirror response propensity.",
        "fail_action": "Keep EFIN as industrial architecture/ranking candidate; avoid claiming learned causal path contribution.",
    },
    "DESCN.propensity_and_exposure": {
        "ablation_family": "exposure_propensity_calibration",
        "required_variants": ["DESCN_default", "DESCN_wide_dropout"],
        "command": "python3 scripts/run_deep_model_tuned_benchmark.py --preset tuned-v1 --rows 400 --known-effect-bootstrap-samples 16",
        "metrics_to_watch": ["valid_p_prpsy_ess", "valid_propensity_loss", "PEHE", "QINI", "OPE clipping sensitivity"],
        "pass_gate": "Exposure/propensity telemetry remains bounded and benchmark metrics remain at least competitive on classification known-CATE datasets.",
        "fail_action": "Keep DESCN task boundary explicit; no biased-exposure robustness claim until calibration and OPE sensitivity pass.",
    },
    "DESCN.entire_space_heads": {
        "ablation_family": "head_calibration_and_spread",
        "required_variants": ["DESCN_default", "DESCN_wide_dropout"],
        "command": "python3 scripts/run_deep_model_tuned_benchmark.py --preset tuned-v1 --rows 400 --known-effect-bootstrap-samples 16",
        "metrics_to_watch": ["valid_tau_std", "valid_head_spread", "mu0/mu1 calibration", "PEHE", "oracle top-K recall"],
        "pass_gate": "Head spread/tau variance is finite and paired with PEHE/top-K recall improvement on classification benchmarks.",
        "fail_action": "Keep many-head architecture as source evidence; do not claim calibrated CATE heads without head-level calibration artifacts.",
    },
    "DESCN.cross_head_constraints": {
        "ablation_family": "cross_constraint_removal",
        "required_variants": ["DESCN_default", "DESCN_no_constraint"],
        "command": "python3 scripts/run_deep_model_tuned_benchmark.py --preset tuned-v1 --rows 400 --known-effect-bootstrap-samples 16",
        "metrics_to_watch": ["valid_cross_head_residual", "PEHE", "QINI", "policy value", "constraint residual"],
        "pass_gate": "Constraint residual decreases and the no-constraint comparison shows no worse PEHE/QINI or a clear policy benefit.",
        "fail_action": "Describe the constraint as implemented/observable, not as performance-improving, until the no-constraint ablation exists.",
    },
}


def latest(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def flatten(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return "" if value is None else str(value)


def scorecards_by_model(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("model")): row
        for row in payload.get("model_scorecards", [])
        if isinstance(row, dict) and row.get("model")
    }


def variant_coverage(required: list[str], available: list[str]) -> dict[str, Any]:
    available_set = set(available)
    present = [variant for variant in required if variant in available_set]
    missing = [variant for variant in required if variant not in available_set]
    if not missing:
        status = "covered_by_current_tuned_benchmark"
    elif present:
        status = "partial_variant_coverage"
    else:
        status = "requires_new_variant"
    return {"status": status, "present": present, "missing": missing}


def gate_decision(linkage_row: dict[str, Any], coverage: dict[str, Any]) -> str:
    baseline_verdict = str(linkage_row.get("tuned_baseline_verdict") or "")
    paper_verdict = str(linkage_row.get("paper_verdict") or "")
    if coverage["missing"]:
        return "blocked_until_variant_exists"
    if baseline_verdict == "baseline_still_stronger":
        return "blocked_until_ablation_beats_baseline"
    if paper_verdict in {"accuracy_and_ranking_candidate", "ranking_candidate"}:
        return "review_candidate_after_ablation"
    return "review_required"


def build_rows(linkage: dict[str, Any], tuned: dict[str, Any]) -> list[dict[str, Any]]:
    tuned_cards = scorecards_by_model(tuned)
    rows: list[dict[str, Any]] = []
    for link in linkage.get("rows") or []:
        contract_id = str(link.get("contract_id") or "")
        spec = GATE_SPECS.get(contract_id)
        if spec is None:
            continue
        model = str(link.get("model") or "")
        tuned_card = tuned_cards.get(model, {})
        available_variants = [str(item) for item in tuned_card.get("variants") or []]
        coverage = variant_coverage(spec["required_variants"], available_variants)
        decision = gate_decision(link, coverage)
        rows.append(
            {
                "gate_id": f"{contract_id}.ablation_gate",
                "model": model,
                "contract_id": contract_id,
                "telemetry_family": link.get("telemetry_family"),
                "ablation_family": spec["ablation_family"],
                "gate_decision": decision,
                "variant_coverage_status": coverage["status"],
                "available_variants": available_variants,
                "required_variants": spec["required_variants"],
                "missing_variants": coverage["missing"],
                "linkage_status": link.get("linkage_status"),
                "paper_verdict": link.get("paper_verdict"),
                "tuned_verdict": link.get("tuned_verdict"),
                "tuned_baseline_verdict": link.get("tuned_baseline_verdict"),
                "telemetry_last": link.get("telemetry_last"),
                "telemetry_delta": link.get("telemetry_delta"),
                "command": spec["command"],
                "metrics_to_watch": spec["metrics_to_watch"],
                "pass_gate": spec["pass_gate"],
                "fail_action": spec["fail_action"],
                "safe_claim": link.get("safe_claim"),
                "blocked_claim": link.get("blocked_claim"),
                "interview_line": (
                    f"{model} {spec['ablation_family']}: gate={decision}; "
                    f"coverage={coverage['status']}; pass gate={spec['pass_gate']}"
                ),
                "evidence_refs": [
                    "reports/deep_model_telemetry_benchmark_linkage_latest.json",
                    "reports/deep_model_tuned_benchmark_latest.json",
                    "reports/deep_model_trainer_collector_smoke_latest.json",
                ],
            }
        )
    return rows


def build_payload(rows: list[dict[str, Any]], source_paths: dict[str, Path | None]) -> dict[str, Any]:
    decision_counts: dict[str, int] = {}
    coverage_counts: dict[str, int] = {}
    for row in rows:
        decision = str(row.get("gate_decision") or "")
        decision_counts[decision] = decision_counts.get(decision, 0) + 1
        coverage = str(row.get("variant_coverage_status") or "")
        coverage_counts[coverage] = coverage_counts.get(coverage, 0) + 1
    return {
        "schema_version": 1,
        "status": "ok" if len(rows) >= 8 else "review",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "source_artifacts": {key: str(path) if path else "" for key, path in source_paths.items()},
        "summary": {
            "models": len({row.get("model") for row in rows}),
            "gates": len(rows),
            "decision_counts": decision_counts,
            "coverage_counts": coverage_counts,
            "missing_variant_rows": sum(1 for row in rows if row.get("missing_variants")),
        },
        "rows": rows,
        "interview_tracks": {
            "30s": "The ablation gate turns each internal telemetry claim into a pass/fail experiment contract.",
            "5min": "For each deep-model hook, it lists required variants, available coverage, command, metrics, pass gate, fail action, safe claim and blocked claim.",
            "15min": "This is the next step after telemetry-benchmark linkage: a reviewer can see exactly which ablation would upgrade a model from architecture evidence to benchmark-backed claim, and which claims remain blocked.",
        },
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "gate_id",
        "model",
        "contract_id",
        "telemetry_family",
        "ablation_family",
        "gate_decision",
        "variant_coverage_status",
        "available_variants",
        "required_variants",
        "missing_variants",
        "linkage_status",
        "paper_verdict",
        "tuned_verdict",
        "tuned_baseline_verdict",
        "telemetry_last",
        "telemetry_delta",
        "command",
        "metrics_to_watch",
        "pass_gate",
        "fail_action",
        "safe_claim",
        "blocked_claim",
        "interview_line",
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
    overview = "\n".join(
        "| {model} | {family} | {decision} | {coverage} | {missing} | {gate} |".format(
            model=str(row.get("model", "")).replace("|", "/"),
            family=str(row.get("ablation_family", "")).replace("|", "/"),
            decision=str(row.get("gate_decision", "")).replace("|", "/"),
            coverage=str(row.get("variant_coverage_status", "")).replace("|", "/"),
            missing=", ".join(row.get("missing_variants") or []).replace("|", "/"),
            gate=str(row.get("pass_gate", "")).replace("|", "/"),
        )
        for row in rows
    )
    details = []
    for row in rows:
        details.append(
            f"""## {row.get("gate_id")}

Model: `{row.get("model")}`
Contract: `{row.get("contract_id")}`
Ablation family: `{row.get("ablation_family")}`
Decision: `{row.get("gate_decision")}`

Variant coverage:
- Available: `{", ".join(row.get("available_variants") or [])}`
- Required: `{", ".join(row.get("required_variants") or [])}`
- Missing: `{", ".join(row.get("missing_variants") or []) or "none"}`

Command:

```bash
{row.get("command")}
```

Metrics to watch: `{", ".join(row.get("metrics_to_watch") or [])}`

Pass gate: {row.get("pass_gate")}

Fail action: {row.get("fail_action")}

Safe claim: {row.get("safe_claim")}

Blocked claim: {row.get("blocked_claim")}

Interview line: {row.get("interview_line")}
"""
        )
    return f"""# DeepUplift Deep Model Telemetry Ablation Gate

Generated at: `{payload.get("generated_at")}`

This report turns telemetry-benchmark linkage into executable ablation gates. It does not claim the ablations have already proven causality. Instead, it names the variants, command, metrics, pass gate and fail action required before a deep-model internal telemetry claim can be promoted.

## Summary

| Metric | Value |
| --- | ---: |
| Models | {summary.get("models", 0)} |
| Gates | {summary.get("gates", 0)} |
| Missing-variant rows | {summary.get("missing_variant_rows", 0)} |

Decision counts: `{json.dumps(summary.get("decision_counts") or {}, ensure_ascii=False, sort_keys=True)}`

Coverage counts: `{json.dumps(summary.get("coverage_counts") or {}, ensure_ascii=False, sort_keys=True)}`

Source artifacts: `{json.dumps(payload.get("source_artifacts") or {}, ensure_ascii=False, sort_keys=True)}`

## Interview Tracks

- 30s: {tracks.get("30s", "")}
- 5min: {tracks.get("5min", "")}
- 15min: {tracks.get("15min", "")}

## Gate Overview

| Model | Ablation family | Decision | Variant coverage | Missing variants | Pass gate |
| --- | --- | --- | --- | --- | --- |
{overview}

{chr(10).join(details)}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ablation gates from deep model telemetry-benchmark linkage.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--json-output", default="reports/deep_model_telemetry_ablation_gate_latest.json")
    parser.add_argument("--csv-output", default="reports/deep_model_telemetry_ablation_gate_latest.csv")
    parser.add_argument("--doc-output", default="docs/DEEPUplift_DEEP_MODEL_TELEMETRY_ABLATION_GATE.md")
    args = parser.parse_args()

    root = Path(args.root)
    reports_dir = root / "reports"
    source_paths = {
        "telemetry_benchmark_linkage": latest(reports_dir, "deep_model_telemetry_benchmark_linkage_latest.json"),
        "deep_model_tuned_benchmark": latest(reports_dir, "deep_model_tuned_benchmark_latest.json"),
    }
    linkage = read_json(source_paths["telemetry_benchmark_linkage"])
    tuned = read_json(source_paths["deep_model_tuned_benchmark"])
    rows = build_rows(linkage, tuned)
    payload = build_payload(rows, source_paths)

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
                "gates": (payload.get("summary") or {}).get("gates"),
                "missing_variant_rows": (payload.get("summary") or {}).get("missing_variant_rows"),
            },
            ensure_ascii=False,
        )
    )
    if payload.get("status") != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
