from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNNER_SCRIPT = "scripts/run_deep_model_tuned_benchmark.py"
SUPPORTED_RUNNER_FLAGS = {
    "--manifest",
    "--preset",
    "--datasets",
    "--variants",
    "--models",
    "--include-variants",
    "--seeds",
    "--rows",
    "--known-effect-bootstrap-samples",
    "--artifacts-dir",
    "--output-prefix",
    "--no-update-latest",
    "--emit-battle-cards",
    "--strict",
}
PLANNING_ONLY_FLAG_REPLACEMENTS = {
    "--models": "Native runner alias: selects all variants for the listed model names and is merged with `--variants`.",
    "--include-variants": "Native runner alias: adds explicit variant ids and is merged with `--variants`.",
    "--emit-battle-cards": "Native compatibility flag; battle cards are always emitted by the tuned benchmark runner.",
}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def flatten(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return "" if value is None else str(value)


def gate_by_variant(gate_payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = {}
    for row in gate_payload.get("rows") or []:
        if not isinstance(row, dict):
            continue
        for variant in row.get("required_variants") or []:
            index.setdefault(str(variant), []).append(row)
    return index


def interpretation_by_variant(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = {}
    for row in payload.get("rows") or []:
        if isinstance(row, dict) and row.get("variant_id"):
            index.setdefault(str(row.get("variant_id")), []).append(row)
    return index


def command_for_row(row: dict[str, Any]) -> str:
    tier = str(row.get("claim_tier") or "")
    model = row.get("model")
    variant = row.get("variant_id")
    python_bin = os.environ.get("PYTHON_BIN") or sys.executable
    if tier.startswith("B_"):
        return (
            f"{python_bin} {RUNNER_SCRIPT} --preset tuned-v1 --rows 800 "
            "--known-effect-bootstrap-samples 32 --models TLearnerGBM DRLearnerGBM "
            f"{model} --include-variants {variant}"
        )
    if tier.startswith("C_"):
        return (
            f"{python_bin} {RUNNER_SCRIPT} --preset tuned-v1 --rows 600 "
            "--known-effect-bootstrap-samples 24 --emit-battle-cards "
            f"--models TLearnerGBM DRLearnerGBM {model} --include-variants {variant}"
        )
    return (
        f"{python_bin} {RUNNER_SCRIPT} --preset tuned-smoke --rows 240 "
        f"--models {model} --include-variants {variant}"
    )


def runner_supported_command_for_row(row: dict[str, Any]) -> str:
    model = str(row.get("model") or "")
    variant = str(row.get("variant_id") or "")
    python_bin = os.environ.get("PYTHON_BIN") or sys.executable
    variants = ["TLearnerGBM_baseline", "DRLearnerGBM_baseline"]
    if variant:
        variants.append(variant)
    elif model:
        variants.append(model)
    unique_variants = " ".join(dict.fromkeys(variants))
    return (
        f"{python_bin} {RUNNER_SCRIPT} --preset tuned-smoke "
        "--datasets acic_style_synthetic_known_cate_1k6 --seeds 20260520 "
        "--rows 120 --known-effect-bootstrap-samples 2 "
        "--artifacts-dir reports/deep_model_ablation_command_contract_smoke_runs "
        "--output-prefix deep_model_ablation_command_contract_smoke --no-update-latest "
        f"--variants {unique_variants}"
    )


def command_flags(command: str) -> list[str]:
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    return [token for token in tokens if token.startswith("--")]


def command_contract(command: str) -> dict[str, Any]:
    flags = command_flags(command)
    unsupported = [flag for flag in flags if flag not in SUPPORTED_RUNNER_FLAGS]
    return {
        "flags": flags,
        "supported_flags": [flag for flag in flags if flag in SUPPORTED_RUNNER_FLAGS],
        "unsupported_flags": unsupported,
        "unsupported_flag_replacements": {
            flag: PLANNING_ONLY_FLAG_REPLACEMENTS.get(flag, "No replacement registered; review runner CLI.")
            for flag in unsupported
        },
        "status": "runner_supported" if not unsupported else "planning_only_requires_runner_adapter",
    }


def plan_for_row(row: dict[str, Any], gates: list[dict[str, Any]], interpretation_rows: list[dict[str, Any]]) -> dict[str, Any]:
    tier = str(row.get("claim_tier") or "")
    variant = str(row.get("variant_id") or "")
    model = str(row.get("model") or "")
    support = int(row.get("support_rows") or 0)
    tradeoff = int(row.get("tradeoff_rows") or 0)
    baseline_stronger = int(row.get("baseline_still_stronger_rows") or 0)
    metrics = ["PEHE", "ATE error", "QINI", "AUUC", "policy_top10_oracle_value", "oracle_top10_recall"]
    expected = [
        "reports/deep_model_tuned_benchmark_latest.json",
        "reports/deep_model_ablation_interpretation_latest.json",
        "reports/deep_model_ablation_promotion_matrix_latest.json",
    ]

    if tier.startswith("B_"):
        priority = "P0"
        experiment_type = "seed_and_baseline_stress"
        hypothesis = (
            f"{variant} has local mechanism evidence for {model}; wider rows, bootstrap and baseline stress should show whether it is a robust candidate or only a small-slice artifact."
        )
        pass_gate = (
            "At least two dataset rows keep PEHE non-worse or improved versus default, one ranking/policy metric improves, and the baseline verdict is no worse than ranking_gain_accuracy_tradeoff."
        )
        fail_action = "Demote to tradeoff/coverage evidence, keep default as demo path, and move deep-model story to telemetry/source-level depth."
        owner = "ML researcher + benchmark owner"
    elif tier.startswith("C_"):
        priority = "P1"
        experiment_type = "metric_conflict_triage"
        hypothesis = (
            f"{variant} is a useful tradeoff case; calibration, Top-K and business-value sensitivity should identify whether the metric conflict is acceptable for a business policy."
        )
        pass_gate = (
            "QINI or policy value improves in the target business slice while PEHE degradation is bounded and the interview claim stays explicitly tradeoff-only."
        )
        fail_action = "Keep the variant as failure-attribution material; do not promote the mechanism or replace the default."
        owner = "Benchmark owner + product/ROI reviewer"
    else:
        priority = "P2"
        experiment_type = "coverage_or_implementation_repair"
        hypothesis = f"{variant} needs coverage or implementation repair before it can support any claim."
        pass_gate = "Variant appears in the latest ablation interpretation artifact with finite PEHE/QINI/policy deltas and no missing gate rows."
        fail_action = "Keep variant out of demo path and document it as missing or blocked evidence."
        owner = "Model implementation owner"

    if baseline_stronger:
        metrics.append("delta_vs_TLearner_or_DRLearner_baseline")
    if gates:
        metrics.extend(sorted({str(metric) for gate in gates for metric in gate.get("metrics_to_watch") or []}))
        expected.append("reports/deep_model_telemetry_ablation_gate_latest.json")
    if interpretation_rows:
        expected.append("reports/deep_model_ablation_interpretation_latest.csv")

    recommended_command = command_for_row(row)
    runnable_fallback_command = runner_supported_command_for_row(row)
    recommended_contract = command_contract(recommended_command)
    fallback_contract = command_contract(runnable_fallback_command)

    return {
        "model": model,
        "variant_id": variant,
        "priority": priority,
        "experiment_type": experiment_type,
        "claim_tier": row.get("claim_tier"),
        "promotion_decision": row.get("promotion_decision"),
        "promotion_score": row.get("promotion_score"),
        "support_rows": support,
        "tradeoff_rows": tradeoff,
        "baseline_still_stronger_rows": baseline_stronger,
        "hypothesis": hypothesis,
        "recommended_command": recommended_command,
        "runnable_fallback_command": runnable_fallback_command,
        "command_contract_status": (
            "fallback_ready_planner_needs_adapter"
            if recommended_contract["unsupported_flags"] and not fallback_contract["unsupported_flags"]
            else "runner_supported"
            if not fallback_contract["unsupported_flags"]
            else "fallback_needs_review"
        ),
        "planner_unsupported_flags": recommended_contract["unsupported_flags"],
        "planner_flag_replacements": recommended_contract["unsupported_flag_replacements"],
        "fallback_supported_flags": fallback_contract["supported_flags"],
        "fallback_unsupported_flags": fallback_contract["unsupported_flags"],
        "fallback_expected_scope": "One known-CATE dataset, one seed, selected baseline variants plus the candidate variant; intended as a smoke executable contract, not a full paper-level benchmark.",
        "metrics_to_watch": sorted(set(metrics)),
        "pass_gate": pass_gate,
        "fail_action": fail_action,
        "owner_role": owner,
        "expected_artifacts": sorted(set(expected)),
        "risk": row.get("failure_attribution") or "No failure attribution available.",
        "safe_interview_line": (
            f"For {variant}, I would run a {experiment_type} next: current tier={row.get('claim_tier')}, "
            f"support={support}, tradeoff={tradeoff}, baseline_stronger={baseline_stronger}. "
            "The pass gate decides whether this remains a local mechanism story or becomes stronger benchmark evidence."
        ),
        "source_refs": sorted(
            set(
                [
                    "reports/deep_model_ablation_promotion_matrix_latest.json",
                    "reports/deep_model_ablation_interpretation_latest.json",
                    "reports/deep_model_telemetry_ablation_gate_latest.json",
                    *[str(ref) for ref in row.get("evidence_refs") or []],
                ]
            )
        ),
    }


def build_payload(promotion: dict[str, Any], interpretation: dict[str, Any], gate: dict[str, Any]) -> dict[str, Any]:
    gates = gate_by_variant(gate)
    interp = interpretation_by_variant(interpretation)
    rows = [
        plan_for_row(row, gates.get(str(row.get("variant_id")), []), interp.get(str(row.get("variant_id")), []))
        for row in promotion.get("rows") or []
        if isinstance(row, dict) and row.get("variant_id")
    ]
    priority_counts: dict[str, int] = {}
    type_counts: dict[str, int] = {}
    command_status_counts: dict[str, int] = {}
    fallback_ready_rows = 0
    planner_adapter_rows = 0
    for row in rows:
        priority_counts[row["priority"]] = priority_counts.get(row["priority"], 0) + 1
        type_counts[row["experiment_type"]] = type_counts.get(row["experiment_type"], 0) + 1
        status = str(row.get("command_contract_status") or "unknown")
        command_status_counts[status] = command_status_counts.get(status, 0) + 1
        if not row.get("fallback_unsupported_flags"):
            fallback_ready_rows += 1
        if row.get("planner_unsupported_flags"):
            planner_adapter_rows += 1
    return {
        "schema_version": 1,
        "status": "ok" if rows else "empty",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "summary": {
            "rows": len(rows),
            "models": len({row.get("model") for row in rows}),
            "variants": len({row.get("variant_id") for row in rows}),
            "priority_counts": priority_counts,
            "experiment_type_counts": type_counts,
            "command_status_counts": command_status_counts,
            "fallback_ready_rows": fallback_ready_rows,
            "planner_adapter_rows": planner_adapter_rows,
            "p0_rows": priority_counts.get("P0", 0),
            "p1_rows": priority_counts.get("P1", 0),
        },
        "rows": rows,
        "interview_tracks": {
            "30s": "The next-experiment plan turns each ablation claim tier into a concrete command, metric set, pass gate and fail action.",
            "5min": "For B-tier variants, widen rows/seeds and stress the T/DR baseline. For C-tier variants, treat them as metric-conflict cases and review PEHE, QINI and policy value together.",
            "15min": "This layer answers the follow-up question after promotion matrix: what exact experiment would you run tomorrow, what would count as success, and how would you prevent over-claiming if it fails.",
            "30min": "Walk one variant from telemetry gate to ablation interpretation, promotion tier, experiment command, metrics, pass gate, expected artifacts and rollback of the claim if baseline or stability still blocks it.",
            "command_contract": "The recommended command now uses runner-native aliases where available; the runnable fallback command remains isolated and audited by the command contract artifact.",
        },
        "source_artifacts": {
            "deep_model_ablation_promotion_matrix": "reports/deep_model_ablation_promotion_matrix_latest.json",
            "deep_model_ablation_interpretation": "reports/deep_model_ablation_interpretation_latest.json",
            "deep_model_telemetry_ablation_gate": "reports/deep_model_telemetry_ablation_gate_latest.json",
        },
    }


CSV_FIELDS = [
    "model",
    "variant_id",
    "priority",
    "experiment_type",
    "claim_tier",
    "promotion_decision",
    "promotion_score",
    "support_rows",
    "tradeoff_rows",
    "baseline_still_stronger_rows",
    "hypothesis",
    "recommended_command",
    "runnable_fallback_command",
    "command_contract_status",
    "planner_unsupported_flags",
    "planner_flag_replacements",
    "fallback_supported_flags",
    "fallback_unsupported_flags",
    "fallback_expected_scope",
    "metrics_to_watch",
    "pass_gate",
    "fail_action",
    "owner_role",
    "expected_artifacts",
    "risk",
    "safe_interview_line",
    "source_refs",
]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: flatten(row.get(field)) for field in CSV_FIELDS})


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> list[str]:
    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = []
        for key, _ in columns:
            value = row.get(key)
            if isinstance(value, (dict, list)):
                value = flatten(value)
            values.append(str(value if value not in (None, "") else "NA").replace("|", "/"))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    summary = payload.get("summary") or {}
    rows = payload.get("rows") or []
    tracks = payload.get("interview_tracks") or {}
    lines = [
        "# DeepUplift Deep Model Ablation Next Experiment Plan",
        "",
        f"Generated at: `{payload.get('generated_at')}`",
        "",
        "This report turns ablation promotion decisions into concrete next experiments. It is designed for the interview follow-up: what exactly would you run next, and what evidence would change your claim?",
        "",
        "## Summary",
        "",
        f"- Rows: `{summary.get('rows')}`",
        f"- Priority counts: `{json.dumps(summary.get('priority_counts') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- Experiment type counts: `{json.dumps(summary.get('experiment_type_counts') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- Command contract status: `{json.dumps(summary.get('command_status_counts') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- Runnable fallback rows: `{summary.get('fallback_ready_rows')}`",
        "",
        "## Interview Tracks",
        "",
        f"- 30 seconds: {tracks.get('30s', '')}",
        f"- 5 minutes: {tracks.get('5min', '')}",
        f"- 15 minutes: {tracks.get('15min', '')}",
        f"- 30 minutes: {tracks.get('30min', '')}",
        f"- Command contract: {tracks.get('command_contract', '')}",
        "",
        "## Experiment Plan",
        "",
    ]
    lines.extend(
        markdown_table(
            rows,
            [
                ("priority", "Priority"),
                ("model", "Model"),
                ("variant_id", "Variant"),
                ("experiment_type", "Experiment"),
                ("claim_tier", "Claim tier"),
                ("promotion_decision", "Promotion decision"),
                ("command_contract_status", "Command contract"),
                ("pass_gate", "Pass gate"),
                ("fail_action", "Fail action"),
            ],
        )
    )
    lines.extend(["", "## Runnable Plans", ""])
    for row in rows:
        lines.extend(
            [
                f"### {row.get('priority')} / {row.get('variant_id')}",
                "",
                f"- Hypothesis: {row.get('hypothesis')}",
                f"- Planning command: `{row.get('recommended_command')}`",
                f"- Runnable fallback command: `{row.get('runnable_fallback_command')}`",
                f"- Unsupported planning flags: `{', '.join(row.get('planner_unsupported_flags') or []) or 'none'}`",
                f"- Command contract status: `{row.get('command_contract_status')}`",
                f"- Metrics to watch: `{', '.join(row.get('metrics_to_watch') or [])}`",
                f"- Pass gate: {row.get('pass_gate')}",
                f"- Fail action: {row.get('fail_action')}",
                f"- Interview line: {row.get('safe_interview_line')}",
                "",
            ]
        )
    lines.extend(["## Evidence Links", ""])
    for name, artifact in (payload.get("source_artifacts") or {}).items():
        lines.append(f"- {name}: `{artifact}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate next-experiment plans for deep-model ablation variants.")
    parser.add_argument("--ablation-promotion", default="reports/deep_model_ablation_promotion_matrix_latest.json")
    parser.add_argument("--ablation-interpretation", default="reports/deep_model_ablation_interpretation_latest.json")
    parser.add_argument("--ablation-gate", default="reports/deep_model_telemetry_ablation_gate_latest.json")
    parser.add_argument("--json-output", default="reports/deep_model_ablation_next_experiment_plan_latest.json")
    parser.add_argument("--csv-output", default="reports/deep_model_ablation_next_experiment_plan_latest.csv")
    parser.add_argument("--doc-output", default="docs/DEEPUplift_DEEP_MODEL_ABLATION_NEXT_EXPERIMENT_PLAN.md")
    args = parser.parse_args()

    promotion = read_json(ROOT / args.ablation_promotion)
    interpretation = read_json(ROOT / args.ablation_interpretation)
    gate = read_json(ROOT / args.ablation_gate)
    payload = build_payload(promotion, interpretation, gate)
    write_json(ROOT / args.json_output, payload)
    write_csv(ROOT / args.csv_output, payload.get("rows") or [])
    write_markdown(ROOT / args.doc_output, payload)
    print(
        json.dumps(
            {
                "status": payload.get("status"),
                "json": args.json_output,
                "csv": args.csv_output,
                "markdown": args.doc_output,
                "rows": (payload.get("summary") or {}).get("rows"),
                "p0_rows": (payload.get("summary") or {}).get("p0_rows"),
                "p1_rows": (payload.get("summary") or {}).get("p1_rows"),
            },
            ensure_ascii=False,
        )
    )
    if payload.get("status") != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
