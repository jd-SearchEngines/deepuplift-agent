from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PYTHON = "python3"


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


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug[:96] or "unknown"


def parse_dataset_and_model(entity: str) -> tuple[str, str]:
    if " / " in entity:
        left, right = entity.split(" / ", 1)
        return left.strip(), right.strip()
    return entity.strip(), ""


def experiment_type_for(gate: str, domain: str) -> str:
    if gate == "blocked":
        return "blocker_repair_rerun"
    if gate == "paper_review_candidate" or domain in {"paper_benchmark", "deep_model_battle"}:
        return "paper_benchmark_or_ablation_review"
    if gate == "shadow_candidate":
        return "shadow_scoring_and_holdout_design"
    return "evidence_refresh_and_monitoring"


def priority_for(gate: str) -> str:
    return {
        "blocked": "P0",
        "shadow_candidate": "P0",
        "paper_review_candidate": "P1",
        "monitor": "P2",
    }.get(gate, "P2")


def command_for(row: dict[str, Any], experiment_type: str) -> str:
    entity = str(row.get("entity") or "")
    dataset_id, model = parse_dataset_and_model(entity)
    if experiment_type == "paper_benchmark_or_ablation_review":
        if model or entity in {"CFRNet", "DragonNet", "EFIN", "DESCN", "TLearnerGBM", "DRLearnerGBM"}:
            selected_model = model or entity
            return (
                f"{PYTHON} scripts/run_paper_benchmark_suite.py --preset benchmark-v3 "
                f"--models {selected_model} --seeds 13 29 47 --rows 600 --known-effect-bootstrap-samples 30"
            )
        return f"{PYTHON} scripts/run_paper_benchmark_suite.py --preset benchmark-v3 --seeds 13 29 47 --rows 600 --known-effect-bootstrap-samples 30"
    if experiment_type in {"blocker_repair_rerun", "shadow_scoring_and_holdout_design"} and dataset_id.startswith("synthetic_"):
        rows = 1000 if experiment_type == "shadow_scoring_and_holdout_design" else 600
        return f"{PYTHON} scripts/smoke_industrial_scenario_compare.py --rows {rows} --dataset-id {dataset_id}"
    if experiment_type == "evidence_refresh_and_monitoring":
        return f"{PYTHON} scripts/generate_evidence_store.py --limit 300"
    return f"{PYTHON} scripts/generate_regression_warning_audit.py"


def refresh_command_for() -> str:
    return (
        f"{PYTHON} scripts/generate_regression_warning_audit.py && "
        f"{PYTHON} scripts/generate_promotion_launch_cards.py && "
        f"{PYTHON} scripts/generate_regression_warning_triage.py && "
        f"{PYTHON} scripts/generate_pilot_readiness_scorecard.py && "
        f"{PYTHON} scripts/generate_pilot_experiment_plan.py"
    )


def metrics_for(row: dict[str, Any], experiment_type: str) -> list[str]:
    metrics = ["pilot_gate", "readiness_score", "launch_blockers", "required_artifacts"]
    risk_families = set(str(item) for item in row.get("risk_families") or [])
    if experiment_type == "paper_benchmark_or_ablation_review":
        metrics += ["PEHE", "ATE error", "QINI", "AUUC", "policy value", "bootstrap CI", "baseline delta"]
    if experiment_type in {"blocker_repair_rerun", "shadow_scoring_and_holdout_design"}:
        metrics += ["policy value", "ROI/iROAS", "Top-K observed uplift", "weak overlap rate", "post-trim QINI", "calibration"]
    if "refutation_stability" in risk_families:
        metrics += ["refutation p-value", "bootstrap CI"]
    if "support_and_overlap" in risk_families:
        metrics += ["ESS", "overlap trim stability"]
    return sorted(set(metrics))


def pass_gate_for(row: dict[str, Any], experiment_type: str) -> str:
    if experiment_type == "blocker_repair_rerun":
        return "pilot_gate moves from blocked to paper_review_candidate/shadow_candidate and launch_blockers decrease."
    if experiment_type == "paper_benchmark_or_ablation_review":
        return "multi-seed PEHE/ATE/QINI/policy value are stable, bootstrap CI is not obviously negative, and baseline delta is explained."
    if experiment_type == "shadow_scoring_and_holdout_design":
        return "shadow scores produce non-negative policy value, overlap/ESS are acceptable, and holdout/A/B design plus rollback metric is documented."
    return "evidence store refreshes, run diff is ok, and no new high-severity governance blocker appears."


def fail_action_for(row: dict[str, Any], experiment_type: str) -> str:
    if experiment_type == "blocker_repair_rerun":
        return "Keep offline-only; open owner review for ROI/overlap root cause and avoid stronger resume claim."
    if experiment_type == "paper_benchmark_or_ablation_review":
        return "Demote to research backlog or source-deconstruction claim; do not claim paper-level reproduction."
    if experiment_type == "shadow_scoring_and_holdout_design":
        return "Do not enter A/B; collect randomized exploration or redesign policy threshold/cost objective."
    return "Keep monitor-only and refresh artifact manifest before any review."


def claim_upgrade_for(row: dict[str, Any], experiment_type: str) -> str:
    if experiment_type == "blocker_repair_rerun":
        return "Blocked -> paper review/shadow only after blocker disappears in refreshed scorecard."
    if experiment_type == "paper_benchmark_or_ablation_review":
        return "Research/source claim -> paper-level benchmark candidate if metrics and failure attribution are stable."
    if experiment_type == "shadow_scoring_and_holdout_design":
        return "Shadow candidate -> small A/B candidate after OPE/holdout design and rollback conditions pass."
    return "Monitor -> low-risk shadow candidate if artifact freshness and guardrails remain clean."


def expected_artifacts_for(row: dict[str, Any], experiment_type: str) -> list[str]:
    artifacts = ["reports/pilot_readiness_scorecard_latest.json", "reports/pilot_experiment_plan_latest.json"]
    if experiment_type == "paper_benchmark_or_ablation_review":
        artifacts += [
            "reports/paper_benchmark_latest.json",
            "reports/paper_benchmark_interpretation_latest.json",
            "reports/deep_model_tuned_benchmark_latest.json",
        ]
    elif experiment_type in {"blocker_repair_rerun", "shadow_scoring_and_holdout_design"}:
        artifacts += [
            "reports/industrial_scenario_compare_smoke_latest.json",
            "reports/regression_warning_audit_latest.json",
            "reports/regression_warning_triage_latest.json",
            "reports/promotion_launch_cards_latest.json",
        ]
    else:
        artifacts += ["reports/evidence_store_latest.json", "reports/evidence_run_diff_latest.json"]
    return sorted(set(artifacts))


def build_plan_row(row: dict[str, Any]) -> dict[str, Any]:
    gate = str(row.get("pilot_gate") or "blocked")
    domain = str(row.get("domain") or "unknown")
    experiment_type = experiment_type_for(gate, domain)
    command = command_for(row, experiment_type)
    refresh = refresh_command_for()
    priority = priority_for(gate)
    return {
        "experiment_id": f"PEX-{slugify(str(row.get('scorecard_id') or row.get('entity') or 'unknown'))}",
        "scorecard_id": row.get("scorecard_id"),
        "domain": domain,
        "entity": row.get("entity"),
        "pilot_gate": gate,
        "readiness_score": row.get("readiness_score"),
        "evidence_grade": row.get("evidence_grade"),
        "priority": priority,
        "experiment_type": experiment_type,
        "owner_role": row.get("owner_role"),
        "hypothesis": (row.get("next_experiments") or ["Refresh evidence and compare gate movement."])[0],
        "recommended_command": command,
        "refresh_command": refresh,
        "command_status": "runner_supported_known_cli",
        "metrics_to_watch": metrics_for(row, experiment_type),
        "expected_artifacts": expected_artifacts_for(row, experiment_type),
        "pass_gate": pass_gate_for(row, experiment_type),
        "fail_action": fail_action_for(row, experiment_type),
        "claim_upgrade": claim_upgrade_for(row, experiment_type),
        "rollback_conditions": row.get("rollback_conditions") or [],
        "demo_route": "Evidence Dashboard -> Pilot Experiment Plan -> Pilot Readiness Scorecard",
        "interview_line": (
            f"`{row.get('entity')}` next experiment is `{experiment_type}` ({priority}); "
            f"pass gate: {pass_gate_for(row, experiment_type)}"
        ),
    }


def build_payload(scorecard_payload: dict[str, Any]) -> dict[str, Any]:
    scorecards = [row for row in scorecard_payload.get("scorecards") or [] if isinstance(row, dict)]
    rows = [build_plan_row(row) for row in scorecards]
    rows.sort(
        key=lambda row: (
            {"P0": 0, "P1": 1, "P2": 2, "P3": 3}.get(str(row.get("priority")), 9),
            {"shadow_scoring_and_holdout_design": 0, "blocker_repair_rerun": 1, "paper_benchmark_or_ablation_review": 2}.get(
                str(row.get("experiment_type")), 9
            ),
            -int(row.get("readiness_score") or 0),
        )
    )
    priority_counts = Counter(str(row.get("priority") or "unknown") for row in rows)
    experiment_counts = Counter(str(row.get("experiment_type") or "unknown") for row in rows)
    command_counts = Counter(str(row.get("command_status") or "unknown") for row in rows)
    return {
        "schema_version": 1,
        "status": "ok" if rows else "fail",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "summary": {
            "experiments": len(rows),
            "p0_rows": priority_counts.get("P0", 0),
            "p1_rows": priority_counts.get("P1", 0),
            "p2_rows": priority_counts.get("P2", 0),
            "priority_counts": dict(priority_counts),
            "experiment_type_counts": dict(experiment_counts),
            "command_status_counts": dict(command_counts),
            "source_pilot_gate": (scorecard_payload.get("summary") or {}).get("pilot_gate"),
            "shadow_or_monitor_rows": sum(1 for row in rows if row.get("pilot_gate") in {"shadow_candidate", "monitor"}),
            "paper_review_rows": sum(1 for row in rows if row.get("pilot_gate") == "paper_review_candidate"),
            "blocked_rows": sum(1 for row in rows if row.get("pilot_gate") == "blocked"),
        },
        "rows": rows,
        "interview_tracks": {
            "30s": "After readiness scoring, I convert each gate into a concrete experiment command, artifact list, pass gate and fail action.",
            "5min": "Show one shadow candidate and one blocked row: the same table tells me what to run, what artifact proves it, and when not to overclaim.",
            "15min": "Use priority -> experiment_type -> command -> expected_artifacts -> pass_gate -> claim_upgrade to explain a real industrial launch loop.",
            "30min": "Walk through offline repair, paper benchmark review, shadow scoring, holdout/A/B design and rollback. The plan is intentionally generated, validated and packaged.",
        },
        "source_artifacts": {
            "pilot_readiness_scorecard": "reports/pilot_readiness_scorecard_latest.json",
            "promotion_launch_cards": "reports/promotion_launch_cards_latest.json",
            "regression_warning_triage": "reports/regression_warning_triage_latest.json",
        },
    }


CSV_FIELDS = [
    "experiment_id",
    "scorecard_id",
    "domain",
    "entity",
    "pilot_gate",
    "readiness_score",
    "evidence_grade",
    "priority",
    "experiment_type",
    "owner_role",
    "hypothesis",
    "recommended_command",
    "refresh_command",
    "command_status",
    "metrics_to_watch",
    "expected_artifacts",
    "pass_gate",
    "fail_action",
    "claim_upgrade",
    "rollback_conditions",
    "demo_route",
    "interview_line",
]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: flatten(row.get(field)) for field in CSV_FIELDS})


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    summary = payload.get("summary") or {}
    rows = payload.get("rows") or []
    tracks = payload.get("interview_tracks") or {}
    lines = [
        "# DeepUplift Pilot Experiment Plan",
        "",
        f"Generated at: `{payload.get('generated_at')}`",
        "",
        "This report turns pilot readiness gates into concrete next experiments. It does not auto-promote a model; it defines what command to run, what artifact to inspect, what pass gate must hold, and how claims can be upgraded.",
        "",
        "## Summary",
        "",
        f"- Experiments: `{summary.get('experiments')}`",
        f"- P0 rows: `{summary.get('p0_rows')}`",
        f"- P1 rows: `{summary.get('p1_rows')}`",
        f"- P2 rows: `{summary.get('p2_rows')}`",
        f"- Experiment types: `{json.dumps(summary.get('experiment_type_counts') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- Command status: `{json.dumps(summary.get('command_status_counts') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Interview Tracks",
        "",
        f"- 30 seconds: {tracks.get('30s', '')}",
        f"- 5 minutes: {tracks.get('5min', '')}",
        f"- 15 minutes: {tracks.get('15min', '')}",
        f"- 30 minutes: {tracks.get('30min', '')}",
        "",
        "## Experiment Table",
        "",
        "| Priority | Type | Gate | Score | Entity | Command status | Pass gate |",
        "| --- | --- | --- | ---: | --- | --- | --- |",
    ]
    for row in rows[:80]:
        lines.append(
            f"| {row.get('priority')} | {row.get('experiment_type')} | {row.get('pilot_gate')} | "
            f"{row.get('readiness_score')} | {row.get('entity')} | {row.get('command_status')} | {row.get('pass_gate')} |"
        )
    lines.extend(["", "## Source Artifacts", ""])
    for name, artifact in (payload.get("source_artifacts") or {}).items():
        lines.append(f"- {name}: `{artifact}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate pilot next-experiment plan from pilot readiness scorecards.")
    parser.add_argument("--scorecard", default="reports/pilot_readiness_scorecard_latest.json")
    parser.add_argument("--json-output", default="reports/pilot_experiment_plan_latest.json")
    parser.add_argument("--csv-output", default="reports/pilot_experiment_plan_latest.csv")
    parser.add_argument("--doc-output", default="docs/DEEPUplift_PILOT_EXPERIMENT_PLAN.md")
    args = parser.parse_args()

    payload = build_payload(read_json(ROOT / args.scorecard))
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
                "experiments": (payload.get("summary") or {}).get("experiments"),
                "p0_rows": (payload.get("summary") or {}).get("p0_rows"),
                "p1_rows": (payload.get("summary") or {}).get("p1_rows"),
                "p2_rows": (payload.get("summary") or {}).get("p2_rows"),
            },
            ensure_ascii=False,
        )
    )
    if payload.get("status") != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
