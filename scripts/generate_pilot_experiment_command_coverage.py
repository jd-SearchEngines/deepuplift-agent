from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


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


def smoke_rows_by_experiment(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in payload.get("rows") or []:
        if not isinstance(row, dict) or not row.get("experiment_id"):
            continue
        rows[str(row.get("experiment_id"))] = row
    return rows


def coverage_status(plan_row: dict[str, Any], smoke_row: dict[str, Any]) -> str:
    if smoke_row and smoke_row.get("status") == "ok":
        return "sampled_smoked_ok"
    if smoke_row:
        return "sampled_smoked_failed"
    if plan_row.get("command_status") == "runner_supported_known_cli":
        return "runner_supported_backlog"
    return "command_needs_review"


def transition_for(status: str) -> str:
    if status == "sampled_smoked_ok":
        return "plan_only_to_sampled_smoked"
    if status == "sampled_smoked_failed":
        return "plan_only_to_failed_smoke"
    if status == "runner_supported_backlog":
        return "plan_only_backlog"
    return "plan_needs_command_contract"


def evidence_level_for(status: str) -> str:
    if status == "sampled_smoked_ok":
        return "smoke_executed"
    if status == "sampled_smoked_failed":
        return "failed_smoke_evidence"
    if status == "runner_supported_backlog":
        return "planned_runner_supported"
    return "planned_only"


def next_action_for(row: dict[str, Any], status: str, total_rows: int) -> str:
    if status == "sampled_smoked_ok":
        return "Escalate this row from command smoke to the full pass gate: larger benchmark/OPE/shadow artifact and readiness refresh."
    if status == "sampled_smoked_failed":
        return "Fix the safe smoke runner or optional dependency, rerun command smoke, and keep the claim offline-only."
    if status == "runner_supported_backlog":
        experiment_id = row.get("experiment_id")
        experiment_type = row.get("experiment_type")
        return (
            "Expand command smoke coverage with "
            f"`python scripts/smoke_pilot_experiment_commands.py --limit {total_rows}` or `--all`; "
            f"prioritize `{experiment_id}` in `{experiment_type}` before claiming executed evidence."
        )
    return "Repair the command contract before smoke execution."


def build_rows(plan: dict[str, Any], smoke: dict[str, Any]) -> list[dict[str, Any]]:
    plan_rows = [row for row in plan.get("rows") or [] if isinstance(row, dict)]
    smoke_by_id = smoke_rows_by_experiment(smoke)
    total_rows = len(plan_rows)
    rows: list[dict[str, Any]] = []
    for plan_row in plan_rows:
        experiment_id = str(plan_row.get("experiment_id") or "")
        smoke_row = smoke_by_id.get(experiment_id) or {}
        status = coverage_status(plan_row, smoke_row)
        runner_artifact = smoke_row.get("runner_manifest") or smoke_row.get("runner_json") or smoke_row.get("runner_csv") or smoke_row.get("runner_doc")
        rows.append(
            {
                "experiment_id": experiment_id,
                "scorecard_id": plan_row.get("scorecard_id"),
                "domain": plan_row.get("domain"),
                "entity": plan_row.get("entity"),
                "pilot_gate": plan_row.get("pilot_gate"),
                "readiness_score": plan_row.get("readiness_score"),
                "evidence_grade": plan_row.get("evidence_grade"),
                "priority": plan_row.get("priority"),
                "experiment_type": plan_row.get("experiment_type"),
                "owner_role": plan_row.get("owner_role"),
                "command_status": plan_row.get("command_status"),
                "coverage_status": status,
                "transition": transition_for(status),
                "evidence_level": evidence_level_for(status),
                "sampled_smoked": bool(smoke_row),
                "sampled_smoked_ok": status == "sampled_smoked_ok",
                "smoke_status": smoke_row.get("status") if smoke_row else "",
                "smoke_returncode": smoke_row.get("returncode") if smoke_row else "",
                "smoke_elapsed_seconds": smoke_row.get("elapsed_seconds") if smoke_row else "",
                "runner_artifact": runner_artifact,
                "safe_smoke_command": smoke_row.get("safe_smoke_command") if smoke_row else "",
                "recommended_command": plan_row.get("recommended_command"),
                "pass_gate": plan_row.get("pass_gate"),
                "fail_action": plan_row.get("fail_action"),
                "claim_upgrade": plan_row.get("claim_upgrade"),
                "claim_boundary": (
                    "executed command smoke only; still needs pass_gate evidence"
                    if status == "sampled_smoked_ok"
                    else "planned command only; do not claim execution yet"
                ),
                "next_action": next_action_for(plan_row, status, total_rows),
                "demo_route": "Evidence Dashboard -> Pilot Experiment Command Coverage -> Pilot Experiment Command Smoke",
                "source_refs": [
                    "reports/pilot_experiment_plan_latest.json",
                    "reports/pilot_experiment_command_smoke_latest.json",
                ],
            }
        )
    return rows


def count_by(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(field) or "unknown")
        counts[value] = counts.get(value, 0) + 1
    return counts


def priority_coverage(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for row in rows:
        priority = str(row.get("priority") or "unknown")
        out.setdefault(priority, {"rows": 0, "sampled_smoked_ok": 0, "backlog": 0, "failed": 0})
        out[priority]["rows"] += 1
        if row.get("sampled_smoked_ok"):
            out[priority]["sampled_smoked_ok"] += 1
        elif row.get("coverage_status") == "sampled_smoked_failed":
            out[priority]["failed"] += 1
        else:
            out[priority]["backlog"] += 1
    return out


def build_payload(plan: dict[str, Any], smoke: dict[str, Any]) -> dict[str, Any]:
    rows = build_rows(plan, smoke)
    total_rows = len(rows)
    sampled_rows = sum(1 for row in rows if row.get("sampled_smoked"))
    ok_rows = sum(1 for row in rows if row.get("sampled_smoked_ok"))
    failed_rows = sum(1 for row in rows if row.get("coverage_status") == "sampled_smoked_failed")
    backlog_rows = total_rows - sampled_rows
    p0_rows = [row for row in rows if row.get("priority") == "P0"]
    p0_ok = sum(1 for row in p0_rows if row.get("sampled_smoked_ok"))
    smoke_summary = smoke.get("summary") or {}
    plan_summary = plan.get("summary") or {}
    sampled_gate = "ok" if sampled_rows and failed_rows == 0 else "review_required"
    full_coverage_gate = "ok" if total_rows and ok_rows == total_rows else "review_required"
    return {
        "schema_version": 1,
        "status": "ok" if rows and failed_rows == 0 else "fail",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "summary": {
            "plan_rows": total_rows,
            "source_plan_rows": plan_summary.get("experiments", total_rows),
            "source_smoke_rows": smoke_summary.get("smoke_rows", sampled_rows),
            "sampled_smoked_rows": sampled_rows,
            "sampled_smoked_ok_rows": ok_rows,
            "sampled_failed_rows": failed_rows,
            "backlog_rows": backlog_rows,
            "coverage_pct": round(ok_rows / max(total_rows, 1), 4),
            "sampled_success_pct": round(ok_rows / max(sampled_rows, 1), 4),
            "p0_rows": len(p0_rows),
            "p0_sampled_ok_rows": p0_ok,
            "p0_coverage_pct": round(p0_ok / max(len(p0_rows), 1), 4),
            "coverage_status_counts": count_by(rows, "coverage_status"),
            "transition_counts": count_by(rows, "transition"),
            "priority_coverage": priority_coverage(rows),
            "experiment_type_counts": count_by(rows, "experiment_type"),
            "sampled_smoke_gate": sampled_gate,
            "full_coverage_gate": full_coverage_gate,
            "coverage_gate_delta": f"plan_only -> sampled_smoke_{sampled_gate}",
        },
        "rows": rows,
        "interview_tracks": {
            "30s": f"I added pilot command coverage governance: {ok_rows}/{total_rows} planned pilot experiments now have sampled smoke execution, with the remaining rows explicit backlog.",
            "5min": "The matrix joins the generated pilot experiment plan with actual command-smoke results, so I can separate planned runner support from executed evidence.",
            "15min": "This is claim hygiene: plan rows prove the next experiment contract; command smoke proves safe execution; full benchmark/OPE/holdout artifacts are still required before launch claims.",
            "30min": "Walk one sampled-smoked P0 row, one unsmoked P0 backlog row, and one paper benchmark row. Then show how --limit/--all expands coverage without overwriting latest benchmark evidence.",
        },
        "safe_claims": [
            "Pilot experiment commands are generated for every current readiness row.",
            "The sampled command-smoke rows executed with safe tiny-row overrides and isolated outputs.",
            "Rows without sampled smoke are visible as backlog rather than hidden behind a stronger claim.",
        ],
        "unsafe_claims": [
            "Do not claim full pilot coverage while backlog_rows is non-zero.",
            "Do not treat command smoke as proof of model superiority or online incrementality.",
            "Do not enter A/B or ramp-up without pass_gate artifacts, OPE/holdout design, rollback metrics, and refreshed readiness.",
        ],
        "source_artifacts": {
            "pilot_experiment_plan": "reports/pilot_experiment_plan_latest.json",
            "pilot_experiment_command_smoke": "reports/pilot_experiment_command_smoke_latest.json",
        },
    }


CSV_FIELDS = [
    "priority",
    "experiment_id",
    "scorecard_id",
    "domain",
    "entity",
    "pilot_gate",
    "readiness_score",
    "evidence_grade",
    "experiment_type",
    "owner_role",
    "command_status",
    "coverage_status",
    "transition",
    "evidence_level",
    "sampled_smoked",
    "sampled_smoked_ok",
    "smoke_status",
    "smoke_returncode",
    "smoke_elapsed_seconds",
    "runner_artifact",
    "safe_smoke_command",
    "recommended_command",
    "pass_gate",
    "fail_action",
    "claim_upgrade",
    "claim_boundary",
    "next_action",
    "demo_route",
    "source_refs",
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
    tracks = payload.get("interview_tracks") or {}
    rows = payload.get("rows") or []
    lines = [
        "# DeepUplift Pilot Experiment Command Coverage",
        "",
        f"Generated at: `{payload.get('generated_at')}`",
        "",
        "This report joins the Pilot Experiment Plan with sampled Pilot Command Smoke results. It makes the evidence boundary explicit: every row can be planned and runner-supported, but only sampled-smoked rows have execution evidence.",
        "",
        "## Summary",
        "",
        f"- Plan rows: `{summary.get('plan_rows')}`",
        f"- Sampled-smoked OK rows: `{summary.get('sampled_smoked_ok_rows')}`",
        f"- Sampled failed rows: `{summary.get('sampled_failed_rows')}`",
        f"- Backlog rows: `{summary.get('backlog_rows')}`",
        f"- Coverage pct: `{summary.get('coverage_pct')}`",
        f"- P0 coverage pct: `{summary.get('p0_coverage_pct')}`",
        f"- Sampled smoke gate: `{summary.get('sampled_smoke_gate')}`",
        f"- Full coverage gate: `{summary.get('full_coverage_gate')}`",
        f"- Status counts: `{json.dumps(summary.get('coverage_status_counts') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Interview Tracks",
        "",
        f"- 30 seconds: {tracks.get('30s', '')}",
        f"- 5 minutes: {tracks.get('5min', '')}",
        f"- 15 minutes: {tracks.get('15min', '')}",
        f"- 30 minutes: {tracks.get('30min', '')}",
        "",
        "## Safe Claims",
        "",
    ]
    for claim in payload.get("safe_claims") or []:
        lines.append(f"- {claim}")
    lines.extend(["", "## Unsafe Claims", ""])
    for claim in payload.get("unsafe_claims") or []:
        lines.append(f"- {claim}")
    lines.extend(
        [
            "",
            "## Coverage Matrix",
            "",
            "| Priority | Experiment | Entity | Type | Coverage | Evidence level | Next action |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row.get('priority')} | {row.get('experiment_id')} | {row.get('entity')} | "
            f"{row.get('experiment_type')} | {row.get('coverage_status')} | "
            f"{row.get('evidence_level')} | {row.get('next_action')} |"
        )
    lines.extend(["", "## Evidence Links", ""])
    for name, artifact in (payload.get("source_artifacts") or {}).items():
        lines.append(f"- {name}: `{artifact}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate coverage matrix for pilot experiment command smoke.")
    parser.add_argument("--plan", default="reports/pilot_experiment_plan_latest.json")
    parser.add_argument("--command-smoke", default="reports/pilot_experiment_command_smoke_latest.json")
    parser.add_argument("--json-output", default="reports/pilot_experiment_command_coverage_latest.json")
    parser.add_argument("--csv-output", default="reports/pilot_experiment_command_coverage_latest.csv")
    parser.add_argument("--doc-output", default="docs/DEEPUplift_PILOT_EXPERIMENT_COMMAND_COVERAGE.md")
    args = parser.parse_args()

    payload = build_payload(read_json(ROOT / args.plan), read_json(ROOT / args.command_smoke))
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
                "plan_rows": (payload.get("summary") or {}).get("plan_rows"),
                "sampled_smoked_ok_rows": (payload.get("summary") or {}).get("sampled_smoked_ok_rows"),
                "backlog_rows": (payload.get("summary") or {}).get("backlog_rows"),
                "sampled_smoke_gate": (payload.get("summary") or {}).get("sampled_smoke_gate"),
            },
            ensure_ascii=False,
        )
    )
    if payload.get("status") != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
