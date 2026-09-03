from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import time
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


def parse_runner_output(output: str) -> dict[str, Any]:
    for line in reversed(output.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("status"):
            return payload
    return {}


def shlex_tokens(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return command.split()


def value_after_flag(command: str, flag: str) -> str | None:
    tokens = shlex_tokens(command)
    for index, token in enumerate(tokens):
        if token == flag and index + 1 < len(tokens):
            return tokens[index + 1]
    return None


def dataset_from_row(row: dict[str, Any]) -> str:
    command_value = value_after_flag(str(row.get("recommended_command") or ""), "--dataset-id")
    if command_value:
        return command_value
    entity = str(row.get("entity") or "")
    if " / " in entity:
        return entity.split(" / ", 1)[0].strip()
    return entity.strip()


def model_from_row(row: dict[str, Any]) -> str:
    command_value = value_after_flag(str(row.get("recommended_command") or ""), "--models")
    if command_value:
        return command_value
    entity = str(row.get("entity") or "")
    if " / " in entity:
        return entity.split(" / ", 1)[1].strip()
    return entity.strip() or "TLearnerGBM"


def sample_rows(rows: list[dict[str, Any]], limit: int, run_all: bool = False) -> list[dict[str, Any]]:
    if run_all:
        return list(rows)
    if limit <= 0:
        return []
    selected: list[dict[str, Any]] = []
    for experiment_type in [
        "shadow_scoring_and_holdout_design",
        "blocker_repair_rerun",
        "paper_benchmark_or_ablation_review",
        "evidence_refresh_and_monitoring",
    ]:
        for row in rows:
            if row.get("experiment_type") == experiment_type and row not in selected:
                selected.append(row)
                break
        if len(selected) >= limit:
            return selected[:limit]
    for row in rows:
        if row not in selected:
            selected.append(row)
        if len(selected) >= limit:
            break
    return selected[:limit]


def safe_command_for(row: dict[str, Any], timestamp: str) -> str:
    experiment_type = str(row.get("experiment_type") or "")
    if experiment_type in {"blocker_repair_rerun", "shadow_scoring_and_holdout_design"}:
        dataset_id = dataset_from_row(row)
        return (
            f"{PYTHON} scripts/smoke_industrial_scenario_compare.py "
            f"--rows 80 --dataset-id {dataset_id} "
            f"--artifacts-dir reports/pilot_experiment_command_smoke_runs/{timestamp} "
            f"--output-prefix pilot_experiment_command_smoke --no-update-latest"
        )
    if experiment_type == "paper_benchmark_or_ablation_review":
        model = model_from_row(row)
        return (
            f"{PYTHON} scripts/run_paper_benchmark_suite.py "
            f"--preset smoke --datasets acic_style_synthetic_known_cate_1k6 "
            f"--models {model} --seeds 20260520 --rows 90 --known-effect-bootstrap-samples 1 "
            f"--artifacts-dir reports/pilot_experiment_command_smoke_runs/{timestamp} "
            f"--output-prefix pilot_experiment_command_smoke --no-update-latest"
        )
    return (
        f"{PYTHON} scripts/generate_evidence_store.py --limit 40 "
        f"--json-output reports/pilot_experiment_command_smoke_evidence_store_{timestamp}.json "
        f"--csv-output reports/pilot_experiment_command_smoke_evidence_store_{timestamp}.csv "
        f"--diff-output reports/pilot_experiment_command_smoke_evidence_diff_{timestamp}.json "
        f"--markdown-output reports/pilot_experiment_command_smoke_evidence_store_{timestamp}.md"
    )


def run_command(row: dict[str, Any], timestamp: str, timeout_seconds: int) -> dict[str, Any]:
    safe_command = safe_command_for(row, timestamp)
    started = time.time()
    try:
        tokens = shlex.split(safe_command)
    except ValueError as exc:
        return {
            "experiment_id": row.get("experiment_id"),
            "status": "fail",
            "error": f"command parse failed: {exc}",
            "recommended_command": row.get("recommended_command"),
            "safe_smoke_command": safe_command,
        }
    result = subprocess.run(
        tokens,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    elapsed = round(time.time() - started, 3)
    combined = f"{result.stdout}\n{result.stderr}"
    runner_payload = parse_runner_output(combined)
    status = "ok" if result.returncode == 0 and runner_payload.get("status") == "ok" else "fail"
    return {
        "experiment_id": row.get("experiment_id"),
        "scorecard_id": row.get("scorecard_id"),
        "entity": row.get("entity"),
        "priority": row.get("priority"),
        "pilot_gate": row.get("pilot_gate"),
        "experiment_type": row.get("experiment_type"),
        "status": status,
        "returncode": result.returncode,
        "elapsed_seconds": elapsed,
        "recommended_command": row.get("recommended_command"),
        "safe_smoke_command": safe_command,
        "runner_json": runner_payload.get("json"),
        "runner_csv": runner_payload.get("csv"),
        "runner_manifest": runner_payload.get("manifest"),
        "runner_doc": runner_payload.get("doc") or runner_payload.get("markdown"),
        "runs": runner_payload.get("runs"),
        "ok_runs": runner_payload.get("ok_runs"),
        "leaderboard_rows": runner_payload.get("leaderboard_rows"),
        "failures": runner_payload.get("failures") or [],
        "pass_gate": row.get("pass_gate"),
        "fail_action": row.get("fail_action"),
        "claim_upgrade": row.get("claim_upgrade"),
        "error": "" if status == "ok" else combined[-2000:],
    }


def build_payload(plan: dict[str, Any], limit: int, run_all: bool, timeout_seconds: int) -> dict[str, Any]:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    source_rows = [row for row in plan.get("rows") or [] if isinstance(row, dict)]
    selected = sample_rows(source_rows, limit=limit, run_all=run_all)
    rows = [run_command(row, timestamp, timeout_seconds=timeout_seconds) for row in selected]
    ok_rows = sum(1 for row in rows if row.get("status") == "ok")
    type_counts: dict[str, int] = {}
    priority_counts: dict[str, int] = {}
    for row in rows:
        experiment_type = str(row.get("experiment_type") or "unknown")
        priority = str(row.get("priority") or "unknown")
        type_counts[experiment_type] = type_counts.get(experiment_type, 0) + 1
        priority_counts[priority] = priority_counts.get(priority, 0) + 1
    return {
        "schema_version": 1,
        "status": "ok" if rows and ok_rows == len(rows) else "fail",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "summary": {
            "source_plan_rows": len(source_rows),
            "smoke_rows": len(rows),
            "ok_rows": ok_rows,
            "failed_rows": len(rows) - ok_rows,
            "run_all": run_all,
            "limit": limit,
            "experiment_type_counts": type_counts,
            "priority_counts": priority_counts,
            "coverage_note": "sampled_smoke_execution_not_full_coverage",
        },
        "rows": rows,
        "interview_tracks": {
            "30s": "The pilot plan is not only generated; a sampled set of next-experiment commands is executed with safe tiny-row overrides.",
            "5min": "Show the source plan row, the safe smoke command, the runner artifact, and the pass/fail boundary. This proves command executability without overwriting main latest benchmark evidence.",
            "15min": "Explain the distinction between runner-supported command, smoke-executed command, and full pilot evidence. Smoke is an engineering gate, not an online launch claim.",
            "30min": "Walk through blocker repair, shadow scoring, paper benchmark smoke and evidence-store refresh as the offline-to-shadow promotion loop.",
        },
        "source_artifacts": {
            "pilot_experiment_plan": "reports/pilot_experiment_plan_latest.json",
            "paper_runner": "scripts/run_paper_benchmark_suite.py",
            "industrial_runner": "scripts/smoke_industrial_scenario_compare.py",
        },
    }


CSV_FIELDS = [
    "experiment_id",
    "scorecard_id",
    "entity",
    "priority",
    "pilot_gate",
    "experiment_type",
    "status",
    "returncode",
    "elapsed_seconds",
    "runs",
    "ok_runs",
    "leaderboard_rows",
    "runner_json",
    "runner_csv",
    "runner_manifest",
    "runner_doc",
    "recommended_command",
    "safe_smoke_command",
    "pass_gate",
    "fail_action",
    "claim_upgrade",
    "failures",
    "error",
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
        "# DeepUplift Pilot Experiment Command Smoke",
        "",
        f"Generated at: `{payload.get('generated_at')}`",
        "",
        "This report executes a sampled set of commands from the Pilot Experiment Plan with safe tiny-row overrides and isolated output prefixes. It proves command executability without replacing the main benchmark/latest evidence.",
        "",
        "## Summary",
        "",
        f"- Source plan rows: `{summary.get('source_plan_rows')}`",
        f"- Smoke rows: `{summary.get('smoke_rows')}`",
        f"- OK rows: `{summary.get('ok_rows')}`",
        f"- Failed rows: `{summary.get('failed_rows')}`",
        f"- Run all: `{summary.get('run_all')}`",
        f"- Experiment types: `{json.dumps(summary.get('experiment_type_counts') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Interview Tracks",
        "",
        f"- 30 seconds: {tracks.get('30s', '')}",
        f"- 5 minutes: {tracks.get('5min', '')}",
        f"- 15 minutes: {tracks.get('15min', '')}",
        f"- 30 minutes: {tracks.get('30min', '')}",
        "",
        "## Smoke Rows",
        "",
        "| Priority | Type | Entity | Status | Runs | Runner artifact |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    for row in rows:
        runner = row.get("runner_manifest") or row.get("runner_json") or row.get("runner_csv") or ""
        lines.append(
            f"| {row.get('priority')} | {row.get('experiment_type')} | {row.get('entity')} | "
            f"{row.get('status')} | {row.get('runs')} | `{runner}` |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Safely execute sampled Pilot Experiment Plan commands.")
    parser.add_argument("--plan", default="reports/pilot_experiment_plan_latest.json")
    parser.add_argument("--json-output", default="reports/pilot_experiment_command_smoke_latest.json")
    parser.add_argument("--csv-output", default="reports/pilot_experiment_command_smoke_latest.csv")
    parser.add_argument("--doc-output", default="docs/DEEPUplift_PILOT_EXPERIMENT_COMMAND_SMOKE.md")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=180)
    args = parser.parse_args()

    payload = build_payload(read_json(ROOT / args.plan), limit=args.limit, run_all=args.all, timeout_seconds=args.timeout_seconds)
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
                "smoke_rows": (payload.get("summary") or {}).get("smoke_rows"),
                "ok_rows": (payload.get("summary") or {}).get("ok_rows"),
            },
            ensure_ascii=False,
        )
    )
    if payload.get("status") != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
