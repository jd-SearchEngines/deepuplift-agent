from __future__ import annotations

import argparse
import csv
import json
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNNER_SCRIPT = "scripts/run_deep_model_tuned_benchmark.py"


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


def runner_help_flags(runner: Path) -> tuple[set[str], str]:
    if not runner.exists():
        return set(), f"missing runner: {runner}"
    result = subprocess.run(
        [sys.executable, str(runner), "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    output = f"{result.stdout}\n{result.stderr}"
    flags = set(re.findall(r"--[a-zA-Z0-9][a-zA-Z0-9-]*", output))
    if result.returncode != 0:
        return flags, f"runner --help returned {result.returncode}"
    return flags, ""


def command_flags(command: str) -> list[str]:
    try:
        tokens = shlex.split(command or "")
    except ValueError:
        tokens = str(command or "").split()
    return [token for token in tokens if token.startswith("--")]


def command_uses_runner(command: str) -> bool:
    try:
        tokens = shlex.split(command or "")
    except ValueError:
        tokens = str(command or "").split()
    return any(token.endswith(RUNNER_SCRIPT) for token in tokens)


def command_row(plan_row: dict[str, Any], supported_flags: set[str]) -> dict[str, Any]:
    recommended = str(plan_row.get("recommended_command") or "")
    fallback = str(plan_row.get("runnable_fallback_command") or "")
    recommended_flags = command_flags(recommended)
    fallback_flags = command_flags(fallback)
    recommended_unsupported = sorted({flag for flag in recommended_flags if flag not in supported_flags})
    fallback_unsupported = sorted({flag for flag in fallback_flags if flag not in supported_flags})
    fallback_ready = bool(fallback and command_uses_runner(fallback) and not fallback_unsupported)
    if fallback_ready and recommended_unsupported:
        status = "fallback_ready_planner_needs_adapter"
    elif fallback_ready:
        status = "runner_supported"
    else:
        status = "fallback_needs_review"
    if recommended_unsupported:
        interview_line = (
            f"`{plan_row.get('variant_id')}` is honest-by-contract: the planning command still requires runner adapter work "
            f"({', '.join(recommended_unsupported)}), while the fallback command is "
            f"{'runnable on the current CLI' if fallback_ready else 'not yet runnable and must be repaired'}."
        )
    else:
        interview_line = (
            f"`{plan_row.get('variant_id')}` is runner-native-by-contract: the recommended command and the isolated fallback "
            "command are both supported by the current tuned benchmark CLI."
        )
    return {
        "model": plan_row.get("model"),
        "variant_id": plan_row.get("variant_id"),
        "priority": plan_row.get("priority"),
        "experiment_type": plan_row.get("experiment_type"),
        "claim_tier": plan_row.get("claim_tier"),
        "recommended_command": recommended,
        "runnable_fallback_command": fallback,
        "recommended_flags": recommended_flags,
        "recommended_unsupported_flags": recommended_unsupported,
        "fallback_flags": fallback_flags,
        "fallback_unsupported_flags": fallback_unsupported,
        "fallback_ready": fallback_ready,
        "contract_status": status,
        "contract_interview_line": interview_line,
        "source_refs": [
            "reports/deep_model_ablation_next_experiment_plan_latest.json",
            "scripts/run_deep_model_tuned_benchmark.py",
        ],
    }


def build_payload(plan_payload: dict[str, Any], supported_flags: set[str], help_error: str) -> dict[str, Any]:
    rows = [
        command_row(row, supported_flags)
        for row in plan_payload.get("rows") or []
        if isinstance(row, dict) and row.get("variant_id")
    ]
    status_counts: dict[str, int] = {}
    unsupported_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("contract_status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        for flag in row.get("recommended_unsupported_flags") or []:
            unsupported_counts[str(flag)] = unsupported_counts.get(str(flag), 0) + 1
        for flag in row.get("fallback_unsupported_flags") or []:
            unsupported_counts[str(flag)] = unsupported_counts.get(str(flag), 0) + 1
    fallback_ready_rows = sum(1 for row in rows if row.get("fallback_ready"))
    planner_adapter_rows = sum(1 for row in rows if row.get("recommended_unsupported_flags"))
    return {
        "schema_version": 1,
        "status": "ok" if rows and fallback_ready_rows == len(rows) and not help_error else "review_required",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "runner": RUNNER_SCRIPT,
        "runner_supported_flags": sorted(supported_flags),
        "runner_help_error": help_error,
        "summary": {
            "rows": len(rows),
            "variants": len({row.get("variant_id") for row in rows}),
            "fallback_ready_rows": fallback_ready_rows,
            "planner_adapter_rows": planner_adapter_rows,
            "status_counts": status_counts,
            "unsupported_flag_counts": unsupported_counts,
        },
        "rows": rows,
        "interview_tracks": {
            "30s": "I added a command contract so ablation next-experiment commands are audited against the current tuned benchmark runner before they become evidence.",
            "5min": "Each ablation variant now carries a recommended command and an isolated fallback command; both are checked against `run_deep_model_tuned_benchmark.py --help`.",
            "15min": "The contract prevents evidence inflation: a planned experiment is not counted as runnable unless all command flags are supported, or the fallback path is explicitly runnable.",
            "30min": "Walk one row from ablation promotion to next experiment, then to command contract and command smoke: unsupported flags become backlog, while runner-native rows can execute immediately.",
        },
        "source_artifacts": {
            "deep_model_ablation_next_experiment_plan": "reports/deep_model_ablation_next_experiment_plan_latest.json",
            "deep_model_tuned_benchmark_runner": RUNNER_SCRIPT,
        },
    }


CSV_FIELDS = [
    "model",
    "variant_id",
    "priority",
    "experiment_type",
    "claim_tier",
    "contract_status",
    "fallback_ready",
    "recommended_unsupported_flags",
    "fallback_unsupported_flags",
    "recommended_command",
    "runnable_fallback_command",
    "contract_interview_line",
    "source_refs",
]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: flatten(row.get(field)) for field in CSV_FIELDS})


def markdown_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Priority | Model | Variant | Status | Fallback ready | Unsupported planner flags |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        unsupported = ", ".join(row.get("recommended_unsupported_flags") or []) or "none"
        lines.append(
            f"| {row.get('priority')} | {row.get('model')} | {row.get('variant_id')} | "
            f"{row.get('contract_status')} | {row.get('fallback_ready')} | {unsupported} |"
        )
    return lines


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    summary = payload.get("summary") or {}
    tracks = payload.get("interview_tracks") or {}
    rows = payload.get("rows") or []
    lines = [
        "# DeepUplift Deep Model Ablation Command Contract",
        "",
        f"Generated at: `{payload.get('generated_at')}`",
        "",
        "This report audits whether ablation next-experiment commands match the current tuned benchmark runner. It keeps the project honest: runner-native commands are marked runnable, unsupported planning flags become adapter backlog, and every row also carries an isolated fallback command.",
        "",
        "## Summary",
        "",
        f"- Rows: `{summary.get('rows')}`",
        f"- Fallback ready rows: `{summary.get('fallback_ready_rows')}`",
        f"- Planner adapter rows: `{summary.get('planner_adapter_rows')}`",
        f"- Status counts: `{json.dumps(summary.get('status_counts') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- Unsupported flag counts: `{json.dumps(summary.get('unsupported_flag_counts') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- Runner flags: `{', '.join(payload.get('runner_supported_flags') or [])}`",
        "",
        "## Interview Tracks",
        "",
        f"- 30 seconds: {tracks.get('30s', '')}",
        f"- 5 minutes: {tracks.get('5min', '')}",
        f"- 15 minutes: {tracks.get('15min', '')}",
        f"- 30 minutes: {tracks.get('30min', '')}",
        "",
        "## Contract Table",
        "",
    ]
    lines.extend(markdown_table(rows))
    lines.extend(["", "## Runnable Fallback Commands", ""])
    for row in rows:
        lines.extend(
            [
                f"### {row.get('variant_id')}",
                "",
                f"- Planning command: `{row.get('recommended_command')}`",
                f"- Runnable fallback: `{row.get('runnable_fallback_command')}`",
                f"- Contract line: {row.get('contract_interview_line')}",
                "",
            ]
        )
    lines.extend(["## Evidence Links", ""])
    for name, artifact in (payload.get("source_artifacts") or {}).items():
        lines.append(f"- {name}: `{artifact}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit ablation next-experiment commands against the tuned benchmark runner CLI.")
    parser.add_argument("--plan", default="reports/deep_model_ablation_next_experiment_plan_latest.json")
    parser.add_argument("--runner", default=RUNNER_SCRIPT)
    parser.add_argument("--json-output", default="reports/deep_model_ablation_command_contract_latest.json")
    parser.add_argument("--csv-output", default="reports/deep_model_ablation_command_contract_latest.csv")
    parser.add_argument("--doc-output", default="docs/DEEPUplift_DEEP_MODEL_ABLATION_COMMAND_CONTRACT.md")
    args = parser.parse_args()

    plan = read_json(ROOT / args.plan)
    supported_flags, help_error = runner_help_flags(ROOT / args.runner)
    payload = build_payload(plan, supported_flags, help_error)
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
                "fallback_ready_rows": (payload.get("summary") or {}).get("fallback_ready_rows"),
                "planner_adapter_rows": (payload.get("summary") or {}).get("planner_adapter_rows"),
            },
            ensure_ascii=False,
        )
    )
    if payload.get("status") != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
