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
        if isinstance(payload, dict) and payload.get("json"):
            return payload
    return {}


def run_command(row: dict[str, Any], timeout_seconds: int) -> dict[str, Any]:
    command = str(row.get("runnable_fallback_command") or "")
    started = time.time()
    try:
        tokens = shlex.split(command)
    except ValueError as exc:
        return {
            "variant_id": row.get("variant_id"),
            "model": row.get("model"),
            "status": "fail",
            "error": f"command parse failed: {exc}",
            "command": command,
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
        "variant_id": row.get("variant_id"),
        "model": row.get("model"),
        "priority": row.get("priority"),
        "contract_status": row.get("contract_status"),
        "status": status,
        "returncode": result.returncode,
        "elapsed_seconds": elapsed,
        "command": command,
        "runner_json": runner_payload.get("json"),
        "runner_manifest": runner_payload.get("manifest"),
        "runner_leaderboard": runner_payload.get("leaderboard"),
        "runner_doc": runner_payload.get("doc"),
        "runs": runner_payload.get("runs"),
        "ok_runs": runner_payload.get("ok_runs"),
        "leaderboard_rows": runner_payload.get("leaderboard_rows"),
        "battle_cards_rows": runner_payload.get("battle_cards_rows"),
        "failures": runner_payload.get("failures") or [],
        "error": "" if status == "ok" else combined[-2000:],
    }


CSV_FIELDS = [
    "variant_id",
    "model",
    "priority",
    "contract_status",
    "status",
    "returncode",
    "elapsed_seconds",
    "runs",
    "ok_runs",
    "leaderboard_rows",
    "battle_cards_rows",
    "runner_json",
    "runner_manifest",
    "runner_leaderboard",
    "runner_doc",
    "command",
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
    lines = [
        "# DeepUplift Deep Model Ablation Command Contract Smoke",
        "",
        f"Generated at: `{payload.get('generated_at')}`",
        "",
        "This smoke proof executes a small sample of runnable fallback commands from the command-contract artifact. It intentionally uses `--output-prefix deep_model_ablation_command_contract_smoke` and `--no-update-latest` so the main tuned benchmark evidence is not overwritten.",
        "",
        "## Summary",
        "",
        f"- Rows executed: `{summary.get('rows')}`",
        f"- OK rows: `{summary.get('ok_rows')}`",
        f"- Failed rows: `{summary.get('failed_rows')}`",
        f"- Sample limit: `{summary.get('sample_limit')}`",
        "",
        "## Smoke Rows",
        "",
        "| Variant | Model | Status | Runs | OK runs | Manifest |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('variant_id')} | {row.get('model')} | {row.get('status')} | "
            f"{row.get('runs')} | {row.get('ok_runs')} | `{row.get('runner_manifest')}` |"
        )
    lines.extend(
        [
            "",
            "## Evidence Contract",
            "",
            "- Source contract: `reports/deep_model_ablation_command_contract_latest.json`",
            "- Smoke JSON: `reports/deep_model_ablation_command_contract_smoke_latest.json`",
            "- Smoke CSV: `reports/deep_model_ablation_command_contract_smoke_latest.csv`",
            "- Main benchmark latest is not updated by this smoke.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute a small sample of ablation command-contract fallback commands.")
    parser.add_argument("--contract", default="reports/deep_model_ablation_command_contract_latest.json")
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=int, default=240)
    parser.add_argument("--json-output", default="reports/deep_model_ablation_command_contract_smoke_latest.json")
    parser.add_argument("--csv-output", default="reports/deep_model_ablation_command_contract_smoke_latest.csv")
    parser.add_argument("--doc-output", default="docs/DEEPUplift_DEEP_MODEL_ABLATION_COMMAND_CONTRACT_SMOKE.md")
    args = parser.parse_args()

    contract = read_json(ROOT / args.contract)
    rows = [row for row in contract.get("rows") or [] if isinstance(row, dict) and row.get("fallback_ready")]
    sample = rows[: max(int(args.limit), 0)]
    smoke_rows = [run_command(row, args.timeout_seconds) for row in sample]
    ok_rows = sum(1 for row in smoke_rows if row.get("status") == "ok")
    payload = {
        "schema_version": 1,
        "status": "ok" if smoke_rows and ok_rows == len(smoke_rows) else "fail",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "summary": {
            "rows": len(smoke_rows),
            "ok_rows": ok_rows,
            "failed_rows": len(smoke_rows) - ok_rows,
            "sample_limit": args.limit,
            "source_contract_rows": len(rows),
        },
        "rows": smoke_rows,
        "source_artifacts": {
            "deep_model_ablation_command_contract": args.contract,
        },
    }
    write_json(ROOT / args.json_output, payload)
    write_csv(ROOT / args.csv_output, smoke_rows)
    write_markdown(ROOT / args.doc_output, payload)
    print(
        json.dumps(
            {
                "status": payload.get("status"),
                "json": args.json_output,
                "csv": args.csv_output,
                "markdown": args.doc_output,
                "rows": len(smoke_rows),
                "ok_rows": ok_rows,
            },
            ensure_ascii=False,
        )
    )
    if payload.get("status") != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
