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
SAFE_OVERRIDE_FLAGS = (
    "--preset tuned-smoke "
    "--datasets acic_style_synthetic_known_cate_1k6 "
    "--seeds 20260520 "
    "--rows 90 "
    "--known-effect-bootstrap-samples 1 "
    "--artifacts-dir reports/deep_model_ablation_recommended_command_smoke_runs "
    "--output-prefix deep_model_ablation_recommended_command_smoke "
    "--no-update-latest"
)


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


def command_flags(command: str) -> set[str]:
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    return {token for token in tokens if token.startswith("--")}


def parse_csv_arg(values: list[str] | None) -> set[str]:
    selected: set[str] = set()
    for value in values or []:
        for item in str(value).split(","):
            item = item.strip()
            if item:
                selected.add(item)
    return selected


def filter_rows(
    rows: list[dict[str, Any]],
    models: set[str],
    variant_ids: set[str],
    priorities: set[str],
) -> list[dict[str, Any]]:
    filtered = rows
    if models:
        filtered = [row for row in filtered if str(row.get("model") or "") in models]
    if variant_ids:
        filtered = [row for row in filtered if str(row.get("variant_id") or "") in variant_ids]
    if priorities:
        filtered = [row for row in filtered if str(row.get("priority") or "") in priorities]
    return filtered


def sample_rows(rows: list[dict[str, Any]], limit: int, run_all: bool = False) -> list[dict[str, Any]]:
    if run_all:
        return list(rows)
    if limit <= 0:
        return []
    selected: list[dict[str, Any]] = []
    for priority in ("P0", "P1", "P2"):
        for row in rows:
            if row.get("priority") == priority and row not in selected:
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


def safe_recommended_command(row: dict[str, Any]) -> str:
    command = str(row.get("recommended_command") or "")
    return f"{command} {SAFE_OVERRIDE_FLAGS}".strip()


def run_command(row: dict[str, Any], timeout_seconds: int) -> dict[str, Any]:
    original_command = str(row.get("recommended_command") or "")
    command = safe_recommended_command(row)
    flags = command_flags(original_command)
    started = time.time()
    try:
        tokens = shlex.split(command)
    except ValueError as exc:
        return {
            "variant_id": row.get("variant_id"),
            "model": row.get("model"),
            "status": "fail",
            "error": f"command parse failed: {exc}",
            "recommended_command": original_command,
            "safe_smoke_command": command,
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
        "uses_models_alias": "--models" in flags,
        "uses_include_variants_alias": "--include-variants" in flags,
        "uses_emit_battle_cards_alias": "--emit-battle-cards" in flags,
        "recommended_command": original_command,
        "safe_smoke_command": command,
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
    "uses_models_alias",
    "uses_include_variants_alias",
    "uses_emit_battle_cards_alias",
    "runs",
    "ok_runs",
    "leaderboard_rows",
    "battle_cards_rows",
    "runner_json",
    "runner_manifest",
    "runner_leaderboard",
    "runner_doc",
    "recommended_command",
    "safe_smoke_command",
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
        "# DeepUplift Deep Model Ablation Recommended Command Smoke",
        "",
        f"Generated at: `{payload.get('generated_at')}`",
        "",
        "This smoke proof executes a small sample of runner-native recommended commands from the command-contract artifact. It appends safe overrides for a tiny dataset slice, isolated artifact directory, `deep_model_ablation_recommended_command_smoke` output prefix, and `--no-update-latest`, so the main tuned benchmark evidence is not overwritten.",
        "",
        "## Summary",
        "",
        f"- Rows executed: `{summary.get('rows')}`",
        f"- OK rows: `{summary.get('ok_rows')}`",
        f"- Failed rows: `{summary.get('failed_rows')}`",
        f"- Source contract rows: `{summary.get('source_contract_rows')}`",
        f"- Filtered contract rows: `{summary.get('filtered_contract_rows')}`",
        f"- Run all filtered rows: `{summary.get('run_all')}`",
        f"- Rows using `--models`: `{summary.get('models_alias_rows')}`",
        f"- Rows using `--include-variants`: `{summary.get('include_variants_alias_rows')}`",
        f"- Rows using `--emit-battle-cards`: `{summary.get('emit_battle_cards_alias_rows')}`",
        "",
        "## Smoke Rows",
        "",
        "| Priority | Variant | Model | Status | Runs | OK runs | Native aliases | Manifest |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        aliases = ", ".join(
            alias
            for alias, used in [
                ("models", row.get("uses_models_alias")),
                ("include-variants", row.get("uses_include_variants_alias")),
                ("emit-battle-cards", row.get("uses_emit_battle_cards_alias")),
            ]
            if used
        ) or "none"
        lines.append(
            f"| {row.get('priority')} | {row.get('variant_id')} | {row.get('model')} | {row.get('status')} | "
            f"{row.get('runs')} | {row.get('ok_runs')} | {aliases} | `{row.get('runner_manifest')}` |"
        )
    lines.extend(
        [
            "",
            "## Evidence Contract",
            "",
            "- Source contract: `reports/deep_model_ablation_command_contract_latest.json`",
            "- Smoke JSON: `reports/deep_model_ablation_recommended_command_smoke_latest.json`",
            "- Smoke CSV: `reports/deep_model_ablation_recommended_command_smoke_latest.csv`",
            "- Main benchmark latest is not updated by this smoke.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute a small sample of runner-native ablation recommended commands.")
    parser.add_argument("--contract", default="reports/deep_model_ablation_command_contract_latest.json")
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--all", action="store_true", help="Run all filtered runner-native recommended command rows.")
    parser.add_argument("--models", nargs="*", default=None, help="Optional model filter; accepts space-separated or comma-separated values.")
    parser.add_argument("--variant-ids", nargs="*", default=None, help="Optional variant_id filter; accepts space-separated or comma-separated values.")
    parser.add_argument("--priorities", nargs="*", default=None, help="Optional priority filter such as P0 P1; accepts space-separated or comma-separated values.")
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--json-output", default="reports/deep_model_ablation_recommended_command_smoke_latest.json")
    parser.add_argument("--csv-output", default="reports/deep_model_ablation_recommended_command_smoke_latest.csv")
    parser.add_argument("--doc-output", default="docs/DEEPUplift_DEEP_MODEL_ABLATION_RECOMMENDED_COMMAND_SMOKE.md")
    args = parser.parse_args()

    contract = read_json(ROOT / args.contract)
    rows = [
        row
        for row in contract.get("rows") or []
        if isinstance(row, dict) and row.get("contract_status") == "runner_supported" and row.get("recommended_command")
    ]
    models = parse_csv_arg(args.models)
    variant_ids = parse_csv_arg(args.variant_ids)
    priorities = parse_csv_arg(args.priorities)
    filtered_rows = filter_rows(rows, models=models, variant_ids=variant_ids, priorities=priorities)
    sample = sample_rows(filtered_rows, max(int(args.limit), 0), run_all=bool(args.all))
    smoke_rows = [run_command(row, args.timeout_seconds) for row in sample]
    ok_rows = sum(1 for row in smoke_rows if row.get("status") == "ok")
    payload = {
        "schema_version": 1,
        "status": "ok" if smoke_rows and ok_rows == len(smoke_rows) else "fail",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "safe_overrides": SAFE_OVERRIDE_FLAGS,
        "summary": {
            "rows": len(smoke_rows),
            "ok_rows": ok_rows,
            "failed_rows": len(smoke_rows) - ok_rows,
            "sample_limit": args.limit,
            "run_all": bool(args.all),
            "source_contract_rows": len(rows),
            "filtered_contract_rows": len(filtered_rows),
            "requested_models": sorted(models),
            "requested_variant_ids": sorted(variant_ids),
            "requested_priorities": sorted(priorities),
            "models_alias_rows": sum(1 for row in smoke_rows if row.get("uses_models_alias")),
            "include_variants_alias_rows": sum(1 for row in smoke_rows if row.get("uses_include_variants_alias")),
            "emit_battle_cards_alias_rows": sum(1 for row in smoke_rows if row.get("uses_emit_battle_cards_alias")),
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
                "models_alias_rows": payload["summary"]["models_alias_rows"],
                "include_variants_alias_rows": payload["summary"]["include_variants_alias_rows"],
                "emit_battle_cards_alias_rows": payload["summary"]["emit_battle_cards_alias_rows"],
            },
            ensure_ascii=False,
        )
    )
    if payload.get("status") != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
