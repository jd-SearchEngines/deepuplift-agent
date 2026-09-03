from __future__ import annotations

import argparse
import csv
import json
import shlex
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


def command_flags(command: str) -> list[str]:
    try:
        tokens = shlex.split(command or "")
    except ValueError:
        tokens = str(command or "").split()
    return [token for token in tokens if token.startswith("--")]


def ok_rows_by_variant(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in payload.get("rows") or []:
        if isinstance(row, dict) and row.get("status") == "ok" and row.get("variant_id"):
            rows[str(row.get("variant_id"))] = row
    return rows


def coverage_status(fallback_smoked: bool, recommended_smoked: bool, contract_status: str, fallback_ready: bool) -> str:
    if fallback_smoked and recommended_smoked:
        return "both_smoked"
    if recommended_smoked:
        return "recommended_smoked"
    if fallback_smoked:
        return "fallback_smoked"
    if contract_status == "runner_supported" and fallback_ready:
        return "contract_ready_not_smoked"
    return "contract_needs_review"


def next_action_for(row: dict[str, Any], status: str, source_contract_rows: int) -> str:
    variant_id = row.get("variant_id")
    if status == "both_smoked":
        return "Use as smoke-covered command evidence; next upgrade is multi-seed benchmark expansion."
    if status == "recommended_smoked":
        return "Optional: run fallback-command smoke for symmetry, but runner-native recommended path already executed."
    if status == "fallback_smoked":
        return "Run runner-native recommended command smoke to prove the --models / --include-variants path."
    if status == "contract_ready_not_smoked":
        return (
            f"Increase recommended command smoke coverage, e.g. "
            f"`python scripts/smoke_deep_model_ablation_recommended_commands.py --all` "
            f"or `python scripts/smoke_deep_model_ablation_recommended_commands.py --limit {source_contract_rows}` "
            f"to include `{variant_id}`."
        )
    return "Repair command contract or runner CLI before executing smoke."


def build_rows(
    contract: dict[str, Any],
    fallback_smoke: dict[str, Any],
    recommended_smoke: dict[str, Any],
) -> list[dict[str, Any]]:
    fallback_by_variant = ok_rows_by_variant(fallback_smoke)
    recommended_by_variant = ok_rows_by_variant(recommended_smoke)
    source_contract_rows = int((contract.get("summary") or {}).get("rows") or len(contract.get("rows") or []))
    rows: list[dict[str, Any]] = []
    for contract_row in contract.get("rows") or []:
        if not isinstance(contract_row, dict) or not contract_row.get("variant_id"):
            continue
        variant_id = str(contract_row.get("variant_id"))
        fallback_row = fallback_by_variant.get(variant_id) or {}
        recommended_row = recommended_by_variant.get(variant_id) or {}
        fallback_smoked = bool(fallback_row)
        recommended_smoked = bool(recommended_row)
        status = coverage_status(
            fallback_smoked=fallback_smoked,
            recommended_smoked=recommended_smoked,
            contract_status=str(contract_row.get("contract_status") or ""),
            fallback_ready=bool(contract_row.get("fallback_ready")),
        )
        recommended_flags = command_flags(str(contract_row.get("recommended_command") or ""))
        smoke_aliases = [
            alias
            for alias, used in [
                ("--models", recommended_row.get("uses_models_alias")),
                ("--include-variants", recommended_row.get("uses_include_variants_alias")),
                ("--emit-battle-cards", recommended_row.get("uses_emit_battle_cards_alias")),
            ]
            if used
        ]
        rows.append(
            {
                "model": contract_row.get("model"),
                "variant_id": variant_id,
                "priority": contract_row.get("priority"),
                "experiment_type": contract_row.get("experiment_type"),
                "claim_tier": contract_row.get("claim_tier"),
                "contract_status": contract_row.get("contract_status"),
                "fallback_ready": bool(contract_row.get("fallback_ready")),
                "fallback_smoked": fallback_smoked,
                "recommended_smoked": recommended_smoked,
                "any_smoked": fallback_smoked or recommended_smoked,
                "coverage_status": status,
                "recommended_runner_aliases": [flag for flag in recommended_flags if flag in {"--models", "--include-variants", "--emit-battle-cards"}],
                "recommended_smoke_aliases": smoke_aliases,
                "fallback_runner_manifest": fallback_row.get("runner_manifest"),
                "recommended_runner_manifest": recommended_row.get("runner_manifest"),
                "recommended_runs": recommended_row.get("runs"),
                "recommended_ok_runs": recommended_row.get("ok_runs"),
                "fallback_runs": fallback_row.get("runs"),
                "fallback_ok_runs": fallback_row.get("ok_runs"),
                "next_action": next_action_for(contract_row, status, source_contract_rows),
                "source_refs": [
                    "reports/deep_model_ablation_command_contract_latest.json",
                    "reports/deep_model_ablation_command_contract_smoke_latest.json",
                    "reports/deep_model_ablation_recommended_command_smoke_latest.json",
                ],
            }
        )
    return rows


def build_payload(contract: dict[str, Any], fallback_smoke: dict[str, Any], recommended_smoke: dict[str, Any]) -> dict[str, Any]:
    rows = build_rows(contract, fallback_smoke, recommended_smoke)
    status_counts: dict[str, int] = {}
    priority_counts: dict[str, dict[str, int]] = {}
    for row in rows:
        status = str(row.get("coverage_status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        priority = str(row.get("priority") or "unknown")
        priority_counts.setdefault(priority, {"rows": 0, "any_smoked": 0, "recommended_smoked": 0})
        priority_counts[priority]["rows"] += 1
        if row.get("any_smoked"):
            priority_counts[priority]["any_smoked"] += 1
        if row.get("recommended_smoked"):
            priority_counts[priority]["recommended_smoked"] += 1
    contract_rows = len(rows)
    any_smoked_rows = sum(1 for row in rows if row.get("any_smoked"))
    recommended_smoked_rows = sum(1 for row in rows if row.get("recommended_smoked"))
    fallback_smoked_rows = sum(1 for row in rows if row.get("fallback_smoked"))
    both_smoked_rows = sum(1 for row in rows if row.get("fallback_smoked") and row.get("recommended_smoked"))
    unsmoked_rows = contract_rows - any_smoked_rows
    return {
        "schema_version": 1,
        "status": "ok" if rows else "fail",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "summary": {
            "contract_rows": contract_rows,
            "runner_supported_rows": sum(1 for row in rows if row.get("contract_status") == "runner_supported"),
            "fallback_ready_rows": sum(1 for row in rows if row.get("fallback_ready")),
            "fallback_smoked_rows": fallback_smoked_rows,
            "recommended_smoked_rows": recommended_smoked_rows,
            "both_smoked_rows": both_smoked_rows,
            "any_smoked_rows": any_smoked_rows,
            "unsmoked_rows": unsmoked_rows,
            "coverage_pct": round(any_smoked_rows / max(contract_rows, 1), 4),
            "recommended_coverage_pct": round(recommended_smoked_rows / max(contract_rows, 1), 4),
            "fallback_coverage_pct": round(fallback_smoked_rows / max(contract_rows, 1), 4),
            "coverage_status_counts": status_counts,
            "priority_counts": priority_counts,
            "coverage_gate": "ok" if unsmoked_rows == 0 else "review_required",
        },
        "rows": rows,
        "interview_tracks": {
            "30s": "I added smoke coverage governance: command contracts can be runnable while only a sampled subset is smoke-executed, and the gap is visible.",
            "5min": "The coverage report joins command contract rows with fallback smoke and runner-native recommended-command smoke, so reviewers can see which ablation experiments are proven by execution.",
            "15min": "This prevents overclaiming: `runner_supported` means the CLI accepts the command, while `recommended_smoked` / `fallback_smoked` means a safe isolated execution produced manifests.",
            "30min": "Walk EFIN_wide as both-smoked, DESCN_no_constraint as runner-native-smoked, then the remaining contract-ready variants as the next coverage expansion plan before larger benchmark runs.",
        },
        "source_artifacts": {
            "deep_model_ablation_command_contract": "reports/deep_model_ablation_command_contract_latest.json",
            "deep_model_ablation_command_contract_smoke": "reports/deep_model_ablation_command_contract_smoke_latest.json",
            "deep_model_ablation_recommended_command_smoke": "reports/deep_model_ablation_recommended_command_smoke_latest.json",
        },
    }


CSV_FIELDS = [
    "priority",
    "model",
    "variant_id",
    "experiment_type",
    "claim_tier",
    "contract_status",
    "fallback_ready",
    "fallback_smoked",
    "recommended_smoked",
    "any_smoked",
    "coverage_status",
    "recommended_runner_aliases",
    "recommended_smoke_aliases",
    "fallback_runner_manifest",
    "recommended_runner_manifest",
    "fallback_runs",
    "fallback_ok_runs",
    "recommended_runs",
    "recommended_ok_runs",
    "next_action",
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
        "# DeepUplift Deep Model Ablation Command Smoke Coverage",
        "",
        f"Generated at: `{payload.get('generated_at')}`",
        "",
        "This report joins the ablation command contract with the fallback-command smoke and runner-native recommended-command smoke. It makes the evidence boundary explicit: a command can be runner-supported while still awaiting smoke execution.",
        "",
        "## Summary",
        "",
        f"- Contract rows: `{summary.get('contract_rows')}`",
        f"- Any-smoked rows: `{summary.get('any_smoked_rows')}`",
        f"- Recommended-smoked rows: `{summary.get('recommended_smoked_rows')}`",
        f"- Fallback-smoked rows: `{summary.get('fallback_smoked_rows')}`",
        f"- Both-smoked rows: `{summary.get('both_smoked_rows')}`",
        f"- Unsmoked rows: `{summary.get('unsmoked_rows')}`",
        f"- Coverage gate: `{summary.get('coverage_gate')}`",
        f"- Coverage status counts: `{json.dumps(summary.get('coverage_status_counts') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Interview Tracks",
        "",
        f"- 30 seconds: {tracks.get('30s', '')}",
        f"- 5 minutes: {tracks.get('5min', '')}",
        f"- 15 minutes: {tracks.get('15min', '')}",
        f"- 30 minutes: {tracks.get('30min', '')}",
        "",
        "## Coverage Table",
        "",
        "| Priority | Model | Variant | Contract | Fallback smoke | Recommended smoke | Coverage | Next action |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('priority')} | {row.get('model')} | {row.get('variant_id')} | "
            f"{row.get('contract_status')} | {row.get('fallback_smoked')} | {row.get('recommended_smoked')} | "
            f"{row.get('coverage_status')} | {row.get('next_action')} |"
        )
    lines.extend(["", "## Evidence Links", ""])
    for name, artifact in (payload.get("source_artifacts") or {}).items():
        lines.append(f"- {name}: `{artifact}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ablation command smoke coverage from contract and smoke artifacts.")
    parser.add_argument("--contract", default="reports/deep_model_ablation_command_contract_latest.json")
    parser.add_argument("--fallback-smoke", default="reports/deep_model_ablation_command_contract_smoke_latest.json")
    parser.add_argument("--recommended-smoke", default="reports/deep_model_ablation_recommended_command_smoke_latest.json")
    parser.add_argument("--json-output", default="reports/deep_model_ablation_command_smoke_coverage_latest.json")
    parser.add_argument("--csv-output", default="reports/deep_model_ablation_command_smoke_coverage_latest.csv")
    parser.add_argument("--doc-output", default="docs/DEEPUplift_DEEP_MODEL_ABLATION_COMMAND_SMOKE_COVERAGE.md")
    args = parser.parse_args()

    payload = build_payload(
        read_json(ROOT / args.contract),
        read_json(ROOT / args.fallback_smoke),
        read_json(ROOT / args.recommended_smoke),
    )
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
                "contract_rows": (payload.get("summary") or {}).get("contract_rows"),
                "any_smoked_rows": (payload.get("summary") or {}).get("any_smoked_rows"),
                "recommended_smoked_rows": (payload.get("summary") or {}).get("recommended_smoked_rows"),
                "coverage_gate": (payload.get("summary") or {}).get("coverage_gate"),
            },
            ensure_ascii=False,
        )
    )
    if payload.get("status") != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
