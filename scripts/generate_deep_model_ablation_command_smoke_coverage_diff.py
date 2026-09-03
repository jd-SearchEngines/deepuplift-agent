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


def pct(numerator: int, denominator: int) -> float:
    return round(numerator / max(denominator, 1), 4)


def coverage_gate(smoked_rows: int, total_rows: int) -> str:
    return "ok" if total_rows > 0 and smoked_rows == total_rows else "review_required"


def transition_for(row: dict[str, Any]) -> str:
    fallback_smoked = bool(row.get("fallback_smoked"))
    recommended_smoked = bool(row.get("recommended_smoked"))
    if fallback_smoked and recommended_smoked:
        return "fallback_to_both_smoked"
    if recommended_smoked:
        return "unsmoked_to_runner_native_smoked"
    if fallback_smoked:
        return "fallback_only_no_runner_native_smoke"
    if row.get("contract_status") == "runner_supported" and row.get("fallback_ready"):
        return "contract_ready_backlog"
    return "contract_needs_review"


def interview_claim_for(row: dict[str, Any]) -> str:
    transition = row.get("transition")
    if transition == "fallback_to_both_smoked":
        return "Strong command evidence: both fallback and runner-native recommended paths executed in isolated smoke."
    if transition == "unsmoked_to_runner_native_smoked":
        return "Runner-native executable evidence: the recommended command now runs with safe smoke overrides."
    if transition == "fallback_only_no_runner_native_smoke":
        return "Fallback executable evidence only; run recommended-command smoke before claiming native CLI coverage."
    if transition == "contract_ready_backlog":
        return "Contract-ready backlog only; do not claim execution until smoke manifest exists."
    return "Review command contract before using this row as evidence."


def next_action_for(row: dict[str, Any]) -> str:
    transition = row.get("transition")
    if transition == "fallback_to_both_smoked":
        return "Upgrade this variant from command smoke to multi-seed tuned benchmark evidence."
    if transition == "unsmoked_to_runner_native_smoked":
        return "Optional fallback symmetry; primary next step is larger benchmark runs and battle-card interpretation."
    if transition == "fallback_only_no_runner_native_smoke":
        return "Run runner-native recommended command smoke for this variant."
    if transition == "contract_ready_backlog":
        return "Run `python scripts/smoke_deep_model_ablation_recommended_commands.py --all` or filter by model/variant."
    return "Repair CLI contract before smoke execution."


def build_rows(coverage_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in coverage_payload.get("rows") or []:
        if not isinstance(source, dict) or not source.get("variant_id"):
            continue
        baseline_smoked = bool(source.get("fallback_smoked"))
        current_smoked = bool(source.get("any_smoked"))
        transition = transition_for(source)
        row = {
            "priority": source.get("priority"),
            "model": source.get("model"),
            "variant_id": source.get("variant_id"),
            "experiment_type": source.get("experiment_type"),
            "claim_tier": source.get("claim_tier"),
            "contract_status": source.get("contract_status"),
            "fallback_ready": bool(source.get("fallback_ready")),
            "baseline_fallback_smoked": baseline_smoked,
            "current_any_smoked": current_smoked,
            "runner_native_recommended_smoked": bool(source.get("recommended_smoked")),
            "transition": transition,
            "smoke_gain": int(current_smoked) - int(baseline_smoked),
            "recommended_runner_aliases": source.get("recommended_runner_aliases") or [],
            "recommended_smoke_aliases": source.get("recommended_smoke_aliases") or [],
            "fallback_runner_manifest": source.get("fallback_runner_manifest"),
            "recommended_runner_manifest": source.get("recommended_runner_manifest"),
            "interview_claim": interview_claim_for({"transition": transition}),
            "next_action": next_action_for({"transition": transition}),
        }
        rows.append(row)
    return rows


def build_payload(coverage_payload: dict[str, Any], recommended_smoke_payload: dict[str, Any]) -> dict[str, Any]:
    rows = build_rows(coverage_payload)
    total_rows = len(rows)
    baseline_smoked_rows = sum(1 for row in rows if row.get("baseline_fallback_smoked"))
    current_smoked_rows = sum(1 for row in rows if row.get("current_any_smoked"))
    runner_native_smoked_rows = sum(1 for row in rows if row.get("runner_native_recommended_smoked"))
    smoke_gain_rows = sum(1 for row in rows if int(row.get("smoke_gain") or 0) > 0)
    transition_counts: dict[str, int] = {}
    priority_counts: dict[str, dict[str, int]] = {}
    for row in rows:
        transition = str(row.get("transition") or "unknown")
        transition_counts[transition] = transition_counts.get(transition, 0) + 1
        priority = str(row.get("priority") or "unknown")
        priority_counts.setdefault(priority, {"rows": 0, "baseline_smoked": 0, "current_smoked": 0, "runner_native_smoked": 0})
        priority_counts[priority]["rows"] += 1
        if row.get("baseline_fallback_smoked"):
            priority_counts[priority]["baseline_smoked"] += 1
        if row.get("current_any_smoked"):
            priority_counts[priority]["current_smoked"] += 1
        if row.get("runner_native_recommended_smoked"):
            priority_counts[priority]["runner_native_smoked"] += 1

    recommended_summary = recommended_smoke_payload.get("summary") or {}
    before_gate = coverage_gate(baseline_smoked_rows, total_rows)
    after_gate = coverage_gate(current_smoked_rows, total_rows)
    baseline_pct = pct(baseline_smoked_rows, total_rows)
    current_pct = pct(current_smoked_rows, total_rows)
    runner_native_pct = pct(runner_native_smoked_rows, total_rows)
    delta_pct = round(current_pct - baseline_pct, 4)
    status = "ok" if total_rows and current_smoked_rows >= baseline_smoked_rows else "fail"
    return {
        "schema_version": 1,
        "status": status,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "summary": {
            "contract_rows": total_rows,
            "baseline_fallback_smoked_rows": baseline_smoked_rows,
            "current_any_smoked_rows": current_smoked_rows,
            "runner_native_recommended_smoked_rows": runner_native_smoked_rows,
            "smoke_gain_rows": smoke_gain_rows,
            "baseline_coverage_pct": baseline_pct,
            "current_coverage_pct": current_pct,
            "runner_native_coverage_pct": runner_native_pct,
            "coverage_delta_pct": delta_pct,
            "coverage_gate_before_runner_native": before_gate,
            "coverage_gate_after_runner_native": after_gate,
            "coverage_gate_delta": f"{before_gate} -> {after_gate}",
            "transition_counts": transition_counts,
            "priority_counts": priority_counts,
            "runner_native_alias_counts": {
                "models_alias_rows": recommended_summary.get("models_alias_rows"),
                "include_variants_alias_rows": recommended_summary.get("include_variants_alias_rows"),
                "emit_battle_cards_alias_rows": recommended_summary.get("emit_battle_cards_alias_rows"),
            },
        },
        "rows": rows,
        "interview_tracks": {
            "30s": f"I turned ablation commands from {baseline_smoked_rows}/{total_rows} fallback-smoked coverage into {current_smoked_rows}/{total_rows} smoke-covered coverage, with gate `{before_gate}` upgraded to `{after_gate}`.",
            "5min": "The diff explains the engineering upgrade: command contract proves the CLI shape, fallback smoke proves compatibility, and runner-native recommended smoke proves the actual review commands run safely.",
            "15min": "The important boundary is claim hygiene: smoke coverage proves executable evidence and manifest generation, while model superiority still depends on tuned/paper-level benchmark leaderboards.",
            "30min": "Walk through one both-smoked row, one runner-native-only row, then connect the next action to multi-seed tuned benchmark expansion and battle-card interpretation.",
        },
        "safe_claims": [
            "The ablation command surface is runner-native for all current planned variants.",
            "All current ablation variants have at least one isolated smoke execution path after the recommended-command smoke expansion.",
            "Coverage diff is executable-evidence governance; it does not by itself prove a model is better.",
        ],
        "unsafe_claims": [
            "Do not claim a variant is paper-level validated from command smoke alone.",
            "Do not treat fallback-ready command contracts as executed evidence without a smoke manifest.",
            "Do not promote deep-model variants to production without multi-seed benchmark, failure attribution, and launch cards.",
        ],
        "source_artifacts": {
            "deep_model_ablation_command_smoke_coverage": "reports/deep_model_ablation_command_smoke_coverage_latest.json",
            "deep_model_ablation_recommended_command_smoke": "reports/deep_model_ablation_recommended_command_smoke_latest.json",
            "deep_model_ablation_command_contract_smoke": "reports/deep_model_ablation_command_contract_smoke_latest.json",
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
    "baseline_fallback_smoked",
    "current_any_smoked",
    "runner_native_recommended_smoked",
    "transition",
    "smoke_gain",
    "recommended_runner_aliases",
    "recommended_smoke_aliases",
    "fallback_runner_manifest",
    "recommended_runner_manifest",
    "interview_claim",
    "next_action",
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
        "# DeepUplift Deep Model Ablation Command Smoke Coverage Diff",
        "",
        f"Generated at: `{payload.get('generated_at')}`",
        "",
        "This report turns the ablation command smoke coverage artifact into a before/after review story. The baseline is fallback-command smoke coverage; the current state includes runner-native recommended-command smoke coverage.",
        "",
        "## What Changed",
        "",
        f"- Baseline fallback-smoked coverage: `{summary.get('baseline_fallback_smoked_rows')}/{summary.get('contract_rows')}` (`{summary.get('baseline_coverage_pct')}`)",
        f"- Current any-smoked coverage: `{summary.get('current_any_smoked_rows')}/{summary.get('contract_rows')}` (`{summary.get('current_coverage_pct')}`)",
        f"- Runner-native recommended-smoked coverage: `{summary.get('runner_native_recommended_smoked_rows')}/{summary.get('contract_rows')}` (`{summary.get('runner_native_coverage_pct')}`)",
        f"- Smoke gain rows: `{summary.get('smoke_gain_rows')}`",
        f"- Gate change: `{summary.get('coverage_gate_delta')}`",
        f"- Transition counts: `{json.dumps(summary.get('transition_counts') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Why It Matters",
        "",
        "- Command contract answers whether the planned experiment command is supported by the runner.",
        "- Smoke coverage answers whether a safe isolated execution actually produced run manifests.",
        "- Coverage diff answers what was upgraded in this iteration and which claim remains only benchmark-level work.",
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
            "## Variant Transitions",
            "",
            "| Priority | Model | Variant | Baseline fallback smoke | Current smoke | Runner-native smoke | Transition | Next action |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row.get('priority')} | {row.get('model')} | {row.get('variant_id')} | "
            f"{row.get('baseline_fallback_smoked')} | {row.get('current_any_smoked')} | "
            f"{row.get('runner_native_recommended_smoked')} | {row.get('transition')} | {row.get('next_action')} |"
        )
    lines.extend(["", "## Evidence Links", ""])
    for name, artifact in (payload.get("source_artifacts") or {}).items():
        lines.append(f"- {name}: `{artifact}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate before/after diff for ablation command smoke coverage.")
    parser.add_argument("--coverage", default="reports/deep_model_ablation_command_smoke_coverage_latest.json")
    parser.add_argument("--recommended-smoke", default="reports/deep_model_ablation_recommended_command_smoke_latest.json")
    parser.add_argument("--json-output", default="reports/deep_model_ablation_command_smoke_coverage_diff_latest.json")
    parser.add_argument("--csv-output", default="reports/deep_model_ablation_command_smoke_coverage_diff_latest.csv")
    parser.add_argument("--doc-output", default="docs/DEEPUplift_DEEP_MODEL_ABLATION_COMMAND_SMOKE_COVERAGE_DIFF.md")
    args = parser.parse_args()

    payload = build_payload(read_json(ROOT / args.coverage), read_json(ROOT / args.recommended_smoke))
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
                "baseline_fallback_smoked_rows": (payload.get("summary") or {}).get("baseline_fallback_smoked_rows"),
                "current_any_smoked_rows": (payload.get("summary") or {}).get("current_any_smoked_rows"),
                "coverage_gate_delta": (payload.get("summary") or {}).get("coverage_gate_delta"),
            },
            ensure_ascii=False,
        )
    )
    if payload.get("status") != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
