from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_VARIANTS = {
    "CFRNet_low_balance",
    "CFRNet_high_balance",
    "DragonNet_no_tarreg",
    "DragonNet_stronger_tarreg",
    "EFIN_wide",
    "EFIN_no_attention",
    "EFIN_path_regularized",
    "DESCN_wide_dropout",
    "DESCN_no_constraint",
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


def num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def metric(value: Any, digits: int = 5) -> str:
    number = num(value)
    if number is None:
        return "NA"
    return f"{number:.{digits}g}"


def group_by(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if isinstance(row, dict) and row.get(key):
            grouped.setdefault(str(row.get(key)), []).append(row)
    return grouped


def model_promotion_index(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("model")): row
        for row in payload.get("rows") or []
        if isinstance(row, dict) and row.get("model")
    }


def gate_index(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    indexed: dict[str, list[dict[str, Any]]] = {}
    for row in payload.get("rows") or []:
        if not isinstance(row, dict):
            continue
        for variant in row.get("required_variants") or []:
            indexed.setdefault(str(variant), []).append(row)
    return indexed


def movement_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "pehe_improved": 0,
        "qini_improved": 0,
        "policy_improved": 0,
        "ate_improved": 0,
        "loss_improved": 0,
    }
    for row in rows:
        movements = row.get("metric_movements") or {}
        counts["pehe_improved"] += 1 if movements.get("pehe_mean") == "improved" else 0
        counts["qini_improved"] += 1 if movements.get("qini_mean") == "improved" else 0
        counts["policy_improved"] += 1 if movements.get("policy_top10_oracle_value_mean") == "improved" else 0
        counts["ate_improved"] += 1 if movements.get("ate_error_mean") == "improved" else 0
        counts["loss_improved"] += 1 if (
            movements.get("train_loss_delta_mean") == "improved"
            or movements.get("valid_loss_delta_mean") == "improved"
        ) else 0
    return counts


def count_values(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(field) or "unknown")
        counts[value] = counts.get(value, 0) + 1
    return counts


def best_row(rows: list[dict[str, Any]], field: str, *, lower_is_better: bool) -> dict[str, Any]:
    candidates = [row for row in rows if num(row.get(field)) is not None]
    if not candidates:
        return {}
    return sorted(candidates, key=lambda row: num(row.get(field)) or 0.0, reverse=not lower_is_better)[0]


def decision_for_variant(rows: list[dict[str, Any]], gates: list[dict[str, Any]]) -> tuple[str, str, str]:
    verdict_counts = count_values(rows, "verdict")
    baseline_counts = count_values(rows, "baseline_verdict")
    support = sum(count for verdict, count in verdict_counts.items() if verdict.startswith("ablation_supports"))
    tradeoff = sum(count for verdict, count in verdict_counts.items() if "tradeoff" in verdict)
    blocked = len(rows) - support - tradeoff
    beats_baseline = baseline_counts.get("beats_baseline_on_accuracy_and_ranking", 0)
    baseline_still_stronger = baseline_counts.get("baseline_still_stronger", 0)
    ranking_baseline_tradeoff = baseline_counts.get("ranking_gain_accuracy_tradeoff", 0)
    gate_blocked = any(str(gate.get("gate_decision") or "").startswith("blocked") for gate in gates)

    if support >= max(2, math.ceil(len(rows) * 0.5)) and beats_baseline:
        return (
            "promote_variant_for_limited_benchmark_slice",
            "A_local_benchmark_candidate",
            "Variant has repeated default-vs-variant support and at least one dataset-level baseline win; keep the claim tied to this benchmark slice.",
        )
    if support and baseline_still_stronger:
        return (
            "mechanism_supported_but_baseline_blocked",
            "B_local_mechanism_evidence",
            "Variant improves over its default on some metrics, but the stronger T/DR baseline still blocks superiority claims.",
        )
    if support:
        return (
            "promote_as_local_mechanism_evidence",
            "B_local_mechanism_evidence",
            "Variant has local default-vs-variant support; use it as mechanism evidence, not production evidence.",
        )
    if tradeoff and ranking_baseline_tradeoff:
        return (
            "promote_as_business_tradeoff_case",
            "C_tradeoff_case_study",
            "Variant creates a useful accuracy-vs-ranking case study and should be presented as a tradeoff, not a win.",
        )
    if tradeoff:
        return (
            "promote_as_default_vs_variant_tradeoff",
            "C_tradeoff_case_study",
            "Variant has mixed movement versus default; it is useful for failure attribution and metric-conflict discussion.",
        )
    if gate_blocked or blocked:
        return (
            "keep_as_ablation_coverage_only",
            "D_coverage_only",
            "Variant coverage is present, but the latest metrics do not support a mechanism promotion claim.",
        )
    return (
        "needs_more_evidence",
        "D_coverage_only",
        "No stable promotion signal is available in the latest artifact.",
    )


def default_or_variant(rows: list[dict[str, Any]], decision: str) -> str:
    counts = movement_counts(rows)
    support = sum(1 for row in rows if str(row.get("verdict") or "").startswith("ablation_supports"))
    accuracy = counts["pehe_improved"] + counts["ate_improved"]
    ranking = counts["qini_improved"] + counts["policy_improved"]
    if decision == "promote_variant_for_limited_benchmark_slice":
        return "variant_preferred_for_this_benchmark_slice"
    if decision in {"mechanism_supported_but_baseline_blocked", "promote_as_local_mechanism_evidence"}:
        return "variant_for_mechanism_story_default_for_demo"
    if ranking > accuracy:
        return "variant_for_ranking_story_default_for_cate_claim"
    if accuracy > ranking and support:
        return "variant_for_effect_accuracy_story_default_for_policy_claim"
    return "default_preferred_until_better_ablation_evidence"


def score_variant(rows: list[dict[str, Any]], decision: str, gates: list[dict[str, Any]]) -> int:
    counts = movement_counts(rows)
    support = sum(1 for row in rows if str(row.get("verdict") or "").startswith("ablation_supports"))
    tradeoff = sum(1 for row in rows if "tradeoff" in str(row.get("verdict") or ""))
    baseline_counts = count_values(rows, "baseline_verdict")
    score = 35
    score += support * 10
    score += tradeoff * 3
    score += counts["pehe_improved"] * 4
    score += counts["qini_improved"] * 3
    score += counts["policy_improved"] * 3
    score += baseline_counts.get("beats_baseline_on_accuracy_and_ranking", 0) * 12
    score -= baseline_counts.get("baseline_still_stronger", 0) * 5
    score -= sum(1 for gate in gates if str(gate.get("gate_decision") or "").startswith("blocked")) * 4
    if decision == "keep_as_ablation_coverage_only":
        score -= 10
    return max(0, min(score, 100))


def safe_claim(row: dict[str, Any]) -> str:
    decision = row.get("promotion_decision")
    variant = row.get("variant_id")
    model = row.get("model")
    if decision == "promote_variant_for_limited_benchmark_slice":
        return f"{variant} can be shown as a limited benchmark-slice candidate for {model}; cite default deltas, baseline context and seed/row scope together."
    if decision == "mechanism_supported_but_baseline_blocked":
        return f"{variant} supports a local mechanism story for {model}, but the model is still blocked by stronger tabular/meta-learner baselines."
    if decision == "promote_as_local_mechanism_evidence":
        return f"{variant} is safe to use as local default-vs-variant mechanism evidence for {model}; keep production and SOTA claims out."
    if str(decision).startswith("promote_as_"):
        return f"{variant} is useful as a tradeoff case: it teaches why PEHE, QINI and policy value must be reviewed together."
    return f"{variant} is runnable and covered, but the current metrics are not enough to promote the mechanism claim."


def blocked_claim(row: dict[str, Any]) -> str:
    blocks = [
        "Do not claim production lift without online incrementality, OPE/holdout and rollback guardrails.",
        "Do not claim paper-level superiority from a tuned-v1 small-row benchmark slice.",
    ]
    if row.get("baseline_still_stronger_rows"):
        blocks.append("Do not claim the deep variant beats the project baseline while T/DR baselines remain stronger.")
    if row.get("support_rows", 0) == 0:
        blocks.append("Do not claim the ablated mechanism improves uplift quality until default-vs-variant metrics support it.")
    return " ".join(blocks)


def failure_attribution(row: dict[str, Any]) -> str:
    if row.get("baseline_still_stronger_rows"):
        return "Stronger tabular/meta-learner baselines still dominate at least one dataset; explain this as sample-size, optimization or inductive-bias mismatch before promoting deep variants."
    if row.get("tradeoff_rows"):
        return "Metric conflict is the main story: the variant may improve one objective while hurting PEHE, QINI or policy value."
    if row.get("support_rows"):
        return "Local mechanism signal exists, but it needs more rows/seeds and baseline stress before becoming a stronger benchmark claim."
    return "Latest metric movement does not validate the mechanism; keep the variant as coverage evidence and rerun wider ablations."


def build_rows(
    interpretation: dict[str, Any],
    gate: dict[str, Any],
    model_matrix: dict[str, Any],
) -> list[dict[str, Any]]:
    by_variant = group_by(interpretation.get("rows") or [], "variant_id")
    gate_by_variant = gate_index(gate)
    model_index = model_promotion_index(model_matrix)
    rows: list[dict[str, Any]] = []
    for variant in sorted(REQUIRED_VARIANTS | set(by_variant)):
        group = by_variant.get(variant, [])
        if not group:
            model = variant.split("_", 1)[0]
            rows.append(
                {
                    "model": model,
                    "variant_id": variant,
                    "mechanism": "unknown",
                    "datasets": 0,
                    "rows": 0,
                    "promotion_decision": "missing_ablation_evidence",
                    "claim_tier": "E_missing",
                    "promotion_reason": "Required variant is not present in the latest ablation interpretation artifact.",
                    "recommended_claim_boundary": "missing_evidence",
                    "promotion_score": 0,
                    "safe_claim": f"{variant} is required by the ablation gate but missing from the latest interpretation artifact.",
                    "blocked_claim": "Do not discuss this variant as runnable or supported until the benchmark artifact contains it.",
                    "failure_attribution": "Variant evidence is missing.",
                    "evidence_refs": [
                        "reports/deep_model_ablation_interpretation_latest.json",
                        "reports/deep_model_telemetry_ablation_gate_latest.json",
                    ],
                }
            )
            continue
        gates = gate_by_variant.get(variant, [])
        decision, tier, reason = decision_for_variant(group, gates)
        recommendation = default_or_variant(group, decision)
        movement = movement_counts(group)
        verdict_counts = count_values(group, "verdict")
        baseline_counts = count_values(group, "baseline_verdict")
        native_counts = count_values(group, "native_ablation_verdict")
        best_pehe = best_row(group, "delta_pehe_mean_vs_default", lower_is_better=True)
        best_qini = best_row(group, "delta_qini_mean_vs_default", lower_is_better=False)
        model = str(group[0].get("model") or variant.split("_", 1)[0])
        model_card = model_index.get(model, {})
        row: dict[str, Any] = {
            "model": model,
            "variant_id": variant,
            "mechanism": group[0].get("mechanism"),
            "hypothesis": group[0].get("hypothesis"),
            "datasets": len({item.get("dataset_id") for item in group}),
            "rows": len(group),
            "support_rows": sum(1 for item in group if str(item.get("verdict") or "").startswith("ablation_supports")),
            "tradeoff_rows": sum(1 for item in group if "tradeoff" in str(item.get("verdict") or "")),
            "blocked_rows": sum(1 for item in group if str(item.get("verdict") or "") == "ablation_not_supported_by_latest_metrics"),
            "pehe_improved_rows": movement["pehe_improved"],
            "qini_improved_rows": movement["qini_improved"],
            "policy_improved_rows": movement["policy_improved"],
            "loss_improved_rows": movement["loss_improved"],
            "baseline_still_stronger_rows": baseline_counts.get("baseline_still_stronger", 0),
            "beats_baseline_rows": baseline_counts.get("beats_baseline_on_accuracy_and_ranking", 0),
            "baseline_tradeoff_rows": baseline_counts.get("ranking_gain_accuracy_tradeoff", 0),
            "promotion_decision": decision,
            "claim_tier": tier,
            "promotion_reason": reason,
            "recommended_claim_boundary": recommendation,
            "promotion_score": score_variant(group, decision, gates),
            "verdict_counts": verdict_counts,
            "baseline_verdict_counts": baseline_counts,
            "native_ablation_verdict_counts": native_counts,
            "gate_decisions": sorted({str(item.get("gate_decision")) for item in gates if item.get("gate_decision")}),
            "model_evidence_stage": model_card.get("evidence_stage"),
            "model_promotion_tier": model_card.get("promotion_tier"),
            "model_launch_decision": model_card.get("launch_decision"),
            "best_pehe_dataset": best_pehe.get("dataset_id"),
            "best_pehe_delta": best_pehe.get("delta_pehe_mean_vs_default"),
            "best_qini_dataset": best_qini.get("dataset_id"),
            "best_qini_delta": best_qini.get("delta_qini_mean_vs_default"),
            "example_interview_line": group[0].get("interview_line"),
            "evidence_refs": sorted(
                {
                    "reports/deep_model_ablation_interpretation_latest.json",
                    "reports/deep_model_ablation_interpretation_latest.csv",
                    "reports/deep_model_telemetry_ablation_gate_latest.json",
                    "reports/model_evidence_promotion_matrix_latest.json",
                    *[str(ref) for item in group for ref in item.get("evidence_refs") or []],
                }
            ),
        }
        row["safe_claim"] = safe_claim(row)
        row["blocked_claim"] = blocked_claim(row)
        row["failure_attribution"] = failure_attribution(row)
        row["interview_line"] = (
            f"{variant}: tier={tier}, decision={decision}, support={row['support_rows']}/{row['rows']}, "
            f"baseline_stronger={row['baseline_still_stronger_rows']}, recommendation={recommendation}."
        )
        rows.append(row)
    return sorted(rows, key=lambda row: (-int(row.get("promotion_score", 0)), str(row.get("model")), str(row.get("variant_id"))))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    decision_counts = count_values(rows, "promotion_decision")
    tier_counts = count_values(rows, "claim_tier")
    model_counts = count_values(rows, "model")
    return {
        "rows": len(rows),
        "models": len(model_counts),
        "variants": len({row.get("variant_id") for row in rows}),
        "decision_counts": decision_counts,
        "claim_tier_counts": tier_counts,
        "model_counts": model_counts,
        "promotion_ready_rows": sum(1 for row in rows if str(row.get("claim_tier") or "").startswith(("A_", "B_"))),
        "tradeoff_rows": sum(1 for row in rows if str(row.get("claim_tier") or "").startswith("C_")),
        "blocked_or_missing_rows": sum(1 for row in rows if str(row.get("claim_tier") or "").startswith(("D_", "E_"))),
        "top_variants": [row.get("variant_id") for row in rows[:5]],
    }


def build_payload(
    interpretation: dict[str, Any],
    gate: dict[str, Any],
    model_matrix: dict[str, Any],
) -> dict[str, Any]:
    rows = build_rows(interpretation, gate, model_matrix)
    return {
        "schema_version": 1,
        "status": "ok" if rows else "empty",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "benchmark_preset": interpretation.get("benchmark_preset"),
        "benchmark_tier": interpretation.get("benchmark_tier"),
        "seeds": interpretation.get("seeds") or [],
        "summary": summarize(rows),
        "rows": rows,
        "interview_tracks": {
            "30s": "The ablation promotion matrix says which runnable deep-model variants can be used as local mechanism evidence and which claims remain blocked.",
            "5min": "I first check variant coverage, then default-vs-variant PEHE/QINI/policy movement, then baseline context. A variant can be runnable, locally useful, and still blocked from superiority claims.",
            "15min": "This is the guardrail between model deconstruction and over-claiming: CFRNet/DragonNet/EFIN/DESCN variants get claim tiers, safe claims, blocked claims, and a default-or-variant recommendation.",
            "30min": "Walk one variant from source change to tuned benchmark row, ablation interpretation, model promotion context, failure attribution, claim tier, and next experiment before any pilot or paper-level claim is made.",
        },
        "source_artifacts": {
            "deep_model_ablation_interpretation": "reports/deep_model_ablation_interpretation_latest.json",
            "deep_model_telemetry_ablation_gate": "reports/deep_model_telemetry_ablation_gate_latest.json",
            "model_evidence_promotion_matrix": "reports/model_evidence_promotion_matrix_latest.json",
        },
    }


CSV_FIELDS = [
    "model",
    "variant_id",
    "mechanism",
    "datasets",
    "rows",
    "support_rows",
    "tradeoff_rows",
    "blocked_rows",
    "pehe_improved_rows",
    "qini_improved_rows",
    "policy_improved_rows",
    "baseline_still_stronger_rows",
    "beats_baseline_rows",
    "promotion_decision",
    "claim_tier",
    "promotion_score",
    "recommended_claim_boundary",
    "model_evidence_stage",
    "model_promotion_tier",
    "model_launch_decision",
    "best_pehe_dataset",
    "best_pehe_delta",
    "best_qini_dataset",
    "best_qini_delta",
    "safe_claim",
    "blocked_claim",
    "failure_attribution",
    "interview_line",
    "evidence_refs",
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
            if isinstance(value, float):
                value = metric(value)
            elif isinstance(value, (dict, list)):
                value = flatten(value)
            values.append(str(value if value not in (None, "") else "NA").replace("|", "/"))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    summary = payload.get("summary") or {}
    rows = payload.get("rows") or []
    tracks = payload.get("interview_tracks") or {}
    lines = [
        "# DeepUplift Deep Model Ablation Promotion Matrix",
        "",
        f"Generated at: `{payload.get('generated_at')}`",
        "",
        "This report sits after the ablation interpretation layer. It answers a stricter review question: can this runnable variant be promoted into a safe interview or review claim, or should it remain coverage/tradeoff evidence?",
        "",
        "## Summary",
        "",
        f"- Benchmark preset: `{payload.get('benchmark_preset')}`",
        f"- Benchmark tier: `{payload.get('benchmark_tier')}`",
        f"- Seeds: `{payload.get('seeds')}`",
        f"- Rows: `{summary.get('rows')}`",
        f"- Claim tiers: `{json.dumps(summary.get('claim_tier_counts') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- Decisions: `{json.dumps(summary.get('decision_counts') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Interview Tracks",
        "",
        f"- 30 seconds: {tracks.get('30s', '')}",
        f"- 5 minutes: {tracks.get('5min', '')}",
        f"- 15 minutes: {tracks.get('15min', '')}",
        f"- 30 minutes: {tracks.get('30min', '')}",
        "",
        "## Promotion Matrix",
        "",
    ]
    lines.extend(
        markdown_table(
            rows,
            [
                ("model", "Model"),
                ("variant_id", "Variant"),
                ("claim_tier", "Claim tier"),
                ("promotion_decision", "Decision"),
                ("promotion_score", "Score"),
                ("support_rows", "Support"),
                ("tradeoff_rows", "Tradeoff"),
                ("baseline_still_stronger_rows", "Baseline stronger"),
                ("recommended_claim_boundary", "Recommendation"),
                ("safe_claim", "Safe claim"),
            ],
        )
    )
    lines.extend(["", "## Claim Boundaries", ""])
    for row in rows:
        lines.extend(
            [
                f"### {row.get('variant_id')}",
                "",
                f"- Mechanism: `{row.get('mechanism')}`",
                f"- Decision: `{row.get('promotion_decision')}`",
                f"- Claim tier: `{row.get('claim_tier')}`",
                f"- Safe claim: {row.get('safe_claim')}",
                f"- Blocked claim: {row.get('blocked_claim')}",
                f"- Failure attribution: {row.get('failure_attribution')}",
                f"- Interview line: {row.get('interview_line')}",
                "",
            ]
        )
    lines.extend(["## Evidence Links", ""])
    for name, artifact in (payload.get("source_artifacts") or {}).items():
        lines.append(f"- {name}: `{artifact}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate claim promotion matrix for deep-model ablation variants.")
    parser.add_argument("--ablation-interpretation", default="reports/deep_model_ablation_interpretation_latest.json")
    parser.add_argument("--ablation-gate", default="reports/deep_model_telemetry_ablation_gate_latest.json")
    parser.add_argument("--model-promotion-matrix", default="reports/model_evidence_promotion_matrix_latest.json")
    parser.add_argument("--json-output", default="reports/deep_model_ablation_promotion_matrix_latest.json")
    parser.add_argument("--csv-output", default="reports/deep_model_ablation_promotion_matrix_latest.csv")
    parser.add_argument("--doc-output", default="docs/DEEPUplift_DEEP_MODEL_ABLATION_PROMOTION_MATRIX.md")
    args = parser.parse_args()

    interpretation = read_json(ROOT / args.ablation_interpretation)
    gate = read_json(ROOT / args.ablation_gate)
    model_matrix = read_json(ROOT / args.model_promotion_matrix)
    payload = build_payload(interpretation, gate, model_matrix)
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
                "promotion_ready_rows": (payload.get("summary") or {}).get("promotion_ready_rows"),
                "blocked_or_missing_rows": (payload.get("summary") or {}).get("blocked_or_missing_rows"),
            },
            ensure_ascii=False,
        )
    )
    if payload.get("status") != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
