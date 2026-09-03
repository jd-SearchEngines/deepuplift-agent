from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


METRICS: dict[str, dict[str, Any]] = {
    "pehe_mean": {"label": "PEHE", "lower_is_better": True},
    "ate_error_mean": {"label": "ATE error", "lower_is_better": True},
    "qini_mean": {"label": "QINI", "lower_is_better": False},
    "auuc_mean": {"label": "AUUC", "lower_is_better": False},
    "policy_top10_oracle_value_mean": {"label": "Policy oracle Top10", "lower_is_better": False},
    "oracle_top10_recall_mean": {"label": "Oracle Top10 recall", "lower_is_better": False},
    "train_loss_delta_mean": {"label": "Train loss delta", "lower_is_better": False},
    "valid_loss_delta_mean": {"label": "Valid loss delta", "lower_is_better": False},
}


ABLATION_FAMILIES: dict[str, dict[str, Any]] = {
    "CFRNet_low_balance": {
        "hypothesis": "Weaker IPM pressure helps if CFRNet is over-balancing small tabular samples.",
        "mechanism": "representation_balance_weight_down",
        "contract_ids": ["CFRNet.phi_x_representation"],
    },
    "CFRNet_high_balance": {
        "hypothesis": "Stronger IPM pressure helps if treatment/control representation imbalance is the bottleneck.",
        "mechanism": "representation_balance_weight_up",
        "contract_ids": ["CFRNet.phi_x_representation"],
    },
    "DragonNet_no_tarreg": {
        "hypothesis": "Removing targeted regularization tests whether the propensity/outcome heads already stabilize the effect estimate.",
        "mechanism": "targeted_regularization_removed",
        "contract_ids": ["DragonNet.e_hat_propensity", "DragonNet.targeted_regularization_components"],
    },
    "DragonNet_stronger_tarreg": {
        "hypothesis": "Stronger targeted regularization should help if treatment assignment confounding is not adequately corrected.",
        "mechanism": "targeted_regularization_beta_up",
        "contract_ids": ["DragonNet.e_hat_propensity", "DragonNet.targeted_regularization_components"],
    },
    "EFIN_wide": {
        "hypothesis": "Wider interaction towers test whether EFIN is capacity-limited before interpreting attention or path signals.",
        "mechanism": "interaction_capacity_up",
        "contract_ids": ["EFIN.interaction_attention", "EFIN.uplift_path_contribution"],
    },
    "EFIN_no_attention": {
        "hypothesis": "Uniform pooling should hurt ranking/policy if treatment-aware interaction attention is genuinely useful.",
        "mechanism": "interaction_attention_removed",
        "contract_ids": ["EFIN.interaction_attention"],
    },
    "EFIN_path_regularized": {
        "hypothesis": "Conservative uplift-path scaling should help if the uplift path over-dominates the base response path.",
        "mechanism": "uplift_path_scaled_down",
        "contract_ids": ["EFIN.uplift_path_contribution"],
    },
    "DESCN_wide_dropout": {
        "hypothesis": "Wider towers plus dropout test whether entire-space heads need more capacity and regularization.",
        "mechanism": "entire_space_capacity_regularization",
        "contract_ids": ["DESCN.propensity_and_exposure", "DESCN.entire_space_heads"],
    },
    "DESCN_no_constraint": {
        "hypothesis": "Disabling cross-head consistency losses tests whether the constraints improve PEHE/QINI or mostly add optimization pressure.",
        "mechanism": "cross_head_constraints_removed",
        "contract_ids": ["DESCN.cross_head_constraints"],
    },
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _fmt(value: Any, digits: int = 5) -> str:
    number = _num(value)
    if number is None:
        return "NA"
    return f"{number:.{digits}g}"


def _delta(current: Any, reference: Any) -> float | None:
    left = _num(current)
    right = _num(reference)
    if left is None or right is None:
        return None
    return left - right


def _metric_movement(delta: float | None, *, lower_is_better: bool, tolerance: float = 1e-9) -> str:
    if delta is None:
        return "missing"
    if abs(delta) <= tolerance:
        return "flat"
    if lower_is_better:
        return "improved" if delta < 0 else "worse"
    return "improved" if delta > 0 else "worse"


def _leaderboard_index(leaderboard: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(row.get("dataset_id") or ""), str(row.get("variant_id") or "")): row
        for row in leaderboard
        if isinstance(row, dict) and row.get("dataset_id") and row.get("variant_id")
    }


def _baseline_index(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (str(row.get("dataset_id") or ""), str(row.get("variant_id") or "")): row
        for row in rows
        if isinstance(row, dict) and row.get("dataset_id") and row.get("variant_id")
    }


def _gate_index(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        for variant in row.get("required_variants") or []:
            index.setdefault(str(variant), []).append(row)
    return index


def _metric_deltas(row: dict[str, Any], default: dict[str, Any]) -> tuple[dict[str, float | None], dict[str, str]]:
    deltas: dict[str, float | None] = {}
    movements: dict[str, str] = {}
    for metric, spec in METRICS.items():
        delta = _delta(row.get(metric), default.get(metric))
        deltas[f"delta_{metric}_vs_default"] = delta
        movements[metric] = _metric_movement(delta, lower_is_better=bool(spec["lower_is_better"]))
    return deltas, movements


def _dominant_verdict(movements: dict[str, str]) -> str:
    pehe = movements.get("pehe_mean")
    ate = movements.get("ate_error_mean")
    ranking = [movements.get("qini_mean"), movements.get("auuc_mean"), movements.get("policy_top10_oracle_value_mean")]
    loss = [movements.get("train_loss_delta_mean"), movements.get("valid_loss_delta_mean")]
    accuracy_improved = pehe == "improved" or (pehe == "flat" and ate == "improved")
    accuracy_worse = pehe == "worse"
    ranking_improved = any(item == "improved" for item in ranking)
    ranking_worse = any(item == "worse" for item in ranking)
    loss_improved = any(item == "improved" for item in loss)
    if accuracy_improved and ranking_improved:
        return "ablation_supports_accuracy_and_ranking"
    if accuracy_improved and not ranking_worse:
        return "ablation_supports_effect_accuracy"
    if ranking_improved and not accuracy_worse:
        return "ablation_supports_ranking_or_policy"
    if ranking_improved and accuracy_worse:
        return "ranking_gain_accuracy_tradeoff"
    if accuracy_improved and ranking_worse:
        return "accuracy_gain_ranking_tradeoff"
    if loss_improved and not accuracy_worse and not ranking_worse:
        return "optimization_signal_only"
    return "ablation_not_supported_by_latest_metrics"


def _safe_claim(model: str, variant_id: str, verdict: str, movements: dict[str, str]) -> str:
    pehe = movements.get("pehe_mean", "missing")
    qini = movements.get("qini_mean", "missing")
    policy = movements.get("policy_top10_oracle_value_mean", "missing")
    if verdict == "ablation_supports_accuracy_and_ranking":
        return f"{variant_id} improves both effect-accuracy and ranking/policy signals versus the default in this latest benchmark slice."
    if verdict == "ablation_supports_effect_accuracy":
        return f"{variant_id} is an effect-accuracy ablation candidate for {model}; keep ranking/policy claims separate."
    if verdict == "ablation_supports_ranking_or_policy":
        return f"{variant_id} is a ranking/policy candidate for {model}; do not sell it as a cleaner CATE estimator yet."
    if verdict == "ranking_gain_accuracy_tradeoff":
        return f"{variant_id} improves ranking/policy but worsens effect accuracy; position it as business-targeting evidence, not paper-level CATE proof."
    if verdict == "accuracy_gain_ranking_tradeoff":
        return f"{variant_id} improves PEHE/ATE but loses ranking/policy value; inspect calibration and top-K curves before promotion."
    return f"{variant_id} is covered by the benchmark, but latest metric movement is not enough to promote the mechanism claim (PEHE={pehe}, QINI={qini}, policy={policy})."


def _blocked_claim(variant_id: str, verdict: str) -> str:
    if verdict in {"ablation_supports_accuracy_and_ranking", "ablation_supports_effect_accuracy", "ablation_supports_ranking_or_policy"}:
        return "Do not generalize beyond this preset, dataset slice and seed count; require wider rows/seeds before production or paper-level superiority claims."
    return f"Do not claim {variant_id}'s mechanism improves uplift quality until the ablation beats default or baseline on the relevant metric family."


def _interview_line(row: dict[str, Any]) -> str:
    return (
        f"{row['variant_id']} vs {row['default_variant']} on {row['dataset_id']}: "
        f"PEHE delta={_fmt(row.get('delta_pehe_mean_vs_default'))}, "
        f"QINI delta={_fmt(row.get('delta_qini_mean_vs_default'))}, "
        f"policy delta={_fmt(row.get('delta_policy_top10_oracle_value_mean_vs_default'))}; "
        f"verdict={row['verdict']}."
    )


def _build_rows(tuned: dict[str, Any], gate: dict[str, Any]) -> list[dict[str, Any]]:
    leaderboard = tuned.get("leaderboard") or []
    defaults = {
        (str(row.get("dataset_id") or ""), str(row.get("model") or "")): row
        for row in leaderboard
        if isinstance(row, dict) and str(row.get("variant_id") or "").endswith("_default")
    }
    baseline_by_variant = _baseline_index(tuned.get("baseline_comparison") or [])
    ablation_by_variant = _baseline_index(tuned.get("ablation_comparison") or [])
    gate_by_variant = _gate_index(gate.get("rows") or [])
    rows: list[dict[str, Any]] = []
    for current in leaderboard:
        variant_id = str(current.get("variant_id") or "")
        family = str(current.get("family") or "")
        if variant_id.endswith("_default") or family.startswith("baseline"):
            continue
        model = str(current.get("model") or "")
        dataset_id = str(current.get("dataset_id") or "")
        default = defaults.get((dataset_id, model))
        if not default:
            continue
        deltas, movements = _metric_deltas(current, default)
        verdict = _dominant_verdict(movements)
        spec = ABLATION_FAMILIES.get(variant_id, {})
        gates = gate_by_variant.get(variant_id, [])
        baseline_delta = baseline_by_variant.get((dataset_id, variant_id), {})
        native_ablation = ablation_by_variant.get((dataset_id, variant_id), {})
        row: dict[str, Any] = {
            "dataset_id": dataset_id,
            "dataset_name": current.get("dataset_name"),
            "model": model,
            "variant_id": variant_id,
            "default_variant": default.get("variant_id"),
            "mechanism": spec.get("mechanism", "architecture_or_training_ablation"),
            "hypothesis": spec.get("hypothesis", current.get("role", "")),
            "contract_ids": spec.get("contract_ids") or [gate_row.get("contract_id") for gate_row in gates if gate_row.get("contract_id")],
            "gate_decisions": [gate_row.get("gate_decision") for gate_row in gates if gate_row.get("gate_decision")],
            "gate_pass_conditions": [gate_row.get("pass_gate") for gate_row in gates if gate_row.get("pass_gate")],
            "ok_runs": current.get("ok_runs"),
            "seeds": current.get("seeds"),
            "verdict": verdict,
            "native_ablation_verdict": native_ablation.get("verdict"),
            "baseline_verdict": baseline_delta.get("verdict"),
            "best_baseline": baseline_delta.get("best_baseline"),
            "delta_pehe_vs_baseline": baseline_delta.get("delta_pehe_vs_baseline"),
            "delta_qini_vs_baseline": baseline_delta.get("delta_qini_vs_baseline"),
            "delta_auuc_vs_baseline": baseline_delta.get("delta_auuc_vs_baseline"),
            "delta_policy_top10_oracle_value_vs_baseline": baseline_delta.get("delta_policy_top10_oracle_value_vs_baseline"),
            "metric_movements": movements,
        }
        row.update(deltas)
        row["safe_claim"] = _safe_claim(model, variant_id, verdict, movements)
        row["blocked_claim"] = _blocked_claim(variant_id, verdict)
        row["interview_line"] = _interview_line(row)
        row["evidence_refs"] = [
            "reports/deep_model_tuned_benchmark_latest.json",
            "reports/deep_model_tuned_benchmark_battle_cards_latest.csv",
            "reports/deep_model_telemetry_ablation_gate_latest.json",
        ]
        rows.append(row)
    return rows


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    verdict_counts: dict[str, int] = {}
    model_counts: dict[str, int] = {}
    mechanism_counts: dict[str, int] = {}
    gate_decision_counts: dict[str, int] = {}
    for row in rows:
        verdict = str(row.get("verdict") or "unknown")
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
        model = str(row.get("model") or "unknown")
        model_counts[model] = model_counts.get(model, 0) + 1
        mechanism = str(row.get("mechanism") or "unknown")
        mechanism_counts[mechanism] = mechanism_counts.get(mechanism, 0) + 1
        for decision in row.get("gate_decisions") or ["no_gate"]:
            key = str(decision or "no_gate")
            gate_decision_counts[key] = gate_decision_counts.get(key, 0) + 1
    support_rows = sum(1 for row in rows if str(row.get("verdict") or "").startswith("ablation_supports"))
    tradeoff_rows = sum(1 for row in rows if "tradeoff" in str(row.get("verdict") or ""))
    blocked_rows = len(rows) - support_rows - tradeoff_rows
    return {
        "rows": len(rows),
        "models": len(model_counts),
        "variants": len({row.get("variant_id") for row in rows}),
        "datasets": len({row.get("dataset_id") for row in rows}),
        "verdict_counts": verdict_counts,
        "model_counts": model_counts,
        "mechanism_counts": mechanism_counts,
        "gate_decision_counts": gate_decision_counts,
        "support_rows": support_rows,
        "tradeoff_rows": tradeoff_rows,
        "blocked_rows": blocked_rows,
    }


def _family_scorecards(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row.get("model") or ""), str(row.get("variant_id") or "")), []).append(row)
    cards = []
    for (model, variant_id), group in sorted(grouped.items()):
        verdict_counts: dict[str, int] = {}
        pehe_improved = 0
        qini_improved = 0
        policy_improved = 0
        for row in group:
            verdict = str(row.get("verdict") or "unknown")
            verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
            movements = row.get("metric_movements") or {}
            pehe_improved += 1 if movements.get("pehe_mean") == "improved" else 0
            qini_improved += 1 if movements.get("qini_mean") == "improved" else 0
            policy_improved += 1 if movements.get("policy_top10_oracle_value_mean") == "improved" else 0
        best_pehe = min(
            (row for row in group if _num(row.get("delta_pehe_mean_vs_default")) is not None),
            key=lambda row: _num(row.get("delta_pehe_mean_vs_default")) or 1e18,
            default=None,
        )
        best_qini = max(
            (row for row in group if _num(row.get("delta_qini_mean_vs_default")) is not None),
            key=lambda row: _num(row.get("delta_qini_mean_vs_default")) or -1e18,
            default=None,
        )
        cards.append(
            {
                "model": model,
                "variant_id": variant_id,
                "mechanism": (group[0].get("mechanism") if group else ""),
                "datasets": len({row.get("dataset_id") for row in group}),
                "rows": len(group),
                "verdict_counts": verdict_counts,
                "pehe_improved_datasets": pehe_improved,
                "qini_improved_datasets": qini_improved,
                "policy_improved_datasets": policy_improved,
                "best_pehe_dataset": best_pehe.get("dataset_id") if best_pehe else None,
                "best_pehe_delta": best_pehe.get("delta_pehe_mean_vs_default") if best_pehe else None,
                "best_qini_dataset": best_qini.get("dataset_id") if best_qini else None,
                "best_qini_delta": best_qini.get("delta_qini_mean_vs_default") if best_qini else None,
                "safe_claim": group[0].get("safe_claim") if group else "",
                "interview_line": (
                    f"{variant_id}: PEHE improved on {pehe_improved}/{len(group)} dataset rows, "
                    f"QINI improved on {qini_improved}/{len(group)}, policy improved on {policy_improved}/{len(group)}."
                ),
            }
        )
    return cards


def _build_payload(tuned: dict[str, Any], gate: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = _summarize(rows)
    return {
        "schema_version": 1,
        "status": "ok" if rows else "empty",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "benchmark_preset": tuned.get("preset"),
        "benchmark_tier": tuned.get("benchmark_tier"),
        "seeds": tuned.get("seeds") or [],
        "rows_per_dataset": tuned.get("rows_per_dataset"),
        "known_effect_bootstrap_samples": tuned.get("known_effect_bootstrap_samples"),
        "summary": summary,
        "rows": rows,
        "variant_scorecards": _family_scorecards(rows),
        "interview_tracks": {
            "30s": "The ablation interpretation layer explains whether each runnable deep-model variant actually supports its mechanism claim versus the default.",
            "5min": "Open the report after the ablation gate: first confirm all variants exist, then compare PEHE/QINI/policy deltas versus default, and keep safe/blocked claims separate.",
            "15min": "Use CFRNet balance, DragonNet tarreg, EFIN attention/path and DESCN constraints as four examples of mechanism-to-metric review. This turns deep models from architecture name-dropping into auditable causal engineering.",
            "30min": "Walk from source code change to tuned benchmark row, ablation delta, gate decision, safe claim, blocked claim, and next experiment. A claim is only promoted when metric movement matches the mechanism and survives wider seeds/rows.",
        },
        "source_artifacts": {
            "deep_model_tuned_benchmark": "reports/deep_model_tuned_benchmark_latest.json",
            "deep_model_tuned_battle_cards": "reports/deep_model_tuned_benchmark_battle_cards_latest.csv",
            "deep_model_telemetry_ablation_gate": "reports/deep_model_telemetry_ablation_gate_latest.json",
        },
    }


def _flatten(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return "" if value is None else str(value)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "dataset_id",
        "model",
        "variant_id",
        "default_variant",
        "mechanism",
        "hypothesis",
        "verdict",
        "native_ablation_verdict",
        "baseline_verdict",
        "best_baseline",
        "delta_pehe_mean_vs_default",
        "delta_ate_error_mean_vs_default",
        "delta_qini_mean_vs_default",
        "delta_auuc_mean_vs_default",
        "delta_policy_top10_oracle_value_mean_vs_default",
        "delta_oracle_top10_recall_mean_vs_default",
        "delta_train_loss_delta_mean_vs_default",
        "delta_valid_loss_delta_mean_vs_default",
        "delta_pehe_vs_baseline",
        "delta_qini_vs_baseline",
        "delta_auuc_vs_baseline",
        "delta_policy_top10_oracle_value_vs_baseline",
        "metric_movements",
        "contract_ids",
        "gate_decisions",
        "safe_claim",
        "blocked_claim",
        "interview_line",
        "evidence_refs",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _flatten(row.get(field)) for field in fields})


def _markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> list[str]:
    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = []
        for key, _ in columns:
            value = row.get(key)
            if isinstance(value, float):
                value = _fmt(value)
            elif isinstance(value, dict):
                value = ", ".join(f"{k}:{v}" for k, v in sorted(value.items())) or "NA"
            elif isinstance(value, list):
                value = ", ".join(str(item) for item in value)
            values.append(str(value if value not in (None, "") else "NA").replace("|", "/"))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    summary = payload.get("summary") or {}
    tracks = payload.get("interview_tracks") or {}
    rows = payload.get("rows") or []
    lines = [
        "# DeepUplift Deep Model Ablation Interpretation",
        "",
        f"Generated at: `{payload.get('generated_at')}`",
        "",
        "This report sits after the telemetry ablation gate. The gate says which variants are required; this report reads the latest tuned benchmark and explains what each runnable variant actually did versus its default model.",
        "",
        "## Summary",
        "",
        f"- Benchmark preset: `{payload.get('benchmark_preset')}`",
        f"- Benchmark tier: `{payload.get('benchmark_tier')}`",
        f"- Seeds: `{payload.get('seeds')}`",
        f"- Rows: `{summary}`",
        "",
        "## Interview Tracks",
        "",
        f"- 30 seconds: {tracks.get('30s', '')}",
        f"- 5 minutes: {tracks.get('5min', '')}",
        f"- 15 minutes: {tracks.get('15min', '')}",
        f"- 30 minutes: {tracks.get('30min', '')}",
        "",
        "## Variant Scorecards",
        "",
    ]
    lines.extend(
        _markdown_table(
            payload.get("variant_scorecards") or [],
            [
                ("model", "Model"),
                ("variant_id", "Variant"),
                ("mechanism", "Mechanism"),
                ("datasets", "Datasets"),
                ("pehe_improved_datasets", "PEHE improved"),
                ("qini_improved_datasets", "QINI improved"),
                ("policy_improved_datasets", "Policy improved"),
                ("best_pehe_delta", "Best PEHE delta"),
                ("best_qini_delta", "Best QINI delta"),
                ("interview_line", "Interview line"),
            ],
        )
    )
    lines.extend(["", "## Dataset-Level Ablation Rows", ""])
    lines.extend(
        _markdown_table(
            rows,
            [
                ("dataset_id", "Dataset"),
                ("model", "Model"),
                ("variant_id", "Variant"),
                ("verdict", "Verdict"),
                ("delta_pehe_mean_vs_default", "Delta PEHE"),
                ("delta_qini_mean_vs_default", "Delta QINI"),
                ("delta_policy_top10_oracle_value_mean_vs_default", "Delta policy"),
                ("baseline_verdict", "Baseline verdict"),
                ("safe_claim", "Safe claim"),
            ],
        )
    )
    lines.extend(["", "## Claim Boundaries", ""])
    for row in rows:
        lines.extend(
            [
                f"### {row.get('variant_id')} on {row.get('dataset_id')}",
                "",
                f"- Hypothesis: {row.get('hypothesis')}",
                f"- Contract IDs: `{', '.join(row.get('contract_ids') or [])}`",
                f"- Gate decisions: `{', '.join(str(item) for item in row.get('gate_decisions') or []) or 'NA'}`",
                f"- Safe claim: {row.get('safe_claim')}",
                f"- Blocked claim: {row.get('blocked_claim')}",
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
    parser = argparse.ArgumentParser(description="Interpret deep-model ablation deltas from the latest tuned benchmark.")
    parser.add_argument("--tuned-benchmark", default="reports/deep_model_tuned_benchmark_latest.json")
    parser.add_argument("--ablation-gate", default="reports/deep_model_telemetry_ablation_gate_latest.json")
    parser.add_argument("--json-output", default="reports/deep_model_ablation_interpretation_latest.json")
    parser.add_argument("--csv-output", default="reports/deep_model_ablation_interpretation_latest.csv")
    parser.add_argument("--doc-output", default="docs/DEEPUplift_DEEP_MODEL_ABLATION_INTERPRETATION.md")
    args = parser.parse_args()

    tuned = _read_json(ROOT / args.tuned_benchmark)
    gate = _read_json(ROOT / args.ablation_gate)
    rows = _build_rows(tuned, gate)
    payload = _build_payload(tuned, gate, rows)
    _write_json(ROOT / args.json_output, payload)
    write_csv(ROOT / args.csv_output, rows)
    write_markdown(ROOT / args.doc_output, payload)
    print(
        json.dumps(
            {
                "status": payload.get("status"),
                "json": args.json_output,
                "csv": args.csv_output,
                "markdown": args.doc_output,
                "rows": (payload.get("summary") or {}).get("rows"),
                "variants": (payload.get("summary") or {}).get("variants"),
                "support_rows": (payload.get("summary") or {}).get("support_rows"),
            },
            ensure_ascii=False,
        )
    )
    if payload.get("status") != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
