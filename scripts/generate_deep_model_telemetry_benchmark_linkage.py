from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any


LINKAGE_SPECS: dict[str, dict[str, Any]] = {
    "CFRNet.phi_x_representation": {
        "telemetry_family": "representation_balance",
        "telemetry_columns": [
            "train_representation_mmd",
            "valid_representation_mmd",
            "valid_representation_mean_l2",
            "valid_phi_std",
        ],
        "benchmark_question": "Does representation balancing improve PEHE/QINI/policy over T/DR baselines?",
        "failure_hypothesis": "If MMD improves but PEHE/QINI lag, the balance weight may be trading off outcome fit or underfitting small tabular data.",
        "next_ablation": "Run alpha sweep for CFRNet, persist representation MMD curves, and join per-seed PEHE/QINI/policy deltas.",
        "safe_claim": "CFRNet now has executable representation telemetry and benchmark verdicts in one review row.",
        "blocked_claim": "Do not claim representation balance caused benchmark improvement until alpha ablation links MMD movement to PEHE/QINI/policy lift.",
    },
    "DragonNet.e_hat_propensity": {
        "telemetry_family": "propensity_overlap",
        "telemetry_columns": [
            "train_propensity_bce",
            "valid_propensity_bce",
            "valid_propensity_ess",
            "valid_e_hat_mean",
            "valid_e_hat_std",
        ],
        "benchmark_question": "Does the propensity head stabilize overlap-sensitive effect and ranking metrics?",
        "failure_hypothesis": "If propensity ESS/calibration look plausible but benchmark lags, the head may be learning treatment assignment without improving CATE ranking.",
        "next_ablation": "Run propensity clipping and tarreg/no-tarreg comparison, then join ESS/calibration with PEHE, ATE error and DR policy value.",
        "safe_claim": "DragonNet propensity telemetry is collected and can be reviewed beside benchmark and failure-attribution rows.",
        "blocked_claim": "Do not claim DragonNet solves overlap or selection bias until propensity diagnostics improve benchmark/OPE sensitivity.",
    },
    "DragonNet.targeted_regularization_components": {
        "telemetry_family": "targeted_regularization",
        "telemetry_columns": [
            "valid_targeted_regularization",
            "epsilon_mean",
            "epsilon_std",
            "train_loss",
            "valid_loss",
        ],
        "benchmark_question": "Does targeted regularization improve PEHE/ATE error or only add loss complexity?",
        "failure_hypothesis": "If tarreg drops but PEHE/ATE do not improve, beta may be mis-scaled or the clever-covariate signal may be noisy.",
        "next_ablation": "Run beta sweep and no-tarreg variant, then require PEHE/ATE improvement before stronger DragonNet claims.",
        "safe_claim": "DragonNet tarreg components are observable during training and reviewable with tuned benchmark variants.",
        "blocked_claim": "Do not sell targeted regularization as a win until beta ablation beats the guarded baseline on effect accuracy.",
    },
    "EFIN.interaction_attention": {
        "telemetry_family": "interaction_attention",
        "telemetry_columns": [
            "valid_attention_entropy",
            "attention_max_mean_weight",
        ],
        "benchmark_question": "Does treatment-aware interaction attention improve ranking or policy utility?",
        "failure_hypothesis": "Attention can be stable without being causal explanation; ranking gains need no-attention ablation and seed stability.",
        "next_ablation": "Run no-attention and wide-attention ablations, then join attention entropy/stability with QINI and policy value.",
        "safe_claim": "EFIN attention telemetry is executable and can be used as debugging evidence for ranking-oriented review.",
        "blocked_claim": "Do not call attention an explanation until stability plus ablation evidence shows it matters.",
    },
    "EFIN.uplift_path_contribution": {
        "telemetry_family": "uplift_path",
        "telemetry_columns": [
            "valid_base_logit_norm",
            "valid_uplift_logit_norm",
            "valid_uplift_path_ratio",
            "valid_loss",
        ],
        "benchmark_question": "Is EFIN learning uplift path signal or mostly response propensity?",
        "failure_hypothesis": "A weak uplift-path ratio can indicate the model is fitting response surfaces rather than heterogenous treatment effect.",
        "next_ablation": "Track uplift-path ratio across seeds and compare with ranking/policy deltas on full-funnel and recommendation scenarios.",
        "safe_claim": "EFIN path telemetry gives a concrete way to discuss response-fit versus uplift-signal risk.",
        "blocked_claim": "Do not claim EFIN learned causal uplift structure until uplift-path movement aligns with QINI/policy lift.",
    },
    "DESCN.propensity_and_exposure": {
        "telemetry_family": "propensity_exposure",
        "telemetry_columns": [
            "valid_p_prpsy_ess",
            "valid_p_prpsy_mean",
            "valid_p_prpsy_std",
            "valid_propensity_loss",
        ],
        "benchmark_question": "Does DESCN's entire-space propensity/exposure path stay stable enough for biased-assignment settings?",
        "failure_hypothesis": "Collapsed exposure propensity can hide selection bias even when outcome heads appear to train.",
        "next_ablation": "Persist propensity calibration and clipping sensitivity, then require ESS/overlap gates before launch claims.",
        "safe_claim": "DESCN propensity/exposure telemetry is available beside classification benchmark verdicts.",
        "blocked_claim": "Do not claim DESCN handles biased exposure until propensity calibration and OPE sensitivity pass.",
    },
    "DESCN.entire_space_heads": {
        "telemetry_family": "entire_space_heads",
        "telemetry_columns": [
            "valid_tau_std",
            "valid_mu0_logit_mean",
            "valid_mu1_logit_mean",
            "valid_head_spread",
        ],
        "benchmark_question": "Are the DESCN heads calibrated enough to support CATE accuracy and ranking claims?",
        "failure_hypothesis": "Low head spread or unstable tau can make many-head architecture look complex without improving CATE.",
        "next_ablation": "Add mu0/mu1 calibration cards and compare head-spread/tau-std movement with PEHE and oracle top-K recall.",
        "safe_claim": "DESCN head telemetry is executable and tied to benchmark scorecards for model review.",
        "blocked_claim": "Do not claim entire-space heads improve uplift until calibration and benchmark deltas pass.",
    },
    "DESCN.cross_head_constraints": {
        "telemetry_family": "cross_head_constraint",
        "telemetry_columns": [
            "valid_cross_head_residual",
            "valid_head_spread",
            "valid_loss",
        ],
        "benchmark_question": "Do cross-head constraints improve consistency or just add regularization burden?",
        "failure_hypothesis": "A lower residual is useful only if it also preserves PEHE/QINI/policy performance.",
        "next_ablation": "Run no-constraint ablation and join residual deltas to PEHE, QINI and policy value movement.",
        "safe_claim": "DESCN cross-head residual is collected and reviewable against tuned/paper benchmark results.",
        "blocked_claim": "Do not claim constraints improve uplift until residual reduction is linked to benchmark or policy gain.",
    },
}


def latest(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def clean_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def metric_slice(metrics: dict[str, Any], columns: list[str]) -> dict[str, float | None]:
    return {column: clean_float(metrics.get(column)) for column in columns if column in metrics}


def flatten(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return "" if value is None else str(value)


def scorecard_by_model(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = payload.get("model_scorecards") or []
    return {str(row.get("model")): row for row in rows if isinstance(row, dict) and row.get("model")}


def average_leaderboard_by_model(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = payload.get("leaderboard") or []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if isinstance(row, dict) and row.get("model"):
            grouped.setdefault(str(row.get("model")), []).append(row)
    out: dict[str, dict[str, Any]] = {}
    metrics = ["pehe_mean", "ate_error_mean", "qini_mean", "auuc_mean", "policy_observed_net_value_mean"]
    for model, model_rows in grouped.items():
        values: dict[str, Any] = {"model": model, "datasets": len({row.get("dataset_id") for row in model_rows})}
        for metric in metrics:
            nums = [clean_float(row.get(metric)) for row in model_rows]
            nums = [number for number in nums if number is not None]
            values[metric] = sum(nums) / len(nums) if nums else None
        out[model] = values
    return out


def baseline_context(tuned_cards: dict[str, dict[str, Any]]) -> dict[str, Any]:
    candidates = [tuned_cards.get("TLearnerGBM", {}), tuned_cards.get("DRLearnerGBM", {})]
    candidates = [row for row in candidates if row]
    best_pehe = min(candidates, key=lambda row: clean_float(row.get("best_pehe")) or float("inf")) if candidates else {}
    best_qini = max(candidates, key=lambda row: clean_float(row.get("best_qini")) or float("-inf")) if candidates else {}
    return {
        "best_baseline_pehe_model": best_pehe.get("model") or "",
        "best_baseline_pehe": clean_float(best_pehe.get("best_pehe")),
        "best_baseline_qini_model": best_qini.get("model") or "",
        "best_baseline_qini": clean_float(best_qini.get("best_qini")),
    }


def paper_metrics(model: str, paper_cards: dict[str, dict[str, Any]], paper_leaderboard: dict[str, dict[str, Any]]) -> dict[str, Any]:
    card = paper_cards.get(model) or {}
    fallback = paper_leaderboard.get(model) or {}
    return {
        "paper_verdict": card.get("verdict") or fallback.get("verdict") or "",
        "paper_pehe_mean": clean_float(card.get("pehe_mean_over_datasets") or fallback.get("pehe_mean")),
        "paper_ate_error_mean": clean_float(card.get("ate_error_mean_over_datasets") or fallback.get("ate_error_mean")),
        "paper_qini_mean": clean_float(card.get("qini_mean_over_datasets") or fallback.get("qini_mean")),
        "paper_auuc_mean": clean_float(card.get("auuc_mean_over_datasets") or fallback.get("auuc_mean")),
        "paper_policy_value_mean": clean_float(
            card.get("policy_top10_oracle_value_mean_over_datasets") or fallback.get("policy_observed_net_value_mean")
        ),
        "paper_stability_counts": card.get("stability_counts") or {},
        "paper_winner_counts": card.get("winner_counts") or {},
        "paper_interview_line": card.get("interview_line") or "",
        "paper_watchout": card.get("watchout") or "",
    }


def tuned_metrics(model: str, tuned_cards: dict[str, dict[str, Any]], baseline: dict[str, Any]) -> dict[str, Any]:
    card = tuned_cards.get(model) or {}
    best_pehe = clean_float(card.get("best_pehe"))
    best_qini = clean_float(card.get("best_qini"))
    baseline_pehe = clean_float(baseline.get("best_baseline_pehe"))
    baseline_qini = clean_float(baseline.get("best_baseline_qini"))
    pehe_delta = best_pehe - baseline_pehe if best_pehe is not None and baseline_pehe is not None else clean_float(card.get("avg_delta_pehe_vs_baseline"))
    qini_delta = best_qini - baseline_qini if best_qini is not None and baseline_qini is not None else clean_float(card.get("avg_delta_qini_vs_baseline"))
    if pehe_delta is not None and qini_delta is not None:
        if pehe_delta <= 0 and qini_delta >= 0:
            baseline_verdict = "beats_or_matches_baseline"
        elif pehe_delta <= 0:
            baseline_verdict = "accuracy_gain_ranking_review"
        elif qini_delta >= 0:
            baseline_verdict = "ranking_gain_accuracy_review"
        else:
            baseline_verdict = "baseline_still_stronger"
    else:
        baseline_verdict = str(card.get("verdict") or "not_available")
    return {
        "tuned_verdict": card.get("verdict") or "",
        "tuned_best_pehe_variant": card.get("best_pehe_variant") or "",
        "tuned_best_qini_variant": card.get("best_qini_variant") or "",
        "tuned_best_pehe": best_pehe,
        "tuned_best_qini": best_qini,
        "tuned_delta_pehe_vs_best_baseline": pehe_delta,
        "tuned_delta_qini_vs_best_baseline": qini_delta,
        "tuned_baseline_verdict": baseline_verdict,
        "tuned_strength": card.get("strength") or "",
        "tuned_watchout": card.get("watchout") or "",
        "tuned_interview_line": card.get("interview_line") or "",
    }


def build_rows(
    collector: dict[str, Any],
    bridge: dict[str, Any],
    tuned_benchmark: dict[str, Any],
    paper_benchmark: dict[str, Any],
    paper_interpretation: dict[str, Any],
) -> list[dict[str, Any]]:
    collector_rows = {
        str(row.get("model")): row
        for row in collector.get("model_rows", [])
        if isinstance(row, dict) and row.get("model")
    }
    latest_epoch_by_model: dict[str, dict[str, Any]] = {}
    for row in collector.get("epoch_rows", []):
        if not isinstance(row, dict) or not row.get("model"):
            continue
        model = str(row.get("model"))
        previous = latest_epoch_by_model.get(model)
        if previous is None or int(row.get("epoch") or 0) >= int(previous.get("epoch") or 0):
            latest_epoch_by_model[model] = row
    bridge_rows = {
        str(row.get("contract_id")): row
        for row in bridge.get("rows", [])
        if isinstance(row, dict) and row.get("contract_id")
    }
    tuned_cards = scorecard_by_model(tuned_benchmark)
    paper_cards = scorecard_by_model(paper_interpretation)
    paper_leaderboard = average_leaderboard_by_model(paper_benchmark)
    baseline = baseline_context(tuned_cards)

    rows: list[dict[str, Any]] = []
    for contract_id, spec in LINKAGE_SPECS.items():
        model = contract_id.split(".", 1)[0]
        collector_row = collector_rows.get(model, {})
        bridge_row = bridge_rows.get(contract_id, {})
        latest_epoch = latest_epoch_by_model.get(model, {})
        metric_source = dict(latest_epoch)
        metric_source.update(collector_row.get("metric_last") or {})
        metric_last = metric_slice(metric_source, spec["telemetry_columns"])
        metric_delta = metric_slice(collector_row.get("metric_delta") or {}, spec["telemetry_columns"])
        source_ready = bool(collector_row and bridge_row and tuned_benchmark.get("status") == "ok" and paper_benchmark.get("status") == "ok")
        p_metrics = paper_metrics(model, paper_cards, paper_leaderboard)
        t_metrics = tuned_metrics(model, tuned_cards, baseline)
        evidence_level = "telemetry_to_benchmark_linkage" if source_ready else "review"
        linkage_status = "link_ready" if source_ready and metric_last else "review"
        rows.append(
            {
                "linkage_id": f"{contract_id}.benchmark_linkage",
                "model": model,
                "contract_id": contract_id,
                "telemetry_family": spec["telemetry_family"],
                "linkage_status": linkage_status,
                "evidence_level": evidence_level,
                "guarded": True,
                "collector_status": collector_row.get("status") or "missing",
                "bridge_status": bridge_row.get("bridge_status") or "missing",
                "contracts_covered": collector_row.get("contracts_covered") or [],
                "telemetry_columns": spec["telemetry_columns"],
                "telemetry_last": metric_last,
                "telemetry_delta": metric_delta,
                "benchmark_question": spec["benchmark_question"],
                "failure_hypothesis": spec["failure_hypothesis"],
                "next_ablation": spec["next_ablation"],
                "paper_verdict": p_metrics["paper_verdict"],
                "paper_pehe_mean": p_metrics["paper_pehe_mean"],
                "paper_ate_error_mean": p_metrics["paper_ate_error_mean"],
                "paper_qini_mean": p_metrics["paper_qini_mean"],
                "paper_auuc_mean": p_metrics["paper_auuc_mean"],
                "paper_policy_value_mean": p_metrics["paper_policy_value_mean"],
                "paper_stability_counts": p_metrics["paper_stability_counts"],
                "paper_winner_counts": p_metrics["paper_winner_counts"],
                "tuned_verdict": t_metrics["tuned_verdict"],
                "tuned_baseline_verdict": t_metrics["tuned_baseline_verdict"],
                "tuned_best_pehe_variant": t_metrics["tuned_best_pehe_variant"],
                "tuned_best_qini_variant": t_metrics["tuned_best_qini_variant"],
                "tuned_best_pehe": t_metrics["tuned_best_pehe"],
                "tuned_best_qini": t_metrics["tuned_best_qini"],
                "tuned_delta_pehe_vs_best_baseline": t_metrics["tuned_delta_pehe_vs_best_baseline"],
                "tuned_delta_qini_vs_best_baseline": t_metrics["tuned_delta_qini_vs_best_baseline"],
                "baseline_context": baseline,
                "safe_claim": spec["safe_claim"],
                "blocked_claim": spec["blocked_claim"],
                "interview_line": (
                    f"{model} {spec['telemetry_family']}: collector={collector_row.get('status', 'missing')}, "
                    f"paper={p_metrics['paper_verdict'] or 'NA'}, tuned={t_metrics['tuned_verdict'] or 'NA'}; "
                    f"next={spec['next_ablation']}"
                ),
            }
        )
    return rows


def build_payload(
    rows: list[dict[str, Any]],
    source_paths: dict[str, Path | None],
) -> dict[str, Any]:
    model_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    families: dict[str, int] = {}
    for row in rows:
        model = str(row.get("model") or "")
        model_counts[model] = model_counts.get(model, 0) + 1
        status = str(row.get("linkage_status") or "")
        status_counts[status] = status_counts.get(status, 0) + 1
        family = str(row.get("telemetry_family") or "")
        families[family] = families.get(family, 0) + 1
    link_ready = status_counts.get("link_ready", 0)
    return {
        "schema_version": 1,
        "status": "ok" if len(rows) >= 8 and link_ready >= 8 else "review",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "source_artifacts": {key: str(value) if value else "" for key, value in source_paths.items()},
        "summary": {
            "models": len(model_counts),
            "linkage_rows": len(rows),
            "link_ready": link_ready,
            "telemetry_families": families,
            "status_counts": status_counts,
            "baseline_context_available": bool(rows and (rows[0].get("baseline_context") or {}).get("best_baseline_pehe_model")),
        },
        "rows": rows,
        "interview_tracks": {
            "30s": "I can now show internal training telemetry beside benchmark verdicts, while keeping causal claims guarded.",
            "5min": "For CFRNet/DragonNet/EFIN/DESCN, each linkage row joins collector metrics, paper scorecard, tuned battle verdict, baseline context, failure hypothesis and next ablation.",
            "15min": "This converts deep model failure attribution into an evidence workflow: collect internal mechanism metrics, compare against PEHE/QINI/policy outcomes, then promote only after ablation links telemetry movement to benchmark movement.",
        },
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "linkage_id",
        "model",
        "contract_id",
        "telemetry_family",
        "linkage_status",
        "evidence_level",
        "guarded",
        "collector_status",
        "bridge_status",
        "contracts_covered",
        "telemetry_columns",
        "telemetry_last",
        "telemetry_delta",
        "benchmark_question",
        "failure_hypothesis",
        "next_ablation",
        "paper_verdict",
        "paper_pehe_mean",
        "paper_ate_error_mean",
        "paper_qini_mean",
        "paper_auuc_mean",
        "paper_policy_value_mean",
        "paper_stability_counts",
        "paper_winner_counts",
        "tuned_verdict",
        "tuned_baseline_verdict",
        "tuned_best_pehe_variant",
        "tuned_best_qini_variant",
        "tuned_best_pehe",
        "tuned_best_qini",
        "tuned_delta_pehe_vs_best_baseline",
        "tuned_delta_qini_vs_best_baseline",
        "baseline_context",
        "safe_claim",
        "blocked_claim",
        "interview_line",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: flatten(row.get(field)) for field in fields})


def build_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary") or {}
    tracks = payload.get("interview_tracks") or {}
    rows = payload.get("rows") or []
    overview = "\n".join(
        "| {model} | {family} | {status} | {paper} | {tuned} | {baseline} | {next_step} |".format(
            model=str(row.get("model", "")).replace("|", "/"),
            family=str(row.get("telemetry_family", "")).replace("|", "/"),
            status=str(row.get("linkage_status", "")).replace("|", "/"),
            paper=str(row.get("paper_verdict", "")).replace("|", "/"),
            tuned=str(row.get("tuned_verdict", "")).replace("|", "/"),
            baseline=str(row.get("tuned_baseline_verdict", "")).replace("|", "/"),
            next_step=str(row.get("next_ablation", "")).replace("|", "/"),
        )
        for row in rows
    )
    details = []
    for row in rows:
        details.append(
            f"""## {row.get("linkage_id")}

Model: `{row.get("model")}`
Contract: `{row.get("contract_id")}`
Telemetry family: `{row.get("telemetry_family")}`
Status: `{row.get("linkage_status")}`

Telemetry:
- Columns: `{", ".join(row.get("telemetry_columns") or [])}`
- Last values: `{json.dumps(row.get("telemetry_last") or {}, ensure_ascii=False, sort_keys=True)}`
- Deltas: `{json.dumps(row.get("telemetry_delta") or {}, ensure_ascii=False, sort_keys=True)}`

Benchmark context:
- Paper verdict: `{row.get("paper_verdict")}`
- Paper PEHE / ATE error / QINI / AUUC: `{row.get("paper_pehe_mean")}` / `{row.get("paper_ate_error_mean")}` / `{row.get("paper_qini_mean")}` / `{row.get("paper_auuc_mean")}`
- Tuned verdict: `{row.get("tuned_verdict")}`
- Tuned baseline verdict: `{row.get("tuned_baseline_verdict")}`
- Tuned best variants: PEHE `{row.get("tuned_best_pehe_variant")}`, QINI `{row.get("tuned_best_qini_variant")}`

Review:
- Benchmark question: {row.get("benchmark_question")}
- Failure hypothesis: {row.get("failure_hypothesis")}
- Next ablation: {row.get("next_ablation")}
- Safe claim: {row.get("safe_claim")}
- Blocked claim: {row.get("blocked_claim")}
- Interview line: {row.get("interview_line")}
"""
        )
    return f"""# DeepUplift Deep Model Telemetry-Benchmark Linkage

Generated at: `{payload.get("generated_at")}`

This report links internal deep-model telemetry to benchmark outcomes without over-claiming causality. It reads:

1. `reports/deep_model_trainer_collector_smoke_latest.json` for executable per-epoch telemetry.
2. `reports/deep_model_hook_training_bridge_latest.json` for guarded contract readiness.
3. `reports/deep_model_tuned_benchmark_latest.json` for tuned battle cards and baseline comparison.
4. `reports/paper_benchmark_latest.json` plus `reports/paper_benchmark_interpretation_latest.json` for paper-level PEHE/QINI/policy scorecards.

The linkage is intentionally guarded: it proves the evidence can be joined, then names the ablation required before claiming an internal metric caused benchmark improvement.

## Summary

| Metric | Value |
| --- | ---: |
| Models | {summary.get("models", 0)} |
| Linkage rows | {summary.get("linkage_rows", 0)} |
| Link-ready rows | {summary.get("link_ready", 0)} |
| Telemetry families | {len(summary.get("telemetry_families") or {})} |

Source artifacts: `{json.dumps(payload.get("source_artifacts") or {}, ensure_ascii=False, sort_keys=True)}`

## Interview Tracks

- 30s: {tracks.get("30s", "")}
- 5min: {tracks.get("5min", "")}
- 15min: {tracks.get("15min", "")}

## Linkage Overview

| Model | Telemetry family | Status | Paper verdict | Tuned verdict | Baseline verdict | Next ablation |
| --- | --- | --- | --- | --- | --- | --- |
{overview}

{chr(10).join(details)}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Join deep model trainer telemetry with paper/tuned benchmark evidence.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--json-output", default="reports/deep_model_telemetry_benchmark_linkage_latest.json")
    parser.add_argument("--csv-output", default="reports/deep_model_telemetry_benchmark_linkage_latest.csv")
    parser.add_argument("--doc-output", default="docs/DEEPUplift_DEEP_MODEL_TELEMETRY_BENCHMARK_LINKAGE.md")
    args = parser.parse_args()

    root = Path(args.root)
    reports_dir = root / "reports"
    source_paths = {
        "trainer_collector": latest(reports_dir, "deep_model_trainer_collector_smoke_latest.json"),
        "hook_training_bridge": latest(reports_dir, "deep_model_hook_training_bridge_latest.json"),
        "deep_model_tuned_benchmark": latest(reports_dir, "deep_model_tuned_benchmark_latest.json"),
        "paper_benchmark": latest(reports_dir, "paper_benchmark_latest.json"),
        "paper_benchmark_interpretation": latest(reports_dir, "paper_benchmark_interpretation_latest.json"),
    }
    collector = read_json(source_paths["trainer_collector"])
    bridge = read_json(source_paths["hook_training_bridge"])
    tuned_benchmark = read_json(source_paths["deep_model_tuned_benchmark"])
    paper_benchmark = read_json(source_paths["paper_benchmark"])
    paper_interpretation = read_json(source_paths["paper_benchmark_interpretation"])
    rows = build_rows(collector, bridge, tuned_benchmark, paper_benchmark, paper_interpretation)
    payload = build_payload(rows, source_paths)

    json_path = Path(args.json_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(Path(args.csv_output), rows)
    doc_path = Path(args.doc_output)
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    doc_path.write_text(build_markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload.get("status"),
                "json": str(json_path),
                "csv": args.csv_output,
                "markdown": str(doc_path),
                "rows": (payload.get("summary") or {}).get("linkage_rows"),
                "link_ready": (payload.get("summary") or {}).get("link_ready"),
            },
            ensure_ascii=False,
        )
    )
    if payload.get("status") != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
