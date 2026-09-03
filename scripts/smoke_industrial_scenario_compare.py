from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepuplift.core import UpliftConfig, train_uplift_model
from deepuplift.core.evaluator import frontier_metric_summary
from deepuplift.core.industrial_scenario_lab import scenario_cards
from deepuplift.core.readiness import decision_readiness
from deepuplift.core.registry import missing_dependencies


MANIFEST = ROOT / "examples" / "datasets" / "manifest.json"

DEFAULT_DATASETS = [
    "synthetic_baidu_waimai_red_packet_8k",
    "synthetic_didi_passenger_subsidy_8k",
    "synthetic_didi_driver_supply_subsidy_8k",
    "synthetic_didi_budget_allocation_5k",
    "synthetic_shopee_ads_full_funnel_8k",
]

SCENARIO_MODELS = {
    "synthetic_coupon_profit_uplift_8k": ["TLearnerLightGBM", "DRLearnerLightGBM", "XLearnerLightGBM"],
    "synthetic_growth_delayed_feedback_7k": ["TLearnerLightGBM", "DRLearnerLightGBM", "PAVCalibratedDRLearnerLightGBM"],
    "synthetic_ads_full_funnel_ecup_7k": ["TLearnerLightGBM", "DRLearnerLightGBM", "RLearnerLightGBM"],
    "synthetic_recommendation_intervention_7k": ["TLearnerLightGBM", "DRLearnerLightGBM", "SkLiftTwoModelsLightGBM"],
    "synthetic_marketplace_subsidy_uplift_7k": ["TLearnerLightGBM", "DRLearnerLightGBM", "RLearnerLightGBM"],
    "synthetic_llm_routing_uplift_8k": ["TLearnerLightGBM", "DRLearnerLightGBM", "PAVCalibratedDRLearnerLightGBM"],
    "synthetic_llm_multi_action_routing_9k": ["DRLearnerLightGBM", "RLearnerLightGBM", "PAVCalibratedDRLearnerLightGBM"],
    "synthetic_baidu_waimai_red_packet_8k": ["DRLearnerLightGBM", "XLearnerLightGBM", "TLearnerLightGBM"],
    "synthetic_didi_passenger_subsidy_8k": ["DRLearnerLightGBM", "RLearnerLightGBM", "TLearnerLightGBM"],
    "synthetic_didi_driver_supply_subsidy_8k": ["DRLearnerLightGBM", "RLearnerLightGBM", "TLearnerLightGBM"],
    "synthetic_didi_budget_allocation_5k": ["DRLearnerLightGBM", "RLearnerLightGBM", "TLearnerLightGBM"],
    "synthetic_shopee_ads_full_funnel_8k": ["DRLearnerLightGBM", "RLearnerLightGBM", "TLearnerLightGBM"],
    "synthetic_spotify_inapp_message_uplift_6k": ["DRLearnerLightGBM", "RLearnerLightGBM", "TLearnerLightGBM"],
    "synthetic_doordash_ghost_ads_lift_6k": ["DRLearnerLightGBM", "RLearnerLightGBM", "TLearnerLightGBM"],
    "synthetic_merchant_subsidy_margin_uplift_6k": ["DRLearnerLightGBM", "RLearnerLightGBM", "TLearnerLightGBM"],
    "synthetic_ecommerce_multi_coupon_uplift_6k": ["DRLearnerLightGBM", "XLearnerLightGBM", "TLearnerLightGBM"],
    "synthetic_crm_overlap_journey_uplift_6k": ["DRLearnerLightGBM", "RLearnerLightGBM", "TLearnerLightGBM"],
    "synthetic_continuous_bid_discount_uplift_5k": ["DRLearnerLightGBM", "RLearnerLightGBM", "TLearnerLightGBM"],
    "synthetic_geo_holdout_incrementality_uplift_6k": ["DRLearnerLightGBM", "RLearnerLightGBM", "TLearnerLightGBM"],
    "synthetic_game_ops_gift_uplift_6k": ["DRLearnerLightGBM", "RLearnerLightGBM", "TLearnerLightGBM"],
    "synthetic_fintech_credit_line_uplift_6k": ["DRLearnerLightGBM", "RLearnerLightGBM", "TLearnerLightGBM"],
    "synthetic_customer_service_escalation_uplift_6k": ["DRLearnerLightGBM", "RLearnerLightGBM", "TLearnerLightGBM"],
    "synthetic_healthcare_followup_uplift_6k": ["DRLearnerLightGBM", "RLearnerLightGBM", "TLearnerLightGBM"],
    "synthetic_saas_retention_offer_uplift_6k": ["DRLearnerLightGBM", "RLearnerLightGBM", "TLearnerLightGBM"],
}


def _model_candidates(dataset_id: str) -> list[str]:
    candidates = list(SCENARIO_MODELS.get(dataset_id, ["TLearnerLightGBM", "DRLearnerLightGBM"]))
    for fallback in ["DRLearnerGBM", "TLearnerGBM"]:
        if fallback not in candidates:
            candidates.append(fallback)
    return candidates


def _load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _dataset_by_id(dataset_id: str) -> dict:
    payload = _load_manifest()
    for row in payload.get("datasets", []):
        if row.get("id") == dataset_id:
            return row
    raise KeyError(f"Dataset not found: {dataset_id}")


def _industrial_top10(metrics: dict[str, Any]) -> dict[str, Any]:
    rows = ((metrics.get("industrial_policy") or {}).get("top10") or [])
    return rows[0] if rows else {}


def _top10_uplift(metrics: dict[str, Any]) -> float | None:
    rows = metrics.get("top_k") or []
    for row in rows:
        if abs(float(row.get("top_fraction", 0.0)) - 0.1) < 1e-9:
            value = row.get("observed_uplift")
            return float(value) if value is not None else None
    return None


def _top_fraction_row(rows: list[dict[str, Any]] | dict[str, Any], fraction: float) -> dict[str, Any]:
    if isinstance(rows, dict):
        rows = rows.get("rows") or []
    for row in rows or []:
        try:
            if abs(float(row.get("top_fraction", 0.0)) - fraction) < 1e-9:
                return row
        except Exception:
            continue
    return {}


def _overlap_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    overlap = metrics.get("overlap_trim") or {}
    propensity = overlap.get("propensity") or {}
    trim05 = {}
    for row in overlap.get("rows") or []:
        try:
            if abs(float(row.get("trim", 0.0)) - 0.05) < 1e-9:
                trim05 = row
                break
        except Exception:
            continue
    return {
        "weak_overlap_rate": propensity.get("weak_overlap_rate"),
        "propensity_p05": propensity.get("p05"),
        "propensity_p95": propensity.get("p95"),
        "trim05_kept_fraction": trim05.get("kept_fraction"),
        "trim05_qini": trim05.get("qini_score"),
    }


def train_one(dataset: dict, scenario: str, model_name: str, rows: int, artifacts_dir: Path) -> dict[str, Any]:
    if missing_dependencies(model_name):
        return {
            "dataset_id": dataset["id"],
            "scenario": scenario,
            "model": model_name,
            "status": "skipped",
            "missing_dependencies": missing_dependencies(model_name),
        }
    df = pd.read_csv(ROOT / dataset["path"]).head(rows).copy()
    cfg = UpliftConfig(
        treatment_col=dataset["treatment_col"],
        outcome_col=dataset["outcome_col"],
        feature_cols=dataset["feature_cols"],
        model_name=model_name,
        task=dataset.get("task", "classification"),
        max_rows=rows,
        test_size=0.30,
        sensitivity_samples=0,
        bootstrap_samples=0,
        artifacts_dir=str(artifacts_dir),
        run_name=f"scenario-compare-{dataset['id']}-{model_name.lower()}",
    )
    result = train_uplift_model(config=cfg, data_frame=df)
    metrics = result.eval_metrics
    readiness = decision_readiness(metrics)
    industrial = _industrial_top10(metrics)
    frontier = frontier_metric_summary(metrics)
    policy_value_sum = industrial.get("policy_value_sum")
    oracle_top10 = _top_fraction_row(metrics.get("oracle_top_k") or {}, 0.1)
    oracle_policy_top10 = _top_fraction_row(metrics.get("oracle_policy_top_k") or [], 0.1)
    overlap = _overlap_summary(metrics)
    business_decision = "needs_policy_or_cost_aware_model"
    if policy_value_sum is None:
        business_decision = "use_metric_specific_frontier_readout"
    elif float(policy_value_sum) > 0:
        business_decision = "candidate_for_controlled_rollout"
    return {
        "dataset_id": dataset["id"],
        "scenario": scenario,
        "model": model_name,
        "status": "ok",
        "rows": rows,
        "run_id": result.run_id,
        "readiness_score": readiness.get("score"),
        "readiness_level": readiness.get("level"),
        "qini": metrics.get("qini_score"),
        "auuc": metrics.get("auuc_score"),
        "top10_observed_uplift": _top10_uplift(metrics),
        "calibration_mae": metrics.get("calibration_mae"),
        "weak_overlap_rate": overlap.get("weak_overlap_rate"),
        "propensity_p05": overlap.get("propensity_p05"),
        "propensity_p95": overlap.get("propensity_p95"),
        "trim05_kept_fraction": overlap.get("trim05_kept_fraction"),
        "trim05_qini": overlap.get("trim05_qini"),
        "oracle_top10_recall": oracle_top10.get("topk_recall"),
        "oracle_top10_gain_capture": oracle_top10.get("oracle_gain_capture"),
        "oracle_policy_top10_recall": oracle_policy_top10.get("topk_overlap_recall"),
        "oracle_policy_top10_capture": oracle_policy_top10.get("policy_value_capture"),
        "industrial_top10_policy_value_sum": policy_value_sum,
        "industrial_top10_roi_proxy": industrial.get("roi_proxy"),
        "business_decision": business_decision,
        "frontier_metric_schema": frontier.get("frontier_metric_schema"),
        "frontier_top10_conversion_uplift": frontier.get("full_funnel_top10_conversion_uplift"),
        "delayed_d30_top10_uplift": frontier.get("delayed_d30_top10_uplift"),
        "run_dir": result.run_dir,
        "metrics_path": result.artifacts.get("metrics"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run scenario-aware multi-model compare on industrial synthetic datasets.")
    parser.add_argument("--rows", type=int, default=800)
    parser.add_argument("--dataset-id", action="append", dest="dataset_ids")
    parser.add_argument("--all", action="store_true", help="Run all industrial scenario datasets instead of the default short set.")
    parser.add_argument("--artifacts-dir", default="reports/industrial_scenario_compare_runs")
    parser.add_argument("--output-prefix", default="industrial_scenario_compare")
    parser.add_argument("--no-update-latest", action="store_true")
    args = parser.parse_args()
    output_prefix = str(args.output_prefix or "industrial_scenario_compare").strip() or "industrial_scenario_compare"

    if args.all:
        selected = list(SCENARIO_MODELS)
    else:
        selected = args.dataset_ids or DEFAULT_DATASETS

    cards = {row["dataset_id"]: row for row in scenario_cards()}
    artifacts_dir = ROOT / args.artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    rows = []
    for dataset_id in selected:
        dataset = _dataset_by_id(dataset_id)
        scenario = cards.get(dataset_id, {}).get("scenario", dataset.get("name"))
        for model_name in _model_candidates(dataset_id):
            try:
                rows.append(train_one(dataset, scenario, model_name, args.rows, artifacts_dir))
            except Exception as exc:
                failures.append(f"{dataset_id}/{model_name}: {exc}")
                rows.append({"dataset_id": dataset_id, "scenario": scenario, "model": model_name, "status": "fail", "error": str(exc)})

    ranking = pd.DataFrame([row for row in rows if row.get("status") == "ok"])
    best_rows = []
    if not ranking.empty:
        score_cols = ["industrial_top10_policy_value_sum", "readiness_score", "qini", "top10_observed_uplift"]
        for dataset_id, part in ranking.groupby("dataset_id"):
            view = part.copy()
            for col in score_cols:
                if col not in view.columns:
                    view[col] = None
            view = view.sort_values(score_cols, ascending=[False, False, False, False], na_position="last")
            best_rows.append(view.iloc[0].to_dict())

    reports_dir = ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    json_path = reports_dir / f"{output_prefix}_smoke_{timestamp}.json"
    csv_path = reports_dir / f"{output_prefix}_{timestamp}.csv"
    latest_json = reports_dir / f"{output_prefix}_smoke_latest.json"
    latest_csv = reports_dir / f"{output_prefix}_latest.csv"
    if rows:
        pd.DataFrame(rows).to_csv(csv_path, index=False)
    payload = {
        "status": "fail" if failures else "ok",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "rows_per_dataset": args.rows,
        "datasets": selected,
        "runs": rows,
        "best_by_dataset": best_rows,
        "csv": str(csv_path.relative_to(ROOT)),
        "failures": failures,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.no_update_latest:
        if rows:
            pd.DataFrame(rows).to_csv(latest_csv, index=False)
        latest_payload = dict(payload)
        latest_payload["csv"] = str(latest_csv.relative_to(ROOT))
        latest_json.write_text(json.dumps(latest_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "json": str(json_path.relative_to(ROOT)), "csv": str(csv_path.relative_to(ROOT)), "runs": len(rows), "failures": failures}, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
