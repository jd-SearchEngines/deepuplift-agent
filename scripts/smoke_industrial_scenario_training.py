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
from deepuplift.core.industrial_scenario_lab import SCENARIO_DATASET_IDS, scenario_cards
from deepuplift.core.readiness import decision_readiness
from deepuplift.core.registry import missing_dependencies


MANIFEST = ROOT / "examples" / "datasets" / "manifest.json"

MODEL_CANDIDATES = {
    "synthetic_coupon_profit_uplift_8k": ["DRLearnerLightGBM", "TLearnerLightGBM", "XLearnerLightGBM", "TLearnerGBM"],
    "synthetic_growth_delayed_feedback_7k": ["TLearnerLightGBM", "DRLearnerLightGBM", "TLearnerGBM"],
    "synthetic_ads_full_funnel_ecup_7k": ["TLearnerLightGBM", "DRLearnerLightGBM", "TLearnerGBM"],
    "synthetic_recommendation_intervention_7k": ["TLearnerLightGBM", "SkLiftTwoModelsLightGBM", "TLearnerGBM"],
    "synthetic_marketplace_subsidy_uplift_7k": ["TLearnerLightGBM", "DRLearnerLightGBM", "TLearnerGBM"],
    "synthetic_llm_routing_uplift_8k": ["TLearnerLightGBM", "DRLearnerLightGBM", "TLearnerGBM"],
    "synthetic_llm_multi_action_routing_9k": ["DRLearnerLightGBM", "RLearnerLightGBM", "PAVCalibratedDRLearnerLightGBM", "TLearnerGBM"],
    "synthetic_baidu_waimai_red_packet_8k": ["TLearnerLightGBM", "XLearnerLightGBM", "DRLearnerLightGBM", "TLearnerGBM"],
    "synthetic_didi_passenger_subsidy_8k": ["RLearnerLightGBM", "DRLearnerLightGBM", "TLearnerLightGBM", "TLearnerGBM"],
    "synthetic_didi_driver_supply_subsidy_8k": ["DRLearnerLightGBM", "RLearnerLightGBM", "TLearnerLightGBM", "TLearnerGBM"],
    "synthetic_didi_budget_allocation_5k": ["DRLearnerLightGBM", "RLearnerLightGBM", "TLearnerLightGBM", "TLearnerGBM"],
    "synthetic_shopee_ads_full_funnel_8k": ["DRLearnerLightGBM", "RLearnerLightGBM", "TLearnerLightGBM", "TLearnerGBM"],
    "synthetic_spotify_inapp_message_uplift_6k": ["DRLearnerLightGBM", "RLearnerLightGBM", "TLearnerLightGBM", "TLearnerGBM"],
    "synthetic_doordash_ghost_ads_lift_6k": ["DRLearnerLightGBM", "RLearnerLightGBM", "TLearnerLightGBM", "TLearnerGBM"],
    "synthetic_merchant_subsidy_margin_uplift_6k": ["DRLearnerLightGBM", "RLearnerLightGBM", "TLearnerLightGBM", "TLearnerGBM"],
    "synthetic_ecommerce_multi_coupon_uplift_6k": ["DRLearnerLightGBM", "XLearnerLightGBM", "TLearnerLightGBM", "TLearnerGBM"],
    "synthetic_crm_overlap_journey_uplift_6k": ["DRLearnerLightGBM", "RLearnerLightGBM", "TLearnerLightGBM", "TLearnerGBM"],
    "synthetic_continuous_bid_discount_uplift_5k": ["DRLearnerLightGBM", "RLearnerLightGBM", "TLearnerLightGBM", "TLearnerGBM"],
    "synthetic_geo_holdout_incrementality_uplift_6k": ["DRLearnerLightGBM", "RLearnerLightGBM", "TLearnerLightGBM", "TLearnerGBM"],
    "synthetic_game_ops_gift_uplift_6k": ["DRLearnerLightGBM", "RLearnerLightGBM", "TLearnerLightGBM", "TLearnerGBM"],
    "synthetic_fintech_credit_line_uplift_6k": ["DRLearnerLightGBM", "RLearnerLightGBM", "TLearnerLightGBM", "TLearnerGBM"],
    "synthetic_customer_service_escalation_uplift_6k": ["DRLearnerLightGBM", "RLearnerLightGBM", "TLearnerLightGBM", "TLearnerGBM"],
    "synthetic_healthcare_followup_uplift_6k": ["DRLearnerLightGBM", "RLearnerLightGBM", "TLearnerLightGBM", "TLearnerGBM"],
    "synthetic_saas_retention_offer_uplift_6k": ["DRLearnerLightGBM", "RLearnerLightGBM", "TLearnerLightGBM", "TLearnerGBM"],
}


def _load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _dataset_by_id(dataset_id: str) -> dict:
    payload = _load_manifest()
    for row in payload.get("datasets", []):
        if row.get("id") == dataset_id:
            return row
    raise KeyError(f"Dataset not found in manifest: {dataset_id}")


def _pick_model(dataset_id: str) -> tuple[str | None, list[str]]:
    checked = []
    for model_name in MODEL_CANDIDATES[dataset_id]:
        missing = missing_dependencies(model_name)
        checked.append(f"{model_name}:{','.join(missing) if missing else 'ready'}")
        if not missing:
            return model_name, checked
    return None, checked


def _top10_policy_capture(metrics: dict[str, Any]) -> float | None:
    rows = metrics.get("oracle_policy_top_k") or []
    for row in rows:
        if abs(float(row.get("top_fraction", 0.0)) - 0.1) < 1e-9:
            value = row.get("policy_value_capture")
            return float(value) if value is not None else None
    return None


def _scenario_frontier_note(metrics: dict[str, Any]) -> str:
    summary = frontier_metric_summary(metrics)
    fields = {key: value for key, value in summary.items() if value is not None}
    if fields:
        return json.dumps(fields, ensure_ascii=False)
    policy_capture = _top10_policy_capture(metrics)
    return f"oracle_policy_top10_capture={policy_capture}" if policy_capture is not None else "generic uplift metrics"


def _industrial_policy_top10(metrics: dict[str, Any]) -> dict[str, Any]:
    rows = ((metrics.get("industrial_policy") or {}).get("top10") or [])
    return rows[0] if rows else {}


def run_case(dataset_id: str, rows: int, artifacts_dir: Path) -> dict[str, Any]:
    dataset = _dataset_by_id(dataset_id)
    card = next((row for row in scenario_cards() if row["dataset_id"] == dataset_id), {})
    df = pd.read_csv(ROOT / dataset["path"]).head(rows).copy()
    model_name, checked = _pick_model(dataset_id)
    if model_name is None:
        return {
            "dataset_id": dataset_id,
            "scenario": card.get("scenario"),
            "status": "skipped",
            "reason": "no runnable candidate",
            "checked_models": checked,
        }

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
        run_name=f"industrial-{dataset_id}-{model_name.lower()}",
        policy_conversion_value=1.0,
        policy_contact_cost=0.0,
    )
    result = train_uplift_model(config=cfg, data_frame=df)
    metrics = result.eval_metrics
    readiness = decision_readiness(metrics)
    industrial_top10 = _industrial_policy_top10(metrics)
    return {
        "dataset_id": dataset_id,
        "dataset_name": dataset.get("name"),
        "scenario": card.get("scenario"),
        "rows": len(df),
        "model": model_name,
        "checked_models": checked,
        "status": "ok",
        "run_id": result.run_id,
        "run_dir": result.run_dir,
        "qini": metrics.get("qini_score"),
        "auuc": metrics.get("auuc_score"),
        "calibration_mae": metrics.get("calibration_mae"),
        "readiness_score": readiness.get("score"),
        "readiness_level": readiness.get("level"),
        "oracle_policy_top10_capture": _top10_policy_capture(metrics),
        "industrial_top10_policy_value_sum": industrial_top10.get("policy_value_sum"),
        "industrial_top10_roi_proxy": industrial_top10.get("roi_proxy"),
        "frontier_metric_note": _scenario_frontier_note(metrics),
        "predictions": result.artifacts.get("predictions"),
        "metrics_path": result.artifacts.get("metrics"),
        "run_note": result.artifacts.get("run_note"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one lightweight runnable model per industrial scenario dataset.")
    parser.add_argument("--rows", type=int, default=1200)
    parser.add_argument("--dataset-id", action="append", dest="dataset_ids")
    parser.add_argument("--artifacts-dir", default="reports/industrial_scenario_training_runs")
    args = parser.parse_args()

    selected = args.dataset_ids or SCENARIO_DATASET_IDS
    artifacts_dir = ROOT / args.artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    runs = []
    for dataset_id in selected:
        try:
            row = run_case(dataset_id, rows=args.rows, artifacts_dir=artifacts_dir)
        except Exception as exc:
            row = {"dataset_id": dataset_id, "status": "fail", "error": str(exc)}
            failures.append(f"{dataset_id}: {exc}")
        if row.get("status") == "fail":
            failures.append(f"{dataset_id}: {row.get('error')}")
        runs.append(row)

    output = ROOT / "reports" / "industrial_scenario_training_smoke_latest.json"
    payload = {
        "status": "fail" if failures else "ok",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "rows_per_dataset": args.rows,
        "runs": runs,
        "failures": failures,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
