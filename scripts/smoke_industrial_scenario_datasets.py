from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepuplift.core.industrial_scenario_lab import SCENARIO_DATASET_IDS, scenario_cards


MANIFEST = ROOT / "examples" / "datasets" / "manifest.json"


COMMON_ORACLE_COLUMNS = {
    "propensity",
    "true_uplift",
    "true_effect",
    "ite",
    "cate",
    "tau",
    "oracle_policy_value",
    "expected_incremental_value",
    "expected_incremental_profit",
    "incremental_profit",
    "incremental_gmv",
    "redemption_prob",
    "expected_coupon_cost",
    "expected_red_packet_cost",
    "channel_cost",
    "fatigue_cost",
    "touch_cost",
    "reorder_value_7d",
    "true_request_uplift",
    "true_completion_uplift",
    "true_supply_uplift",
    "true_match_uplift",
    "true_click_uplift",
    "true_conversion_uplift",
    "true_direct_uplift",
    "completed_order_value",
    "retention_value_d7",
    "expected_subsidy_cost",
    "driver_subsidy_cost",
    "wait_time_reduction_value",
    "match_rate_value",
    "city_incremental_profit",
    "budget_policy_value",
    "net_marketplace_value",
    "message_cost",
    "fatigue_penalty",
    "optout_risk",
    "retention_value",
    "incremental_minutes",
    "ghost_ad_cost",
    "incremental_sales",
    "sales_lift",
    "merchant_subsidy_cost",
    "merchant_incremental_profit",
    "platform_margin_value",
    "incremental_cost",
    "latency_penalty",
    "cheap_quality",
    "strong_quality",
    "rag_quality",
    "tool_quality",
    "human_quality",
    "selected_action_quality",
    "hallucination_risk",
    "evidence_failure_risk",
    "oracle_best_policy_value",
    "optout_uplift",
    "long_term_value",
    "discount_cost",
    "media_cost",
}

DATASET_CHECKS = {
    "synthetic_coupon_profit_uplift_8k": {
        "kind": "synthetic_coupon_profit_uplift",
        "min_rows": 8000,
        "required_columns": {
            "gross_margin",
            "coupon_face_value",
            "expected_coupon_cost",
            "subsidy_abuse_risk",
            "fatigue_cost",
            "uplift_type",
            "true_uplift",
            "oracle_policy_value",
            "treatment",
            "outcome",
        },
    },
    "synthetic_growth_delayed_feedback_7k": {
        "kind": "synthetic_delayed_feedback_uplift",
        "min_rows": 7000,
        "required_columns": {
            "converted_d1",
            "converted_d7",
            "converted_d14",
            "converted_d30",
            "conversion_delay_days",
            "label_window_days",
            "censored",
            "true_uplift_d30",
            "true_uplift",
            "oracle_policy_value",
            "treatment",
            "outcome",
        },
    },
    "synthetic_ads_full_funnel_ecup_7k": {
        "kind": "synthetic_full_funnel_uplift",
        "min_rows": 7000,
        "required_columns": {
            "impression",
            "click",
            "conversion",
            "true_impression_uplift",
            "true_click_uplift",
            "true_conversion_uplift",
            "true_uplift",
            "media_cost",
            "oracle_policy_value",
            "treatment",
            "outcome",
        },
    },
    "synthetic_recommendation_intervention_7k": {
        "kind": "synthetic_recommendation_intervention_uplift",
        "min_rows": 7000,
        "required_columns": {
            "creative_type",
            "item_category",
            "impression",
            "click",
            "conversion",
            "true_request_uplift",
            "true_click_uplift",
            "true_conversion_uplift",
            "negative_uplift_risk",
            "segment_stability_score",
            "oracle_policy_value",
            "treatment",
            "outcome",
        },
    },
    "synthetic_marketplace_subsidy_uplift_7k": {
        "kind": "synthetic_marketplace_subsidy_uplift",
        "min_rows": 7000,
        "required_columns": {
            "planned_subsidy_amount",
            "marketplace_balance",
            "spillover_risk",
            "interference_risk",
            "true_direct_uplift",
            "net_marketplace_value",
            "oracle_policy_value",
            "treatment",
            "outcome",
        },
    },
    "synthetic_llm_routing_uplift_8k": {
        "kind": "synthetic_llm_routing",
        "min_rows": 8000,
        "required_columns": {
            "small_model_quality",
            "strong_model_quality",
            "true_uplift",
            "incremental_cost",
            "latency_penalty",
            "oracle_policy_value",
            "treatment",
            "outcome",
        },
    },
    "synthetic_llm_multi_action_routing_9k": {
        "kind": "synthetic_llm_multi_action_routing",
        "min_rows": 9000,
        "required_columns": {
            "route_action",
            "best_action",
            "cheap_quality",
            "selected_action_quality",
            "true_uplift",
            "incremental_cost",
            "latency_penalty",
            "hallucination_risk",
            "evidence_failure_risk",
            "budget_pressure",
            "oracle_policy_value",
            "oracle_best_policy_value",
            "treatment",
            "outcome",
        },
    },
    "synthetic_baidu_waimai_red_packet_8k": {
        "kind": "resume_baidu_waimai_red_packet_uplift",
        "min_rows": 8000,
        "required_columns": {
            "meal_intent_score",
            "red_packet_amount",
            "min_order_value",
            "expected_red_packet_cost",
            "incremental_gmv",
            "incremental_profit",
            "subsidy_abuse_risk",
            "uplift_type",
            "true_uplift",
            "oracle_policy_value",
            "treatment",
            "outcome",
        },
    },
    "synthetic_didi_passenger_subsidy_8k": {
        "kind": "resume_didi_passenger_subsidy_uplift",
        "min_rows": 8000,
        "required_columns": {
            "ride_intent_score",
            "passenger_subsidy_amount",
            "discount_rate",
            "demand_supply_gap",
            "request",
            "completed_order",
            "expected_subsidy_cost",
            "true_request_uplift",
            "true_completion_uplift",
            "retention_value_d7",
            "price_sensitivity_drift",
            "oracle_policy_value",
            "treatment",
            "outcome",
        },
    },
    "synthetic_didi_driver_supply_subsidy_8k": {
        "kind": "resume_didi_driver_supply_subsidy_uplift",
        "min_rows": 8000,
        "required_columns": {
            "driver_subsidy_amount",
            "incentive_type",
            "online_and_accept",
            "online_minutes",
            "accepted_orders",
            "completed_orders",
            "true_supply_uplift",
            "true_match_uplift",
            "wait_time_reduction_value",
            "driver_subsidy_cost",
            "spillover_risk",
            "interference_risk",
            "oracle_policy_value",
            "treatment",
            "outcome",
        },
    },
    "synthetic_didi_budget_allocation_5k": {
        "kind": "resume_didi_budget_allocation_uplift",
        "min_rows": 5000,
        "required_columns": {
            "city_tier",
            "city_cluster",
            "budget_pool",
            "allocated_budget",
            "budget_share",
            "c_side_share",
            "b_side_share",
            "marginal_roi",
            "incremental_orders",
            "city_incremental_profit",
            "budget_policy_value",
            "cross_market_spillover_risk",
            "true_uplift",
            "oracle_policy_value",
            "treatment",
            "outcome",
        },
    },
    "synthetic_shopee_ads_full_funnel_8k": {
        "kind": "resume_shopee_ads_full_funnel_uplift",
        "min_rows": 8000,
        "required_columns": {
            "impression",
            "click",
            "conversion",
            "bid_cpc",
            "ad_cost",
            "media_cost",
            "incremental_gmv",
            "iroas",
            "geo_holdout_cell",
            "true_impression_uplift",
            "true_click_uplift",
            "true_conversion_uplift",
            "true_uplift",
            "oracle_policy_value",
            "treatment",
            "outcome",
        },
    },
    "synthetic_spotify_inapp_message_uplift_6k": {
        "kind": "synthetic_spotify_inapp_message_uplift",
        "min_rows": 6500,
        "required_columns": {
            "baseline_listening_hours_7d",
            "message_topic",
            "channel",
            "prior_messages_14d",
            "message_fatigue",
            "message_cost",
            "fatigue_penalty",
            "optout_risk",
            "retention_value",
            "incremental_minutes",
            "true_engagement_uplift",
            "true_uplift",
            "oracle_policy_value",
            "treatment",
            "outcome",
        },
    },
    "synthetic_doordash_ghost_ads_lift_6k": {
        "kind": "synthetic_doordash_ghost_ads_lift",
        "min_rows": 6500,
        "required_columns": {
            "impression",
            "ghost_impression",
            "click",
            "conversion",
            "holdout_cell",
            "ghost_ad_cost",
            "incremental_sales",
            "sales_lift",
            "iroas",
            "true_impression_uplift",
            "true_click_uplift",
            "true_conversion_uplift",
            "true_uplift",
            "oracle_policy_value",
            "treatment",
            "outcome",
        },
    },
    "synthetic_merchant_subsidy_margin_uplift_6k": {
        "kind": "synthetic_merchant_subsidy_margin_uplift",
        "min_rows": 6500,
        "required_columns": {
            "merchant_quality",
            "merchant_margin_rate",
            "subsidy_budget_share",
            "cofund_rate",
            "promo_type",
            "merchant_subsidy_cost",
            "merchant_incremental_profit",
            "platform_margin_value",
            "cannibalization_risk",
            "marketplace_spillover_risk",
            "true_order_uplift",
            "true_uplift",
            "oracle_policy_value",
            "treatment",
            "outcome",
        },
    },
    "synthetic_ecommerce_multi_coupon_uplift_6k": {
        "kind": "synthetic_ecommerce_multi_coupon_uplift",
        "min_rows": 6200,
        "required_columns": {
            "coupon_face_value",
            "min_spend",
            "treatment_level",
            "gross_margin",
            "expected_coupon_cost",
            "arbitrage_risk",
            "true_uplift",
            "oracle_policy_value",
            "treatment",
            "outcome",
        },
    },
    "synthetic_crm_overlap_journey_uplift_6k": {
        "kind": "synthetic_crm_overlap_journey_uplift",
        "min_rows": 6400,
        "required_columns": {
            "journey_count",
            "overlap_index",
            "recent_push_count",
            "sms_count",
            "fatigue",
            "optout_uplift",
            "contact_cost",
            "long_term_value",
            "cannibalization_risk",
            "true_uplift",
            "oracle_policy_value",
            "treatment",
            "outcome",
        },
    },
    "synthetic_continuous_bid_discount_uplift_5k": {
        "kind": "synthetic_continuous_bid_discount_uplift",
        "min_rows": 5800,
        "required_columns": {
            "bid_multiplier",
            "discount_rate",
            "treatment_strength",
            "saturation",
            "media_cost",
            "discount_cost",
            "order_value",
            "true_uplift",
            "oracle_policy_value",
            "treatment",
            "outcome",
        },
    },
    "synthetic_geo_holdout_incrementality_uplift_6k": {
        "kind": "synthetic_geo_holdout_incrementality_uplift",
        "min_rows": 6100,
        "required_columns": {
            "geo_id",
            "day_index",
            "holdout_cell",
            "switchback_block",
            "planned_spend",
            "weak_overlap_risk",
            "geo_spillover_risk",
            "synthetic_control_gap",
            "incremental_revenue",
            "media_cost",
            "true_uplift",
            "oracle_policy_value",
            "treatment",
            "outcome",
        },
    },
    "synthetic_game_ops_gift_uplift_6k": {
        "kind": "synthetic_game_ops_gift_uplift",
        "min_rows": 6200,
        "required_columns": {
            "player_level",
            "gift_type",
            "gift_cost",
            "churn_risk",
            "pay_to_win_backlash_risk",
            "incremental_ltv",
            "true_uplift",
            "oracle_policy_value",
            "treatment",
            "outcome",
        },
    },
    "synthetic_fintech_credit_line_uplift_6k": {
        "kind": "synthetic_fintech_credit_line_uplift",
        "min_rows": 6200,
        "required_columns": {
            "credit_score",
            "offer_type",
            "credit_cost",
            "capital_cost",
            "default_risk_lift",
            "incremental_interest_revenue",
            "true_uplift",
            "oracle_policy_value",
            "treatment",
            "outcome",
        },
    },
    "synthetic_customer_service_escalation_uplift_6k": {
        "kind": "synthetic_customer_service_escalation_uplift",
        "min_rows": 6200,
        "required_columns": {
            "issue_complexity",
            "bot_confidence",
            "escalation_cost",
            "retention_value",
            "handle_time_penalty",
            "csat_uplift",
            "true_uplift",
            "oracle_policy_value",
            "treatment",
            "outcome",
        },
    },
    "synthetic_healthcare_followup_uplift_6k": {
        "kind": "synthetic_healthcare_followup_uplift",
        "min_rows": 6200,
        "required_columns": {
            "chronic_score",
            "prior_no_show_rate",
            "channel",
            "outreach_cost",
            "clinical_risk",
            "no_show_risk",
            "adherence_value",
            "true_uplift",
            "oracle_policy_value",
            "treatment",
            "outcome",
        },
    },
    "synthetic_saas_retention_offer_uplift_6k": {
        "kind": "synthetic_saas_retention_offer_uplift",
        "min_rows": 6200,
        "required_columns": {
            "arr",
            "offer_type",
            "churn_risk",
            "expansion_potential",
            "offer_cost",
            "discount_dependency_risk",
            "expansion_value",
            "true_uplift",
            "oracle_policy_value",
            "treatment",
            "outcome",
        },
    },
}


def _load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _safe_mean(df: pd.DataFrame, col: str) -> float | None:
    if col not in df.columns:
        return None
    value = pd.to_numeric(df[col], errors="coerce").mean()
    if pd.isna(value):
        return None
    return float(value)


def main() -> None:
    payload = _load_manifest()
    manifest_rows = {row.get("id"): row for row in payload.get("datasets", [])}
    cards = {row["dataset_id"]: row for row in scenario_cards()}
    failures: list[str] = []
    summaries = []

    for dataset_id in SCENARIO_DATASET_IDS:
        checks = DATASET_CHECKS[dataset_id]
        dataset = manifest_rows.get(dataset_id)
        if not dataset:
            failures.append(f"Missing manifest entry: {dataset_id}")
            continue
        if dataset.get("kind") != checks["kind"]:
            failures.append(f"{dataset_id} kind mismatch: {dataset.get('kind')}")
        path = ROOT / dataset["path"]
        if not path.exists():
            failures.append(f"Missing dataset file: {path}")
            continue
        df = pd.read_csv(path)
        missing = sorted(checks["required_columns"] - set(df.columns))
        if missing:
            failures.append(f"{dataset_id} missing columns: {missing}")
        feature_cols = set(dataset.get("feature_cols") or [])
        leakage = sorted(COMMON_ORACLE_COLUMNS & feature_cols)
        if leakage:
            failures.append(f"{dataset_id} oracle/policy columns should not be features: {leakage}")
        treatment_col = dataset["treatment_col"]
        outcome_col = dataset["outcome_col"]
        treatment_rate = _safe_mean(df, treatment_col)
        outcome_rate = _safe_mean(df, outcome_col)
        if len(df) < int(checks["min_rows"]):
            failures.append(f"{dataset_id} expected at least {checks['min_rows']} rows, found {len(df)}")
        if treatment_rate is not None and not 0.15 <= treatment_rate <= 0.85:
            failures.append(f"{dataset_id} treatment rate outside smoke bounds: {treatment_rate:.3f}")
        if outcome_rate is not None and not 0.002 <= outcome_rate <= 0.90:
            failures.append(f"{dataset_id} outcome rate outside smoke bounds: {outcome_rate:.3f}")
        if "true_uplift" in df.columns:
            uplift = pd.to_numeric(df["true_uplift"], errors="coerce")
            if uplift.max() <= 0.12:
                failures.append(f"{dataset_id} true_uplift max too small: {uplift.max():.3f}")
            if uplift.quantile(0.9) <= uplift.quantile(0.1):
                failures.append(f"{dataset_id} true_uplift has weak spread")
        if "oracle_policy_value" in df.columns:
            policy = pd.to_numeric(df["oracle_policy_value"], errors="coerce")
            if policy.quantile(0.9) <= policy.quantile(0.1):
                failures.append(f"{dataset_id} oracle_policy_value has weak spread")
        summaries.append(
            {
                "dataset_id": dataset_id,
                "scenario": cards.get(dataset_id, {}).get("scenario"),
                "name": dataset.get("name"),
                "kind": dataset.get("kind"),
                "path": dataset.get("path"),
                "rows": len(df),
                "features": len(dataset.get("feature_cols") or []),
                "treatment_rate": treatment_rate,
                "outcome_rate": outcome_rate,
                "true_uplift_min": float(pd.to_numeric(df.get("true_uplift"), errors="coerce").min()) if "true_uplift" in df.columns else None,
                "true_uplift_max": float(pd.to_numeric(df.get("true_uplift"), errors="coerce").max()) if "true_uplift" in df.columns else None,
                "policy_value_p10": float(pd.to_numeric(df["oracle_policy_value"], errors="coerce").quantile(0.1)) if "oracle_policy_value" in df.columns else None,
                "policy_value_p90": float(pd.to_numeric(df["oracle_policy_value"], errors="coerce").quantile(0.9)) if "oracle_policy_value" in df.columns else None,
                "recommended_models": "; ".join(dataset.get("recommended_models") or []),
            }
        )

    output = ROOT / "reports" / "industrial_scenario_datasets_smoke_latest.json"
    output.parent.mkdir(exist_ok=True)
    result = {"status": "fail" if failures else "ok", "datasets": summaries, "failures": failures}
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
