from __future__ import annotations

from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler

from deepuplift.utils.evaluate import calculate_metrics_by_treatment, plot_bins_uplift, uplift_metric

from .policy import policy_value_curve

ORACLE_UPLIFT_COLS = [
    "true_uplift",
    "true_effect",
    "ite",
    "cate",
    "tau",
    "treatment_effect",
    "mu1_minus_mu0",
    "true_impression_uplift",
    "true_click_uplift",
    "true_conversion_uplift",
    "true_uplift_d1",
    "true_uplift_d7",
    "true_uplift_d14",
    "true_uplift_d30",
]

BUSINESS_SCORE_COLS = [
    "dm_score",
    "dm_uplift",
    "delivery_score",
    "business_score",
    "campaign_score",
    "online_score",
    "sim_score",
    "expected_value",
    "expected_incremental_value",
    "predicted_incremental_value",
    "expected_incremental_profit",
    "incremental_profit",
    "incremental_gmv",
    "completed_order_value",
    "wait_time_reduction_value",
    "match_rate_value",
    "city_incremental_profit",
    "budget_policy_value",
    "net_marketplace_value",
    "quality_gain_value",
    "incremental_minutes",
    "retention_value",
    "incremental_sales",
    "merchant_incremental_profit",
    "platform_margin_value",
    "incremental_revenue",
]

FRONTIER_EVIDENCE_COLS = [
    "impression",
    "click",
    "conversion",
    "order_value",
    "media_cost",
    "contact_cost",
    "oracle_policy_value",
    "converted_d1",
    "converted_d7",
    "converted_d14",
    "converted_d30",
    "conversion_delay_days",
    "label_window_days",
    "censored",
    "gross_margin",
    "coupon_face_value",
    "expected_coupon_cost",
    "red_packet_amount",
    "min_order_value",
    "red_packet_expiry_hours",
    "expected_red_packet_cost",
    "touch_cost",
    "reorder_value_7d",
    "incremental_gmv",
    "incremental_profit",
    "channel_cost",
    "fatigue_cost",
    "subsidy_abuse_risk",
    "uplift_type",
    "gmv",
    "request",
    "completed_order",
    "passenger_subsidy_amount",
    "discount_rate",
    "expected_subsidy_cost",
    "price_sensitivity_drift",
    "retention_value_d7",
    "driver_subsidy_amount",
    "driver_subsidy_cost",
    "online_and_accept",
    "online_minutes",
    "accepted_orders",
    "completed_orders",
    "wait_time_reduction_value",
    "match_rate_value",
    "budget_pool",
    "allocated_budget",
    "budget_share",
    "c_side_share",
    "b_side_share",
    "marginal_roi",
    "city_incremental_profit",
    "budget_policy_value",
    "cross_market_spillover_risk",
    "request_value",
    "intervention_cost",
    "negative_uplift_risk",
    "segment_stability_score",
    "planned_subsidy_amount",
    "marketplace_value",
    "subsidy_cost",
    "spillover_risk",
    "interference_risk",
    "ad_cost",
    "bid_cpc",
    "incremental_gmv",
    "iroas",
    "geo_holdout_cell",
    "message_cost",
    "fatigue_penalty",
    "optout_risk",
    "engagement_value",
    "incremental_minutes",
    "retention_value",
    "ghost_impression",
    "ghost_ad_cost",
    "gross_sales",
    "incremental_sales",
    "sales_lift",
    "holdout_cell",
    "merchant_subsidy_cost",
    "merchant_incremental_profit",
    "platform_margin_value",
    "cannibalization_risk",
    "marketplace_spillover_risk",
    "incremental_cost",
    "latency_penalty",
    "user_value",
    "route_action",
    "best_action",
    "cheap_quality",
    "strong_quality",
    "rag_quality",
    "tool_quality",
    "human_quality",
    "selected_action_quality",
    "hallucination_risk",
    "evidence_failure_risk",
    "judge_noise_risk",
    "budget_pressure",
    "oracle_best_policy_value",
    "min_spend",
    "treatment_level",
    "arbitrage_risk",
    "journey_count",
    "overlap_index",
    "recent_push_count",
    "sms_count",
    "fatigue",
    "optout_uplift",
    "long_term_value",
    "bid_multiplier",
    "treatment_strength",
    "saturation",
    "discount_cost",
    "geo_id",
    "day_index",
    "week",
    "switchback_block",
    "planned_spend",
    "weak_overlap_risk",
    "geo_spillover_risk",
    "synthetic_control_gap",
    "incremental_revenue",
    "gift_cost",
    "incremental_ltv",
    "pay_to_win_backlash_risk",
    "credit_cost",
    "capital_cost",
    "default_risk_lift",
    "incremental_interest_revenue",
    "relationship_value",
    "escalation_cost",
    "handle_time_penalty",
    "csat_uplift",
    "outreach_cost",
    "adherence_value",
    "clinical_risk",
    "no_show_risk",
    "offer_cost",
    "expansion_value",
    "discount_dependency_risk",
    "arr",
    "churn_risk",
]


def _clean_float(value: Any):
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(number) or np.isinf(number):
        return None
    return number


def _series_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    clean = df.replace([np.inf, -np.inf], np.nan).reset_index()
    clean = clean.rename(columns={clean.columns[0]: "sample"})
    records = []
    for row in clean.to_dict(orient="records"):
        records.append({key: _clean_float(value) for key, value in row.items()})
    return records


def _safe_response_metrics(df: pd.DataFrame, outcome_col: str, treatment_col: str) -> Dict[str, Any]:
    try:
        metrics = calculate_metrics_by_treatment(
            df,
            outcome_col=outcome_col,
            treatment_col=treatment_col,
            threshold=0.5,
            if_plot=False,
        )
    except Exception as exc:
        return {"error": str(exc)}

    return {
        "auc_y0": _clean_float(metrics.get("auc_y0")),
        "auc_y1": _clean_float(metrics.get("auc_y1")),
        "confusion_matrix_y0": metrics.get("confusion_matrix_y0").tolist(),
        "confusion_matrix_y1": metrics.get("confusion_matrix_y1").tolist(),
    }


def _top_k_stats(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    uplift_col: str,
    fractions: Iterable[float],
) -> List[Dict[str, Any]]:
    sorted_df = df.sort_values(uplift_col, ascending=False).reset_index(drop=True)
    rows = []
    for fraction in fractions:
        n = max(1, int(len(sorted_df) * fraction))
        top = sorted_df.head(n)
        treated = top[top[treatment_col] == 1]
        control = top[top[treatment_col] == 0]
        treated_rate = treated[outcome_col].mean() if len(treated) else np.nan
        control_rate = control[outcome_col].mean() if len(control) else np.nan
        rows.append(
            {
                "top_fraction": fraction,
                "rows": int(n),
                "treated_rows": int(len(treated)),
                "control_rows": int(len(control)),
                "treated_outcome_mean": _clean_float(treated_rate),
                "control_outcome_mean": _clean_float(control_rate),
                "observed_uplift": _clean_float(treated_rate - control_rate),
                "mean_predicted_uplift": _clean_float(top[uplift_col].mean()),
            }
        )
    return rows


def oracle_top_k_recall(
    df: pd.DataFrame,
    uplift_col: str,
    fractions: Iterable[float],
    oracle_cols: Iterable[str] = ORACLE_UPLIFT_COLS,
) -> Dict[str, Any]:
    oracle_col = next((col for col in oracle_cols if col in df.columns), None)
    if oracle_col is None:
        return {}

    work = df[[uplift_col, oracle_col]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if work.empty:
        return {}

    by_model = work.sort_values(uplift_col, ascending=False).reset_index()
    by_oracle = work.sort_values(oracle_col, ascending=False).reset_index()
    rows = []
    for fraction in fractions:
        n = max(1, int(len(work) * fraction))
        model_top = by_model.head(n)
        oracle_top = by_oracle.head(n)
        overlap = len(set(model_top["index"]).intersection(set(oracle_top["index"])))
        oracle_gain = float(oracle_top[oracle_col].sum())
        model_gain = float(model_top[oracle_col].sum())
        rows.append(
            {
                "top_fraction": float(fraction),
                "rows": int(n),
                "oracle_col": oracle_col,
                "topk_recall": _clean_float(overlap / n),
                "mean_true_uplift_in_model_topk": _clean_float(model_top[oracle_col].mean()),
                "mean_true_uplift_in_oracle_topk": _clean_float(oracle_top[oracle_col].mean()),
                "oracle_gain_capture": _clean_float(model_gain / oracle_gain) if abs(oracle_gain) > 1e-12 else None,
            }
        )
    return {"oracle_col": oracle_col, "rows": rows}


def business_score_top_k_alignment(
    df: pd.DataFrame,
    uplift_col: str,
    fractions: Iterable[float],
    business_cols: Iterable[str] = BUSINESS_SCORE_COLS,
) -> Dict[str, Any]:
    business_col = next((col for col in business_cols if col in df.columns), None)
    if business_col is None:
        return {}

    work = df[[uplift_col, business_col]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if work.empty:
        return {}

    by_model = work.sort_values(uplift_col, ascending=False).reset_index()
    by_business = work.sort_values(business_col, ascending=False).reset_index()
    rows = []
    for fraction in fractions:
        n = max(1, int(len(work) * fraction))
        model_top = by_model.head(n)
        business_top = by_business.head(n)
        overlap = len(set(model_top["index"]).intersection(set(business_top["index"])))
        business_total = float(business_top[business_col].sum())
        model_total = float(model_top[business_col].sum())
        rows.append(
            {
                "top_fraction": float(fraction),
                "rows": int(n),
                "business_col": business_col,
                "topk_overlap_recall": _clean_float(overlap / n),
                "mean_business_score_in_model_topk": _clean_float(model_top[business_col].mean()),
                "mean_business_score_in_business_topk": _clean_float(business_top[business_col].mean()),
                "business_score_capture": _clean_float(model_total / business_total) if abs(business_total) > 1e-12 else None,
            }
        )
    return {"business_col": business_col, "rows": rows}


def _observed_effect_for_col(df: pd.DataFrame, treatment_col: str, value_col: str) -> Dict[str, Any]:
    work = df[[treatment_col, value_col]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if work.empty:
        return {
            "treated_rows": 0,
            "control_rows": 0,
            "treated_mean": None,
            "control_mean": None,
            "observed_uplift": None,
        }

    treated = work[work[treatment_col] == 1]
    control = work[work[treatment_col] == 0]
    treated_mean = treated[value_col].mean() if len(treated) else np.nan
    control_mean = control[value_col].mean() if len(control) else np.nan
    return {
        "treated_rows": int(len(treated)),
        "control_rows": int(len(control)),
        "treated_mean": _clean_float(treated_mean),
        "control_mean": _clean_float(control_mean),
        "observed_uplift": _clean_float(treated_mean - control_mean),
    }


def _top_sorted(df: pd.DataFrame, uplift_col: str, cols: list[str]) -> pd.DataFrame:
    keep = [col for col in [uplift_col] + cols if col in df.columns]
    if uplift_col not in keep:
        return pd.DataFrame()
    work = df[keep].replace([np.inf, -np.inf], np.nan).dropna(subset=[uplift_col]).copy()
    if work.empty:
        return work
    return work.sort_values(uplift_col, ascending=False)


def _oracle_capture_rows(
    df: pd.DataFrame,
    uplift_col: str,
    oracle_col: str,
    fractions: Iterable[float],
) -> list[dict[str, Any]]:
    if oracle_col not in df.columns:
        return []
    work = _top_sorted(df, uplift_col, [oracle_col]).dropna(subset=[oracle_col]).copy()
    if work.empty:
        return []

    by_model = work.sort_values(uplift_col, ascending=False).reset_index()
    by_oracle = work.sort_values(oracle_col, ascending=False).reset_index()
    rows = []
    for fraction in fractions:
        n = max(1, int(np.ceil(len(work) * fraction)))
        model_top = by_model.head(n)
        oracle_top = by_oracle.head(n)
        overlap = len(set(model_top["index"]).intersection(set(oracle_top["index"])))
        oracle_total = float(oracle_top[oracle_col].sum())
        model_total = float(model_top[oracle_col].sum())
        rows.append(
            {
                "top_fraction": float(fraction),
                "rows": int(n),
                "oracle_col": oracle_col,
                "topk_recall": _clean_float(overlap / n),
                "mean_oracle_in_model_topk": _clean_float(model_top[oracle_col].mean()),
                "mean_oracle_in_oracle_topk": _clean_float(oracle_top[oracle_col].mean()),
                "oracle_gain_capture": _clean_float(model_total / oracle_total) if abs(oracle_total) > 1e-12 else None,
            }
        )
    return rows


def _policy_capture_rows(
    df: pd.DataFrame,
    uplift_col: str,
    fractions: Iterable[float],
    policy_col: str = "oracle_policy_value",
) -> list[dict[str, Any]]:
    if policy_col not in df.columns:
        return []
    work = _top_sorted(df, uplift_col, [policy_col]).dropna(subset=[policy_col]).copy()
    if work.empty:
        return []

    by_model = work.sort_values(uplift_col, ascending=False).reset_index()
    by_policy = work.sort_values(policy_col, ascending=False).reset_index()
    rows = []
    for fraction in fractions:
        n = max(1, int(np.ceil(len(work) * fraction)))
        model_top = by_model.head(n)
        policy_top = by_policy.head(n)
        overlap = len(set(model_top["index"]).intersection(set(policy_top["index"])))
        policy_total = float(policy_top[policy_col].sum())
        model_total = float(model_top[policy_col].sum())
        rows.append(
            {
                "top_fraction": float(fraction),
                "rows": int(n),
                "policy_col": policy_col,
                "topk_overlap_recall": _clean_float(overlap / n),
                "mean_policy_value_in_model_topk": _clean_float(model_top[policy_col].mean()),
                "mean_policy_value_in_policy_topk": _clean_float(policy_top[policy_col].mean()),
                "policy_value_capture": _clean_float(model_total / policy_total) if abs(policy_total) > 1e-12 else None,
                "model_topk_policy_value_sum": _clean_float(model_total),
            }
        )
    return rows


def full_funnel_uplift_metrics(
    df: pd.DataFrame,
    treatment_col: str,
    uplift_col: str,
    stage_cols: Iterable[str] = ("impression", "click", "conversion"),
    fractions: Iterable[float] = (0.05, 0.1, 0.2, 0.3),
) -> Dict[str, Any]:
    stages = [col for col in stage_cols if col in df.columns]
    if not stages:
        return {}

    sorted_df = _top_sorted(df, uplift_col, [treatment_col] + stages + ["oracle_policy_value"])
    if sorted_df.empty or treatment_col not in sorted_df.columns:
        return {}

    overall_rows = []
    for stage in stages:
        effect = _observed_effect_for_col(sorted_df, treatment_col, stage)
        overall_rows.append({"stage": stage, "scope": "all_holdout", **effect})

    top_k_rows = []
    for fraction in fractions:
        n = max(1, int(np.ceil(len(sorted_df) * fraction)))
        top = sorted_df.head(n)
        for stage in stages:
            effect = _observed_effect_for_col(top, treatment_col, stage)
            top_k_rows.append(
                {
                    "top_fraction": float(fraction),
                    "rows": int(n),
                    "stage": stage,
                    **effect,
                    "mean_predicted_uplift": _clean_float(top[uplift_col].mean()),
                }
            )

    stage_oracle_map = {
        "impression": "true_impression_uplift",
        "click": "true_click_uplift",
        "conversion": "true_conversion_uplift",
    }
    oracle_rows = []
    for stage, oracle_col in stage_oracle_map.items():
        if stage in stages and oracle_col in df.columns:
            for row in _oracle_capture_rows(df, uplift_col, oracle_col, fractions):
                oracle_rows.append({"stage": stage, **row})

    payload = {
        "stage_cols": stages,
        "overall": overall_rows,
        "top_k": top_k_rows,
        "oracle_top_k": oracle_rows,
        "policy_top_k": _policy_capture_rows(df, uplift_col, fractions),
    }
    if "order_value" in df.columns:
        payload["order_value_mean"] = _clean_float(pd.to_numeric(df["order_value"], errors="coerce").mean())
    if "media_cost" in df.columns:
        payload["media_cost_mean"] = _clean_float(pd.to_numeric(df["media_cost"], errors="coerce").mean())
    return payload


def delayed_feedback_uplift_metrics(
    df: pd.DataFrame,
    treatment_col: str,
    uplift_col: str,
    fractions: Iterable[float] = (0.05, 0.1, 0.2, 0.3),
) -> Dict[str, Any]:
    window_cols = sorted(
        [col for col in df.columns if col.startswith("converted_d")],
        key=lambda col: int(col.replace("converted_d", "")) if col.replace("converted_d", "").isdigit() else 10**9,
    )
    if not window_cols:
        return {}

    sorted_df = _top_sorted(
        df,
        uplift_col,
        [treatment_col] + window_cols + ["conversion_delay_days", "label_window_days", "censored", "oracle_policy_value"],
    )
    if sorted_df.empty or treatment_col not in sorted_df.columns:
        return {}

    overall_rows = []
    for window_col in window_cols:
        effect = _observed_effect_for_col(sorted_df, treatment_col, window_col)
        window = window_col.replace("converted_d", "D")
        overall_rows.append({"window": window, "window_col": window_col, "scope": "all_holdout", **effect})

    top_k_rows = []
    for fraction in fractions:
        n = max(1, int(np.ceil(len(sorted_df) * fraction)))
        top = sorted_df.head(n)
        for window_col in window_cols:
            effect = _observed_effect_for_col(top, treatment_col, window_col)
            top_k_rows.append(
                {
                    "top_fraction": float(fraction),
                    "rows": int(n),
                    "window": window_col.replace("converted_d", "D"),
                    "window_col": window_col,
                    **effect,
                    "mean_predicted_uplift": _clean_float(top[uplift_col].mean()),
                }
            )

    oracle_rows = []
    for window_col in window_cols:
        suffix = window_col.replace("converted_d", "")
        oracle_col = f"true_uplift_d{suffix}"
        if oracle_col in df.columns:
            for row in _oracle_capture_rows(df, uplift_col, oracle_col, fractions):
                oracle_rows.append({"window": f"D{suffix}", "window_col": window_col, **row})

    summary: Dict[str, Any] = {}
    if "label_window_days" in df.columns:
        label_window = pd.to_numeric(df["label_window_days"], errors="coerce").dropna()
        if not label_window.empty:
            summary["label_window_days_min"] = _clean_float(label_window.min())
            summary["label_window_days_median"] = _clean_float(label_window.median())
            summary["label_window_days_max"] = _clean_float(label_window.max())
    if "censored" in df.columns:
        summary["censored_rate"] = _clean_float(pd.to_numeric(df["censored"], errors="coerce").mean())
    if "conversion_delay_days" in df.columns:
        delay_source = df
        if "censored" in df.columns:
            delay_source = df[pd.to_numeric(df["censored"], errors="coerce").fillna(1) == 0]
        delay = pd.to_numeric(delay_source["conversion_delay_days"], errors="coerce").dropna()
        if not delay.empty:
            summary["observed_conversion_delay_days_p50"] = _clean_float(delay.quantile(0.5))
            summary["observed_conversion_delay_days_p90"] = _clean_float(delay.quantile(0.9))
            summary["observed_conversion_delay_rows"] = int(len(delay))

    return {
        "window_cols": window_cols,
        "summary": summary,
        "overall": overall_rows,
        "top_k": top_k_rows,
        "oracle_top_k": oracle_rows,
        "policy_top_k": _policy_capture_rows(df, uplift_col, fractions),
    }


def industrial_policy_metrics(
    df: pd.DataFrame,
    uplift_col: str,
    fractions: Iterable[float] = (0.05, 0.1, 0.2, 0.3),
) -> Dict[str, Any]:
    scenario_specs = []
    columns = set(df.columns)
    if {"gross_margin", "expected_coupon_cost", "oracle_policy_value"} <= columns:
        scenario_specs.append(
            {
                "scenario": "coupon_profit",
                "value_cols": ["gross_margin", "expected_coupon_cost", "subsidy_abuse_risk", "fatigue_cost", "oracle_policy_value"],
                "risk_cols": ["subsidy_abuse_risk", "fatigue_cost"],
                "cost_col": "expected_coupon_cost",
                "policy_col": "oracle_policy_value",
            }
        )
    if {"coupon_face_value", "min_spend", "arbitrage_risk", "oracle_policy_value"} <= columns:
        scenario_specs.append(
            {
                "scenario": "ecommerce_multi_coupon",
                "value_cols": [
                    "coupon_face_value",
                    "min_spend",
                    "gross_margin",
                    "expected_coupon_cost",
                    "arbitrage_risk",
                    "oracle_policy_value",
                ],
                "risk_cols": ["arbitrage_risk"],
                "cost_col": "expected_coupon_cost",
                "policy_col": "oracle_policy_value",
            }
        )
    if {"expected_red_packet_cost", "incremental_profit", "oracle_policy_value"} <= columns:
        scenario_specs.append(
            {
                "scenario": "baidu_waimai_red_packet",
                "value_cols": [
                    "gmv",
                    "gross_margin",
                    "red_packet_amount",
                    "expected_red_packet_cost",
                    "subsidy_abuse_risk",
                    "reorder_value_7d",
                    "incremental_gmv",
                    "incremental_profit",
                    "oracle_policy_value",
                ],
                "risk_cols": ["subsidy_abuse_risk"],
                "cost_col": "expected_red_packet_cost",
                "policy_col": "oracle_policy_value",
            }
        )
    if {"expected_subsidy_cost", "completed_order_value", "oracle_policy_value"} <= columns:
        scenario_specs.append(
            {
                "scenario": "didi_passenger_subsidy",
                "value_cols": [
                    "passenger_subsidy_amount",
                    "discount_rate",
                    "expected_subsidy_cost",
                    "completed_order_value",
                    "retention_value_d7",
                    "price_sensitivity_drift",
                    "subsidy_abuse_risk",
                    "oracle_policy_value",
                ],
                "risk_cols": ["subsidy_abuse_risk", "price_sensitivity_drift"],
                "cost_col": "expected_subsidy_cost",
                "policy_col": "oracle_policy_value",
            }
        )
    if {"driver_subsidy_cost", "wait_time_reduction_value", "oracle_policy_value"} <= columns:
        scenario_specs.append(
            {
                "scenario": "didi_driver_supply_subsidy",
                "value_cols": [
                    "driver_subsidy_amount",
                    "driver_subsidy_cost",
                    "wait_time_reduction_value",
                    "match_rate_value",
                    "spillover_risk",
                    "interference_risk",
                    "oracle_policy_value",
                ],
                "risk_cols": ["spillover_risk", "interference_risk"],
                "cost_col": "driver_subsidy_cost",
                "policy_col": "oracle_policy_value",
            }
        )
    if {"allocated_budget", "city_incremental_profit", "oracle_policy_value"} <= columns:
        scenario_specs.append(
            {
                "scenario": "didi_budget_allocation",
                "value_cols": [
                    "allocated_budget",
                    "budget_share",
                    "c_side_share",
                    "b_side_share",
                    "marginal_roi",
                    "city_incremental_profit",
                    "cross_market_spillover_risk",
                    "oracle_policy_value",
                ],
                "risk_cols": ["cross_market_spillover_risk"],
                "cost_col": "allocated_budget",
                "policy_col": "oracle_policy_value",
            }
        )
    if {"ad_cost", "incremental_gmv", "iroas", "oracle_policy_value"} <= columns:
        scenario_specs.append(
            {
                "scenario": "shopee_ads_iroas",
                "value_cols": [
                    "ad_cost",
                    "media_cost",
                    "bid_cpc",
                    "incremental_gmv",
                    "iroas",
                    "order_value",
                    "oracle_policy_value",
                ],
                "risk_cols": [],
                "cost_col": "ad_cost",
                "policy_col": "oracle_policy_value",
            }
        )
    if {"message_cost", "fatigue_penalty", "optout_risk", "oracle_policy_value"} <= columns:
        scenario_specs.append(
            {
                "scenario": "spotify_inapp_message_diet",
                "value_cols": [
                    "engagement_value",
                    "incremental_minutes",
                    "message_cost",
                    "fatigue_penalty",
                    "optout_risk",
                    "retention_value",
                    "oracle_policy_value",
                ],
                "risk_cols": ["optout_risk", "fatigue_penalty"],
                "cost_col": "message_cost",
                "policy_col": "oracle_policy_value",
            }
        )
    if {"contact_cost", "optout_uplift", "cannibalization_risk", "long_term_value", "oracle_policy_value"} <= columns:
        scenario_specs.append(
            {
                "scenario": "crm_overlap_journey",
                "value_cols": [
                    "journey_count",
                    "overlap_index",
                    "contact_cost",
                    "optout_uplift",
                    "cannibalization_risk",
                    "long_term_value",
                    "oracle_policy_value",
                ],
                "risk_cols": ["optout_uplift", "cannibalization_risk"],
                "cost_col": "contact_cost",
                "policy_col": "oracle_policy_value",
            }
        )
    if {"ghost_ad_cost", "incremental_sales", "sales_lift", "oracle_policy_value"} <= columns:
        scenario_specs.append(
            {
                "scenario": "doordash_ghost_ads_lift",
                "value_cols": [
                    "ghost_ad_cost",
                    "incremental_sales",
                    "sales_lift",
                    "iroas",
                    "gross_sales",
                    "oracle_policy_value",
                ],
                "risk_cols": [],
                "cost_col": "ghost_ad_cost",
                "policy_col": "oracle_policy_value",
            }
        )
    if {"merchant_subsidy_cost", "merchant_incremental_profit", "platform_margin_value", "oracle_policy_value"} <= columns:
        scenario_specs.append(
            {
                "scenario": "merchant_subsidy_margin",
                "value_cols": [
                    "merchant_subsidy_cost",
                    "merchant_incremental_profit",
                    "platform_margin_value",
                    "cannibalization_risk",
                    "marketplace_spillover_risk",
                    "oracle_policy_value",
                ],
                "risk_cols": ["cannibalization_risk", "marketplace_spillover_risk"],
                "cost_col": "merchant_subsidy_cost",
                "policy_col": "oracle_policy_value",
            }
        )
    if {"bid_multiplier", "discount_rate", "discount_cost", "media_cost", "oracle_policy_value"} <= columns:
        scenario_specs.append(
            {
                "scenario": "continuous_bid_discount",
                "value_cols": [
                    "bid_multiplier",
                    "discount_rate",
                    "treatment_strength",
                    "saturation",
                    "order_value",
                    "media_cost",
                    "discount_cost",
                    "oracle_policy_value",
                ],
                "risk_cols": ["saturation"],
                "cost_col": "media_cost",
                "policy_col": "oracle_policy_value",
            }
        )
    if {"holdout_cell", "switchback_block", "media_cost", "incremental_revenue", "oracle_policy_value"} <= columns:
        scenario_specs.append(
            {
                "scenario": "geo_holdout_incrementality",
                "value_cols": [
                    "planned_spend",
                    "media_cost",
                    "incremental_revenue",
                    "weak_overlap_risk",
                    "geo_spillover_risk",
                    "synthetic_control_gap",
                    "oracle_policy_value",
                ],
                "risk_cols": ["weak_overlap_risk", "geo_spillover_risk"],
                "cost_col": "media_cost",
                "policy_col": "oracle_policy_value",
            }
        )
    if {"gift_cost", "incremental_ltv", "pay_to_win_backlash_risk", "oracle_policy_value"} <= columns:
        scenario_specs.append(
            {
                "scenario": "game_ops_gift",
                "value_cols": ["gift_cost", "incremental_ltv", "churn_risk", "pay_to_win_backlash_risk", "oracle_policy_value"],
                "risk_cols": ["churn_risk", "pay_to_win_backlash_risk"],
                "cost_col": "gift_cost",
                "policy_col": "oracle_policy_value",
            }
        )
    if {"credit_cost", "incremental_interest_revenue", "default_risk_lift", "capital_cost", "oracle_policy_value"} <= columns:
        scenario_specs.append(
            {
                "scenario": "fintech_credit_line",
                "value_cols": [
                    "credit_cost",
                    "capital_cost",
                    "default_risk_lift",
                    "incremental_interest_revenue",
                    "relationship_value",
                    "oracle_policy_value",
                ],
                "risk_cols": ["default_risk_lift"],
                "cost_col": "credit_cost",
                "policy_col": "oracle_policy_value",
            }
        )
    if {"escalation_cost", "retention_value", "handle_time_penalty", "csat_uplift", "oracle_policy_value"} <= columns:
        scenario_specs.append(
            {
                "scenario": "customer_service_escalation",
                "value_cols": ["escalation_cost", "retention_value", "handle_time_penalty", "csat_uplift", "oracle_policy_value"],
                "risk_cols": ["handle_time_penalty"],
                "cost_col": "escalation_cost",
                "policy_col": "oracle_policy_value",
            }
        )
    if {"outreach_cost", "adherence_value", "clinical_risk", "no_show_risk", "oracle_policy_value"} <= columns:
        scenario_specs.append(
            {
                "scenario": "healthcare_followup",
                "value_cols": ["outreach_cost", "adherence_value", "clinical_risk", "no_show_risk", "oracle_policy_value"],
                "risk_cols": ["clinical_risk", "no_show_risk"],
                "cost_col": "outreach_cost",
                "policy_col": "oracle_policy_value",
            }
        )
    if {"offer_cost", "expansion_value", "discount_dependency_risk", "churn_risk", "oracle_policy_value"} <= columns:
        scenario_specs.append(
            {
                "scenario": "saas_retention_offer",
                "value_cols": ["arr", "offer_cost", "expansion_value", "discount_dependency_risk", "churn_risk", "oracle_policy_value"],
                "risk_cols": ["discount_dependency_risk", "churn_risk"],
                "cost_col": "offer_cost",
                "policy_col": "oracle_policy_value",
            }
        )
    if {"request_value", "negative_uplift_risk", "oracle_policy_value"} <= columns:
        scenario_specs.append(
            {
                "scenario": "recommendation_intervention",
                "value_cols": ["request_value", "intervention_cost", "negative_uplift_risk", "segment_stability_score", "oracle_policy_value"],
                "risk_cols": ["negative_uplift_risk"],
                "cost_col": "intervention_cost",
                "policy_col": "oracle_policy_value",
            }
        )
    if {"marketplace_value", "subsidy_cost", "spillover_risk", "oracle_policy_value"} <= columns:
        scenario_specs.append(
            {
                "scenario": "marketplace_subsidy",
                "value_cols": ["marketplace_value", "subsidy_cost", "spillover_risk", "interference_risk", "oracle_policy_value"],
                "risk_cols": ["spillover_risk", "interference_risk"],
                "cost_col": "subsidy_cost",
                "policy_col": "oracle_policy_value",
            }
        )
    if {"route_action", "incremental_cost", "latency_penalty", "hallucination_risk", "evidence_failure_risk", "oracle_policy_value"} <= columns:
        scenario_specs.append(
            {
                "scenario": "llm_multi_action_routing",
                "value_cols": [
                    "route_action",
                    "best_action",
                    "user_value",
                    "selected_action_quality",
                    "cheap_quality",
                    "incremental_cost",
                    "latency_penalty",
                    "hallucination_risk",
                    "evidence_failure_risk",
                    "judge_noise_risk",
                    "budget_pressure",
                    "oracle_policy_value",
                    "oracle_best_policy_value",
                ],
                "risk_cols": ["latency_penalty", "hallucination_risk", "evidence_failure_risk", "judge_noise_risk", "budget_pressure"],
                "cost_col": "incremental_cost",
                "policy_col": "oracle_policy_value",
            }
        )
    elif {"incremental_cost", "latency_penalty", "oracle_policy_value"} <= columns:
        scenario_specs.append(
            {
                "scenario": "llm_routing",
                "value_cols": ["incremental_cost", "latency_penalty", "oracle_policy_value"],
                "risk_cols": ["latency_penalty"],
                "cost_col": "incremental_cost",
                "policy_col": "oracle_policy_value",
            }
        )
    if not scenario_specs:
        return {}

    rows: list[dict[str, Any]] = []
    for spec in scenario_specs:
        cols = [col for col in [uplift_col] + spec["value_cols"] if col in df.columns]
        work = df[cols].dropna(subset=[uplift_col]).copy()
        if work.empty:
            continue
        sorted_df = work.sort_values(uplift_col, ascending=False).reset_index(drop=True)
        policy_col = spec["policy_col"]
        cost_col = spec.get("cost_col")
        for fraction in fractions:
            n = max(1, int(np.ceil(len(sorted_df) * fraction)))
            top = sorted_df.head(n)
            policy_sum = float(top[policy_col].sum()) if policy_col in top.columns else np.nan
            cost_sum = float(top[cost_col].sum()) if cost_col in top.columns else np.nan
            row = {
                "scenario": spec["scenario"],
                "top_fraction": float(fraction),
                "rows": int(n),
                "mean_predicted_uplift": _clean_float(top[uplift_col].mean()),
                "policy_value_sum": _clean_float(policy_sum),
                "policy_value_mean": _clean_float(top[policy_col].mean()) if policy_col in top.columns else None,
                "cost_sum": _clean_float(cost_sum) if cost_col in top.columns else None,
                "roi_proxy": _clean_float(policy_sum / cost_sum) if cost_col in top.columns and abs(cost_sum) > 1e-12 else None,
            }
            for risk_col in spec["risk_cols"]:
                if risk_col in top.columns:
                    row[f"mean_{risk_col}"] = _clean_float(top[risk_col].mean())
            rows.append(row)

    if not rows:
        return {}
    top10 = [row for row in rows if abs(row.get("top_fraction", 0) - 0.1) < 1e-9]
    return {"scenarios": sorted({row["scenario"] for row in rows}), "top_k": rows, "top10": top10}


def frontier_metric_summary(metrics: Dict[str, Any]) -> Dict[str, Any]:
    full_funnel = metrics.get("full_funnel") or {}
    delayed_feedback = metrics.get("delayed_feedback") or {}
    industrial_policy = metrics.get("industrial_policy") or {}
    full_rows = full_funnel.get("top_k") or []
    delayed_rows = delayed_feedback.get("top_k") or []
    industrial_rows = industrial_policy.get("top_k") or []

    def pick(rows: list[dict[str, Any]], *, fraction: float, key: str, value: str) -> dict[str, Any]:
        return next(
            (
                row
                for row in rows
                if abs(float(row.get("top_fraction", -1)) - fraction) < 1e-6 and row.get(key) == value
            ),
            {},
        )

    ff_conversion = pick(full_rows, fraction=0.1, key="stage", value="conversion")
    ff_click = pick(full_rows, fraction=0.1, key="stage", value="click")
    ff_impression = pick(full_rows, fraction=0.1, key="stage", value="impression")
    delayed_d30 = pick(delayed_rows, fraction=0.1, key="window_col", value="converted_d30")
    delayed_d14 = pick(delayed_rows, fraction=0.1, key="window_col", value="converted_d14")
    delayed_d7 = pick(delayed_rows, fraction=0.1, key="window_col", value="converted_d7")
    summary = delayed_feedback.get("summary") or {}
    schemas = []
    if full_rows:
        schemas.append("full_funnel")
    if delayed_rows:
        schemas.append("delayed_feedback")
    if industrial_rows:
        schemas.extend(industrial_policy.get("scenarios") or ["industrial_policy"])
    industrial_top10 = next(
        (row for row in industrial_rows if abs(float(row.get("top_fraction", -1)) - 0.1) < 1e-6),
        {},
    )

    return {
        "frontier_metric_schema": ", ".join(schemas),
        "full_funnel_top10_impression_uplift": ff_impression.get("observed_uplift"),
        "full_funnel_top10_click_uplift": ff_click.get("observed_uplift"),
        "full_funnel_top10_conversion_uplift": ff_conversion.get("observed_uplift"),
        "delayed_d7_top10_uplift": delayed_d7.get("observed_uplift"),
        "delayed_d14_top10_uplift": delayed_d14.get("observed_uplift"),
        "delayed_d30_top10_uplift": delayed_d30.get("observed_uplift"),
        "delayed_censored_rate": summary.get("censored_rate"),
        "delayed_observed_delay_p90": summary.get("observed_conversion_delay_days_p90"),
        "industrial_top10_policy_value_sum": industrial_top10.get("policy_value_sum"),
        "industrial_top10_roi_proxy": industrial_top10.get("roi_proxy"),
    }


def uplift_calibration(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    uplift_col: str,
    bins: int = 10,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    work = df[[outcome_col, treatment_col, uplift_col]].copy()
    work["calibration_bin"] = pd.qcut(work[uplift_col], bins, duplicates="drop")
    rows = []
    errors = []
    for interval, part in work.groupby("calibration_bin", observed=True):
        treated = part[part[treatment_col] == 1]
        control = part[part[treatment_col] == 0]
        treated_rate = treated[outcome_col].mean() if len(treated) else np.nan
        control_rate = control[outcome_col].mean() if len(control) else np.nan
        observed = treated_rate - control_rate
        predicted = part[uplift_col].mean()
        error = predicted - observed
        if not np.isnan(error):
            errors.append(abs(error))
        rows.append(
            {
                "bin": str(interval),
                "rows": int(len(part)),
                "treated_rows": int(len(treated)),
                "control_rows": int(len(control)),
                "mean_predicted_uplift": _clean_float(predicted),
                "observed_uplift": _clean_float(observed),
                "calibration_error": _clean_float(error),
            }
        )
    metrics = {"calibration_mae": _clean_float(np.mean(errors)) if errors else None}
    return rows, metrics


def segment_analysis(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    uplift_col: str,
    segment_cols: List[str] | None,
    min_count: int = 50,
) -> List[Dict[str, Any]]:
    if not segment_cols:
        return []
    rows = []
    for col in segment_cols:
        if col not in df.columns:
            continue
        work = df[[col, outcome_col, treatment_col, uplift_col]].copy()
        if pd.api.types.is_numeric_dtype(work[col]) and work[col].nunique(dropna=True) > 8:
            try:
                work["segment"] = pd.qcut(work[col], 4, duplicates="drop").astype(str)
            except ValueError:
                work["segment"] = work[col].astype(str)
        else:
            top_values = work[col].astype("object").where(work[col].notna(), "__missing__").astype(str)
            if top_values.nunique() > 12:
                keep = set(top_values.value_counts().head(12).index)
                top_values = top_values.where(top_values.isin(keep), "__other__")
            work["segment"] = top_values

        for segment, part in work.groupby("segment", observed=True):
            if len(part) < min_count:
                continue
            treated = part[part[treatment_col] == 1]
            control = part[part[treatment_col] == 0]
            if len(treated) == 0 or len(control) == 0:
                continue
            observed = treated[outcome_col].mean() - control[outcome_col].mean()
            rows.append(
                {
                    "feature": col,
                    "segment": str(segment),
                    "rows": int(len(part)),
                    "treated_rows": int(len(treated)),
                    "control_rows": int(len(control)),
                    "mean_predicted_uplift": _clean_float(part[uplift_col].mean()),
                    "observed_uplift": _clean_float(observed),
                }
            )
    return sorted(rows, key=lambda row: abs(row.get("observed_uplift") or 0), reverse=True)


def distilled_segment_tree(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    uplift_col: str,
    feature_cols: List[str] | None,
    max_leaf_nodes: int = 8,
    min_leaf_fraction: float = 0.04,
    random_state: int = 42,
) -> List[Dict[str, Any]]:
    if not feature_cols:
        return []

    cols = [outcome_col, treatment_col, uplift_col] + [col for col in feature_cols if col in df.columns]
    work = df[cols].replace([np.inf, -np.inf], np.nan).dropna(subset=[outcome_col, treatment_col, uplift_col]).copy()
    feature_cols = [col for col in feature_cols if col in work.columns]
    if len(work) < 100 or not feature_cols or work[uplift_col].nunique(dropna=True) < 2:
        return []

    encoded_source: Dict[str, pd.Series] = {}
    for col in feature_cols:
        series = work[col]
        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce")
            encoded_source[col] = numeric.fillna(numeric.median() if numeric.notna().any() else 0.0)
        else:
            encoded_source[col] = series.astype("object").where(series.notna(), "__missing__").astype(str)

    x_tree = pd.get_dummies(pd.DataFrame(encoded_source), dummy_na=False)
    if x_tree.empty:
        return []

    min_samples_leaf = max(30, int(len(work) * min_leaf_fraction))
    tree = DecisionTreeRegressor(
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    tree.fit(x_tree, work[uplift_col].to_numpy())
    leaves = tree.apply(x_tree)
    feature_names = list(x_tree.columns)

    paths: Dict[int, List[str]] = {}

    def walk(node: int, conditions: List[str]) -> None:
        left = tree.tree_.children_left[node]
        right = tree.tree_.children_right[node]
        if left == right:
            paths[node] = conditions
            return
        feature_name = feature_names[tree.tree_.feature[node]]
        threshold = tree.tree_.threshold[node]
        if threshold <= 0.5 and feature_name not in feature_cols:
            left_rule = f"{feature_name} = 0"
            right_rule = f"{feature_name} = 1"
        else:
            left_rule = f"{feature_name} <= {threshold:.4g}"
            right_rule = f"{feature_name} > {threshold:.4g}"
        walk(left, conditions + [left_rule])
        walk(right, conditions + [right_rule])

    walk(0, [])

    result_rows = []
    work = work.reset_index(drop=True)
    for leaf_id in sorted(set(leaves)):
        part = work[np.asarray(leaves) == leaf_id]
        if len(part) < min_samples_leaf:
            continue
        treated = part[part[treatment_col] == 1]
        control = part[part[treatment_col] == 0]
        observed = np.nan
        if len(treated) and len(control):
            observed = treated[outcome_col].mean() - control[outcome_col].mean()
        rule = " AND ".join(paths.get(int(leaf_id), [])) or "ALL"
        result_rows.append(
            {
                "leaf": int(leaf_id),
                "rule": rule,
                "rows": int(len(part)),
                "support": _clean_float(len(part) / len(work)),
                "treated_rows": int(len(treated)),
                "control_rows": int(len(control)),
                "mean_predicted_uplift": _clean_float(part[uplift_col].mean()),
                "observed_uplift": _clean_float(observed),
                "tree_predicted_uplift": _clean_float(tree.tree_.value[int(leaf_id)][0][0]),
            }
        )

    return sorted(result_rows, key=lambda row: row.get("mean_predicted_uplift") or 0.0, reverse=True)


def bootstrap_metric_ci(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    uplift_col: str,
    bins: int = 10,
    n_bootstrap: int = 0,
    random_state: int = 42,
    conversion_value: float = 1.0,
    contact_cost: float = 0.0,
) -> Dict[str, Any]:
    if n_bootstrap <= 0:
        return {}

    rng = np.random.default_rng(random_state)
    samples: Dict[str, List[float]] = {
        "qini_score": [],
        "auuc_score": [],
        "top10_observed_uplift": [],
        "calibration_mae": [],
        "policy_best_net_value": [],
    }

    for _ in range(n_bootstrap):
        indexes = rng.integers(0, len(df), len(df))
        boot = df.iloc[indexes].reset_index(drop=True)
        if boot[treatment_col].nunique(dropna=True) < 2:
            continue
        try:
            _, qini_scores = uplift_metric(
                df=boot[[treatment_col, outcome_col, uplift_col]],
                outcome_col=outcome_col,
                treatment_col=treatment_col,
                kind="qini",
                if_plot=False,
            )
            _, auuc_scores = uplift_metric(
                df=boot[[treatment_col, outcome_col, uplift_col]],
                outcome_col=outcome_col,
                treatment_col=treatment_col,
                kind="auuc",
                if_plot=False,
            )
            top10 = _top_k_stats(boot, outcome_col, treatment_col, uplift_col, fractions=[0.1])[0]
            _, calibration_metrics = uplift_calibration(boot, outcome_col, treatment_col, uplift_col, bins=bins)
            policy = policy_value_curve(
                boot,
                uplift_col=uplift_col,
                outcome_col=outcome_col,
                treatment_col=treatment_col,
                conversion_value=conversion_value,
                contact_cost=contact_cost,
            )
        except Exception:
            continue

        best_policy = policy.get("best") or {}
        values = {
            "qini_score": qini_scores.get(uplift_col),
            "auuc_score": auuc_scores.get(uplift_col),
            "top10_observed_uplift": top10.get("observed_uplift"),
            "calibration_mae": calibration_metrics.get("calibration_mae"),
            "policy_best_net_value": best_policy.get(policy.get("ranking_col", "observed_net_value")),
        }
        for key, value in values.items():
            clean = _clean_float(value)
            if clean is not None:
                samples[key].append(clean)

    ci = {}
    for key, values in samples.items():
        if not values:
            continue
        array = np.asarray(values, dtype="float64")
        ci[key] = {
            "mean": _clean_float(np.mean(array)),
            "low": _clean_float(np.quantile(array, 0.025)),
            "high": _clean_float(np.quantile(array, 0.975)),
        }

    return {
        "n_bootstrap": int(n_bootstrap),
        "n_success": max((len(values) for values in samples.values()), default=0),
        "metrics": ci,
    }


def _metric_scores(df: pd.DataFrame, outcome_col: str, treatment_col: str, uplift_col: str) -> Dict[str, Any]:
    if df[treatment_col].nunique(dropna=True) < 2:
        return {"qini_score": None, "auuc_score": None}
    _, qini_scores = uplift_metric(
        df=df[[treatment_col, outcome_col, uplift_col]],
        outcome_col=outcome_col,
        treatment_col=treatment_col,
        kind="qini",
        if_plot=False,
    )
    _, auuc_scores = uplift_metric(
        df=df[[treatment_col, outcome_col, uplift_col]],
        outcome_col=outcome_col,
        treatment_col=treatment_col,
        kind="auuc",
        if_plot=False,
    )
    return {
        "qini_score": _clean_float(qini_scores.get(uplift_col)),
        "auuc_score": _clean_float(auuc_scores.get(uplift_col)),
    }


def _summarize_null(observed: float | None, values: List[float]) -> Dict[str, Any]:
    if observed is None or not values:
        return {
            "null_mean": None,
            "null_p95": None,
            "excess_over_p95": None,
            "p_value": None,
        }
    array = np.asarray(values, dtype="float64")
    p95 = float(np.quantile(array, 0.95))
    p_value = float((np.sum(array >= observed) + 1) / (len(array) + 1))
    return {
        "null_mean": _clean_float(np.mean(array)),
        "null_p95": _clean_float(p95),
        "excess_over_p95": _clean_float(observed - p95),
        "p_value": _clean_float(p_value),
    }


def sensitivity_refutation_checks(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    uplift_col: str,
    n_permutations: int = 0,
    random_state: int = 42,
) -> Dict[str, Any]:
    if n_permutations <= 0:
        return {}

    work = df[[treatment_col, outcome_col, uplift_col]].copy().reset_index(drop=True)
    observed = _metric_scores(work, outcome_col, treatment_col, uplift_col)
    rng = np.random.default_rng(random_state)
    nulls = {
        "random_score": {"qini_score": [], "auuc_score": []},
        "permuted_treatment": {"qini_score": [], "auuc_score": []},
        "permuted_outcome": {"qini_score": [], "auuc_score": []},
    }

    for _ in range(n_permutations):
        random_score = work.copy()
        random_score[uplift_col] = rng.permutation(random_score[uplift_col].to_numpy())
        scores = _metric_scores(random_score, outcome_col, treatment_col, uplift_col)
        for key, value in scores.items():
            clean = _clean_float(value)
            if clean is not None:
                nulls["random_score"][key].append(clean)

        permuted_treatment = work.copy()
        permuted_treatment[treatment_col] = rng.permutation(permuted_treatment[treatment_col].to_numpy())
        scores = _metric_scores(permuted_treatment, outcome_col, treatment_col, uplift_col)
        for key, value in scores.items():
            clean = _clean_float(value)
            if clean is not None:
                nulls["permuted_treatment"][key].append(clean)

        permuted_outcome = work.copy()
        permuted_outcome[outcome_col] = rng.permutation(permuted_outcome[outcome_col].to_numpy())
        scores = _metric_scores(permuted_outcome, outcome_col, treatment_col, uplift_col)
        for key, value in scores.items():
            clean = _clean_float(value)
            if clean is not None:
                nulls["permuted_outcome"][key].append(clean)

    tests = {}
    for test_name, payload in nulls.items():
        tests[test_name] = {
            "qini_score": _summarize_null(observed.get("qini_score"), payload["qini_score"]),
            "auuc_score": _summarize_null(observed.get("auuc_score"), payload["auuc_score"]),
        }

    qini_random = tests["random_score"]["qini_score"]
    qini_treatment = tests["permuted_treatment"]["qini_score"]
    pass_random = (qini_random.get("excess_over_p95") or 0) > 0
    pass_treatment = (qini_treatment.get("excess_over_p95") or 0) > 0
    if pass_random and pass_treatment:
        verdict = "pass"
        message = "Observed ranking beats random-score and permuted-treatment null checks."
    elif pass_random or pass_treatment:
        verdict = "caution"
        message = "Observed ranking beats one null check but not all; inspect calibration, Top-K and bootstrap CI before deployment."
    else:
        verdict = "fail"
        message = "Observed ranking does not beat the refutation null checks; treat this run as exploratory."

    return {
        "n_permutations": int(n_permutations),
        "observed": observed,
        "verdict": verdict,
        "message": message,
        "tests": tests,
    }


def overlap_trim_diagnostics(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    uplift_col: str,
    segment_cols: List[str] | None,
    trims: Iterable[float] = (0.01, 0.05, 0.1),
) -> Dict[str, Any]:
    if not segment_cols:
        return {}
    if df[treatment_col].nunique(dropna=True) != 2:
        return {}

    feature_cols = [col for col in segment_cols if col in df.columns]
    if not feature_cols:
        return {}

    try:
        features = pd.get_dummies(df[feature_cols], dummy_na=True)
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(features.median(numeric_only=True)).fillna(0.0)
        if features.shape[1] == 0:
            return {}
        model = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=1000))
        treatment = df[treatment_col].astype(int).to_numpy()
        model.fit(features, treatment)
        proba = model.predict_proba(features)
        class_index = list(model.named_steps["logisticregression"].classes_).index(1)
        propensity = np.clip(proba[:, class_index], 1e-4, 1 - 1e-4)
    except Exception as exc:
        return {"error": str(exc)}

    rows = []
    for trim in trims:
        mask = (propensity >= trim) & (propensity <= 1 - trim)
        trimmed = df.loc[mask].copy()
        if len(trimmed) == 0 or trimmed[treatment_col].nunique(dropna=True) < 2:
            rows.append(
                {
                    "trim": float(trim),
                    "rows": int(len(trimmed)),
                    "kept_fraction": _clean_float(len(trimmed) / len(df)),
                    "qini_score": None,
                    "auuc_score": None,
                    "top10_observed_uplift": None,
                }
            )
            continue
        scores = _metric_scores(trimmed, outcome_col, treatment_col, uplift_col)
        top10 = _top_k_stats(trimmed, outcome_col, treatment_col, uplift_col, fractions=[0.1])[0]
        rows.append(
            {
                "trim": float(trim),
                "rows": int(len(trimmed)),
                "kept_fraction": _clean_float(len(trimmed) / len(df)),
                "qini_score": scores.get("qini_score"),
                "auuc_score": scores.get("auuc_score"),
                "top10_observed_uplift": top10.get("observed_uplift"),
                "top10_mean_predicted_uplift": top10.get("mean_predicted_uplift"),
            }
        )

    weak_rate = float(np.mean((propensity < 0.05) | (propensity > 0.95)))
    return {
        "propensity": {
            "min": _clean_float(np.min(propensity)),
            "p05": _clean_float(np.quantile(propensity, 0.05)),
            "median": _clean_float(np.quantile(propensity, 0.5)),
            "p95": _clean_float(np.quantile(propensity, 0.95)),
            "max": _clean_float(np.max(propensity)),
            "weak_overlap_rate": _clean_float(weak_rate),
        },
        "rows": rows,
    }


def evaluate_uplift_predictions(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    uplift_col: str = "uplift_score",
    bins: int = 10,
    segment_cols: List[str] | None = None,
    bootstrap_samples: int = 0,
    sensitivity_samples: int = 0,
    random_state: int = 42,
    policy_contact_cost: float = 0.0,
    policy_conversion_value: float = 1.0,
) -> Dict[str, Any]:
    oracle_cols = [col for col in ORACLE_UPLIFT_COLS if col in df.columns]
    business_score_cols = [col for col in BUSINESS_SCORE_COLS if col in df.columns]
    frontier_cols = [col for col in FRONTIER_EVIDENCE_COLS if col in df.columns]
    keep_cols = (
        [treatment_col, outcome_col, uplift_col, "y0_pred", "y1_pred"]
        + [col for col in (segment_cols or []) if col in df.columns]
        + oracle_cols
        + business_score_cols
        + frontier_cols
    )
    keep_cols = [col for col in dict.fromkeys(keep_cols) if col in df.columns]
    eval_df = df[keep_cols].copy()

    qini_curve, qini_scores = uplift_metric(
        df=eval_df[[treatment_col, outcome_col, uplift_col]],
        outcome_col=outcome_col,
        treatment_col=treatment_col,
        kind="qini",
        if_plot=False,
    )
    auuc_curve, auuc_scores = uplift_metric(
        df=eval_df[[treatment_col, outcome_col, uplift_col]],
        outcome_col=outcome_col,
        treatment_col=treatment_col,
        kind="auuc",
        if_plot=False,
    )
    bins_df = plot_bins_uplift(
        eval_df,
        uplift_col=uplift_col,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        bins=bins,
        if_plot=False,
    )
    bins_df = bins_df.copy()
    bins_df["bins"] = bins_df["bins"].astype(str)

    response_metrics = _safe_response_metrics(eval_df, outcome_col, treatment_col)
    top_k = _top_k_stats(eval_df, outcome_col, treatment_col, uplift_col, fractions=[0.05, 0.1, 0.2, 0.3])
    oracle_top_k = oracle_top_k_recall(eval_df, uplift_col, fractions=[0.05, 0.1, 0.2, 0.3])
    business_alignment = business_score_top_k_alignment(eval_df, uplift_col, fractions=[0.05, 0.1, 0.2, 0.3])
    oracle_policy_top_k = _policy_capture_rows(eval_df, uplift_col, fractions=[0.05, 0.1, 0.2, 0.3])
    full_funnel = full_funnel_uplift_metrics(eval_df, treatment_col, uplift_col)
    delayed_feedback = delayed_feedback_uplift_metrics(eval_df, treatment_col, uplift_col)
    industrial_policy = industrial_policy_metrics(eval_df, uplift_col)
    calibration, calibration_metrics = uplift_calibration(eval_df, outcome_col, treatment_col, uplift_col, bins=bins)
    segments = segment_analysis(eval_df, outcome_col, treatment_col, uplift_col, segment_cols=segment_cols)
    distilled_segments = distilled_segment_tree(
        eval_df,
        outcome_col,
        treatment_col,
        uplift_col,
        feature_cols=segment_cols,
        random_state=random_state,
    )
    policy = policy_value_curve(
        eval_df,
        uplift_col=uplift_col,
        outcome_col=outcome_col,
        treatment_col=treatment_col,
        conversion_value=policy_conversion_value,
        contact_cost=policy_contact_cost,
    )
    bootstrap = bootstrap_metric_ci(
        eval_df,
        outcome_col,
        treatment_col,
        uplift_col,
        bins=bins,
        n_bootstrap=bootstrap_samples,
        random_state=random_state,
        conversion_value=policy_conversion_value,
        contact_cost=policy_contact_cost,
    )
    sensitivity = sensitivity_refutation_checks(
        eval_df,
        outcome_col,
        treatment_col,
        uplift_col,
        n_permutations=sensitivity_samples,
        random_state=random_state,
    )
    overlap_trim = overlap_trim_diagnostics(
        eval_df,
        outcome_col,
        treatment_col,
        uplift_col,
        segment_cols=segment_cols,
    )

    return {
        "metrics": {
            "qini_score": _clean_float(qini_scores.get(uplift_col)),
            "auuc_score": _clean_float(auuc_scores.get(uplift_col)),
            "random_qini_score": _clean_float(qini_scores.get("random")),
            "random_auuc_score": _clean_float(auuc_scores.get("random")),
            **calibration_metrics,
            "response": response_metrics,
            "top_k": top_k,
            "oracle_top_k": oracle_top_k,
            "business_score_alignment": business_alignment,
            "oracle_policy_top_k": oracle_policy_top_k,
            "full_funnel": full_funnel,
            "delayed_feedback": delayed_feedback,
            "industrial_policy": industrial_policy,
            "policy_best": policy.get("best"),
            "bootstrap": bootstrap,
            "sensitivity": sensitivity,
            "overlap_trim": overlap_trim,
        },
        "curves": {
            "qini": _series_to_records(qini_curve),
            "auuc": _series_to_records(auuc_curve),
            "bins": bins_df.to_dict(orient="records"),
            "calibration": calibration,
            "segments": segments,
            "distilled_segments": distilled_segments,
            "policy_value": policy.get("rows", []),
            "oracle_top_k": oracle_top_k.get("rows", []),
            "business_score_alignment": business_alignment.get("rows", []),
            "oracle_policy_top_k": oracle_policy_top_k,
            "full_funnel_top_k": full_funnel.get("top_k", []),
            "delayed_feedback_top_k": delayed_feedback.get("top_k", []),
            "industrial_policy_top_k": industrial_policy.get("top_k", []),
        },
    }
