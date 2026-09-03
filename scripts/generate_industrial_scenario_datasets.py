from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "examples" / "datasets"
MANIFEST_PATH = DATASET_DIR / "manifest.json"


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-value))


def _update_manifest(new_entries: list[dict]) -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    else:
        manifest = {"datasets": []}
    existing = {row.get("id"): row for row in manifest.get("datasets", [])}
    for entry in new_entries:
        existing[entry["id"]] = entry
    ordered = []
    seen = set()
    for row in manifest.get("datasets", []):
        dataset_id = row.get("id")
        if dataset_id in existing and dataset_id not in seen:
            ordered.append(existing[dataset_id])
            seen.add(dataset_id)
    for entry in new_entries:
        if entry["id"] not in seen:
            ordered.append(entry)
            seen.add(entry["id"])
    MANIFEST_PATH.write_text(json.dumps({"datasets": ordered}, ensure_ascii=False, indent=2), encoding="utf-8")


def write_coupon_profit_dataset() -> dict:
    rng = np.random.default_rng(202605173)
    rows = 8000
    age = rng.normal(35, 9.5, rows).clip(18, 70)
    tenure_days = rng.exponential(360, rows).clip(1, 2600)
    prior_orders = rng.poisson(2.2, rows).clip(0, 28)
    recency_days = rng.exponential(21, rows).clip(0, 160)
    price_sensitivity = rng.beta(2.8, 2.2, rows)
    brand_loyalty = rng.beta(2.5, 2.4, rows)
    margin_rate = rng.beta(4.0, 3.0, rows).clip(0.12, 0.72)
    basket_value = rng.gamma(2.2, 38, rows).clip(12, 520)
    gross_margin = (basket_value * margin_rate).clip(2, 220)
    coupon_face_value = rng.choice([3.0, 5.0, 8.0, 12.0], rows, p=[0.25, 0.37, 0.27, 0.11])
    channel = rng.choice(["app_push", "sms", "email", "feed_card"], rows, p=[0.38, 0.19, 0.21, 0.22])
    member_tier = rng.choice(["new", "silver", "gold", "platinum"], rows, p=[0.23, 0.39, 0.27, 0.11])
    category = rng.choice(["fresh", "beauty", "electronics", "fashion", "local_service"], rows, p=[0.24, 0.20, 0.15, 0.25, 0.16])

    high_loyalty = brand_loyalty > 0.72
    deal_sensitive = price_sensitivity > 0.64
    lapsed = recency_days > 38
    high_margin = gross_margin > np.quantile(gross_margin, 0.62)
    abuse_risk = np.clip(0.08 + 0.42 * deal_sensitive + 0.20 * (prior_orders <= 1) - 0.25 * high_loyalty, 0.01, 0.86)
    fatigue_score = rng.beta(1.8, 4.2, rows)
    redemption_prob = np.clip(0.10 + 0.52 * deal_sensitive + 0.12 * lapsed + 0.08 * (coupon_face_value >= 8) - 0.16 * high_loyalty, 0.02, 0.90)
    expected_coupon_cost = coupon_face_value * redemption_prob
    channel_cost = np.select(
        [channel == "sms", channel == "app_push", channel == "email"],
        [0.035, 0.006, 0.002],
        default=0.010,
    )
    fatigue_cost = np.clip(0.015 + 0.075 * fatigue_score + 0.030 * (channel == "sms"), 0.0, 0.15)

    uplift_type = np.full(rows, "lost_cause", dtype=object)
    uplift_type[deal_sensitive & lapsed & ~high_loyalty] = "persuadable"
    uplift_type[high_loyalty & (prior_orders >= 2)] = "sure_thing"
    uplift_type[(abuse_risk > 0.62) | ((fatigue_score > 0.75) & ~lapsed)] = "sleeping_dog"

    propensity = _sigmoid(
        -0.15
        + 0.95 * deal_sensitive
        + 0.48 * lapsed
        + 0.25 * high_margin
        + 0.18 * (member_tier == "gold")
        - 0.50 * high_loyalty
        - 0.35 * (abuse_risk > 0.62)
    )
    treatment = rng.binomial(1, propensity)

    base_prob = _sigmoid(
        -2.5
        + 0.50 * np.log1p(prior_orders)
        - 0.012 * recency_days
        + 1.15 * brand_loyalty
        + 0.25 * high_margin
        - 0.32 * (member_tier == "new")
    )
    true_uplift = np.clip(
        0.015
        + 0.22 * (uplift_type == "persuadable")
        + 0.055 * (deal_sensitive & high_margin)
        - 0.035 * (uplift_type == "sure_thing")
        - 0.11 * (uplift_type == "sleeping_dog")
        - 0.025 * (expected_coupon_cost > gross_margin * 0.35),
        -0.16,
        0.34,
    )
    outcome_prob = np.clip(base_prob + treatment * true_uplift, 0.002, 0.92)
    outcome = rng.binomial(1, outcome_prob)
    oracle_policy_value = true_uplift * gross_margin - expected_coupon_cost - channel_cost - fatigue_cost - 0.35 * abuse_risk
    expected_incremental_value = true_uplift * gross_margin

    df = pd.DataFrame(
        {
            "age": age.round(2),
            "tenure_days": tenure_days.round(1),
            "prior_orders": prior_orders,
            "recency_days": recency_days.round(2),
            "price_sensitivity": price_sensitivity.round(4),
            "brand_loyalty": brand_loyalty.round(4),
            "margin_rate": margin_rate.round(4),
            "basket_value": basket_value.round(4),
            "gross_margin": gross_margin.round(4),
            "coupon_face_value": coupon_face_value,
            "channel": channel,
            "member_tier": member_tier,
            "category": category,
            "fatigue_score": fatigue_score.round(4),
            "subsidy_abuse_risk": abuse_risk.round(4),
            "propensity": propensity.round(4),
            "treatment": treatment,
            "outcome": outcome,
            "true_uplift": true_uplift.round(4),
            "uplift_type": uplift_type,
            "redemption_prob": redemption_prob.round(4),
            "expected_coupon_cost": expected_coupon_cost.round(5),
            "channel_cost": channel_cost.round(5),
            "fatigue_cost": fatigue_cost.round(5),
            "expected_incremental_value": expected_incremental_value.round(5),
            "oracle_policy_value": oracle_policy_value.round(5),
        }
    )
    output = DATASET_DIR / "synthetic_coupon_profit_uplift_8k.csv"
    df.to_csv(output, index=False)
    feature_cols = [
        "age",
        "tenure_days",
        "prior_orders",
        "recency_days",
        "price_sensitivity",
        "brand_loyalty",
        "margin_rate",
        "basket_value",
        "gross_margin",
        "coupon_face_value",
        "channel",
        "member_tier",
        "category",
        "fatigue_score",
        "subsidy_abuse_risk",
    ]
    return {
        "id": "synthetic_coupon_profit_uplift_8k",
        "name": "Synthetic Coupon Profit Uplift 8k",
        "kind": "synthetic_coupon_profit_uplift",
        "path": _relative(output),
        "rows": int(len(df)),
        "description": "Synthetic coupon/subsidy uplift dataset with margin, expected coupon payout, abuse risk, fatigue and four uplift personas.",
        "source_url": "docs/UPLIFT_INDUSTRIAL_SCENARIO_LAB.md",
        "treatment_col": "treatment",
        "outcome_col": "outcome",
        "feature_cols": feature_cols,
        "task": "classification",
        "recommended_models": [
            "TLearnerLightGBM",
            "XLearnerLightGBM",
            "DRLearnerLightGBM",
            "RLearnerLightGBM",
            "CausalForest",
            "CFRNet",
            "DragonNet",
            "EFIN",
            "ContrastiveUpliftNet",
        ],
        "quick_train": {"epochs": 2, "batch_size": 256, "learning_rate": 0.001, "max_rows": 8000},
    }


def write_recommendation_intervention_dataset() -> dict:
    rng = np.random.default_rng(202605174)
    rows = 7000
    user_intent = rng.beta(2.4, 2.5, rows)
    item_affinity = rng.beta(2.5, 2.0, rows)
    creator_affinity = rng.beta(2.2, 2.8, rows)
    novelty_score = rng.beta(2.0, 3.0, rows)
    position_bias = rng.beta(2.8, 2.2, rows)
    session_depth = rng.poisson(6.0, rows).clip(1, 45)
    history_click_rate = rng.beta(2.3, 5.5, rows)
    fatigue_score = rng.beta(2.1, 4.6, rows)
    device = rng.choice(["ios", "android", "web"], rows, p=[0.37, 0.43, 0.20])
    creative_type = rng.choice(["price_badge", "social_proof", "new_arrival", "editor_pick"], rows, p=[0.30, 0.28, 0.22, 0.20])
    item_category = rng.choice(["short_video", "local_deal", "ecommerce", "content", "service"], rows, p=[0.28, 0.18, 0.24, 0.18, 0.12])

    intent_match = user_intent * item_affinity
    creative_match = np.select(
        [creative_type == "price_badge", creative_type == "social_proof", creative_type == "new_arrival"],
        [0.16 * novelty_score + 0.10 * (history_click_rate < 0.25), 0.12 * creator_affinity, 0.18 * novelty_score],
        default=0.10 * item_affinity,
    )
    high_negative_risk = (fatigue_score > 0.72) | ((session_depth > 22) & (user_intent < 0.35))
    propensity = _sigmoid(
        -0.25
        + 0.70 * intent_match
        + 0.45 * novelty_score
        + 0.30 * (creative_type == "price_badge")
        + 0.18 * (device != "web")
        - 0.42 * high_negative_risk
    )
    treatment = rng.binomial(1, propensity)
    base_impression = _sigmoid(-0.50 + 1.10 * position_bias + 0.60 * user_intent + 0.20 * (device == "ios"))
    request_uplift = np.clip(0.02 + 0.20 * intent_match + 0.16 * creative_match - 0.12 * high_negative_risk, -0.08, 0.34)
    impression_prob = np.clip(base_impression + treatment * request_uplift, 0.01, 0.98)
    impression = rng.binomial(1, impression_prob)

    base_click = _sigmoid(-2.15 + 1.35 * intent_match + 0.42 * history_click_rate + 0.25 * creator_affinity)
    click_uplift = np.clip(0.01 + 0.65 * request_uplift + 0.07 * (creative_type == "social_proof") - 0.04 * high_negative_risk, -0.07, 0.28)
    click = rng.binomial(1, np.clip(base_click + treatment * click_uplift, 0.002, 0.88) * np.maximum(impression, 0.20))

    base_conversion = _sigmoid(-3.0 + 1.55 * intent_match + 0.35 * creator_affinity + 0.45 * (item_category == "ecommerce"))
    conversion_uplift = np.clip(0.004 + 0.42 * click_uplift + 0.06 * (item_category == "local_deal") - 0.06 * high_negative_risk, -0.07, 0.24)
    conversion = rng.binomial(1, np.clip(base_conversion + treatment * conversion_uplift, 0.001, 0.72) * np.maximum(click, 0.16))
    request_value = rng.gamma(2.0, 19.0, rows).clip(3, 260)
    intervention_cost = np.clip(0.006 + 0.010 * (creative_type == "new_arrival") + 0.009 * (creative_type == "editor_pick"), 0.004, 0.04)
    negative_uplift_risk = np.clip(0.04 + 0.50 * high_negative_risk + 0.12 * fatigue_score, 0.01, 0.78)
    segment_stability_score = np.clip(1 - 0.55 * novelty_score - 0.35 * high_negative_risk + 0.15 * session_depth / 45, 0.05, 0.98)
    oracle_policy_value = conversion_uplift * request_value - intervention_cost - 0.12 * negative_uplift_risk

    df = pd.DataFrame(
        {
            "user_intent": user_intent.round(4),
            "item_affinity": item_affinity.round(4),
            "creator_affinity": creator_affinity.round(4),
            "novelty_score": novelty_score.round(4),
            "position_bias": position_bias.round(4),
            "session_depth": session_depth,
            "history_click_rate": history_click_rate.round(4),
            "fatigue_score": fatigue_score.round(4),
            "device": device,
            "creative_type": creative_type,
            "item_category": item_category,
            "propensity": propensity.round(4),
            "treatment": treatment,
            "impression": impression,
            "click": click,
            "conversion": conversion,
            "outcome": conversion,
            "true_request_uplift": request_uplift.round(4),
            "true_click_uplift": click_uplift.round(4),
            "true_conversion_uplift": conversion_uplift.round(4),
            "true_uplift": conversion_uplift.round(4),
            "request_value": request_value.round(4),
            "intervention_cost": intervention_cost.round(5),
            "negative_uplift_risk": negative_uplift_risk.round(4),
            "segment_stability_score": segment_stability_score.round(4),
            "oracle_policy_value": oracle_policy_value.round(5),
        }
    )
    output = DATASET_DIR / "synthetic_recommendation_intervention_7k.csv"
    df.to_csv(output, index=False)
    feature_cols = [
        "user_intent",
        "item_affinity",
        "creator_affinity",
        "novelty_score",
        "position_bias",
        "session_depth",
        "history_click_rate",
        "fatigue_score",
        "device",
        "creative_type",
        "item_category",
    ]
    return {
        "id": "synthetic_recommendation_intervention_7k",
        "name": "Synthetic Recommendation Intervention 7k",
        "kind": "synthetic_recommendation_intervention_uplift",
        "path": _relative(output),
        "rows": int(len(df)),
        "description": "Synthetic feed/recommendation intervention dataset with request, click and conversion uplift, creative treatment features and negative uplift risk.",
        "source_url": "docs/UPLIFT_INDUSTRIAL_SCENARIO_LAB.md",
        "treatment_col": "treatment",
        "outcome_col": "outcome",
        "feature_cols": feature_cols,
        "task": "classification",
        "recommended_models": ["EFIN", "DESCN", "DRLearnerLightGBM", "CausalForest", "ContrastiveUpliftNet", "SkLiftTwoModelsLightGBM"],
        "quick_train": {"epochs": 2, "batch_size": 256, "learning_rate": 0.001, "max_rows": 7000},
    }


def write_marketplace_subsidy_dataset() -> dict:
    rng = np.random.default_rng(202605175)
    rows = 7000
    demand_pressure = rng.beta(2.4, 2.3, rows)
    supply_pressure = rng.beta(2.1, 2.5, rows)
    driver_accept_rate = rng.beta(3.2, 2.4, rows)
    buyer_price_sensitivity = rng.beta(2.7, 2.3, rows)
    competitor_intensity = rng.beta(2.2, 3.0, rows)
    wait_time_minutes = rng.gamma(2.1, 3.2, rows).clip(1, 35)
    city_tier = rng.choice(["tier1", "tier2", "tier3", "tier4"], rows, p=[0.20, 0.36, 0.29, 0.15])
    hour_bucket = rng.choice(["morning_peak", "daytime", "evening_peak", "night"], rows, p=[0.26, 0.31, 0.28, 0.15])
    supply_gap = np.clip(demand_pressure - 0.72 * supply_pressure + 0.20 * (wait_time_minutes > 10), -0.65, 1.0)
    planned_subsidy_amount = np.select(
        [supply_gap > 0.45, supply_gap > 0.20, supply_gap > 0.0],
        [8.0, 5.0, 3.0],
        default=1.5,
    )
    marketplace_balance = np.clip(1 - np.abs(demand_pressure - supply_pressure), 0.02, 1.0)
    spillover_risk = np.clip(0.05 + 0.45 * (supply_gap > 0.35) + 0.22 * competitor_intensity + 0.12 * (city_tier == "tier1"), 0.02, 0.88)
    interference_risk = np.clip(0.04 + 0.52 * spillover_risk + 0.20 * (hour_bucket != "daytime"), 0.02, 0.92)

    propensity = _sigmoid(
        -0.20
        + 0.95 * supply_gap
        + 0.45 * buyer_price_sensitivity
        + 0.35 * competitor_intensity
        + 0.24 * (hour_bucket == "evening_peak")
        - 0.40 * (spillover_risk > 0.65)
    )
    treatment = rng.binomial(1, propensity)
    base_order = _sigmoid(
        -2.15
        + 0.75 * demand_pressure
        + 0.65 * driver_accept_rate
        - 0.045 * wait_time_minutes
        - 0.40 * competitor_intensity
        + 0.20 * (city_tier == "tier2")
    )
    true_direct_uplift = np.clip(
        0.01
        + 0.17 * buyer_price_sensitivity
        + 0.12 * (supply_gap > 0.10)
        + 0.07 * (planned_subsidy_amount >= 5)
        - 0.10 * (spillover_risk > 0.65)
        - 0.06 * (marketplace_balance < 0.30),
        -0.10,
        0.30,
    )
    outcome_prob = np.clip(base_order + treatment * true_direct_uplift, 0.002, 0.82)
    outcome = rng.binomial(1, outcome_prob)
    marketplace_value = rng.gamma(2.1, 18.0, rows).clip(5, 260) * (0.8 + 0.4 * marketplace_balance)
    subsidy_cost = planned_subsidy_amount * np.clip(0.40 + 0.40 * buyer_price_sensitivity, 0.20, 0.95)
    spillover_buffer = 0.35 * spillover_risk + 0.28 * interference_risk
    net_marketplace_value = true_direct_uplift * marketplace_value - subsidy_cost - spillover_buffer
    oracle_policy_value = net_marketplace_value

    df = pd.DataFrame(
        {
            "demand_pressure": demand_pressure.round(4),
            "supply_pressure": supply_pressure.round(4),
            "driver_accept_rate": driver_accept_rate.round(4),
            "buyer_price_sensitivity": buyer_price_sensitivity.round(4),
            "competitor_intensity": competitor_intensity.round(4),
            "wait_time_minutes": wait_time_minutes.round(3),
            "city_tier": city_tier,
            "hour_bucket": hour_bucket,
            "planned_subsidy_amount": planned_subsidy_amount,
            "marketplace_balance": marketplace_balance.round(4),
            "spillover_risk": spillover_risk.round(4),
            "interference_risk": interference_risk.round(4),
            "propensity": propensity.round(4),
            "treatment": treatment,
            "outcome": outcome,
            "true_direct_uplift": true_direct_uplift.round(4),
            "true_uplift": true_direct_uplift.round(4),
            "marketplace_value": marketplace_value.round(4),
            "subsidy_cost": subsidy_cost.round(4),
            "net_marketplace_value": net_marketplace_value.round(5),
            "oracle_policy_value": oracle_policy_value.round(5),
        }
    )
    output = DATASET_DIR / "synthetic_marketplace_subsidy_uplift_7k.csv"
    df.to_csv(output, index=False)
    feature_cols = [
        "demand_pressure",
        "supply_pressure",
        "driver_accept_rate",
        "buyer_price_sensitivity",
        "competitor_intensity",
        "wait_time_minutes",
        "city_tier",
        "hour_bucket",
        "planned_subsidy_amount",
        "marketplace_balance",
        "spillover_risk",
        "interference_risk",
    ]
    return {
        "id": "synthetic_marketplace_subsidy_uplift_7k",
        "name": "Synthetic Marketplace Subsidy Uplift 7k",
        "kind": "synthetic_marketplace_subsidy_uplift",
        "path": _relative(output),
        "rows": int(len(df)),
        "description": "Synthetic marketplace subsidy dataset with supply-demand pressure, subsidy amount, spillover/interference risk and net marketplace value.",
        "source_url": "docs/UPLIFT_INDUSTRIAL_SCENARIO_LAB.md",
        "treatment_col": "treatment",
        "outcome_col": "outcome",
        "feature_cols": feature_cols,
        "task": "classification",
        "recommended_models": ["DRLearnerLightGBM", "RLearnerLightGBM", "CausalForest", "MultiTLearnerGBM", "DoseResponseGBM"],
        "quick_train": {"epochs": 2, "batch_size": 256, "learning_rate": 0.001, "max_rows": 7000},
    }


def write_baidu_waimai_red_packet_dataset() -> dict:
    rng = np.random.default_rng(202605181)
    rows = 8000
    meal_intent_score = rng.beta(2.6, 2.2, rows)
    historical_orders_30d = rng.poisson(3.4, rows).clip(0, 36)
    days_since_last_order = rng.exponential(8.5, rows).clip(0, 90)
    avg_order_value = rng.gamma(2.0, 24.0, rows).clip(12, 220)
    merchant_distance_km = rng.gamma(1.7, 1.5, rows).clip(0.2, 12)
    delivery_eta_minutes = (18 + 5.2 * merchant_distance_km + rng.normal(0, 5, rows)).clip(8, 75)
    weather = rng.choice(["clear", "rain", "snow", "hot"], rows, p=[0.58, 0.26, 0.04, 0.12])
    meal_time = rng.choice(["breakfast", "lunch", "dinner", "late_night"], rows, p=[0.14, 0.34, 0.38, 0.14])
    cuisine_preference = rng.choice(["fast_food", "noodle", "hotpot", "coffee", "healthy"], rows, p=[0.30, 0.24, 0.16, 0.14, 0.16])
    merchant_margin_rate = rng.beta(4.2, 3.0, rows).clip(0.10, 0.62)
    red_packet_amount = rng.choice([3.0, 5.0, 8.0, 12.0], rows, p=[0.32, 0.36, 0.23, 0.09])
    min_order_value = red_packet_amount * rng.choice([3.0, 4.0, 5.0], rows, p=[0.35, 0.42, 0.23])
    red_packet_expiry_hours = rng.choice([2, 6, 24, 72], rows, p=[0.24, 0.30, 0.32, 0.14])
    channel = rng.choice(["app_push", "sms", "feed_popup", "home_banner"], rows, p=[0.38, 0.16, 0.26, 0.20])
    price_sensitivity = rng.beta(3.0, 2.1, rows)

    high_intent = meal_intent_score > 0.72
    lapsed = days_since_last_order > 14
    short_eta = delivery_eta_minutes < 28
    rainy_meal = (weather == "rain") & np.isin(meal_time, ["lunch", "dinner"])
    subsidy_abuse_risk = np.clip(
        0.05
        + 0.34 * price_sensitivity
        + 0.18 * (historical_orders_30d <= 1)
        + 0.10 * (red_packet_amount >= 8)
        - 0.18 * (historical_orders_30d >= 6),
        0.01,
        0.84,
    )
    redemption_prob = np.clip(
        0.08
        + 0.44 * price_sensitivity
        + 0.10 * high_intent
        + 0.10 * (red_packet_expiry_hours <= 6)
        + 0.08 * rainy_meal
        - 0.12 * (red_packet_amount < 5),
        0.02,
        0.88,
    )
    expected_red_packet_cost = red_packet_amount * redemption_prob
    gross_margin = avg_order_value * merchant_margin_rate
    touch_cost = np.select([channel == "sms", channel == "app_push"], [0.035, 0.006], default=0.010)
    uplift_type = np.full(rows, "lost_cause", dtype=object)
    uplift_type[(price_sensitivity > 0.62) & lapsed & short_eta] = "persuadable"
    uplift_type[high_intent & (historical_orders_30d >= 4)] = "sure_thing"
    uplift_type[(subsidy_abuse_risk > 0.60) | ((delivery_eta_minutes > 45) & (meal_intent_score < 0.45))] = "sleeping_dog"

    propensity = _sigmoid(
        -0.22
        + 0.70 * price_sensitivity
        + 0.42 * lapsed
        + 0.25 * rainy_meal
        + 0.18 * (meal_time == "dinner")
        - 0.35 * (uplift_type == "sure_thing")
        - 0.40 * (subsidy_abuse_risk > 0.68)
    )
    treatment = rng.binomial(1, propensity)
    base_order_prob = _sigmoid(
        -2.75
        + 1.28 * meal_intent_score
        + 0.42 * np.log1p(historical_orders_30d)
        - 0.014 * days_since_last_order
        - 0.026 * delivery_eta_minutes
        + 0.28 * rainy_meal
        + 0.18 * (cuisine_preference == "fast_food")
    )
    true_uplift = np.clip(
        0.012
        + 0.21 * (uplift_type == "persuadable")
        + 0.045 * (rainy_meal & short_eta)
        + 0.035 * (red_packet_amount >= 8)
        - 0.032 * (uplift_type == "sure_thing")
        - 0.120 * (uplift_type == "sleeping_dog")
        - 0.022 * (expected_red_packet_cost > gross_margin * 0.45),
        -0.15,
        0.34,
    )
    placed_order_prob = np.clip(base_order_prob + treatment * true_uplift, 0.002, 0.90)
    placed_order = rng.binomial(1, placed_order_prob)
    gmv = np.where(placed_order == 1, avg_order_value * rng.normal(1.03, 0.13, rows).clip(0.7, 1.55), 0.0)
    reorder_value_7d = np.clip(true_uplift, 0, None) * avg_order_value * 0.30
    incremental_gmv = true_uplift * avg_order_value
    incremental_profit = true_uplift * gross_margin + reorder_value_7d - expected_red_packet_cost - touch_cost - 0.24 * subsidy_abuse_risk
    oracle_policy_value = incremental_profit

    df = pd.DataFrame(
        {
            "meal_intent_score": meal_intent_score.round(4),
            "historical_orders_30d": historical_orders_30d,
            "days_since_last_order": days_since_last_order.round(3),
            "avg_order_value": avg_order_value.round(4),
            "merchant_distance_km": merchant_distance_km.round(3),
            "delivery_eta_minutes": delivery_eta_minutes.round(3),
            "weather": weather,
            "meal_time": meal_time,
            "cuisine_preference": cuisine_preference,
            "merchant_margin_rate": merchant_margin_rate.round(4),
            "red_packet_amount": red_packet_amount,
            "min_order_value": min_order_value.round(2),
            "red_packet_expiry_hours": red_packet_expiry_hours,
            "channel": channel,
            "price_sensitivity": price_sensitivity.round(4),
            "subsidy_abuse_risk": subsidy_abuse_risk.round(4),
            "propensity": propensity.round(4),
            "treatment": treatment,
            "outcome": placed_order,
            "placed_order": placed_order,
            "gmv": gmv.round(4),
            "gross_margin": gross_margin.round(4),
            "true_uplift": true_uplift.round(4),
            "uplift_type": uplift_type,
            "redemption_prob": redemption_prob.round(4),
            "expected_red_packet_cost": expected_red_packet_cost.round(5),
            "touch_cost": touch_cost.round(5),
            "reorder_value_7d": reorder_value_7d.round(5),
            "incremental_gmv": incremental_gmv.round(5),
            "incremental_profit": incremental_profit.round(5),
            "expected_incremental_profit": incremental_profit.round(5),
            "oracle_policy_value": oracle_policy_value.round(5),
        }
    )
    output = DATASET_DIR / "synthetic_baidu_waimai_red_packet_8k.csv"
    df.to_csv(output, index=False)
    feature_cols = [
        "meal_intent_score",
        "historical_orders_30d",
        "days_since_last_order",
        "avg_order_value",
        "merchant_distance_km",
        "delivery_eta_minutes",
        "weather",
        "meal_time",
        "cuisine_preference",
        "merchant_margin_rate",
        "red_packet_amount",
        "min_order_value",
        "red_packet_expiry_hours",
        "channel",
        "price_sensitivity",
        "subsidy_abuse_risk",
    ]
    return {
        "id": "synthetic_baidu_waimai_red_packet_8k",
        "name": "Baidu Waimai Red Packet Uplift 8k",
        "kind": "resume_baidu_waimai_red_packet_uplift",
        "path": _relative(output),
        "rows": int(len(df)),
        "description": "Resume-style synthetic Baidu Waimai red-packet dataset with hunger intent, coupon cost, GMV, margin, abuse risk and reorder value.",
        "source_url": "docs/UPLIFT_INDUSTRIAL_SCENARIO_LAB.md",
        "treatment_col": "treatment",
        "outcome_col": "outcome",
        "feature_cols": feature_cols,
        "task": "classification",
        "recommended_models": ["DRLearnerLightGBM", "XLearnerLightGBM", "TLearnerLightGBM", "CausalForest", "SkLiftUpliftRandomForest", "EFIN", "ContrastiveUpliftNet"],
        "quick_train": {"epochs": 2, "batch_size": 256, "learning_rate": 0.001, "max_rows": 8000},
    }


def write_didi_passenger_subsidy_dataset() -> dict:
    rng = np.random.default_rng(202605182)
    rows = 8000
    ride_intent_score = rng.beta(2.7, 2.2, rows)
    historical_rides_30d = rng.poisson(5.0, rows).clip(0, 55)
    days_since_last_ride = rng.exponential(5.5, rows).clip(0, 80)
    fare_estimate = rng.gamma(2.4, 16.0, rows).clip(12, 260)
    eta_minutes = rng.gamma(2.1, 2.6, rows).clip(1, 32)
    distance_km = rng.gamma(2.0, 3.0, rows).clip(1, 42)
    city_tier = rng.choice(["tier1", "tier2", "tier3", "tier4"], rows, p=[0.24, 0.35, 0.27, 0.14])
    weather = rng.choice(["clear", "rain", "storm", "hot"], rows, p=[0.62, 0.24, 0.04, 0.10])
    hour_bucket = rng.choice(["morning_peak", "daytime", "evening_peak", "night"], rows, p=[0.27, 0.28, 0.31, 0.14])
    price_sensitivity = rng.beta(2.9, 2.2, rows)
    demand_supply_gap = rng.normal(0.08, 0.42, rows).clip(-0.8, 1.1)
    passenger_subsidy_amount = np.select(
        [fare_estimate > 95, fare_estimate > 55, fare_estimate > 30],
        [15.0, 10.0, 6.0],
        default=3.0,
    )
    discount_rate = np.clip(passenger_subsidy_amount / fare_estimate, 0.02, 0.65)
    coupon_type = rng.choice(["cash_coupon", "percentage_discount", "rush_hour_coupon"], rows, p=[0.44, 0.28, 0.28])
    competition_intensity = rng.beta(2.2, 2.8, rows)

    peak_bad_gap = demand_supply_gap > 0.35
    lapsed = days_since_last_ride > 12
    high_intent = ride_intent_score > 0.70
    price_sensitivity_drift = np.clip(0.02 + 0.12 * price_sensitivity + 0.08 * (discount_rate > 0.25), 0.0, 0.28)
    subsidy_abuse_risk = np.clip(0.04 + 0.24 * price_sensitivity + 0.13 * (historical_rides_30d <= 1) + 0.10 * (coupon_type == "cash_coupon"), 0.01, 0.70)
    expected_subsidy_cost = passenger_subsidy_amount * np.clip(0.16 + 0.26 * price_sensitivity + 0.07 * high_intent, 0.10, 0.62)
    platform_take_rate = np.clip(0.18 + 0.04 * (city_tier == "tier1") + rng.normal(0, 0.01, rows), 0.12, 0.28)
    propensity = _sigmoid(
        -0.18
        + 0.62 * price_sensitivity
        + 0.42 * lapsed
        + 0.32 * competition_intensity
        + 0.24 * (hour_bucket == "evening_peak")
        - 0.28 * peak_bad_gap
    )
    treatment = rng.binomial(1, propensity)
    base_request = _sigmoid(
        -2.35
        + 1.35 * ride_intent_score
        + 0.34 * np.log1p(historical_rides_30d)
        - 0.022 * days_since_last_ride
        - 0.030 * eta_minutes
        - 0.45 * competition_intensity
        + 0.18 * (weather == "rain")
    )
    true_request_uplift = np.clip(
        0.012
        + 0.18 * price_sensitivity
        + 0.12 * lapsed
        + 0.05 * (discount_rate > 0.20)
        - 0.09 * peak_bad_gap
        - 0.05 * (subsidy_abuse_risk > 0.55),
        -0.10,
        0.31,
    )
    request = rng.binomial(1, np.clip(base_request + treatment * true_request_uplift, 0.002, 0.90))
    base_completion = _sigmoid(-1.20 + 0.62 * request + 0.55 * ride_intent_score - 0.038 * eta_minutes - 0.30 * peak_bad_gap)
    true_completion_uplift = np.clip(true_request_uplift * 0.72 + 0.035 * (demand_supply_gap < 0.15) - 0.035 * peak_bad_gap, -0.10, 0.25)
    completed_order = rng.binomial(1, np.clip(base_completion + treatment * true_completion_uplift, 0.002, 0.94))
    gmv = np.where(completed_order == 1, fare_estimate * rng.normal(1.0, 0.10, rows).clip(0.75, 1.35), 0.0)
    retention_value_d7 = np.clip(true_completion_uplift, 0, None) * fare_estimate * platform_take_rate * 0.85
    completed_order_value = true_completion_uplift * fare_estimate * (platform_take_rate + 0.10)
    oracle_policy_value = completed_order_value + retention_value_d7 - expected_subsidy_cost - 0.30 * subsidy_abuse_risk - price_sensitivity_drift * fare_estimate * 0.025

    df = pd.DataFrame(
        {
            "ride_intent_score": ride_intent_score.round(4),
            "historical_rides_30d": historical_rides_30d,
            "days_since_last_ride": days_since_last_ride.round(3),
            "fare_estimate": fare_estimate.round(4),
            "eta_minutes": eta_minutes.round(3),
            "distance_km": distance_km.round(3),
            "city_tier": city_tier,
            "weather": weather,
            "hour_bucket": hour_bucket,
            "price_sensitivity": price_sensitivity.round(4),
            "demand_supply_gap": demand_supply_gap.round(4),
            "passenger_subsidy_amount": passenger_subsidy_amount,
            "discount_rate": discount_rate.round(4),
            "coupon_type": coupon_type,
            "competition_intensity": competition_intensity.round(4),
            "subsidy_abuse_risk": subsidy_abuse_risk.round(4),
            "propensity": propensity.round(4),
            "treatment": treatment,
            "request": request,
            "completed_order": completed_order,
            "outcome": completed_order,
            "gmv": gmv.round(4),
            "platform_take_rate": platform_take_rate.round(4),
            "expected_subsidy_cost": expected_subsidy_cost.round(5),
            "price_sensitivity_drift": price_sensitivity_drift.round(5),
            "true_request_uplift": true_request_uplift.round(4),
            "true_completion_uplift": true_completion_uplift.round(4),
            "true_uplift": true_completion_uplift.round(4),
            "completed_order_value": completed_order_value.round(5),
            "retention_value_d7": retention_value_d7.round(5),
            "oracle_policy_value": oracle_policy_value.round(5),
        }
    )
    output = DATASET_DIR / "synthetic_didi_passenger_subsidy_8k.csv"
    df.to_csv(output, index=False)
    feature_cols = [
        "ride_intent_score",
        "historical_rides_30d",
        "days_since_last_ride",
        "fare_estimate",
        "eta_minutes",
        "distance_km",
        "city_tier",
        "weather",
        "hour_bucket",
        "price_sensitivity",
        "demand_supply_gap",
        "passenger_subsidy_amount",
        "discount_rate",
        "coupon_type",
        "competition_intensity",
        "subsidy_abuse_risk",
    ]
    return {
        "id": "synthetic_didi_passenger_subsidy_8k",
        "name": "DiDi Passenger Subsidy Uplift 8k",
        "kind": "resume_didi_passenger_subsidy_uplift",
        "path": _relative(output),
        "rows": int(len(df)),
        "description": "Resume-style synthetic passenger subsidy dataset with ride intent, ETA, demand-supply gap, completion uplift, retention and subsidy ROI.",
        "source_url": "docs/UPLIFT_INDUSTRIAL_SCENARIO_LAB.md",
        "treatment_col": "treatment",
        "outcome_col": "outcome",
        "feature_cols": feature_cols,
        "task": "classification",
        "recommended_models": ["DRLearnerLightGBM", "RLearnerLightGBM", "CausalForest", "DoseResponseGBM", "MultiTLearnerGBM", "DelayedFeedbackUplift"],
        "quick_train": {"epochs": 2, "batch_size": 256, "learning_rate": 0.001, "max_rows": 8000},
    }


def write_didi_driver_supply_subsidy_dataset() -> dict:
    rng = np.random.default_rng(202605183)
    rows = 8000
    driver_tenure_days = rng.exponential(520, rows).clip(5, 4200)
    historical_online_hours_7d = rng.gamma(2.2, 8.0, rows).clip(0.5, 86)
    accept_rate = rng.beta(4.0, 2.4, rows)
    completion_rate = rng.beta(5.0, 1.8, rows)
    city_tier = rng.choice(["tier1", "tier2", "tier3", "tier4"], rows, p=[0.22, 0.34, 0.29, 0.15])
    zone_type = rng.choice(["airport", "business", "residential", "railway", "suburb"], rows, p=[0.14, 0.27, 0.31, 0.12, 0.16])
    hour_bucket = rng.choice(["morning_peak", "daytime", "evening_peak", "night"], rows, p=[0.26, 0.30, 0.30, 0.14])
    weather = rng.choice(["clear", "rain", "storm", "hot"], rows, p=[0.60, 0.25, 0.04, 0.11])
    supply_pressure = rng.beta(2.1, 2.6, rows)
    demand_pressure = rng.beta(2.7, 2.1, rows)
    expected_income_gap = rng.normal(0.12, 0.38, rows).clip(-0.7, 1.2)
    driver_subsidy_amount = np.select(
        [expected_income_gap > 0.55, expected_income_gap > 0.25, expected_income_gap > 0.0],
        [30.0, 18.0, 10.0],
        default=5.0,
    )
    incentive_type = rng.choice(["quest", "hot_zone_bonus", "guarantee", "step_bonus"], rows, p=[0.30, 0.32, 0.18, 0.20])
    driver_fatigue_score = rng.beta(2.0, 3.6, rows)
    supply_gap = np.clip(demand_pressure - supply_pressure + 0.16 * (weather != "clear") + 0.12 * (hour_bucket != "daytime"), -0.75, 1.25)
    spillover_risk = np.clip(0.05 + 0.38 * (zone_type == "business") + 0.25 * (supply_gap > 0.55) + 0.16 * (city_tier == "tier1"), 0.02, 0.85)
    interference_risk = np.clip(0.04 + 0.48 * spillover_risk + 0.12 * (incentive_type == "hot_zone_bonus"), 0.02, 0.90)
    propensity = _sigmoid(
        -0.12
        + 0.92 * supply_gap
        + 0.48 * expected_income_gap
        + 0.22 * (incentive_type == "hot_zone_bonus")
        - 0.34 * (driver_fatigue_score > 0.72)
        - 0.28 * (spillover_risk > 0.70)
    )
    treatment = rng.binomial(1, propensity)
    base_online = _sigmoid(
        -1.85
        + 0.030 * historical_online_hours_7d
        + 1.10 * accept_rate
        + 0.40 * completion_rate
        + 0.55 * demand_pressure
        - 0.48 * driver_fatigue_score
    )
    true_supply_uplift = np.clip(
        0.010
        + 0.17 * supply_gap
        + 0.11 * expected_income_gap
        + 0.05 * (driver_subsidy_amount >= 18)
        - 0.08 * (driver_fatigue_score > 0.72)
        - 0.06 * (spillover_risk > 0.72),
        -0.11,
        0.32,
    )
    online_and_accept = rng.binomial(1, np.clip(base_online + treatment * true_supply_uplift, 0.002, 0.92))
    true_match_uplift = np.clip(true_supply_uplift * 0.75 + 0.05 * (zone_type == "airport") - 0.045 * interference_risk, -0.10, 0.27)
    online_minutes = np.where(online_and_accept == 1, rng.gamma(2.3, 35.0, rows).clip(15, 360), 0.0)
    accepted_orders = np.where(online_and_accept == 1, rng.poisson((online_minutes / 50.0) * accept_rate * (1 + demand_pressure)), 0)
    completed_orders = rng.binomial(np.maximum(accepted_orders, 0), completion_rate.clip(0.01, 0.98))
    wait_time_reduction_value = np.clip(true_match_uplift, 0, None) * (34.0 + 70.0 * demand_pressure)
    match_rate_value = true_match_uplift * (42.0 + 58.0 * supply_gap.clip(0, None))
    driver_subsidy_cost = driver_subsidy_amount * np.clip(0.22 + 0.22 * online_and_accept + 0.08 * (incentive_type == "quest"), 0.12, 0.68)
    oracle_policy_value = wait_time_reduction_value + match_rate_value - driver_subsidy_cost - 0.70 * spillover_risk - 0.55 * interference_risk

    df = pd.DataFrame(
        {
            "driver_tenure_days": driver_tenure_days.round(1),
            "historical_online_hours_7d": historical_online_hours_7d.round(3),
            "accept_rate": accept_rate.round(4),
            "completion_rate": completion_rate.round(4),
            "city_tier": city_tier,
            "zone_type": zone_type,
            "hour_bucket": hour_bucket,
            "weather": weather,
            "supply_pressure": supply_pressure.round(4),
            "demand_pressure": demand_pressure.round(4),
            "expected_income_gap": expected_income_gap.round(4),
            "driver_subsidy_amount": driver_subsidy_amount,
            "incentive_type": incentive_type,
            "driver_fatigue_score": driver_fatigue_score.round(4),
            "spillover_risk": spillover_risk.round(4),
            "interference_risk": interference_risk.round(4),
            "propensity": propensity.round(4),
            "treatment": treatment,
            "outcome": online_and_accept,
            "online_and_accept": online_and_accept,
            "online_minutes": online_minutes.round(3),
            "accepted_orders": accepted_orders,
            "completed_orders": completed_orders,
            "true_supply_uplift": true_supply_uplift.round(4),
            "true_match_uplift": true_match_uplift.round(4),
            "true_uplift": true_match_uplift.round(4),
            "wait_time_reduction_value": wait_time_reduction_value.round(5),
            "match_rate_value": match_rate_value.round(5),
            "driver_subsidy_cost": driver_subsidy_cost.round(5),
            "oracle_policy_value": oracle_policy_value.round(5),
        }
    )
    output = DATASET_DIR / "synthetic_didi_driver_supply_subsidy_8k.csv"
    df.to_csv(output, index=False)
    feature_cols = [
        "driver_tenure_days",
        "historical_online_hours_7d",
        "accept_rate",
        "completion_rate",
        "city_tier",
        "zone_type",
        "hour_bucket",
        "weather",
        "supply_pressure",
        "demand_pressure",
        "expected_income_gap",
        "driver_subsidy_amount",
        "incentive_type",
        "driver_fatigue_score",
        "spillover_risk",
        "interference_risk",
    ]
    return {
        "id": "synthetic_didi_driver_supply_subsidy_8k",
        "name": "DiDi Driver Supply Subsidy Uplift 8k",
        "kind": "resume_didi_driver_supply_subsidy_uplift",
        "path": _relative(output),
        "rows": int(len(df)),
        "description": "Resume-style synthetic driver supply subsidy dataset with supply pressure, incentive type, online/accept outcome, wait-time value and spillover risk.",
        "source_url": "docs/UPLIFT_INDUSTRIAL_SCENARIO_LAB.md",
        "treatment_col": "treatment",
        "outcome_col": "outcome",
        "feature_cols": feature_cols,
        "task": "classification",
        "recommended_models": ["DRLearnerLightGBM", "RLearnerLightGBM", "CausalForest", "MultiTLearnerGBM", "DoseResponseGBM", "MarketplaceSubsidyPolicy"],
        "quick_train": {"epochs": 2, "batch_size": 256, "learning_rate": 0.001, "max_rows": 8000},
    }


def write_didi_budget_allocation_dataset() -> dict:
    rng = np.random.default_rng(202605184)
    rows = 5000
    city_tier = rng.choice(["tier1", "tier2", "tier3", "tier4"], rows, p=[0.20, 0.34, 0.31, 0.15])
    city_cluster = rng.choice(["north", "south", "east", "west", "central"], rows)
    hour_bucket = rng.choice(["morning_peak", "daytime", "evening_peak", "night"], rows, p=[0.26, 0.30, 0.30, 0.14])
    baseline_orders = rng.gamma(2.4, 180.0, rows).clip(30, 1800)
    demand_supply_gap = rng.normal(0.10, 0.38, rows).clip(-0.75, 1.10)
    historical_roi = rng.gamma(2.2, 0.55, rows).clip(0.15, 5.5)
    competition_intensity = rng.beta(2.2, 2.8, rows)
    weather_risk = rng.beta(1.8, 5.5, rows)
    passenger_elasticity = rng.beta(2.7, 2.3, rows)
    driver_elasticity = rng.beta(2.3, 2.6, rows)
    budget_pool = rng.choice([5000.0, 8000.0, 12000.0, 20000.0], rows, p=[0.28, 0.35, 0.25, 0.12])
    budget_share = np.clip(0.04 + 0.18 * demand_supply_gap.clip(0, None) + 0.10 * historical_roi / 5.5 + rng.normal(0, 0.035, rows), 0.01, 0.42)
    allocated_budget = budget_pool * budget_share
    c_side_share = np.clip(0.45 + 0.25 * passenger_elasticity - 0.18 * driver_elasticity + rng.normal(0, 0.05, rows), 0.15, 0.85)
    b_side_share = 1.0 - c_side_share
    high_budget = allocated_budget > np.quantile(allocated_budget, 0.55)
    cross_market_spillover_risk = np.clip(0.04 + 0.33 * (city_tier == "tier1") + 0.22 * competition_intensity + 0.18 * (demand_supply_gap > 0.50), 0.02, 0.82)
    propensity = _sigmoid(-0.15 + 0.72 * historical_roi + 0.68 * demand_supply_gap + 0.28 * competition_intensity - 0.34 * cross_market_spillover_risk)
    treatment = rng.binomial(1, propensity)
    baseline_growth_prob = _sigmoid(
        -1.45
        + 0.36 * historical_roi
        + 0.58 * demand_supply_gap
        + 0.22 * (hour_bucket != "daytime")
        - 0.32 * competition_intensity
        - 0.20 * weather_risk
    )
    marginal_roi = np.clip(
        0.20
        + 1.30 * historical_roi / (1.0 + 2.8 * budget_share)
        + 0.85 * demand_supply_gap.clip(0, None)
        + 0.28 * (0.55 * passenger_elasticity + 0.45 * driver_elasticity)
        - 0.70 * cross_market_spillover_risk,
        -0.50,
        4.50,
    )
    true_uplift = np.clip(0.018 + 0.11 * (marginal_roi > 1.0) + 0.08 * demand_supply_gap.clip(0, None) - 0.06 * (cross_market_spillover_risk > 0.65), -0.08, 0.26)
    outcome = rng.binomial(1, np.clip(baseline_growth_prob + treatment * true_uplift, 0.002, 0.88))
    incremental_orders = true_uplift * baseline_orders
    city_incremental_profit = incremental_orders * 3.4 * marginal_roi - allocated_budget * 0.18 - 260.0 * cross_market_spillover_risk
    budget_policy_value = city_incremental_profit
    oracle_policy_value = budget_policy_value

    df = pd.DataFrame(
        {
            "city_tier": city_tier,
            "city_cluster": city_cluster,
            "hour_bucket": hour_bucket,
            "baseline_orders": baseline_orders.round(3),
            "demand_supply_gap": demand_supply_gap.round(4),
            "historical_roi": historical_roi.round(4),
            "competition_intensity": competition_intensity.round(4),
            "weather_risk": weather_risk.round(4),
            "passenger_elasticity": passenger_elasticity.round(4),
            "driver_elasticity": driver_elasticity.round(4),
            "budget_pool": budget_pool,
            "allocated_budget": allocated_budget.round(4),
            "budget_share": budget_share.round(4),
            "c_side_share": c_side_share.round(4),
            "b_side_share": b_side_share.round(4),
            "high_budget": high_budget.astype(int),
            "cross_market_spillover_risk": cross_market_spillover_risk.round(4),
            "propensity": propensity.round(4),
            "treatment": treatment,
            "outcome": outcome,
            "marginal_roi": marginal_roi.round(5),
            "incremental_orders": incremental_orders.round(5),
            "city_incremental_profit": city_incremental_profit.round(5),
            "budget_policy_value": budget_policy_value.round(5),
            "true_uplift": true_uplift.round(4),
            "oracle_policy_value": oracle_policy_value.round(5),
        }
    )
    output = DATASET_DIR / "synthetic_didi_budget_allocation_5k.csv"
    df.to_csv(output, index=False)
    feature_cols = [
        "city_tier",
        "city_cluster",
        "hour_bucket",
        "baseline_orders",
        "demand_supply_gap",
        "historical_roi",
        "competition_intensity",
        "weather_risk",
        "passenger_elasticity",
        "driver_elasticity",
        "budget_pool",
        "allocated_budget",
        "budget_share",
        "c_side_share",
        "b_side_share",
        "high_budget",
        "cross_market_spillover_risk",
    ]
    return {
        "id": "synthetic_didi_budget_allocation_5k",
        "name": "DiDi City-Time Budget Allocation 5k",
        "kind": "resume_didi_budget_allocation_uplift",
        "path": _relative(output),
        "rows": int(len(df)),
        "description": "Resume-style city/time budget allocation dataset with C/B-side budget split, marginal ROI, incremental orders, budget policy value and spillover guardrail.",
        "source_url": "docs/UPLIFT_INDUSTRIAL_SCENARIO_LAB.md",
        "treatment_col": "treatment",
        "outcome_col": "outcome",
        "feature_cols": feature_cols,
        "task": "classification",
        "recommended_models": ["DRLearnerLightGBM", "RLearnerLightGBM", "CausalForest", "DoseResponseGBM", "BudgetConstrainedPolicyLearner"],
        "quick_train": {"epochs": 2, "batch_size": 256, "learning_rate": 0.001, "max_rows": 5000},
    }


def write_shopee_ads_full_funnel_dataset() -> dict:
    rng = np.random.default_rng(202605185)
    rows = 8000
    user_purchase_intent = rng.beta(2.5, 2.3, rows)
    category_affinity = rng.beta(2.7, 2.2, rows)
    seller_quality_score = rng.beta(3.0, 2.4, rows)
    price_sensitivity = rng.beta(2.8, 2.3, rows)
    prior_clicks_7d = rng.poisson(2.4, rows).clip(0, 30)
    prior_orders_30d = rng.poisson(1.7, rows).clip(0, 20)
    ad_slot_quality = rng.beta(2.4, 2.5, rows)
    bid_cpc = rng.gamma(2.0, 0.12, rows).clip(0.02, 1.8)
    campaign_objective = rng.choice(["traffic", "conversion", "gmv", "new_buyer"], rows, p=[0.28, 0.33, 0.24, 0.15])
    creative_type = rng.choice(["discount_badge", "free_shipping", "live_shop", "new_arrival"], rows, p=[0.34, 0.26, 0.18, 0.22])
    market = rng.choice(["id", "th", "vn", "ph", "sg"], rows, p=[0.30, 0.24, 0.22, 0.18, 0.06])
    device = rng.choice(["ios", "android", "web"], rows, p=[0.28, 0.56, 0.16])
    geo_holdout_cell = rng.choice(["holdout_A", "holdout_B", "eligible"], rows, p=[0.08, 0.07, 0.85])

    user_item_match = user_purchase_intent * category_affinity
    creative_match = np.select(
        [creative_type == "discount_badge", creative_type == "free_shipping", creative_type == "live_shop"],
        [0.16 * price_sensitivity, 0.14 * price_sensitivity + 0.04 * seller_quality_score, 0.12 * seller_quality_score],
        default=0.10 * category_affinity,
    )
    propensity = _sigmoid(
        -0.20
        + 0.75 * user_item_match
        + 0.48 * ad_slot_quality
        + 0.34 * bid_cpc
        + 0.20 * (campaign_objective == "conversion")
        - 0.50 * (geo_holdout_cell != "eligible")
    )
    treatment = rng.binomial(1, propensity)
    base_impression = _sigmoid(-0.30 + 1.15 * ad_slot_quality + 0.46 * seller_quality_score + 0.20 * (device != "web"))
    true_impression_uplift = np.clip(0.02 + 0.16 * ad_slot_quality + 0.06 * (bid_cpc > 0.30) - 0.05 * (geo_holdout_cell != "eligible"), -0.03, 0.28)
    impression = rng.binomial(1, np.clip(base_impression + treatment * true_impression_uplift, 0.01, 0.99))
    base_click = _sigmoid(-2.25 + 1.55 * user_item_match + 0.42 * np.log1p(prior_clicks_7d) + 0.35 * creative_match)
    true_click_uplift = np.clip(0.008 + 0.55 * true_impression_uplift + 0.10 * creative_match + 0.05 * (campaign_objective == "traffic"), -0.04, 0.25)
    click = rng.binomial(1, np.clip(base_click + treatment * true_click_uplift, 0.002, 0.82) * np.maximum(impression, 0.20))
    base_conversion = _sigmoid(-3.15 + 1.75 * user_item_match + 0.52 * seller_quality_score + 0.32 * np.log1p(prior_orders_30d))
    true_conversion_uplift = np.clip(0.004 + 0.44 * true_click_uplift + 0.05 * (campaign_objective != "traffic") + 0.04 * (creative_type == "free_shipping"), -0.04, 0.22)
    conversion = rng.binomial(1, np.clip(base_conversion + treatment * true_conversion_uplift, 0.001, 0.70) * np.maximum(click, 0.14))
    order_value = rng.gamma(2.2, 18.0, rows).clip(5, 360)
    gmv = np.where(conversion == 1, order_value * rng.normal(1.0, 0.12, rows).clip(0.75, 1.55), 0.0)
    media_cost = np.where(impression == 1, bid_cpc * np.clip(0.22 + 0.76 * click + rng.normal(0, 0.04, rows), 0.05, 1.25), 0.0)
    ad_cost = media_cost
    incremental_gmv = true_conversion_uplift * order_value
    margin_value = incremental_gmv * np.clip(0.10 + 0.06 * seller_quality_score, 0.06, 0.22)
    oracle_policy_value = margin_value - ad_cost - 0.02 * (campaign_objective == "traffic")
    iroas = np.divide(incremental_gmv, ad_cost, out=np.zeros_like(incremental_gmv), where=ad_cost > 1e-6)

    df = pd.DataFrame(
        {
            "user_purchase_intent": user_purchase_intent.round(4),
            "category_affinity": category_affinity.round(4),
            "seller_quality_score": seller_quality_score.round(4),
            "price_sensitivity": price_sensitivity.round(4),
            "prior_clicks_7d": prior_clicks_7d,
            "prior_orders_30d": prior_orders_30d,
            "ad_slot_quality": ad_slot_quality.round(4),
            "bid_cpc": bid_cpc.round(5),
            "campaign_objective": campaign_objective,
            "creative_type": creative_type,
            "market": market,
            "device": device,
            "geo_holdout_cell": geo_holdout_cell,
            "propensity": propensity.round(4),
            "treatment": treatment,
            "impression": impression,
            "click": click,
            "conversion": conversion,
            "outcome": conversion,
            "order_value": order_value.round(4),
            "gmv": gmv.round(4),
            "media_cost": media_cost.round(5),
            "ad_cost": ad_cost.round(5),
            "true_impression_uplift": true_impression_uplift.round(4),
            "true_click_uplift": true_click_uplift.round(4),
            "true_conversion_uplift": true_conversion_uplift.round(4),
            "true_uplift": true_conversion_uplift.round(4),
            "incremental_gmv": incremental_gmv.round(5),
            "iroas": pd.Series(iroas).replace([np.inf, -np.inf], np.nan).fillna(0).round(5),
            "oracle_policy_value": oracle_policy_value.round(5),
        }
    )
    output = DATASET_DIR / "synthetic_shopee_ads_full_funnel_8k.csv"
    df.to_csv(output, index=False)
    feature_cols = [
        "user_purchase_intent",
        "category_affinity",
        "seller_quality_score",
        "price_sensitivity",
        "prior_clicks_7d",
        "prior_orders_30d",
        "ad_slot_quality",
        "bid_cpc",
        "campaign_objective",
        "creative_type",
        "market",
        "device",
        "geo_holdout_cell",
    ]
    return {
        "id": "synthetic_shopee_ads_full_funnel_8k",
        "name": "Shopee Ads Full-Funnel iROAS Uplift 8k",
        "kind": "resume_shopee_ads_full_funnel_uplift",
        "path": _relative(output),
        "rows": int(len(df)),
        "description": "Resume-style Shopee ads dataset with impression-click-conversion funnel, bid cost, incremental GMV, iROAS and geo/audience holdout cells.",
        "source_url": "docs/UPLIFT_INDUSTRIAL_SCENARIO_LAB.md",
        "treatment_col": "treatment",
        "outcome_col": "outcome",
        "feature_cols": feature_cols,
        "task": "classification",
        "recommended_models": ["DRLearnerLightGBM", "RLearnerLightGBM", "TLearnerLightGBM", "CausalForest", "ECUP", "DESCN", "EFIN"],
        "quick_train": {"epochs": 2, "batch_size": 256, "learning_rate": 0.001, "max_rows": 8000},
    }


def write_spotify_inapp_message_dataset() -> dict:
    rng = np.random.default_rng(202605186)
    rows = 6500
    baseline_listening_hours_7d = rng.gamma(2.1, 2.8, rows).clip(0.1, 45)
    skip_rate = rng.beta(2.2, 4.2, rows)
    discovery_affinity = rng.beta(2.5, 2.3, rows)
    playlist_depth = rng.poisson(7.0, rows).clip(0, 60)
    subscriber_tenure_days = rng.exponential(420, rows).clip(1, 3600)
    churn_risk = rng.beta(2.3, 3.0, rows)
    message_fatigue = rng.beta(2.0, 4.0, rows)
    prior_messages_14d = rng.poisson(2.0, rows).clip(0, 18)
    plan_type = rng.choice(["free", "student", "individual", "family"], rows, p=[0.46, 0.10, 0.32, 0.12])
    device = rng.choice(["ios", "android", "desktop", "web"], rows, p=[0.34, 0.42, 0.16, 0.08])
    country_tier = rng.choice(["tier1", "tier2", "emerging"], rows, p=[0.36, 0.38, 0.26])
    message_topic = rng.choice(["new_release", "playlist_recap", "premium_trial", "concert_alert"], rows, p=[0.32, 0.26, 0.24, 0.18])
    channel = rng.choice(["in_app_card", "push", "email"], rows, p=[0.58, 0.30, 0.12])

    high_music_intent = (baseline_listening_hours_7d > 6) & (skip_rate < 0.42)
    content_match = np.clip(
        0.18 * discovery_affinity
        + 0.10 * (message_topic == "new_release")
        + 0.09 * (message_topic == "playlist_recap")
        + 0.07 * (message_topic == "concert_alert")
        - 0.09 * (message_topic == "premium_trial") * (plan_type != "free"),
        -0.06,
        0.32,
    )
    optout_risk = np.clip(0.03 + 0.28 * message_fatigue + 0.07 * prior_messages_14d + 0.10 * (channel == "push"), 0.01, 0.78)
    fatigue_penalty = np.clip(0.006 + 0.036 * message_fatigue + 0.006 * prior_messages_14d + 0.020 * (channel == "push"), 0.002, 0.18)
    message_cost = np.select([channel == "push", channel == "email"], [0.006, 0.002], default=0.003)
    propensity = _sigmoid(
        -0.10
        + 0.62 * discovery_affinity
        + 0.42 * churn_risk
        + 0.28 * (plan_type == "free")
        + 0.18 * (message_topic == "premium_trial")
        - 0.55 * optout_risk
    )
    treatment = rng.binomial(1, propensity)
    base_engagement = _sigmoid(
        -1.75
        + 0.18 * np.log1p(baseline_listening_hours_7d)
        + 0.70 * discovery_affinity
        + 0.18 * np.log1p(playlist_depth)
        - 0.88 * churn_risk
        - 0.52 * skip_rate
    )
    true_engagement_uplift = np.clip(
        0.006
        + 0.16 * content_match
        + 0.09 * ((churn_risk > 0.45) & high_music_intent)
        + 0.06 * ((plan_type == "free") & (message_topic == "premium_trial"))
        - 0.10 * (optout_risk > 0.55)
        - 0.05 * (prior_messages_14d > 5),
        -0.09,
        0.24,
    )
    engaged = rng.binomial(1, np.clip(base_engagement + treatment * true_engagement_uplift, 0.002, 0.88))
    engagement_value = np.clip(0.25 + 0.75 * discovery_affinity + 0.30 * churn_risk + 0.12 * (plan_type == "free"), 0.08, 1.6)
    incremental_minutes = true_engagement_uplift * (38 + 44 * discovery_affinity + 25 * high_music_intent)
    retention_value = np.clip(true_engagement_uplift, 0, None) * engagement_value * 2.4
    oracle_policy_value = true_engagement_uplift * engagement_value + retention_value - message_cost - fatigue_penalty - 0.16 * optout_risk

    df = pd.DataFrame(
        {
            "baseline_listening_hours_7d": baseline_listening_hours_7d.round(4),
            "skip_rate": skip_rate.round(4),
            "discovery_affinity": discovery_affinity.round(4),
            "playlist_depth": playlist_depth,
            "subscriber_tenure_days": subscriber_tenure_days.round(1),
            "churn_risk": churn_risk.round(4),
            "message_fatigue": message_fatigue.round(4),
            "prior_messages_14d": prior_messages_14d,
            "plan_type": plan_type,
            "device": device,
            "country_tier": country_tier,
            "message_topic": message_topic,
            "channel": channel,
            "propensity": propensity.round(4),
            "treatment": treatment,
            "outcome": engaged,
            "engaged": engaged,
            "true_engagement_uplift": true_engagement_uplift.round(4),
            "true_uplift": true_engagement_uplift.round(4),
            "engagement_value": engagement_value.round(5),
            "incremental_minutes": incremental_minutes.round(5),
            "message_cost": pd.Series(message_cost).round(5),
            "fatigue_penalty": fatigue_penalty.round(5),
            "optout_risk": optout_risk.round(4),
            "retention_value": retention_value.round(5),
            "oracle_policy_value": oracle_policy_value.round(5),
        }
    )
    output = DATASET_DIR / "synthetic_spotify_inapp_message_uplift_6k.csv"
    df.to_csv(output, index=False)
    feature_cols = [
        "baseline_listening_hours_7d",
        "skip_rate",
        "discovery_affinity",
        "playlist_depth",
        "subscriber_tenure_days",
        "churn_risk",
        "message_fatigue",
        "prior_messages_14d",
        "plan_type",
        "device",
        "country_tier",
        "message_topic",
        "channel",
    ]
    return {
        "id": "synthetic_spotify_inapp_message_uplift_6k",
        "name": "Spotify In-App Message Diet Uplift 6.5k",
        "kind": "synthetic_spotify_inapp_message_uplift",
        "path": _relative(output),
        "rows": int(len(df)),
        "description": "Synthetic in-app messaging uplift dataset inspired by message-diet incrementality: engagement uplift minus fatigue and opt-out risk.",
        "source_url": "docs/UPLIFT_INDUSTRIAL_SCENARIO_LAB.md",
        "treatment_col": "treatment",
        "outcome_col": "outcome",
        "feature_cols": feature_cols,
        "task": "classification",
        "recommended_models": ["DRLearnerLightGBM", "RLearnerLightGBM", "PAVCalibratedDRLearnerLightGBM", "CausalForest", "CFRNet", "DragonNet"],
        "quick_train": {"epochs": 2, "batch_size": 256, "learning_rate": 0.001, "max_rows": 6500},
    }


def write_doordash_ghost_ads_dataset() -> dict:
    rng = np.random.default_rng(202605187)
    rows = 6500
    eater_purchase_intent = rng.beta(2.6, 2.4, rows)
    category_affinity = rng.beta(2.4, 2.2, rows)
    merchant_brand_loyalty = rng.beta(2.7, 2.6, rows)
    distance_miles = rng.gamma(1.8, 1.2, rows).clip(0.2, 12)
    ad_slot_quality = rng.beta(2.7, 2.2, rows)
    bid_cpc = rng.gamma(2.0, 0.18, rows).clip(0.03, 2.2)
    basket_value = rng.gamma(2.0, 22.0, rows).clip(8, 260)
    margin_rate = rng.beta(3.8, 3.2, rows).clip(0.08, 0.58)
    prior_orders_30d = rng.poisson(2.4, rows).clip(0, 30)
    cuisine = rng.choice(["pizza", "burger", "asian", "grocery", "coffee"], rows, p=[0.22, 0.24, 0.24, 0.18, 0.12])
    market_density = rng.choice(["dense", "mid", "sparse"], rows, p=[0.44, 0.38, 0.18])
    holdout_cell = rng.choice(["eligible", "ghost_holdout", "geo_holdout"], rows, p=[0.78, 0.14, 0.08])
    auction_pressure = rng.beta(2.1, 2.8, rows)

    ad_relevance = eater_purchase_intent * category_affinity * (0.75 + 0.25 * merchant_brand_loyalty)
    propensity = _sigmoid(
        -0.22
        + 0.82 * ad_relevance
        + 0.54 * ad_slot_quality
        + 0.32 * bid_cpc
        + 0.20 * auction_pressure
        - 0.92 * (holdout_cell != "eligible")
    )
    treatment = rng.binomial(1, propensity)
    base_impression = _sigmoid(-0.35 + 1.20 * ad_slot_quality + 0.45 * auction_pressure + 0.18 * (market_density == "dense"))
    impression_uplift = np.clip(0.018 + 0.15 * ad_slot_quality + 0.07 * (bid_cpc > 0.35) - 0.07 * (holdout_cell != "eligible"), -0.04, 0.26)
    impression = rng.binomial(1, np.clip(base_impression + treatment * impression_uplift, 0.01, 0.98))
    base_click = _sigmoid(-2.1 + 1.35 * ad_relevance + 0.32 * np.log1p(prior_orders_30d) - 0.06 * distance_miles)
    click_uplift = np.clip(0.012 + 0.50 * impression_uplift + 0.08 * category_affinity - 0.025 * (distance_miles > 5), -0.04, 0.23)
    click = rng.binomial(1, np.clip(base_click + treatment * click_uplift, 0.002, 0.82) * np.maximum(impression, 0.18))
    base_conversion = _sigmoid(-2.9 + 1.55 * ad_relevance + 0.30 * merchant_brand_loyalty - 0.08 * distance_miles)
    sales_lift = np.clip(0.004 + 0.45 * click_uplift + 0.05 * (cuisine == "grocery") + 0.035 * (market_density == "dense"), -0.035, 0.20)
    conversion = rng.binomial(1, np.clip(base_conversion + treatment * sales_lift, 0.001, 0.72) * np.maximum(click, 0.14))
    ghost_impression = rng.binomial(1, np.clip(base_impression + 0.25 * impression_uplift, 0.01, 0.95))
    gross_sales = np.where(conversion == 1, basket_value * rng.normal(1.0, 0.12, rows).clip(0.75, 1.45), 0.0)
    ghost_ad_cost = np.where(impression == 1, bid_cpc * np.clip(0.24 + 0.70 * click + 0.10 * auction_pressure, 0.06, 1.2), 0.0)
    incremental_sales = sales_lift * basket_value
    margin_value = incremental_sales * margin_rate
    iroas = np.divide(incremental_sales, ghost_ad_cost, out=np.zeros_like(incremental_sales), where=ghost_ad_cost > 1e-6)
    oracle_policy_value = margin_value - ghost_ad_cost - 0.025 * (holdout_cell != "eligible")

    df = pd.DataFrame(
        {
            "eater_purchase_intent": eater_purchase_intent.round(4),
            "category_affinity": category_affinity.round(4),
            "merchant_brand_loyalty": merchant_brand_loyalty.round(4),
            "distance_miles": distance_miles.round(3),
            "ad_slot_quality": ad_slot_quality.round(4),
            "bid_cpc": bid_cpc.round(5),
            "basket_value": basket_value.round(4),
            "margin_rate": margin_rate.round(4),
            "prior_orders_30d": prior_orders_30d,
            "cuisine": cuisine,
            "market_density": market_density,
            "holdout_cell": holdout_cell,
            "auction_pressure": auction_pressure.round(4),
            "propensity": propensity.round(4),
            "treatment": treatment,
            "impression": impression,
            "ghost_impression": ghost_impression,
            "click": click,
            "conversion": conversion,
            "outcome": conversion,
            "gross_sales": gross_sales.round(4),
            "ghost_ad_cost": ghost_ad_cost.round(5),
            "incremental_sales": incremental_sales.round(5),
            "sales_lift": sales_lift.round(4),
            "true_impression_uplift": impression_uplift.round(4),
            "true_click_uplift": click_uplift.round(4),
            "true_conversion_uplift": sales_lift.round(4),
            "true_uplift": sales_lift.round(4),
            "iroas": pd.Series(iroas).replace([np.inf, -np.inf], np.nan).fillna(0).round(5),
            "oracle_policy_value": oracle_policy_value.round(5),
        }
    )
    output = DATASET_DIR / "synthetic_doordash_ghost_ads_lift_6k.csv"
    df.to_csv(output, index=False)
    feature_cols = [
        "eater_purchase_intent",
        "category_affinity",
        "merchant_brand_loyalty",
        "distance_miles",
        "ad_slot_quality",
        "bid_cpc",
        "basket_value",
        "margin_rate",
        "prior_orders_30d",
        "cuisine",
        "market_density",
        "holdout_cell",
        "auction_pressure",
    ]
    return {
        "id": "synthetic_doordash_ghost_ads_lift_6k",
        "name": "DoorDash Ghost Ads Sales Lift 6.5k",
        "kind": "synthetic_doordash_ghost_ads_lift",
        "path": _relative(output),
        "rows": int(len(df)),
        "description": "Synthetic ads incrementality dataset with ghost-ad holdout, sales lift, iROAS and ad-cost policy value.",
        "source_url": "docs/UPLIFT_INDUSTRIAL_SCENARIO_LAB.md",
        "treatment_col": "treatment",
        "outcome_col": "outcome",
        "feature_cols": feature_cols,
        "task": "classification",
        "recommended_models": ["DRLearnerLightGBM", "RLearnerLightGBM", "TLearnerLightGBM", "CausalForest", "ECUP", "PAVCalibratedDRLearnerLightGBM"],
        "quick_train": {"epochs": 2, "batch_size": 256, "learning_rate": 0.001, "max_rows": 6500},
    }


def write_merchant_subsidy_dataset() -> dict:
    rng = np.random.default_rng(202605188)
    rows = 6500
    merchant_quality = rng.beta(3.0, 2.2, rows)
    merchant_margin_rate = rng.beta(3.8, 3.0, rows).clip(0.08, 0.62)
    historical_orders_30d = rng.gamma(2.1, 38.0, rows).clip(1, 650)
    prep_time_minutes = rng.gamma(2.0, 8.0, rows).clip(5, 80)
    avg_ticket = rng.gamma(2.2, 28.0, rows).clip(10, 320)
    demand_pressure = rng.beta(2.5, 2.2, rows)
    commission_rate = rng.beta(4.0, 5.0, rows).clip(0.08, 0.36)
    merchant_churn_risk = rng.beta(2.2, 3.0, rows)
    subsidy_budget_share = rng.beta(2.0, 5.0, rows).clip(0.01, 0.45)
    cofund_rate = rng.beta(2.5, 3.0, rows).clip(0.05, 0.80)
    cuisine = rng.choice(["fast_food", "local", "coffee", "grocery", "premium"], rows, p=[0.28, 0.28, 0.14, 0.18, 0.12])
    city_tier = rng.choice(["tier1", "tier2", "tier3"], rows, p=[0.34, 0.42, 0.24])
    promo_type = rng.choice(["delivery_fee_waiver", "merchant_coupon", "ranking_boost", "cofund_bundle"], rows, p=[0.28, 0.32, 0.22, 0.18])

    merchant_fit = merchant_quality * demand_pressure * (0.7 + 0.3 * merchant_margin_rate)
    cannibalization_risk = np.clip(0.06 + 0.40 * merchant_quality + 0.18 * (historical_orders_30d > 180) - 0.20 * merchant_churn_risk, 0.02, 0.82)
    marketplace_spillover_risk = np.clip(0.04 + 0.24 * (city_tier == "tier1") + 0.20 * (promo_type == "ranking_boost") + 0.10 * demand_pressure, 0.02, 0.74)
    merchant_subsidy_cost = avg_ticket * subsidy_budget_share * (1 - cofund_rate)
    propensity = _sigmoid(
        -0.16
        + 0.72 * demand_pressure
        + 0.50 * merchant_churn_risk
        + 0.32 * (promo_type == "cofund_bundle")
        + 0.18 * (city_tier == "tier2")
        - 0.42 * cannibalization_risk
    )
    treatment = rng.binomial(1, propensity)
    base_order = _sigmoid(
        -2.15
        + 0.72 * merchant_quality
        + 0.54 * demand_pressure
        + 0.20 * np.log1p(historical_orders_30d)
        - 0.025 * prep_time_minutes
        - 0.32 * merchant_churn_risk
    )
    true_order_uplift = np.clip(
        0.010
        + 0.18 * merchant_churn_risk
        + 0.12 * merchant_fit
        + 0.065 * (promo_type == "cofund_bundle")
        - 0.08 * cannibalization_risk
        - 0.04 * (prep_time_minutes > 45),
        -0.10,
        0.30,
    )
    outcome = rng.binomial(1, np.clip(base_order + treatment * true_order_uplift, 0.002, 0.90))
    incremental_orders = true_order_uplift * historical_orders_30d * 0.20
    platform_margin_value = incremental_orders * avg_ticket * commission_rate
    merchant_incremental_profit = incremental_orders * avg_ticket * merchant_margin_rate - merchant_subsidy_cost
    oracle_policy_value = platform_margin_value + 0.35 * merchant_incremental_profit - merchant_subsidy_cost - 0.50 * cannibalization_risk - 0.45 * marketplace_spillover_risk

    df = pd.DataFrame(
        {
            "merchant_quality": merchant_quality.round(4),
            "merchant_margin_rate": merchant_margin_rate.round(4),
            "historical_orders_30d": historical_orders_30d.round(3),
            "prep_time_minutes": prep_time_minutes.round(3),
            "avg_ticket": avg_ticket.round(4),
            "demand_pressure": demand_pressure.round(4),
            "commission_rate": commission_rate.round(4),
            "merchant_churn_risk": merchant_churn_risk.round(4),
            "subsidy_budget_share": subsidy_budget_share.round(4),
            "cofund_rate": cofund_rate.round(4),
            "cuisine": cuisine,
            "city_tier": city_tier,
            "promo_type": promo_type,
            "cannibalization_risk": cannibalization_risk.round(4),
            "marketplace_spillover_risk": marketplace_spillover_risk.round(4),
            "merchant_subsidy_cost": merchant_subsidy_cost.round(5),
            "propensity": propensity.round(4),
            "treatment": treatment,
            "outcome": outcome,
            "true_order_uplift": true_order_uplift.round(4),
            "true_uplift": true_order_uplift.round(4),
            "incremental_orders": incremental_orders.round(5),
            "platform_margin_value": platform_margin_value.round(5),
            "merchant_incremental_profit": merchant_incremental_profit.round(5),
            "oracle_policy_value": oracle_policy_value.round(5),
        }
    )
    output = DATASET_DIR / "synthetic_merchant_subsidy_margin_uplift_6k.csv"
    df.to_csv(output, index=False)
    feature_cols = [
        "merchant_quality",
        "merchant_margin_rate",
        "historical_orders_30d",
        "prep_time_minutes",
        "avg_ticket",
        "demand_pressure",
        "commission_rate",
        "merchant_churn_risk",
        "subsidy_budget_share",
        "cofund_rate",
        "cuisine",
        "city_tier",
        "promo_type",
        "cannibalization_risk",
        "marketplace_spillover_risk",
    ]
    return {
        "id": "synthetic_merchant_subsidy_margin_uplift_6k",
        "name": "Merchant Subsidy Margin Uplift 6.5k",
        "kind": "synthetic_merchant_subsidy_margin_uplift",
        "path": _relative(output),
        "rows": int(len(df)),
        "description": "Synthetic merchant-side subsidy/cofund dataset with merchant margin, cannibalization, churn risk and marketplace spillover guardrails.",
        "source_url": "docs/UPLIFT_INDUSTRIAL_SCENARIO_LAB.md",
        "treatment_col": "treatment",
        "outcome_col": "outcome",
        "feature_cols": feature_cols,
        "task": "classification",
        "recommended_models": ["DRLearnerLightGBM", "RLearnerLightGBM", "CausalForest", "MultiTLearnerGBM", "DoseResponseGBM", "UTBoostGBM"],
        "quick_train": {"epochs": 2, "batch_size": 256, "learning_rate": 0.001, "max_rows": 6500},
    }


def write_ecommerce_multi_coupon_dataset() -> dict:
    rng = np.random.default_rng(2026051781)
    rows = 6200
    user_value = rng.gamma(2.1, 42, rows).clip(8, 520)
    margin_rate = rng.beta(3.8, 3.0, rows).clip(0.08, 0.68)
    price_sensitivity = rng.beta(2.8, 2.2, rows)
    loyalty = rng.beta(2.5, 2.7, rows)
    recency_days = rng.exponential(26, rows).clip(0, 180)
    prior_coupon_uses = rng.poisson(1.4, rows).clip(0, 18)
    cart_value = rng.gamma(2.2, 31, rows).clip(10, 380)
    coupon_face_value = rng.choice([0.0, 5.0, 10.0, 20.0, 30.0], rows, p=[0.22, 0.28, 0.24, 0.18, 0.08])
    min_spend = rng.choice([39.0, 79.0, 129.0, 199.0], rows, p=[0.30, 0.36, 0.22, 0.12])
    channel = rng.choice(["app_push", "feed_card", "seller_message", "email"], rows, p=[0.35, 0.28, 0.22, 0.15])
    category = rng.choice(["fashion", "beauty", "fmcg", "electronics", "home"], rows, p=[0.26, 0.22, 0.24, 0.14, 0.14])
    treatment = (coupon_face_value > 0).astype(int)
    treatment_level = np.select(
        [coupon_face_value == 0, coupon_face_value <= 5, coupon_face_value <= 10, coupon_face_value <= 20],
        [0, 1, 2, 3],
        default=4,
    )
    meets_threshold = cart_value >= min_spend
    arbitrage_risk = np.clip(0.05 + 0.34 * (prior_coupon_uses > 3) + 0.24 * price_sensitivity - 0.18 * loyalty, 0.01, 0.86)
    true_uplift = np.clip(
        0.012
        + 0.065 * treatment_level
        + 0.13 * price_sensitivity
        + 0.055 * (recency_days > 45)
        + 0.035 * meets_threshold
        - 0.045 * loyalty
        - 0.055 * (coupon_face_value > cart_value * margin_rate * 0.8)
        - 0.085 * (arbitrage_risk > 0.62),
        -0.10,
        0.38,
    )
    base_prob = _sigmoid(-2.35 + 0.012 * user_value + 0.42 * loyalty - 0.009 * recency_days + 0.18 * meets_threshold)
    outcome = rng.binomial(1, np.clip(base_prob + treatment * true_uplift, 0.002, 0.94))
    gross_margin = cart_value * margin_rate
    expected_coupon_cost = coupon_face_value * np.clip(0.12 + 0.52 * price_sensitivity + 0.18 * meets_threshold, 0.01, 0.92)
    oracle_policy_value = true_uplift * gross_margin - expected_coupon_cost - 0.22 * arbitrage_risk - 0.006 * (channel == "app_push")
    df = pd.DataFrame(
        {
            "user_value": user_value.round(4),
            "margin_rate": margin_rate.round(4),
            "price_sensitivity": price_sensitivity.round(4),
            "loyalty": loyalty.round(4),
            "recency_days": recency_days.round(3),
            "prior_coupon_uses": prior_coupon_uses,
            "cart_value": cart_value.round(4),
            "coupon_face_value": coupon_face_value,
            "min_spend": min_spend,
            "treatment_level": treatment_level,
            "channel": channel,
            "category": category,
            "meets_threshold": meets_threshold.astype(int),
            "arbitrage_risk": arbitrage_risk.round(4),
            "treatment": treatment,
            "outcome": outcome,
            "true_uplift": true_uplift.round(5),
            "gross_margin": gross_margin.round(5),
            "expected_coupon_cost": expected_coupon_cost.round(5),
            "oracle_policy_value": oracle_policy_value.round(5),
        }
    )
    output = DATASET_DIR / "synthetic_ecommerce_multi_coupon_uplift_6k.csv"
    df.to_csv(output, index=False)
    feature_cols = [
        "user_value",
        "margin_rate",
        "price_sensitivity",
        "loyalty",
        "recency_days",
        "prior_coupon_uses",
        "cart_value",
        "coupon_face_value",
        "min_spend",
        "treatment_level",
        "channel",
        "category",
        "meets_threshold",
        "arbitrage_risk",
    ]
    return {
        "id": "synthetic_ecommerce_multi_coupon_uplift_6k",
        "name": "E-commerce Multi-Coupon Uplift 6.2k",
        "kind": "synthetic_ecommerce_multi_coupon_uplift",
        "path": _relative(output),
        "rows": int(len(df)),
        "description": "Multi-face-value coupon dataset with minimum-spend thresholds, arbitrage risk and cost-aware policy value.",
        "source_url": "docs/UPLIFT_INDUSTRIAL_CASE_EXPANSION.md",
        "treatment_col": "treatment",
        "outcome_col": "outcome",
        "feature_cols": feature_cols,
        "task": "classification",
        "recommended_models": ["DRLearnerLightGBM", "XLearnerLightGBM", "RLearnerLightGBM", "MultiTLearnerGBM", "DoseResponseGBM", "UTBoostGBM"],
        "quick_train": {"epochs": 2, "batch_size": 256, "learning_rate": 0.001, "max_rows": 6200},
    }


def write_crm_overlap_journey_dataset() -> dict:
    rng = np.random.default_rng(2026051782)
    rows = 6400
    lifecycle_score = rng.beta(2.2, 2.8, rows)
    churn_risk = rng.beta(2.6, 2.1, rows)
    purchase_intent = rng.beta(2.4, 2.7, rows)
    recent_push_count = rng.poisson(3.2, rows).clip(0, 22)
    email_count = rng.poisson(1.6, rows).clip(0, 12)
    sms_count = rng.poisson(0.7, rows).clip(0, 7)
    journey_count = rng.choice([1, 2, 3, 4], rows, p=[0.32, 0.36, 0.22, 0.10])
    overlap_index = np.clip(0.15 * journey_count + 0.04 * recent_push_count + 0.06 * sms_count, 0, 1.8)
    channel = rng.choice(["push", "sms", "email", "in_app"], rows, p=[0.40, 0.17, 0.19, 0.24])
    topic = rng.choice(["discount", "new_feature", "retention", "recommendation"], rows, p=[0.32, 0.18, 0.26, 0.24])
    fatigue = np.clip(0.04 * recent_push_count + 0.10 * sms_count + 0.12 * (journey_count >= 3), 0, 0.92)
    propensity = _sigmoid(-0.3 + 0.72 * churn_risk + 0.45 * purchase_intent + 0.24 * (topic == "discount") - 0.55 * fatigue)
    treatment = rng.binomial(1, propensity)
    true_uplift = np.clip(
        0.018
        + 0.14 * churn_risk
        + 0.08 * purchase_intent
        + 0.045 * (topic == "retention")
        - 0.10 * fatigue
        - 0.065 * (overlap_index > 0.9)
        - 0.045 * (channel == "sms"),
        -0.12,
        0.30,
    )
    base_prob = _sigmoid(-2.25 + 1.0 * lifecycle_score + 0.65 * purchase_intent - 0.40 * churn_risk)
    outcome = rng.binomial(1, np.clip(base_prob + treatment * true_uplift, 0.002, 0.88))
    optout_uplift = np.clip(0.005 + 0.08 * fatigue + 0.05 * (channel == "sms") + 0.035 * (journey_count >= 3), 0, 0.28)
    contact_cost = np.select([channel == "sms", channel == "push", channel == "email"], [0.035, 0.006, 0.002], default=0.004)
    long_term_value = rng.gamma(2.0, 25, rows).clip(2, 240)
    cannibalization_risk = np.clip(0.06 + 0.30 * (overlap_index > 0.9) + 0.20 * lifecycle_score - 0.12 * churn_risk, 0.01, 0.72)
    oracle_policy_value = true_uplift * long_term_value - contact_cost - 0.8 * optout_uplift - 0.35 * cannibalization_risk
    df = pd.DataFrame(
        {
            "lifecycle_score": lifecycle_score.round(4),
            "churn_risk": churn_risk.round(4),
            "purchase_intent": purchase_intent.round(4),
            "recent_push_count": recent_push_count,
            "email_count": email_count,
            "sms_count": sms_count,
            "journey_count": journey_count,
            "overlap_index": overlap_index.round(4),
            "channel": channel,
            "topic": topic,
            "fatigue": fatigue.round(4),
            "propensity": propensity.round(4),
            "treatment": treatment,
            "outcome": outcome,
            "true_uplift": true_uplift.round(5),
            "optout_uplift": optout_uplift.round(5),
            "contact_cost": contact_cost.round(5),
            "long_term_value": long_term_value.round(5),
            "cannibalization_risk": cannibalization_risk.round(4),
            "oracle_policy_value": oracle_policy_value.round(5),
        }
    )
    output = DATASET_DIR / "synthetic_crm_overlap_journey_uplift_6k.csv"
    df.to_csv(output, index=False)
    feature_cols = [
        "lifecycle_score",
        "churn_risk",
        "purchase_intent",
        "recent_push_count",
        "email_count",
        "sms_count",
        "journey_count",
        "overlap_index",
        "channel",
        "topic",
        "fatigue",
    ]
    return {
        "id": "synthetic_crm_overlap_journey_uplift_6k",
        "name": "CRM Overlap Journey Uplift 6.4k",
        "kind": "synthetic_crm_overlap_journey_uplift",
        "path": _relative(output),
        "rows": int(len(df)),
        "description": "CRM multi-journey overlap dataset with fatigue, opt-out uplift, cannibalization risk and long-term value.",
        "source_url": "docs/UPLIFT_INDUSTRIAL_CASE_EXPANSION.md",
        "treatment_col": "treatment",
        "outcome_col": "outcome",
        "feature_cols": feature_cols,
        "task": "classification",
        "recommended_models": ["DRLearnerLightGBM", "RLearnerLightGBM", "PAVCalibratedDRLearnerLightGBM", "CausalForest", "CFRNet"],
        "quick_train": {"epochs": 2, "batch_size": 256, "learning_rate": 0.001, "max_rows": 6400},
    }


def write_continuous_bid_discount_dataset() -> dict:
    rng = np.random.default_rng(2026051783)
    rows = 5800
    baseline_intent = rng.beta(2.4, 2.5, rows)
    price_elasticity = rng.beta(2.8, 2.0, rows)
    supply_pressure = rng.beta(2.2, 2.3, rows)
    competitor_pressure = rng.beta(2.1, 2.4, rows)
    margin_rate = rng.beta(3.3, 3.2, rows).clip(0.08, 0.62)
    bid_multiplier = rng.uniform(0.75, 2.6, rows)
    discount_rate = rng.uniform(0.00, 0.32, rows)
    treatment_strength = np.clip(0.55 * (bid_multiplier - 0.75) / 1.85 + 0.45 * discount_rate / 0.32, 0, 1)
    treatment = (treatment_strength > 0.35).astype(int)
    channel = rng.choice(["search_ads", "feed_ads", "coupon", "push"], rows, p=[0.30, 0.25, 0.28, 0.17])
    market = rng.choice(["id", "th", "vn", "sg", "br"], rows, p=[0.28, 0.20, 0.18, 0.17, 0.17])
    saturation = np.clip(0.16 + 0.48 * competitor_pressure + 0.34 * (bid_multiplier > 2.0), 0, 0.95)
    marginal_response = np.clip(
        0.03
        + 0.20 * baseline_intent
        + 0.18 * price_elasticity
        + 0.12 * supply_pressure
        - 0.18 * saturation
        - 0.06 * (treatment_strength > 0.82),
        -0.06,
        0.36,
    )
    true_uplift = np.clip(marginal_response * treatment_strength - 0.055 * treatment_strength**2, -0.08, 0.32)
    base_prob = _sigmoid(-2.45 + 1.25 * baseline_intent + 0.38 * supply_pressure - 0.24 * competitor_pressure)
    outcome = rng.binomial(1, np.clip(base_prob + treatment * true_uplift, 0.002, 0.88))
    order_value = rng.gamma(2.1, 36, rows).clip(5, 360)
    media_cost = np.clip(0.02 + 0.11 * bid_multiplier + 0.08 * (channel == "search_ads"), 0.01, 0.48)
    discount_cost = order_value * discount_rate * 0.35
    oracle_policy_value = true_uplift * order_value * margin_rate - media_cost - discount_cost
    df = pd.DataFrame(
        {
            "baseline_intent": baseline_intent.round(4),
            "price_elasticity": price_elasticity.round(4),
            "supply_pressure": supply_pressure.round(4),
            "competitor_pressure": competitor_pressure.round(4),
            "margin_rate": margin_rate.round(4),
            "bid_multiplier": bid_multiplier.round(4),
            "discount_rate": discount_rate.round(4),
            "treatment_strength": treatment_strength.round(4),
            "channel": channel,
            "market": market,
            "saturation": saturation.round(4),
            "treatment": treatment,
            "outcome": outcome,
            "true_uplift": true_uplift.round(5),
            "order_value": order_value.round(5),
            "media_cost": media_cost.round(5),
            "discount_cost": discount_cost.round(5),
            "oracle_policy_value": oracle_policy_value.round(5),
        }
    )
    output = DATASET_DIR / "synthetic_continuous_bid_discount_uplift_5k.csv"
    df.to_csv(output, index=False)
    feature_cols = [
        "baseline_intent",
        "price_elasticity",
        "supply_pressure",
        "competitor_pressure",
        "margin_rate",
        "bid_multiplier",
        "discount_rate",
        "treatment_strength",
        "channel",
        "market",
        "saturation",
    ]
    return {
        "id": "synthetic_continuous_bid_discount_uplift_5k",
        "name": "Continuous Bid Discount Uplift 5.8k",
        "kind": "synthetic_continuous_bid_discount_uplift",
        "path": _relative(output),
        "rows": int(len(df)),
        "description": "Continuous treatment proxy dataset for bid multiplier and discount intensity with saturation and diminishing returns.",
        "source_url": "docs/UPLIFT_INDUSTRIAL_CASE_EXPANSION.md",
        "treatment_col": "treatment",
        "outcome_col": "outcome",
        "feature_cols": feature_cols,
        "task": "classification",
        "recommended_models": ["DRLearnerLightGBM", "RLearnerLightGBM", "TLearnerLightGBM", "DoseResponseGBM", "CausalForest"],
        "quick_train": {"epochs": 2, "batch_size": 256, "learning_rate": 0.001, "max_rows": 5800},
    }


def write_geo_holdout_incrementality_dataset() -> dict:
    rng = np.random.default_rng(202605196)
    geos = 96
    days = 64
    rows = geos * days
    geo_id = np.repeat(np.arange(geos), days)
    day_index = np.tile(np.arange(days), geos)
    week = day_index // 7
    city_tier = rng.choice(["tier1", "tier2", "tier3", "growth"], geos, p=[0.18, 0.32, 0.34, 0.16])
    category_focus = rng.choice(["food", "grocery", "ride", "retail"], geos, p=[0.30, 0.24, 0.22, 0.24])
    geo_size = rng.gamma(4.5, 1800, geos).clip(900, 22000)
    ad_sensitivity = rng.beta(2.2, 2.4, geos)
    baseline_cvr = rng.beta(2.5, 12.0, geos).clip(0.015, 0.48)
    competition = rng.beta(2.4, 2.8, geos)
    organic_growth = rng.normal(0.0, 0.035, geos)

    tier_rep = city_tier[geo_id]
    category_rep = category_focus[geo_id]
    geo_size_rep = geo_size[geo_id]
    ad_sensitivity_rep = ad_sensitivity[geo_id]
    baseline_cvr_rep = baseline_cvr[geo_id]
    competition_rep = competition[geo_id]
    organic_growth_rep = organic_growth[geo_id]
    weekday = day_index % 7
    seasonality = 0.08 * np.sin(day_index / 6.0) + 0.05 * (weekday >= 5)
    local_event_intensity = rng.beta(1.4, 4.5, rows)
    auction_pressure = rng.beta(2.0, 2.7, rows)
    budget_pressure = rng.beta(2.4, 3.2, rows)
    planned_spend = (
        4.0
        + 0.004 * geo_size_rep
        + 9.0 * auction_pressure
        + 5.0 * local_event_intensity
        + 6.5 * (tier_rep == "tier1")
    ).clip(5, 140)
    avg_order_value = rng.gamma(2.5, 18, rows).clip(12, 180)
    margin_rate = rng.beta(4.0, 4.2, rows).clip(0.10, 0.55)
    media_cpm = (3.2 + 2.4 * auction_pressure + 1.3 * (tier_rep == "tier1")).clip(2.5, 9.8)
    holdout_cell = ((geo_id % 8) == 0) | (((geo_id + week) % 19) == 0)
    switchback_block = ((geo_id // 6) + week) % 4
    randomized_intensity = rng.beta(2.0 + 1.2 * ad_sensitivity_rep, 2.7, rows)
    treatment_prob = np.clip(
        0.16
        + 0.48 * randomized_intensity
        + 0.15 * budget_pressure
        + 0.10 * (tier_rep == "growth")
        - 0.22 * holdout_cell,
        0.03,
        0.88,
    )
    treatment = rng.binomial(1, treatment_prob)
    treatment = np.where(holdout_cell, 0, treatment)

    weak_overlap_risk = np.clip(np.abs(treatment_prob - 0.5) * 1.7 + 0.18 * holdout_cell, 0, 0.96)
    geo_spillover_risk = np.clip(0.04 + 0.26 * competition_rep + 0.13 * (switchback_block == 3), 0.01, 0.72)
    synthetic_control_gap = rng.normal(0.0, 0.018, rows) + 0.035 * organic_growth_rep - 0.018 * competition_rep
    saturation = np.clip(planned_spend / (planned_spend + 35 + 0.006 * geo_size_rep), 0.02, 0.91)
    true_uplift = np.clip(
        0.018
        + 0.17 * ad_sensitivity_rep
        + 0.035 * local_event_intensity
        + 0.025 * (category_rep == "grocery")
        - 0.055 * competition_rep
        - 0.060 * saturation
        - 0.025 * geo_spillover_risk
        + synthetic_control_gap,
        -0.12,
        0.32,
    )
    base_prob = np.clip(
        baseline_cvr_rep
        + 0.035 * seasonality
        + 0.020 * local_event_intensity
        + organic_growth_rep
        - 0.018 * competition_rep,
        0.01,
        0.86,
    )
    outcome_prob = np.clip(base_prob + treatment * true_uplift, 0.002, 0.95)
    outcome = rng.binomial(1, outcome_prob)
    incremental_revenue = true_uplift * avg_order_value
    media_cost = planned_spend * (0.018 + 0.0045 * media_cpm)
    oracle_policy_value = incremental_revenue * margin_rate - media_cost - 0.85 * geo_spillover_risk - 0.45 * weak_overlap_risk

    df = pd.DataFrame(
        {
            "geo_id": geo_id,
            "day_index": day_index,
            "week": week,
            "weekday": weekday,
            "city_tier": tier_rep,
            "category_focus": category_rep,
            "geo_size": geo_size_rep.round(2),
            "baseline_cvr": baseline_cvr_rep.round(5),
            "ad_sensitivity": ad_sensitivity_rep.round(5),
            "competition": competition_rep.round(5),
            "organic_growth": organic_growth_rep.round(5),
            "seasonality": seasonality.round(5),
            "local_event_intensity": local_event_intensity.round(5),
            "auction_pressure": auction_pressure.round(5),
            "budget_pressure": budget_pressure.round(5),
            "planned_spend": planned_spend.round(4),
            "avg_order_value": avg_order_value.round(4),
            "margin_rate": margin_rate.round(5),
            "media_cpm": media_cpm.round(4),
            "holdout_cell": holdout_cell.astype(int),
            "switchback_block": switchback_block,
            "treatment_prob": treatment_prob.round(5),
            "weak_overlap_risk": weak_overlap_risk.round(5),
            "geo_spillover_risk": geo_spillover_risk.round(5),
            "synthetic_control_gap": synthetic_control_gap.round(5),
            "saturation": saturation.round(5),
            "true_uplift": true_uplift.round(5),
            "incremental_revenue": incremental_revenue.round(5),
            "media_cost": media_cost.round(5),
            "oracle_policy_value": oracle_policy_value.round(5),
            "treatment": treatment.astype(int),
            "outcome": outcome.astype(int),
        }
    )
    output = DATASET_DIR / "synthetic_geo_holdout_incrementality_uplift_6k.csv"
    df.to_csv(output, index=False)
    feature_cols = [
        "geo_id",
        "day_index",
        "week",
        "weekday",
        "city_tier",
        "category_focus",
        "geo_size",
        "baseline_cvr",
        "ad_sensitivity",
        "competition",
        "organic_growth",
        "seasonality",
        "local_event_intensity",
        "auction_pressure",
        "budget_pressure",
        "planned_spend",
        "avg_order_value",
        "margin_rate",
        "media_cpm",
        "holdout_cell",
        "switchback_block",
        "treatment_prob",
        "weak_overlap_risk",
        "geo_spillover_risk",
        "synthetic_control_gap",
        "saturation",
    ]
    return {
        "id": "synthetic_geo_holdout_incrementality_uplift_6k",
        "name": "Geo Holdout Incrementality Uplift 6k",
        "kind": "synthetic_geo_holdout_incrementality_uplift",
        "path": _relative(output),
        "rows": int(len(df)),
        "description": "Geo/time holdout scenario for ads or subsidy incrementality, switchback blocks, weak overlap and spillover risk.",
        "source_url": "docs/UPLIFT_INDUSTRIAL_CASE_EXPANSION.md",
        "treatment_col": "treatment",
        "outcome_col": "outcome",
        "feature_cols": feature_cols,
        "task": "classification",
        "recommended_models": ["DRLearnerLightGBM", "RLearnerLightGBM", "TLearnerLightGBM", "CausalForest", "GeoLift optional"],
        "quick_train": {"epochs": 2, "batch_size": 256, "learning_rate": 0.001, "max_rows": 6144},
    }


def write_game_ops_gift_uplift_dataset() -> dict:
    rng = np.random.default_rng(202605181)
    rows = 6200
    player_level = rng.integers(1, 80, rows)
    days_since_install = rng.exponential(28, rows).clip(1, 360)
    sessions_7d = rng.poisson(6.5, rows).clip(0, 60)
    spend_30d = rng.gamma(1.4, 12.0, rows).clip(0, 320)
    skill_gap = rng.beta(2.2, 2.4, rows)
    social_degree = rng.beta(2.0, 3.0, rows)
    difficulty_spike = rng.beta(2.6, 2.1, rows)
    churn_risk = np.clip(0.12 + 0.55 * difficulty_spike - 0.04 * sessions_7d + 0.20 * (days_since_install < 7), 0.02, 0.94)
    payer_segment = rng.choice(["non_payer", "minnow", "dolphin", "whale"], rows, p=[0.64, 0.22, 0.11, 0.03])
    gift_type = rng.choice(["starter_pack", "energy", "skin_trial", "battle_pass_coupon"], rows, p=[0.30, 0.28, 0.22, 0.20])
    gift_cost = np.select(
        [gift_type == "starter_pack", gift_type == "energy", gift_type == "skin_trial"],
        [0.65, 0.25, 0.45],
        default=1.20,
    )
    frustration_relief = np.clip(0.15 + 0.55 * difficulty_spike + 0.18 * skill_gap - 0.20 * (payer_segment == "whale"), 0, 1)
    pay_to_win_backlash_risk = np.clip(0.04 + 0.35 * social_degree + 0.30 * (gift_type == "battle_pass_coupon") - 0.15 * (sessions_7d < 3), 0, 0.82)
    propensity = _sigmoid(-0.30 + 0.85 * churn_risk + 0.55 * frustration_relief + 0.25 * (payer_segment != "non_payer") - 0.45 * (payer_segment == "whale"))
    treatment = rng.binomial(1, propensity)
    base_retention = _sigmoid(-1.95 + 0.10 * np.log1p(sessions_7d) + 0.015 * player_level + 0.55 * social_degree - 1.15 * churn_risk)
    true_uplift = np.clip(
        0.012
        + 0.18 * frustration_relief
        + 0.06 * (gift_type == "energy")
        + 0.04 * (payer_segment == "minnow")
        - 0.08 * (payer_segment == "whale")
        - 0.10 * pay_to_win_backlash_risk,
        -0.14,
        0.30,
    )
    outcome = rng.binomial(1, np.clip(base_retention + treatment * true_uplift, 0.002, 0.94))
    incremental_ltv = true_uplift * (4.0 + 0.35 * spend_30d + 2.6 * social_degree)
    oracle_policy_value = incremental_ltv - gift_cost - 0.80 * pay_to_win_backlash_risk - 0.25 * churn_risk
    df = pd.DataFrame(
        {
            "player_level": player_level,
            "days_since_install": days_since_install.round(2),
            "sessions_7d": sessions_7d,
            "spend_30d": spend_30d.round(4),
            "skill_gap": skill_gap.round(5),
            "social_degree": social_degree.round(5),
            "difficulty_spike": difficulty_spike.round(5),
            "churn_risk": churn_risk.round(5),
            "payer_segment": payer_segment,
            "gift_type": gift_type,
            "gift_cost": gift_cost.round(5),
            "frustration_relief": frustration_relief.round(5),
            "pay_to_win_backlash_risk": pay_to_win_backlash_risk.round(5),
            "propensity": propensity.round(5),
            "treatment": treatment.astype(int),
            "outcome": outcome.astype(int),
            "true_uplift": true_uplift.round(5),
            "incremental_ltv": incremental_ltv.round(5),
            "oracle_policy_value": oracle_policy_value.round(5),
        }
    )
    output = DATASET_DIR / "synthetic_game_ops_gift_uplift_6k.csv"
    df.to_csv(output, index=False)
    feature_cols = [
        "player_level",
        "days_since_install",
        "sessions_7d",
        "spend_30d",
        "skill_gap",
        "social_degree",
        "difficulty_spike",
        "churn_risk",
        "payer_segment",
        "gift_type",
        "gift_cost",
        "frustration_relief",
        "pay_to_win_backlash_risk",
    ]
    return {
        "id": "synthetic_game_ops_gift_uplift_6k",
        "name": "Game Ops Gift / First-Purchase Uplift 6.2k",
        "kind": "synthetic_game_ops_gift_uplift",
        "path": _relative(output),
        "rows": int(len(df)),
        "description": "Game operations intervention dataset for礼包/首充/energy gifts with retention uplift, LTV, backlash risk and churn risk.",
        "source_url": "docs/UPLIFT_INDUSTRIAL_CASE_EXPANSION.md",
        "treatment_col": "treatment",
        "outcome_col": "outcome",
        "feature_cols": feature_cols,
        "task": "classification",
        "recommended_models": ["DRLearnerLightGBM", "RLearnerLightGBM", "TLearnerLightGBM", "CausalForest", "UpliftTree", "ContrastiveUpliftNet"],
        "quick_train": {"epochs": 2, "batch_size": 256, "learning_rate": 0.001, "max_rows": 6200},
    }


def write_fintech_credit_line_uplift_dataset() -> dict:
    rng = np.random.default_rng(202605182)
    rows = 6200
    credit_score = rng.normal(680, 65, rows).clip(420, 840)
    utilization = rng.beta(2.4, 2.0, rows)
    income_band = rng.choice(["low", "mid", "high", "premium"], rows, p=[0.24, 0.43, 0.25, 0.08])
    tenure_months = rng.exponential(28, rows).clip(1, 180)
    delinquency_12m = rng.poisson(np.clip(1.2 - (credit_score - 600) / 260, 0.05, 2.5), rows).clip(0, 8)
    liquidity_need = rng.beta(2.1, 2.4, rows)
    price_sensitivity = rng.beta(2.4, 2.6, rows)
    relationship_value = rng.gamma(2.0, 160, rows).clip(40, 2500)
    offer_type = rng.choice(["limit_increase", "apr_discount", "fee_waiver", "installment_coupon"], rows, p=[0.35, 0.25, 0.18, 0.22])
    capital_cost = np.clip(0.006 * relationship_value + 120 * utilization + 18 * delinquency_12m, 12, 480)
    credit_cost = np.select(
        [offer_type == "limit_increase", offer_type == "apr_discount", offer_type == "fee_waiver"],
        [0.018 * relationship_value, 0.014 * relationship_value, 8.0],
        default=16.0,
    )
    default_risk_lift = np.clip(0.015 + 0.12 * utilization + 0.055 * delinquency_12m - 0.00012 * (credit_score - 650), 0.0, 0.55)
    propensity = _sigmoid(-1.05 + 1.2 * liquidity_need + 0.55 * utilization + 0.32 * (credit_score > 660) - 0.62 * (delinquency_12m >= 3))
    treatment = rng.binomial(1, propensity)
    base_takeup = _sigmoid(-2.8 + 1.15 * liquidity_need + 0.45 * price_sensitivity + 0.25 * (income_band == "mid") - 0.40 * (delinquency_12m > 2))
    true_uplift = np.clip(
        0.018
        + 0.19 * liquidity_need
        + 0.07 * (offer_type == "apr_discount")
        + 0.05 * (offer_type == "limit_increase")
        - 0.13 * default_risk_lift
        - 0.055 * (credit_score < 560),
        -0.12,
        0.32,
    )
    outcome = rng.binomial(1, np.clip(base_takeup + treatment * true_uplift, 0.002, 0.90))
    incremental_interest_revenue = true_uplift * relationship_value * np.clip(0.06 + 0.18 * utilization, 0.02, 0.32)
    oracle_policy_value = incremental_interest_revenue - credit_cost - capital_cost * default_risk_lift
    df = pd.DataFrame(
        {
            "credit_score": credit_score.round(2),
            "utilization": utilization.round(5),
            "income_band": income_band,
            "tenure_months": tenure_months.round(2),
            "delinquency_12m": delinquency_12m,
            "liquidity_need": liquidity_need.round(5),
            "price_sensitivity": price_sensitivity.round(5),
            "relationship_value": relationship_value.round(4),
            "offer_type": offer_type,
            "credit_cost": credit_cost.round(5),
            "capital_cost": capital_cost.round(5),
            "default_risk_lift": default_risk_lift.round(5),
            "propensity": propensity.round(5),
            "treatment": treatment.astype(int),
            "outcome": outcome.astype(int),
            "true_uplift": true_uplift.round(5),
            "incremental_interest_revenue": incremental_interest_revenue.round(5),
            "oracle_policy_value": oracle_policy_value.round(5),
        }
    )
    output = DATASET_DIR / "synthetic_fintech_credit_line_uplift_6k.csv"
    df.to_csv(output, index=False)
    feature_cols = [
        "credit_score",
        "utilization",
        "income_band",
        "tenure_months",
        "delinquency_12m",
        "liquidity_need",
        "price_sensitivity",
        "relationship_value",
        "offer_type",
        "credit_cost",
        "capital_cost",
        "default_risk_lift",
    ]
    return {
        "id": "synthetic_fintech_credit_line_uplift_6k",
        "name": "FinTech Credit-Line / APR Offer Uplift 6.2k",
        "kind": "synthetic_fintech_credit_line_uplift",
        "path": _relative(output),
        "rows": int(len(df)),
        "description": "Financial-services treatment dataset for credit-line increases, APR discounts and fee waivers with revenue uplift and default-risk lift.",
        "source_url": "docs/UPLIFT_INDUSTRIAL_CASE_EXPANSION.md",
        "treatment_col": "treatment",
        "outcome_col": "outcome",
        "feature_cols": feature_cols,
        "task": "classification",
        "recommended_models": ["DRLearnerLightGBM", "RLearnerLightGBM", "CausalForest", "TLearnerLightGBM", "DML", "PolicyLearner"],
        "quick_train": {"epochs": 2, "batch_size": 256, "learning_rate": 0.001, "max_rows": 6200},
    }


def write_customer_service_escalation_uplift_dataset() -> dict:
    rng = np.random.default_rng(202605183)
    rows = 6200
    customer_value = rng.gamma(2.4, 120, rows).clip(30, 2200)
    issue_complexity = rng.beta(2.5, 2.2, rows)
    sentiment_score = rng.normal(0, 1, rows).clip(-3, 3)
    bot_confidence = rng.beta(3.0, 2.0, rows)
    prior_tickets_90d = rng.poisson(2.1, rows).clip(0, 20)
    wait_time_minutes = rng.gamma(2.0, 2.7, rows).clip(0.2, 45)
    channel = rng.choice(["chat", "phone", "email", "in_app"], rows, p=[0.42, 0.20, 0.20, 0.18])
    issue_type = rng.choice(["billing", "delivery", "technical", "refund", "account"], rows, p=[0.21, 0.24, 0.22, 0.18, 0.15])
    escalation_cost = np.clip(1.1 + 5.2 * issue_complexity + 0.10 * wait_time_minutes + 1.4 * (channel == "phone"), 0.8, 16.0)
    retention_value = customer_value * np.clip(0.10 + 0.22 * (issue_type == "billing") + 0.18 * (sentiment_score < -0.8), 0.05, 0.55)
    handle_time_penalty = np.clip(0.25 + 0.08 * wait_time_minutes + 0.80 * issue_complexity, 0.1, 6.0)
    propensity = _sigmoid(-0.80 + 1.35 * issue_complexity - 1.00 * bot_confidence + 0.50 * (sentiment_score < -0.5) + 0.18 * prior_tickets_90d)
    treatment = rng.binomial(1, propensity)
    base_resolution = _sigmoid(1.0 + 1.2 * bot_confidence - 1.4 * issue_complexity - 0.20 * prior_tickets_90d + 0.35 * (sentiment_score > 0))
    csat_uplift = np.clip(0.025 + 0.23 * issue_complexity + 0.16 * (sentiment_score < -0.6) - 0.18 * bot_confidence - 0.04 * (wait_time_minutes > 20), -0.10, 0.34)
    outcome = rng.binomial(1, np.clip(base_resolution + treatment * csat_uplift, 0.002, 0.96))
    oracle_policy_value = csat_uplift * retention_value - escalation_cost - handle_time_penalty
    df = pd.DataFrame(
        {
            "customer_value": customer_value.round(4),
            "issue_complexity": issue_complexity.round(5),
            "sentiment_score": sentiment_score.round(5),
            "bot_confidence": bot_confidence.round(5),
            "prior_tickets_90d": prior_tickets_90d,
            "wait_time_minutes": wait_time_minutes.round(4),
            "channel": channel,
            "issue_type": issue_type,
            "escalation_cost": escalation_cost.round(5),
            "retention_value": retention_value.round(5),
            "handle_time_penalty": handle_time_penalty.round(5),
            "propensity": propensity.round(5),
            "treatment": treatment.astype(int),
            "outcome": outcome.astype(int),
            "true_uplift": csat_uplift.round(5),
            "csat_uplift": csat_uplift.round(5),
            "oracle_policy_value": oracle_policy_value.round(5),
        }
    )
    output = DATASET_DIR / "synthetic_customer_service_escalation_uplift_6k.csv"
    df.to_csv(output, index=False)
    feature_cols = [
        "customer_value",
        "issue_complexity",
        "sentiment_score",
        "bot_confidence",
        "prior_tickets_90d",
        "wait_time_minutes",
        "channel",
        "issue_type",
        "escalation_cost",
        "handle_time_penalty",
    ]
    return {
        "id": "synthetic_customer_service_escalation_uplift_6k",
        "name": "Customer Service Human Escalation Uplift 6.2k",
        "kind": "synthetic_customer_service_escalation_uplift",
        "path": _relative(output),
        "rows": int(len(df)),
        "description": "Customer-support routing dataset for escalating bot/chat cases to human agents with CSAT uplift, retention value and handling-cost penalties.",
        "source_url": "docs/UPLIFT_INDUSTRIAL_CASE_EXPANSION.md",
        "treatment_col": "treatment",
        "outcome_col": "outcome",
        "feature_cols": feature_cols,
        "task": "classification",
        "recommended_models": ["DRLearnerLightGBM", "RLearnerLightGBM", "TLearnerLightGBM", "CausalForest", "LLMRoutingUplift"],
        "quick_train": {"epochs": 2, "batch_size": 256, "learning_rate": 0.001, "max_rows": 6200},
    }


def write_healthcare_followup_uplift_dataset() -> dict:
    rng = np.random.default_rng(202605184)
    rows = 6200
    age = rng.normal(54, 16, rows).clip(18, 92)
    chronic_score = rng.beta(2.3, 2.1, rows)
    prior_no_show_rate = rng.beta(2.1, 4.0, rows)
    medication_adherence = rng.beta(3.0, 2.8, rows)
    risk_score = np.clip(0.20 + 0.45 * chronic_score + 0.20 * (age > 65) + 0.20 * prior_no_show_rate - 0.18 * medication_adherence, 0.02, 0.96)
    access_barrier = rng.beta(2.0, 3.2, rows)
    channel = rng.choice(["sms", "nurse_call", "app_reminder", "community_worker"], rows, p=[0.36, 0.24, 0.26, 0.14])
    outreach_cost = np.select(
        [channel == "sms", channel == "app_reminder", channel == "nurse_call"],
        [0.05, 0.02, 5.5],
        default=11.0,
    )
    clinical_risk = np.clip(0.08 + 0.42 * risk_score + 0.16 * (age > 70), 0.02, 0.90)
    no_show_risk = np.clip(0.06 + 0.58 * prior_no_show_rate + 0.22 * access_barrier - 0.18 * medication_adherence, 0.01, 0.88)
    propensity = _sigmoid(-0.55 + 1.10 * risk_score + 0.70 * access_barrier + 0.35 * no_show_risk + 0.30 * (channel == "nurse_call"))
    treatment = rng.binomial(1, propensity)
    base_adherence = _sigmoid(-0.75 + 1.40 * medication_adherence - 0.95 * no_show_risk - 0.35 * chronic_score)
    true_uplift = np.clip(
        0.012
        + 0.20 * access_barrier * (channel != "app_reminder")
        + 0.12 * clinical_risk * (channel == "nurse_call")
        + 0.10 * no_show_risk * (channel == "community_worker")
        - 0.08 * (medication_adherence > 0.76)
        - 0.04 * (clinical_risk > 0.78),
        -0.10,
        0.30,
    )
    outcome = rng.binomial(1, np.clip(base_adherence + treatment * true_uplift, 0.002, 0.96))
    adherence_value = np.clip(12 + 220 * clinical_risk + 80 * chronic_score, 5, 450)
    oracle_policy_value = true_uplift * adherence_value - outreach_cost - 16.0 * np.maximum(clinical_risk - 0.80, 0)
    df = pd.DataFrame(
        {
            "age": age.round(2),
            "chronic_score": chronic_score.round(5),
            "prior_no_show_rate": prior_no_show_rate.round(5),
            "medication_adherence": medication_adherence.round(5),
            "risk_score": risk_score.round(5),
            "access_barrier": access_barrier.round(5),
            "channel": channel,
            "outreach_cost": outreach_cost.round(5),
            "clinical_risk": clinical_risk.round(5),
            "no_show_risk": no_show_risk.round(5),
            "adherence_value": adherence_value.round(5),
            "propensity": propensity.round(5),
            "treatment": treatment.astype(int),
            "outcome": outcome.astype(int),
            "true_uplift": true_uplift.round(5),
            "oracle_policy_value": oracle_policy_value.round(5),
        }
    )
    output = DATASET_DIR / "synthetic_healthcare_followup_uplift_6k.csv"
    df.to_csv(output, index=False)
    feature_cols = [
        "age",
        "chronic_score",
        "prior_no_show_rate",
        "medication_adherence",
        "risk_score",
        "access_barrier",
        "channel",
        "outreach_cost",
        "clinical_risk",
        "no_show_risk",
        "adherence_value",
    ]
    return {
        "id": "synthetic_healthcare_followup_uplift_6k",
        "name": "Healthcare Follow-Up / Adherence Uplift 6.2k",
        "kind": "synthetic_healthcare_followup_uplift",
        "path": _relative(output),
        "rows": int(len(df)),
        "description": "Healthcare follow-up outreach dataset for SMS/nurse/app/community interventions with adherence uplift, no-show risk and clinical-risk guardrails.",
        "source_url": "docs/UPLIFT_INDUSTRIAL_CASE_EXPANSION.md",
        "treatment_col": "treatment",
        "outcome_col": "outcome",
        "feature_cols": feature_cols,
        "task": "classification",
        "recommended_models": ["DRLearnerLightGBM", "RLearnerLightGBM", "CausalForest", "TLearnerLightGBM", "DML", "InterpretablePolicyTree"],
        "quick_train": {"epochs": 2, "batch_size": 256, "learning_rate": 0.001, "max_rows": 6200},
    }


def write_saas_retention_offer_uplift_dataset() -> dict:
    rng = np.random.default_rng(202605185)
    rows = 6200
    arr = rng.lognormal(7.1, 0.75, rows).clip(600, 120000)
    seats = rng.poisson(28, rows).clip(1, 500)
    product_usage_30d = rng.beta(2.3, 2.8, rows)
    support_tickets_90d = rng.poisson(2.5, rows).clip(0, 30)
    nps = rng.normal(24, 38, rows).clip(-100, 100)
    renewal_days_left = rng.integers(7, 180, rows)
    contract_type = rng.choice(["monthly", "annual", "multi_year"], rows, p=[0.32, 0.52, 0.16])
    offer_type = rng.choice(["discount", "csm_call", "training", "feature_credit"], rows, p=[0.25, 0.34, 0.23, 0.18])
    churn_risk = np.clip(0.10 + 0.62 * (1 - product_usage_30d) + 0.018 * support_tickets_90d - 0.0025 * nps + 0.14 * (contract_type == "monthly"), 0.02, 0.96)
    expansion_potential = np.clip(0.10 + 0.45 * product_usage_30d + 0.10 * np.log1p(seats) + 0.15 * (contract_type != "monthly"), 0.02, 0.95)
    offer_cost = np.select(
        [offer_type == "discount", offer_type == "csm_call", offer_type == "training"],
        [0.07 * arr, 180.0, 350.0],
        default=0.025 * arr,
    )
    discount_dependency_risk = np.clip(0.04 + 0.40 * (offer_type == "discount") + 0.20 * churn_risk - 0.12 * product_usage_30d, 0.01, 0.78)
    propensity = _sigmoid(-0.75 + 1.15 * churn_risk + 0.35 * expansion_potential + 0.35 * (arr > np.median(arr)) - 0.30 * (renewal_days_left > 120))
    treatment = rng.binomial(1, propensity)
    base_renewal = _sigmoid(1.45 - 2.15 * churn_risk + 0.90 * product_usage_30d + 0.008 * nps + 0.35 * (contract_type == "multi_year"))
    true_uplift = np.clip(
        0.015
        + 0.20 * churn_risk * (offer_type != "feature_credit")
        + 0.10 * expansion_potential * (offer_type == "training")
        + 0.07 * (offer_type == "csm_call") * (support_tickets_90d > 3)
        - 0.10 * discount_dependency_risk
        - 0.055 * (product_usage_30d > 0.82),
        -0.12,
        0.34,
    )
    outcome = rng.binomial(1, np.clip(base_renewal + treatment * true_uplift, 0.002, 0.98))
    expansion_value = true_uplift * arr * np.clip(0.35 + 0.55 * expansion_potential, 0.15, 0.95)
    oracle_policy_value = expansion_value - offer_cost - 0.08 * arr * discount_dependency_risk
    df = pd.DataFrame(
        {
            "arr": arr.round(4),
            "seats": seats,
            "product_usage_30d": product_usage_30d.round(5),
            "support_tickets_90d": support_tickets_90d,
            "nps": nps.round(2),
            "renewal_days_left": renewal_days_left,
            "contract_type": contract_type,
            "offer_type": offer_type,
            "churn_risk": churn_risk.round(5),
            "expansion_potential": expansion_potential.round(5),
            "offer_cost": offer_cost.round(5),
            "discount_dependency_risk": discount_dependency_risk.round(5),
            "propensity": propensity.round(5),
            "treatment": treatment.astype(int),
            "outcome": outcome.astype(int),
            "true_uplift": true_uplift.round(5),
            "expansion_value": expansion_value.round(5),
            "oracle_policy_value": oracle_policy_value.round(5),
        }
    )
    output = DATASET_DIR / "synthetic_saas_retention_offer_uplift_6k.csv"
    df.to_csv(output, index=False)
    feature_cols = [
        "arr",
        "seats",
        "product_usage_30d",
        "support_tickets_90d",
        "nps",
        "renewal_days_left",
        "contract_type",
        "offer_type",
        "churn_risk",
        "expansion_potential",
        "offer_cost",
        "discount_dependency_risk",
    ]
    return {
        "id": "synthetic_saas_retention_offer_uplift_6k",
        "name": "SaaS Retention / Renewal Offer Uplift 6.2k",
        "kind": "synthetic_saas_retention_offer_uplift",
        "path": _relative(output),
        "rows": int(len(df)),
        "description": "B2B SaaS retention dataset for discounts, CSM calls, training and feature credits with ARR, churn risk, offer cost and discount-dependency risk.",
        "source_url": "docs/UPLIFT_INDUSTRIAL_CASE_EXPANSION.md",
        "treatment_col": "treatment",
        "outcome_col": "outcome",
        "feature_cols": feature_cols,
        "task": "classification",
        "recommended_models": ["DRLearnerLightGBM", "RLearnerLightGBM", "CausalForest", "TLearnerLightGBM", "UpliftTree", "PolicyValueOptimizer"],
        "quick_train": {"epochs": 2, "batch_size": 256, "learning_rate": 0.001, "max_rows": 6200},
    }


def main() -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    entries = [
        write_coupon_profit_dataset(),
        write_recommendation_intervention_dataset(),
        write_marketplace_subsidy_dataset(),
        write_baidu_waimai_red_packet_dataset(),
        write_didi_passenger_subsidy_dataset(),
        write_didi_driver_supply_subsidy_dataset(),
        write_didi_budget_allocation_dataset(),
        write_shopee_ads_full_funnel_dataset(),
        write_spotify_inapp_message_dataset(),
        write_doordash_ghost_ads_dataset(),
        write_merchant_subsidy_dataset(),
        write_ecommerce_multi_coupon_dataset(),
        write_crm_overlap_journey_dataset(),
        write_continuous_bid_discount_dataset(),
        write_geo_holdout_incrementality_dataset(),
        write_game_ops_gift_uplift_dataset(),
        write_fintech_credit_line_uplift_dataset(),
        write_customer_service_escalation_uplift_dataset(),
        write_healthcare_followup_uplift_dataset(),
        write_saas_retention_offer_uplift_dataset(),
    ]
    _update_manifest(entries)
    print(json.dumps({"status": "ok", "datasets": entries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
