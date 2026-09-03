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


def write_full_funnel_ecup_dataset() -> dict:
    rng = np.random.default_rng(202605171)
    rows = 7000
    age = rng.normal(36, 10, rows).clip(18, 72)
    income_score = rng.beta(2.3, 3.2, rows)
    prior_sessions = rng.poisson(5, rows).clip(0, 35)
    prior_purchases = rng.poisson(1.3, rows).clip(0, 12)
    category_affinity = rng.beta(2.0, 2.8, rows)
    price_sensitivity = rng.beta(2.6, 2.4, rows)
    ad_slot_quality = rng.beta(2.8, 2.0, rows)
    device = rng.choice(["ios", "android", "web"], rows, p=[0.34, 0.39, 0.27])
    channel = rng.choice(["search", "feed", "video", "retargeting"], rows, p=[0.26, 0.34, 0.18, 0.22])

    high_intent = (category_affinity > 0.58) | (prior_purchases >= 2)
    deal_sensitive = price_sensitivity > 0.62
    mobile = np.isin(device, ["ios", "android"])
    retargeting = channel == "retargeting"

    propensity = _sigmoid(
        -0.45
        + 0.75 * high_intent
        + 0.45 * retargeting
        + 0.35 * ad_slot_quality
        + 0.25 * mobile
        - 0.30 * deal_sensitive
    )
    treatment = rng.binomial(1, propensity)

    base_impression = _sigmoid(-0.4 + 0.9 * ad_slot_quality + 0.35 * mobile + 0.25 * retargeting)
    impression_uplift = np.clip(0.04 + 0.20 * ad_slot_quality + 0.06 * retargeting - 0.05 * deal_sensitive, -0.03, 0.34)
    impression_prob = np.clip(base_impression + treatment * impression_uplift, 0.01, 0.98)
    impression = rng.binomial(1, impression_prob)

    base_click = _sigmoid(
        -2.4
        + 1.1 * category_affinity
        + 0.45 * high_intent
        + 0.35 * ad_slot_quality
        + 0.18 * (channel == "feed")
    )
    click_uplift = np.clip(0.01 + 0.13 * high_intent + 0.08 * ad_slot_quality + 0.05 * retargeting - 0.04 * deal_sensitive, -0.04, 0.26)
    click_prob = np.clip(base_click + treatment * click_uplift, 0.005, 0.92)
    click = rng.binomial(1, click_prob * np.maximum(impression, 0.25))

    base_conversion = _sigmoid(
        -3.1
        + 1.4 * category_affinity
        + 0.38 * prior_purchases
        + 0.55 * income_score
        - 0.45 * price_sensitivity
        + 0.25 * (device == "ios")
    )
    conversion_uplift = np.clip(
        0.005
        + 0.09 * high_intent
        + 0.08 * click_uplift
        + 0.04 * retargeting
        - 0.05 * (deal_sensitive & ~high_intent),
        -0.05,
        0.24,
    )
    conversion_prob = np.clip(base_conversion + treatment * conversion_uplift, 0.002, 0.75)
    conversion = rng.binomial(1, conversion_prob * np.maximum(click, 0.18))
    order_value = rng.gamma(shape=2.0, scale=38, size=rows).clip(8, 480)
    media_cost = (0.015 + 0.035 * ad_slot_quality + 0.010 * retargeting + 0.006 * (channel == "video")).clip(0.006, 0.09)
    oracle_policy_value = conversion_uplift * order_value - media_cost

    df = pd.DataFrame(
        {
            "age": age.round(2),
            "income_score": income_score.round(4),
            "prior_sessions": prior_sessions,
            "prior_purchases": prior_purchases,
            "category_affinity": category_affinity.round(4),
            "price_sensitivity": price_sensitivity.round(4),
            "ad_slot_quality": ad_slot_quality.round(4),
            "device": device,
            "channel": channel,
            "propensity": propensity.round(4),
            "treatment": treatment,
            "impression": impression,
            "click": click,
            "conversion": conversion,
            "outcome": conversion,
            "true_impression_uplift": impression_uplift.round(4),
            "true_click_uplift": click_uplift.round(4),
            "true_conversion_uplift": conversion_uplift.round(4),
            "true_uplift": conversion_uplift.round(4),
            "order_value": order_value.round(4),
            "media_cost": media_cost.round(5),
            "oracle_policy_value": oracle_policy_value.round(5),
        }
    )
    output = DATASET_DIR / "synthetic_ads_full_funnel_ecup_7k.csv"
    df.to_csv(output, index=False)
    feature_cols = [
        "age",
        "income_score",
        "prior_sessions",
        "prior_purchases",
        "category_affinity",
        "price_sensitivity",
        "ad_slot_quality",
        "device",
        "channel",
    ]
    return {
        "id": "synthetic_ads_full_funnel_ecup_7k",
        "name": "Synthetic Ads Full-Funnel ECUP 7k",
        "kind": "synthetic_full_funnel_uplift",
        "path": _relative(output),
        "rows": int(len(df)),
        "description": "Synthetic full-funnel ad uplift dataset with impression, click and conversion stages plus known stage-level uplift.",
        "source_url": "docs/UPLIFT_MODEL_CAPABILITY_ARCHITECTURE.md",
        "treatment_col": "treatment",
        "outcome_col": "outcome",
        "feature_cols": feature_cols,
        "task": "classification",
        "recommended_models": ["DRLearnerLightGBM", "RLearnerLightGBM", "TLearnerLightGBM", "EFIN", "DESCN", "CausalForest"],
        "quick_train": {"epochs": 2, "batch_size": 256, "learning_rate": 0.001, "max_rows": 7000},
    }


def write_delayed_feedback_dataset() -> dict:
    rng = np.random.default_rng(202605172)
    rows = 7000
    tenure_days = rng.exponential(280, rows).clip(1, 2200)
    recency_days = rng.exponential(26, rows).clip(0, 180)
    sessions_7d = rng.poisson(2.8, rows).clip(0, 28)
    prior_orders = rng.poisson(1.6, rows).clip(0, 18)
    churn_risk = rng.beta(2.4, 3.0, rows)
    value_score = rng.gamma(shape=2.1, scale=2.6, size=rows).clip(0.2, 22)
    fatigue_score = rng.beta(2.2, 4.2, rows)
    device = rng.choice(["ios", "android", "web"], rows, p=[0.36, 0.37, 0.27])
    lifecycle_stage = rng.choice(["new", "active", "at_risk", "dormant"], rows, p=[0.20, 0.42, 0.24, 0.14])

    at_risk = lifecycle_stage == "at_risk"
    dormant = lifecycle_stage == "dormant"
    active = lifecycle_stage == "active"
    high_value = value_score > np.quantile(value_score, 0.68)
    fatigued = fatigue_score > 0.68

    propensity = _sigmoid(
        -0.30
        + 0.85 * at_risk
        + 0.55 * dormant
        + 0.35 * high_value
        - 0.40 * fatigued
        - 0.22 * active
    )
    treatment = rng.binomial(1, propensity)

    base_30d = _sigmoid(
        -2.6
        - 0.018 * recency_days
        + 0.16 * sessions_7d
        + 0.24 * prior_orders
        + 0.42 * high_value
        - 0.55 * churn_risk
        - 0.35 * dormant
    )
    uplift_30d = np.clip(
        0.015
        + 0.13 * at_risk
        + 0.11 * dormant
        + 0.06 * high_value
        + 0.04 * (device == "ios")
        - 0.09 * fatigued
        - 0.04 * active,
        -0.08,
        0.30,
    )
    outcome_30d_prob = np.clip(base_30d + treatment * uplift_30d, 0.002, 0.88)
    converted_30d = rng.binomial(1, outcome_30d_prob)

    fast_responder_prob = np.clip(0.34 + 0.25 * active + 0.12 * sessions_7d / 10 - 0.20 * dormant, 0.05, 0.78)
    medium_responder_prob = np.clip(0.35 + 0.18 * at_risk + 0.10 * high_value, 0.08, 0.70)
    delay_bucket = rng.choice([3, 10, 21], rows, p=[0.34, 0.36, 0.30])
    delay_bucket = np.where(rng.random(rows) < fast_responder_prob, rng.integers(1, 8, rows), delay_bucket)
    delay_bucket = np.where(rng.random(rows) < medium_responder_prob, rng.integers(8, 15, rows), delay_bucket)
    conversion_delay_days = np.where(converted_30d == 1, delay_bucket, 999)
    converted_d1 = ((converted_30d == 1) & (conversion_delay_days <= 1)).astype(int)
    converted_d7 = ((converted_30d == 1) & (conversion_delay_days <= 7)).astype(int)
    converted_d14 = ((converted_30d == 1) & (conversion_delay_days <= 14)).astype(int)

    uplift_d7 = np.clip(uplift_30d * (0.45 + 0.15 * active - 0.10 * dormant), -0.06, 0.18)
    uplift_d14 = np.clip(uplift_30d * (0.72 + 0.08 * at_risk), -0.07, 0.24)
    contact_cost = (0.012 + 0.012 * fatigued + 0.006 * (device == "ios")).clip(0.006, 0.04)
    oracle_policy_value_d30 = uplift_30d * value_score - contact_cost - 0.025 * fatigued

    df = pd.DataFrame(
        {
            "tenure_days": tenure_days.round(1),
            "recency_days": recency_days.round(2),
            "sessions_7d": sessions_7d,
            "prior_orders": prior_orders,
            "churn_risk": churn_risk.round(4),
            "value_score": value_score.round(4),
            "fatigue_score": fatigue_score.round(4),
            "device": device,
            "lifecycle_stage": lifecycle_stage,
            "propensity": propensity.round(4),
            "treatment": treatment,
            "converted_d1": converted_d1,
            "converted_d7": converted_d7,
            "converted_d14": converted_d14,
            "converted_d30": converted_30d,
            "outcome": converted_30d,
            "conversion_delay_days": conversion_delay_days,
            "label_window_days": np.full(rows, 30),
            "censored": (conversion_delay_days == 999).astype(int),
            "true_uplift_d7": uplift_d7.round(4),
            "true_uplift_d14": uplift_d14.round(4),
            "true_uplift_d30": uplift_30d.round(4),
            "true_uplift": uplift_30d.round(4),
            "contact_cost": contact_cost.round(5),
            "oracle_policy_value": oracle_policy_value_d30.round(5),
        }
    )
    output = DATASET_DIR / "synthetic_growth_delayed_feedback_7k.csv"
    df.to_csv(output, index=False)
    feature_cols = [
        "tenure_days",
        "recency_days",
        "sessions_7d",
        "prior_orders",
        "churn_risk",
        "value_score",
        "fatigue_score",
        "device",
        "lifecycle_stage",
    ]
    return {
        "id": "synthetic_growth_delayed_feedback_7k",
        "name": "Synthetic Growth Delayed Feedback 7k",
        "kind": "synthetic_delayed_feedback_uplift",
        "path": _relative(output),
        "rows": int(len(df)),
        "description": "Synthetic growth/push dataset with D1/D7/D14/D30 conversion windows, delay labels and known delayed true uplift.",
        "source_url": "docs/UPLIFT_MODEL_CAPABILITY_ARCHITECTURE.md",
        "treatment_col": "treatment",
        "outcome_col": "outcome",
        "feature_cols": feature_cols,
        "task": "classification",
        "recommended_models": ["DRLearnerLightGBM", "RLearnerLightGBM", "PAVCalibratedDRLearnerLightGBM", "CFRNet", "DragonNet", "CausalForest"],
        "quick_train": {"epochs": 2, "batch_size": 256, "learning_rate": 0.001, "max_rows": 7000},
    }


def update_manifest(new_entries: list[dict]) -> None:
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


def main() -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    entries = [write_full_funnel_ecup_dataset(), write_delayed_feedback_dataset()]
    update_manifest(entries)
    print(json.dumps({"status": "ok", "datasets": entries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
