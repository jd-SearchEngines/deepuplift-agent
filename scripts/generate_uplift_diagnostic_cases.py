from __future__ import annotations

import json
import sys
from hashlib import blake2b
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepuplift.core.failure_modes import diagnostic_case_rows


DATA_DIR = ROOT / "examples" / "diagnostic_cases"
REPORTS = ROOT / "reports"


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def stable_seed(*parts: object) -> int:
    text = "::".join(str(part) for part in parts)
    digest = blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % (2**32)


def base_frame(case_id: str, n: int = 520, seed: int = 2026051701) -> pd.DataFrame:
    rng = np.random.default_rng(stable_seed(case_id, seed))
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    intent = sigmoid(1.2 * x1 + 0.8 * x2 + rng.normal(0, 0.4, n))
    price_sensitivity = sigmoid(-0.6 * x1 + 0.9 * x2 + rng.normal(0, 0.5, n))
    cost = rng.gamma(2.0, 1.6, n).clip(0.2, 12)
    value = rng.gamma(2.4, 9.0, n).clip(2, 90)
    return pd.DataFrame(
        {
            "case_id": case_id,
            "x1": x1.round(5),
            "x2": x2.round(5),
            "intent": intent.round(5),
            "price_sensitivity": price_sensitivity.round(5),
            "treatment_cost": cost.round(5),
            "business_value": value.round(5),
        }
    )


def build_case(case_id: str) -> pd.DataFrame:
    df = base_frame(case_id)
    n = len(df)
    rng = np.random.default_rng(stable_seed(case_id))
    intent = df["intent"].to_numpy()
    sensitivity = df["price_sensitivity"].to_numpy()
    cost = df["treatment_cost"].to_numpy()
    value = df["business_value"].to_numpy()

    propensity = np.full(n, 0.5)
    true_uplift = 0.06 + 0.12 * sensitivity - 0.04 * intent
    mock_score = true_uplift + rng.normal(0, 0.025, n)
    policy_value = true_uplift * value - cost
    outcome_d1 = np.zeros(n, dtype=int)
    outcome_d30 = np.zeros(n, dtype=int)
    click = np.zeros(n, dtype=int)
    conversion = np.zeros(n, dtype=int)
    geo_holdout_lift = np.full(n, np.nan)
    dose = np.zeros(n)
    quality_uplift = np.zeros(n)
    latency_penalty = np.zeros(n)

    if case_id == "weak_uplift_signal":
        true_uplift = 0.008 + 0.018 * (sensitivity - 0.5)
        mock_score = intent + 0.02 * rng.normal(size=n)
    elif case_id == "imbalanced_treatment":
        propensity = np.clip(0.88 + 0.09 * intent - 0.03 * sensitivity, 0.80, 0.98)
        true_uplift = 0.05 + 0.09 * sensitivity - 0.03 * intent
        mock_score = true_uplift + rng.normal(0, 0.04, n)
    elif case_id == "extreme_propensity":
        propensity = np.where(intent > 0.58, 0.97, 0.04)
        true_uplift = 0.04 + 0.13 * sensitivity - 0.02 * intent
        mock_score = true_uplift + rng.normal(0, 0.05, n)
    elif case_id == "negative_uplift_sleeping_dog":
        sleeping_dog = (intent > 0.74) & (sensitivity < 0.48)
        true_uplift = 0.12 * sensitivity - 0.18 * sleeping_dog.astype(float)
        mock_score = intent + 0.03 * rng.normal(size=n)
        df["sleeping_dog"] = sleeping_dog.astype(int)
        df["abuse_risk"] = np.clip(0.12 + 0.55 * sensitivity + 0.22 * sleeping_dog, 0, 1).round(5)
        policy_value = true_uplift * value - cost - 2.5 * df["abuse_risk"].to_numpy()
    elif case_id == "high_qini_low_roi":
        high_cost = 4.0 + 18.0 * sensitivity
        true_uplift = 0.08 + 0.16 * sensitivity
        mock_score = true_uplift + rng.normal(0, 0.015, n)
        df["treatment_cost"] = high_cost.round(5)
        cost = high_cost
        policy_value = true_uplift * value * 0.18 - cost
    elif case_id == "delayed_feedback_trap":
        true_uplift = 0.03 + 0.15 * sensitivity
        mock_score = true_uplift + rng.normal(0, 0.02, n)
        outcome_d1 = rng.binomial(1, np.clip(0.04 + 0.01 * true_uplift, 0, 1), n)
        outcome_d30 = rng.binomial(1, np.clip(0.10 + 0.55 * true_uplift, 0, 1), n)
        df["converted_d1"] = outcome_d1
        df["converted_d30"] = outcome_d30
        df["censored"] = rng.binomial(1, 0.18, n)
    elif case_id == "full_funnel_click_trap":
        true_uplift = 0.02 + 0.11 * sensitivity
        mock_score = true_uplift + rng.normal(0, 0.02, n)
        click_uplift = 0.18 + 0.18 * sensitivity
        conversion_uplift = -0.03 + 0.03 * sensitivity
        click = rng.binomial(1, np.clip(0.08 + click_uplift, 0, 1), n)
        conversion = rng.binomial(1, np.clip(0.04 + conversion_uplift, 0, 1), n)
        df["click"] = click
        df["conversion"] = conversion
        df["true_click_uplift"] = click_uplift.round(5)
        df["true_conversion_uplift"] = conversion_uplift.round(5)
        policy_value = conversion_uplift * value - cost
    elif case_id == "geo_holdout_conflict":
        true_uplift = 0.04 + 0.12 * sensitivity
        mock_score = true_uplift + rng.normal(0, 0.02, n)
        geo = np.arange(n) % 40
        holdout_noise = rng.normal(-0.02, 0.10, 40)
        geo_holdout_lift = true_uplift + holdout_noise[geo]
        df["geo_id"] = geo
        df["holdout_cell"] = (geo % 7 == 0).astype(int)
        df["geo_holdout_lift"] = geo_holdout_lift.round(5)
        policy_value = geo_holdout_lift * value - cost
    elif case_id == "llm_routing_cost_trap":
        true_uplift = 0.015 + 0.045 * (intent > 0.68).astype(float)
        mock_score = true_uplift + 0.04 * rng.normal(size=n)
        model_cost = 0.18 + 1.8 * intent
        latency_penalty = 0.08 + 0.6 * intent
        quality_uplift = true_uplift
        df["model_cost"] = model_cost.round(5)
        df["latency_penalty"] = latency_penalty.round(5)
        df["quality_uplift"] = quality_uplift.round(5)
        policy_value = quality_uplift * value * 0.35 - model_cost - latency_penalty
    elif case_id == "continuous_dose_saturation":
        dose = np.clip(0.2 + 0.95 * sensitivity + 0.15 * rng.normal(size=n), 0, 1.4)
        true_uplift = 0.02 + 0.20 * (1 - np.exp(-2.4 * dose)) - 0.05 * np.maximum(dose - 0.9, 0)
        mock_score = true_uplift + rng.normal(0, 0.02, n)
        dose_cost = 1.2 + 10.5 * dose**2
        df["treatment_strength"] = dose.round(5)
        df["dose_cost"] = dose_cost.round(5)
        policy_value = true_uplift * value * 0.35 - dose_cost

    propensity = np.clip(propensity, 0.01, 0.99)
    treatment = rng.binomial(1, propensity)
    base_prob = np.clip(0.05 + 0.55 * intent - 0.08 * sensitivity, 0.01, 0.88)
    outcome_prob = np.clip(base_prob + treatment * true_uplift, 0.001, 0.95)
    outcome = rng.binomial(1, outcome_prob)
    df["propensity"] = propensity.round(5)
    df["treatment"] = treatment.astype(int)
    df["outcome"] = outcome.astype(int)
    df["true_uplift"] = true_uplift.round(5)
    df["mock_uplift_score"] = mock_score.round(5)
    df["oracle_policy_value"] = policy_value.round(5)
    return df


def summarize_case(case: dict[str, str], df: pd.DataFrame) -> dict[str, object]:
    sorted_df = df.sort_values("mock_uplift_score", ascending=False)
    top_n = max(1, int(len(sorted_df) * 0.1))
    top = sorted_df.head(top_n)
    treatment_rate = float(df["treatment"].mean())
    weak_overlap_rate = float(((df["propensity"] < 0.08) | (df["propensity"] > 0.92)).mean())
    summary: dict[str, object] = {
        **case,
        "rows": int(len(df)),
        "treatment_rate": round(treatment_rate, 6),
        "weak_overlap_rate": round(weak_overlap_rate, 6),
        "top10_true_uplift": round(float(top["true_uplift"].mean()), 6),
        "top10_policy_value": round(float(top["oracle_policy_value"].sum()), 6),
        "policy_value_mean": round(float(df["oracle_policy_value"].mean()), 6),
        "mock_csv": str((DATA_DIR / f"{case['case_id']}.csv").relative_to(ROOT)),
    }
    if "converted_d1" in df.columns and "converted_d30" in df.columns:
        summary["d1_rate_top10"] = round(float(top["converted_d1"].mean()), 6)
        summary["d30_rate_top10"] = round(float(top["converted_d30"].mean()), 6)
    if "true_click_uplift" in df.columns and "true_conversion_uplift" in df.columns:
        summary["click_uplift_top10"] = round(float(top["true_click_uplift"].mean()), 6)
        summary["conversion_uplift_top10"] = round(float(top["true_conversion_uplift"].mean()), 6)
    if "geo_holdout_lift" in df.columns:
        summary["geo_holdout_lift_top10"] = round(float(top["geo_holdout_lift"].mean()), 6)
    if "quality_uplift" in df.columns:
        summary["quality_uplift_top10"] = round(float(top["quality_uplift"].mean()), 6)
    if "treatment_strength" in df.columns:
        summary["dose_top10"] = round(float(top["treatment_strength"].mean()), 6)
    return summary


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in diagnostic_case_rows():
        df = build_case(case["case_id"])
        out = DATA_DIR / f"{case['case_id']}.csv"
        df.to_csv(out, index=False)
        rows.append(summarize_case(case, df))

    summary = pd.DataFrame(rows)
    summary_csv = REPORTS / "uplift_diagnostic_cases_latest.csv"
    summary_json = REPORTS / "uplift_diagnostic_cases_latest.json"
    summary.to_csv(summary_csv, index=False)
    summary_json.write_text(
        json.dumps({"status": "ok", "cases": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"status": "ok", "cases": len(rows), "csv": str(summary_csv.relative_to(ROOT))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
