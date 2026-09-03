from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


MANIFEST = ROOT / "examples" / "datasets" / "manifest.json"

DATASET_CHECKS = {
    "synthetic_ads_full_funnel_ecup_7k": {
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
        "oracle_columns": {
            "propensity",
            "true_impression_uplift",
            "true_click_uplift",
            "true_conversion_uplift",
            "true_uplift",
            "media_cost",
            "oracle_policy_value",
            "order_value",
        },
        "kind": "synthetic_full_funnel_uplift",
    },
    "synthetic_growth_delayed_feedback_7k": {
        "required_columns": {
            "converted_d1",
            "converted_d7",
            "converted_d14",
            "converted_d30",
            "conversion_delay_days",
            "label_window_days",
            "censored",
            "true_uplift_d7",
            "true_uplift_d14",
            "true_uplift_d30",
            "true_uplift",
            "contact_cost",
            "oracle_policy_value",
            "treatment",
            "outcome",
        },
        "oracle_columns": {
            "propensity",
            "converted_d1",
            "converted_d7",
            "converted_d14",
            "converted_d30",
            "conversion_delay_days",
            "label_window_days",
            "censored",
            "true_uplift_d7",
            "true_uplift_d14",
            "true_uplift_d30",
            "true_uplift",
            "contact_cost",
            "oracle_policy_value",
        },
        "kind": "synthetic_delayed_feedback_uplift",
    },
}


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest_rows = {row.get("id"): row for row in manifest.get("datasets", [])}
    failures: list[str] = []
    summaries = []
    for dataset_id, checks in DATASET_CHECKS.items():
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
        leakage = sorted(checks["oracle_columns"] & set(dataset.get("feature_cols", [])))
        if leakage:
            failures.append(f"{dataset_id} oracle columns should not be features: {leakage}")
        treatment_rate = float(df[dataset["treatment_col"]].mean())
        outcome_rate = float(df[dataset["outcome_col"]].mean())
        if len(df) < 7000:
            failures.append(f"{dataset_id} expected at least 7000 rows, found {len(df)}")
        if not 0.20 <= treatment_rate <= 0.80:
            failures.append(f"{dataset_id} treatment rate outside smoke bounds: {treatment_rate:.3f}")
        if not 0.01 <= outcome_rate <= 0.75:
            failures.append(f"{dataset_id} outcome rate outside smoke bounds: {outcome_rate:.3f}")
        uplift = df["true_uplift"]
        if uplift.max() <= 0.15:
            failures.append(f"{dataset_id} true_uplift max too small: {uplift.max():.3f}")
        if df["oracle_policy_value"].quantile(0.9) <= df["oracle_policy_value"].quantile(0.1):
            failures.append(f"{dataset_id} oracle_policy_value has weak spread")
        summaries.append(
            {
                "dataset_id": dataset_id,
                "name": dataset.get("name"),
                "path": dataset.get("path"),
                "rows": len(df),
                "treatment_rate": treatment_rate,
                "outcome_rate": outcome_rate,
                "true_uplift_min": float(uplift.min()),
                "true_uplift_max": float(uplift.max()),
                "policy_value_p10": float(df["oracle_policy_value"].quantile(0.1)),
                "policy_value_p90": float(df["oracle_policy_value"].quantile(0.9)),
            }
        )
    payload = {"status": "fail" if failures else "ok", "datasets": summaries, "failures": failures}
    output = ROOT / "reports" / "frontier_synthetic_datasets_smoke_latest.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
