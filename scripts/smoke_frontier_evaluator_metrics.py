from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepuplift.core.evaluator import evaluate_uplift_predictions


MANIFEST = ROOT / "examples" / "datasets" / "manifest.json"


def load_dataset(dataset_id: str, rows: int = 1600) -> tuple[dict, pd.DataFrame]:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for item in payload.get("datasets", []):
        if item.get("id") == dataset_id:
            return item, pd.read_csv(ROOT / item["path"]).head(rows).copy()
    raise ValueError(f"Dataset not found: {dataset_id}")


def attach_score(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    work = df.copy()
    base = pd.to_numeric(work.get("true_uplift", pd.Series(np.zeros(len(work)))), errors="coerce").fillna(0.0)
    work["uplift_score"] = base + rng.normal(0, 0.015, size=len(work))
    work["y0_pred"] = np.clip(pd.to_numeric(work["outcome"], errors="coerce").fillna(0).mean() - work["uplift_score"] / 2, 0, 1)
    work["y1_pred"] = np.clip(work["y0_pred"] + work["uplift_score"], 0, 1)
    return work


def main() -> None:
    checks = []
    failures: list[str] = []

    ads_meta, ads = load_dataset("synthetic_ads_full_funnel_ecup_7k")
    ads_result = evaluate_uplift_predictions(
        attach_score(ads, seed=7),
        outcome_col=ads_meta["outcome_col"],
        treatment_col=ads_meta["treatment_col"],
        uplift_col="uplift_score",
        bins=8,
    )
    ads_metrics = ads_result["metrics"].get("full_funnel") or {}
    ads_rows = ads_metrics.get("top_k") or []
    checks.append(
        {
            "dataset_id": ads_meta["id"],
            "metric": "full_funnel",
            "stage_cols": ads_metrics.get("stage_cols"),
            "top_k_rows": len(ads_rows),
            "oracle_rows": len(ads_metrics.get("oracle_top_k") or []),
            "policy_rows": len(ads_metrics.get("policy_top_k") or []),
        }
    )
    if {"impression", "click", "conversion"} - set(ads_metrics.get("stage_cols") or []):
        failures.append("full_funnel missing expected stages")
    if len(ads_rows) < 9:
        failures.append("full_funnel top_k rows too small")

    growth_meta, growth = load_dataset("synthetic_growth_delayed_feedback_7k")
    growth_result = evaluate_uplift_predictions(
        attach_score(growth, seed=13),
        outcome_col=growth_meta["outcome_col"],
        treatment_col=growth_meta["treatment_col"],
        uplift_col="uplift_score",
        bins=8,
    )
    delayed_metrics = growth_result["metrics"].get("delayed_feedback") or {}
    delayed_rows = delayed_metrics.get("top_k") or []
    checks.append(
        {
            "dataset_id": growth_meta["id"],
            "metric": "delayed_feedback",
            "window_cols": delayed_metrics.get("window_cols"),
            "top_k_rows": len(delayed_rows),
            "oracle_rows": len(delayed_metrics.get("oracle_top_k") or []),
            "policy_rows": len(delayed_metrics.get("policy_top_k") or []),
            "summary": delayed_metrics.get("summary"),
        }
    )
    if {"converted_d1", "converted_d7", "converted_d14", "converted_d30"} - set(delayed_metrics.get("window_cols") or []):
        failures.append("delayed_feedback missing expected windows")
    if len(delayed_rows) < 12:
        failures.append("delayed_feedback top_k rows too small")

    payload = {"status": "fail" if failures else "ok", "checks": checks, "failures": failures}
    output = ROOT / "reports" / "frontier_evaluator_metrics_smoke_latest.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
