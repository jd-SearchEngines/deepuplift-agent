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
SCENARIO_DATASETS = [
    "synthetic_coupon_profit_uplift_8k",
    "synthetic_recommendation_intervention_7k",
    "synthetic_marketplace_subsidy_uplift_7k",
    "synthetic_llm_routing_uplift_8k",
    "synthetic_llm_multi_action_routing_9k",
    "synthetic_baidu_waimai_red_packet_8k",
    "synthetic_didi_passenger_subsidy_8k",
    "synthetic_didi_driver_supply_subsidy_8k",
    "synthetic_didi_budget_allocation_5k",
    "synthetic_shopee_ads_full_funnel_8k",
    "synthetic_spotify_inapp_message_uplift_6k",
    "synthetic_doordash_ghost_ads_lift_6k",
    "synthetic_merchant_subsidy_margin_uplift_6k",
    "synthetic_ecommerce_multi_coupon_uplift_6k",
    "synthetic_crm_overlap_journey_uplift_6k",
    "synthetic_continuous_bid_discount_uplift_5k",
    "synthetic_geo_holdout_incrementality_uplift_6k",
    "synthetic_game_ops_gift_uplift_6k",
    "synthetic_fintech_credit_line_uplift_6k",
    "synthetic_customer_service_escalation_uplift_6k",
    "synthetic_healthcare_followup_uplift_6k",
    "synthetic_saas_retention_offer_uplift_6k",
]


def load_dataset(dataset_id: str, rows: int = 1200) -> tuple[dict, pd.DataFrame]:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for item in payload.get("datasets", []):
        if item.get("id") == dataset_id:
            return item, pd.read_csv(ROOT / item["path"]).head(rows).copy()
    raise ValueError(f"Dataset not found: {dataset_id}")


def attach_score(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    work = df.copy()
    base = pd.to_numeric(work["true_uplift"], errors="coerce").fillna(0.0)
    work["uplift_score"] = base + rng.normal(0, 0.012, size=len(work))
    outcome_mean = pd.to_numeric(work["outcome"], errors="coerce").fillna(0).mean()
    work["y0_pred"] = np.clip(outcome_mean - work["uplift_score"] / 2, 0, 1)
    work["y1_pred"] = np.clip(work["y0_pred"] + work["uplift_score"], 0, 1)
    return work


def main() -> None:
    failures: list[str] = []
    checks = []
    for index, dataset_id in enumerate(SCENARIO_DATASETS, start=1):
        meta, df = load_dataset(dataset_id)
        result = evaluate_uplift_predictions(
            attach_score(df, seed=100 + index),
            outcome_col=meta["outcome_col"],
            treatment_col=meta["treatment_col"],
            uplift_col="uplift_score",
            bins=8,
        )
        industrial = result["metrics"].get("industrial_policy") or {}
        top10 = industrial.get("top10") or []
        checks.append(
            {
                "dataset_id": dataset_id,
                "scenarios": industrial.get("scenarios"),
                "top_k_rows": len(industrial.get("top_k") or []),
                "top10_rows": len(top10),
                "top10_policy_value_sum": top10[0].get("policy_value_sum") if top10 else None,
                "top10_roi_proxy": top10[0].get("roi_proxy") if top10 else None,
            }
        )
        if not top10:
            failures.append(f"{dataset_id} did not emit industrial_policy top10")
        if top10 and top10[0].get("policy_value_sum") is None:
            failures.append(f"{dataset_id} missing policy value sum")

    output = ROOT / "reports" / "industrial_policy_metrics_smoke_latest.json"
    output.parent.mkdir(exist_ok=True)
    payload = {"status": "fail" if failures else "ok", "checks": checks, "failures": failures}
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
