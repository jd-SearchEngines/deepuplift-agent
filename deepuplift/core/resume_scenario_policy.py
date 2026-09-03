from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .industrial_scenario_lab import SCENARIO_CARDS


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "examples" / "datasets" / "manifest.json"

RESUME_SCENARIO_IDS = [
    "synthetic_baidu_waimai_red_packet_8k",
    "synthetic_didi_passenger_subsidy_8k",
    "synthetic_didi_driver_supply_subsidy_8k",
    "synthetic_didi_budget_allocation_5k",
    "synthetic_shopee_ads_full_funnel_8k",
]

SCENARIO_COST_COL = {
    "synthetic_baidu_waimai_red_packet_8k": "expected_red_packet_cost",
    "synthetic_didi_passenger_subsidy_8k": "expected_subsidy_cost",
    "synthetic_didi_driver_supply_subsidy_8k": "driver_subsidy_cost",
    "synthetic_didi_budget_allocation_5k": "allocated_budget",
    "synthetic_shopee_ads_full_funnel_8k": "ad_cost",
}

SCENARIO_VALUE_COL = {
    "synthetic_baidu_waimai_red_packet_8k": "incremental_profit",
    "synthetic_didi_passenger_subsidy_8k": "completed_order_value",
    "synthetic_didi_driver_supply_subsidy_8k": "wait_time_reduction_value",
    "synthetic_didi_budget_allocation_5k": "city_incremental_profit",
    "synthetic_shopee_ads_full_funnel_8k": "incremental_gmv",
}

SCENARIO_RISK_COLS = {
    "synthetic_baidu_waimai_red_packet_8k": ["subsidy_abuse_risk"],
    "synthetic_didi_passenger_subsidy_8k": ["subsidy_abuse_risk", "price_sensitivity_drift"],
    "synthetic_didi_driver_supply_subsidy_8k": ["spillover_risk", "interference_risk"],
    "synthetic_didi_budget_allocation_5k": ["cross_market_spillover_risk"],
    "synthetic_shopee_ads_full_funnel_8k": [],
}


def _manifest_rows() -> dict[str, dict[str, Any]]:
    if not MANIFEST.exists():
        return {}
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {row.get("id"): row for row in payload.get("datasets", [])}


def _load_dataset(dataset_id: str) -> tuple[dict[str, Any], pd.DataFrame]:
    rows = _manifest_rows()
    meta = rows.get(dataset_id)
    if not meta:
        raise KeyError(f"Dataset not found in manifest: {dataset_id}")
    path = ROOT / str(meta["path"])
    return meta, pd.read_csv(path)


def _clean_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(number) or np.isinf(number):
        return None
    return number


def resume_scenario_policy_benchmarks(fractions: Iterable[float] = (0.05, 0.10, 0.20)) -> list[dict[str, Any]]:
    cards = {row["dataset_id"]: row for row in SCENARIO_CARDS}
    rows: list[dict[str, Any]] = []
    for dataset_id in RESUME_SCENARIO_IDS:
        try:
            meta, df = _load_dataset(dataset_id)
        except Exception:
            continue
        if "oracle_policy_value" not in df.columns:
            continue
        work = df.copy()
        work["_score"] = pd.to_numeric(work["oracle_policy_value"], errors="coerce")
        work = work.dropna(subset=["_score"]).sort_values("_score", ascending=False).reset_index(drop=True)
        if work.empty:
            continue
        cost_col = SCENARIO_COST_COL.get(dataset_id)
        value_col = SCENARIO_VALUE_COL.get(dataset_id)
        for fraction in fractions:
            n = max(1, int(np.ceil(len(work) * float(fraction))))
            top = work.head(n)
            policy_sum = _clean_float(top["oracle_policy_value"].sum())
            cost_sum = _clean_float(pd.to_numeric(top[cost_col], errors="coerce").sum()) if cost_col in top.columns else None
            value_sum = _clean_float(pd.to_numeric(top[value_col], errors="coerce").sum()) if value_col in top.columns else None
            row = {
                "scenario": cards.get(dataset_id, {}).get("scenario", meta.get("name")),
                "dataset_id": dataset_id,
                "top_fraction": float(fraction),
                "rows": int(n),
                "policy_value_sum": policy_sum,
                "cost_sum": cost_sum,
                "value_sum": value_sum,
                "roi_proxy": _clean_float(policy_sum / cost_sum) if cost_sum and abs(cost_sum) > 1e-12 else None,
                "threshold_oracle_policy_value": _clean_float(top["_score"].min()),
                "mean_true_uplift": _clean_float(pd.to_numeric(top.get("true_uplift"), errors="coerce").mean()) if "true_uplift" in top.columns else None,
                "recommended_models": "; ".join(meta.get("recommended_models") or []),
            }
            for risk_col in SCENARIO_RISK_COLS.get(dataset_id, []):
                if risk_col in top.columns:
                    row[f"mean_{risk_col}"] = _clean_float(pd.to_numeric(top[risk_col], errors="coerce").mean())
            rows.append(row)
    return rows


def didi_budget_allocation_plan(total_budget: float = 100_000.0, max_cells: int = 40) -> dict[str, Any]:
    dataset_id = "synthetic_didi_budget_allocation_5k"
    meta, df = _load_dataset(dataset_id)
    work = df.copy()
    work["allocated_budget"] = pd.to_numeric(work["allocated_budget"], errors="coerce").fillna(0.0)
    work["budget_policy_value"] = pd.to_numeric(work["budget_policy_value"], errors="coerce").fillna(0.0)
    work["marginal_roi"] = pd.to_numeric(work["marginal_roi"], errors="coerce").fillna(0.0)
    work["score"] = work["budget_policy_value"] / work["allocated_budget"].clip(lower=1e-6)
    work = work[(work["allocated_budget"] > 0) & (work["budget_policy_value"] > 0)].sort_values(
        ["score", "marginal_roi", "budget_policy_value"], ascending=False
    )
    selected = []
    used_budget = 0.0
    for _, row in work.iterrows():
        cost = float(row["allocated_budget"])
        if used_budget + cost > float(total_budget):
            continue
        selected.append(row)
        used_budget += cost
        if len(selected) >= max_cells:
            break
    if selected:
        table = pd.DataFrame(selected)
    else:
        table = work.head(0)
    keep_cols = [
        "city_tier",
        "city_cluster",
        "hour_bucket",
        "allocated_budget",
        "budget_share",
        "c_side_share",
        "b_side_share",
        "marginal_roi",
        "city_incremental_profit",
        "cross_market_spillover_risk",
        "budget_policy_value",
        "score",
    ]
    visible = table[[col for col in keep_cols if col in table.columns]].copy()
    summary = {
        "dataset_id": dataset_id,
        "dataset_name": meta.get("name"),
        "total_budget": float(total_budget),
        "selected_cells": int(len(visible)),
        "used_budget": _clean_float(used_budget),
        "remaining_budget": _clean_float(float(total_budget) - used_budget),
        "expected_policy_value": _clean_float(pd.to_numeric(visible.get("budget_policy_value"), errors="coerce").sum()) if not visible.empty else 0.0,
        "expected_city_incremental_profit": _clean_float(pd.to_numeric(visible.get("city_incremental_profit"), errors="coerce").sum()) if not visible.empty else 0.0,
        "mean_marginal_roi": _clean_float(pd.to_numeric(visible.get("marginal_roi"), errors="coerce").mean()) if not visible.empty else None,
        "mean_spillover_risk": _clean_float(pd.to_numeric(visible.get("cross_market_spillover_risk"), errors="coerce").mean()) if not visible.empty else None,
    }
    return {"summary": summary, "allocations": visible.head(max_cells).to_dict(orient="records")}
