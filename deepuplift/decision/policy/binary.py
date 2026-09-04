from __future__ import annotations

from typing import Any

import numpy as np

from deepuplift.contracts import EffectPrediction, PolicyResult, TreatmentType
from deepuplift.decision.economics.iroas import calculate_iroas


def build_binary_policy(
    prediction: EffectPrediction,
    *,
    treatment_cost: float,
    outcome_value: float,
    budget: float | None = None,
    max_contacts: int | None = None,
    min_net_value: float = 0.0,
    policy_name: str = "coupon_uplift_policy",
) -> PolicyResult:
    if prediction.treatment_type != TreatmentType.BINARY:
        raise ValueError("build_binary_policy requires a binary EffectPrediction.")
    frame = prediction.to_frame().sort_values("recommended_effect", ascending=False).reset_index(drop=True)
    frame["expected_incremental_value"] = frame["recommended_effect"] * float(outcome_value)
    frame["treatment_cost"] = float(treatment_cost)
    frame["net_value"] = frame["expected_incremental_value"] - float(treatment_cost)
    eligible = frame["net_value"] >= float(min_net_value)
    if budget is not None and treatment_cost > 0:
        eligible &= eligible.cumsum() * float(treatment_cost) <= float(budget)
    if max_contacts is not None:
        eligible &= eligible.cumsum() <= int(max_contacts)
    rows = []
    for _, row in frame.iterrows():
        selected = bool(eligible.loc[row.name])
        rows.append({
            "unit_id": row["unit_id"],
            "recommended_treatment": 1 if selected else 0,
            "estimated_effect": float(row["recommended_effect"]),
            "expected_incremental_value": float(row["expected_incremental_value"]) if selected else 0.0,
            "treatment_cost": float(treatment_cost) if selected else 0.0,
            "net_value": float(row["net_value"]) if selected else 0.0,
            "policy_score": float(row["net_value"]),
            "eligible": selected,
            "reason": "positive expected net value" if selected else "not selected by value or constraint",
        })
    selected_rows = [row for row in rows if row["recommended_treatment"] == 1]
    total_cost = sum(row["treatment_cost"] for row in selected_rows)
    total_value = sum(row["expected_incremental_value"] for row in selected_rows)
    summary = {
        "target_count": len(selected_rows),
        "total_cost": total_cost,
        "expected_incremental_outcome": sum(row["estimated_effect"] for row in selected_rows),
        "expected_incremental_value": total_value,
        "expected_net_value": sum(row["net_value"] for row in selected_rows),
        "iroas": calculate_iroas(total_value, total_cost),
        "budget": budget,
        "budget_utilization": total_cost / float(budget) if budget and budget > 0 else None,
    }
    return PolicyResult(rows=rows, summary=summary, policy_name=policy_name, metadata={"offline_estimate": True})
