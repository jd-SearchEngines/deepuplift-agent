from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from deepuplift.contracts import EffectPrediction, PolicyResult, TreatmentType


def build_multi_policy(
    prediction: EffectPrediction,
    *,
    treatment_costs: Mapping[Any, float] | Callable[[Any], float],
    outcome_value: float = 1.0,
    budget: float | None = None,
    no_treatment: Any = "NO_TREATMENT",
    optimizer: str = "net_value",
    policy_name: str = "multi_net_value_policy",
) -> PolicyResult:
    """Choose the action with maximum incremental net value, including no-op."""
    if prediction.treatment_type != TreatmentType.MULTI_DISCRETE:
        raise ValueError("build_multi_policy requires a multi_discrete EffectPrediction.")
    if optimizer not in {"net_value", "value_per_cost"}:
        raise ValueError("optimizer must be 'net_value' or 'value_per_cost'.")
    effects = {key: np.asarray(value, dtype="float64") for key, value in prediction.treatment_effects.items()}
    n = len(np.asarray(prediction.unit_id))
    if not effects:
        return PolicyResult(rows=[{"unit_id": unit, "recommended_treatment": no_treatment, "net_value": 0.0, "eligible": False} for unit in prediction.unit_id], summary={"target_count": 0, "total_cost": 0.0, "expected_incremental_outcome": 0.0, "expected_incremental_value": 0.0, "expected_net_value": 0.0, "budget": budget, "budget_utilization": 0.0}, policy_name=policy_name, metadata={"offline_only": True})
    cost_fn = treatment_costs if callable(treatment_costs) else lambda treatment: float(treatment_costs.get(treatment, treatment_costs.get(str(treatment), 0.0)))
    candidates: dict[Any, np.ndarray] = {key: effect * float(outcome_value) - float(cost_fn(key)) for key, effect in effects.items()}
    selected = []
    for index in range(n):
        best_treatment = no_treatment
        best_net = 0.0
        for treatment, values in candidates.items():
            if values[index] > best_net:
                best_treatment, best_net = treatment, float(values[index])
        cost = float(cost_fn(best_treatment)) if best_treatment != no_treatment else 0.0
        selected.append({"index": index, "treatment": best_treatment, "net_value": best_net, "cost": cost, "effect": float(effects[best_treatment][index]) if best_treatment != no_treatment else 0.0})
    if optimizer == "value_per_cost":
        selected.sort(key=lambda row: row["net_value"] / row["cost"] if row["cost"] > 0 else -np.inf, reverse=True)
    else:
        selected.sort(key=lambda row: row["net_value"], reverse=True)
    spent = 0.0
    chosen: dict[int, bool] = {}
    for row in selected:
        allowed = row["net_value"] > 0 and (budget is None or spent + row["cost"] <= float(budget))
        chosen[row["index"]] = allowed
        if allowed:
            spent += row["cost"]
    rows = []
    for row in sorted(selected, key=lambda item: item["index"]):
        eligible = chosen[row["index"]]
        treatment = row["treatment"] if eligible else no_treatment
        rows.append({"unit_id": prediction.unit_id[row["index"]], "recommended_treatment": treatment, "estimated_effect": row["effect"] if eligible else 0.0, "expected_incremental_value": row["effect"] * float(outcome_value) if eligible else 0.0, "treatment_cost": row["cost"] if eligible else 0.0, "net_value": row["net_value"] if eligible else 0.0, "eligible": eligible, "reason": "maximum positive net value" if eligible else "no positive net value or budget unavailable"})
    active = [row for row in rows if row["eligible"]]
    value = sum(row["expected_incremental_value"] for row in active)
    return PolicyResult(rows=rows, summary={"target_count": len(active), "total_cost": spent, "expected_incremental_outcome": sum(row["estimated_effect"] for row in active), "expected_incremental_value": value, "expected_net_value": sum(row["net_value"] for row in active), "iroas": value / spent if spent > 0 else None, "budget": budget, "budget_utilization": spent / float(budget) if budget and budget > 0 else None}, policy_name=policy_name, metadata={"offline_only": True, "decision_rule": f"{optimizer}: action-specific net value including no-treatment"})


__all__ = ["build_multi_policy"]
