from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from deepuplift.contracts import EffectPrediction, PolicyResult, TreatmentType


def build_continuous_policy(prediction: EffectPrediction, *, dose_cost: Mapping[float, float] | Callable[[float], float], outcome_value: float = 1.0, budget: float | None = None, no_treatment: float = 0.0, policy_name: str = "continuous_net_value_policy") -> PolicyResult:
    if prediction.treatment_type != TreatmentType.CONTINUOUS:
        raise ValueError("build_continuous_policy requires a continuous EffectPrediction.")
    cost_fn = dose_cost if callable(dose_cost) else lambda dose: float(dose_cost.get(float(dose), dose_cost.get(str(float(dose)), 0.0)))
    effects_by_dose = prediction.dose_effect_predictions
    if not effects_by_dose:
        raise ValueError("Continuous policy requires dose_effect_predictions for every dose in the prediction.")
    dose_values = [float(dose) for dose in (prediction.dose_grid if prediction.dose_grid is not None else effects_by_dose.keys())]
    candidates = []
    for index, unit in enumerate(prediction.unit_id):
        best = {"unit_id": unit, "dose": float(no_treatment), "effect": 0.0, "value": 0.0, "cost": 0.0, "net": 0.0}
        for dose in dose_values:
            effect = float(np.asarray(effects_by_dose[dose])[index])
            cost = 0.0 if dose == float(no_treatment) else float(cost_fn(dose))
            value = effect * float(outcome_value)
            net = value - cost
            if net > best["net"]:
                best = {"unit_id": unit, "dose": dose, "effect": effect, "value": value, "cost": cost, "net": net}
        candidates.append(best)
    remaining = float(budget) if budget is not None else None
    eligible_indices = set()
    for index, candidate in sorted(enumerate(candidates), key=lambda item: item[1]["net"], reverse=True):
        if candidate["net"] > 0 and (remaining is None or candidate["cost"] <= remaining):
            eligible_indices.add(index)
            if remaining is not None:
                remaining -= candidate["cost"]
    rows = []
    for index, candidate in enumerate(candidates):
        eligible = index in eligible_indices
        rows.append({"unit_id": candidate["unit_id"], "recommended_treatment": candidate["dose"] if eligible else no_treatment, "estimated_effect": candidate["effect"] if eligible else 0.0, "expected_incremental_value": candidate["value"] if eligible else 0.0, "treatment_cost": candidate["cost"] if eligible else 0.0, "net_value": candidate["net"] if eligible else 0.0, "eligible": eligible, "reason": "maximum positive dose net value" if eligible else "no positive net value or budget unavailable"})
    active = [row for row in rows if row["eligible"]]
    total_cost = sum(row["treatment_cost"] for row in active); value = sum(row["expected_incremental_value"] for row in active)
    return PolicyResult(rows=rows, summary={"target_count": len(active), "total_cost": total_cost, "expected_incremental_outcome": sum(row["estimated_effect"] for row in active), "expected_incremental_value": value, "expected_net_value": sum(row["net_value"] for row in active), "iroas": value / total_cost if total_cost else None, "budget": budget, "budget_utilization": total_cost / float(budget) if budget and budget > 0 else None}, policy_name=policy_name, metadata={"offline_only": True, "maturity": "EXPERIMENTAL", "status": "EXPERIMENTAL", "decision_rule": "for each user and dose: effect * outcome_value - dose_cost; include no-treatment"})


__all__ = ["build_continuous_policy"]
