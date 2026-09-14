from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from deepuplift.contracts import EffectPrediction, PolicyResult, TreatmentType


def build_continuous_policy(
    prediction: EffectPrediction,
    *,
    dose_cost: Mapping[float, float] | Callable[[float], float],
    outcome_value: float = 1.0,
    budget: float | None = None,
    no_treatment: float = 0.0,
    policy_name: str = "continuous_net_value_policy",
    respect_observed_support: bool = True,
) -> PolicyResult:
    """Select the maximum positive net-value dose within empirical support."""
    if prediction.treatment_type != TreatmentType.CONTINUOUS:
        raise ValueError("build_continuous_policy requires a continuous EffectPrediction.")
    cost_fn = dose_cost if callable(dose_cost) else lambda dose: float(dose_cost.get(float(dose), dose_cost.get(str(float(dose)), 0.0)))
    effects_by_dose = prediction.dose_effect_predictions
    if not effects_by_dose:
        raise ValueError("Continuous policy requires dose_effect_predictions for every dose in the prediction.")
    dose_values = [float(dose) for dose in (prediction.dose_grid if prediction.dose_grid is not None else effects_by_dose.keys())]
    metadata = prediction.metadata or {}
    support = metadata.get("dose_support")
    if support and not respect_observed_support and not support.get("extrapolation_enabled", False):
        raise ValueError("Disabling observed-support filtering requires model predictions made with allow_extrapolation=True.")
    filtered_doses = []
    if support and respect_observed_support:
        lo, hi = float(support["observed_min"]), float(support["observed_max"])
        valid = {d for d in dose_values if lo <= d <= hi}
        filtered_doses.extend(d for d in dose_values if d not in valid)
        dose_values = [d for d in dose_values if d in valid]
        if (float(no_treatment) == 0.0 and not support.get("no_treatment_supported", False)) or no_treatment < lo or no_treatment > hi:
            baseline_matches_no_treatment = (
                prediction.baseline_dose is not None
                and float(prediction.baseline_dose) == float(no_treatment)
            )
            extrapolation_authorized = bool(support.get("extrapolation_enabled", False))
            unsupported_no_treatment_baseline = not (baseline_matches_no_treatment and extrapolation_authorized)
        else:
            unsupported_no_treatment_baseline = False
    else:
        unsupported_no_treatment_baseline = bool(
            support and (no_treatment < float(support["observed_min"]) or no_treatment > float(support["observed_max"]))
            and not support.get("extrapolation_enabled", False)
        )
    local = metadata.get("local_treatment_support")
    local_mask = None
    if local and local.get("dose_grid") == dose_values:
        local_mask = np.asarray(local.get("supported_dose_mask", []), dtype=bool)
    elif local:
        local_grid = [float(d) for d in local.get("dose_grid", [])]
        raw_mask = np.asarray(local.get("supported_dose_mask", []), dtype=bool)
        local_mask = np.zeros((len(prediction.unit_id), len(dose_values)), dtype=bool)
        for target_idx, dose in enumerate(dose_values):
            matches = [i for i, item in enumerate(local_grid) if item == dose]
            if matches and raw_mask.ndim == 2 and raw_mask.shape[0] == len(prediction.unit_id):
                local_mask[:, target_idx] = raw_mask[:, matches[0]]

    candidates = []
    reasons = []
    for index, unit in enumerate(prediction.unit_id):
        best = {"unit_id": unit, "dose": float(no_treatment), "effect": 0.0, "value": 0.0, "cost": 0.0, "net": 0.0}
        if unsupported_no_treatment_baseline:
            candidates.append(best)
            reasons.append("unsupported_no_treatment_baseline")
            continue
        has_local_candidate = False
        for dose_index, dose in enumerate(dose_values):
            if local_mask is not None and (local_mask.ndim != 2 or not local_mask[index, dose_index]):
                continue
            has_local_candidate = True
            effect = float(np.asarray(effects_by_dose[dose])[index])
            cost = 0.0 if dose == float(no_treatment) else float(cost_fn(dose))
            value = effect * float(outcome_value)
            net = value - cost
            if net > best["net"]:
                best = {"unit_id": unit, "dose": dose, "effect": effect, "value": value, "cost": cost, "net": net}
        candidates.append(best)
        if local_mask is not None and not has_local_candidate:
            reasons.append("outside_local_treatment_support")
        elif not dose_values and filtered_doses:
            reasons.append("outside_observed_treatment_support")
        else:
            reasons.append("no positive net value or budget unavailable")
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
        rows.append({
            "unit_id": candidate["unit_id"],
            "recommended_treatment": candidate["dose"] if eligible else no_treatment,
            "estimated_effect": candidate["effect"] if eligible else 0.0,
            "expected_incremental_value": candidate["value"] if eligible else 0.0,
            "treatment_cost": candidate["cost"] if eligible else 0.0,
            "net_value": candidate["net"] if eligible else 0.0,
            "eligible": eligible,
            "reason": "maximum positive dose net value" if eligible else reasons[index],
        })
    active = [row for row in rows if row["eligible"]]
    total_cost = sum(row["treatment_cost"] for row in active)
    value = sum(row["expected_incremental_value"] for row in active)
    policy_metadata = {
        "offline_only": True,
        "maturity": "EXPERIMENTAL",
        "status": "EXPERIMENTAL",
        "decision_rule": "for each user and supported dose: effect * outcome_value - dose_cost; include no-treatment",
        "respect_observed_support": bool(respect_observed_support),
        "filtered_unsupported_doses": sorted(set(filtered_doses)),
        "unsupported_no_treatment_baseline": unsupported_no_treatment_baseline,
        "dose_support": support,
        "extrapolation_enabled": bool(support and support.get("extrapolation_enabled", False)),
        "local_support_diagnostic": local.get("diagnostic_name") if local else None,
    }
    if support and support.get("warning"):
        policy_metadata["warning"] = support["warning"]
    return PolicyResult(
        rows=rows,
        summary={
            "target_count": len(active), "total_cost": total_cost,
            "expected_incremental_outcome": sum(row["estimated_effect"] for row in active),
            "expected_incremental_value": value,
            "expected_net_value": sum(row["net_value"] for row in active),
            "iroas": value / total_cost if total_cost else None,
            "budget": budget,
            "budget_utilization": total_cost / float(budget) if budget and budget > 0 else None,
        },
        policy_name=policy_name,
        metadata=policy_metadata,
    )


__all__ = ["build_continuous_policy"]
