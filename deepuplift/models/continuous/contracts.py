from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from deepuplift.contracts import EffectPrediction, TreatmentType


EXTRAPOLATION_WARNING = (
    "Predictions outside observed treatment support are extrapolations and do not have empirical overlap support."
)


def resolve_dose_support(
    observed_min: float,
    observed_max: float,
    *,
    grid_size: int,
    baseline_dose: float | None = None,
    dose_grid: Any | None = None,
    allow_extrapolation: bool = False,
    no_treatment_observed: bool | None = None,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Resolve a prediction grid and reference baseline against factual support."""
    lower, upper = float(observed_min), float(observed_max)
    if not np.isfinite([lower, upper]).all() or upper <= lower:
        raise ValueError("Observed continuous treatment support must have finite, distinct endpoints.")
    zero_supported = (lower <= 0.0 <= upper) if no_treatment_observed is None else bool(no_treatment_observed)
    baseline = (0.0 if zero_supported else lower) if baseline_dose is None else float(baseline_dose)
    if not np.isfinite(baseline):
        raise ValueError("baseline_dose must be finite.")
    baseline_supported = lower <= baseline <= upper and (baseline != 0.0 or zero_supported)
    if not baseline_supported and not allow_extrapolation:
        if baseline == 0.0 and not zero_supported:
            detail = "dose 0 was not observed as a no-treatment baseline in the factual training rows."
        else:
            detail = f"it is outside observed treatment support [{lower:g}, {upper:g}]."
        raise ValueError(
            f"baseline_dose={baseline:g} is unsupported: {detail} "
            "Pass allow_extrapolation=True to explicitly request extrapolation."
        )
    if dose_grid is None:
        start, end = (lower, upper) if not allow_extrapolation else (min(lower, baseline), max(upper, baseline))
        grid = np.linspace(start, end, max(3, int(grid_size)))
        if not zero_supported:
            grid = grid[~np.isclose(grid, 0.0, atol=1e-12)]
        if start <= baseline <= end:
            grid = np.unique(np.append(grid, baseline))
    else:
        grid = np.asarray(dose_grid, dtype="float64").reshape(-1)
    if len(grid) == 0 or not np.isfinite(grid).all() or (len(grid) > 1 and np.any(np.diff(grid) <= 0)):
        raise ValueError("dose_grid must be finite and strictly increasing.")
    unsupported_mask = (grid < lower) | (grid > upper) | (np.isclose(grid, 0.0, atol=1e-12) & (not zero_supported))
    unsupported = grid[unsupported_mask]
    if len(unsupported) and not allow_extrapolation:
        raise ValueError(
            "dose_grid contains doses without empirical training support "
            f"(observed interval [{lower:g}, {upper:g}], no_treatment_supported={zero_supported}). "
            "Pass allow_extrapolation=True to explicitly request extrapolation."
        )
    support = {
        "observed_min": lower,
        "observed_max": upper,
        "grid_min": float(grid.min()),
        "grid_max": float(grid.max()),
        "contains_extrapolation": bool(len(unsupported) or not baseline_supported),
        "baseline_supported": bool(baseline_supported),
        "no_treatment_supported": bool(zero_supported),
        "extrapolation_enabled": bool(allow_extrapolation),
        "unsupported_doses": sorted(set(float(x) for x in unsupported) | ({baseline} if not baseline_supported else set())),
    }
    if support["contains_extrapolation"]:
        support["warning"] = EXTRAPOLATION_WARNING
    return grid, baseline, support


def effect_prediction_from_curve(
    *,
    unit_id: Any,
    dose_grid: Any,
    dose_outcomes: Any,
    baseline_dose: float = 0.0,
    metadata: Mapping[str, Any] | None = None,
    uncertainty: Any | None = None,
) -> EffectPrediction:
    """Convert an ``(n_units, n_doses)`` outcome surface to EffectPrediction.

    ``recommended_treatment`` is the per-unit maximum-effect dose for
    compatibility and diagnostics only. Economic selection remains owned by
    the continuous decision layer.
    """
    grid = np.asarray(dose_grid, dtype="float64").reshape(-1)
    outcomes = np.asarray(dose_outcomes, dtype="float64")
    ids = np.asarray(unit_id)
    if outcomes.ndim != 2 or outcomes.shape != (len(ids), len(grid)):
        raise ValueError("dose_outcomes must have shape (number of units, number of doses).")
    if len(grid) == 0 or not np.isfinite(grid).all() or (len(grid) > 1 and np.any(np.diff(grid) <= 0)):
        raise ValueError("dose_grid must be finite and strictly increasing.")
    if not np.isfinite(outcomes).all():
        raise ValueError("dose_outcomes must contain only finite values.")
    baseline_idx = int(np.argmin(np.abs(grid - float(baseline_dose))))
    baseline = outcomes[:, baseline_idx]
    effects = outcomes - baseline[:, None]
    best = effects.argmax(axis=1)
    rows = np.arange(len(ids))
    meta = dict(metadata or {})
    meta.setdefault("baseline_dose", float(grid[baseline_idx]))
    meta.setdefault("max_effect_dose_mean", float(np.mean(grid[best])))
    meta.setdefault("recommendation_type", "maximum_effect_reference_only")
    outcomes_by_dose = {float(dose): outcomes[:, i] for i, dose in enumerate(grid)}
    effects_by_dose = {float(dose): effects[:, i] for i, dose in enumerate(grid)}
    uncertainty_by_dose = {}
    if uncertainty is not None:
        uncertainty_values = np.asarray(uncertainty, dtype="float64")
        if uncertainty_values.shape != outcomes.shape or not np.isfinite(uncertainty_values).all():
            raise ValueError("uncertainty must be finite and match dose_outcomes shape.")
        uncertainty_by_dose = {float(dose): uncertainty_values[:, i] for i, dose in enumerate(grid)}
    return EffectPrediction(
        unit_id=ids,
        treatment_type=TreatmentType.CONTINUOUS,
        y0=baseline,
        y1=outcomes[rows, best],
        uplift=effects[rows, best],
        recommended_treatment=grid[best],
        recommended_effect=effects[rows, best],
        dose_grid=grid,
        baseline_dose=float(grid[baseline_idx]),
        dose_outcome_predictions=outcomes_by_dose,
        dose_effect_predictions=effects_by_dose,
        uncertainty_by_dose=uncertainty_by_dose,
        metadata=meta,
    )


__all__ = ["EXTRAPOLATION_WARNING", "resolve_dose_support", "effect_prediction_from_curve"]
