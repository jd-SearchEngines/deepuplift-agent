from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from deepuplift.contracts import EffectPrediction, TreatmentType


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


__all__ = ["effect_prediction_from_curve"]
