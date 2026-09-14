from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from deepuplift.contracts import EffectPrediction, TreatmentType


def _matrix_on_grid(prediction: EffectPrediction, dose_grid: Any, *, effects: bool = False) -> np.ndarray:
    if prediction.treatment_type != TreatmentType.CONTINUOUS:
        raise ValueError("Continuous metrics require a continuous EffectPrediction.")
    mapping = prediction.dose_effect_predictions if effects else prediction.dose_outcome_predictions
    if not mapping:
        raise ValueError("Continuous metrics require a full per-dose prediction curve.")
    source_dose = np.asarray(sorted(float(x) for x in mapping), dtype="float64")
    source_values = np.column_stack([np.asarray(mapping[float(dose)], dtype="float64") for dose in source_dose])
    target = np.asarray(dose_grid, dtype="float64").reshape(-1)
    return np.vstack([np.interp(target, source_dose, row) for row in source_values])


def continuous_curve_metrics(
    prediction: EffectPrediction,
    *,
    dose_grid: Any,
    true_response_curves: Any,
    true_effect_curves: Any,
    true_optimal_effect_dose: Any,
    outcome_value: float = 1.0,
    dose_cost: Callable[[Any], Any] | None = None,
    true_optimal_economic_dose: Any | None = None,
) -> dict[str, float | None]:
    """Counterfactual curve and policy metrics on one shared dose grid."""
    grid = np.asarray(dose_grid, dtype="float64").reshape(-1)
    truth_y = np.asarray(true_response_curves, dtype="float64")
    truth_tau = np.asarray(true_effect_curves, dtype="float64")
    truth_opt = np.asarray(true_optimal_effect_dose, dtype="float64").reshape(-1)
    pred_y = _matrix_on_grid(prediction, grid)
    pred_tau = _matrix_on_grid(prediction, grid, effects=True)
    if pred_y.shape != truth_y.shape or pred_tau.shape != truth_tau.shape or len(truth_opt) != len(pred_y):
        raise ValueError("Prediction and ground-truth curve shapes must agree over units and dose grid.")
    interval = float(grid[-1] - grid[0])
    squared_error = (pred_y - truth_y) ** 2
    absolute_error = np.abs(pred_y - truth_y)
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    mise = float(np.mean(integrate(squared_error, x=grid, axis=1) / interval)) if interval > 0 else float(np.mean(squared_error))
    iae = float(np.mean(integrate(absolute_error, x=grid, axis=1) / interval)) if interval > 0 else float(np.mean(absolute_error))
    predicted_opt = grid[np.argmax(pred_tau, axis=1)]
    opt_positions = np.abs(grid[None, :] - predicted_opt[:, None]).argmin(axis=1)
    rows = np.arange(len(pred_y))
    best_effect = truth_tau.max(axis=1)
    selected_effect = truth_tau[rows, opt_positions]
    costs = np.zeros_like(grid) if dose_cost is None else np.asarray(dose_cost(grid), dtype="float64")
    predicted_net = outcome_value * pred_tau - costs[None, :]
    economic_positions = np.argmax(predicted_net, axis=1)
    true_net = outcome_value * truth_tau - costs[None, :]
    true_net[:, np.abs(grid).argmin()] = np.maximum(true_net[:, np.abs(grid).argmin()], 0.0)
    selected_economic_net = true_net[rows, economic_positions]
    optimal_economic_net = true_net.max(axis=1)
    economic_regret = optimal_economic_net - selected_economic_net
    economic_dose_error = None
    if true_optimal_economic_dose is not None:
        truth_economic_dose = np.asarray(true_optimal_economic_dose, dtype="float64").reshape(-1)
        if len(truth_economic_dose) != len(pred_y):
            raise ValueError("true_optimal_economic_dose must contain one value per unit.")
        economic_dose_error = float(np.mean(np.abs(grid[economic_positions] - truth_economic_dose)))
    true_adrf = truth_y.mean(axis=0)
    pred_adrf = pred_y.mean(axis=0)
    return {
        "mise": mise,
        "dose_response_rmse": float(np.sqrt(np.mean(squared_error))),
        "integrated_absolute_error": iae,
        "cate_icte_rmse": float(np.sqrt(np.mean((pred_tau - truth_tau) ** 2))),
        "adrf_rmse": float(np.sqrt(np.mean((pred_adrf - true_adrf) ** 2))),
        "optimal_dose_error": float(np.mean(np.abs(predicted_opt - truth_opt))),
        "optimal_dose_regret": float(np.mean(best_effect - selected_effect)),
        "economic_optimal_dose_error": economic_dose_error,
        "economic_policy_regret": float(np.mean(economic_regret)),
        "predicted_max_effect_dose_mean": float(np.mean(predicted_opt)),
        "predicted_max_net_value_dose_mean": float(np.mean(grid[economic_positions])),
        "true_max_effect_dose_mean": float(np.mean(truth_opt)),
        "true_max_net_value_dose_mean": float(np.mean(true_optimal_economic_dose)) if true_optimal_economic_dose is not None else None,
    }


__all__ = ["continuous_curve_metrics"]
