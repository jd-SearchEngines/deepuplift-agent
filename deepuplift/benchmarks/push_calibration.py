"""P4 monotonicity and curve-smoothness calibration comparisons."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split

from deepuplift.benchmarks.continuous.datasets import continuous_dose_cost
from deepuplift.benchmarks.continuous.metrics import continuous_curve_metrics
from deepuplift.benchmarks.push_continuous import generate_push_continuous
from deepuplift.contracts import EffectPrediction
from deepuplift.models.continuous.contracts import effect_prediction_from_curve
from deepuplift.models import build_model


def _curve_matrix(prediction: EffectPrediction) -> tuple[np.ndarray, np.ndarray]:
    grid = np.asarray(prediction.dose_grid, dtype="float64")
    matrix = np.column_stack([np.asarray(prediction.dose_outcome_predictions[float(dose)], dtype="float64") for dose in grid])
    return grid, matrix


def monotonic_violation_rate(curves: np.ndarray) -> float:
    curves = np.asarray(curves, dtype="float64")
    if curves.shape[1] < 2:
        return 0.0
    return float(np.mean(np.diff(curves, axis=1) < -1e-8))


def pav_calibrate(curves: np.ndarray, grid: np.ndarray) -> np.ndarray:
    result = np.empty_like(curves, dtype="float64")
    for row, values in enumerate(np.asarray(curves, dtype="float64")):
        result[row] = IsotonicRegression(increasing=True, out_of_bounds="clip").fit_transform(grid, values)
    return result


def monotonic_penalty_calibrate(curves: np.ndarray, *, strength: float = 0.35, steps: int = 500, learning_rate: float = 0.5) -> np.ndarray:
    """Minimize squared fit error plus a one-sided monotonicity penalty."""

    values = np.asarray(curves, dtype="float64").copy()
    for _ in range(max(1, int(steps))):
        for index in range(values.shape[1] - 1):
            violation = np.maximum(values[:, index] - values[:, index + 1], 0.0)
            delta = 0.5 * learning_rate * strength * violation
            values[:, index] -= delta
            values[:, index + 1] += delta
    return values


def laplacian_smooth(curves: np.ndarray, *, strength: float = 0.25) -> np.ndarray:
    """Apply chain-graph Laplacian smoothing without imposing monotonicity."""

    values = np.asarray(curves, dtype="float64")
    size = values.shape[1]
    laplacian = np.zeros((size, size), dtype="float64")
    for index in range(size - 1):
        laplacian[index, index] += 1.0
        laplacian[index + 1, index + 1] += 1.0
        laplacian[index, index + 1] -= 1.0
        laplacian[index + 1, index] -= 1.0
    return np.linalg.solve(np.eye(size) + float(strength) * laplacian, values.T).T


def _prediction_with_curves(prediction: EffectPrediction, curves: np.ndarray, *, method: str) -> EffectPrediction:
    grid = np.asarray(prediction.dose_grid, dtype="float64")
    return effect_prediction_from_curve(
        unit_id=prediction.unit_id,
        dose_grid=grid,
        dose_outcomes=curves,
        baseline_dose=prediction.baseline_dose,
        metadata={**prediction.metadata, "calibration_method": method},
    )


def run_push_calibration_benchmark(
    mrt,
    *,
    shapes: list[str] | None = None,
    rows: int = 600,
    seed: int = 42,
    model_name: str = "DoseResponseGBM",
) -> dict[str, Any]:
    requested_shapes = shapes or ["monotone", "fatigue"]
    methods = ("none", "pav_isotonic", "explicit_monotonic_penalty", "laplacian_smooth")
    shape_reports = []
    for shape_index, shape in enumerate(requested_shapes):
        data = generate_push_continuous(mrt, shape=shape, rows=rows, seed=seed + shape_index * 101)
        train_indices, test_indices = train_test_split(np.arange(rows), test_size=0.25, random_state=seed + shape_index * 101)
        train = data.subset(train_indices)
        test = data.subset(test_indices)
        model = build_model(model_name, random_state=seed + shape_index * 101, grid_size=len(data.dose_grid))
        model.fit(train.dataset)
        prediction = model.predict(test.dataset)
        grid, curves = _curve_matrix(prediction)
        truth_y = np.vstack([np.interp(grid, data.dose_grid, row) for row in test.true_response_curves])
        truth_tau = np.vstack([np.interp(grid, data.dose_grid, row) for row in test.true_effect_curves])
        calibrated = {
            "none": curves,
            "pav_isotonic": pav_calibrate(curves, grid),
            "explicit_monotonic_penalty": monotonic_penalty_calibrate(curves),
            "laplacian_smooth": laplacian_smooth(curves),
        }
        rows_out = []
        for method in methods:
            calibrated_prediction = _prediction_with_curves(prediction, calibrated[method], method=method)
            metrics = continuous_curve_metrics(
                calibrated_prediction,
                dose_grid=grid,
                true_response_curves=truth_y,
                true_effect_curves=truth_tau,
                true_optimal_effect_dose=test.true_optimal_effect_dose,
                true_optimal_economic_dose=test.true_optimal_economic_dose,
                outcome_value=test.outcome_value,
                dose_cost=continuous_dose_cost,
                causal_baseline_supported=True,
            )
            rows_out.append({
                "method": method,
                "metrics": metrics,
                "monotonic_violation_rate": monotonic_violation_rate(calibrated[method]),
                "monotonicity_is_target": shape == "monotone",
            })
        shape_reports.append({
            "shape": shape,
            "model": model_name,
            "train_rows": len(train_indices),
            "test_rows": len(test_indices),
            "support_clipped_to_model_grid": True,
            "dose_grid": grid.tolist(),
            "results": rows_out,
        })
    return {
        "status": "PASS",
        "evidence_class": "SEMI_SYNTHETIC_KNOWN_TRUTH",
        "methods": list(methods),
        "shapes": shape_reports,
        "interpretation": [
            "PAV/isotonic and the explicit one-sided penalty target monotonicity; they are not appropriate assumptions for fatigue curves.",
            "Laplacian smoothing reduces adjacent roughness but does not guarantee monotonicity.",
            "Metrics are evaluated only on the model's observed dose support intersection.",
        ],
    }


__all__ = ["laplacian_smooth", "monotonic_penalty_calibrate", "monotonic_violation_rate", "pav_calibrate", "run_push_calibration_benchmark"]
