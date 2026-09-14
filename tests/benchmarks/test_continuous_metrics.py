import numpy as np

from deepuplift.benchmarks.continuous.metrics import continuous_curve_metrics
from deepuplift.models.continuous.contracts import effect_prediction_from_curve


def test_continuous_metrics_recover_exact_curves_and_separate_economic_dose():
    grid = np.array([0.0, 8.0, 20.0])
    truth_y = np.array([[0.0, 5.0, 7.0], [0.0, 4.0, 6.0]])
    truth_tau = truth_y - truth_y[:, :1]
    prediction = effect_prediction_from_curve(unit_id=["u1", "u2"], dose_grid=grid, dose_outcomes=truth_y)
    metrics = continuous_curve_metrics(
        prediction,
        dose_grid=grid,
        true_response_curves=truth_y,
        true_effect_curves=truth_tau,
        true_optimal_effect_dose=[20.0, 20.0],
        true_optimal_economic_dose=[8.0, 8.0],
        dose_cost=lambda dose: np.array([0.0, 1.0, 10.0]),
    )
    assert metrics["mise"] == 0.0
    assert metrics["optimal_dose_error"] == 0.0
    assert metrics["predicted_max_effect_dose_mean"] == 20.0
    assert metrics["predicted_max_net_value_dose_mean"] == 8.0
    assert metrics["economic_policy_regret"] == 0.0
