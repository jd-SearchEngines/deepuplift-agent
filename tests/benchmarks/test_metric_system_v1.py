from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import json

from deepuplift.contracts import CausalDataset, EffectPrediction, TreatmentType
from deepuplift.decision.evaluation import fixed_budget_policy_value, policy_value_contributions, ranking_metrics
from deepuplift.decision.evaluation.ranking import qini_curve
from deepuplift.benchmarks.continuous.metrics import continuous_curve_metrics
from deepuplift.models.continuous.contracts import effect_prediction_from_curve


def _binary_fixture(seed: int) -> tuple[EffectPrediction, CausalDataset]:
    rng = np.random.default_rng(seed)
    rows = 80
    treatment = np.arange(rows) % 2
    uplift = np.linspace(1.0, -1.0, rows) + rng.normal(0.0, 0.01, rows)
    outcome = (0.5 + treatment * uplift + rng.normal(0.0, 0.05, rows) > 0.5).astype(float)
    frame = pd.DataFrame({"unit_id": np.arange(rows), "x": uplift, "treatment": treatment, "outcome": outcome})
    dataset = CausalDataset(frame, feature_cols=["x"], treatment_col="treatment", outcome_col="outcome", id_column="unit_id")
    return EffectPrediction(unit_id=frame["unit_id"], treatment_type=TreatmentType.BINARY, uplift=uplift), dataset


def test_qini_and_auuc_are_distinct_and_bootstrap_is_seeded():
    prediction, dataset = _binary_fixture(0)
    metrics = ranking_metrics(prediction, dataset, include_curves=False, bootstrap_samples=32, random_state=17)

    assert metrics["definitions"]["qini"] != metrics["definitions"]["auuc"]
    assert metrics["qini"] == metrics["qini_raw_area"] - metrics["random_baseline_area"]
    assert metrics["bootstrap_stability"]["n"] > 0
    assert metrics["bootstrap_stability"]["qini_ci_95"] == ranking_metrics(
        prediction, dataset, include_curves=False, bootstrap_samples=32, random_state=17
    )["bootstrap_stability"]["qini_ci_95"]


def test_ranking_curve_excludes_prefixes_without_both_treatment_arms():
    curve = qini_curve(
        pd.DataFrame(
            {
                "uplift": [4.0, 3.0, 2.0, 1.0],
                "treatment": [1, 1, 0, 0],
                "outcome": [1.0, 1.0, 0.0, 0.0],
            }
        )
    )
    assert len(curve) == 2
    assert np.isfinite(curve[["qini", "uplift", "auuc_gain"]].to_numpy()).all()


def test_auuc_is_unavailable_with_only_one_identifiable_prefix():
    frame = pd.DataFrame({"unit_id": [0, 1], "x": [1.0, 0.0], "treatment": [1, 0], "outcome": [1.0, 0.0]})
    dataset = CausalDataset(frame, feature_cols=["x"], treatment_col="treatment", outcome_col="outcome", id_column="unit_id")
    prediction = EffectPrediction(unit_id=frame["unit_id"], treatment_type=TreatmentType.BINARY, uplift=frame["x"])
    assert ranking_metrics(prediction, dataset, include_curves=False, bootstrap_samples=0)["auuc"] is None


def test_fixed_budget_policy_value_uses_topk_and_original_row_bootstrap():
    result = fixed_budget_policy_value(
        scores=[0.2, 0.9, 0.5, 0.8],
        treatment=[1, 1, 0, 0],
        outcome=[1, 0, 1, 0],
        propensity=0.5,
        budget=2.0,
        treatment_cost=1.0,
        bootstrap_samples=64,
        random_state=3,
    )

    assert result["selected_indices"] == [1, 3]
    json.dumps(result)
    assert result["target_count"] == 2
    assert result["bootstrap"]["samples"] == 64
    assert result["policy_value_ci"]["low"] <= result["policy_value"] <= result["policy_value_ci"]["high"]


def test_fixed_budget_handles_decimal_capacity_and_rejects_nonfinite_propensity():
    result = fixed_budget_policy_value(
        scores=[0.4, 0.3, 0.2],
        treatment=[1, 0, 1],
        outcome=[1, 1, 0],
        propensity=0.5,
        budget=0.3,
        treatment_cost=0.1,
        bootstrap_samples=8,
    )
    assert result["target_count"] == 3
    with pytest.raises(ValueError, match="propensity"):
        fixed_budget_policy_value(
            scores=[0.4, 0.3],
            treatment=[1, 0],
            outcome=[1, 0],
            propensity=[0.5, np.nan],
            budget=1.0,
            treatment_cost=1.0,
            bootstrap_samples=8,
        )
    with pytest.raises(ValueError, match="binary"):
        fixed_budget_policy_value(
            scores=[0.4, 0.3],
            treatment=[1, 2],
            outcome=[1, 0],
            propensity=0.5,
            budget=1.0,
            treatment_cost=1.0,
            bootstrap_samples=8,
        )
    with pytest.raises(ValueError, match="budget"):
        fixed_budget_policy_value(
            scores=[0.4, 0.3],
            treatment=[1, 0],
            outcome=[1, 0],
            propensity=0.5,
            budget=np.inf,
            treatment_cost=1.0,
            bootstrap_samples=8,
        )
    with pytest.raises(ValueError, match="outcome_value"):
        fixed_budget_policy_value(
            scores=[0.4, 0.3],
            treatment=[1, 0],
            outcome=[1, 0],
            propensity=0.5,
            budget=1.0,
            treatment_cost=1.0,
            outcome_value=np.nan,
            bootstrap_samples=8,
        )
    with pytest.raises(ValueError, match="policy_treatment"):
        policy_value_contributions([1, 0], [1, 0], [2, 0], 0.5)


def test_three_seed_metric_smoke_keeps_estimator_and_uncertainty_fields():
    for seed in (0, 1, 2):
        prediction, dataset = _binary_fixture(seed)
        metrics = ranking_metrics(prediction, dataset, include_curves=False, bootstrap_samples=16, random_state=seed)
        assert metrics["definitions"]["qini"]
        assert metrics["bootstrap_stability"]["random_state"] == seed


def test_continuous_metrics_separate_response_and_effect_mise():
    grid = np.array([0.0, 1.0, 2.0])
    truth_y = np.array([[0.0, 1.0, 1.5], [0.0, 2.0, 2.5]])
    truth_tau = truth_y - truth_y[:, :1]
    prediction = effect_prediction_from_curve(unit_id=["a", "b"], dose_grid=grid, dose_outcomes=truth_y)
    metrics = continuous_curve_metrics(
        prediction,
        dose_grid=grid,
        true_response_curves=truth_y,
        true_effect_curves=truth_tau,
        true_optimal_effect_dose=[2.0, 2.0],
    )
    assert metrics["response_mise"] == 0.0
    assert metrics["effect_mise"] == 0.0
