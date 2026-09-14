import numpy as np
import pandas as pd
import pytest

from deepuplift.contracts import TreatmentType
from deepuplift.data.schema import create_causal_dataset
from deepuplift.decision import build_continuous_policy
from deepuplift.models.continuous import DoseResponseGBM
from deepuplift.models.continuous.contracts import effect_prediction_from_curve, resolve_dose_support


def _dataset(lower=5.0, upper=20.0, rows=80):
    rng = np.random.default_rng(123)
    dose = np.linspace(lower, upper, rows)
    x = rng.normal(size=rows)
    frame = pd.DataFrame({"id": np.arange(rows), "x": x, "dose": dose, "y": 0.3 * x + 0.2 * dose})
    return create_causal_dataset(frame, feature_cols=["x"], treatment_col="dose", outcome_col="y", treatment_type=TreatmentType.CONTINUOUS, id_column="id")


def test_zero_to_twenty_observed_support_uses_supported_no_treatment_baseline():
    grid, baseline, support = resolve_dose_support(0.0, 20.0, grid_size=11)
    assert grid[0] == 0.0 and grid[-1] == 20.0 and baseline == 0.0
    assert support["no_treatment_supported"] is True
    assert support["contains_extrapolation"] is False


def test_five_to_twenty_support_does_not_silently_predict_zero_baseline():
    data = _dataset()
    model = DoseResponseGBM(grid_size=9).fit(data)
    prediction = model.predict(data.subset(np.arange(5)))
    assert prediction.baseline_dose == 5.0
    assert prediction.dose_grid[0] == 5.0
    assert prediction.metadata["dose_support"]["no_treatment_supported"] is False
    policy = build_continuous_policy(prediction, dose_cost=lambda dose: 0.01 * dose)
    assert all(row["recommended_treatment"] == 0.0 for row in policy.rows)
    assert all(row["reason"] == "unsupported_no_treatment_baseline" for row in policy.rows)
    with pytest.raises(ValueError, match="baseline_dose=0 is unsupported"):
        DoseResponseGBM(baseline_dose=0.0).fit(data)
    with pytest.raises(ValueError, match="dose_grid contains doses without empirical training support"):
        model.predict(data.subset(np.arange(5)), dose_grid=[0.0, 10.0, 20.0])


def test_zero_is_not_a_supported_baseline_without_factual_zero_rows():
    grid, baseline, support = resolve_dose_support(
        -1.0, 20.0, grid_size=9, no_treatment_observed=False
    )
    assert baseline == -1.0
    assert not np.any(np.isclose(grid, 0.0))
    assert support["no_treatment_supported"] is False
    with pytest.raises(ValueError, match="baseline_dose=0"):
        resolve_dose_support(-1.0, 20.0, grid_size=9, baseline_dose=0.0, no_treatment_observed=False)


def test_explicit_extrapolation_records_warning_and_doses():
    data = _dataset()
    model = DoseResponseGBM(
        grid_size=9, baseline_dose=0.0, dose_grid=np.linspace(0.0, 30.0, 7), allow_extrapolation=True
    ).fit(data)
    prediction = model.predict(data.subset(np.arange(4)))
    support = prediction.metadata["dose_support"]
    assert prediction.baseline_dose == 0.0
    assert support["extrapolation_enabled"] is True
    assert support["unsupported_doses"]
    assert "do not have empirical overlap support" in support["warning"]
    policy = build_continuous_policy(prediction, dose_cost=lambda dose: 0.001 * dose)
    assert policy.metadata["warning"] == support["warning"]


def test_decision_filters_doses_beyond_observed_support_and_uses_economic_value():
    prediction = effect_prediction_from_curve(
        unit_id=["u"], dose_grid=[0.0, 10.0, 25.0], dose_outcomes=[[0.0, 5.0, 20.0]],
        metadata={"dose_support": {"observed_min": 0.0, "observed_max": 20.0, "no_treatment_supported": True, "extrapolation_enabled": False}},
    )
    policy = build_continuous_policy(prediction, dose_cost={0.0: 0.0, 10.0: 1.0, 25.0: 0.0})
    assert policy.rows[0]["recommended_treatment"] == 10.0
    assert policy.metadata["filtered_unsupported_doses"] == [25.0]
    assert prediction.recommended_treatment[0] == 25.0  # maximum effect stays diagnostic, not the business decision


def test_policy_can_require_local_empirical_support():
    prediction = effect_prediction_from_curve(
        unit_id=["supported", "unsupported"], dose_grid=[0.0, 10.0, 20.0],
        dose_outcomes=[[0.0, 7.0, 9.0], [0.0, 7.0, 9.0]],
        metadata={
            "dose_support": {"observed_min": 0.0, "observed_max": 20.0, "no_treatment_supported": True, "extrapolation_enabled": False},
            "local_treatment_support": {
                "diagnostic_name": "local_empirical_treatment_support_diagnostic",
                "dose_grid": [0.0, 10.0, 20.0],
                "supported_dose_mask": [[True, True, False], [False, False, False]],
            },
        },
    )
    policy = build_continuous_policy(prediction, dose_cost=lambda dose: 0.1)
    assert policy.rows[0]["recommended_treatment"] == 10.0
    assert policy.rows[1]["recommended_treatment"] == 0.0
    assert policy.rows[1]["reason"] == "outside_local_treatment_support"
