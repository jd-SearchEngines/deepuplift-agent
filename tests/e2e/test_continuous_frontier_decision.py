import numpy as np
import pytest

pytest.importorskip("torch")

from deepuplift.decision import build_continuous_policy
from deepuplift.models.continuous.contracts import effect_prediction_from_curve


def test_shared_contract_selects_economic_dose_instead_of_maximum_effect():
    grid = np.array([0.0, 8.0, 20.0])
    outcomes = np.array([[0.0, 5.0, 7.0]])
    prediction = effect_prediction_from_curve(
        unit_id=["user"], dose_grid=grid, dose_outcomes=outcomes,
        metadata={"model_families_checked": ["DoseResponseGBM", "VCNet", "GIKS"]},
    )
    policy = build_continuous_policy(
        prediction,
        dose_cost={0.0: 0.0, 8.0: 1.0, 20.0: 10.0},
        outcome_value=1.0,
    )
    assert prediction.recommended_treatment[0] == 20.0
    assert policy.rows[0]["recommended_treatment"] == 8.0
    assert policy.rows[0]["net_value"] == 4.0
