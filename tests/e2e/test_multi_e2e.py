import numpy as np

from deepuplift.contracts import EffectPrediction, TreatmentType
from deepuplift.decision import build_multi_policy


def test_multi_decision_optimizes_net_value_not_raw_uplift():
    prediction = EffectPrediction(unit_id=np.array(["u1"]), treatment_type=TreatmentType.MULTI_DISCRETE, treatment_effects={5: np.array([.20]), 10: np.array([.30]), 20: np.array([.50])}, recommended_treatment=np.array([20]), recommended_effect=np.array([.50]))
    policy = build_multi_policy(prediction, treatment_costs={5: 1.0, 10: 20.0, 20: 45.0}, outcome_value=100.0)
    assert policy.rows[0]["recommended_treatment"] == 5

