import numpy as np

from deepuplift.contracts import EffectPrediction, TreatmentType
from deepuplift.decision import build_binary_policy, simulate_budget


def test_budget_policy_is_cost_aware():
    prediction = EffectPrediction(
        unit_id=["u1", "u2", "u3"],
        treatment_type=TreatmentType.BINARY,
        uplift=np.array([0.5, 0.2, -0.1]),
        recommended_effect=np.array([0.5, 0.2, -0.1]),
    )
    policy = build_binary_policy(prediction, treatment_cost=10, outcome_value=30, budget=10)
    assert policy.summary["target_count"] == 1
    assert policy.summary["total_cost"] == 10
    assert policy.summary["expected_net_value"] == 5


def test_budget_simulator_returns_business_columns():
    frame = __import__("pandas").DataFrame({"uplift": [0.4, 0.2, 0.1]})
    result = simulate_budget(frame, budgets=[10, 20], outcome_value=30, treatment_cost=10)
    assert {"budget", "target_count", "net_value", "iroas"}.issubset(result.columns)
