import numpy as np

from deepuplift.contracts import CausalDataset, EffectPrediction, PolicyResult, TreatmentType


def test_effect_prediction_is_model_independent():
    prediction = EffectPrediction(
        unit_id=["u1", "u2"],
        treatment_type=TreatmentType.BINARY,
        y0=[0.1, 0.2],
        y1=[0.3, 0.1],
        uplift=np.array([0.2, -0.1]),
    )
    assert prediction.to_frame()["uplift"].tolist() == [0.2, -0.1]


def test_policy_result_has_rows_and_summary():
    result = PolicyResult(rows=[{"unit_id": "u1", "eligible": True}], summary={"target_count": 1})
    assert result.to_frame().iloc[0]["unit_id"] == "u1"
    assert result.to_dict()["summary"]["target_count"] == 1
