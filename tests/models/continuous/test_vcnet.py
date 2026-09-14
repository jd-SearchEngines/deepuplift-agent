import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from deepuplift.contracts import TreatmentType
from deepuplift.data.schema import create_causal_dataset
from deepuplift.decision import build_continuous_policy
from deepuplift.models.continuous import VCNet


def test_vcnet_uses_varying_coefficients_and_emits_continuous_curve():
    rng = np.random.default_rng(12)
    x = rng.normal(size=120)
    dose = np.clip(8 + 5 * x + rng.normal(0, 2, 120), 0, 20)
    y = 0.3 + (0.8 + 0.25 * np.tanh(x)) * dose / 20 - 1.1 * (dose / 20) ** 2 + rng.normal(0, 0.05, 120)
    frame = pd.DataFrame({"id": np.arange(120), "x": x, "dose": dose, "y": y})
    data = create_causal_dataset(frame, feature_cols=["x"], treatment_col="dose", outcome_col="y", treatment_type=TreatmentType.CONTINUOUS, id_column="id")
    model = VCNet(random_state=7, epochs=6, grid_size=11, basis_degree=3, hidden_dim=12).fit(data)
    prediction = model.predict(data.subset(np.arange(6)))
    assert hasattr(model.model, "coefficients")
    assert prediction.metadata["recommendation_type"] == "maximum_effect_reference_only"
    assert prediction.dose_grid[0] == 0.0
    nearby_features = pd.concat([frame[["x"]].iloc[:1], frame[["x"]].iloc[:1]], ignore_index=True)
    values = model._predict_at_raw_features(nearby_features, np.array([8.0, 8.001]))
    assert np.isfinite(values).all() and abs(values[1] - values[0]) < 0.01
    policy = build_continuous_policy(prediction, dose_cost=lambda dose: 0.02 * dose)
    assert policy.rows
