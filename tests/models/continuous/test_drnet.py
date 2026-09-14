import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from deepuplift.contracts import TreatmentType
from deepuplift.data.schema import create_causal_dataset
from deepuplift.decision import build_continuous_policy
from deepuplift.models import build_model


def _dataset(seed=31, rows=100):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=rows)
    dose = np.clip(10 + 4 * x + rng.normal(0, 2, rows), 0, 20)
    y = 0.2 + 0.08 * dose - 0.002 * dose**2 + 0.1 * x + rng.normal(0, 0.1, rows)
    frame = pd.DataFrame({"id": np.arange(rows), "x": x, "dose": dose, "y": y})
    return create_causal_dataset(frame, feature_cols=["x"], treatment_col="dose", outcome_col="y", treatment_type=TreatmentType.CONTINUOUS, id_column="id")


def test_drnet_fits_stratified_heads_and_emits_full_curve_for_decision():
    from deepuplift.models.continuous import DRNet

    dataset = _dataset()
    model = DRNet(random_state=3, epochs=5, grid_size=9, num_dose_bins=4, hidden_dim=12).fit(dataset)
    prediction = model.predict(dataset.subset(np.arange(8)))
    assert len(model.model.heads) == 4
    assert len(prediction.dose_outcome_predictions) == len(prediction.dose_grid)
    assert prediction.baseline_dose == 0.0
    assert all(np.isfinite(values).all() for values in prediction.dose_effect_predictions.values())
    policy = build_continuous_policy(prediction, dose_cost=lambda dose: 0.01 * dose)
    assert policy.rows and policy.metadata["decision_rule"].startswith("for each user")


def test_registry_build_model_drnet_defaults_to_regression():
    model = build_model("DRNet", random_state=8, epochs=2)
    assert model.task == "regression"
