import numpy as np
import pandas as pd

from deepuplift.data import create_causal_dataset
from deepuplift.models.continuous import DoseResponseGBM
from deepuplift.models.multi import MultiTreatmentOutcomeModel


def test_multi_treatment_prediction_contract():
    rng = np.random.default_rng(3)
    frame = pd.DataFrame({"id": range(180), "x": rng.normal(size=180), "t": np.tile([0, 1, 2], 60), "y": rng.normal(size=180)})
    dataset = create_causal_dataset(frame, feature_cols=["x"], treatment_col="t", outcome_col="y", id_column="id", treatment_type="multi_discrete")
    prediction = MultiTreatmentOutcomeModel(task="regression", random_state=3).fit(dataset).predict(dataset)
    assert prediction.treatment_type.value == "multi_discrete"
    assert len(prediction.treatment_effects) == 2
    assert len(prediction.recommended_treatment) == 180


def test_continuous_treatment_prediction_contract():
    rng = np.random.default_rng(4)
    frame = pd.DataFrame({"id": range(180), "x": rng.normal(size=180), "dose": rng.uniform(0, 10, 180), "y": rng.normal(size=180)})
    dataset = create_causal_dataset(frame, feature_cols=["x"], treatment_col="dose", outcome_col="y", id_column="id", treatment_type="continuous")
    prediction = DoseResponseGBM(task="regression", random_state=4, grid_size=7).fit(dataset).predict(dataset)
    assert prediction.treatment_type.value == "continuous"
    assert len(prediction.recommended_treatment) == 180
    assert len(prediction.metadata["dose_grid"]) == 7
