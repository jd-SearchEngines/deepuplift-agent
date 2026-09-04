from deepuplift.data import randomized_dataset
from deepuplift.models import available_models, build_model, model_info
import numpy as np
import pandas as pd


def test_random_forest_meta_models_are_runnable():
    rng = np.random.default_rng(2)
    frame = pd.DataFrame({"x": rng.normal(size=140), "t": rng.binomial(1, .5, 140)})
    frame["y"] = ((frame["x"] + frame["t"] * .4 + rng.normal(size=140) * .2) > 0).astype(int)
    dataset = randomized_dataset(frame, feature_cols=["x"], treatment_col="t", outcome_col="y")
    for name in ["S-Learner-RF", "T-Learner-RF", "X-Learner-RF", "DR-Learner-RF"]:
        model = build_model(name)
        model.fit(dataset)
        assert len(model.predict(dataset).uplift) == 140
    assert model_info("CausalMLUpliftTree")["runnable"] is False
    assert "S-Learner-RF" in available_models(only_available=True)

