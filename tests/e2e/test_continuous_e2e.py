import numpy as np
import pandas as pd

from deepuplift.data import TreatmentType, create_causal_dataset
from deepuplift.decision import build_continuous_policy
from deepuplift.models import build_model


def test_continuous_reference_produces_experimental_policy():
    rng = np.random.default_rng(4)
    dose = np.linspace(0, 20, 160)
    frame = pd.DataFrame({"id": np.arange(len(dose)), "x": rng.normal(size=len(dose)), "dose": dose, "y": .2 + .05 * dose - .002 * dose**2 + rng.normal(0, .02, len(dose))})
    dataset = create_causal_dataset(frame, feature_cols=["x"], treatment_col="dose", outcome_col="y", treatment_type=TreatmentType.CONTINUOUS, id_column="id")
    model = build_model("DoseResponseGBM"); model.fit(dataset); prediction = model.predict(dataset)
    policy = build_continuous_policy(prediction, dose_cost=lambda value: .01 * value)
    assert np.isfinite(prediction.recommended_effect).all()
    assert policy.rows and policy.metadata["status"] == "EXPERIMENTAL"

