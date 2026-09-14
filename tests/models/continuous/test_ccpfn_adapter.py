import numpy as np
import pandas as pd

from deepuplift.contracts import AssignmentType, TreatmentType
from deepuplift.data.schema import create_causal_dataset
from deepuplift.models.continuous import CCPFNAdapter


class MockCEPO:
    def fit(self, X, t, y):
        self.context_size = len(X)
        self.context_mean = float(np.mean(y))

    def estimate_cepo(self, X, t):
        return self.context_mean + X[:, 0] * 0.1 + np.asarray(t) * 0.2


def test_ccpfn_adapter_contract_with_mock_backend_does_not_import_optional_package():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(32, 2))
    dose = np.linspace(0, 1, 32)
    frame = pd.DataFrame({"id": np.arange(32), "x1": x[:, 0], "x2": x[:, 1], "dose": dose, "y": x[:, 0] + dose})
    dataset = create_causal_dataset(frame, feature_cols=["x1", "x2"], treatment_col="dose", outcome_col="y", treatment_type=TreatmentType.CONTINUOUS, assignment_type=AssignmentType.OBSERVATIONAL, id_column="id")
    model = CCPFNAdapter(backend=MockCEPO(), grid_size=5, max_context_length=16, max_query_length=3, random_state=11).fit(dataset)
    prediction = model.predict(dataset.subset(np.arange(7)))
    assert model.context_size == 16
    assert prediction.metadata["implementation_source"] == "external_adapter"
    assert prediction.dose_grid.shape == (5,)
    assert prediction.baseline_dose == 0.0
    assert all(np.isfinite(v).all() for v in prediction.dose_outcome_predictions.values())
    assert len(prediction.dose_effect_predictions) == 5
