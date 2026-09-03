import numpy as np

from deepuplift.data import estimate_nuisance, synthetic_ground_truth
from deepuplift.models import build_model


def test_dr_reuses_unified_nuisance_result():
    dataset = synthetic_ground_truth(240, 9)
    nuisance = estimate_nuisance(dataset, n_splits=4, weighting="overlap")
    model = build_model("DR-Learner")
    model.fit(dataset, nuisance=nuisance)
    prediction = model.predict(dataset)
    assert len(prediction.uplift) == 240
    assert np.isfinite(prediction.uplift).all()
    assert prediction.metadata["nuisance_provenance"]["provenance"] == "deepuplift unified nuisance layer"

