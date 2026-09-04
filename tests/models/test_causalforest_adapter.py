import importlib.util

import numpy as np
import pytest

from deepuplift.models import build_model, model_info


@pytest.mark.skipif(importlib.util.find_spec("econml") is None, reason="econml not installed")
def test_causalforest_adapter_is_real_when_optional_dependency_exists():
    from deepuplift.benchmarks.metrics import evaluate_prediction
    from deepuplift.data import synthetic_ground_truth
    model = build_model("CausalForestDML")
    dataset = synthetic_ground_truth(160, seed=42)
    prediction = model.fit(dataset).predict(dataset)
    metrics = evaluate_prediction(prediction, dataset, true_effect_col="true_effect")
    assert model.nuisance_mode == "internal"
    assert model_info("CausalForestDML")["runnable"] is True
    assert prediction.metadata["backend"] == "econml"
    assert len(prediction.unit_id) == 160
    assert np.isfinite(prediction.recommended_effect).all()
    assert metrics["ranking"]["qini"] is not None
