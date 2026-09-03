import importlib.util

import numpy as np
import pytest

from deepuplift.models import build_model


@pytest.mark.skipif(importlib.util.find_spec("causalml") is None, reason="causalml not installed")
def test_causalml_tree_adapter_is_constructible_when_optional_dependency_exists():
    from deepuplift.benchmarks.metrics import evaluate_prediction
    from deepuplift.data import synthetic_ground_truth
    dataset = synthetic_ground_truth(160, seed=42)
    prediction = build_model("CausalMLUpliftTree").fit(dataset).predict(dataset)
    metrics = evaluate_prediction(prediction, dataset, true_effect_col="true_effect")
    assert prediction.metadata["backend"] == "causalml"
    assert len(prediction.unit_id) == 160
    assert np.isfinite(prediction.recommended_effect).all()
    assert metrics["ranking"]["qini"] is not None
