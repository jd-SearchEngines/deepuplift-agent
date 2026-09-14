import numpy as np
import pytest

pytest.importorskip("torch")

from deepuplift.benchmarks.continuous import synthetic_dose_response
from deepuplift.decision import build_continuous_policy
from deepuplift.models.continuous import GIKSEstimator


def test_giks_runs_factual_gradient_and_gp_augmentation_stages():
    data = synthetic_dose_response(rows=120, seed=43)
    model = GIKSEstimator(
        "VCNet", random_state=43, grid_size=11, factual_epochs=4, giks_epochs=3,
        augmentation_ratio=0.75, near_threshold=0.12, kernel_bandwidth=0.25,
        max_gp_context=96,
    ).fit(data.dataset)
    prediction = model.predict(data.dataset.subset(np.arange(12)))
    provenance = prediction.metadata["giks"]
    assert provenance["factual_epochs"] == 4
    assert provenance["giks_epochs"] == 3
    assert provenance["gradient_interpolation_attempted"] > 0
    assert provenance["gradient_interpolation_accepted"] > 0
    assert provenance["kernel_smoothing_attempted"] > 0
    assert provenance["kernel_smoothing_accepted"] > 0
    assert provenance["accepted_pseudo_label_count"] > 0
    assert provenance["augmentation_executed"] is True
    assert len(prediction.dose_outcome_predictions) == len(prediction.dose_grid)
    assert all(np.isfinite(v).all() for v in prediction.dose_effect_predictions.values())
    policy = build_continuous_policy(prediction, dose_cost=lambda dose: 0.02 * dose)
    assert policy.rows


def test_giks_does_not_claim_unvalidated_base_models():
    with pytest.raises(ValueError, match="currently validates"):
        GIKSEstimator("DoseResponseGBM")
