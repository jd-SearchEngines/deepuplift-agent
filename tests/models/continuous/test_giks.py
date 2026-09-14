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
    assert provenance["observed_support"]["observed_min"] == model.dose_lower
    assert provenance["pseudo_extrapolation_fraction"] == 0.0
    assert provenance["near_pseudo_count"] + provenance["far_pseudo_count"] == provenance["accepted_pseudo_label_count"]
    assert provenance["accepted_ratio"] > 0
    assert provenance["mean_gp_variance"] is not None and provenance["p95_gp_variance"] is not None
    assert provenance["effective_pseudo_weight_sum"] > 0
    assert provenance["augmentation_executed"] is True
    assert len(prediction.dose_outcome_predictions) == len(prediction.dose_grid)
    assert all(np.isfinite(v).all() for v in prediction.dose_effect_predictions.values())
    policy = build_continuous_policy(prediction, dose_cost=lambda dose: 0.02 * dose)
    assert policy.rows


def test_giks_does_not_claim_unvalidated_base_models():
    with pytest.raises(ValueError, match="currently validates"):
        GIKSEstimator("DoseResponseGBM")


def test_giks_fails_closed_for_too_few_and_near_constant_dose_rows():
    dataset = synthetic_dose_response(rows=20, seed=31).dataset
    tiny = dataset.subset(np.arange(7))
    with pytest.raises(ValueError, match="at least eight"):
        GIKSEstimator("DRNet", factual_epochs=1, giks_epochs=1).fit(tiny)
    frame = dataset.to_pandas().iloc[:16].copy()
    frame[dataset.treatment_col] = 4.0 + np.linspace(0.0, 1e-10, len(frame))
    near_constant = type(dataset)(
        data=frame, feature_cols=dataset.feature_cols, treatment_col=dataset.treatment_col,
        outcome_col=dataset.outcome_col, treatment_type=dataset.treatment_type,
        assignment_type=dataset.assignment_type, id_column=dataset.id_column,
    )
    with pytest.raises(ValueError, match="near-constant"):
        GIKSEstimator("DRNet", factual_epochs=1, giks_epochs=1).fit(near_constant)


def test_giks_fails_closed_when_gp_variance_rejects_every_pseudo_label(monkeypatch):
    data = synthetic_dose_response(rows=36, seed=91)
    model = GIKSEstimator(
        "DRNet", random_state=91, factual_epochs=1, giks_epochs=1,
        augmentation_ratio=0.5, near_threshold=1e-5, max_gp_variance=0.01,
    )
    monkeypatch.setattr(model, "_gp_counterfactuals", lambda _x, _t, _y, qx, _qt: (np.zeros(len(qx)), np.ones(len(qx))))
    with pytest.raises(RuntimeError, match="no accepted pseudo-counterfactual"):
        model.fit(data.dataset)
