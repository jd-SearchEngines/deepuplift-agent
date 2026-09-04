import numpy as np

from deepuplift.data import estimate_nuisance, synthetic_ground_truth


def test_crossfit_has_oof_fold_contract():
    result = estimate_nuisance(synthetic_ground_truth(240, 11), n_splits=4, weighting="none")
    assert set(result.fold_ids) == {0, 1, 2, 3}
    assert len(result.propensity_scores) == 240
    assert np.isfinite(result.outcome_control_oof).all()
    assert np.isfinite(result.outcome_treated_oof).all()
    assert "p05" in result.diagnostics and "p95" in result.diagnostics

