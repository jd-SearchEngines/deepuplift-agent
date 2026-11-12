from __future__ import annotations

from pathlib import Path

import numpy as np

from deepuplift.benchmarks import fit_group_crossfit_proximal, run_mrt_proximal_baseline
from deepuplift.benchmarks.push_longitudinal import generate_push_longitudinal_dgp
from deepuplift.data import load_mybpmylife_mrt
from tests.data.test_mrt_loader import _fixture


def test_group_crossfit_proximal_uses_history_and_has_zero_group_overlap(tmp_path: Path):
    world = generate_push_longitudinal_dgp(load_mybpmylife_mrt(_fixture(tmp_path)), users=30, horizon=4, seed=13)
    logged = world.rollout("logged")
    features = ["x0", "x1", "x2", "previous_treatments", "time_since_last_push", "fatigue_load"]
    result = fit_group_crossfit_proximal(
        logged,
        feature_cols=features,
        group_col="user_id",
        treatment_col="treatment_assignment",
        outcome_col="observed_outcome",
        propensity_col="push_probability",
        n_splits=5,
        seed=13,
    )
    effects = result["predicted_effect"]
    assert result["status"] == "PASS"
    assert result["participant_overlap"] == 0
    assert np.isfinite(effects).all()
    assert len(effects) == len(logged)


def test_real_mrt_proximal_baseline_keeps_missingness_and_pehe_limits(tmp_path: Path):
    report = run_mrt_proximal_baseline(
        load_mybpmylife_mrt(_fixture(tmp_path)),
        outcome="activity_steps_60min",
        arm="activity",
        n_splits=2,
        seed=5,
    )
    assert report["status"] == "NOT_AVAILABLE"
    assert report["evidence_class"] == "REAL_RANDOMIZED_MRT"
    assert report["rows_complete_outcome"] == 2
    assert "Too few complete outcome rows" in report["reason"]
