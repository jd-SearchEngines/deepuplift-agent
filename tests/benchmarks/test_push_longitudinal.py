from __future__ import annotations

from pathlib import Path

import numpy as np

from deepuplift.benchmarks import generate_push_longitudinal_dgp, run_push_longitudinal_suite
from deepuplift.data import load_mybpmylife_mrt
from tests.data.test_mrt_loader import _fixture


def test_longitudinal_dgp_preserves_truth_and_state_order(tmp_path: Path):
    world = generate_push_longitudinal_dgp(load_mybpmylife_mrt(_fixture(tmp_path)), users=4, horizon=5, seed=9)
    rollout = world.rollout("random_push")
    assert len(rollout) == 20
    assert set(rollout["evidence_class"]) == {"SEMI_SYNTHETIC_KNOWN_TRUTH"}
    assert (rollout["treatment_assignment"] <= rollout["availability"].astype(int)).all()
    assert np.isfinite(rollout[["potential_outcome_no_push", "potential_outcome_push", "true_proximal_effect"]].to_numpy()).all()
    for _, group in rollout.groupby("user_id", sort=False):
        history = group["previous_treatments"].tolist()
        assert all(0 <= value <= index for index, value in enumerate(history))
        assert history == sorted(history)


def test_longitudinal_suite_reports_policy_truth_without_real_claims(tmp_path: Path):
    result = run_push_longitudinal_suite(
        load_mybpmylife_mrt(_fixture(tmp_path)),
        scenarios=["fatigue", "carryover"],
        policies=["no_push", "fatigue_aware", "oracle"],
        users=5,
        horizon=4,
        seed=11,
    )
    assert result["status"] == "PASS"
    assert result["evidence_class"] == "SEMI_SYNTHETIC_KNOWN_TRUTH"
    assert len(result["scenarios"]) == 2
    for scenario in result["scenarios"]:
        assert len(scenario["policies"]) == 3
        assert all(np.isfinite(row["cumulative_policy_value"]) for row in scenario["policies"])
    assert "real MRT causal effects" in result["limitations"][0]
