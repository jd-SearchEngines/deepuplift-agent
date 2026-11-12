from __future__ import annotations

import numpy as np
import pytest

from deepuplift.decision.ope import evaluate_sequential_ope_policy


def test_sequential_ope_reports_per_decision_and_trajectory_metrics():
    result = evaluate_sequential_ope_policy(
        logged_action=["none", "push", "none", "push"],
        reward=[1.0, 2.0, 1.5, 2.5],
        logged_propensity=[0.5, 0.5, 0.5, 0.5],
        target_action=["none", "push", "push", "push"],
        cluster_id=["u1", "u1", "u2", "u2"],
        decision_time=[0, 1, 0, 1],
        target_reward_pred=[1.0, 2.0, 1.8, 2.5],
        logged_reward_pred=[1.0, 2.0, 1.5, 2.5],
        n_bootstrap=30,
        random_state=7,
    )
    assert result["status"] == "ok"
    assert set(result["metrics"]) == {
        "per_decision_ips",
        "per_decision_snips",
        "trajectory_ips",
        "trajectory_snips",
        "per_decision_dr",
    }
    assert result["diagnostics"]["cluster_count"] == 2
    assert result["diagnostics"]["bootstrap_unit"] == "cluster"
    assert result["diagnostics"]["sequential_dr_status"].startswith("NOT_AVAILABLE")
    assert result["bootstrap"]["bootstrap_unit"] == "cluster"
    assert np.isfinite(result["metrics"]["trajectory_ips"])


def test_sequential_ope_rejects_duplicate_decision_time():
    with pytest.raises(ValueError, match="at most one row"):
        evaluate_sequential_ope_policy(
            logged_action=["none", "push"],
            reward=[1.0, 2.0],
            logged_propensity=[0.5, 0.5],
            target_action=["none", "push"],
            cluster_id=["u1", "u1"],
            decision_time=[0, 0],
        )
