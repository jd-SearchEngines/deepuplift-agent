from __future__ import annotations

import numpy as np
import pytest

from deepuplift.core.ope import evaluate_sequential_dr_policy


def test_sequential_dr_is_exact_with_exact_continuation_values():
    logged = ["control", "treat", "control", "treat"]
    target = ["treat", "treat", "treat", "treat"]
    reward = [0.0, 1.0, 0.0, 1.0]
    # Two one-step trajectories, each with exact V and Q nuisance values.
    result = evaluate_sequential_dr_policy(
        logged_action=logged,
        reward=reward,
        logged_propensity=[0.5] * 4,
        target_action=target,
        cluster_id=["a", "a", "b", "b"],
        decision_time=[0, 1, 0, 1],
        target_value_pred=[2.0, 1.0, 2.0, 1.0],
        logged_action_value_pred=[2.0, 1.0, 2.0, 1.0],
        n_bootstrap=20,
    )
    assert result["status"] == "ok"
    assert result["diagnostics"]["sequential_dr_status"] == "AVAILABLE"
    assert result["metrics"]["sequential_dr"] == pytest.approx(2.0)
    assert result["bootstrap"]["sequential_dr"]["low"] <= 2.0 <= result["bootstrap"]["sequential_dr"]["high"]


def test_sequential_dr_uses_cluster_bootstrap_and_is_not_one_step_dr():
    logged = np.array(["treat", "control", "treat", "control"])
    target = np.array(["treat", "treat", "treat", "treat"])
    result = evaluate_sequential_dr_policy(
        logged_action=logged,
        reward=[1.0, 0.0, 1.0, 0.0],
        logged_propensity=[0.5] * 4,
        target_action=target,
        cluster_id=["u1", "u1", "u2", "u2"],
        decision_time=[0, 1, 0, 1],
        target_value_pred=[2.0, 1.0, 2.0, 1.0],
        logged_action_value_pred=[1.0, 1.0, 1.0, 1.0],
        n_bootstrap=10,
    )
    assert result["diagnostics"]["bootstrap_unit"] == "cluster"
    assert result["diagnostics"]["continuation_value_model"] == "provided_target_V_and_logged_Q"
    assert result["diagnostics"]["max_horizon"] == 2


def test_sequential_dr_rejects_invalid_propensity():
    with pytest.raises(ValueError, match="logged_propensity"):
        evaluate_sequential_dr_policy(
            logged_action=["a"], reward=[1], logged_propensity=[0], target_action=["a"],
            cluster_id=["u"], decision_time=[0], target_value_pred=[1], logged_action_value_pred=[1]
        )
