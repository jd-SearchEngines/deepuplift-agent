from __future__ import annotations

import numpy as np
import pytest

from deepuplift.decision.ope import evaluate_ope_policy


def test_ope_cluster_bootstrap_resamples_decision_points_by_cluster():
    logged = np.array(["a", "b", "a", "b", "a", "b", "a", "b"])
    target = np.array(["a", "a", "b", "b", "a", "a", "b", "b"])
    reward = np.array([1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 2.0, 0.0])
    propensity = np.full(8, 0.5)
    clusters = np.array(["u1", "u1", "u2", "u2", "u3", "u3", "u4", "u4"])
    first = evaluate_ope_policy(
        logged_action=logged,
        reward=reward,
        logged_propensity=propensity,
        target_action=target,
        cluster_id=clusters,
        n_bootstrap=40,
        random_state=19,
    )
    second = evaluate_ope_policy(
        logged_action=logged,
        reward=reward,
        logged_propensity=propensity,
        target_action=target,
        cluster_id=clusters,
        n_bootstrap=40,
        random_state=19,
    )
    assert first["status"] == "ok"
    assert first["diagnostics"]["cluster_count"] == 4
    assert first["diagnostics"]["bootstrap_unit"] == "cluster"
    assert first["bootstrap"]["bootstrap_unit"] == "cluster"
    assert first["bootstrap"]["ips"] == second["bootstrap"]["ips"]


def test_ope_validates_prediction_and_cluster_lengths():
    kwargs = {
        "logged_action": ["a", "b"],
        "reward": [1.0, 0.0],
        "logged_propensity": [0.5, 0.5],
        "target_action": ["a", "a"],
    }
    with pytest.raises(ValueError, match="target_reward_pred"):
        evaluate_ope_policy(**kwargs, target_reward_pred=[1.0])
    with pytest.raises(ValueError, match="cluster_id"):
        evaluate_ope_policy(**kwargs, cluster_id=["u1"])


def test_one_step_ope_rejects_invalid_raw_propensity_and_supports_stochastic_target():
    with pytest.raises(ValueError, match="raw values are not silently clipped"):
        evaluate_ope_policy(
            logged_action=["a", "b"],
            reward=[1.0, 2.0],
            logged_propensity=[0.0, 1.2],
            target_action=["a", "b"],
        )
    result = evaluate_ope_policy(
        logged_action=["a", "b"],
        reward=[1.0, 2.0],
        logged_propensity=[0.6, 0.4],
        target_action=["a", "a"],
        target_logged_action_probability=[0.7, 0.2],
    )
    assert result["diagnostics"]["target_policy_type"] == "stochastic"
    assert result["diagnostics"]["target_support_rate"] == 1.0
