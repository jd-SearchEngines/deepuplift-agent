from __future__ import annotations

import numpy as np
import pytest

from deepuplift.benchmarks import generate_push_longitudinal_dgp
from deepuplift.data import load_mybpmylife_mrt
from deepuplift.decision.ope import evaluate_sequential_ope_policy
from tests.data.test_mrt_loader import _fixture


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
        "one_step_contextual_ips",
        "one_step_contextual_snips",
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


def test_sequential_ope_rejects_invalid_raw_propensity_and_accepts_stochastic_target():
    with pytest.raises(ValueError, match="raw values are not silently clipped"):
        evaluate_sequential_ope_policy(
            logged_action=["none", "push"],
            reward=[1.0, 2.0],
            logged_propensity=[0.0, 1.2],
            target_action=["none", "push"],
            cluster_id=["u1", "u1"],
            decision_time=[0, 1],
        )
    result = evaluate_sequential_ope_policy(
        logged_action=["none", "push"],
        reward=[1.0, 2.0],
        logged_propensity=[0.6, 0.4],
        target_action=["none", "push"],
        target_logged_action_probability=[0.7, 0.3],
        cluster_id=["u1", "u1"],
        decision_time=[0, 1],
    )
    assert result["diagnostics"]["target_policy_type"] == "stochastic"
    assert result["diagnostics"]["target_support_rate"] == 1.0


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


def test_sequential_ope_exact_toy_weight_units():
    result = evaluate_sequential_ope_policy(
        logged_action=["push", "push"],
        reward=[1.0, 2.0],
        logged_propensity=[0.5, 0.5],
        target_action=["push", "push"],
        cluster_id=["u1", "u1"],
        decision_time=[0, 1],
    )
    assert result["metrics"]["one_step_contextual_ips"] == 3.0
    assert result["metrics"]["per_decision_ips"] == 10.0
    assert result["metrics"]["trajectory_ips"] == 12.0
    assert result["metrics"]["trajectory_snips"] == 3.0
    assert result["diagnostics"]["cumulative_weight_max"] == 4.0


def test_sequential_ope_five_seed_known_truth_smoke(tmp_path):
    mrt = load_mybpmylife_mrt(_fixture(tmp_path))
    errors = []
    for seed in (11, 22, 33, 44, 55):
        world = generate_push_longitudinal_dgp(mrt, scenario="fatigue", users=20, horizon=4, seed=seed)
        logged = world.rollout("logged")
        target = world.rollout("fatigue_aware")
        result = evaluate_sequential_ope_policy(
            logged_action=logged["treatment_assignment"],
            reward=logged["observed_outcome"],
            logged_propensity=logged["assignment_probability"],
            target_action=target["treatment_assignment"],
            cluster_id=logged["user_id"],
            decision_time=logged["decision_time"],
        )
        truth = float((target["baseline_y0"] + target["treatment_assignment"] * target["true_proximal_effect"]).sum() / world.users)
        estimate = result["metrics"]["one_step_contextual_ips"] * world.horizon
        assert np.isfinite(truth)
        assert np.isfinite(estimate)
        errors.append(abs(estimate - truth))
    assert float(np.mean(errors)) < 1.0
