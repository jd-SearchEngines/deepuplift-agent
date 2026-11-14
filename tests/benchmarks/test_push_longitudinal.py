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


def test_longitudinal_scenario_isolation_and_dynamic_oracle(tmp_path: Path):
    mrt = load_mybpmylife_mrt(_fixture(tmp_path))
    no_carry = generate_push_longitudinal_dgp(mrt, scenario="no_fatigue_no_carryover", users=10, horizon=6, seed=21)
    carry = generate_push_longitudinal_dgp(mrt, scenario="carryover", users=10, horizon=6, seed=21)
    no_random = no_carry.rollout("random_push")
    no_none = no_carry.rollout("no_push")
    carry_random = carry.rollout("random_push")
    assert np.allclose(no_random["potential_outcome_no_push"], no_none["potential_outcome_no_push"])
    assert carry_random["potential_outcome_no_push"].to_numpy().max() > no_random["potential_outcome_no_push"].to_numpy().max()
    rolling = generate_push_longitudinal_dgp(mrt, scenario="rolling_fatigue", users=10, horizon=6, seed=21)
    decayed = generate_push_longitudinal_dgp(mrt, scenario="decayed_fatigue", users=10, horizon=6, seed=21)
    assert rolling.parameters["fatigue_mode"] == "rolling"
    assert decayed.parameters["fatigue_mode"] == "decayed"
    assert np.any(rolling.rollout("random_push")["rolling_fatigue"] > 0)
    assert np.any(decayed.rollout("random_push")["decayed_fatigue"] > 0)
    myopic = carry.rollout("myopic_oracle")
    dynamic = carry.rollout("dynamic_oracle")
    myopic_value = float((myopic["baseline_y0"] + myopic["treatment_assignment"] * myopic["true_proximal_effect"]).sum())
    dynamic_value = float((dynamic["baseline_y0"] + dynamic["treatment_assignment"] * dynamic["true_proximal_effect"]).sum())
    assert dynamic_value >= myopic_value - 1e-12
