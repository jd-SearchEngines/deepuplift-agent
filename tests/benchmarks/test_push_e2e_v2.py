from __future__ import annotations

from deepuplift.benchmarks.push_e2e_v2 import generate_push_e2e_world, run_push_e2e_v2


def test_push_e2e_world_has_known_truth_and_longitudinal_state():
    world = generate_push_e2e_world(users=40, horizon=3, seed=42)
    assert len(world.frame) == 120
    assert {"true_effect_push", "true_effect_rich", "fatigue_load", "carryover_state"}.issubset(world.frame)
    assert set(world.frame["treatment"].unique()).issubset({0, 1, 2})


def test_push_e2e_smoke_keeps_solver_feasibility_separate():
    report = run_push_e2e_v2(users=50, horizon=3, seeds=(42,))
    assert report["evidence_class"] == "SEMI_SYNTHETIC_KNOWN_TRUTH"
    assert report["gates"]["real_mrt_lane"] == "BLOCKED_LICENSE_AND_LOCAL_ARCHIVE"
    assert report["solver_rows"]
    assert all(row["feasible"] is True for row in report["solver_rows"])
    assert report["gates"]["full_sequential_dr"] == "NOT_AVAILABLE_CONTINUATION_MODEL_MISSING"
