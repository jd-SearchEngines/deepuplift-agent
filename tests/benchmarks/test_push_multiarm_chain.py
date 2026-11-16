from __future__ import annotations

from scripts.run_push_multiarm_chain import run


def test_multiarm_candidate_optimizer_ope_chain_is_feasible():
    report = run(seeds=[42], users=60, horizon=3, bootstrap=8)
    assert report["status"] == "PASS"
    assert report["gates"]["three_arm_logged_probability_ope"] == "PASS"
    assert all(row["feasible"] is True for result in report["results"] for row in result["solvers"])
