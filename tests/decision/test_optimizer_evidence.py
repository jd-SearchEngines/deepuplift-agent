from __future__ import annotations

from deepuplift.decision.optimization import run_constrained_optimizer_suite


def test_optimizer_evidence_keeps_feasibility_and_quality_separate():
    report = run_constrained_optimizer_suite(seed=17, scales=(24,))
    assert report["status"] in {"PASS", "PARTIAL"}
    assert report["small_instance"]["exact_reference"]["status"] == "PASS"
    for row in report["small_instance"]["solvers"]:
        assert row["feasibility"]["feasible"] is True
        assert "objective_net_value" in row
        assert "runtime_seconds" in row
