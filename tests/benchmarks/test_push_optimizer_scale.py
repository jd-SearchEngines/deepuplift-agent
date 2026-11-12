from __future__ import annotations

from deepuplift.decision.optimization import run_constrained_optimizer_suite


def test_push_optimizer_scale_report_contains_auditable_metrics():
    report = run_constrained_optimizer_suite(seed=42, scales=(1_000,))
    assert report["status"] == "PASS"
    scale = report["scales"][0]
    assert scale["status"] == "PASS"
    for row in scale["solvers"]:
        assert row["feasibility"]["feasible"]
        assert row["primal_constraint_violation"] == 0.0
        assert "signed_residuals" in row
        assert "complementarity" in row
        assert row["runtime_seconds"] >= 0.0
        assert row["candidates_per_second"] > 0.0
