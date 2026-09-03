from deepuplift.benchmarks.release.gates import build_release_gate


def test_release_gate_reports_alpha_when_core_paths_pass():
    gate = build_release_gate(synthetic_pass=True, observational_pass=True, application_pass=True, rct_pass=True, multi_pass=True, continuous_pass=True, scale_results=[{"rows": 1000, "prediction_finite": True, "policy_generated": True, "peak_memory_mb": 10}], hillstrom_status="NOT_RUN", criteo_status="NOT_RUN", reproducibility_pass=True, packaging_pass=True, ci_pass=True, optional_statuses={"CausalForestDML": "NOT_INSTALLED"})
    assert gate["overall"] == "READY_FOR_ALPHA_RELEASE"
    assert gate["SCALE"]["1000"] == "PASS"

