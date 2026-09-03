from deepuplift.benchmarks.release import ReleaseConfig, run_release_benchmark


def test_release_benchmark_writes_release_bundle(tmp_path):
    report = run_release_benchmark(ReleaseConfig(synthetic_rows=180, scale_sizes=(300,), output_dir=tmp_path, benchmark_models=("S-Learner", "DR-Learner", "CausalForestDML"), packaging_status="PASS", ci_status="PASS"))
    run_dir = tmp_path / report["run_id"]
    assert report["release_gate"]["overall"] in {"READY_FOR_ALPHA_RELEASE", "READY_FOR_BETA_RELEASE"}
    assert (run_dir / "release_gate.json").exists()
    assert (run_dir / "RELEASE_BENCHMARK_REPORT.md").exists()
    assert report["rct"]["status"] == "PASS"
