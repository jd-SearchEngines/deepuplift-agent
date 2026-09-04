from deepuplift.benchmarks.benchmark_scalability import run_scalability_benchmark


def test_scale_smoke_records_measured_runtime():
    result = run_scalability_benchmark([1200], seed=3)
    assert result[0]["rows"] == 1200
    assert result[0]["total_seconds"] > 0
    assert result[0]["memory_status"] == "MEASURED_PROCESS_MAXRSS"
    assert result[0]["peak_memory_mb"] > 0
