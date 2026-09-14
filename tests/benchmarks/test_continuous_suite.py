from deepuplift.benchmarks.continuous import run_continuous_suite, synthetic_dose_response
from deepuplift.benchmarks.continuous import runner as continuous_runner


def test_continuous_suite_runs_reproducible_synthetic_gbm_benchmark():
    data = synthetic_dose_response(rows=100, seed=6)
    result = run_continuous_suite(data=data, model_names=["DoseResponseGBM"])
    assert result["status"] == "PASS"
    assert result["dataset"]["ground_truth_available"] is True
    assert result["models"][0]["metrics"]["mise"] >= 0
    assert result["models"][0]["train_seconds"] >= 0
    assert result["models"][0]["predict_curve_seconds"] >= 0
    assert result["models"][0]["decision_seconds"] >= 0
    assert "Continuous Treatment Benchmark" in result["report"]


def test_continuous_suite_does_not_hide_a_blocked_requested_model(monkeypatch):
    def model_result(name, *_args, **_kwargs):
        return {"model": name, "status": "PASS" if name == "DoseResponseGBM" else "BLOCKED"}

    monkeypatch.setattr(continuous_runner, "run_continuous_model", model_result)
    result = continuous_runner.run_continuous_dataset(
        synthetic_dose_response(rows=40, seed=7),
        model_names=["DoseResponseGBM", "DRNet"],
    )
    assert result["status"] == "PARTIAL"
