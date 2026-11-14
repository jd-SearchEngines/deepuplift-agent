import numpy as np

from deepuplift.benchmarks.experiment_a import _multi_dataset, run_binary_comparison, run_multi_comparison
from deepuplift.benchmarks.continuous.datasets import synthetic_dose_response


def test_binary_runner_keeps_method_specific_prediction_evidence():
    report = run_binary_comparison(sizes=[180], seeds=[42], models=["S-Learner", "T-Learner"])
    rows = [row for row in report["rows"] if row["status"] == "PASS"]
    assert len(rows) == 2
    assert len({row["prediction_fingerprint"] for row in rows}) == 2
    assert all(row["pehe"] is not None for row in rows)


def test_multi_dataset_has_explicit_eligibility_mask():
    dataset = _multi_dataset(300, 42)
    frame = dataset.to_pandas()
    assert {"eligible_1", "eligible_2", "true_tau_1", "true_tau_2"}.issubset(frame.columns)
    assert frame["eligible_1"].isin([0.0, 1.0]).all()
    assert frame["eligible_2"].isin([0.0, 1.0]).all()


def test_multi_runner_independently_checks_eligibility():
    report = run_multi_comparison(sizes=[180], seeds=[42])
    rows = [row for row in report["rows"] if row["status"] == "PASS"]
    assert len(rows) == 2
    assert all(row["eligibility_violations"] == 0 for row in rows)


def test_continuous_dgp_exposes_three_declared_shapes():
    for shape in ("monotone", "saturating", "fatigue"):
        benchmark = synthetic_dose_response(rows=80, seed=42, shape=shape)
        assert benchmark.dataset.metadata["shape"] == shape
        assert benchmark.true_effect_curves.shape == (80, len(benchmark.dose_grid))
