from deepuplift.benchmarks import run_benchmark
from deepuplift.data import synthetic_ground_truth


def test_fixed_seed_reproduces_dataset_and_metric_values():
    dataset_a = synthetic_ground_truth(160, 21)
    dataset_b = synthetic_ground_truth(160, 21)
    first = run_benchmark(dataset_a, models=["DR-Learner"], seed=21)
    second = run_benchmark(dataset_b, models=["DR-Learner"], seed=21)
    assert first["manifest"]["dataset_hash"] == second["manifest"]["dataset_hash"]
    assert first["results"][0]["metrics"]["ground_truth"] == second["results"][0]["metrics"]["ground_truth"]
