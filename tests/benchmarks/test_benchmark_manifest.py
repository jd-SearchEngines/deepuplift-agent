import json

from deepuplift.benchmarks import run_benchmark
from deepuplift.data import synthetic_ground_truth


def test_benchmark_writes_complete_evidence_bundle(tmp_path):
    report = run_benchmark(synthetic_ground_truth(180, 13), models=["DR-Learner"], output_dir=tmp_path)
    run_dir = tmp_path / report["manifest"]["run_id"]
    expected = {"config.json", "dataset_manifest.json", "diagnostics.json", "nuisance.json", "model.json", "metrics.json", "policy.json", "environment.json", "evidence_manifest.json"}
    assert expected == {p.name for p in run_dir.iterdir()}
    manifest = json.loads((run_dir / "evidence_manifest.json").read_text())
    assert manifest["files"]["metrics.json"]
    assert report["results"][0]["metrics"]["ground_truth"]["available"] is True

