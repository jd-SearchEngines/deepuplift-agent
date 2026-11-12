from __future__ import annotations

from pathlib import Path

from deepuplift.benchmarks.push_semisynthetic import generate_push_semisynthetic, run_push_semisynthetic_scenario
from deepuplift.data import load_mybpmylife_mrt
from tests.data.test_mrt_loader import _fixture


def test_semisynthetic_truth_is_not_in_model_frame(tmp_path: Path):
    mrt = load_mybpmylife_mrt(_fixture(tmp_path))
    benchmark = generate_push_semisynthetic(mrt, scenario="measured_confounding", rows=40, seed=3)
    frame = benchmark.dataset.to_pandas()
    assert "true_effect" not in frame.columns
    assert len(benchmark.truth.y0) == 40
    assert benchmark.dataset.metadata["truth_not_in_model_frame"] is True


def test_semisynthetic_scenarios_report_known_truth_and_mnar_limit(tmp_path: Path):
    mrt = load_mybpmylife_mrt(_fixture(tmp_path))
    for scenario in ("measured_confounding", "sample_selection_mar", "overlap_failure", "hidden_confounding", "sample_selection_mnar"):
        report = run_push_semisynthetic_scenario(mrt, scenario=scenario, rows=40, seed=5)
        assert report["status"] == "PASS"
        assert report["evidence_class"] == "SEMI_SYNTHETIC_KNOWN_TRUTH"
        assert report["methods"]
    mnar = run_push_semisynthetic_scenario(mrt, scenario="sample_selection_mnar", rows=40, seed=5)
    assert "not identified" in mnar["mnar_interpretation"]
