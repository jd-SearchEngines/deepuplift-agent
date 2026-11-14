from __future__ import annotations

from pathlib import Path

from deepuplift.benchmarks import run_push_mrt_baseline
from deepuplift.data import load_mybpmylife_mrt, participant_group_split
from tests.data.test_mrt_loader import _fixture


def test_participant_group_split_has_no_overlap(tmp_path: Path):
    dataset = load_mybpmylife_mrt(_fixture(tmp_path))
    train, test = participant_group_split(dataset, test_fraction=0.5, seed=11)
    frame = dataset.analysis_frame().reset_index(drop=True)
    assert set(frame.iloc[train]["participant_id"]).isdisjoint(set(frame.iloc[test]["participant_id"]))


def test_push_baseline_runs_separate_outcomes_without_pehe(tmp_path: Path):
    dataset = load_mybpmylife_mrt(_fixture(tmp_path))
    report = run_push_mrt_baseline(dataset, bootstrap_repetitions=20, seed=7)
    assert report["status"] == "PASS"
    assert [row["outcome"] for row in report["outcomes"]] == ["activity_steps_60min", "diet_low_salt_24hr"]
    assert all(row["ground_truth"]["pehe"] == "NOT_AVAILABLE" for row in report["outcomes"])
    assert all("activity_vs_none" in row["itt_contrasts"] for row in report["outcomes"])
    assert all(row["delivery_separate_from_assignment"]["causal_delivery_effect_estimated"] is False for row in report["outcomes"])
