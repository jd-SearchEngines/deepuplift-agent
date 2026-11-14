from __future__ import annotations

from pathlib import Path

import pandas as pd

from deepuplift.data import (
    PUSH_DECISION_COLUMNS,
    build_mrt_decision_points,
    load_mybpmylife_mrt,
    validate_push_decision_points,
)
from tests.data.test_mrt_loader import _fixture


def test_mrt_decision_schema_keeps_unavailable_fields_explicit(tmp_path: Path):
    dataset = load_mybpmylife_mrt(_fixture(tmp_path))
    decision_points = build_mrt_decision_points(dataset, outcome="activity_steps_60min")
    frame = decision_points.to_pandas()

    assert set(PUSH_DECISION_COLUMNS).issubset(frame.columns)
    assert len(frame) == 4
    assert decision_points.validation["status"] == "PASS"
    assert decision_points.validation["unresolved_assignment_rows"] == 1
    assert frame["assignment_probability"].tolist()[:3] == [0.75, 0.125, 0.125]
    assert frame["assignment_probability"].isna().sum() == 1
    assert frame["availability"].isna().all()
    assert frame["decision_time"].isna().all()
    assert frame["time_since_last_push"].isna().all()
    assert frame.loc[1, "previous_treatments"] == ("none",)
    assert frame.loc[0, "treatment_exposure"] is pd.NA or pd.isna(frame.loc[0, "treatment_exposure"])
    assert frame["missing_indicator"].sum() == 0


def test_decision_schema_history_is_prior_only_and_validation_is_independent(tmp_path: Path):
    dataset = load_mybpmylife_mrt(_fixture(tmp_path))
    frame = build_mrt_decision_points(dataset).to_pandas()
    assert all(
        len(row["previous_treatments"]) == row["decision_sequence"]
        for _, row in frame.iterrows()
    )
    report = validate_push_decision_points(frame)
    assert report["status"] == "PASS"
    assert report["assignment_probability_mismatch_rows"] == 0
    assert report["history_leakage_rows"] == 0


def test_decision_schema_rejects_unknown_outcome(tmp_path: Path):
    dataset = load_mybpmylife_mrt(_fixture(tmp_path))
    try:
        build_mrt_decision_points(dataset, outcome="long_term_lift")
    except ValueError as exc:
        assert "Unknown MRT outcome" in str(exc)
    else:
        raise AssertionError("unknown outcome must fail closed")
