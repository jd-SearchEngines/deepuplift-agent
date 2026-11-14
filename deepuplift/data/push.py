"""Canonical decision-point construction for repeated Push experiments.

The public MRT table contains repeated decision rows, but it does not expose
every field needed by a production Push system.  This module creates a stable
decision-point frame while keeping unavailable fields explicitly missing.  In
particular, it never treats notification delivery as randomized assignment and
never invents calendar timestamps or availability values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from deepuplift.data.public.mrt import (
    MRTDataset,
    MRT_DECLARED_ASSIGNMENT_PROBABILITIES,
    load_mybpmylife_mrt,
)


PUSH_DECISION_COLUMNS = (
    "user_id",
    "decision_time",
    "user_state",
    "user_history",
    "availability",
    "treatment_assignment",
    "assignment_probability",
    "treatment_exposure",
    "previous_treatments",
    "time_since_last_push",
    "outcome",
    "outcome_window",
    "missing_indicator",
)

_OUTCOME_SPECS = {
    "activity_steps_60min": ("SumValues_Steps_60minAfter", "60 minutes after decision"),
    "diet_low_salt_24hr": ("NumValues_LowSaltChoices_24hrAfter", "24 hours after decision"),
}
_STATE_COLUMNS = (
    "TimeEnrollment_to_PrefDate_days",
    "DoesPrefDateHaveRepeat",
    "Phase",
    "Phase_1dayBefore",
    "TimeOfDayID",
    "TimeOfDay",
    "PrefRepeatRank",
    "IsPresent_HeartRate_30minBefore",
    "NumValues_HeartRate_30minBefore",
    "SumValues_WearTime_hr_30minBefore",
    "SumValues_Steps_30minBefore",
    "NumValues_LowSaltChoices_30minBefore",
    "SodiumIntakeScore",
)
_ORDER_COLUMNS = (
    "TimeEnrollment_to_PrefDate_days",
    "TimeOfDayID",
    "PrefRepeatRank",
)


def _json_safe(value: Any) -> Any:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _state_record(row: pd.Series, columns: list[str]) -> dict[str, Any]:
    return {column: _json_safe(row[column]) for column in columns}


def _ordered_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["_source_row_order"] = np.arange(len(result), dtype="int64")
    sort_columns = [column for column in _ORDER_COLUMNS if column in result.columns]
    result = result.sort_values(
        ["participant_id", *sort_columns, "_source_row_order"],
        kind="mergesort",
        na_position="last",
    )
    return result


@dataclass(frozen=True)
class PushDecisionDataset:
    """Canonical repeated-decision frame plus provenance and validation."""

    frame: pd.DataFrame
    metadata: dict[str, Any]
    validation: dict[str, Any]

    def __post_init__(self) -> None:
        missing = sorted(set(PUSH_DECISION_COLUMNS) - set(self.frame.columns))
        if missing:
            raise ValueError(f"Push decision frame is missing canonical columns: {missing}")

    def to_pandas(self) -> pd.DataFrame:
        return self.frame.copy()

    @property
    def analysis_ready(self) -> pd.DataFrame:
        """Return rows with explicit assignment and a non-missing outcome."""

        return self.frame.loc[
            self.frame["treatment_assignment"].notna()
            & ~self.frame["missing_indicator"].astype(bool)
        ].copy()


def validate_push_decision_points(frame: pd.DataFrame) -> dict[str, Any]:
    """Independently validate canonical fields and history leakage guards."""

    missing = sorted(set(PUSH_DECISION_COLUMNS) - set(frame.columns))
    if missing:
        return {"status": "FAIL", "missing_columns": missing}
    assignment = frame["treatment_assignment"].astype("string")
    expected_probability = assignment.map(MRT_DECLARED_ASSIGNMENT_PROBABILITIES)
    probabilities = pd.to_numeric(frame["assignment_probability"], errors="coerce")
    probability_mismatch = (
        assignment.notna()
        & (probabilities - expected_probability.astype("float64")).abs().gt(1e-12)
    )
    history_length_mismatch = 0
    if "decision_sequence" in frame:
        history_length_mismatch = sum(
            len(history) != int(sequence)
            for history, sequence in zip(frame["previous_treatments"], frame["decision_sequence"])
            if isinstance(history, (list, tuple))
        )
    return {
        "status": "PASS" if not probability_mismatch.any() else "FAIL",
        "required_columns": list(PUSH_DECISION_COLUMNS),
        "missing_columns": missing,
        "rows": int(len(frame)),
        "users": int(frame["user_id"].nunique(dropna=True)),
        "assignment_rows": int(assignment.notna().sum()),
        "unresolved_assignment_rows": int(assignment.isna().sum()),
        "missing_outcome_rows": int(frame["missing_indicator"].astype(bool).sum()),
        "assignment_probability_mismatch_rows": int(probability_mismatch.sum()),
        "history_leakage_rows": int(history_length_mismatch),
        "availability_observed_rows": int(frame["availability"].notna().sum()),
        "decision_time_observed_rows": int(frame["decision_time"].notna().sum()),
    }


def build_mrt_decision_points(
    dataset: MRTDataset | str,
    *,
    outcome: str = "activity_steps_60min",
) -> PushDecisionDataset:
    """Construct canonical Push decision points from a local MRT source.

    The returned frame includes unresolved assignment rows so the audit can
    distinguish raw coverage from analysis-ready coverage.  Missing upstream
    availability and calendar timestamps remain missing rather than being
    inferred from assignment or ordinal study-day fields.
    """

    if outcome not in _OUTCOME_SPECS:
        raise ValueError(f"Unknown MRT outcome {outcome!r}; expected {sorted(_OUTCOME_SPECS)}")
    mrt = dataset if isinstance(dataset, MRTDataset) else load_mybpmylife_mrt(dataset)
    outcome_column, outcome_window = _OUTCOME_SPECS[outcome]
    if outcome_column not in mrt.frame.columns:
        raise ValueError(f"MRT outcome column {outcome_column!r} is not available.")

    source = _ordered_frame(mrt.frame.reset_index(drop=True))
    state_columns = [column for column in _STATE_COLUMNS if column in source.columns]
    previous_treatments: dict[int, tuple[str, ...]] = {}
    sequence_by_row: dict[int, int] = {}
    last_push_sequence: dict[Any, int | None] = {}
    for user_id, group in source.groupby("participant_id", sort=False, dropna=False):
        history: list[str] = []
        for sequence, (row_index, row) in enumerate(group.iterrows()):
            previous_treatments[int(row_index)] = tuple(history)
            sequence_by_row[int(row_index)] = sequence
            current = row.get("assignment_arm")
            if pd.notna(current):
                history.append(str(current))
            last_push_sequence.setdefault(user_id, None)

    output = pd.DataFrame(index=source.index)
    output["user_id"] = source["participant_id"]
    # The upstream table has ordinal study fields but no calendar timestamp.
    output["decision_time"] = pd.Series(pd.NaT, index=source.index, dtype="datetime64[ns]")
    output["user_state"] = [
        _state_record(row, state_columns) for _, row in source.iterrows()
    ]
    output["user_history"] = [
        {
            "prior_assignment_count": len(previous_treatments[int(index)]),
            "prior_assignments": list(previous_treatments[int(index)]),
        }
        for index in source.index
    ]
    output["availability"] = pd.Series(pd.NA, index=source.index, dtype="boolean")
    output["treatment_assignment"] = source["assignment_arm"].astype("string")
    output["assignment_probability"] = output["treatment_assignment"].map(MRT_DECLARED_ASSIGNMENT_PROBABILITIES).astype("Float64")
    output["treatment_exposure"] = source["delivery_arm"].astype("string")
    output["previous_treatments"] = [previous_treatments[int(index)] for index in source.index]
    output["time_since_last_push"] = pd.Series(pd.NA, index=source.index, dtype="Float64")
    output["outcome"] = pd.to_numeric(source[outcome_column], errors="coerce")
    output["outcome_window"] = outcome_window
    output["missing_indicator"] = output["outcome"].isna()

    # Preserve audit-relevant fields without pretending they are canonical
    # timestamps or availability observations.
    output["decision_sequence"] = [sequence_by_row[int(index)] for index in source.index]
    output["assignment_valid"] = output["treatment_assignment"].notna()
    output["source_row_index"] = source["_source_row_order"].astype("int64").to_numpy()
    for column in _ORDER_COLUMNS:
        if column in source.columns:
            output[f"source_{column}"] = source[column].to_numpy()
    output = output.reset_index(drop=True)
    validation = validate_push_decision_points(output)
    validation.update(
        {
            "outcome": outcome,
            "outcome_source_column": outcome_column,
            "outcome_window": outcome_window,
            "status": "PASS" if validation["status"] == "PASS" else "FAIL",
        }
    )
    metadata = {
        **mrt.metadata,
        "schema_name": "push_decision_point",
        "schema_version": 1,
        "evidence_class": "REAL_RANDOMIZED_MRT",
        "decision_unit": "participant x repeated decision row",
        "assignment_field": "treatment_assignment",
        "exposure_field": "treatment_exposure",
        "availability_status": "NOT_AVAILABLE_IN_UPSTREAM_TABLE",
        "decision_time_status": "NOT_AVAILABLE_IN_UPSTREAM_TABLE",
        "time_since_last_push_status": "NOT_AVAILABLE_WITHOUT_CALENDAR_TIMESTAMP_OR_EXPOSURE_TIME",
        "history_definition": "prior resolved assignment arms in deterministic study-order fields; unresolved rows are not recoded",
        "raw_data_redistributed": False,
    }
    return PushDecisionDataset(frame=output, metadata=metadata, validation=validation)


__all__ = [
    "PUSH_DECISION_COLUMNS",
    "PushDecisionDataset",
    "build_mrt_decision_points",
    "validate_push_decision_points",
]
