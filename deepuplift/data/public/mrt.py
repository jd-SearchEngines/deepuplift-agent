"""Local-only myBPmyLife Push MRT loader and P0 audit.

The upstream release is a repeated-decision table, not a one-row-per-user
dataset.  This module deliberately keeps the longitudinal fields in a
separate contract and only exposes a filtered ``CausalDataset`` adapter after
assignment rows have been resolved from explicit upstream fields.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np
import pandas as pd

from deepuplift.contracts import AssignmentType, CausalDataset, TreatmentType
from deepuplift.data.schema import create_causal_dataset


MRT_SOURCE_URL = "https://github.com/WIRED-L-um/MRT_myBPmyLife"
MRT_DECLARED_ASSIGNMENT_PROBABILITIES = {
    "none": 0.75,
    "activity": 0.125,
    "diet": 0.125,
}

_MRT_REQUIRED = (
    "PrefDatetimeRowID",
    "ParticipantIdentifier",
    "TimeEnrollment_to_PrefDate_days",
    "TimeOfDayID",
    "TimeOfDay",
    "PrefRepeatRank",
    "DiceRollRowID",
    "DiceRollType",
    "IsDiceRollSuccess",
    "NotificationRowID",
    "NotificationType",
)
_ACTIVITY_OUTCOME = "SumValues_Steps_60minAfter"
_DIET_OUTCOME = "NumValues_LowSaltChoices_24hrAfter"
_PRETREATMENT_CANDIDATES = (
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
_POTENTIAL_LEAKAGE_COLUMNS = (
    "Phase_1dayAfter",
    "SumValues_WearTime_hr_60minAfter",
    "SumValues_Steps_60minAfter",
    "NumValues_LowSaltChoices_60minAfter",
    "SumValues_WearTime_hr_03hrAfter",
    "SumValues_Steps_03hrAfter",
    "NumValues_LowSaltChoices_03hrAfter",
    "SumValues_WearTime_hr_24hrAfter",
    "SumValues_Steps_24hrAfter",
    "NumValues_LowSaltChoices_24hrAfter",
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if pd.isna(value):
        return None
    return value


def _read_csv_from_zip(archive: zipfile.ZipFile, filename: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    candidates = [name for name in archive.namelist() if Path(name).name.lower() == filename.lower()]
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected exactly one {filename!r} in the MRT archive; found {candidates}.")
    name = candidates[0]
    payload = archive.read(name)
    return pd.read_csv(BytesIO(payload)), {
        "name": name,
        "bytes": len(payload),
        "sha256": _sha256_bytes(payload),
    }


def _read_local_csv(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    payload = path.read_bytes()
    return pd.read_csv(BytesIO(payload)), {
        "name": path.name,
        "path": str(path),
        "bytes": len(payload),
        "sha256": _sha256_bytes(payload),
    }


def _source_frames(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = Path(path).expanduser()
    if not source.exists():
        raise FileNotFoundError(
            f"myBPmyLife MRT source was not found at {source}. "
            f"Obtain it from {MRT_SOURCE_URL}; raw data is not redistributed."
        )
    source_files: dict[str, Any] = {"source_path": str(source), "source_url": MRT_SOURCE_URL}
    if source.is_file() and source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            frame, mrt_info = _read_csv_from_zip(archive, "MRT Data.csv")
            source_files["archive"] = {"path": str(source), "bytes": source.stat().st_size, "sha256": _sha256_file(source)}
            source_files["mrt_csv"] = mrt_info
            try:
                moderator, moderator_info = _read_csv_from_zip(archive, "Moderator.csv")
            except FileNotFoundError:
                moderator = None
            else:
                source_files["moderator_csv"] = moderator_info
    elif source.is_dir():
        candidate = source / "MRT Data.csv"
        if not candidate.exists():
            candidate = source / "Data_MRT" / "MRT Data.csv"
        if not candidate.exists():
            raise FileNotFoundError(f"Could not find 'MRT Data.csv' below {source}.")
        frame, mrt_info = _read_local_csv(candidate)
        source_files["mrt_csv"] = mrt_info
        moderator_path = candidate.with_name("Moderator.csv")
        if moderator_path.exists():
            moderator, moderator_info = _read_local_csv(moderator_path)
            source_files["moderator_csv"] = moderator_info
        else:
            moderator = None
    elif source.suffix.lower() == ".csv":
        frame, mrt_info = _read_local_csv(source)
        source_files["mrt_csv"] = mrt_info
        moderator = None
    else:
        raise ValueError("myBPmyLife MRT path must be a directory, CSV, or ZIP archive.")
    if moderator is not None:
        join_keys = ["PrefDatetimeRowID"]
        if "DiceRollRowID" in moderator.columns and "DiceRollRowID" in frame.columns:
            join_keys.append("DiceRollRowID")
        if any(key not in moderator.columns for key in join_keys):
            raise ValueError(f"Moderator.csv is missing the join keys {join_keys!r}.")
        extra = [column for column in moderator.columns if column not in join_keys and column not in frame.columns]
        if extra:
            frame = frame.merge(moderator[[*join_keys, *extra]], on=join_keys, how="left", validate="many_to_one")
            source_files["moderator_join_columns"] = extra
    return frame, source_files


def _derive_assignment(frame: pd.DataFrame) -> pd.Series:
    success = pd.to_numeric(frame["IsDiceRollSuccess"], errors="coerce")
    dice_type = frame["DiceRollType"].astype("string").str.lower()
    resolved = success.isin([0.0, 1.0]) & dice_type.isin(["activity", "diet"])
    assignment = pd.Series(pd.NA, index=frame.index, dtype="string")
    assignment.loc[resolved & success.eq(0.0)] = "none"
    assignment.loc[resolved & success.eq(1.0)] = dice_type.loc[resolved & success.eq(1.0)]
    return assignment


@dataclass
class MRTDataset:
    """Longitudinal Push MRT table with explicit assignment and delivery fields."""

    frame: pd.DataFrame
    source_files: dict[str, Any]
    license_status: str = "LICENSE_UNVERIFIED"

    def __post_init__(self) -> None:
        self.frame = self.frame.copy()
        missing = sorted(set(_MRT_REQUIRED) - set(self.frame.columns))
        if missing:
            raise ValueError(f"MRT Data.csv is missing required columns: {missing}")
        self.frame["assignment_arm"] = _derive_assignment(self.frame)
        self.frame["delivery_observed"] = self.frame["NotificationType"].notna()
        self.frame["delivery_arm"] = self.frame["NotificationType"].astype("string").str.lower()
        self.frame["mrt_decision_id"] = self.frame["DiceRollRowID"]
        self.frame["participant_id"] = self.frame["ParticipantIdentifier"]

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "name": "myBPmyLife_push_mrt",
            "source": MRT_SOURCE_URL,
            "license": self.license_status,
            "raw_data_redistributed": False,
            "assignment_estimand": "intent_to_treat_by_dice_assignment",
            "cluster_column": "participant_id",
            "decision_column": "mrt_decision_id",
        }

    def analysis_frame(self) -> pd.DataFrame:
        """Return only rows with an explicit, valid dice assignment.

        Unresolved rows are intentionally excluded.  They remain available in
        ``frame`` and are counted by the audit instead of being recoded as no
        notification.
        """

        return self.frame.loc[self.frame["assignment_arm"].notna()].copy()

    def to_causal_dataset(
        self,
        outcome: str,
        *,
        feature_cols: Sequence[str] | None = None,
    ) -> CausalDataset:
        """Create a safe multi-arm adapter for one proximal outcome.

        This adapter preserves repeated decisions and puts the participant
        cluster in metadata.  It never uses delivery as the treatment column.
        """

        specs = {
            "activity_steps_60min": (_ACTIVITY_OUTCOME, "activity notification -> steps in the following 60 minutes"),
            "diet_low_salt_24hr": (_DIET_OUTCOME, "dietary notification -> low-sodium choices in the following 24 hours"),
        }
        if outcome not in specs:
            raise ValueError(f"Unknown MRT outcome {outcome!r}; expected {sorted(specs)}.")
        outcome_column, outcome_definition = specs[outcome]
        if outcome_column not in self.frame.columns:
            raise ValueError(f"MRT outcome column {outcome_column!r} is not available.")
        frame = self.analysis_frame().reset_index(drop=True)
        frame["mrt_outcome"] = pd.to_numeric(frame[outcome_column], errors="coerce")
        default_features = [column for column in _PRETREATMENT_CANDIDATES if column in frame.columns]
        selected = list(feature_cols or default_features)
        invalid = sorted(set(selected) - set(frame.columns))
        if invalid:
            raise ValueError(f"Unknown MRT feature columns: {invalid}")
        if any(column in _POTENTIAL_LEAKAGE_COLUMNS for column in selected):
            raise ValueError("MRT adapter rejects post-treatment or future feature columns.")
        return create_causal_dataset(
            frame,
            feature_cols=selected,
            treatment_col="assignment_arm",
            outcome_col="mrt_outcome",
            treatment_type=TreatmentType.MULTI_DISCRETE,
            assignment_type=AssignmentType.RANDOMIZED,
            id_column="mrt_decision_id",
            metadata={
                **self.metadata,
                "outcome_name": outcome,
                "outcome_definition": outcome_definition,
                "outcome_source_column": outcome_column,
                "outcome_window": "60 minutes after decision" if outcome.startswith("activity") else "24 hours after decision",
                "participant_column": "participant_id",
                "declared_assignment_probabilities": dict(MRT_DECLARED_ASSIGNMENT_PROBABILITIES),
                "delivery_column": "delivery_arm",
                "raw_rows": int(len(self.frame)),
                "analysis_rows": int(len(frame)),
                "unresolved_rows_excluded": int(self.frame["assignment_arm"].isna().sum()),
            },
        )


def load_mybpmylife_mrt(path_or_directory: str | Path) -> MRTDataset:
    """Load the upstream MRT table from a local directory, CSV, or ZIP.

    No network access or implicit download occurs.  The loader reads the data
    dictionary-compatible upstream column names and records file hashes.
    """

    frame, source_files = _source_frames(path_or_directory)
    return MRTDataset(frame=frame, source_files=source_files)


def _summary_stats(values: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype="float64")
    if not len(numeric):
        return {"count": 0}
    return {
        "count": int(len(numeric)),
        "min": float(np.min(numeric)),
        "p25": float(np.quantile(numeric, 0.25)),
        "median": float(np.median(numeric)),
        "p75": float(np.quantile(numeric, 0.75)),
        "max": float(np.max(numeric)),
    }


def _standardized_mean_difference(left: pd.Series, right: pd.Series) -> float | None:
    a = pd.to_numeric(left, errors="coerce").dropna().to_numpy(dtype="float64")
    b = pd.to_numeric(right, errors="coerce").dropna().to_numpy(dtype="float64")
    if not len(a) or not len(b):
        return None
    a_var = np.var(a, ddof=1) if len(a) > 1 else 0.0
    b_var = np.var(b, ddof=1) if len(b) > 1 else 0.0
    pooled = float(np.sqrt((a_var + b_var) / 2.0))
    return float((np.mean(b) - np.mean(a)) / pooled) if pooled > 0 else 0.0


def participant_group_split(
    dataset: MRTDataset,
    *,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return positional analysis-row indices with participant isolation."""

    frame = dataset.analysis_frame().reset_index(drop=True)
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")
    participants = np.sort(frame["participant_id"].dropna().unique())
    rng = np.random.default_rng(seed)
    shuffled = participants.copy()
    rng.shuffle(shuffled)
    test_count = min(max(1, int(round(len(participants) * test_fraction))), len(participants) - 1)
    test_participants = shuffled[:test_count]
    test_mask = frame["participant_id"].isin(test_participants).to_numpy()
    return np.flatnonzero(~test_mask), np.flatnonzero(test_mask)


def _participant_split(frame: pd.DataFrame, *, test_fraction: float, seed: int) -> dict[str, Any]:
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")
    participants = np.sort(frame["participant_id"].dropna().unique())
    rng = np.random.default_rng(seed)
    shuffled = participants.copy()
    rng.shuffle(shuffled)
    test_count = min(max(1, int(round(len(participants) * test_fraction))), len(participants) - 1)
    test_participants = shuffled[:test_count]
    test_mask = frame["participant_id"].isin(test_participants).to_numpy()
    return {
        "method": "participant_group_holdout",
        "seed": int(seed),
        "test_fraction": float(test_fraction),
        "train_rows": int((~test_mask).sum()),
        "test_rows": int(test_mask.sum()),
        "train_participants": int(frame.loc[~test_mask, "participant_id"].nunique()),
        "test_participants": int(frame.loc[test_mask, "participant_id"].nunique()),
        "participant_overlap": int(len(set(frame.loc[~test_mask, "participant_id"]) & set(frame.loc[test_mask, "participant_id"]))),
    }


def audit_mrt_dataset(
    dataset: MRTDataset | str | Path,
    *,
    split_seed: int = 42,
    test_fraction: float = 0.2,
) -> dict[str, Any]:
    """Produce a non-sensitive, reproducible P0 schema/readiness audit."""

    mrt = load_mybpmylife_mrt(dataset) if isinstance(dataset, (str, Path)) else dataset
    frame = mrt.frame
    analysis = mrt.analysis_frame()
    success = pd.to_numeric(frame["IsDiceRollSuccess"], errors="coerce")
    dice_type = frame["DiceRollType"].astype("string").str.lower()
    invalid_success = int((success.notna() & ~success.isin([0.0, 1.0])).sum())
    invalid_type = int((dice_type.notna() & ~dice_type.isin(["activity", "diet"])).sum())
    invalid_assignment_rows = int((invalid_success + invalid_type) > 0)
    unresolved_rows = int(frame["assignment_arm"].isna().sum())
    counts = analysis["assignment_arm"].value_counts().reindex(["none", "activity", "diet"], fill_value=0)
    empirical = (counts / max(len(analysis), 1)).astype(float).to_dict()
    probability_delta = {arm: float(empirical[arm] - probability) for arm, probability in MRT_DECLARED_ASSIGNMENT_PROBABILITIES.items()}
    delivery_invalid = int((frame["delivery_arm"].notna() & ~frame["delivery_arm"].isin(["activity", "diet"])).sum())
    participant_counts = analysis.groupby("participant_id").size()
    numeric_features = [column for column in _PRETREATMENT_CANDIDATES if column in analysis.columns and pd.api.types.is_numeric_dtype(analysis[column])]
    balance: dict[str, Any] = {}
    for arm in ("activity", "diet"):
        balance[arm] = {
            column: _standardized_mean_difference(
                analysis.loc[analysis["assignment_arm"] == "none", column],
                analysis.loc[analysis["assignment_arm"] == arm, column],
            )
            for column in numeric_features
        }
    outcome_fields = {
        "activity_steps_60min": {
            "column": _ACTIVITY_OUTCOME,
            "window": "60 minutes after decision",
            "missing_rate": float(frame[_ACTIVITY_OUTCOME].isna().mean()) if _ACTIVITY_OUTCOME in frame else None,
        },
        "diet_low_salt_24hr": {
            "column": _DIET_OUTCOME,
            "window": "24 hours after decision",
            "missing_rate": float(frame[_DIET_OUTCOME].isna().mean()) if _DIET_OUTCOME in frame else None,
        },
    }
    key = ["participant_id", "PrefDatetimeRowID", "DiceRollRowID"]
    duplicate_exact_keys = int(frame.duplicated(key, keep=False).sum())
    duplicate_pref_keys = int(frame.duplicated(["participant_id", "PrefDatetimeRowID"], keep=False).sum())
    decision_non_null = frame["DiceRollRowID"].notna()
    duplicate_decision_ids = int(frame.loc[decision_non_null, "DiceRollRowID"].duplicated(keep=False).sum())
    order_frame = frame[["participant_id", "TimeEnrollment_to_PrefDate_days", "TimeOfDayID", "PrefRepeatRank"]].copy()
    order_frame["source_order"] = np.arange(len(order_frame))
    ordering_violations = 0
    for _, group in order_frame.groupby("participant_id", sort=False):
        keys = group[["TimeEnrollment_to_PrefDate_days", "TimeOfDayID", "PrefRepeatRank"]].apply(tuple, axis=1).tolist()
        ordering_violations += sum(left > right for left, right in zip(keys, keys[1:]))
    split = _participant_split(analysis, test_fraction=test_fraction, seed=split_seed) if len(analysis) else {}
    audit = {
        "schema_version": 1,
        "status": "PASS" if not invalid_assignment_rows and not delivery_invalid and len(analysis) else "DATA_SCHEMA_UNRESOLVED",
        "gate_status": "PASS" if not invalid_assignment_rows and not delivery_invalid and len(analysis) else "FAIL",
        "failure_mode": None if not invalid_assignment_rows and not delivery_invalid else "DATA_SCHEMA_UNRESOLVED",
        "source": {**mrt.metadata, "files": mrt.source_files},
        "license_status": mrt.license_status,
        "schema": {
            "required_columns": list(_MRT_REQUIRED),
            "missing_required_columns": sorted(set(_MRT_REQUIRED) - set(frame.columns)),
            "assignment_mapping": "assignment_arm = none when IsDiceRollSuccess=0; DiceRollType when IsDiceRollSuccess=1; unresolved otherwise",
            "delivery_mapping": "delivery_arm = NotificationType when present; kept separate from assignment_arm",
            "pre_treatment_features": [column for column in _PRETREATMENT_CANDIDATES if column in frame.columns],
            "potential_leakage_columns": [column for column in _POTENTIAL_LEAKAGE_COLUMNS if column in frame.columns],
            "timestamp_status": "NO_CALENDAR_TIMESTAMP_IN_UPSTREAM_TABLE; ordinal study day/time-of-day fields preserved",
        },
        "counts": {
            "raw_rows": int(len(frame)),
            "analysis_rows": int(len(analysis)),
            "unresolved_assignment_rows": unresolved_rows,
            "participants_raw": int(frame["participant_id"].nunique()),
            "participants_analysis": int(analysis["participant_id"].nunique()),
            "participant_decision_counts": _summary_stats(participant_counts),
            "duplicate_exact_decision_keys": duplicate_exact_keys,
            "duplicate_preference_keys": duplicate_pref_keys,
            "duplicate_non_null_decision_ids": duplicate_decision_ids,
        },
        "assignment": {
            "declared_probabilities": dict(MRT_DECLARED_ASSIGNMENT_PROBABILITIES),
            "empirical_probabilities_analysis_rows": {key: float(value) for key, value in empirical.items()},
            "empirical_minus_declared": probability_delta,
            "counts_analysis_rows": {str(key): int(value) for key, value in counts.items()},
            "invalid_success_values": invalid_success,
            "invalid_dice_type_values": invalid_type,
            "known_randomization_used": True,
            "propensity_estimation_used": False,
            "unresolved_policy": "FAIL_CLOSED_EXCLUDE_FROM_ANALYSIS_SNAPSHOT",
        },
        "delivery": {
            "delivery_rate_all_rows": float(frame["delivery_observed"].mean()),
            "delivery_rate_assigned_notification_rows": float(analysis.loc[analysis["assignment_arm"].isin(["activity", "diet"]), "delivery_observed"].mean()),
            "invalid_delivery_values": delivery_invalid,
        },
        "outcomes": outcome_fields,
        "missingness": {
            column: float(frame[column].isna().mean())
            for column in [*(_PRETREATMENT_CANDIDATES), _ACTIVITY_OUTCOME, _DIET_OUTCOME]
            if column in frame.columns
        },
        "pre_treatment_balance_smd_vs_none": balance,
        "time_ordering": {
            "ordering_fields": ["TimeEnrollment_to_PrefDate_days", "TimeOfDayID", "PrefRepeatRank"],
            "within_participant_adjacent_ordering_violations": int(ordering_violations),
            "calendar_timestamp_available": False,
        },
        "split": split,
        "warnings": [
            "REAL_RANDOMIZED_MRT; rows are repeated decision points clustered by participant.",
            "No individual counterfactual outcome is observed; PEHE is not available for this dataset.",
            "Upstream repository did not include a LICENSE file; redistribution terms remain unverified.",
            "4,482 rows without explicit dice assignment are retained for audit but excluded from the analysis snapshot.",
        ],
    }
    return json.loads(json.dumps(audit, default=_json_safe))


def export_mrt_causal_snapshot(
    dataset: MRTDataset | str | Path,
    output_path: str | Path,
    *,
    outcome: str,
    feature_cols: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Explicitly export an analysis snapshot; never called implicitly."""

    mrt = load_mybpmylife_mrt(dataset) if isinstance(dataset, (str, Path)) else dataset
    causal = mrt.to_causal_dataset(outcome, feature_cols=feature_cols)
    destination = Path(output_path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame = causal.to_pandas()
    if destination.suffix.lower() == ".parquet":
        frame.to_parquet(destination, index=False)
    else:
        frame.to_csv(destination, index=False)
    return {
        "path": str(destination),
        "rows": int(len(frame)),
        "outcome": outcome,
        "raw_data_redistributed": False,
        "assignment_estimand": "intent_to_treat_by_dice_assignment",
    }


__all__ = [
    "MRTDataset",
    "MRT_DECLARED_ASSIGNMENT_PROBABILITIES",
    "MRT_SOURCE_URL",
    "audit_mrt_dataset",
    "export_mrt_causal_snapshot",
    "load_mybpmylife_mrt",
    "participant_group_split",
]
