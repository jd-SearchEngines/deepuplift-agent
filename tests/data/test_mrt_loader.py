from __future__ import annotations

from pathlib import Path
import zipfile

import pandas as pd
import pytest

from deepuplift.data import audit_mrt_dataset, load_mybpmylife_mrt


def _fixture(tmp_path: Path) -> Path:
    frame = pd.DataFrame(
        {
            "PrefDatetimeRowID": [1, 2, 3, 4],
            "ParticipantIdentifier": [10, 10, 11, 12],
            "TimeEnrollment_to_PrefDate_days": [1, 1, 1, 2],
            "DoesPrefDateHaveRepeat": [0, 0, 0, 0],
            "Phase": ["active"] * 4,
            "Phase_1dayBefore": ["active"] * 4,
            "Phase_1dayAfter": ["active"] * 4,
            "TimeOfDayID": [1, 2, 1, 1],
            "TimeOfDay": ["Morning", "Lunch", "Morning", "Morning"],
            "PrefRepeatRank": [1, 1, 1, 1],
            "DiceRollRowID": [101, 102, 103, None],
            "DiceRollType": ["activity", "diet", "activity", None],
            "IsDiceRollSuccess": [0, 1, 1, None],
            "NotificationRowID": [None, 2, None, None],
            "NotificationType": [None, "diet", None, None],
            "SumValues_Steps_60minAfter": [1, 2, 3, 4],
            "NumValues_LowSaltChoices_24hrAfter": [0, 1, 0, 1],
            "NumValues_Steps_30minBefore": [1, 1, 1, 1],
            "SumValues_Steps_30minBefore": [1, 1, 1, 1],
        }
    )
    directory = tmp_path / "Data_MRT"
    directory.mkdir()
    frame.to_csv(directory / "MRT Data.csv", index=False)
    return directory


def test_mrt_audit_fails_closed_and_keeps_unresolved_rows_out_of_snapshot(tmp_path):
    dataset = load_mybpmylife_mrt(_fixture(tmp_path))
    report = audit_mrt_dataset(dataset)
    assert report["gate_status"] == "PASS"
    assert report["counts"]["raw_rows"] == 4
    assert report["counts"]["analysis_rows"] == 3
    assert report["counts"]["unresolved_assignment_rows"] == 1
    assert report["assignment"]["known_randomization_used"] is True
    assert report["assignment"]["propensity_estimation_used"] is False
    causal = dataset.to_causal_dataset("activity_steps_60min")
    assert len(causal.to_pandas()) == 3
    assert set(causal.to_pandas()["assignment_arm"]) == {"none", "activity", "diet"}


def test_mrt_loader_accepts_zip_and_rejects_invalid_assignment(tmp_path):
    directory = _fixture(tmp_path)
    archive = tmp_path / "Data_MRT.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.write(directory / "MRT Data.csv", arcname="Data_MRT/MRT Data.csv")
    zipped = load_mybpmylife_mrt(archive)
    assert len(zipped.frame) == 4

    invalid = zipped.frame.copy()
    invalid.loc[0, "IsDiceRollSuccess"] = 2
    invalid.to_csv(directory / "MRT Data.csv", index=False)
    report = audit_mrt_dataset(load_mybpmylife_mrt(directory))
    assert report["gate_status"] == "FAIL"
    assert report["failure_mode"] == "DATA_SCHEMA_UNRESOLVED"


def test_mrt_adapter_rejects_future_features(tmp_path):
    dataset = load_mybpmylife_mrt(_fixture(tmp_path))
    with pytest.raises(ValueError, match="post-treatment"):
        dataset.to_causal_dataset("diet_low_salt_24hr", feature_cols=["Phase_1dayAfter"])
