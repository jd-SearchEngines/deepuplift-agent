from __future__ import annotations

from typing import Any, Iterable

from deepuplift.contracts import AssignmentType, TreatmentType
from deepuplift.data.schema import create_causal_dataset


def observational_dataset(data: Any, *, feature_cols: Iterable[str], treatment_col: str, outcome_col: str, treatment_type: TreatmentType | str = TreatmentType.BINARY, id_column: str | None = None):
    return create_causal_dataset(data, feature_cols=feature_cols, treatment_col=treatment_col, outcome_col=outcome_col, treatment_type=treatment_type, assignment_type=AssignmentType.OBSERVATIONAL, id_column=id_column)
