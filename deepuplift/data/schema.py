from __future__ import annotations

from typing import Any, Iterable

import pandas as pd

from deepuplift.contracts import AssignmentType, CausalDataset, TreatmentType


def infer_treatment_type(series: pd.Series) -> TreatmentType:
    values = series.dropna()
    unique_count = values.nunique()
    if unique_count == 2:
        return TreatmentType.BINARY
    if pd.api.types.is_numeric_dtype(values) and unique_count > 10:
        return TreatmentType.CONTINUOUS
    return TreatmentType.MULTI_DISCRETE


def create_causal_dataset(
    data: Any,
    *,
    feature_cols: Iterable[str],
    treatment_col: str,
    outcome_col: str,
    treatment_type: TreatmentType | str | None = None,
    assignment_type: AssignmentType | str = AssignmentType.RANDOMIZED,
    id_column: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> CausalDataset:
    frame = data if isinstance(data, pd.DataFrame) else data.to_pandas() if hasattr(data, "to_pandas") else data
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("create_causal_dataset accepts pandas-like data with a to_pandas method.")
    features = list(feature_cols)
    if treatment_type is None:
        treatment_type = infer_treatment_type(frame[treatment_col])
    return CausalDataset(
        data=data,
        feature_cols=features,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        treatment_type=treatment_type,
        assignment_type=assignment_type,
        id_column=id_column,
        metadata=dict(metadata or {}),
    )
