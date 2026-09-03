from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import pandas as pd

from .treatment import AssignmentType, TreatmentType, coerce_assignment_type, coerce_treatment_type


@dataclass
class CausalDataset:
    """Backend-neutral description of a causal/uplift dataset.

    ``data`` may be a pandas frame, a Polars frame/lazy frame, or another
    backend object implementing ``to_pandas``. The contract stores column
    semantics separately so model and decision code does not infer them from
    a concrete dataframe library.
    """

    data: Any
    feature_cols: List[str]
    treatment_col: str
    outcome_col: str
    treatment_type: TreatmentType | str = TreatmentType.BINARY
    assignment_type: AssignmentType | str = AssignmentType.RANDOMIZED
    id_column: str | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.treatment_type = coerce_treatment_type(self.treatment_type)
        self.assignment_type = coerce_assignment_type(self.assignment_type)
        self.feature_cols = list(self.feature_cols)
        columns = getattr(self.data, "columns", None)
        if columns is not None:
            missing = [
                col
                for col in [self.treatment_col, self.outcome_col, *self.feature_cols]
                if col not in columns
            ]
            if self.id_column and self.id_column not in columns:
                missing.append(self.id_column)
            if missing:
                raise ValueError(f"CausalDataset columns not found: {sorted(set(missing))}")

    def to_pandas(self) -> pd.DataFrame:
        if isinstance(self.data, pd.DataFrame):
            return self.data.copy()
        if hasattr(self.data, "to_pandas"):
            return self.data.to_pandas()
        if hasattr(self.data, "collect"):
            collected = self.data.collect()
            if hasattr(collected, "to_pandas"):
                return collected.to_pandas()
        raise TypeError("CausalDataset data must be a pandas-compatible backend object.")

    @property
    def features(self) -> pd.DataFrame:
        return self.to_pandas()[self.feature_cols].copy()

    @property
    def treatment(self) -> pd.Series:
        return self.to_pandas()[self.treatment_col].copy()

    @property
    def outcome(self) -> pd.Series:
        return self.to_pandas()[self.outcome_col].copy()

    @property
    def unit_ids(self) -> pd.Series:
        frame = self.to_pandas()
        if self.id_column and self.id_column in frame.columns:
            return frame[self.id_column].copy()
        return pd.Series(frame.index, index=frame.index, name="unit_id")

    def subset(self, indices: Any) -> "CausalDataset":
        frame = self.to_pandas().iloc[indices].reset_index(drop=True)
        return CausalDataset(
            data=frame,
            feature_cols=self.feature_cols,
            treatment_col=self.treatment_col,
            outcome_col=self.outcome_col,
            treatment_type=self.treatment_type,
            assignment_type=self.assignment_type,
            id_column=self.id_column,
            metadata=dict(self.metadata),
        )

    def with_metadata(self, **values: Any) -> "CausalDataset":
        metadata = dict(self.metadata)
        metadata.update(values)
        return CausalDataset(
            data=self.to_pandas(),
            feature_cols=self.feature_cols,
            treatment_col=self.treatment_col,
            outcome_col=self.outcome_col,
            treatment_type=self.treatment_type,
            assignment_type=self.assignment_type,
            id_column=self.id_column,
            metadata=metadata,
        )
