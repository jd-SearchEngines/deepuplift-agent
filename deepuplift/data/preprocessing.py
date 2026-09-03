from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class TabularPreprocessor:
    """Small, deterministic pandas preprocessor for model adapters."""

    feature_cols: List[str]
    categorical_cols: Optional[List[str]] = None
    numeric_cols: List[str] = field(default_factory=list)
    fitted_categorical_cols: List[str] = field(default_factory=list)
    medians: Dict[str, float] = field(default_factory=dict)
    output_feature_cols: List[str] = field(default_factory=list)

    def fit(self, frame: pd.DataFrame) -> "TabularPreprocessor":
        missing = [col for col in self.feature_cols if col not in frame.columns]
        if missing:
            raise ValueError(f"Feature columns not found: {missing}")
        self.fitted_categorical_cols = (
            list(self.categorical_cols)
            if self.categorical_cols is not None
            else [col for col in self.feature_cols if not pd.api.types.is_numeric_dtype(frame[col])]
        )
        self.numeric_cols = [col for col in self.feature_cols if col not in self.fitted_categorical_cols]
        self.medians = {}
        for col in self.numeric_cols:
            values = pd.to_numeric(frame[col], errors="coerce")
            median = values.median()
            self.medians[col] = 0.0 if pd.isna(median) else float(median)
        self.output_feature_cols = self._transform_unaligned(frame).columns.tolist()
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.output_feature_cols:
            raise RuntimeError("TabularPreprocessor must be fitted before transform.")
        return self._transform_unaligned(frame).reindex(columns=self.output_feature_cols, fill_value=0.0)

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        self.fit(frame)
        return self.transform(frame)

    def _transform_unaligned(self, frame: pd.DataFrame) -> pd.DataFrame:
        parts = []
        if self.numeric_cols:
            numeric = pd.DataFrame(index=frame.index)
            for col in self.numeric_cols:
                values = pd.to_numeric(frame[col], errors="coerce")
                numeric[col] = values.fillna(self.medians.get(col, 0.0))
            parts.append(numeric.astype("float32"))
        if self.fitted_categorical_cols:
            categorical = pd.DataFrame(index=frame.index)
            for col in self.fitted_categorical_cols:
                categorical[col] = frame[col].astype("object").where(frame[col].notna(), "__missing__").astype(str)
            parts.append(pd.get_dummies(categorical, prefix=self.fitted_categorical_cols, dtype="float32"))
        if not parts:
            raise ValueError("No usable feature columns after preprocessing.")
        return pd.concat(parts, axis=1)
