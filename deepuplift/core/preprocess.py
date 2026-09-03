from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd
from pandas.api.types import is_numeric_dtype


@dataclass
class TabularPreprocessor:
    feature_cols: List[str]
    categorical_cols: Optional[List[str]] = None
    numeric_cols: List[str] = field(default_factory=list)
    fitted_categorical_cols: List[str] = field(default_factory=list)
    medians: Dict[str, float] = field(default_factory=dict)
    output_feature_cols: List[str] = field(default_factory=list)

    def fit(self, df: pd.DataFrame) -> "TabularPreprocessor":
        missing = [col for col in self.feature_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Feature columns not found: {missing}")

        if self.categorical_cols is None:
            self.fitted_categorical_cols = [
                col for col in self.feature_cols if not is_numeric_dtype(df[col])
            ]
        else:
            self.fitted_categorical_cols = list(self.categorical_cols)

        self.numeric_cols = [
            col for col in self.feature_cols if col not in self.fitted_categorical_cols
        ]
        self.medians = {}
        for col in self.numeric_cols:
            values = pd.to_numeric(df[col], errors="coerce")
            median = values.median()
            self.medians[col] = 0.0 if pd.isna(median) else float(median)

        transformed = self._transform_unaligned(df)
        self.output_feature_cols = transformed.columns.tolist()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        transformed = self._transform_unaligned(df)
        return transformed.reindex(columns=self.output_feature_cols, fill_value=0.0)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)

    def _transform_unaligned(self, df: pd.DataFrame) -> pd.DataFrame:
        frames = []

        if self.numeric_cols:
            numeric = pd.DataFrame(index=df.index)
            for col in self.numeric_cols:
                median = self.medians.get(col, 0.0)
                numeric[col] = pd.to_numeric(df[col], errors="coerce").fillna(median)
            frames.append(numeric.astype("float32"))

        if self.fitted_categorical_cols:
            categorical = pd.DataFrame(index=df.index)
            for col in self.fitted_categorical_cols:
                values = df[col].astype("object").where(df[col].notna(), "__missing__")
                categorical[col] = values.astype(str)
            frames.append(pd.get_dummies(categorical, prefix=self.fitted_categorical_cols, dtype="float32"))

        if not frames:
            raise ValueError("No usable feature columns after preprocessing.")

        return pd.concat(frames, axis=1)
