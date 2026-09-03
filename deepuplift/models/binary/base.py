from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from deepuplift.contracts import CausalDataset, EffectPrediction, TreatmentType
from deepuplift.data.preprocessing import TabularPreprocessor


class OutcomeEstimator:
    def __init__(self, task: str, random_state: int = 42):
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

        self.task = task
        self.model = (
            GradientBoostingClassifier(random_state=random_state)
            if task == "classification"
            else GradientBoostingRegressor(random_state=random_state, loss="huber")
        )

    def fit(self, x: pd.DataFrame, y: Any) -> "OutcomeEstimator":
        self.model.fit(x, np.asarray(y))
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        if self.task == "classification":
            return np.asarray(self.model.predict_proba(x)[:, 1], dtype="float64")
        return np.asarray(self.model.predict(x), dtype="float64")


def normalize_binary_treatment(series: pd.Series) -> tuple[np.ndarray, dict[Any, int]]:
    values = series.dropna().unique().tolist()
    if len(values) != 2:
        raise ValueError(f"Binary uplift models require exactly two treatment values; got {values}.")
    ordered = sorted(values, key=lambda item: str(item))
    mapping = {ordered[0]: 0, ordered[1]: 1}
    return series.map(mapping).to_numpy(dtype="float64"), mapping


class BaseBinaryUpliftModel:
    name = "BaseBinaryUpliftModel"

    def __init__(self, task: str = "classification", random_state: int = 42):
        if task not in {"classification", "regression"}:
            raise ValueError("task must be 'classification' or 'regression'.")
        self.task = task
        self.random_state = random_state
        self.preprocessor: TabularPreprocessor | None = None
        self.treatment_mapping: dict[Any, int] = {}
        self.feature_cols: list[str] = []

    def fit(self, dataset: CausalDataset) -> "BaseBinaryUpliftModel":
        if dataset.treatment_type != TreatmentType.BINARY:
            raise ValueError(f"{self.name} supports binary treatment only.")
        frame = dataset.to_pandas().dropna(subset=[dataset.treatment_col, dataset.outcome_col]).reset_index(drop=True)
        treatment, mapping = normalize_binary_treatment(frame[dataset.treatment_col])
        if len(np.unique(treatment)) != 2:
            raise ValueError("Both treatment arms are required for fitting.")
        self.treatment_mapping = mapping
        self.feature_cols = list(dataset.feature_cols)
        self.preprocessor = TabularPreprocessor(self.feature_cols)
        x = self.preprocessor.fit_transform(frame[self.feature_cols])
        y = pd.to_numeric(frame[dataset.outcome_col], errors="coerce").to_numpy(dtype="float64")
        if np.isnan(y).any():
            raise ValueError("Outcome must be numeric for the reference model adapters.")
        self._fit(x, y, treatment, dataset.assignment_type.value)
        return self

    def _fit(self, x: pd.DataFrame, y: np.ndarray, treatment: np.ndarray, assignment_type: str) -> None:
        raise NotImplementedError

    def _predict_effects(self, x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError

    def predict(self, dataset: CausalDataset) -> EffectPrediction:
        if self.preprocessor is None:
            raise RuntimeError("Model must be fitted before predict.")
        frame = dataset.to_pandas()
        x = self.preprocessor.transform(frame[self.feature_cols])
        y0, y1, propensity = self._predict_effects(x)
        return EffectPrediction(
            unit_id=dataset.unit_ids.to_numpy(),
            treatment_type=TreatmentType.BINARY,
            uplift=y1 - y0,
            cate=y1 - y0,
            y0=y0,
            y1=y1,
            recommended_effect=y1 - y0,
            propensity=propensity,
            metadata={"model_name": self.name, "task": self.task, "treatment_mapping": self.treatment_mapping},
        )
