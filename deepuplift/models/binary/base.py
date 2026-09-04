from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from deepuplift.contracts import CausalDataset, EffectPrediction, TreatmentType
from deepuplift.data.preprocessing import TabularPreprocessor


class OutcomeEstimator:
    def __init__(self, task: str, random_state: int = 42, model: str = "gbm"):
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

        self.task = task
        key = model.lower().replace("_", "-")
        if key in {"random-forest", "rf"}:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            cls = RandomForestClassifier if task == "classification" else RandomForestRegressor
            self.model = cls(n_estimators=120, min_samples_leaf=5, random_state=random_state, n_jobs=-1)
        elif key in {"hist-gradient-boosting", "histgb"}:
            from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
            cls = HistGradientBoostingClassifier if task == "classification" else HistGradientBoostingRegressor
            self.model = cls(random_state=random_state)
        elif key == "lightgbm":
            try:
                from lightgbm import LGBMClassifier, LGBMRegressor
            except ImportError as exc:
                raise ImportError("lightgbm outcome estimation requires the optional 'lightgbm' extra.") from exc
            cls = LGBMClassifier if task == "classification" else LGBMRegressor
            self.model = cls(random_state=random_state, verbosity=-1)
        else:
            self.model = GradientBoostingClassifier(random_state=random_state) if task == "classification" else GradientBoostingRegressor(random_state=random_state, loss="huber")

    def fit(self, x: pd.DataFrame, y: Any, sample_weight: Any | None = None) -> "OutcomeEstimator":
        if sample_weight is None:
            self.model.fit(x, np.asarray(y))
        else:
            self.model.fit(x, np.asarray(y), sample_weight=np.asarray(sample_weight))
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

    def __init__(self, task: str = "classification", random_state: int = 42, outcome_model: str = "gbm"):
        if task not in {"classification", "regression"}:
            raise ValueError("task must be 'classification' or 'regression'.")
        self.task = task
        self.random_state = random_state
        self.outcome_model = outcome_model
        self.preprocessor: TabularPreprocessor | None = None
        self.treatment_mapping: dict[Any, int] = {}
        self.feature_cols: list[str] = []

    def fit(self, dataset: CausalDataset, nuisance=None) -> "BaseBinaryUpliftModel":
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
        if self.task == "classification" and len(np.unique(y)) > 2:
            self.task = "regression"
            self._outcome_task_inferred = True
        if nuisance is None and dataset.assignment_type.value == "observational":
            from deepuplift.data.nuisance import estimate_nuisance
            nuisance = estimate_nuisance(dataset, cross_fit=True, n_splits=5, random_state=self.random_state)
        if nuisance is not None and len(nuisance.propensity_scores) != len(frame):
            raise ValueError("nuisance must contain one estimate per training row")
        self.nuisance = nuisance
        self._fit(x, y, treatment, dataset.assignment_type.value, nuisance)
        return self

    def _fit(self, x: pd.DataFrame, y: np.ndarray, treatment: np.ndarray, assignment_type: str, nuisance=None) -> None:
        raise NotImplementedError

    def _predict_effects(self, x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError

    def _prediction_propensity(self, x: pd.DataFrame, fallback: float) -> np.ndarray:
        nuisance = getattr(self, "nuisance", None)
        if nuisance is not None:
            return np.clip(nuisance.predict_propensity(x), 1e-6, 1 - 1e-6)
        return np.full(len(x), fallback, dtype="float64")

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
            metadata={"model_name": self.name, "task": self.task, "outcome_model": self.outcome_model, "treatment_mapping": self.treatment_mapping, "nuisance_provenance": getattr(getattr(self, "nuisance", None), "metadata", None)},
        )
