from __future__ import annotations

import numpy as np
import pandas as pd

from deepuplift.contracts import CausalDataset, EffectPrediction, TreatmentType
from deepuplift.data.preprocessing import TabularPreprocessor
from deepuplift.models.binary.base import OutcomeEstimator
from .contracts import effect_prediction_from_curve, resolve_dose_support


class DoseResponseGBM:
    """Reference dose-response adapter using a dose-conditioned GBM."""

    name = "DoseResponseGBM"

    def __init__(self, task: str = "regression", random_state: int = 42, grid_size: int = 21,
                 baseline_dose: float | None = None, dose_grid=None, allow_extrapolation: bool = False):
        self.task = task
        self.random_state = random_state
        self.grid_size = max(3, int(grid_size))
        self.requested_baseline_dose = baseline_dose
        self.requested_dose_grid = dose_grid
        self.allow_extrapolation = bool(allow_extrapolation)

    def fit(self, dataset: CausalDataset, *, sample_weight=None, warm_start: bool = False) -> "DoseResponseGBM":
        if dataset.treatment_type != TreatmentType.CONTINUOUS:
            raise ValueError("DoseResponseGBM requires continuous treatment.")
        frame = dataset.to_pandas()
        keep = frame[[dataset.treatment_col, dataset.outcome_col]].notna().all(axis=1).to_numpy()
        weights = None
        if sample_weight is not None:
            supplied_weights = np.asarray(sample_weight, dtype="float64").reshape(-1)
            if len(supplied_weights) != len(frame):
                raise ValueError("sample_weight must match the number of dataset rows.")
            weights = supplied_weights[keep]
        frame = frame.loc[keep].reset_index(drop=True)
        dose = pd.to_numeric(frame[dataset.treatment_col], errors="coerce")
        if dose.nunique() < 3:
            raise ValueError("Continuous treatment needs at least three distinct numeric doses.")
        self.preprocessor = TabularPreprocessor(dataset.feature_cols)
        x = self.preprocessor.fit_transform(frame[dataset.feature_cols])
        x["__dose__"] = dose.to_numpy(dtype="float64")
        y = pd.to_numeric(frame[dataset.outcome_col], errors="coerce").to_numpy(dtype="float64")
        if np.isnan(y).any():
            raise ValueError("Outcome must be numeric for DoseResponseGBM.")
        if self.task == "classification" and len(np.unique(y)) > 2:
            self.task = "regression"
        self.model = OutcomeEstimator(self.task, self.random_state).fit(x, y, sample_weight=weights)
        self.observed_dose_min, self.observed_dose_max = float(dose.min()), float(dose.max())
        self.no_treatment_observed = bool(np.isclose(dose.to_numpy(dtype="float64"), 0.0, atol=1e-12).any())
        self.dose_grid, self.baseline_dose, self.dose_support = resolve_dose_support(
            self.observed_dose_min, self.observed_dose_max, grid_size=self.grid_size,
            baseline_dose=self.requested_baseline_dose, dose_grid=self.requested_dose_grid,
            allow_extrapolation=self.allow_extrapolation,
            no_treatment_observed=self.no_treatment_observed,
        )
        self.feature_cols = list(dataset.feature_cols)
        return self

    def _predict_at_raw_features(self, features: pd.DataFrame, doses: np.ndarray) -> np.ndarray:
        if not getattr(self, "model", None):
            raise RuntimeError("Model must be fitted before prediction.")
        x = self.preprocessor.transform(features[self.feature_cols])
        x["__dose__"] = np.asarray(doses, dtype="float64").reshape(-1)
        return self.model.predict(x)

    def _predict_dose_gradient(self, features: pd.DataFrame, doses: np.ndarray) -> np.ndarray:
        doses = np.asarray(doses, dtype="float64").reshape(-1)
        step = max(float(np.ptp(self.dose_grid)) * 1e-4, 1e-5)
        right = self._predict_at_raw_features(features, doses + step)
        left = self._predict_at_raw_features(features, doses - step)
        return (right - left) / (2.0 * step)

    def predict(self, dataset: CausalDataset, *, dose_grid=None) -> EffectPrediction:
        if not getattr(self, "model", None):
            raise RuntimeError("Model must be fitted before predict.")
        base_x = self.preprocessor.transform(dataset.to_pandas()[self.feature_cols])
        grid, baseline, support = resolve_dose_support(
            self.observed_dose_min, self.observed_dose_max, grid_size=self.grid_size,
            baseline_dose=self.baseline_dose, dose_grid=self.dose_grid if dose_grid is None else dose_grid,
            allow_extrapolation=self.allow_extrapolation,
            no_treatment_observed=self.no_treatment_observed,
        )
        predictions = []
        for dose in grid:
            x = base_x.copy()
            x["__dose__"] = float(dose)
            predictions.append(self.model.predict(x))
        values = np.vstack(predictions).T
        return effect_prediction_from_curve(
            unit_id=dataset.unit_ids.to_numpy(),
            dose_grid=grid,
            dose_outcomes=values,
            baseline_dose=baseline,
            metadata={
                "model_name": self.name,
                "maturity": "EXPERIMENTAL",
                "status": "EXPERIMENTAL",
                "offline_only": True,
                "dose_grid": grid.tolist(),
                "observed_dose_min": self.observed_dose_min,
                "observed_dose_max": self.observed_dose_max,
                "dose_support": support,
            },
        )
