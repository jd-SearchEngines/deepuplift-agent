from __future__ import annotations

import numpy as np
import pandas as pd

from deepuplift.contracts import CausalDataset, EffectPrediction, TreatmentType
from deepuplift.data.preprocessing import TabularPreprocessor
from deepuplift.models.binary.base import OutcomeEstimator


class DoseResponseGBM:
    """Reference dose-response adapter using a dose-conditioned GBM."""

    name = "DoseResponseGBM"

    def __init__(self, task: str = "regression", random_state: int = 42, grid_size: int = 21):
        self.task = task
        self.random_state = random_state
        self.grid_size = max(3, int(grid_size))

    def fit(self, dataset: CausalDataset) -> "DoseResponseGBM":
        if dataset.treatment_type != TreatmentType.CONTINUOUS:
            raise ValueError("DoseResponseGBM requires continuous treatment.")
        frame = dataset.to_pandas().dropna(subset=[dataset.treatment_col, dataset.outcome_col]).reset_index(drop=True)
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
        self.model = OutcomeEstimator(self.task, self.random_state).fit(x, y)
        lower, upper = float(dose.quantile(0.01)), float(dose.quantile(0.99))
        # Dose applications use a non-negative intervention scale. Keep the
        # requested grid size while reserving the first point for no-treatment.
        self.dose_grid = np.linspace(0.0, max(upper, 0.0), self.grid_size)
        self.feature_cols = list(dataset.feature_cols)
        return self

    def predict(self, dataset: CausalDataset) -> EffectPrediction:
        if not getattr(self, "model", None):
            raise RuntimeError("Model must be fitted before predict.")
        base_x = self.preprocessor.transform(dataset.to_pandas()[self.feature_cols])
        predictions = []
        for dose in self.dose_grid:
            x = base_x.copy()
            x["__dose__"] = float(dose)
            predictions.append(self.model.predict(x))
        values = np.vstack(predictions).T
        baseline_idx = int(np.argmin(np.abs(self.dose_grid - 0.0)))
        baseline = values[:, baseline_idx]
        effects = values - baseline[:, None]
        best_idx = effects.argmax(axis=1)
        recommended = values[np.arange(len(values)), best_idx]
        outcome_by_dose = {float(dose): values[:, index] for index, dose in enumerate(self.dose_grid)}
        effect_by_dose = {float(dose): effects[:, index] for index, dose in enumerate(self.dose_grid)}
        return EffectPrediction(
            unit_id=dataset.unit_ids.to_numpy(),
            treatment_type=TreatmentType.CONTINUOUS,
            y0=baseline,
            y1=recommended,
            uplift=recommended - baseline,
            recommended_effect=recommended - baseline,
            recommended_treatment=self.dose_grid[best_idx],
            dose_grid=self.dose_grid,
            dose_outcome_predictions=outcome_by_dose,
            dose_effect_predictions=effect_by_dose,
            metadata={"model_name": self.name, "dose_grid": self.dose_grid.tolist(), "baseline_dose": float(self.dose_grid[baseline_idx]), "maturity": "EXPERIMENTAL", "status": "EXPERIMENTAL", "offline_only": True},
        )
