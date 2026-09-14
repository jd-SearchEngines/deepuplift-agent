from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pandas as pd

from deepuplift.contracts import CausalDataset, EffectPrediction, TreatmentType
from deepuplift.data.preprocessing import TabularPreprocessor
from .contracts import effect_prediction_from_curve


def _require_torch():
    try:
        return importlib.import_module("torch")
    except ImportError as exc:
        raise ImportError("This continuous neural estimator requires PyTorch; install deepuplift[continuous].") from exc


class TorchContinuousEstimator:
    """Shared fit/predict mechanics for optional PyTorch continuous estimators."""

    name = "TorchContinuousEstimator"
    architecture_metadata: dict[str, Any] = {}

    def __init__(
        self,
        *,
        task: str = "regression",
        random_state: int = 42,
        grid_size: int = 21,
        epochs: int = 120,
        batch_size: int = 256,
        hidden_dim: int = 32,
        learning_rate: float = 0.01,
        weight_decay: float = 1e-4,
        baseline_dose: float = 0.0,
    ) -> None:
        if task not in {"regression", "classification"}:
            raise ValueError("task must be 'regression' or 'classification'.")
        self.task = task
        self.random_state = int(random_state)
        self.grid_size = max(3, int(grid_size))
        self.epochs = max(1, int(epochs))
        self.batch_size = max(1, int(batch_size))
        self.hidden_dim = max(4, int(hidden_dim))
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.baseline_dose = float(baseline_dose)
        self._fitted = False

    def _new_network(self, input_dim: int, dose_boundaries: np.ndarray):
        raise NotImplementedError

    def _fit_frame(
        self,
        frame: pd.DataFrame,
        feature_cols: list[str],
        treatment_col: str,
        outcome_col: str,
        sample_weight: Any | None = None,
    ):
        torch = _require_torch()
        if not self._fitted:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)
            self.feature_cols = list(feature_cols)
            self.preprocessor = TabularPreprocessor(self.feature_cols)
            encoded = self.preprocessor.fit_transform(frame[self.feature_cols])
            self.observed_dose_min = float(frame[treatment_col].min())
            self.observed_dose_max = float(frame[treatment_col].max())
            self.dose_min = min(self.observed_dose_min, self.baseline_dose)
            self.dose_max = max(self.observed_dose_max, self.baseline_dose)
            if not np.isfinite([self.dose_min, self.dose_max]).all() or self.dose_max <= self.dose_min:
                raise ValueError("Continuous treatment must contain at least two finite, distinct doses.")
            self.dose_grid = np.unique(np.append(np.linspace(self.dose_min, self.dose_max, self.grid_size), self.baseline_dose))
            self.dose_scale = self.dose_max - self.dose_min
            normalized_dose = (frame[treatment_col].to_numpy(dtype="float64") - self.dose_min) / self.dose_scale
            bins = max(1, int(getattr(self, "num_dose_bins", 1)))
            quantiles = np.arange(1, bins, dtype="float64") / bins
            self.dose_boundaries = np.unique(np.quantile(normalized_dose, quantiles)).astype("float32") if len(quantiles) else np.empty(0, dtype="float32")
            self.model = self._new_network(encoded.shape[1], self.dose_boundaries)
        else:
            if list(feature_cols) != self.feature_cols:
                raise ValueError("A warm-start fit must use the same feature columns as the factual fit.")
            encoded = self.preprocessor.transform(frame[self.feature_cols])

        # Pandas copy-on-write may expose a read-only NumPy view; Torch tensors
        # must own writable memory even though the training loop does not mutate it.
        x_np = encoded.to_numpy(dtype="float32", copy=True)
        dose = frame[treatment_col].to_numpy(dtype="float32")
        y_np = frame[outcome_col].to_numpy(dtype="float32")
        if sample_weight is None:
            weights_np = np.ones(len(frame), dtype="float32")
        else:
            weights_np = np.asarray(sample_weight, dtype="float32").reshape(-1)
            if len(weights_np) != len(frame) or not np.isfinite(weights_np).all() or np.any(weights_np < 0):
                raise ValueError("sample_weight must be finite, nonnegative, and match the number of rows.")
        if not np.isfinite(x_np).all() or not np.isfinite(dose).all() or not np.isfinite(y_np).all():
            raise ValueError("Features, treatment, and outcome must be finite after preprocessing.")

        dose_np = (dose - self.dose_min) / self.dose_scale
        x = torch.as_tensor(x_np, dtype=torch.float32)
        t = torch.as_tensor(dose_np, dtype=torch.float32)
        y = torch.as_tensor(y_np, dtype=torch.float32)
        weights = torch.as_tensor(weights_np, dtype=torch.float32)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        rng = np.random.default_rng(self.random_state + int(getattr(self, "_fit_count", 0)))
        for _ in range(self.epochs):
            self.model.train()
            order = rng.permutation(len(x_np))
            for start in range(0, len(order), self.batch_size):
                ids = torch.as_tensor(order[start : start + self.batch_size], dtype=torch.long)
                optimizer.zero_grad(set_to_none=True)
                prediction = self.model(x[ids], t[ids]).reshape(-1)
                per_row = (prediction - y[ids]).square()
                batch_weight = weights[ids]
                denom = batch_weight.sum().clamp_min(1e-8)
                loss = (per_row * batch_weight).sum() / denom
                loss.backward()
                optimizer.step()
        self._fit_count = int(getattr(self, "_fit_count", 0)) + 1
        self._fitted = True
        return x_np

    def fit(self, dataset: CausalDataset, *, sample_weight: Any | None = None, warm_start: bool = False):
        if dataset.treatment_type != TreatmentType.CONTINUOUS:
            raise ValueError(f"{self.name} requires continuous treatment.")
        frame = dataset.to_pandas().dropna(subset=[dataset.treatment_col, dataset.outcome_col]).reset_index(drop=True)
        frame[dataset.treatment_col] = pd.to_numeric(frame[dataset.treatment_col], errors="coerce")
        frame[dataset.outcome_col] = pd.to_numeric(frame[dataset.outcome_col], errors="coerce")
        frame = frame.dropna(subset=[dataset.treatment_col, dataset.outcome_col]).reset_index(drop=True)
        if len(frame) < 4:
            raise ValueError("At least four complete rows are required for continuous neural estimation.")
        if self.task == "classification" and frame[dataset.outcome_col].nunique() > 2:
            self.task = "regression"
        # A warm-start is used by GIKS to preserve factual initialization while
        # optimizing the combined factual and accepted pseudo-counterfactual data.
        if not warm_start or not self._fitted:
            self._fitted = False
        self._fit_frame(frame, dataset.feature_cols, dataset.treatment_col, dataset.outcome_col, sample_weight)
        self.treatment_col = dataset.treatment_col
        self.outcome_col = dataset.outcome_col
        return self

    def _encoded(self, features: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model must be fitted before prediction.")
        return self.preprocessor.transform(features[self.feature_cols]).to_numpy(dtype="float32", copy=True)

    def _predict_raw(self, x_encoded: np.ndarray, doses: np.ndarray, *, gradient: bool = False):
        torch = _require_torch()
        x_encoded = np.asarray(x_encoded, dtype="float32")
        doses = np.asarray(doses, dtype="float32").reshape(-1)
        if len(x_encoded) != len(doses):
            raise ValueError("One dose is required per feature row.")
        t_np = (doses - self.dose_min) / self.dose_scale
        self.model.eval()
        x = torch.as_tensor(x_encoded, dtype=torch.float32)
        t = torch.as_tensor(t_np, dtype=torch.float32).requires_grad_(gradient)
        if gradient:
            y = self.model(x, t).reshape(-1)
            grad = torch.autograd.grad(y.sum(), t)[0] / self.dose_scale
            return y.detach().cpu().numpy().astype("float64"), grad.detach().cpu().numpy().astype("float64")
        with torch.no_grad():
            y = self.model(x, t).reshape(-1)
        return y.cpu().numpy().astype("float64")

    def _predict_at_raw_features(self, features: pd.DataFrame, doses: np.ndarray) -> np.ndarray:
        x = self._encoded(features)
        return self._predict_raw(x, doses)

    def _predict_dose_gradient(self, features: pd.DataFrame, doses: np.ndarray) -> np.ndarray:
        x = self._encoded(features)
        _, grad = self._predict_raw(x, doses, gradient=True)
        return grad

    def predict(self, dataset: CausalDataset) -> EffectPrediction:
        if dataset.treatment_type != TreatmentType.CONTINUOUS:
            raise ValueError(f"{self.name} requires continuous treatment.")
        frame = dataset.to_pandas()
        x = self._encoded(frame[self.feature_cols])
        n, k = len(x), len(self.dose_grid)
        x_grid = np.tile(x, (k, 1))
        dose_grid_rows = np.repeat(self.dose_grid, n)
        outcome = self._predict_raw(x_grid, dose_grid_rows).reshape(k, n).T
        return effect_prediction_from_curve(
            unit_id=dataset.unit_ids.to_numpy(),
            dose_grid=self.dose_grid.copy(),
            dose_outcomes=outcome,
            baseline_dose=self.baseline_dose,
            metadata={
                "model_name": self.name,
                "maturity": "OPTIONAL/EXPERIMENTAL",
                "status": "EXPERIMENTAL",
                "recommendation_type": "maximum_effect_reference_only",
                "device": "cpu",
                "seed": self.random_state,
                **self.architecture_metadata,
            },
        )
