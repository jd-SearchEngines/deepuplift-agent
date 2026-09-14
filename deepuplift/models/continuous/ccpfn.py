from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from deepuplift.contracts import CausalDataset, EffectPrediction, TreatmentType
from deepuplift.data.preprocessing import TabularPreprocessor
from .contracts import effect_prediction_from_curve


class CCPFNAdapter:
    """Adapter for the published CCPFN CEPO inference package.

    The optional backend is imported only on ``fit``. CCPFN uses factual rows
    as in-context observations and exposes conditional expected potential
    outcomes through ``CEPOEstimator.estimate_cepo``; DeepUplift converts those
    values into its shared dose-response contract.
    """

    name = "CCPFN"
    implementation_source = "external_adapter"
    paper = "Causal Foundation Models with Continuous Treatments (2026)"
    source_project = "https://github.com/layer6ai-labs/CCPFN-inference"
    license = "Apache-2.0"

    def __init__(
        self,
        *,
        device: str = "cpu",
        model_path: str = "Layer6/CCPFN",
        grid_size: int = 21,
        random_state: int = 42,
        max_context_length: int = 4096,
        max_query_length: int = 4096,
        cache_dir: str | None = None,
        backend: Any | None = None,
    ) -> None:
        self.device = device
        self.model_path = model_path
        self.grid_size = max(3, int(grid_size))
        self.random_state = int(random_state)
        self.max_context_length = max(1, int(max_context_length))
        self.max_query_length = max(1, int(max_query_length))
        self.cache_dir = cache_dir
        self.backend = backend
        self._backend_injected = backend is not None
        self._fitted = False

    def fit(self, dataset: CausalDataset) -> "CCPFNAdapter":
        if dataset.treatment_type != TreatmentType.CONTINUOUS:
            raise ValueError("CCPFNAdapter requires continuous treatment.")
        frame = dataset.to_pandas().dropna(subset=[dataset.treatment_col, dataset.outcome_col]).reset_index(drop=True)
        frame[dataset.treatment_col] = pd.to_numeric(frame[dataset.treatment_col], errors="coerce")
        frame[dataset.outcome_col] = pd.to_numeric(frame[dataset.outcome_col], errors="coerce")
        frame = frame.dropna(subset=[dataset.treatment_col, dataset.outcome_col]).reset_index(drop=True)
        self.feature_cols = list(dataset.feature_cols)
        self.preprocessor = TabularPreprocessor(self.feature_cols)
        x = self.preprocessor.fit_transform(frame[self.feature_cols]).to_numpy(dtype="float64", copy=True)
        treatment = frame[dataset.treatment_col].to_numpy(dtype="float64")
        outcome = frame[dataset.outcome_col].to_numpy(dtype="float64")
        if not np.isfinite(x).all() or not np.isfinite(treatment).all() or not np.isfinite(outcome).all():
            raise ValueError("CCPFN context features, treatment, and outcome must be finite.")
        self.baseline_dose = 0.0
        self.dose_grid = np.unique(np.append(np.linspace(min(0.0, treatment.min()), max(0.0, treatment.max()), self.grid_size), self.baseline_dose))
        self.training_dose_range = [float(treatment.min()), float(treatment.max())]
        self.context_size = min(len(x), self.max_context_length)
        if len(x) > self.context_size:
            rng = np.random.default_rng(self.random_state)
            self.context_indices = np.sort(rng.choice(len(x), size=self.context_size, replace=False))
        else:
            self.context_indices = np.arange(len(x))
        self.backend_package_version = None
        if self.backend is None:
            try:
                module = importlib.import_module("ccpfn")
            except ImportError as exc:
                raise ImportError("CCPFN requires the optional deepuplift[ccpfn] extra.") from exc
            version = importlib.import_module("importlib.metadata").version
            self.backend_package_version = version("ccpfn")
            estimator = module.CEPOEstimator(
                device=self.device,
                model_path=self.model_path,
                max_context_length=self.max_context_length,
                max_query_length=self.max_query_length,
                cache_dir=self.cache_dir,
            )
            self.backend = estimator
        # CEPOEstimator.fit stores this factual context and loads the published
        # pretrained prior; it does not fine-tune the foundation model.
        self.backend.fit(x[self.context_indices], treatment[self.context_indices], outcome[self.context_indices])
        self._fitted = True
        return self

    def predict(self, dataset: CausalDataset) -> EffectPrediction:
        if not self._fitted:
            raise RuntimeError("Model must be fitted before predict.")
        if dataset.treatment_type != TreatmentType.CONTINUOUS:
            raise ValueError("CCPFNAdapter requires continuous treatment.")
        frame = dataset.to_pandas()
        x = self.preprocessor.transform(frame[self.feature_cols]).to_numpy(dtype="float64", copy=True)
        n, k = len(x), len(self.dose_grid)
        outcomes = np.empty((n, k), dtype="float64")
        for column, dose in enumerate(self.dose_grid):
            column_values = np.empty(n, dtype="float64")
            for start in range(0, n, self.max_query_length):
                end = min(start + self.max_query_length, n)
                target = np.full(end - start, float(dose), dtype="float64")
                column_values[start:end] = np.asarray(self.backend.estimate_cepo(x[start:end], target), dtype="float64").reshape(-1)
            outcomes[:, column] = column_values
        if not np.isfinite(outcomes).all():
            raise ValueError("CCPFN returned non-finite CEPO predictions.")
        metadata = {
            "model_name": self.name,
            "backend": "ccpfn.CEPOEstimator",
            "backend_package_version": self.backend_package_version,
            "implementation_source": self.implementation_source,
            "paper": self.paper,
            "source_project": self.source_project,
            "license": self.license,
            "weights_model_id": self.model_path,
            "context_size": self.context_size,
            "seed": self.random_state,
            "training_dose_range": self.training_dose_range,
            "requires_network_on_first_use": not self._backend_injected,
            "maturity": "OPTIONAL/FRONTIER_EXPERIMENTAL",
            "status": "EXPERIMENTAL",
            "recommendation_type": "maximum_effect_reference_only",
        }
        return effect_prediction_from_curve(
            unit_id=dataset.unit_ids.to_numpy(),
            dose_grid=self.dose_grid.copy(),
            dose_outcomes=outcomes,
            baseline_dose=self.baseline_dose,
            metadata=metadata,
        )


def local_weights_available(model_path: str = "Layer6/CCPFN", cache_dir: str | None = None) -> bool:
    """Return whether the CCPFN Hub snapshot appears in its configured cache."""
    root = Path(cache_dir or (Path.home() / ".cache" / "ccpfn"))
    owner, _, name = model_path.partition("/")
    blobs = root / f"models--{owner}--{name}" / "blobs"
    return blobs.is_dir() and any(path.is_file() and path.stat().st_size > 0 for path in blobs.iterdir())


__all__ = ["CCPFNAdapter", "local_weights_available"]
