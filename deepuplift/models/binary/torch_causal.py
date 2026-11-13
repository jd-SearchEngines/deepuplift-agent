"""Unified adapters for the repository's binary neural causal models.

The legacy modules in ``deepuplift.models`` predate the stable model contract.
This adapter reuses their network and loss definitions while adding a
backend-neutral ``fit``/``predict`` interface.  It is deliberately labelled
``ARCHITECTURE_INSPIRED_VARIANT``: the official external repositories and
their historical training stacks are not copied or claimed as reproduced.
"""
from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pandas as pd

from deepuplift.contracts import CausalDataset, EffectPrediction, TreatmentType
from deepuplift.data.preprocessing import TabularPreprocessor
from deepuplift.models.CFRNet import CFRNet, cfrnet_loss
from deepuplift.models.DragonNet import DragonNet, dragonnet_loss
from deepuplift.models.TarNet import TarNet, tarnet_loss


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise ImportError("TorchBinaryCausalEstimator requires PyTorch.") from exc
    return torch


class TorchBinaryCausalEstimator:
    """TARNet/CFRNet/DragonNet adapter with explicit mechanism toggles."""

    _MODEL_CLASSES: ClassVar[dict[str, type]] = {"TarNet": TarNet, "CFRNet": CFRNet, "DragonNet": DragonNet}

    def __init__(
        self,
        *,
        model_name: str = "TarNet",
        task: str = "regression",
        random_state: int = 42,
        epochs: int = 20,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        share_dim: int = 8,
        hidden_dim: int = 32,
        balance_strength: float = 1.0,
        ipm_mode: str = "mmd_rbf",
        targeted_regularization: bool = False,
        device: str = "auto",
    ) -> None:
        if model_name not in self._MODEL_CLASSES:
            raise ValueError(f"model_name must be one of {sorted(self._MODEL_CLASSES)}")
        if task not in {"regression", "classification"}:
            raise ValueError("task must be regression or classification")
        self.model_name = model_name
        self.task = task
        self.random_state = int(random_state)
        self.epochs = max(1, int(epochs))
        self.batch_size = max(2, int(batch_size))
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.share_dim = max(2, int(share_dim))
        self.hidden_dim = max(4, int(hidden_dim))
        self.balance_strength = float(balance_strength)
        self.ipm_mode = str(ipm_mode)
        self.targeted_regularization = bool(targeted_regularization)
        self.requested_device = str(device).lower()
        self.device = self._resolve_device(self.requested_device)
        self.preprocessor: TabularPreprocessor | None = None
        self.model = None
        self.feature_cols: list[str] = []
        self.treatment_mapping: dict[Any, int] = {}

    @staticmethod
    def _resolve_device(requested: str) -> str:
        torch = _torch()
        if requested == "auto":
            return "mps" if torch.backends.mps.is_available() else "cpu"
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("Requested MPS device is unavailable")
        if requested not in {"cpu", "mps"}:
            raise ValueError("device must be auto, cpu, or mps")
        return requested

    def _make_model(self, input_dim: int):
        cls = self._MODEL_CLASSES[self.model_name]
        model = cls(
            input_dim=input_dim,
            share_dim=self.share_dim,
            share_hidden_dims=[self.hidden_dim, self.hidden_dim],
            base_hidden_dims=[self.hidden_dim, self.hidden_dim],
            task=self.task,
        ).to(self.device)
        # The retained legacy TowerUnit stores its own device string and
        # otherwise moves inputs back to CPU even after ``model.to("mps")``.
        # Keep the public adapter honest about the actual backend.
        for module in model.modules():
            if hasattr(module, "device"):
                module.device = self.device
        return model

    def _loss(self, outputs, t, y):
        t_pred, y_preds, *extras = outputs
        phi = extras[0] if extras else None
        if self.model_name == "CFRNet":
            return cfrnet_loss(
                t_pred,
                y_preds,
                t,
                y,
                phi,
                alpha=self.balance_strength,
                task=self.task,
                ipm_mode=self.ipm_mode,
            )[0]
        if self.model_name == "DragonNet":
            epsilon = extras[1] if len(extras) > 1 else None
            return dragonnet_loss(
                t_pred,
                y_preds,
                t,
                y,
                phi,
                alpha=1.0,
                beta=1.0,
                tarreg=self.targeted_regularization,
                tarreg_eps=epsilon,
                task=self.task,
            )[0]
        return tarnet_loss(t_pred, y_preds, t, y, phi, task=self.task)[0]

    def fit(self, dataset: CausalDataset, nuisance=None) -> TorchBinaryCausalEstimator:
        if dataset.treatment_type != TreatmentType.BINARY:
            raise ValueError(f"{self.model_name} supports binary treatment only")
        torch = _torch()
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        frame = dataset.to_pandas().dropna(subset=[dataset.treatment_col, dataset.outcome_col]).reset_index(drop=True)
        values = sorted(frame[dataset.treatment_col].unique().tolist(), key=lambda value: str(value))
        if len(values) != 2:
            raise ValueError("Binary neural causal models require exactly two treatment arms")
        self.treatment_mapping = {values[0]: 0, values[1]: 1}
        treatment = frame[dataset.treatment_col].map(self.treatment_mapping).to_numpy(dtype="float32")
        outcome = pd.to_numeric(frame[dataset.outcome_col], errors="coerce").to_numpy(dtype="float32")
        if not np.isfinite(outcome).all():
            raise ValueError("Outcome must be finite")
        if self.task == "classification" and not set(np.unique(outcome)).issubset({0.0, 1.0}):
            raise ValueError("classification task requires a binary outcome")
        self.feature_cols = list(dataset.feature_cols)
        self.preprocessor = TabularPreprocessor(self.feature_cols)
        encoded = self.preprocessor.fit_transform(frame[self.feature_cols]).to_numpy(dtype="float32", copy=True)
        self.model = self._make_model(encoded.shape[1])
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        x_tensor = torch.as_tensor(encoded, dtype=torch.float32, device=self.device)
        t_tensor = torch.as_tensor(treatment[:, None], dtype=torch.float32, device=self.device)
        y_tensor = torch.as_tensor(outcome[:, None], dtype=torch.float32, device=self.device)
        rng = np.random.default_rng(self.random_state)
        history = []
        for epoch in range(self.epochs):
            self.model.train()
            order = rng.permutation(len(encoded))
            epoch_losses = []
            for start in range(0, len(order), self.batch_size):
                batch = torch.as_tensor(order[start : start + self.batch_size], dtype=torch.long, device=self.device)
                optimizer.zero_grad(set_to_none=True)
                loss = self._loss(self.model(x_tensor[batch], t_tensor[batch]), t_tensor[batch], y_tensor[batch])
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"{self.model_name} loss became non-finite at epoch {epoch}")
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu()))
            history.append(float(np.mean(epoch_losses)))
        self.training_history_ = history
        self.nuisance = nuisance
        return self

    def predict(self, dataset: CausalDataset) -> EffectPrediction:
        if self.model is None or self.preprocessor is None:
            raise RuntimeError("Model must be fitted before prediction")
        torch = _torch()
        frame = dataset.to_pandas()
        encoded = self.preprocessor.transform(frame[self.feature_cols]).to_numpy(dtype="float32", copy=True)
        x_tensor = torch.as_tensor(encoded, dtype=torch.float32, device=self.device)
        dummy_t = torch.zeros((len(frame), 1), dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            t_pred, y_preds, *_ = self.model(x_tensor, dummy_t)
        y0 = y_preds[0].reshape(-1).detach().cpu().numpy().astype("float64")
        y1 = y_preds[1].reshape(-1).detach().cpu().numpy().astype("float64")
        if t_pred is None:
            propensity = np.full(len(frame), .5, dtype="float64")
        else:
            propensity = t_pred.reshape(-1).detach().cpu().numpy().astype("float64")
        return EffectPrediction(
            unit_id=dataset.unit_ids.to_numpy(),
            treatment_type=TreatmentType.BINARY,
            y0=y0,
            y1=y1,
            uplift=y1 - y0,
            cate=y1 - y0,
            recommended_effect=y1 - y0,
            propensity=np.clip(propensity, 1e-6, 1 - 1e-6),
            metadata={
                "model_name": self.model_name,
                "maturity": "ARCHITECTURE_INSPIRED_VARIANT",
                "backend": "pytorch",
                "device": self.device,
                "task": self.task,
                "balance_strength": self.balance_strength,
                "ipm_mode": self.ipm_mode if self.model_name == "CFRNet" else None,
                "targeted_regularization": self.targeted_regularization if self.model_name == "DragonNet" else None,
                "training_epochs": self.epochs,
                "treatment_mapping": {str(key): value for key, value in self.treatment_mapping.items()},
                "source_boundary": "native adapter over retained DeepUplift legacy definitions; not official repository code reproduction",
            },
        )

    def model_info(self) -> dict[str, Any]:
        return {
            "name": self.model_name,
            "maturity": "ARCHITECTURE_INSPIRED_VARIANT",
            "backend": "pytorch",
            "device": self.device,
            "mechanism": {
                "TarNet": "shared representation plus treatment-specific outcome heads",
                "CFRNet": "TarNet plus differentiable representation IPM penalty",
                "DragonNet": "outcome heads plus propensity head and optional targeted regularization",
            }[self.model_name],
        }


__all__ = ["TorchBinaryCausalEstimator"]
