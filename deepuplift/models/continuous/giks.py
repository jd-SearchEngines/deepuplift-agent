from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from deepuplift.contracts import CausalDataset, EffectPrediction, TreatmentType
from .drnet import DRNet
from .vcnet import VCNet


class GIKSEstimator:
    """Two-stage GIKS augmentation around a differentiable dose estimator.

    Stage one fits only factual observations. Stage two samples independent
    counterfactual doses, labels nearby doses with first-order gradient
    interpolation anchored at the factual outcome, and labels distant doses
    with a Gaussian-process RBF smoother over covariates and treatment. GP
    posterior variance filters and down-weights weakly supported labels.

    This is an independent implementation of the published method's central
    mechanism. It does not copy the official Apache-2.0 repository code.
    """

    name = "GIKS"
    _BASES = {"DRNET": DRNet, "VCNET": VCNet}

    def __init__(
        self,
        base_model: str = "VCNet",
        *,
        random_state: int = 42,
        grid_size: int = 21,
        factual_epochs: int = 80,
        giks_epochs: int = 40,
        augmentation_ratio: float = 1.0,
        near_threshold: float = 0.15,
        kernel_bandwidth: float = 0.20,
        max_gp_variance: float = 0.99,
        gp_noise: float = 0.05,
        max_gp_context: int = 512,
        model_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.base_model_name = str(base_model).upper()
        if self.base_model_name not in self._BASES:
            raise ValueError("GIKS currently validates differentiable DRNet and VCNet bases; choose base_model='DRNet' or 'VCNet'.")
        if augmentation_ratio <= 0:
            raise ValueError("augmentation_ratio must be positive.")
        if not 0 < near_threshold < 1 or not 0 < kernel_bandwidth <= 1:
            raise ValueError("near_threshold and kernel_bandwidth are fractions in (0, 1].")
        if not 0 < max_gp_variance <= 1:
            raise ValueError("max_gp_variance must be in (0, 1].")
        self.random_state = int(random_state)
        self.grid_size = max(3, int(grid_size))
        self.factual_epochs = max(1, int(factual_epochs))
        self.giks_epochs = max(1, int(giks_epochs))
        self.augmentation_ratio = float(augmentation_ratio)
        self.near_threshold = float(near_threshold)
        self.kernel_bandwidth = float(kernel_bandwidth)
        self.max_gp_variance = float(max_gp_variance)
        self.gp_noise = float(gp_noise)
        self.max_gp_context = max(8, int(max_gp_context))
        self.model_kwargs = dict(model_kwargs or {})

    def _make_base(self, *, epochs: int):
        cls = self._BASES[self.base_model_name]
        kwargs = {"task": "regression", "random_state": self.random_state, "grid_size": self.grid_size, "epochs": epochs}
        kwargs.update(self.model_kwargs)
        return cls(**kwargs)

    def _gp_counterfactuals(
        self,
        context_x: np.ndarray,
        context_t: np.ndarray,
        context_y: np.ndarray,
        query_x: np.ndarray,
        query_t: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Exact RBF Gaussian-process posterior mean and normalized variance."""
        from numpy.linalg import LinAlgError, solve

        rng = np.random.default_rng(self.random_state)
        if len(context_x) > self.max_gp_context:
            selected = np.sort(rng.choice(len(context_x), size=self.max_gp_context, replace=False))
            context_x, context_t, context_y = context_x[selected], context_t[selected], context_y[selected]
        context_norm = np.sum(context_x * context_x, axis=1)
        context_distance2 = np.maximum(context_norm[:, None] + context_norm[None, :] - 2.0 * (context_x @ context_x.T), 0.0)
        x_scale = np.sqrt(np.median(context_distance2)) if len(context_x) > 1 else 1.0
        x_scale = max(float(x_scale), 1e-3)
        y_center = float(np.mean(context_y))
        y_scale = float(np.std(context_y)) or 1.0
        y_std = (context_y - y_center) / y_scale

        train_x = context_x / x_scale
        query_x_scaled = query_x / x_scale
        dt = (context_t[:, None] - context_t[None, :]) / self.kernel_bandwidth
        train_norm = np.sum(train_x * train_x, axis=1)
        distance2 = np.maximum(train_norm[:, None] + train_norm[None, :] - 2.0 * (train_x @ train_x.T), 0.0)
        kernel = np.exp(-0.5 * (distance2 + dt * dt))
        kernel.flat[:: len(kernel) + 1] += self.gp_noise
        try:
            alpha = solve(kernel, y_std)
            cross_dt = (query_t[:, None] - context_t[None, :]) / self.kernel_bandwidth
            query_norm = np.sum(query_x_scaled * query_x_scaled, axis=1)
            cross_distance2 = np.maximum(query_norm[:, None] + train_norm[None, :] - 2.0 * (query_x_scaled @ train_x.T), 0.0)
            cross = np.exp(-0.5 * (cross_distance2 + cross_dt * cross_dt))
            posterior = cross @ alpha
            variance = 1.0 - np.sum(cross * solve(kernel, cross.T).T, axis=1)
        except LinAlgError:
            # A larger diagonal jitter is a numerical repair of the same GP,
            # not a fallback to nearest-neighbour interpolation.
            kernel.flat[:: len(kernel) + 1] += 1e-3
            alpha = solve(kernel, y_std)
            cross_dt = (query_t[:, None] - context_t[None, :]) / self.kernel_bandwidth
            query_norm = np.sum(query_x_scaled * query_x_scaled, axis=1)
            cross_distance2 = np.maximum(query_norm[:, None] + train_norm[None, :] - 2.0 * (query_x_scaled @ train_x.T), 0.0)
            cross = np.exp(-0.5 * (cross_distance2 + cross_dt * cross_dt))
            posterior = cross @ alpha
            variance = 1.0 - np.sum(cross * solve(kernel, cross.T).T, axis=1)
        return y_center + y_scale * posterior, np.clip(variance, 0.0, 1.0)

    def fit(self, dataset: CausalDataset) -> "GIKSEstimator":
        if dataset.treatment_type != TreatmentType.CONTINUOUS:
            raise ValueError("GIKS requires continuous treatment.")
        started = time.perf_counter()
        frame = dataset.to_pandas().dropna(subset=[dataset.treatment_col, dataset.outcome_col]).reset_index(drop=True)
        frame[dataset.treatment_col] = pd.to_numeric(frame[dataset.treatment_col], errors="coerce")
        frame[dataset.outcome_col] = pd.to_numeric(frame[dataset.outcome_col], errors="coerce")
        frame = frame.dropna(subset=[dataset.treatment_col, dataset.outcome_col]).reset_index(drop=True)
        if len(frame) < 8:
            raise ValueError("GIKS needs at least eight complete factual observations.")
        factual_t = frame[dataset.treatment_col].to_numpy(dtype="float64")
        factual_y = frame[dataset.outcome_col].to_numpy(dtype="float64")
        lower, upper = float(factual_t.min()), float(factual_t.max())
        if upper - lower <= 1e-8 * max(1.0, abs(lower), abs(upper)):
            raise ValueError("GIKS requires a non-degenerate observed treatment support; near-constant doses are unsupported.")
        self.dose_lower, self.dose_upper = lower, upper
        self.feature_cols = list(dataset.feature_cols)
        factual = CausalDataset(
            data=frame,
            feature_cols=dataset.feature_cols,
            treatment_col=dataset.treatment_col,
            outcome_col=dataset.outcome_col,
            treatment_type=dataset.treatment_type,
            assignment_type=dataset.assignment_type,
            id_column=dataset.id_column,
            metadata={**dataset.metadata, "giks_stage": "factual"},
        )

        # Stage one is a genuine factual-only estimator fit.
        self.base_estimator = self._make_base(epochs=self.factual_epochs)
        factual_started = time.perf_counter()
        self.base_estimator.fit(factual)
        factual_fit_seconds = time.perf_counter() - factual_started

        count = max(1, int(round(len(frame) * self.augmentation_ratio)))
        rng = np.random.default_rng(self.random_state)
        source_ids = rng.integers(0, len(frame), size=count)
        cf_dose = rng.uniform(lower, upper, size=count)
        source_dose = factual_t[source_ids]
        normalized_gap = np.abs(cf_dose - source_dose) / (upper - lower)
        near = normalized_gap <= self.near_threshold
        features = frame[self.feature_cols]
        pseudo_x = features.iloc[source_ids].reset_index(drop=True)

        # Gradient interpolation: factual outcome anchor plus the learned local
        # dose derivative times the counterfactual dose displacement.
        gradients = self.base_estimator._predict_dose_gradient(pseudo_x, source_dose)
        pseudo_y = np.full(count, np.nan, dtype="float64")
        pseudo_y[near] = factual_y[source_ids[near]] + gradients[near] * (cf_dose[near] - source_dose[near])

        # Kernel smoothing uses all factual outcomes in an exact Gaussian
        # process over standardized covariates and normalized dose.
        far = ~near
        context_x = self.base_estimator.preprocessor.transform(frame[self.feature_cols]).to_numpy(dtype="float64")
        pseudo_encoded = self.base_estimator.preprocessor.transform(pseudo_x).to_numpy(dtype="float64")
        context_t = (factual_t - lower) / (upper - lower)
        far_mean = np.empty(0, dtype="float64")
        far_variance = np.empty(0, dtype="float64")
        if far.any():
            far_mean, far_variance = self._gp_counterfactuals(
                context_x, context_t, factual_y,
                pseudo_encoded[far], (cf_dose[far] - lower) / (upper - lower),
            )
            pseudo_y[far] = far_mean
        variance_full = np.zeros(count, dtype="float64")
        if far.any():
            variance_full[far] = far_variance
        accepted = np.isfinite(pseudo_y) & (variance_full <= self.max_gp_variance)
        if not accepted.any():
            raise RuntimeError("GIKS generated no accepted pseudo-counterfactual labels; inspect treatment overlap or relax the GP variance threshold.")
        pseudo_rows = frame.iloc[source_ids[accepted]].copy().reset_index(drop=True)
        pseudo_rows[dataset.treatment_col] = cf_dose[accepted]
        pseudo_rows[dataset.outcome_col] = pseudo_y[accepted]
        augmented_frame = pd.concat([frame, pseudo_rows], ignore_index=True)
        augmented = CausalDataset(
            data=augmented_frame,
            feature_cols=dataset.feature_cols,
            treatment_col=dataset.treatment_col,
            outcome_col=dataset.outcome_col,
            treatment_type=dataset.treatment_type,
            assignment_type=dataset.assignment_type,
            id_column=dataset.id_column,
            metadata={**dataset.metadata, "giks_stage": "factual_plus_counterfactual"},
        )
        far_confidence = np.clip(1.0 - variance_full[accepted], 0.0, 1.0)
        pseudo_weights = np.where(near[accepted], 1.0, np.maximum(far_confidence, 1e-3))
        sample_weight = np.concatenate([np.ones(len(frame), dtype="float64"), pseudo_weights])
        accepted_doses = cf_dose[accepted]
        edge_width = 0.05 * (upper - lower)
        edge_mask = (accepted_doses <= lower + edge_width) | (accepted_doses >= upper - edge_width)
        edge_fraction = float(edge_mask.mean()) if len(edge_mask) else 0.0
        if far.any():
            mean_gp_variance = float(np.mean(far_variance))
            p95_gp_variance = float(np.quantile(far_variance, 0.95))
        else:
            mean_gp_variance = p95_gp_variance = None

        # Stage two continues optimization from factual weights on the mixed
        # factual and pseudo-counterfactual sample.
        self.base_estimator.epochs = self.giks_epochs
        self.base_estimator.fit(augmented, sample_weight=sample_weight, warm_start=True)
        self.dataset_metadata = dict(dataset.metadata)
        self.fit_metadata = {
            "base_model": self.base_model_name,
            "factual_epochs": self.factual_epochs,
            "giks_epochs": self.giks_epochs,
            "augmentation_ratio": self.augmentation_ratio,
            "near_threshold": self.near_threshold,
            "kernel_bandwidth": self.kernel_bandwidth,
            "pseudo_label_count": int(count),
            "accepted_pseudo_label_count": int(accepted.sum()),
            "observed_support": {"observed_min": lower, "observed_max": upper},
            "pseudo_dose_support": {
                "min": float(accepted_doses.min()), "max": float(accepted_doses.max()),
                "contains_extrapolation": bool(np.any((accepted_doses < lower) | (accepted_doses > upper))),
            },
            "pseudo_extrapolation_fraction": float(np.mean((accepted_doses < lower) | (accepted_doses > upper))),
            "near_pseudo_count": int((accepted & near).sum()),
            "far_pseudo_count": int((accepted & far).sum()),
            "accepted_ratio": float(accepted.mean()),
            "mean_gp_variance": mean_gp_variance,
            "p95_gp_variance": p95_gp_variance,
            "effective_pseudo_weight_sum": float(pseudo_weights.sum()),
            "accepted_pseudo_support_edge_fraction": edge_fraction,
            "support_warning": (
                "Many accepted pseudo-labels are near factual support edges; local support is weaker there."
                if edge_fraction >= 0.25 else None
            ),
            "gradient_interpolation_attempted": int(near.sum()),
            "gradient_interpolation_accepted": int((accepted & near).sum()),
            "kernel_smoothing_attempted": int(far.sum()),
            "kernel_smoothing_accepted": int((accepted & far).sum()),
            "gp_context_size": min(len(frame), self.max_gp_context),
            "seed": self.random_state,
            "implementation_reference": "https://github.com/nlokeshiisc/GIKS_release",
            "implementation_license": "Apache-2.0 (reference repository); no code copied.",
            "differences_from_reference": "Independent exact RBF GP over covariates and treatment; posterior-variance weighting; default differentiable bases DRNet/VCNet; compact full-data two-stage training.",
            "factual_fit_seconds": factual_fit_seconds,
            "total_fit_seconds": time.perf_counter() - started,
        }
        return self

    def predict(self, dataset: CausalDataset) -> EffectPrediction:
        prediction = self.base_estimator.predict(dataset)
        prediction.metadata.update({
            "model_name": self.name,
            "maturity": "OPTIONAL/EXPERIMENTAL",
            "giks": {**self.fit_metadata, "augmentation_executed": True},
        })
        return prediction


GIKS = GIKSEstimator

__all__ = ["GIKS", "GIKSEstimator"]
