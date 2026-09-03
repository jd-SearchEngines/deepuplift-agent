from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from deepuplift.contracts import CausalDataset

from .contracts import NuisanceResult


def effective_sample_size(weights: np.ndarray, mask: np.ndarray | None = None) -> float | None:
    values = np.asarray(weights, dtype="float64") if mask is None else np.asarray(weights, dtype="float64")[mask]
    if len(values) == 0 or np.sum(values * values) == 0:
        return None
    return float(np.sum(values) ** 2 / np.sum(values * values))


def _smd(frame: pd.DataFrame, treatment: np.ndarray, features: list[str], weights: np.ndarray | None = None) -> dict[str, float]:
    result: dict[str, float] = {}
    for col in features:
        if not pd.api.types.is_numeric_dtype(frame[col]):
            continue
        values = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype="float64")
        valid = np.isfinite(values)
        if weights is None:
            w = np.ones(len(frame))
        else:
            w = np.asarray(weights, dtype="float64")
        left, right = valid & (treatment == 0), valid & (treatment == 1)
        if not left.any() or not right.any():
            continue
        mean0 = np.average(values[left], weights=w[left]); mean1 = np.average(values[right], weights=w[right])
        var0 = np.average((values[left] - mean0) ** 2, weights=w[left]); var1 = np.average((values[right] - mean1) ** 2, weights=w[right])
        pooled = np.sqrt((var0 + var1) / 2)
        result[col] = float((mean1 - mean0) / pooled) if pooled > 0 else 0.0
    return result


def apply_weighting(dataset: CausalDataset, nuisance: NuisanceResult, *, strategy: str = "ipw", trim_threshold: float | None = None) -> NuisanceResult:
    """Apply explicit IPW/stabilized-IPW/overlap weights and record all changes."""
    frame = dataset.to_pandas().reset_index(drop=True)
    values = sorted(frame[dataset.treatment_col].dropna().unique().tolist(), key=lambda item: str(item))
    treatment = frame[dataset.treatment_col].map({values[0]: 0, values[1]: 1}).to_numpy(dtype="float64")
    p = np.asarray(nuisance.propensity_scores, dtype="float64")
    if len(p) != len(frame):
        raise ValueError("Nuisance arrays must have the same row count as the dataset.")
    trim = np.ones(len(frame), dtype=bool)
    threshold = trim_threshold
    if threshold is not None:
        if not 0 < threshold < .5:
            raise ValueError("trim_threshold must be between 0 and 0.5.")
        trim = (p >= threshold) & (p <= 1 - threshold)
    safe = np.clip(p, 1e-6, 1 - 1e-6)
    if strategy == "ipw":
        weights = np.where(treatment == 1, 1 / safe, 1 / (1 - safe))
    elif strategy in {"stabilized_ipw", "stabilized-ipw"}:
        marginal = float(np.mean(treatment)); weights = np.where(treatment == 1, marginal / safe, (1 - marginal) / (1 - safe))
    elif strategy == "overlap":
        weights = np.where(treatment == 1, 1 - safe, safe)
    elif strategy in {"none", "unweighted"}:
        weights = np.ones(len(frame), dtype="float64")
    else:
        raise ValueError("strategy must be ipw, stabilized_ipw, overlap, or none.")
    weights = np.where(trim, weights, 0.0)
    before = _smd(frame, treatment, dataset.feature_cols)
    after = _smd(frame, treatment, dataset.feature_cols, weights)
    ess = {"treated": effective_sample_size(weights, treatment == 1), "control": effective_sample_size(weights, treatment == 0), "overall": effective_sample_size(weights, trim)}
    diagnostics = dict(nuisance.diagnostics)
    max_before = max((abs(v) for v in before.values()), default=0.0)
    max_after = max((abs(v) for v in after.values()), default=0.0)
    diagnostics.update({"weighting": {"strategy": strategy, "trim_threshold": threshold, "rows_removed": int((~trim).sum()), "percentage_removed": float((~trim).mean()), "ess": ess, "ess_unweighted": float(len(frame)), "ess_change": (ess["overall"] - len(frame)) if ess["overall"] is not None else None, "balance_before_smd": before, "balance_after_smd": after, "balance_change_max_abs_smd": max_after - max_before, "max_abs_smd_before": max_before, "max_abs_smd_after": max_after}})
    nuisance.sample_weights = weights
    nuisance.trim_mask = trim
    nuisance.effective_sample_size = ess
    nuisance.diagnostics = diagnostics
    nuisance.metadata = {**nuisance.metadata, "weighting_strategy": strategy, "trim_threshold": threshold}
    return nuisance


__all__ = ["apply_weighting", "effective_sample_size"]
