"""Lightweight empirical treatment support diagnostics for continuous doses.

These summaries describe observed coverage; they are diagnostics, not a proof
of positivity or causal identification.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from deepuplift.contracts import CausalDataset, TreatmentType
from .preprocessing import TabularPreprocessor


def diagnose_continuous_treatment(
    dataset: CausalDataset,
    *,
    segment_cols: Sequence[str] | None = None,
    bins: int = 10,
    quantiles: Sequence[float] = (0.05, 0.25, 0.5, 0.75, 0.95),
) -> dict[str, Any]:
    """Summarize global dose distribution and dose coverage by feature segment."""
    if dataset.treatment_type != TreatmentType.CONTINUOUS:
        raise ValueError("Continuous treatment diagnostics require a continuous CausalDataset.")
    frame = dataset.to_pandas().dropna(subset=[dataset.treatment_col]).copy()
    dose = pd.to_numeric(frame[dataset.treatment_col], errors="coerce")
    valid = dose.notna() & np.isfinite(dose.to_numpy(dtype="float64"))
    frame, dose = frame.loc[valid], dose.loc[valid].astype("float64")
    if dose.empty:
        raise ValueError("No finite observed treatment doses are available.")
    lo, hi = float(dose.min()), float(dose.max())
    edges = np.linspace(lo, hi if hi > lo else lo + 1.0, max(1, int(bins)) + 1)
    counts, edges = np.histogram(dose.to_numpy(), bins=edges)
    qvalues = np.quantile(dose, quantiles)
    requested = list(segment_cols or [])
    unknown = sorted(set(requested) - set(frame.columns))
    if unknown:
        raise ValueError(f"Unknown segment columns: {unknown}")
    segments = {}
    for column in requested:
        groups = frame.groupby(column, dropna=False, sort=False)[dataset.treatment_col]
        rows = []
        for key, values in groups:
            numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype="float64")
            if not len(numeric):
                continue
            rows.append({
                "value": None if pd.isna(key) else str(key), "count": int(len(numeric)),
                "treatment_min": float(np.min(numeric)), "treatment_max": float(np.max(numeric)),
                "treatment_quantiles": {str(q): float(np.quantile(numeric, q)) for q in quantiles},
                "dose_coverage_fraction": float(np.ptp(numeric) / np.ptp(dose)) if np.ptp(dose) else 0.0,
            })
        segments[column] = rows
    return {
        "diagnostic_name": "observed_continuous_treatment_coverage",
        "interpretation": "Empirical coverage summary; not a positivity proof or causal identification guarantee.",
        "sample_count": int(len(dose)),
        "treatment_min": lo,
        "treatment_max": hi,
        "treatment_quantiles": {str(q): float(v) for q, v in zip(quantiles, qvalues)},
        "dose_histogram": {"counts": counts.astype(int).tolist(), "bin_edges": edges.astype(float).tolist()},
        "feature_segments": segments,
    }


def local_treatment_support(
    train: CausalDataset,
    prediction: CausalDataset,
    *,
    k: int = 25,
    dose_grid: Any | None = None,
) -> dict[str, Any]:
    """Estimate local empirical dose coverage among standardized feature kNNs."""
    for item in (train, prediction):
        if item.treatment_type != TreatmentType.CONTINUOUS:
            raise ValueError("Local treatment support requires continuous datasets.")
    if list(train.feature_cols) != list(prediction.feature_cols):
        raise ValueError("Training and prediction datasets must use the same feature columns.")
    train_frame = train.to_pandas().dropna(subset=[train.treatment_col]).reset_index(drop=True)
    treatment = pd.to_numeric(train_frame[train.treatment_col], errors="coerce").to_numpy(dtype="float64")
    valid = np.isfinite(treatment)
    train_frame, treatment = train_frame.loc[valid].reset_index(drop=True), treatment[valid]
    if len(treatment) < 2 or np.ptp(treatment) <= 0:
        raise ValueError("Local support requires at least two distinct observed doses.")
    prep = TabularPreprocessor(train.feature_cols)
    train_x = prep.fit_transform(train_frame[train.feature_cols]).to_numpy(dtype="float64")
    pred_x = prep.transform(prediction.to_pandas()[prediction.feature_cols]).to_numpy(dtype="float64")
    neighbor_count = min(max(2, int(k)), len(train_x))
    nn = NearestNeighbors(n_neighbors=neighbor_count).fit(train_x)
    distances, indices = nn.kneighbors(pred_x)
    local_doses = treatment[indices]
    low = np.quantile(local_doses, 0.05, axis=1)
    high = np.quantile(local_doses, 0.95, axis=1)
    global_span = float(np.ptp(treatment))
    score = np.clip((high - low) / global_span, 0.0, 1.0)
    grid = np.asarray([] if dose_grid is None else dose_grid, dtype="float64").reshape(-1)
    mask = ((grid[None, :] >= low[:, None]) & (grid[None, :] <= high[:, None])) if len(grid) else np.empty((len(pred_x), 0), dtype=bool)
    nearest_dose_distance = np.min(np.abs(local_doses[:, :, None] - grid[None, None, :]), axis=1) if len(grid) else np.empty((len(pred_x), 0))
    return {
        "diagnostic_name": "local_empirical_treatment_support_diagnostic",
        "interpretation": "Feature-standardized kNN dose coverage; not a positivity proof or causal identification guarantee.",
        "k": neighbor_count,
        "observed_dose_range": [float(np.min(treatment)), float(np.max(treatment))],
        "dose_grid": grid.tolist(),
        "local_support_min": low.tolist(),
        "local_support_max": high.tolist(),
        "support_score": score.tolist(),
        "nearest_observed_dose_distance": nearest_dose_distance.tolist(),
        "supported_dose_mask": mask.tolist(),
        "mean_support_score": float(np.mean(score)) if len(score) else 0.0,
        "unsupported_row_count": int(np.sum(~mask.any(axis=1))) if len(grid) else 0,
    }


__all__ = ["diagnose_continuous_treatment", "local_treatment_support"]
