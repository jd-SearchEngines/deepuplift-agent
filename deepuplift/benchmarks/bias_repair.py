"""Auditable bias diagnosis and repair for real observational data.

The module intentionally keeps the estimands explicit.  It is not a claim that
an observational estimate is a causal truth: identification assumptions and
support diagnostics are returned with every estimate.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, norm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neighbors import NearestNeighbors

from deepuplift.contracts import CausalDataset, TreatmentType
from deepuplift.data.continuous_diagnostics import diagnose_continuous_treatment
from deepuplift.data.nuisance.propensity import estimate_propensity
from deepuplift.data.nuisance.weighting import effective_sample_size
from deepuplift.data.preprocessing import TabularPreprocessor


def _finite(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_finite(v) for v in value.tolist()]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _smd(frame: pd.DataFrame, treatment: np.ndarray, features: list[str], weights: np.ndarray | None = None) -> dict[str, float]:
    """SMD over numeric features and one-hot levels, using pooled SD."""
    x = pd.get_dummies(frame[features], dummy_na=True, dtype=float)
    w = np.ones(len(frame), dtype=float) if weights is None else np.asarray(weights, dtype=float)
    result: dict[str, float] = {}
    for col in x.columns:
        values = x[col].to_numpy(dtype=float)
        left, right = treatment == 0, treatment == 1
        if not left.any() or not right.any() or w[left].sum() <= 0 or w[right].sum() <= 0:
            continue
        mean0 = np.average(values[left], weights=w[left])
        mean1 = np.average(values[right], weights=w[right])
        var0 = np.average((values[left] - mean0) ** 2, weights=w[left])
        var1 = np.average((values[right] - mean1) ** 2, weights=w[right])
        pooled = np.sqrt((var0 + var1) / 2.0)
        result[str(col)] = float((mean1 - mean0) / pooled) if pooled > 1e-12 else 0.0
    return result


def _weight_stats(weights: np.ndarray, treatment: np.ndarray, mask: np.ndarray | None = None) -> dict[str, Any]:
    w = np.asarray(weights, dtype=float)
    use = np.ones(len(w), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    active = w[use]
    return {
        "min": float(np.min(active)) if len(active) else None,
        "p50": float(np.quantile(active, .50)) if len(active) else None,
        "p95": float(np.quantile(active, .95)) if len(active) else None,
        "p99": float(np.quantile(active, .99)) if len(active) else None,
        "max": float(np.max(active)) if len(active) else None,
        "ess_overall": effective_sample_size(active),
        "ess_treated": effective_sample_size(w, use & (treatment == 1)),
        "ess_control": effective_sample_size(w, use & (treatment == 0)),
        "active_rows": int(use.sum()),
        "zero_weight_rows": int(np.sum(w <= 0)),
    }


def _weighted_diff(y: np.ndarray, t: np.ndarray, weights: np.ndarray) -> float:
    treated, control = t == 1, t == 0
    if weights[treated].sum() <= 0 or weights[control].sum() <= 0:
        return float("nan")
    return float(np.average(y[treated], weights=weights[treated]) - np.average(y[control], weights=weights[control]))


def _bootstrap(calc: Callable[[np.ndarray], float], n: int, seed: int, reps: int) -> tuple[float | None, float | None]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(max(20, int(reps))):
        value = calc(rng.integers(0, n, size=n))
        if np.isfinite(value):
            values.append(float(value))
    if not values:
        return None, None
    return float(np.quantile(values, .025)), float(np.quantile(values, .975))


def _matching_effect(y: np.ndarray, t: np.ndarray, p: np.ndarray, indices: np.ndarray | None = None) -> tuple[float, np.ndarray, np.ndarray]:
    ids = np.arange(len(y)) if indices is None else np.asarray(indices, dtype=int)
    yy, tt, pp = y[ids], t[ids], p[ids]
    treated, control = np.flatnonzero(tt == 1), np.flatnonzero(tt == 0)
    if not len(treated) or not len(control):
        return float("nan"), np.zeros(len(ids)), np.zeros(len(ids), dtype=bool)
    logit = np.log(np.clip(pp, 1e-6, 1 - 1e-6) / np.clip(1 - pp, 1e-6, 1 - 1e-6))
    caliper = max(0.01, .2 * float(np.std(logit)))
    nn0 = NearestNeighbors(n_neighbors=1).fit(logit[control, None])
    d0, j0 = nn0.kneighbors(logit[treated, None])
    keep = d0[:, 0] <= caliper
    if not keep.any():
        return float("nan"), np.zeros(len(ids)), np.zeros(len(ids), dtype=bool)
    tr = treated[keep]
    co = control[j0[keep, 0]]
    effect = float(np.mean(yy[tr] - yy[co]))
    weights = np.zeros(len(ids), dtype=float)
    weights[tr] = 1.0
    for item in co:
        weights[item] += 1.0
    active = weights > 0
    return effect, weights, active


def run_binary_bias_repair(
    dataset: CausalDataset,
    *,
    seed: int = 42,
    estimator: str = "logistic",
    bootstrap_reps: int = 250,
) -> dict[str, Any]:
    """Run the binary correction ladder with cross-fitted nuisances."""
    if dataset.treatment_type != TreatmentType.BINARY:
        raise ValueError("run_binary_bias_repair requires a binary CausalDataset")
    frame = dataset.to_pandas().reset_index(drop=True)
    treatment_values = sorted(frame[dataset.treatment_col].dropna().unique().tolist(), key=lambda x: str(x))
    t = frame[dataset.treatment_col].map({treatment_values[0]: 0, treatment_values[1]: 1}).to_numpy(dtype=float)
    y = pd.to_numeric(frame[dataset.outcome_col], errors="coerce").to_numpy(dtype=float)
    nuisance = estimate_propensity(dataset, estimator=estimator, cross_fit=True, n_splits=5, random_state=seed)
    p = np.asarray(nuisance.propensity_scores, dtype=float)
    safe = np.clip(p, .01, .99)
    raw_ipw = np.where(t == 1, 1.0 / safe, 1.0 / (1.0 - safe))
    marginal = float(t.mean())
    stabilized = np.where(t == 1, marginal / safe, (1.0 - marginal) / (1.0 - safe))
    overlap = np.where(t == 1, 1.0 - safe, safe)
    before = _smd(frame, t, dataset.feature_cols)
    max_before = max((abs(v) for v in before.values()), default=0.0)
    methods: list[dict[str, Any]] = []

    m0, m1 = nuisance.outcome_control_oof, nuisance.outcome_treated_oof
    dr_rows = m1 - m0 + t / safe * (y - m1) - (1.0 - t) / (1.0 - safe) * (y - m0)

    specs: list[tuple[str, str, np.ndarray, str, Callable[[np.ndarray], float], np.ndarray]] = [
        ("unadjusted", "difference in observed arm means", np.ones(len(y)), "association; no adjustment", lambda ids: _weighted_diff(y[ids], t[ids], np.ones(len(ids))), np.ones(len(y))),
        ("outcome_regression_crossfit", "cross-fitted E[Y|X,T=1]-E[Y|X,T=0]", np.ones(len(y)), "model-based standardization", lambda ids: float(np.mean((m1 - m0)[ids])), np.ones(len(y))),
        ("ipw", "Horvitz-Thompson inverse propensity weighting", raw_ipw, "ATE under consistency, positivity and no unmeasured confounding", lambda ids: _weighted_diff(y[ids], t[ids], raw_ipw[ids]), raw_ipw),
        ("stabilized_ipw", "marginal-treatment stabilized inverse propensity weighting", stabilized, "ATE under the same identification assumptions; lower variance is not guaranteed", lambda ids: _weighted_diff(y[ids], t[ids], stabilized[ids]), stabilized),
        ("overlap_weighting", "treated gets 1-e(X), control gets e(X)", overlap, "overlap-population effect under positivity", lambda ids: _weighted_diff(y[ids], t[ids], overlap[ids]), overlap),
        ("aipw_dr_crossfit", "cross-fitted augmented IPW pseudo-outcome", np.ones(len(y)), "doubly robust ATE if either nuisance component is consistently estimated", lambda ids: float(np.mean(dr_rows[ids])), np.ones(len(y))),
    ]
    for q in (.99, .995, .999):
        cap = float(np.quantile(raw_ipw, q))
        clipped = np.minimum(raw_ipw, cap)
        specs.append((f"ipw_clip_p{int(q*1000):03d}", f"IPW capped at raw-weight p{q:.3f}", clipped, "ATE target changes in finite samples when extreme weights are capped", lambda ids, w=clipped: _weighted_diff(y[ids], t[ids], w[ids]), clipped))

    for name, calculation, weights, estimand, calc, balance_weights in specs:
        estimate = calc(np.arange(len(y)))
        low, high = _bootstrap(calc, len(y), seed + len(methods), bootstrap_reps)
        after = _smd(frame, t, dataset.feature_cols, balance_weights)
        methods.append({
            "method": name, "calculation": calculation, "estimand_and_assumptions": estimand,
            "estimate": estimate, "ci95": [low, high], "uncertainty": "row bootstrap over fixed cross-fitted nuisance predictions",
            "max_abs_smd_before": max_before, "max_abs_smd_after": max((abs(v) for v in after.values()), default=None),
            "balance_after_smd": after, "weight_diagnostics": _weight_stats(weights, t),
            "cross_fitting": {"enabled": True, "folds": 5, "fold_ids_present": sorted(set(nuisance.fold_ids.tolist()))},
            "status": "PASS" if np.isfinite(estimate) else "FAIL",
        })

    match_estimate, match_weights, match_active = _matching_effect(y, t, p)
    match_fn = lambda ids: _matching_effect(y, t, p, ids)[0]
    low, high = _bootstrap(match_fn, len(y), seed + 101, max(50, bootstrap_reps // 2))
    match_after = _smd(frame, t, dataset.feature_cols, match_weights)
    methods.append({
        "method": "propensity_matching", "calculation": "nearest control on logit propensity, replacement allowed, 0.2 SD caliper",
        "estimand_and_assumptions": "ATT among matched treated; exchangeability and caliper support",
        "estimate": match_estimate, "ci95": [low, high], "uncertainty": "row bootstrap; matching is recomputed per resample",
        "matched_rows": int(match_active.sum()), "max_abs_smd_before": max_before,
        "max_abs_smd_after": max((abs(v) for v in match_after.values()), default=None), "balance_after_smd": match_after,
        "weight_diagnostics": _weight_stats(match_weights, t), "status": "PASS" if np.isfinite(match_estimate) else "FAIL",
    })
    return _finite({
        "dataset": dataset.metadata.get("dataset_name", "observational_binary"),
        "sample_size": len(frame), "treatment_mapping": {str(v): int(i) for i, v in enumerate(treatment_values)},
        "data_audit": {"treatment_counts": {str(v): int((frame[dataset.treatment_col] == v).sum()) for v in treatment_values}, "treatment_rate": float(np.mean(t)), "missingness": {str(k): int(v) for k, v in frame[[*dataset.feature_cols, dataset.treatment_col, dataset.outcome_col]].isna().sum().items()}, "duplicate_rows": int(frame.duplicated().sum()), "selection_note": "The causallib restricted loader supplies a complete-case analytical table; this result does not identify selection effects from rows excluded by the upstream loader."},
        "identification_warning": "NHEFS is observational; no method proves no unmeasured confounding and no true ATE is available.",
        "propensity": {"min": float(p.min()), "p01": float(np.quantile(p, .01)), "p50": float(np.median(p)), "p99": float(np.quantile(p, .99)), "max": float(p.max()), "common_support_05_95": float(np.mean((p >= .05) & (p <= .95))), "treated_mean": float(p[t == 1].mean()), "control_mean": float(p[t == 0].mean())},
        "balance_before_smd": before, "methods": methods,
        "nuisance_diagnostics": nuisance.diagnostics, "source": dataset.metadata.get("source"),
    })


def run_continuous_gps_repair(dataset: CausalDataset, *, seed: int = 42, clipping_quantiles: tuple[float, ...] = (.99, .995, .999)) -> dict[str, Any]:
    """Fit a Gaussian GPS nuisance and report density-ratio support diagnostics."""
    if dataset.treatment_type != TreatmentType.CONTINUOUS:
        raise ValueError("run_continuous_gps_repair requires a continuous CausalDataset")
    frame = dataset.to_pandas().reset_index(drop=True)
    dose = pd.to_numeric(frame[dataset.treatment_col], errors="coerce").to_numpy(dtype=float)
    outcome = pd.to_numeric(frame[dataset.outcome_col], errors="coerce").to_numpy(dtype=float)
    prep = TabularPreprocessor(dataset.feature_cols)
    x = prep.fit_transform(frame[dataset.feature_cols])
    gps_model = HistGradientBoostingRegressor(max_iter=200, learning_rate=.05, max_leaf_nodes=15, random_state=seed)
    gps_model.fit(x, dose)
    mean = gps_model.predict(x)
    residual_sd = max(float(np.std(dose - mean)), 1e-3)
    conditional = norm.pdf(dose, loc=mean, scale=residual_sd)
    try:
        marginal = gaussian_kde(dose)(dose)
    except (ValueError, np.linalg.LinAlgError):
        marginal = norm.pdf(dose, loc=float(np.mean(dose)), scale=max(float(np.std(dose)), 1e-3))
    raw_ratio = marginal / np.maximum(conditional, 1e-12)
    bins = pd.qcut(dose, q=min(10, max(2, len(np.unique(dose)) // 10)), duplicates="drop")
    bin_rows = []
    for key, ids in pd.Series(np.arange(len(dose))).groupby(bins, observed=False):
        idx = ids.to_numpy(dtype=int)
        bin_rows.append({"dose_bin": str(key), "count": len(idx), "dose_mean": float(np.mean(dose[idx])), "outcome_mean": float(np.mean(outcome[idx])), "gps_weighted_outcome_mean": float(np.average(outcome[idx], weights=raw_ratio[idx]))})
    weight_rows = []
    for q in clipping_quantiles:
        cap = float(np.quantile(raw_ratio, q))
        weights = np.minimum(raw_ratio, cap)
        weight_rows.append({"method": f"gps_density_ratio_clip_p{int(q*1000):03d}", "cap": cap, "weight_diagnostics": {"min": float(weights.min()), "p50": float(np.quantile(weights,.5)), "p95": float(np.quantile(weights,.95)), "p99": float(np.quantile(weights,.99)), "max": float(weights.max()), "ess": effective_sample_size(weights)}, "weighted_outcome_mean": float(np.average(outcome, weights=weights)), "weighted_dose_mean": float(np.average(dose, weights=weights)), "support_warning": bool(np.any(weights >= cap))})
    return _finite({
        "dataset": dataset.metadata.get("dataset_name", "observational_continuous"), "sample_size": len(frame),
        "data_audit": {"missingness": {str(k): int(v) for k, v in frame[[*dataset.feature_cols, dataset.treatment_col, dataset.outcome_col]].isna().sum().items()}, "duplicate_rows": int(frame.duplicated().sum()), "selection_note": "Rows missing continuous dose or outcome were excluded; baseline covariates are retained and imputed by the model preprocessor."},
        "gps": {"model": "HistGradientBoostingRegressor conditional mean + Gaussian residual", "residual_sd": residual_sd, "dose_range": [float(dose.min()), float(dose.max())], "dose_quantiles": {str(q): float(np.quantile(dose,q)) for q in (.01,.05,.5,.95,.99)}, "conditional_density_min": float(conditional.min()), "raw_density_ratio": {"p50": float(np.quantile(raw_ratio,.5)), "p95": float(np.quantile(raw_ratio,.95)), "p99": float(np.quantile(raw_ratio,.99)), "max": float(raw_ratio.max())}},
        "support_diagnostic": diagnose_continuous_treatment(dataset, bins=10), "dose_response_bins": bin_rows, "clipping_ablations": weight_rows,
        "identification_warning": "Continuous GPS relies on conditional-density specification and generalized positivity; dose-response estimates outside local support are not identified.", "source": dataset.metadata.get("source"),
    })


__all__ = ["run_binary_bias_repair", "run_continuous_gps_repair"]
