from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


def _as_series(values: Any, name: str) -> pd.Series:
    if isinstance(values, pd.Series):
        return values.reset_index(drop=True)
    return pd.Series(values, name=name)


def _clean_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def _ci(values: np.ndarray, alpha: float = 0.05) -> dict[str, float | None]:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return {"mean": None, "low": None, "high": None, "std": None}
    return {
        "mean": _clean_float(np.mean(finite)),
        "low": _clean_float(np.quantile(finite, alpha / 2)),
        "high": _clean_float(np.quantile(finite, 1 - alpha / 2)),
        "std": _clean_float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
    }


def effective_sample_size(weights: Any) -> float:
    values = np.asarray(weights, dtype="float64")
    values = values[np.isfinite(values)]
    denom = float(np.sum(values**2))
    if denom <= 0:
        return 0.0
    return float(np.sum(values) ** 2 / denom)


def evaluate_ope_policy(
    *,
    logged_action: Any,
    reward: Any,
    logged_propensity: Any,
    target_action: Any,
    target_reward_pred: Any | None = None,
    logged_reward_pred: Any | None = None,
    clip_min: float = 0.02,
    clip_max: float = 1.0,
    n_bootstrap: int = 0,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Estimate target-policy value from logged data with IPS/SNIPS/DR.

    Inputs are intentionally array-like rather than framework-specific so this
    can support binary uplift, multi-treatment coupons, and LLM routing logs.
    For DR, pass the direct-method prediction for the target action and the
    logged action. Without those two arrays, the function reports IPS/SNIPS
    only and marks DR as unavailable.
    """

    logged = _as_series(logged_action, "logged_action").astype(str)
    target = _as_series(target_action, "target_action").astype(str)
    y = pd.to_numeric(_as_series(reward, "reward"), errors="coerce").astype("float64")
    p_raw = pd.to_numeric(_as_series(logged_propensity, "logged_propensity"), errors="coerce").astype("float64")
    if not (len(logged) == len(target) == len(y) == len(p_raw)):
        raise ValueError("logged_action, reward, logged_propensity, and target_action must have the same length.")

    mask = logged.notna() & target.notna() & y.notna() & p_raw.notna()
    logged = logged[mask].reset_index(drop=True)
    target = target[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)
    p_raw = p_raw[mask].reset_index(drop=True)
    if len(y) == 0:
        return {"status": "empty", "rows": 0, "metrics": {}, "diagnostics": {"valid_rows": 0}}

    p = p_raw.clip(lower=float(clip_min), upper=float(clip_max))
    match = (logged == target).to_numpy(dtype="float64")
    weights = match / p.to_numpy(dtype="float64")
    weighted_reward = weights * y.to_numpy(dtype="float64")
    ips = float(np.mean(weighted_reward))
    weight_sum = float(np.sum(weights))
    snips = float(np.sum(weighted_reward) / weight_sum) if weight_sum > 1e-12 else None

    metrics: dict[str, Any] = {
        "ips": _clean_float(ips),
        "snips": _clean_float(snips),
        "dr": None,
    }
    dr_values = None
    if target_reward_pred is not None and logged_reward_pred is not None:
        mu_target = (
            pd.to_numeric(_as_series(target_reward_pred, "target_reward_pred")[mask], errors="coerce")
            .astype("float64")
            .reset_index(drop=True)
        )
        mu_logged = (
            pd.to_numeric(_as_series(logged_reward_pred, "logged_reward_pred")[mask], errors="coerce")
            .astype("float64")
            .reset_index(drop=True)
        )
        valid = mu_target.notna() & mu_logged.notna()
        if valid.any():
            dr_values = mu_target[valid].to_numpy(dtype="float64") + weights[valid.to_numpy()] * (
                y[valid].to_numpy(dtype="float64") - mu_logged[valid].to_numpy(dtype="float64")
            )
            metrics["dr"] = _clean_float(np.mean(dr_values))

    diagnostics = {
        "valid_rows": int(len(y)),
        "match_rows": int(match.sum()),
        "coverage_rate": _clean_float(match.mean()),
        "raw_propensity_min": _clean_float(p_raw.min()),
        "raw_propensity_p05": _clean_float(p_raw.quantile(0.05)),
        "raw_propensity_median": _clean_float(p_raw.median()),
        "raw_propensity_p95": _clean_float(p_raw.quantile(0.95)),
        "raw_propensity_max": _clean_float(p_raw.max()),
        "clip_min": float(clip_min),
        "clip_max": float(clip_max),
        "clipped_rate": _clean_float(((p_raw < clip_min) | (p_raw > clip_max)).mean()),
        "effective_sample_size": _clean_float(effective_sample_size(weights)),
        "effective_sample_fraction": _clean_float(effective_sample_size(weights) / len(y)),
    }

    bootstrap = {}
    if n_bootstrap > 0:
        rng = np.random.default_rng(random_state)
        ips_samples = []
        snips_samples = []
        dr_samples = []
        y_np = y.to_numpy(dtype="float64")
        for _ in range(int(n_bootstrap)):
            indices = rng.integers(0, len(y_np), len(y_np))
            w_b = weights[indices]
            y_b = y_np[indices]
            ips_samples.append(float(np.mean(w_b * y_b)))
            denom = float(np.sum(w_b))
            snips_samples.append(float(np.sum(w_b * y_b) / denom) if denom > 1e-12 else np.nan)
            if dr_values is not None and len(dr_values) == len(y_np):
                dr_samples.append(float(np.mean(dr_values[indices])))
        bootstrap = {
            "n_bootstrap": int(n_bootstrap),
            "ips": _ci(np.asarray(ips_samples, dtype="float64")),
            "snips": _ci(np.asarray(snips_samples, dtype="float64")),
        }
        if dr_samples:
            bootstrap["dr"] = _ci(np.asarray(dr_samples, dtype="float64"))

    return {
        "status": "ok",
        "rows": int(len(y)),
        "metrics": metrics,
        "diagnostics": diagnostics,
        "bootstrap": bootstrap,
        "guidance": _ope_guidance(diagnostics, metrics),
    }


def _ope_guidance(diagnostics: dict[str, Any], metrics: dict[str, Any]) -> list[str]:
    rows = []
    if (diagnostics.get("coverage_rate") or 0.0) < 0.05:
        rows.append("Target policy has very low overlap with logged actions; treat OPE estimates as high risk.")
    if (diagnostics.get("effective_sample_fraction") or 0.0) < 0.1:
        rows.append("Effective sample size is low; require more randomized exploration or stronger clipping sensitivity checks.")
    if metrics.get("dr") is None:
        rows.append("DR estimate is unavailable because direct-method predictions were not supplied.")
    if not rows:
        rows.append("OPE overlap is usable for offline review, but online holdout is still required before rollout.")
    return rows


def clipping_sensitivity(
    *,
    logged_action: Any,
    reward: Any,
    logged_propensity: Any,
    target_action: Any,
    target_reward_pred: Any | None = None,
    logged_reward_pred: Any | None = None,
    clip_grid: list[float] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for clip in clip_grid or [0.01, 0.02, 0.05, 0.10]:
        result = evaluate_ope_policy(
            logged_action=logged_action,
            reward=reward,
            logged_propensity=logged_propensity,
            target_action=target_action,
            target_reward_pred=target_reward_pred,
            logged_reward_pred=logged_reward_pred,
            clip_min=float(clip),
        )
        rows.append(
            {
                "clip_min": float(clip),
                "ips": result.get("metrics", {}).get("ips"),
                "snips": result.get("metrics", {}).get("snips"),
                "dr": result.get("metrics", {}).get("dr"),
                "coverage_rate": result.get("diagnostics", {}).get("coverage_rate"),
                "effective_sample_size": result.get("diagnostics", {}).get("effective_sample_size"),
                "effective_sample_fraction": result.get("diagnostics", {}).get("effective_sample_fraction"),
            }
        )
    return rows
