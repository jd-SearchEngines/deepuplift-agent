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


def _bootstrap_indices(rng: np.random.Generator, size: int, clusters: np.ndarray | None = None) -> np.ndarray:
    if clusters is None:
        return rng.integers(0, size, size)
    unique_clusters = pd.unique(clusters)
    sampled_clusters = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
    pieces = [np.flatnonzero(clusters == cluster) for cluster in sampled_clusters]
    return np.concatenate(pieces) if pieces else np.empty(0, dtype="int64")


def evaluate_ope_policy(
    *,
    logged_action: Any,
    reward: Any,
    logged_propensity: Any,
    target_action: Any,
    target_reward_pred: Any | None = None,
    logged_reward_pred: Any | None = None,
    cluster_id: Any | None = None,
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
    only and marks DR as unavailable. For repeated-decision data, pass
    ``cluster_id`` to bootstrap complete participants/users instead of iid
    decision rows.
    """

    logged = _as_series(logged_action, "logged_action").astype(str)
    target = _as_series(target_action, "target_action").astype(str)
    y = pd.to_numeric(_as_series(reward, "reward"), errors="coerce").astype("float64")
    p_raw = pd.to_numeric(_as_series(logged_propensity, "logged_propensity"), errors="coerce").astype("float64")
    original_size = len(logged)
    if not (len(logged) == len(target) == len(y) == len(p_raw)):
        raise ValueError("logged_action, reward, logged_propensity, and target_action must have the same length.")
    if target_reward_pred is not None and len(_as_series(target_reward_pred, "target_reward_pred")) != original_size:
        raise ValueError("target_reward_pred must have the same length as logged_action.")
    if logged_reward_pred is not None and len(_as_series(logged_reward_pred, "logged_reward_pred")) != original_size:
        raise ValueError("logged_reward_pred must have the same length as logged_action.")
    if cluster_id is not None and len(_as_series(cluster_id, "cluster_id")) != original_size:
        raise ValueError("cluster_id must have the same length as logged_action.")

    mask = logged.notna() & target.notna() & y.notna() & p_raw.notna()
    clusters = None
    if cluster_id is not None:
        clusters = _as_series(cluster_id, "cluster_id")
        mask = mask & clusters.notna()
    logged = logged[mask].reset_index(drop=True)
    target = target[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)
    p_raw = p_raw[mask].reset_index(drop=True)
    if clusters is not None:
        clusters = clusters[mask].astype(str).reset_index(drop=True)
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
        "cluster_count": int(clusters.nunique()) if clusters is not None else None,
        "bootstrap_unit": "cluster" if clusters is not None else "row",
    }

    bootstrap = {}
    if n_bootstrap > 0:
        rng = np.random.default_rng(random_state)
        ips_samples = []
        snips_samples = []
        dr_samples = []
        y_np = y.to_numpy(dtype="float64")
        for _ in range(int(n_bootstrap)):
            indices = _bootstrap_indices(rng, len(y_np), clusters.to_numpy() if clusters is not None else None)
            w_b = weights[indices]
            y_b = y_np[indices]
            ips_samples.append(float(np.mean(w_b * y_b)))
            denom = float(np.sum(w_b))
            snips_samples.append(float(np.sum(w_b * y_b) / denom) if denom > 1e-12 else np.nan)
            if dr_values is not None and len(dr_values) == len(y_np):
                dr_samples.append(float(np.mean(dr_values[indices])))
        bootstrap = {
            "n_bootstrap": int(n_bootstrap),
            "bootstrap_unit": "cluster" if clusters is not None else "row",
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


def _sequential_metrics(
    ratios: np.ndarray,
    rewards: np.ndarray,
    clusters: np.ndarray,
    target_reward_pred: np.ndarray | None = None,
    logged_reward_pred: np.ndarray | None = None,
) -> dict[str, float | None]:
    row_ips = float(np.mean(ratios * rewards))
    ratio_sum = float(np.sum(ratios))
    row_snips = float(np.sum(ratios * rewards) / ratio_sum) if ratio_sum > 1e-12 else None
    trajectory_weights: list[float] = []
    trajectory_returns: list[float] = []
    for cluster in pd.unique(clusters):
        indices = np.flatnonzero(clusters == cluster)
        trajectory_weights.append(float(np.prod(ratios[indices])))
        trajectory_returns.append(float(np.sum(rewards[indices])))
    trajectory_w = np.asarray(trajectory_weights, dtype="float64")
    trajectory_y = np.asarray(trajectory_returns, dtype="float64")
    trajectory_ips = float(np.mean(trajectory_w * trajectory_y)) if len(trajectory_y) else None
    trajectory_sum = float(np.sum(trajectory_w))
    trajectory_snips = float(np.sum(trajectory_w * trajectory_y) / trajectory_sum) if trajectory_sum > 1e-12 else None
    result: dict[str, float | None] = {
        "per_decision_ips": row_ips,
        "per_decision_snips": row_snips,
        "trajectory_ips": trajectory_ips,
        "trajectory_snips": trajectory_snips,
    }
    if target_reward_pred is not None and logged_reward_pred is not None:
        result["per_decision_dr"] = float(
            np.mean(target_reward_pred + ratios * (rewards - logged_reward_pred))
        )
    return result


def evaluate_sequential_ope_policy(
    *,
    logged_action: Any,
    reward: Any,
    logged_propensity: Any,
    target_action: Any,
    cluster_id: Any,
    decision_time: Any,
    target_reward_pred: Any | None = None,
    logged_reward_pred: Any | None = None,
    clip_min: float = 0.02,
    clip_max: float = 1.0,
    n_bootstrap: int = 0,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Evaluate a target policy on repeated decision trajectories.

    The evaluator reports per-decision IPS/SNIPS, trajectory IPS/SNIPS, and a
    per-decision DR correction when direct-method predictions are supplied.
    A full sequential DR estimator needs continuation-value models and is not
    silently substituted here. Bootstrap resamples complete clusters.
    """

    logged = _as_series(logged_action, "logged_action").astype(str)
    target = _as_series(target_action, "target_action").astype(str)
    y = pd.to_numeric(_as_series(reward, "reward"), errors="coerce").astype("float64")
    p_raw = pd.to_numeric(_as_series(logged_propensity, "logged_propensity"), errors="coerce").astype("float64")
    clusters = _as_series(cluster_id, "cluster_id")
    times = _as_series(decision_time, "decision_time")
    lengths = {len(logged), len(target), len(y), len(p_raw), len(clusters), len(times)}
    if len(lengths) != 1:
        raise ValueError("sequential OPE inputs must have the same length.")
    original_size = len(logged)
    if target_reward_pred is not None and len(_as_series(target_reward_pred, "target_reward_pred")) != original_size:
        raise ValueError("target_reward_pred must have the same length as logged_action.")
    if logged_reward_pred is not None and len(_as_series(logged_reward_pred, "logged_reward_pred")) != original_size:
        raise ValueError("logged_reward_pred must have the same length as logged_action.")
    mask = logged.notna() & target.notna() & y.notna() & p_raw.notna() & clusters.notna() & times.notna()
    if not mask.any():
        return {"status": "empty", "rows": 0, "metrics": {}, "diagnostics": {"valid_rows": 0}}
    frame = pd.DataFrame(
        {
            "logged": logged[mask].to_numpy(),
            "target": target[mask].to_numpy(),
            "reward": y[mask].to_numpy(dtype="float64"),
            "propensity": p_raw[mask].to_numpy(dtype="float64"),
            "cluster": clusters[mask].astype(str).to_numpy(),
            "time": times[mask].to_numpy(),
        }
    )
    if target_reward_pred is not None:
        frame["target_reward_pred"] = pd.to_numeric(_as_series(target_reward_pred, "target_reward_pred")[mask], errors="coerce").to_numpy()
    if logged_reward_pred is not None:
        frame["logged_reward_pred"] = pd.to_numeric(_as_series(logged_reward_pred, "logged_reward_pred")[mask], errors="coerce").to_numpy()
    frame = frame.sort_values(["cluster", "time"], kind="mergesort").reset_index(drop=True)
    if frame.duplicated(["cluster", "time"]).any():
        raise ValueError("Each cluster must have at most one row per decision_time.")
    p = frame["propensity"].clip(lower=float(clip_min), upper=float(clip_max)).to_numpy(dtype="float64")
    match = (frame["logged"].to_numpy() == frame["target"].to_numpy()).astype("float64")
    ratios = match / p
    rewards = frame["reward"].to_numpy(dtype="float64")
    target_pred = frame["target_reward_pred"].to_numpy(dtype="float64") if "target_reward_pred" in frame else None
    logged_pred = frame["logged_reward_pred"].to_numpy(dtype="float64") if "logged_reward_pred" in frame else None
    metrics = _sequential_metrics(ratios, rewards, frame["cluster"].to_numpy(), target_pred, logged_pred)
    diagnostics = {
        "valid_rows": int(len(frame)),
        "cluster_count": int(frame["cluster"].nunique()),
        "max_horizon": int(frame.groupby("cluster", sort=False).size().max()),
        "coverage_rate": _clean_float(match.mean()),
        "raw_propensity_min": _clean_float(frame["propensity"].min()),
        "raw_propensity_max": _clean_float(frame["propensity"].max()),
        "clipped_rate": _clean_float(((frame["propensity"] < clip_min) | (frame["propensity"] > clip_max)).mean()),
        "per_decision_weight_max": _clean_float(np.max(ratios)),
        "per_decision_weight_variance": _clean_float(np.var(ratios)),
        "trajectory_weight_max": _clean_float(max(np.prod(ratios[frame["cluster"].to_numpy() == cluster]) for cluster in pd.unique(frame["cluster"]))),
        "bootstrap_unit": "cluster",
        "sequential_dr_status": "NOT_AVAILABLE; continuation-value model not supplied",
    }
    bootstrap: dict[str, Any] = {}
    if n_bootstrap > 0:
        rng = np.random.default_rng(random_state)
        unique_clusters = pd.unique(frame["cluster"])
        samples: dict[str, list[float]] = {key: [] for key, value in metrics.items() if value is not None}
        groups = {cluster: np.flatnonzero(frame["cluster"].to_numpy() == cluster) for cluster in unique_clusters}
        for _ in range(int(n_bootstrap)):
            sampled_clusters = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
            pieces = [groups[cluster] for cluster in sampled_clusters]
            indices = np.concatenate(pieces) if pieces else np.empty(0, dtype="int64")
            boot_group = np.concatenate([np.full(len(piece), position, dtype="int64") for position, piece in enumerate(pieces)])
            values = _sequential_metrics(
                ratios[indices],
                rewards[indices],
                boot_group,
                target_pred[indices] if target_pred is not None else None,
                logged_pred[indices] if logged_pred is not None else None,
            )
            for key in samples:
                if values.get(key) is not None:
                    samples[key].append(float(values[key]))
        bootstrap = {
            "n_bootstrap": int(n_bootstrap),
            "bootstrap_unit": "cluster",
            **{key: _ci(np.asarray(values, dtype="float64")) for key, values in samples.items()},
        }
    return {
        "status": "ok",
        "rows": int(len(frame)),
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
    cluster_id: Any | None = None,
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
            cluster_id=cluster_id,
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
