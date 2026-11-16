from __future__ import annotations

from typing import Any
from collections.abc import Callable

import numpy as np


def bootstrap_ci(
    values: Any,
    statistic: Callable[[np.ndarray], float] = np.mean,
    *,
    samples: int = 1000,
    random_state: int = 42,
) -> dict[str, float | int | None]:
    """Bootstrap a row-level statistic and return a percentile 95% interval."""
    array = np.asarray(values, dtype="float64").reshape(-1)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {"samples": 0, "estimate": None, "low": None, "high": None, "random_state": int(random_state)}
    estimate = float(statistic(array))
    rng = np.random.default_rng(random_state)
    draws = [float(statistic(array[rng.integers(0, len(array), len(array))])) for _ in range(max(1, int(samples)))]
    return {
        "samples": len(draws),
        "estimate": estimate,
        "low": float(np.quantile(draws, 0.025)),
        "high": float(np.quantile(draws, 0.975)),
        "random_state": int(random_state),
    }


def bootstrap_cluster_ci(
    values: Any,
    clusters: Any,
    statistic: Callable[[np.ndarray], float] = np.mean,
    *,
    samples: int = 1000,
    random_state: int = 42,
) -> dict[str, float | int | None]:
    """Bootstrap a statistic by resampling independent clusters."""
    array = np.asarray(values, dtype="float64").reshape(-1)
    cluster_array = np.asarray(clusters).reshape(-1)
    if len(array) != len(cluster_array):
        raise ValueError("values and clusters must have equal length.")
    finite = np.isfinite(array)
    array = array[finite]
    cluster_array = cluster_array[finite]
    if len(array) == 0:
        return {
            "samples": 0,
            "estimate": None,
            "low": None,
            "high": None,
            "random_state": int(random_state),
            "cluster_count": 0,
        }
    unique_clusters, inverse = np.unique(cluster_array, return_inverse=True)
    members = [np.flatnonzero(inverse == index) for index in range(len(unique_clusters))]
    estimate = float(statistic(array))
    rng = np.random.default_rng(random_state)
    draws = []
    for _ in range(max(1, int(samples))):
        selected = rng.integers(0, len(members), len(members))
        row_indices = np.concatenate([members[index] for index in selected])
        draws.append(float(statistic(array[row_indices])))
    return {
        "samples": len(draws),
        "estimate": estimate,
        "low": float(np.quantile(draws, 0.025)),
        "high": float(np.quantile(draws, 0.975)),
        "random_state": int(random_state),
        "cluster_count": int(len(unique_clusters)),
    }


def bootstrap_mean(values: Any, *, samples: int = 100, random_state: int = 42) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype="float64")
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"samples": 0, "mean": None, "low": None, "high": None}
    rng = np.random.default_rng(random_state)
    estimates = [float(np.mean(values[rng.integers(0, len(values), len(values))])) for _ in range(max(1, samples))]
    return {"samples": len(estimates), "mean": float(np.mean(values)), "low": float(np.quantile(estimates, 0.025)), "high": float(np.quantile(estimates, 0.975))}
