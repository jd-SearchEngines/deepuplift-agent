from __future__ import annotations

from typing import Any

import numpy as np


def bootstrap_mean(values: Any, *, samples: int = 100, random_state: int = 42) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype="float64")
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"samples": 0, "mean": None, "low": None, "high": None}
    rng = np.random.default_rng(random_state)
    estimates = [float(np.mean(values[rng.integers(0, len(values), len(values))])) for _ in range(max(1, samples))]
    return {"samples": len(estimates), "mean": float(np.mean(values)), "low": float(np.quantile(estimates, 0.025)), "high": float(np.quantile(estimates, 0.975))}
