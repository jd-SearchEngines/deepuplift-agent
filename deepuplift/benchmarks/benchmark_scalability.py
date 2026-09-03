from __future__ import annotations

import gc
import resource
import time
from typing import Iterable

import numpy as np
import pandas as pd

from deepuplift.data.public import synthetic_ground_truth
from deepuplift.data import estimate_nuisance
from deepuplift.models import build_model
from deepuplift.decision import build_binary_policy


def run_scalability_benchmark(sizes: Iterable[int] = (10_000,), *, seed: int = 42, model_name: str = "DR-Learner") -> list[dict]:
    """Run measured local scale smoke(s); never extrapolates to unsupported sizes."""
    results = []
    for rows in sizes:
        started = time.perf_counter()
        dataset = synthetic_ground_truth(rows=rows, seed=seed)
        loaded = time.perf_counter()
        nuisance = estimate_nuisance(dataset, n_splits=5, weighting="none")
        nuisanced = time.perf_counter()
        model = build_model(model_name, random_state=seed)
        model.fit(dataset, nuisance=nuisance)
        fitted = time.perf_counter()
        prediction = model.predict(dataset)
        predicted = time.perf_counter()
        build_binary_policy(prediction, treatment_cost=0.01, outcome_value=1.0)
        decided = time.perf_counter()
        maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        memory_mb = maxrss / (1024 * 1024) if maxrss > 10_000_000 else maxrss / 1024
        results.append({"rows": rows, "load_seconds": loaded - started, "fit_seconds": fitted - loaded, "predict_seconds": predicted - fitted, "decision_seconds": decided - predicted, "total_seconds": decided - started, "rows_per_second": rows / max(decided - started, 1e-9), "memory_mb": memory_mb, "memory_status": "MEASURED_PROCESS_MAXRSS", "model": model_name, "seed": seed})
        results[-1].update({"preprocess_seconds": loaded - started, "nuisance_seconds": nuisanced - loaded, "fit_seconds": fitted - nuisanced, "peak_memory_mb": memory_mb})
        del dataset, nuisance, model; gc.collect()
    return results


__all__ = ["run_scalability_benchmark"]
