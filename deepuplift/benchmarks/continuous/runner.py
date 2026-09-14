from __future__ import annotations

import platform
import resource
import sys
import threading
import time
from typing import Any

import numpy as np

from deepuplift.contracts import CausalDataset
from deepuplift.decision import build_continuous_policy
from deepuplift.models import build_model, model_info
from .datasets import SyntheticContinuousData, continuous_dose_cost
from .metrics import continuous_curve_metrics


def peak_memory_mb() -> float:
    peak = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak / (1024.0 * 1024.0) if sys.platform == "darwin" else peak / 1024.0


class _PeakMemorySampler:
    def __init__(self, interval: float = 0.02) -> None:
        self.interval = interval
        self.peak_mb = 0.0
        self._stop = threading.Event()
        self._thread = None
        try:
            import psutil
            self._process = psutil.Process()
            self.method = "psutil_rss_sampled"
        except ImportError:
            self._process = None
            self.method = "process_ru_maxrss_lifetime"

    def _sample(self) -> None:
        while not self._stop.is_set():
            if self._process is not None:
                self.peak_mb = max(self.peak_mb, self._process.memory_info().rss / (1024.0 * 1024.0))
            self._stop.wait(self.interval)

    def start(self) -> None:
        if self._process is None:
            self.peak_mb = peak_memory_mb()
            return
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def stop(self) -> float:
        if self._process is None:
            return peak_memory_mb()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        return max(self.peak_mb, self._process.memory_info().rss / (1024.0 * 1024.0))


def environment_provenance() -> dict[str, Any]:
    import importlib.metadata

    import numpy
    import pandas
    import sklearn

    optional = {}
    for name in ("torch", "ccpfn", "econml", "causalml"):
        try:
            optional[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            optional[name] = None
    try:
        optional["psutil"] = importlib.metadata.version("psutil")
    except importlib.metadata.PackageNotFoundError:
        optional["psutil"] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scikit_learn": sklearn.__version__,
        "optional_dependencies": optional,
        "device": "cpu",
    }


def run_continuous_model(
    model_name: str,
    train: CausalDataset,
    test: CausalDataset,
    *,
    seed: int,
    dose_grid: np.ndarray,
    true_response_curves: np.ndarray,
    true_effect_curves: np.ndarray,
    true_optimal_effect_dose: np.ndarray,
    true_optimal_economic_dose: np.ndarray,
    outcome_value: float = 1.0,
    budget: float | None = None,
    dose_cost=None,
    model_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    options = dict(model_options or {})
    options.setdefault("grid_size", len(dose_grid))
    if model_name == "DoseResponseGBM":
        options.setdefault("task", "regression")
    memory_sampler = _PeakMemorySampler()
    memory_sampler.start()
    try:
        start = time.perf_counter()
        model = build_model(model_name, random_state=seed, **options)
        model.fit(train)
        train_seconds = time.perf_counter() - start
        start = time.perf_counter()
        prediction = model.predict(test)
        predict_seconds = time.perf_counter() - start
        start = time.perf_counter()
        cost_fn = continuous_dose_cost if dose_cost is None else dose_cost
        policy = build_continuous_policy(
            prediction,
            dose_cost=lambda dose: float(cost_fn(dose)),
            outcome_value=outcome_value,
            budget=budget,
        )
        decision_seconds = time.perf_counter() - start
    except BaseException:
        memory_sampler.stop()
        raise
    peak_mb = memory_sampler.stop()
    metrics = continuous_curve_metrics(
        prediction,
        dose_grid=dose_grid,
        true_response_curves=true_response_curves,
        true_effect_curves=true_effect_curves,
        true_optimal_effect_dose=true_optimal_effect_dose,
        true_optimal_economic_dose=true_optimal_economic_dose,
        outcome_value=outcome_value,
        dose_cost=cost_fn,
    )
    info = model_info(model_name)
    preview_rows = []
    preview_count = min(3, len(prediction.unit_id))
    grid = np.asarray(prediction.dose_grid, dtype="float64")
    for index in range(preview_count):
        preview_rows.append({
            "unit_id": str(prediction.unit_id[index]),
            "dose_grid": grid.tolist(),
            "outcome_curve": [float(np.asarray(prediction.dose_outcome_predictions[float(dose)])[index]) for dose in grid],
            "effect_curve": [float(np.asarray(prediction.dose_effect_predictions[float(dose)])[index]) for dose in grid],
            "max_effect_dose": float(prediction.recommended_treatment[index]),
            "economic_dose": float(policy.rows[index]["recommended_treatment"]),
            "economic_net_value": float(policy.rows[index]["net_value"]),
        })
    return {
        "model": model_name,
        "status": "PASS" if policy.rows and np.isfinite(list(metrics.values())[:-1]).all() else "INCONCLUSIVE",
        "maturity": info["maturity"],
        "backend": info["backend"],
        "runnable": info["runnable"],
        "train_seconds": train_seconds,
        "predict_curve_seconds": predict_seconds,
        "decision_seconds": decision_seconds,
        "peak_memory_mb": peak_mb,
        "memory_measurement": memory_sampler.method,
        "device": "cpu",
        "metrics": metrics,
        "prediction_metadata": prediction.metadata,
        "curve_preview": preview_rows,
        "policy": {"rows": policy.rows, "summary": policy.summary, "metadata": policy.metadata},
        "model_info": info,
    }


def run_continuous_dataset(
    data: SyntheticContinuousData,
    *,
    model_names: list[str] | None = None,
    test_fraction: float = 0.25,
    seed: int | None = None,
    outcome_value: float | None = None,
    budget: float | None = None,
    model_options: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    names = model_names or ["DoseResponseGBM", "DRNet", "GIKS-DRNet", "VCNet", "GIKS-VCNet"]
    seed = data.seed if seed is None else int(seed)
    outcome_value = data.outcome_value if outcome_value is None else float(outcome_value)
    row_count = len(data.dataset.to_pandas())
    if data.fixed_train_indices is not None and data.fixed_test_indices is not None:
        train_ids = np.asarray(data.fixed_train_indices, dtype="int64")
        test_ids = np.asarray(data.fixed_test_indices, dtype="int64")
        if np.intersect1d(train_ids, test_ids).size or len(np.union1d(train_ids, test_ids)) != row_count:
            raise ValueError("Fixed benchmark split must partition all rows without overlap.")
        split_description = "official pre-generated non-overlapping split"
    else:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(row_count)
        test_count = max(1, int(round(len(indices) * test_fraction)))
        test_ids, train_ids = indices[:test_count], indices[test_count:]
        split_description = "deterministic random permutation, no overlap"
    train, test = data.dataset.subset(train_ids), data.dataset.subset(test_ids)
    options = model_options or {}
    dose_cost = lambda dose: continuous_dose_cost(np.asarray(dose, dtype="float64") * (20.0 / data.cost_dose_range))
    results = []
    for name in names:
        try:
            results.append(
                run_continuous_model(
                    name,
                    train,
                    test,
                    seed=seed,
                    dose_grid=data.dose_grid,
                    true_response_curves=data.true_response_curves[test_ids],
                    true_effect_curves=data.true_effect_curves[test_ids],
                    true_optimal_effect_dose=data.true_optimal_effect_dose[test_ids],
                    true_optimal_economic_dose=data.true_optimal_economic_dose[test_ids],
                    outcome_value=outcome_value,
                    budget=budget,
                    dose_cost=dose_cost,
                    model_options=options.get(name),
                )
            )
        except Exception as exc:
            results.append({"model": name, "status": "BLOCKED", "error": f"{type(exc).__name__}: {exc}", "model_info": model_info(name) if name in {"DoseResponseGBM", "DRNet", "VCNet", "GIKS-VCNet", "GIKS-DRNet", "CCPFN", "TransTEE"} else None})
    return {
        "dataset": {
            **data.dataset.metadata,
            "sample_size": len(data.dataset.to_pandas()),
            "train_size": len(train_ids),
            "test_size": len(test_ids),
            "split": split_description,
            "split_seed": seed,
            "ground_truth_available": True,
            "treatment_range": [float(data.dataset.treatment.min()), float(data.dataset.treatment.max())],
            "feature_dimension": len(data.dataset.feature_cols),
        },
        "config": {"dataset": data.dataset.metadata.get("name"), "models": names, "seed": seed, "test_fraction": test_fraction, "outcome_value": outcome_value, "budget": budget, "dose_grid": data.dose_grid.tolist(), "model_options": options, "fixed_split": data.fixed_train_indices is not None},
        "environment": environment_provenance(),
        "models": results,
        "status": (
            "PASS" if results and all(row.get("status") == "PASS" for row in results)
            else "PARTIAL" if any(row.get("status") == "PASS" for row in results)
            else "BLOCKED"
        ),
    }


__all__ = ["environment_provenance", "peak_memory_mb", "run_continuous_model", "run_continuous_dataset"]
