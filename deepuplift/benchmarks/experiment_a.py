"""Experiment A local model-comparison runners.

The runners in this module deliberately keep evidence classes separate.  When
the MRT is not available locally, the binary and multi-treatment fixtures are
labelled ``SYNTHETIC_SCALE_STRESS`` rather than being presented as Push-derived
semi-synthetic data.  The continuous runner reuses the existing benchmark
contract and does not alter the public model APIs.
"""

from __future__ import annotations

from collections.abc import Iterable
import hashlib
import math
import resource
import sys
import threading
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from deepuplift.contracts import AssignmentType, TreatmentType
from deepuplift.data.public.datasets import synthetic_ground_truth
from deepuplift.data.schema import create_causal_dataset
from deepuplift.data.nuisance import estimate_nuisance
from deepuplift.benchmarks.continuous.datasets import synthetic_dose_response
from deepuplift.benchmarks.continuous.suite import run_continuous_suite
from deepuplift.benchmarks.metrics import evaluate_prediction
from deepuplift.models import build_model, model_info
from deepuplift.models.binary import TLearner


class _PeakRSS:
    def __init__(self, interval: float = 0.01) -> None:
        self.interval = interval
        self.peak = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        try:
            import psutil

            self._process = psutil.Process()
            self.method = "psutil_rss_sampled"
        except Exception:
            self._process = None
            self.method = "ru_maxrss_process_lifetime"

    def _sample(self) -> None:
        while not self._stop.is_set():
            if self._process is not None:
                self.peak = max(self.peak, self._process.memory_info().rss / (1024**2))
            self._stop.wait(self.interval)

    def start(self) -> None:
        if self._process is None:
            self.peak = _ru_maxrss_mb()
            return
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def stop(self) -> float:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._process is not None:
            self.peak = max(self.peak, self._process.memory_info().rss / (1024**2))
            return float(self.peak)
        return float(_ru_maxrss_mb())


def _ru_maxrss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024**2) if sys.platform == "darwin" else value / 1024.0


def _fingerprint(values: Any) -> str:
    array = np.asarray(values, dtype="float64").reshape(-1)
    return hashlib.sha256(np.round(array, 10).tobytes()).hexdigest()[:16]


def _ci95(values: Iterable[float]) -> list[float] | None:
    values = np.asarray(list(values), dtype="float64")
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    mean = float(values.mean())
    if len(values) == 1:
        return [mean, mean]
    half = 1.96 * float(values.std(ddof=1)) / math.sqrt(len(values))
    return [mean - half, mean + half]


def _aggregate_rows(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for model in sorted({row["model"] for row in rows}):
        values = [row[metric] for row in rows if row["model"] == model and row.get("status") == "PASS" and row.get(metric) is not None]
        if not values:
            result[model] = {"n": 0, "mean": None, "std": None, "ci95": None}
            continue
        array = np.asarray(values, dtype="float64")
        result[model] = {"n": int(len(array)), "mean": float(array.mean()), "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0, "ci95": _ci95(array)}
    return result


def _paired_delta(rows: list[dict[str, Any]], metric: str, reference: str) -> dict[str, Any]:
    by_key = {(row["size"], row["seed"], row["model"]): row for row in rows if row.get("status") == "PASS"}
    result = {}
    models = sorted({row["model"] for row in rows if row["model"] != reference})
    for model in models:
        deltas = []
        keys = {(row["size"], row["seed"]) for row in rows if row["model"] == model}
        for key in keys:
            left, right = by_key.get((*key, model)), by_key.get((*key, reference))
            if left and right and left.get(metric) is not None and right.get(metric) is not None:
                deltas.append(float(left[metric]) - float(right[metric]))
        result[model] = {"n": len(deltas), "mean_delta_vs_" + reference: float(np.mean(deltas)) if deltas else None, "ci95": _ci95(deltas)}
    return result


def _binary_dataset(rows: int, seed: int):
    dataset = synthetic_ground_truth(rows=rows, seed=seed)
    return dataset.with_metadata(
        evidence_class="SYNTHETIC_SCALE_STRESS",
        ground_truth_available=True,
        dgp="DeepUplift deterministic binary ground truth; not MRT-derived because raw MRT is unavailable",
        sample_size=rows,
    )


def run_binary_comparison(
    *,
    sizes: Iterable[int] = (5000, 20000),
    seeds: Iterable[int] = (42, 52, 62, 72, 82),
    models: Iterable[str] = ("S-Learner", "T-Learner", "X-Learner", "R-Learner", "DR-Learner", "CausalForestDML"),
    test_size: float = 0.25,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for size in sizes:
        for seed in seeds:
            dataset = _binary_dataset(int(size), int(seed))
            indices = np.arange(len(dataset.to_pandas()))
            train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=int(seed), stratify=dataset.treatment)
            train, test = dataset.subset(train_idx), dataset.subset(test_idx)
            nuisance = estimate_nuisance(train, cross_fit=True, n_splits=3, random_state=int(seed))
            for name in models:
                info = model_info(name)
                if not info["runnable"]:
                    rows.append({"model": name, "size": int(size), "seed": int(seed), "status": "NOT_AVAILABLE", "maturity": info["maturity"], "reason": info["notes"] or str(info["missing_dependencies"])})
                    continue
                sampler = _PeakRSS()
                sampler.start()
                started = time.perf_counter()
                try:
                    model = build_model(name, random_state=int(seed))
                    model.fit(train, nuisance=nuisance if info["capabilities"].get("requires_propensity") else None)
                    prediction = model.predict(test)
                    metrics = evaluate_prediction(prediction, test, true_effect_col="true_effect")
                    ground_truth = metrics["ground_truth"]
                    rows.append({
                        "model": name, "size": int(size), "seed": int(seed), "status": "PASS", "maturity": info["maturity"],
                        "device": "cpu", "backend": info["backend"], "runtime_seconds": time.perf_counter() - started,
                        "peak_rss_mb": sampler.stop(), "memory_measurement": sampler.method,
                        "pehe": ground_truth.get("pehe"), "ate_bias": ground_truth.get("ate_bias"),
                        "qini": metrics["ranking"].get("qini"), "auuc": metrics["ranking"].get("auuc"),
                        "calibration_mae": metrics["calibration"].get("mae"), "policy_net_value": metrics["policy"].get("net_value"),
                        "prediction_fingerprint": _fingerprint(prediction.cate),
                        "split": "stratified row split; deterministic seed",
                    })
                except Exception as exc:
                    rows.append({"model": name, "size": int(size), "seed": int(seed), "status": "FAIL", "maturity": info["maturity"], "runtime_seconds": time.perf_counter() - started, "peak_rss_mb": sampler.stop(), "memory_measurement": sampler.method, "error": f"{type(exc).__name__}: {exc}"})
    metrics = ["pehe", "ate_bias", "qini", "auuc", "calibration_mae", "policy_net_value", "runtime_seconds", "peak_rss_mb"]
    return {"evidence_class": "SYNTHETIC_SCALE_STRESS", "config": {"sizes": list(map(int, sizes)), "seeds": list(map(int, seeds)), "models": list(models), "test_size": test_size}, "rows": rows, "aggregate": {metric: _aggregate_rows(rows, metric) for metric in metrics}, "paired_vs_s_learner": _paired_delta(rows, "pehe", "S-Learner"), "limitations": ["MRT raw data was not available locally; these rows are not Push-derived semi-synthetic evidence.", "Peak RSS is sampled process RSS and includes shared interpreter state.", "Real randomized datasets are not represented in this section."], "status": "PASS" if any(row.get("status") == "PASS" for row in rows) else "BLOCKED"}


def _multi_dataset(rows: int, seed: int):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(rows, 3))
    eligible_1 = rng.random(rows) > 0.10
    eligible_2 = rng.random(rows) > 0.20
    tau_1 = 0.10 + 0.16 * (x[:, 0] > 0) - 0.04 * x[:, 1]
    tau_2 = 0.14 + 0.10 * np.tanh(x[:, 1]) + 0.08 * np.sin(x[:, 2])
    baseline = 0.30 + 0.10 * x[:, 0] + 0.06 * x[:, 1]
    available = np.column_stack([np.ones(rows, dtype=bool), eligible_1, eligible_2])
    probs = np.where(available, np.array([0.50, 0.30, 0.20])[None, :], 0.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    treatment = np.asarray([rng.choice(3, p=row) for row in probs], dtype=int)
    effect = np.choose(treatment, [np.zeros(rows), tau_1, tau_2])
    outcome = baseline + effect + rng.normal(0.0, 0.08, rows)
    frame = pd.DataFrame({"unit_id": [f"multi-{seed}-{i}" for i in range(rows)], "x1": x[:, 0], "x2": x[:, 1], "x3": x[:, 2], "eligible_1": eligible_1.astype(float), "eligible_2": eligible_2.astype(float), "treatment": treatment, "outcome": outcome, "true_tau_1": tau_1, "true_tau_2": tau_2})
    dataset = create_causal_dataset(frame, feature_cols=["x1", "x2", "x3", "eligible_1", "eligible_2"], treatment_col="treatment", outcome_col="outcome", treatment_type=TreatmentType.MULTI_DISCRETE, assignment_type=AssignmentType.RANDOMIZED, id_column="unit_id", metadata={"name": "experiment_a_multi_synthetic", "evidence_class": "SYNTHETIC_SCALE_STRESS", "ground_truth_available": True, "eligible_action_columns": {1: "eligible_1", 2: "eligible_2"}, "dgp": "three-arm known-truth synthetic with per-user eligibility masks; not MRT-derived"})
    return dataset


def _binary_view(dataset, arm: int, indices: np.ndarray):
    frame = dataset.to_pandas().iloc[indices].reset_index(drop=True).copy()
    frame["treatment_binary"] = (frame["treatment"].to_numpy() == arm).astype(int)
    return create_causal_dataset(frame, feature_cols=dataset.feature_cols, treatment_col="treatment_binary", outcome_col="outcome", treatment_type=TreatmentType.BINARY, assignment_type=AssignmentType.RANDOMIZED, id_column="unit_id", metadata={"name": f"one_vs_control_arm_{arm}", "evidence_class": "SYNTHETIC_SCALE_STRESS"})


def _multi_policy_metrics(test_frame: pd.DataFrame, effects: dict[int, np.ndarray], *, mask: np.ndarray | None = None) -> dict[str, Any]:
    n = len(test_frame)
    mask = np.ones((n, 3), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    predicted = np.zeros(n, dtype=int)
    truth = np.zeros(n, dtype=int)
    pred_values = np.zeros((n, 3), dtype=float)
    true_values = np.column_stack([np.zeros(n), test_frame["true_tau_1"].to_numpy(), test_frame["true_tau_2"].to_numpy()])
    for arm in (1, 2):
        pred_values[:, arm] = np.asarray(effects[arm], dtype=float)
    pred_values[~mask] = -np.inf
    true_values[~mask] = -np.inf
    predicted = np.argmax(pred_values, axis=1)
    truth = np.argmax(true_values, axis=1)
    pred_effect = np.where(predicted == 0, 0.0, np.choose(predicted, [np.zeros(n), test_frame["true_tau_1"], test_frame["true_tau_2"]]))
    oracle_effect = np.max(np.where(np.isfinite(true_values), true_values, -np.inf), axis=1)
    invalid = np.any(~mask[np.arange(n), predicted])
    return {"policy_regret": float(np.mean(oracle_effect - pred_effect)), "best_action_accuracy": float(np.mean(predicted == truth)), "eligibility_violations": int(invalid), "oracle_mean_effect": float(np.mean(oracle_effect)), "policy_mean_effect": float(np.mean(pred_effect))}


def run_multi_comparison(*, sizes: Iterable[int] = (5000, 20000), seeds: Iterable[int] = (42, 52, 62, 72, 82)) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for size in sizes:
        for seed in seeds:
            dataset = _multi_dataset(int(size), int(seed))
            indices = np.arange(len(dataset.to_pandas()))
            train_idx, test_idx = train_test_split(indices, test_size=.25, random_state=int(seed), stratify=dataset.treatment)
            train_frame, test_frame = dataset.to_pandas().iloc[train_idx], dataset.to_pandas().iloc[test_idx]
            test_mask = np.column_stack([np.ones(len(test_frame), dtype=bool), test_frame["eligible_1"].to_numpy().astype(bool), test_frame["eligible_2"].to_numpy().astype(bool)])
            # One-vs-control uses the same outer split and has one method-specific fit per arm.
            started, sampler = time.perf_counter(), _PeakRSS()
            sampler.start()
            try:
                arm_effects = {}
                for arm in (1, 2):
                    train_view = _binary_view(dataset, arm, train_idx)
                    model = TLearner(task="regression", random_state=int(seed))
                    model.fit(train_view)
                    test_view = _binary_view(dataset, arm, test_idx)
                    arm_effects[arm] = model.predict(test_view).cate
                policy = _multi_policy_metrics(test_frame, arm_effects, mask=test_mask)
                truth = {arm: test_frame[f"true_tau_{arm}"].to_numpy() for arm in (1, 2)}
                arm_metrics = {str(arm): {"pehe": float(np.sqrt(np.mean((arm_effects[arm][test_mask[:, arm]] - truth[arm][test_mask[:, arm]]) ** 2))), "ate_bias": float(np.mean(arm_effects[arm][test_mask[:, arm]]) - np.mean(truth[arm][test_mask[:, arm]])), "eligible_test_rows": int(test_mask[:, arm].sum())} for arm in (1, 2)}
                rows.append({"model": "One-vs-Control", "size": int(size), "seed": int(seed), "status": "PASS", "runtime_seconds": time.perf_counter() - started, "peak_rss_mb": sampler.stop(), "device": "cpu", "arm_metrics": arm_metrics, **policy})
            except Exception as exc:
                rows.append({"model": "One-vs-Control", "size": int(size), "seed": int(seed), "status": "FAIL", "runtime_seconds": time.perf_counter() - started, "peak_rss_mb": sampler.stop(), "error": f"{type(exc).__name__}: {exc}"})
            started, sampler = time.perf_counter(), _PeakRSS()
            sampler.start()
            try:
                model = build_model("MultiTreatmentOutcome", task="regression", random_state=int(seed))
                model.fit(dataset.subset(train_idx))
                prediction = model.predict(dataset.subset(test_idx))
                effects = {int(arm): np.asarray(values) for arm, values in prediction.treatment_effects.items() if int(arm) in (1, 2)}
                policy = _multi_policy_metrics(test_frame, effects, mask=test_mask)
                arm_metrics = {str(arm): {"pehe": float(np.sqrt(np.mean((effects[arm][test_mask[:, arm]] - test_frame.loc[test_frame.index, f"true_tau_{arm}"].to_numpy()[test_mask[:, arm]]) ** 2))), "ate_bias": float(np.mean(effects[arm][test_mask[:, arm]]) - np.mean(test_frame[f"true_tau_{arm}"].to_numpy()[test_mask[:, arm]])), "eligible_test_rows": int(test_mask[:, arm].sum())} for arm in (1, 2)}
                rows.append({"model": "MultiTreatmentOutcome", "size": int(size), "seed": int(seed), "status": "PASS", "runtime_seconds": time.perf_counter() - started, "peak_rss_mb": sampler.stop(), "device": "cpu", "arm_metrics": arm_metrics, **policy})
            except Exception as exc:
                rows.append({"model": "MultiTreatmentOutcome", "size": int(size), "seed": int(seed), "status": "FAIL", "runtime_seconds": time.perf_counter() - started, "peak_rss_mb": sampler.stop(), "error": f"{type(exc).__name__}: {exc}"})
    return {"evidence_class": "SYNTHETIC_SCALE_STRESS", "config": {"sizes": list(map(int, sizes)), "seeds": list(map(int, seeds)), "models": ["One-vs-Control", "MultiTreatmentOutcome"], "arms": [0, 1, 2], "eligibility_mask": True}, "rows": rows, "limitations": ["The MRT raw context was unavailable, so this is not Push-derived semi-synthetic evidence.", "The unified model is the existing shared API adapter, not MEMENTO/HydraNet/M3TN/MTMT or EFIN.", "Policy metrics use generated truth and an explicit eligibility mask."], "status": "PASS" if any(row.get("status") == "PASS" for row in rows) else "BLOCKED"}


def run_continuous_comparison(*, sizes: Iterable[int] = (5000, 20000), seeds: Iterable[int] = (42, 52, 62, 72, 82), models: Iterable[str] = ("DoseResponseGBM", "DRNet", "VCNet", "GIKS-DRNet", "GIKS-VCNet"), epochs: int = 10, factual_epochs: int = 8, giks_epochs: int = 4, neural_device: str = "cpu") -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for size in sizes:
        for seed in seeds:
            for shape in ("monotone", "saturating", "fatigue"):
                data = synthetic_dose_response(rows=int(size), seed=int(seed), shape=shape)
                data.dataset.metadata.update({"evidence_class": "SYNTHETIC_SCALE_STRESS", "source": "DeepUplift deterministic continuous DGP; not MRT-derived", "push_context_available": False})
                report = run_continuous_suite(data=data, seed=int(seed), model_names=list(models), model_options={"DRNet": {"epochs": epochs, "device": neural_device}, "VCNet": {"epochs": epochs, "device": neural_device}, "GIKS-DRNet": {"factual_epochs": factual_epochs, "giks_epochs": giks_epochs, "model_kwargs": {"device": neural_device}}, "GIKS-VCNet": {"factual_epochs": factual_epochs, "giks_epochs": giks_epochs, "model_kwargs": {"device": neural_device}}})
                for item in report["models"]:
                    compact = dict(item)
                    # A full policy row/curve preview is useful for a one-off
                    # interactive run but makes a 20K x 5-seed artifact huge.
                    # Keep aggregate policy metadata and a single preview row;
                    # all per-model metrics remain intact.
                    if isinstance(compact.get("policy"), dict):
                        compact["policy"] = {"summary": compact["policy"].get("summary")}
                    metadata = compact.get("prediction_metadata")
                    if isinstance(metadata, dict):
                        compact["prediction_metadata"] = {
                            key: metadata.get(key)
                            for key in ("model_name", "maturity", "status", "device", "seed", "observed_dose_min", "observed_dose_max", "dose_support", "giks")
                            if key in metadata
                        }
                        local_support = metadata.get("local_treatment_support")
                        if isinstance(local_support, dict):
                            compact["prediction_metadata"]["local_treatment_support"] = {
                                key: local_support.get(key)
                                for key in ("diagnostic_name", "k", "observed_dose_range", "mean_support_score", "min_support_score", "max_support_score")
                                if key in local_support
                            }
                    compact["curve_preview"] = compact.get("curve_preview", [])[:1]
                    rows.append({"size": int(size), "seed": int(seed), "shape": shape, **compact})
    return {"evidence_class": "SYNTHETIC_SCALE_STRESS", "config": {"sizes": list(map(int, sizes)), "seeds": list(map(int, seeds)), "models": list(models), "epochs": epochs, "factual_epochs": factual_epochs, "giks_epochs": giks_epochs, "neural_device": neural_device, "shapes": ["monotone", "saturating", "fatigue"]}, "rows": rows, "limitations": ["Without local MRT, this uses the existing generic DeepUplift continuous DGP and is not a Push-context semi-synthetic result.", "No real continuous randomized Push result is claimed.", "DoseResponseGBM and the GIKS Gaussian-process stage remain CPU operations; DRNet/VCNet device is explicit per row.", "A mixed CPU/MPS run is not labelled MPS-only."], "status": "PASS" if any(item.get("status") == "PASS" for item in rows) else "BLOCKED"}


def frontier_status() -> list[dict[str, Any]]:
    models = [
        ("TARNet", "A1"), ("CFRNet", "A1"), ("DragonNet", "A1"), ("FlexTENet", "A1"), ("EFIN", "A1"), ("DESCN", "A1"), ("EUEN/EEUEN", "A1"), ("TransTEE", "A1/A3"), ("HyperITE", "A1/A2"),
        ("MEMENTO", "A2"), ("HydraNet", "A2"), ("M3TN", "A2"), ("MTMT", "A2"),
        ("SCIGAN", "A3"), ("CRNet/WDBN", "A3"), ("HNet", "A3"), ("DR2CE", "A3"), ("DRVAE", "A3"),
    ]
    statuses = []
    for name, stage in models:
        if name == "TransTEE":
            maturity, status, reason = "PAPER_ONLY_NOT_IMPLEMENTED", "BLOCKED_DEPENDENCY_OR_DATA", "registry marks continuous integration as OPTIONAL_NOT_VALIDATED; no faithful local adapter"
        else:
            maturity, status, reason = "PAPER_ONLY_NOT_IMPLEMENTED", "BLOCKED_DEPENDENCY_OR_DATA", "no tested EffectPrediction adapter or verified local implementation in this repository"
        statuses.append({"model": name, "stage": stage, "maturity": maturity, "status": status, "reason": reason})
    return statuses


__all__ = ["run_binary_comparison", "run_multi_comparison", "run_continuous_comparison", "frontier_status"]
