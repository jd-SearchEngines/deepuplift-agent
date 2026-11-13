"""Run the real-data Experiment A bias diagnosis lane on NHEFS.

The causallib loader retrieves NHEFS into its local cache.  Only aggregate,
machine-readable diagnostics and model metadata are written to the repository;
the raw table is never copied here.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from causallib.datasets import load_nhefs

from deepuplift.benchmarks.bias_repair import (
    run_binary_bias_repair,
    run_continuous_gps_repair,
)
from deepuplift.contracts import AssignmentType, TreatmentType
from deepuplift.data.schema import create_causal_dataset
from deepuplift.models import build_model, model_info

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "benchmarks"


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _nhefs_binary():
    bunch = load_nhefs(raw=False)
    frame = bunch.X.copy()
    frame["qsmk"] = bunch.a.to_numpy()
    frame["wt82_71"] = bunch.y.to_numpy()
    return create_causal_dataset(
        frame,
        feature_cols=list(bunch.X.columns),
        treatment_col="qsmk",
        outcome_col="wt82_71",
        treatment_type=TreatmentType.BINARY,
        assignment_type=AssignmentType.OBSERVATIONAL,
        metadata={"dataset_name": "NHEFS", "source": "causallib.datasets.load_nhefs(raw=False)", "target_trial": "1971 baseline; qsmk between 1971 and 1982; wt82_71 weight change"},
    )


def _nhefs_continuous():
    frame, _ = load_nhefs(raw=True)
    # Baseline-only covariates.  Follow-up outcomes, treatment-era variables,
    # and qsmk are deliberately excluded to avoid post-treatment leakage.
    excluded = {
        "seqn", "qsmk", "death", "yrdth", "modth", "dadth", "wt82", "wt82_71",
        "smkintensity82_71", "price82", "tax82", "price71_82", "tax71_82",
        "hightax82", "price71", "tax71", "smokeintensity",
    }
    features = [column for column in frame.columns if column not in excluded]
    data = frame[features + ["smokeintensity", "wt82_71"]].copy()
    data["smokeintensity"] = pd.to_numeric(data["smokeintensity"], errors="coerce")
    data["wt82_71"] = pd.to_numeric(data["wt82_71"], errors="coerce")
    data = data.dropna(subset=["smokeintensity", "wt82_71"]).reset_index(drop=True)
    return create_causal_dataset(
        data,
        feature_cols=features,
        treatment_col="smokeintensity",
        outcome_col="wt82_71",
        treatment_type=TreatmentType.CONTINUOUS,
        assignment_type=AssignmentType.OBSERVATIONAL,
        metadata={"dataset_name": "NHEFS", "source": "causallib.datasets.load_nhefs(raw=True)", "target_trial": "1971 baseline; continuous cigarettes/day; wt82_71 weight change"},
    )


def _run_real_continuous_models(dataset, seed: int, epochs: int) -> list[dict]:
    frame = dataset.to_pandas().reset_index(drop=True)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(frame))
    split = max(4, int(len(order) * .8))
    train, test = dataset.subset(order[:split]), dataset.subset(order[split:])
    try:
        import torch
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    except (ImportError, AttributeError, RuntimeError):
        device = "cpu"
    results = []
    for name in ("DoseResponseGBM", "DRNet", "VCNet"):
        started = time.perf_counter()
        options = {"grid_size": 11}
        if name == "DoseResponseGBM":
            options["task"] = "regression"
        else:
            options.update({"epochs": epochs, "hidden_dim": 16, "batch_size": 128, "device": device, "learning_rate": .005})
        try:
            model = build_model(name, random_state=seed, **options)
            model.fit(train)
            prediction = model.predict(test)
            doses = test.to_pandas()[dataset.treatment_col].to_numpy(dtype=float)
            outcomes = test.to_pandas()[dataset.outcome_col].to_numpy(dtype=float)
            grid = np.asarray(prediction.dose_grid, dtype=float)
            nearest = np.argmin(np.abs(doses[:, None] - grid[None, :]), axis=1)
            curves = np.column_stack([prediction.dose_outcome_predictions[float(dose)] for dose in grid])
            factual = curves[np.arange(len(doses)), nearest]
            results.append({
                "model": name, "status": "PASS", "device": prediction.metadata.get("device", "cpu"),
                "train_seconds": time.perf_counter() - started,
                "factual_holdout_rmse_nearest_grid": float(np.sqrt(np.mean((factual - outcomes) ** 2))),
                "observed_train_dose_range": [float(train.to_pandas()[dataset.treatment_col].min()), float(train.to_pandas()[dataset.treatment_col].max())],
                "test_dose_range": [float(doses.min()), float(doses.max())],
                "curve_grid": grid.tolist(), "architecture": model_info(name),
                "causal_metrics": "NOT_AVAILABLE: NHEFS has no individual-level dose-response ground truth",
            })
        except (ImportError, RuntimeError, ValueError, OSError) as exc:
            results.append({"model": name, "status": "FAIL", "device": device, "train_seconds": time.perf_counter() - started, "error_type": type(exc).__name__, "error": str(exc)[:1000], "causal_metrics": "NOT_AVAILABLE"})
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-reps", type=int, default=250)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--skip-models", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    binary = _nhefs_binary()
    continuous = _nhefs_continuous()
    binary_result = run_binary_bias_repair(binary, seed=args.seed, bootstrap_reps=args.bootstrap_reps)
    continuous_result = run_continuous_gps_repair(continuous, seed=args.seed)
    continuous_models = [] if args.skip_models else _run_real_continuous_models(continuous, args.seed, args.epochs)
    payload = {
        "experiment": "Experiment A bias diagnosis and repair",
        "run_scope": "Local Mac; real NHEFS only; auxiliary synthetic and unavailable public datasets are not mixed into these estimates",
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "retrieval": {"loader": "causallib.datasets.load_nhefs", "raw_data_committed": False, "cache_policy": "causallib local cache only", "loader_package": "causallib 0.10.0"},
        "binary": binary_result, "continuous": continuous_result, "continuous_models": continuous_models,
        "elapsed_seconds": time.perf_counter() - started,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "bias_diagnostics.json"
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(path), "binary_rows": binary_result["sample_size"], "continuous_rows": continuous_result["sample_size"], "models": continuous_models}, indent=2, default=str))


if __name__ == "__main__":
    main()
