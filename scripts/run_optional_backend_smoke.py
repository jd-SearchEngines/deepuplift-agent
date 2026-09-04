#!/usr/bin/env python3
"""Run real fit/predict/metric probes for optional model adapters."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from deepuplift.benchmarks.metrics import evaluate_prediction
from deepuplift.data import synthetic_ground_truth
from deepuplift.models import build_model, model_info


MODELS = {
    "econml": ["CausalForestDML"],
    "causalml": ["CausalMLUpliftTree", "CausalMLUpliftRandomForest"],
    "sklift": ["SkLiftTwoModels"],
}


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def run(backend: str, output: str | Path | None = None) -> dict:
    dataset = synthetic_ground_truth(500, seed=42)
    statuses: dict[str, str] = {}
    results: dict[str, dict] = {}
    for name in MODELS[backend]:
        info = model_info(name)
        if not info["runnable"]:
            statuses[name] = "INTERFACE_ONLY" if not info["missing_dependencies"] and "interface-only" in info["notes"].lower() else "NOT_INSTALLED"
            results[name] = {"status": statuses[name], "reason": info["notes"] or info["missing_dependencies"], "model_info": info}
            continue
        started = time.perf_counter()
        try:
            model = build_model(name, random_state=42)
            model.fit(dataset)
            prediction = model.predict(dataset)
            full_metrics = evaluate_prediction(prediction, dataset, true_effect_col="true_effect")
            metrics = {
                "ranking": {key: full_metrics["ranking"].get(key) for key in ("qini", "auuc", "uplift_at_10", "uplift_at_20")},
                "calibration": {"mae": full_metrics["calibration"].get("mae")},
                "ground_truth": full_metrics["ground_truth"],
            }
            finite = bool(np.isfinite(np.asarray(prediction.recommended_effect, dtype="float64")).all())
            metric_ok = metrics.get("ranking", {}).get("qini") is not None and metrics.get("ground_truth", {}).get("available") is True
            statuses[name] = "PASS" if finite and metric_ok else "FAIL"
            results[name] = {"status": statuses[name], "fit_predict": "PASS" if finite else "FAIL", "metric": "PASS" if metric_ok else "FAIL", "rows": len(prediction.unit_id), "backend": prediction.metadata.get("backend"), "metrics": metrics, "runtime_seconds": time.perf_counter() - started, "model_info": info}
        except Exception as exc:
            statuses[name] = "FAIL"
            results[name] = {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}", "runtime_seconds": time.perf_counter() - started, "model_info": info}
    report = {"backend": backend, "dataset": {"name": "synthetic_ground_truth", "rows": 500, "seed": 42, "source": "DeepUplift deterministic DGP"}, "statuses": statuses, "results": results}
    if output:
        Path(output).write_text(json.dumps(_jsonable(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(_jsonable(report), ensure_ascii=False, indent=2, allow_nan=False))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=sorted(MODELS), required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    report = run(args.backend, args.output)
    return 0 if all(value in {"PASS", "NOT_INSTALLED", "INTERFACE_ONLY"} for value in report["statuses"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
