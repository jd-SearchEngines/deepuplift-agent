from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from deepuplift.data import diagnose_dataset, estimate_nuisance
from deepuplift.models import build_model, model_info

from .metrics import evaluate_prediction


def dataset_fingerprint(dataset) -> str:
    raw = dataset.to_pandas().to_csv(index=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _jsonable(value):
    if isinstance(value, dict): return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)): return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None: return value
    return str(value)


def run_benchmark(dataset, *, models: Iterable[str] | None = None, output_dir: str | Path | None = None, seed: int = 42, test_size: float = .3, nuisance_config: dict[str, Any] | None = None, policy_config: dict[str, Any] | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    names = list(models or ["S-Learner", "T-Learner", "X-Learner", "DR-Learner"])
    frame = dataset.to_pandas().reset_index(drop=True)
    indices = np.arange(len(frame))
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=seed, stratify=frame[dataset.treatment_col])
    train, test = dataset.subset(train_idx), dataset.subset(test_idx)
    nuisance = None
    config = nuisance_config or {}
    if dataset.assignment_type.value == "observational":
        nuisance = estimate_nuisance(train, random_state=seed, **config)
    rows = []
    predictions = {}
    for name in names:
        info = model_info(name)
        if not info["runnable"]:
            rows.append({"model": name, "status": "NOT_RUN", "reason": info["notes"] or f"missing optional dependency: {info['missing_dependencies']}"})
            continue
        begin = time.perf_counter()
        try:
            model = build_model(name, random_state=seed)
            model.fit(train, nuisance=nuisance) if nuisance is not None else model.fit(train)
            prediction = model.predict(test)
            metrics = evaluate_prediction(prediction, test, true_effect_col=dataset.metadata.get("true_effect_col"))
            rows.append({"model": name, "status": "PASS", "runtime_seconds": time.perf_counter() - begin, "metrics": metrics, "model_info": info})
            predictions[name] = prediction
        except Exception as exc:
            rows.append({"model": name, "status": "FAIL", "error": str(exc), "model_info": info})
    passed = [r for r in rows if r.get("status") == "PASS"]
    def score(row):
        rank = row["metrics"]["ranking"].get("qini") or -np.inf
        cal = row["metrics"]["calibration"].get("mae")
        return float(rank) - (float(cal) if cal is not None else 0.0)
    best_ranking = max(passed, key=lambda r: r["metrics"]["ranking"].get("qini") if r["metrics"]["ranking"].get("qini") is not None else -np.inf)["model"] if passed else None
    best_calibration = min(passed, key=lambda r: r["metrics"]["calibration"].get("mae") if r["metrics"]["calibration"].get("mae") is not None else np.inf)["model"] if passed else None
    recommended = max(passed, key=score)["model"] if passed else None
    manifest = {"run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{seed}", "timestamp": datetime.now(timezone.utc).isoformat(), "git_sha": _git_sha(), "dataset_name": dataset.metadata.get("name", "unnamed"), "dataset_source": dataset.metadata.get("source", "unknown"), "dataset_hash": dataset_fingerprint(dataset), "sample_size": len(dataset.to_pandas()), "random_seed": seed, "split_seed": seed, "models": names, "nuisance_config": config, "policy_config": policy_config or {}, "python_version": sys.version, "dependencies": {"numpy": np.__version__, "pandas": pd.__version__}, "runtime_seconds": time.perf_counter() - started, "warnings": ["Offline benchmark evidence is not online lift or production validation."]}
    report = {"manifest": manifest, "dataset_metadata": _jsonable(dataset.metadata), "diagnostics": diagnose_dataset(dataset).to_dict(), "results": rows, "selection": {"BEST_RANKING": best_ranking, "BEST_CALIBRATION": best_calibration, "BEST_POLICY_VALUE": recommended, "FASTEST": min(passed, key=lambda r: r.get("runtime_seconds", np.inf))["model"] if passed else None, "RECOMMENDED": recommended, "reason": "Highest simple rank-minus-calibration score among passed models; inspect all four metric dimensions."}, "readiness": "PASS" if passed else "FAIL"}
    if output_dir is not None:
        _write_evidence(Path(output_dir) / manifest["run_id"], manifest, dataset, nuisance, report)
    return report


def _git_sha() -> str | None:
    import subprocess
    try: return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception: return None


def _write_evidence(directory: Path, manifest: dict[str, Any], dataset, nuisance, report: dict[str, Any]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    payloads = {"config.json": {"seed": manifest["random_seed"], "models": manifest["models"]}, "dataset_manifest.json": _jsonable({**dataset.metadata, "name": manifest["dataset_name"], "fingerprint": manifest["dataset_hash"], "rows": manifest["sample_size"]}), "diagnostics.json": report["diagnostics"], "nuisance.json": nuisance.to_dict() if nuisance is not None else {"status": "not_requested"}, "model.json": {"models": report["results"]}, "metrics.json": report["results"], "policy.json": {"offline_only": True}, "environment.json": {"python": platform.python_version(), "platform": platform.platform(), "git_sha": manifest["git_sha"]}}
    hashes = {}
    for name, payload in payloads.items():
        path = directory / name; path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    evidence = {"run_id": manifest["run_id"], "files": hashes, "claims": ["Ground truth metrics are emitted only when dataset supplies true_effect.", "Observational results depend on unconfoundedness/ignorability and positivity assumptions.", "Policy value is offline evidence only."]}
    (directory / "evidence_manifest.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    return directory


__all__ = ["run_benchmark", "dataset_fingerprint"]
