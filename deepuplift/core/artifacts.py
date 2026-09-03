from __future__ import annotations

import hashlib
import json
import math
import pickle
import platform
import subprocess
import sys
import time
import uuid
from importlib import metadata
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd


def _json_default(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Interval):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _json_clean(value: Any):
    if isinstance(value, dict):
        return {str(key): _json_clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_clean(item) for item in value]
    if isinstance(value, tuple):
        return [_json_clean(item) for item in value]
    if isinstance(value, np.generic):
        return _json_clean(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Interval):
        return str(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def create_run_dir(base_dir: str = "runs", run_name: str | None = None) -> tuple[str, Path]:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    suffix = uuid.uuid4().hex[:8]
    clean_name = f"{run_name}-" if run_name else ""
    run_id = f"{clean_name}{timestamp}-{suffix}"
    run_dir = Path(base_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_id, run_dir


def save_json(path: Path, payload: Dict[str, Any]) -> str:
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_clean(payload), f, ensure_ascii=False, indent=2, default=_json_default, allow_nan=False)
    return str(path)


def save_dataframe(path: Path, df: pd.DataFrame) -> str:
    df.to_csv(path, index=False)
    return str(path)


def save_text(path: Path, content: str) -> str:
    path.write_text(content, encoding="utf-8")
    return str(path)


def save_pickle(path: Path, payload: Any) -> str:
    with path.open("wb") as f:
        pickle.dump(payload, f)
    return str(path)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def save_torch_model(path: Path, model: Any, metadata: Dict[str, Any]) -> str:
    import torch

    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, path)
    return str(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataframe_fingerprint(
    df: pd.DataFrame,
    *,
    columns: list[str] | None = None,
    max_rows_for_hash: int = 20_000,
) -> Dict[str, Any]:
    selected = df[columns].copy() if columns else df.copy()
    sample = selected.head(max_rows_for_hash)
    try:
        row_hashes = pd.util.hash_pandas_object(sample, index=True).values.tobytes()
        content_sha256 = hashlib.sha256(row_hashes).hexdigest()
    except Exception:
        content_sha256 = hashlib.sha256(sample.to_csv(index=True).encode("utf-8", errors="replace")).hexdigest()
    return {
        "rows": int(len(selected)),
        "columns": int(len(selected.columns)),
        "column_names": [str(col) for col in selected.columns],
        "hashed_rows": int(len(sample)),
        "hash_method": "pandas.util.hash_pandas_object/head",
        "content_sha256": content_sha256,
    }


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _git_snapshot(root: Path) -> Dict[str, Any]:
    def run_git(args: list[str]) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=root,
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except Exception:
            return None
        if result.returncode != 0:
            return None
        return result.stdout.strip()

    commit = run_git(["rev-parse", "HEAD"])
    branch = run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    status = run_git(["status", "--short"]) or ""
    status_lines = [line for line in status.splitlines() if line.strip()]
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status_lines),
        "changed_files": len(status_lines),
    }


def build_environment_snapshot(root: str | Path = ".") -> Dict[str, Any]:
    package_names = [
        "pandas",
        "numpy",
        "torch",
        "scikit-learn",
        "scipy",
        "lightgbm",
        "econml",
        "scikit-uplift",
        "xgboost",
        "catboost",
        "causalml",
        "utboost",
        "streamlit",
    ]
    packages = {name: _package_version(name) for name in package_names}
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "git": _git_snapshot(Path(root).resolve()),
    }


def build_evidence_manifest(
    *,
    run_id: str,
    run_dir: Path,
    artifacts: Dict[str, str],
    config: Dict[str, Any],
    model_metadata: Dict[str, Any],
    data_fingerprint: Dict[str, Any],
    metrics: Dict[str, Any],
    readiness: Dict[str, Any],
    environment_path: str | None = None,
) -> Dict[str, Any]:
    files = []
    for name, artifact_path in sorted(artifacts.items()):
        path = Path(artifact_path)
        if not path.exists() or not path.is_file():
            files.append({"artifact": name, "path": str(path), "present": False})
            continue
        files.append(
            {
                "artifact": name,
                "file": path.name,
                "path": str(path),
                "present": True,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    return {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "run_id": run_id,
        "run_dir": str(run_dir),
        "contract": {
            "required_prediction_columns": ["treatment", "outcome", "y0_pred", "y1_pred", "uplift_score"],
            "required_core_artifacts": ["config", "metrics", "readiness", "promotion", "predictions", "run_note"],
            "decision_gate": "decision_readiness",
        },
        "model": {
            "name": model_metadata.get("model_name"),
            "task": model_metadata.get("task"),
            "input_dim": model_metadata.get("input_dim"),
            "feature_count": len(model_metadata.get("feature_cols") or []),
            "encoded_feature_count": len(model_metadata.get("encoded_feature_cols") or []),
        },
        "config": {
            "treatment_col": config.get("treatment_col"),
            "outcome_col": config.get("outcome_col"),
            "task": config.get("task"),
            "test_size": config.get("test_size"),
            "random_state": config.get("random_state"),
            "bootstrap_samples": config.get("bootstrap_samples"),
            "sensitivity_samples": config.get("sensitivity_samples"),
        },
        "data_fingerprint": data_fingerprint,
        "metrics_summary": {
            "qini_score": metrics.get("qini_score"),
            "auuc_score": metrics.get("auuc_score"),
            "calibration_mae": metrics.get("calibration_mae"),
            "policy_best": metrics.get("policy_best"),
        },
        "readiness": {
            "score": readiness.get("score"),
            "level": readiness.get("level"),
            "blockers": readiness.get("blockers") or [],
        },
        "environment": environment_path,
        "files": files,
    }
