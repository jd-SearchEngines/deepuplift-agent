from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def discover_evidence_manifests(root: str | Path = ".") -> list[Path]:
    root_path = Path(root)
    search_roots = [root_path / "runs", root_path / "reports"]
    paths: list[Path] = []
    for search_root in search_roots:
        if search_root.exists():
            paths.extend(search_root.rglob("evidence_manifest.json"))
    return sorted(set(paths), key=lambda path: path.stat().st_mtime, reverse=True)


def _artifact_path(manifest: dict[str, Any], artifact: str) -> str | None:
    for row in manifest.get("files") or []:
        if row.get("artifact") == artifact and row.get("present"):
            return row.get("path")
    return None


def _promotion_level(manifest: dict[str, Any]) -> str | None:
    path = _artifact_path(manifest, "promotion")
    if not path:
        return None
    payload = _read_json(Path(path))
    return payload.get("level") or payload.get("promotion_level")


def evidence_record(path: Path, root: str | Path = ".") -> dict[str, Any]:
    manifest = _read_json(path)
    run_dir = Path(manifest.get("run_dir") or path.parent)
    root_path = Path(root).resolve()
    try:
        rel_manifest = str(path.resolve().relative_to(root_path))
    except ValueError:
        rel_manifest = str(path)
    metrics = manifest.get("metrics_summary") or {}
    readiness = manifest.get("readiness") or {}
    model = manifest.get("model") or {}
    data_fp = manifest.get("data_fingerprint") or {}
    files = manifest.get("files") or []
    missing_files = [row.get("artifact") for row in files if not row.get("present")]
    return {
        "manifest_path": rel_manifest,
        "manifest_mtime": path.stat().st_mtime,
        "run_id": manifest.get("run_id"),
        "run_dir": str(run_dir),
        "model": model.get("name"),
        "task": model.get("task"),
        "feature_count": model.get("feature_count"),
        "encoded_feature_count": model.get("encoded_feature_count"),
        "qini_score": metrics.get("qini_score"),
        "auuc_score": metrics.get("auuc_score"),
        "calibration_mae": metrics.get("calibration_mae"),
        "readiness_score": readiness.get("score"),
        "readiness_level": readiness.get("level"),
        "promotion_level": _promotion_level(manifest),
        "data_rows": data_fp.get("rows"),
        "data_columns": data_fp.get("columns"),
        "data_hash": data_fp.get("content_sha256"),
        "artifact_count": len([row for row in files if row.get("present")]),
        "missing_artifacts": "; ".join(str(item) for item in missing_files if item),
        "environment": manifest.get("environment"),
    }


def build_evidence_store(root: str | Path = ".", limit: int | None = None) -> dict[str, Any]:
    paths = discover_evidence_manifests(root)
    if limit is not None:
        paths = paths[: int(limit)]
    records = [evidence_record(path, root=root) for path in paths]
    frame = pd.DataFrame(records)
    summary: dict[str, Any] = {
        "manifest_count": len(records),
        "models": int(frame["model"].nunique()) if not frame.empty and "model" in frame else 0,
        "ready_levels": frame["readiness_level"].value_counts(dropna=False).to_dict() if not frame.empty and "readiness_level" in frame else {},
        "promotion_levels": frame["promotion_level"].value_counts(dropna=False).to_dict() if not frame.empty and "promotion_level" in frame else {},
    }
    return {"schema_version": 1, "summary": summary, "records": records}


def evidence_store_frame(payload: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(payload.get("records") or [])


def diff_evidence_records(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    metrics = ["qini_score", "auuc_score", "calibration_mae", "readiness_score"]
    diff: dict[str, Any] = {
        "previous_run_id": previous.get("run_id"),
        "current_run_id": current.get("run_id"),
        "previous_model": previous.get("model"),
        "current_model": current.get("model"),
        "same_data_hash": previous.get("data_hash") == current.get("data_hash"),
        "metric_deltas": {},
        "readiness_transition": f"{previous.get('readiness_level')} -> {current.get('readiness_level')}",
        "promotion_transition": f"{previous.get('promotion_level')} -> {current.get('promotion_level')}",
    }
    for metric in metrics:
        prev_value = previous.get(metric)
        curr_value = current.get(metric)
        if prev_value is None or curr_value is None:
            delta = None
        else:
            delta = float(curr_value) - float(prev_value)
        diff["metric_deltas"][metric] = {
            "previous": prev_value,
            "current": curr_value,
            "delta": delta,
        }
    return diff


def latest_run_diff(payload: dict[str, Any]) -> dict[str, Any]:
    records = payload.get("records") or []
    if len(records) < 2:
        return {"status": "not_enough_runs", "diff": None}
    return {"status": "ok", "diff": diff_evidence_records(records[1], records[0])}
