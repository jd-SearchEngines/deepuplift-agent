from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable

import pandas as pd

from .artifacts import dataframe_fingerprint, file_sha256


REQUIRED_DATASET_FIELDS = {
    "id",
    "name",
    "kind",
    "path",
    "treatment_col",
    "outcome_col",
    "feature_cols",
    "task",
}


def load_dataset_manifest(path: str | Path = "examples/datasets/manifest.json") -> list[dict[str, Any]]:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    datasets = payload.get("datasets") or []
    if not isinstance(datasets, list):
        raise ValueError(f"Dataset manifest should contain a list under `datasets`: {manifest_path}")
    return [item for item in datasets if isinstance(item, dict)]


def _license_policy(item: dict[str, Any]) -> dict[str, str]:
    source = str(item.get("source_url") or "")
    kind = str(item.get("kind") or "")
    if not source:
        return {
            "license_status": "project_generated",
            "reuse_policy": "safe_for_demo",
            "note": "Synthetic or project-local dataset; keep generation script or provenance note with releases.",
        }
    if "criteo" in source.lower():
        return {
            "license_status": "external_public_dataset",
            "reuse_policy": "sample_only_review_before_redistribution",
            "note": "Criteo uplift data is public for research-style use; keep source citation and avoid claiming production data.",
        }
    if "github.com" in source.lower():
        return {
            "license_status": "external_repository",
            "reuse_policy": "verify_upstream_license_before_redistribution",
            "note": "Repository-sourced data needs upstream license verification before packaging beyond demo use.",
        }
    if "rdatasets" in source.lower() or "doubleml" in source.lower():
        return {
            "license_status": "external_public_reference",
            "reuse_policy": "cite_source_and_keep_local_sample_hash",
            "note": "Public reference dataset; retain citation, hash, and transformation notes.",
        }
    return {
        "license_status": "external_reference",
        "reuse_policy": "cite_source_and_review_terms",
        "note": f"External {kind} dataset; retain source URL and check redistribution terms before publishing.",
    }


def _benchmark_tier(item: dict[str, Any]) -> str:
    kind = str(item.get("kind") or "").lower()
    dataset_id = str(item.get("id") or "").lower()
    if any(token in kind for token in ["counterfactual", "public"]) or dataset_id in {"criteo_visit_10k", "ihdp_npci_1"}:
        return "paper_level_candidate"
    if any(token in kind for token in ["llm", "industrial", "coupon", "ads", "marketplace"]):
        return "production_pilot_candidate"
    if "synthetic" in kind:
        return "smoke"
    return "research"


def _causal_assumption_card(item: dict[str, Any], frame: pd.DataFrame | None) -> dict[str, Any]:
    treatment_col = item.get("treatment_col")
    outcome_col = item.get("outcome_col")
    feature_cols = item.get("feature_cols") or []
    assumptions = {
        "unit": "row-level user/context/request observation",
        "treatment": treatment_col,
        "outcome": outcome_col,
        "identification": ["SUTVA", "unconfoundedness conditional on listed features", "positivity / overlap"],
        "known_risks": [],
    }
    if frame is None or treatment_col not in frame.columns:
        assumptions["known_risks"].append("data file unavailable or treatment column missing")
        return assumptions
    treatment_counts = frame[treatment_col].value_counts(dropna=False)
    if len(treatment_counts) < 2:
        assumptions["known_risks"].append("single treatment arm observed")
    if len(treatment_counts) == 2:
        treatment_rate = float((frame[treatment_col] == treatment_counts.index.max()).mean())
        if treatment_rate < 0.05 or treatment_rate > 0.95:
            assumptions["known_risks"].append("extreme treatment imbalance")
    missing_features = [col for col in feature_cols if col not in frame.columns]
    if missing_features:
        assumptions["known_risks"].append(f"missing feature columns: {missing_features[:5]}")
    if outcome_col not in frame.columns:
        assumptions["known_risks"].append("outcome column missing")
    return assumptions


def _column_profile(frame: pd.DataFrame, columns: Iterable[str]) -> list[dict[str, Any]]:
    rows = []
    for col in columns:
        if col not in frame.columns:
            rows.append({"column": col, "present": False})
            continue
        series = frame[col]
        rows.append(
            {
                "column": col,
                "present": True,
                "dtype": str(series.dtype),
                "missing_rate": float(series.isna().mean()),
                "n_unique": int(series.nunique(dropna=True)),
            }
        )
    return rows


def build_dataset_card(item: dict[str, Any], root: str | Path = ".") -> dict[str, Any]:
    missing_fields = sorted(REQUIRED_DATASET_FIELDS.difference(item))
    root_path = Path(root)
    data_path = root_path / str(item.get("path", ""))
    frame: pd.DataFrame | None = None
    load_error = None
    if data_path.exists() and data_path.is_file():
        try:
            frame = pd.read_csv(data_path)
        except Exception as exc:  # pragma: no cover - defensive path for corrupted local data.
            load_error = str(exc)
    elif item.get("path"):
        load_error = f"file not found: {data_path}"

    feature_cols = list(item.get("feature_cols") or [])
    columns_for_hash = [col for col in [item.get("treatment_col"), item.get("outcome_col"), *feature_cols] if col]
    data_profile: dict[str, Any] = {
        "path": str(item.get("path") or ""),
        "exists": data_path.exists(),
        "load_error": load_error,
    }
    if frame is not None:
        data_profile.update(
            {
                "rows_observed": int(len(frame)),
                "columns_observed": int(len(frame.columns)),
                "declared_rows": item.get("rows"),
                "file_sha256": file_sha256(data_path),
                "fingerprint": dataframe_fingerprint(frame, columns=[col for col in columns_for_hash if col in frame.columns]),
                "column_profile": _column_profile(frame, [item.get("treatment_col"), item.get("outcome_col"), *feature_cols]),
            }
        )

    return {
        "schema_version": 1,
        "dataset_id": item.get("id"),
        "name": item.get("name"),
        "kind": item.get("kind"),
        "benchmark_tier": _benchmark_tier(item),
        "task": item.get("task"),
        "description": item.get("description"),
        "source_url": item.get("source_url"),
        "license": _license_policy(item),
        "schema": {
            "treatment_col": item.get("treatment_col"),
            "outcome_col": item.get("outcome_col"),
            "feature_cols": feature_cols,
            "feature_count": len(feature_cols),
        },
        "recommended_models": item.get("recommended_models") or [],
        "quick_train": item.get("quick_train") or {},
        "causal_assumptions": _causal_assumption_card(item, frame),
        "data_profile": data_profile,
        "validation": {
            "missing_manifest_fields": missing_fields,
            "status": "fail" if missing_fields or load_error else "ok",
        },
    }


def build_dataset_registry(
    manifest_path: str | Path = "examples/datasets/manifest.json",
    root: str | Path = ".",
) -> dict[str, Any]:
    cards = [build_dataset_card(item, root=root) for item in load_dataset_manifest(manifest_path)]
    failures = [
        {
            "dataset_id": card.get("dataset_id"),
            "status": card["validation"]["status"],
            "missing_fields": card["validation"].get("missing_manifest_fields"),
            "load_error": card["data_profile"].get("load_error"),
        }
        for card in cards
        if card["validation"]["status"] != "ok"
    ]
    tier_counts = pd.Series([card["benchmark_tier"] for card in cards]).value_counts().to_dict() if cards else {}
    return {
        "schema_version": 1,
        "manifest": str(manifest_path),
        "dataset_count": len(cards),
        "tier_counts": {str(k): int(v) for k, v in tier_counts.items()},
        "cards": cards,
        "failures": failures,
    }


def dataset_cards_frame(cards: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for card in cards:
        profile = card.get("data_profile") or {}
        license_payload = card.get("license") or {}
        rows.append(
            {
                "dataset_id": card.get("dataset_id"),
                "name": card.get("name"),
                "kind": card.get("kind"),
                "benchmark_tier": card.get("benchmark_tier"),
                "task": card.get("task"),
                "rows_observed": profile.get("rows_observed"),
                "declared_rows": profile.get("declared_rows"),
                "feature_count": (card.get("schema") or {}).get("feature_count"),
                "license_status": license_payload.get("license_status"),
                "reuse_policy": license_payload.get("reuse_policy"),
                "validation_status": (card.get("validation") or {}).get("status"),
                "known_risks": "; ".join((card.get("causal_assumptions") or {}).get("known_risks") or []),
                "source_url": card.get("source_url"),
            }
        )
    return pd.DataFrame(rows)
