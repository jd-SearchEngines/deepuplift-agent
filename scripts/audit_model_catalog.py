from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepuplift.core.registry import MODEL_REGISTRY, missing_dependencies
from deepuplift.core.model_cards import build_model_card, model_parameter_presets, recommended_model_preset


ALLOWED_NON_REGISTRY_DESCRIPTIONS = {
    "MultiTLearnerGBM",
    "MultiTLearnerRF",
    "MultiDRLearnerGBM",
    "MultiDRLearnerRF",
    "DoseResponseGBM",
    "DoseResponseRF",
}


def _literal_dict(node: ast.AST) -> dict[str, str]:
    value = ast.literal_eval(node)
    if not isinstance(value, dict):
        raise TypeError("Expected a dict literal.")
    return {str(key): str(item) for key, item in value.items()}


def extract_model_descriptions(app_path: Path) -> dict[str, str]:
    """Extract MODEL_DESCRIPTIONS without importing Streamlit app side effects."""
    tree = ast.parse(app_path.read_text(encoding="utf-8"))
    descriptions: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "MODEL_DESCRIPTIONS":
                    descriptions.update(_literal_dict(node.value))
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
            if (
                isinstance(call.func, ast.Attribute)
                and call.func.attr == "update"
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "MODEL_DESCRIPTIONS"
                and call.args
            ):
                descriptions.update(_literal_dict(call.args[0]))
    return descriptions


def model_source_and_family(model: str) -> tuple[str, str]:
    if model.startswith("EconML"):
        return "EconML", "official_cate"
    if model.startswith("CausalML"):
        return "CausalML", "official_uplift"
    if model.startswith("SkLift"):
        return "scikit-uplift", "official_uplift"
    if model.startswith("UTBoost"):
        return "UTBoost", "industrial_uplift_gbdt"
    if "LightGBM" in model:
        return "LightGBM", "industrial_boosting"
    if "XGBoost" in model or model.endswith("XGB"):
        return "XGBoost", "industrial_boosting"
    if "CatBoost" in model:
        return "CatBoost", "industrial_boosting"
    if model in {"TarNet", "CFRNet", "DragonNet", "DragonDeepFM", "EFIN", "DESCN", "ESX", "EUEN", "EEUEN"}:
        return "DeepUplift", "neural"
    if "PAV" in model or "GGBM" in model:
        return "DeepUplift", "calibrated_ranking"
    if "Forest" in model or model == "CausalForest":
        return "DeepUplift", "forest_style"
    return "DeepUplift", "meta_learner"


def model_readiness(model: str, missing: list[str]) -> tuple[str, str]:
    if not missing:
        return "ready", "current environment can train it"
    if any("DEEPUPLIFT_ENABLE_XGBOOST" in item or "experimental" in item for item in missing):
        return "guarded", "enable with DEEPUPLIFT_ENABLE_XGBOOST=1 after local runtime validation"
    if any("python>=3.11" in item for item in missing):
        return "needs python 3.11", "run scripts/setup_causalml_py311.sh"
    return "needs dependency", "install missing optional dependencies"


def build_catalog_rows(descriptions: dict[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, spec in sorted(MODEL_REGISTRY.items()):
        missing = missing_dependencies(name)
        status, setup_hint = model_readiness(name, missing)
        source, family = model_source_and_family(name)
        card = build_model_card(name, description=descriptions.get(name, ""))
        presets = model_parameter_presets(name)
        preset_serializable = True
        try:
            json.dumps(presets, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError):
            preset_serializable = False
        rows.append(
            {
                "model": name,
                "status": status,
                "source": source,
                "family": family,
                "stage": card.get("stage", ""),
                "tasks": ",".join(spec.supported_tasks),
                "dependencies": ",".join(spec.dependencies),
                "missing": ",".join(missing),
                "setup_hint": setup_hint,
                "description_present": name in descriptions and bool(descriptions.get(name, "").strip()),
                "model_card_present": bool(card.get("best_for") and card.get("assumptions") and card.get("risks")),
                "preset_count": len(presets),
                "recommended_preset": recommended_model_preset(name)[0],
                "model_preset_present": bool(presets) and preset_serializable,
                "description": descriptions.get(name, ""),
                "decision_use": card.get("decision_use", ""),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def source_readiness_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        source = row["source"]
        item = grouped.setdefault(
            source,
            {
                "source": source,
                "registered": 0,
                "ready": 0,
                "guarded": 0,
                "needs_dependency": 0,
                "needs_python_3_11": 0,
            },
        )
        item["registered"] += 1
        status = row["status"]
        if status == "ready":
            item["ready"] += 1
        elif status == "guarded":
            item["guarded"] += 1
        elif status == "needs python 3.11":
            item["needs_python_3_11"] += 1
        else:
            item["needs_dependency"] += 1
    return sorted(grouped.values(), key=lambda item: item["source"])


def family_readiness_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        family = row["family"]
        item = grouped.setdefault(
            family,
            {
                "family": family,
                "registered": 0,
                "ready": 0,
                "guarded": 0,
                "needs_dependency": 0,
                "needs_python_3_11": 0,
                "sources": set(),
            },
        )
        item["registered"] += 1
        item["sources"].add(row["source"])
        status = row["status"]
        if status == "ready":
            item["ready"] += 1
        elif status == "guarded":
            item["guarded"] += 1
        elif status == "needs python 3.11":
            item["needs_python_3_11"] += 1
        else:
            item["needs_dependency"] += 1
    output = []
    for item in grouped.values():
        output.append({**item, "sources": ",".join(sorted(item["sources"]))})
    return sorted(output, key=lambda item: item["family"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit DeepUplift model registry, descriptions, and dependency status.")
    parser.add_argument("--app", default="app.py")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--min-registered", type=int, default=50)
    parser.add_argument("--min-ready", type=int, default=1)
    parser.add_argument("--min-ready-source", action="append", default=[], help="Require a source-specific ready count, e.g. LightGBM=8.")
    parser.add_argument("--strict", action="store_true", help="Fail when any registered model lacks a UI description.")
    args = parser.parse_args()

    app_path = Path(args.app)
    descriptions = extract_model_descriptions(app_path)
    rows = build_catalog_rows(descriptions)
    source_rows = source_readiness_rows(rows)
    family_rows = family_readiness_rows(rows)
    registry_models = set(MODEL_REGISTRY)
    description_models = set(descriptions)
    missing_descriptions = sorted(name for name in registry_models if not descriptions.get(name, "").strip())
    stale_descriptions = sorted(description_models - registry_models - ALLOWED_NON_REGISTRY_DESCRIPTIONS)
    missing_model_cards = sorted(row["model"] for row in rows if not row.get("model_card_present"))
    missing_model_presets = sorted(row["model"] for row in rows if not row.get("model_preset_present"))
    status_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    ready_source_counts: dict[str, int] = {}
    ready_family_counts: dict[str, int] = {}
    for row in rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
        source_counts[row["source"]] = source_counts.get(row["source"], 0) + 1
        if row["status"] == "ready":
            ready_source_counts[row["source"]] = ready_source_counts.get(row["source"], 0) + 1
            ready_family_counts[row["family"]] = ready_family_counts.get(row["family"], 0) + 1

    audit = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "python": sys.version.split()[0],
        "xgboost_enabled": os.environ.get("DEEPUPLIFT_ENABLE_XGBOOST") == "1",
        "registered_models": len(rows),
        "ready_models": status_counts.get("ready", 0),
        "status_counts": status_counts,
        "source_counts": source_counts,
        "ready_source_counts": ready_source_counts,
        "ready_family_counts": ready_family_counts,
        "missing_descriptions": missing_descriptions,
        "missing_model_cards": missing_model_cards,
        "missing_model_presets": missing_model_presets,
        "stale_descriptions": stale_descriptions,
        "catalog": rows,
        "source_readiness": source_rows,
        "family_readiness": family_rows,
    }

    failures = []
    if len(rows) < args.min_registered:
        failures.append(f"registered model count {len(rows)} < {args.min_registered}")
    if status_counts.get("ready", 0) < args.min_ready:
        failures.append(f"ready model count {status_counts.get('ready', 0)} < {args.min_ready}")
    for item in args.min_ready_source:
        if "=" not in item:
            failures.append(f"invalid --min-ready-source value {item!r}; expected Source=N")
            continue
        source, minimum_text = item.split("=", 1)
        try:
            minimum = int(minimum_text)
        except ValueError:
            failures.append(f"invalid ready threshold {item!r}; expected integer N")
            continue
        actual = ready_source_counts.get(source, 0)
        if actual < minimum:
            failures.append(f"ready model count for {source} {actual} < {minimum}")
    if args.strict and missing_descriptions:
        failures.append(f"{len(missing_descriptions)} registered models lack descriptions")
    if args.strict and missing_model_cards:
        failures.append(f"{len(missing_model_cards)} registered models lack generated model cards")
    if args.strict and missing_model_presets:
        failures.append(f"{len(missing_model_presets)} registered models lack generated parameter presets")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    json_path = output_dir / f"model_catalog_audit_{stamp}.json"
    csv_path = output_dir / f"model_catalog_snapshot_{stamp}.csv"
    source_csv_path = output_dir / f"model_source_readiness_{stamp}.csv"
    family_csv_path = output_dir / f"model_family_readiness_{stamp}.csv"
    json_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    write_csv(csv_path, rows)
    write_csv(source_csv_path, source_rows)
    write_csv(family_csv_path, family_rows)

    print(
        json.dumps(
            {
                "json": str(json_path),
                "csv": str(csv_path),
                "source_csv": str(source_csv_path),
                "family_csv": str(family_csv_path),
                "registered_models": len(rows),
                "ready_models": status_counts.get("ready", 0),
                "ready_source_counts": ready_source_counts,
                "missing_descriptions": len(missing_descriptions),
                "missing_model_cards": len(missing_model_cards),
                "missing_model_presets": len(missing_model_presets),
                "stale_descriptions": len(stale_descriptions),
                "failures": failures,
            },
            ensure_ascii=False,
        )
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
