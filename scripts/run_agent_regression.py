from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepuplift.core.diagnostics import diagnose_uplift_data
from deepuplift.core.knowledge import knowledge_model_recommendations, matched_knowledge
from deepuplift.core.registry import available_models
from deepuplift.core.schema import UpliftConfig
from deepuplift.core.trainer import train_uplift_model


MANIFEST = ROOT / "examples" / "datasets" / "manifest.json"
REPORT_DIR = ROOT / "reports"


DEFAULT_CASES = [
    {"dataset_id": "synthetic_retail_uplift_5k", "model": "TLearnerGBM", "rows": 800, "task": "classification"},
    {"dataset_id": "synthetic_retail_uplift_5k", "model": "TransformedOutcomeGBM", "rows": 800, "task": "classification"},
    {"dataset_id": "doubleml_coupon_uplift", "model": "DRLearnerGBM", "rows": 800, "task": "classification"},
    {"dataset_id": "doubleml_coupon_uplift", "model": "DomainAdaptationLearner", "rows": 800, "task": "classification"},
    {"dataset_id": "aer_smoke_ban", "model": "RLearnerGBM", "rows": 800, "task": "classification"},
    {"dataset_id": "ihdp_npci_1", "model": "CausalForest", "rows": 747, "task": "regression"},
]


def _load_manifest() -> Dict[str, Any]:
    if not MANIFEST.exists():
        raise FileNotFoundError(f"Missing dataset manifest: {MANIFEST}")
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _dataset_by_id(manifest: Dict[str, Any], dataset_id: str) -> Dict[str, Any]:
    for item in manifest.get("datasets", []):
        if item.get("id") == dataset_id:
            return item
    raise KeyError(f"Dataset not found in manifest: {dataset_id}")


def run_case(case: Dict[str, Any], bootstrap_samples: int = 0, sensitivity_samples: int = 0) -> Dict[str, Any]:
    manifest = _load_manifest()
    dataset = _dataset_by_id(manifest, case["dataset_id"])
    df = pd.read_csv(ROOT / dataset["path"]).head(int(case.get("rows") or dataset.get("rows") or 1000))
    task = case.get("task") or dataset["task"]
    model_name = case["model"]

    diagnostics = diagnose_uplift_data(df, dataset["treatment_col"], dataset["outcome_col"], dataset["feature_cols"])
    knowledge_matches = matched_knowledge(diagnostics, task)
    knowledge_models = knowledge_model_recommendations(diagnostics, available_models(task))

    started = time.time()
    config = UpliftConfig(
        treatment_col=dataset["treatment_col"],
        outcome_col=dataset["outcome_col"],
        feature_cols=dataset["feature_cols"],
        model_name=model_name,
        task=task,
        valid_perc=0.0,
        test_size=0.3,
        bootstrap_samples=bootstrap_samples,
        sensitivity_samples=sensitivity_samples,
        model_params={"max_iter": 12, "n_estimators": 40, "max_depth": 4},
        run_name=f"agent-regression-{dataset['id']}-{model_name.lower()}",
    )
    result = train_uplift_model(data_frame=df, config=config)
    elapsed = time.time() - started

    return {
        "dataset_id": dataset["id"],
        "model": model_name,
        "task": task,
        "rows": int(len(df)),
        "diagnostic_warnings": diagnostics.get("warnings", []),
        "knowledge_rule_ids": [item["id"] for item in knowledge_matches],
        "knowledge_models": knowledge_models,
        "run_id": result.run_id,
        "qini": result.eval_metrics.get("qini_score"),
        "auuc": result.eval_metrics.get("auuc_score"),
        "calibration_mae": result.eval_metrics.get("calibration_mae"),
        "policy_best": result.eval_metrics.get("policy_best"),
        "bootstrap": result.eval_metrics.get("bootstrap"),
        "sensitivity": result.eval_metrics.get("sensitivity"),
        "overlap_trim": result.eval_metrics.get("overlap_trim"),
        "run_note": result.artifacts.get("run_note"),
        "elapsed_seconds": round(elapsed, 3),
        "status": "ok",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a lightweight DeepUplift Agent regression suite.")
    parser.add_argument("--bootstrap-samples", type=int, default=0)
    parser.add_argument("--sensitivity-samples", type=int, default=0)
    parser.add_argument("--case", action="append", help="Optional dataset_id:model:rows case override.")
    args = parser.parse_args()

    cases = DEFAULT_CASES
    if args.case:
        cases = []
        for raw in args.case:
            dataset_id, model, rows = raw.split(":")
            cases.append({"dataset_id": dataset_id, "model": model, "rows": int(rows)})

    results: List[Dict[str, Any]] = []
    for case in cases:
        try:
            row = run_case(case, bootstrap_samples=args.bootstrap_samples, sensitivity_samples=args.sensitivity_samples)
        except Exception as exc:
            row = {**case, "status": "error", "error": str(exc)}
        results.append(row)
        print(json.dumps(row, ensure_ascii=False))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    report_path = REPORT_DIR / f"agent_regression_{timestamp}.json"
    report_path.write_text(json.dumps({"results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {report_path}")

    failed = [row for row in results if row.get("status") != "ok"]
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
