from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepuplift.core import UpliftConfig, train_uplift_model
from deepuplift.core.artifacts import save_json
from deepuplift.core.diagnostics import diagnose_uplift_data
from deepuplift.core.evaluator import frontier_metric_summary
from deepuplift.core.model_cards import build_model_card
from deepuplift.core.readiness import decision_readiness
from deepuplift.core.registry import missing_dependencies


def load_dataset(dataset_id: str, manifest_path: Path) -> tuple[dict, pd.DataFrame]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in payload.get("datasets", []):
        if item.get("id") == dataset_id:
            return item, pd.read_csv(item["path"])
    raise ValueError(f"Dataset id not found in {manifest_path}: {dataset_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a no-UI compare smoke and save a manifest JSON.")
    parser.add_argument("--dataset-id", default="synthetic_retail_uplift_5k")
    parser.add_argument("--manifest", default="examples/datasets/manifest.json")
    parser.add_argument("--rows", type=int, default=400)
    parser.add_argument("--models", nargs="+", default=["TLearnerGBM", "DRLearnerGBM"])
    parser.add_argument("--sensitivity-samples", type=int, default=1)
    parser.add_argument("--bootstrap-samples", type=int, default=0)
    args = parser.parse_args()

    dataset, df = load_dataset(args.dataset_id, Path(args.manifest))
    df = df.head(args.rows).copy()
    diagnostics = diagnose_uplift_data(df, dataset["treatment_col"], dataset["outcome_col"], dataset["feature_cols"])

    ranking = []
    runs = []
    for model_name in args.models:
        missing = missing_dependencies(model_name)
        if missing:
            ranking.append({"model": model_name, "error": f"missing dependencies: {', '.join(missing)}"})
            continue
        cfg = UpliftConfig(
            treatment_col=dataset["treatment_col"],
            outcome_col=dataset["outcome_col"],
            feature_cols=dataset["feature_cols"],
            model_name=model_name,
            task=dataset.get("task", "classification"),
            max_rows=args.rows,
            sensitivity_samples=args.sensitivity_samples,
            bootstrap_samples=args.bootstrap_samples,
            run_name=f"no-ui-compare-{model_name.lower()}",
        )
        result = train_uplift_model(config=cfg, data_frame=df)
        metrics = result.eval_metrics
        readiness = decision_readiness(metrics)
        row = {
            "model": model_name,
            "run_id": result.run_id,
            "readiness_score": readiness["score"],
            "readiness_level": readiness["level"],
            "qini": metrics.get("qini_score"),
            "auuc": metrics.get("auuc_score"),
            "calibration_mae": metrics.get("calibration_mae"),
            "sensitivity_verdict": (metrics.get("sensitivity") or {}).get("verdict"),
            **frontier_metric_summary(metrics),
            "predictions": result.artifacts.get("predictions"),
            "run_note": result.artifacts.get("run_note"),
            "readiness": result.artifacts.get("readiness"),
        }
        ranking.append(row)
        runs.append({"run_id": result.run_id, "model": model_name, "run_dir": result.run_dir, "artifacts": result.artifacts})

    ranking = sorted(
        ranking,
        key=lambda row: (row.get("readiness_score") is not None, row.get("readiness_score") or -1, row.get("qini") or -1),
        reverse=True,
    )
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    path = reports_dir / f"no_ui_compare_manifest_{time.strftime('%Y%m%d-%H%M%S')}.json"
    save_json(
        path,
        {
            "dataset": dataset["name"],
            "dataset_id": args.dataset_id,
            "rows": len(df),
            "models": args.models,
            "diagnostic_warnings": diagnostics.get("warnings", []),
            "ranking": ranking,
            "runs": runs,
            "model_cards": [build_model_card(model_name) for model_name in args.models],
        },
    )
    print(path)


if __name__ == "__main__":
    main()
