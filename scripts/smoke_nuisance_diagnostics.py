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


def _load_dataset(dataset_id: str, manifest_path: Path) -> tuple[dict, pd.DataFrame]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    for dataset in payload.get("datasets", []):
        if dataset.get("id") == dataset_id:
            return dataset, pd.read_csv(dataset["path"])
    raise ValueError(f"Dataset id not found in {manifest_path}: {dataset_id}")


def _validate_diagnostics(path: Path, *, expected_method: str) -> list[str]:
    failures: list[str] = []
    if not path.exists():
        return [f"missing nuisance diagnostics: {path}"]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("method") != expected_method:
        failures.append(f"expected method={expected_method}, got {payload.get('method')}")
    cross_fit = payload.get("cross_fit") or {}
    if not cross_fit.get("enabled") or int(cross_fit.get("folds") or 0) < 2:
        failures.append("cross-fit diagnostics are not enabled")
    if len(cross_fit.get("fold_rows") or []) < 2:
        failures.append("cross-fit fold rows are incomplete")
    propensity = payload.get("propensity") or {}
    if propensity.get("ipw_ess") is None:
        failures.append("propensity ESS is missing")
    if not (propensity.get("clipped") or {}).get("finite_count"):
        failures.append("propensity summary is missing finite rows")
    if not (payload.get("pseudo_outcome") or {}).get("finite_count"):
        failures.append("pseudo-outcome summary is missing finite rows")
    if not payload.get("nuisance_models"):
        failures.append("nuisance model list is missing")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke cross-fitted DR/R nuisance diagnostics artifacts.")
    parser.add_argument("--dataset-id", default="synthetic_retail_uplift_5k")
    parser.add_argument("--manifest", default="examples/datasets/manifest.json")
    parser.add_argument("--rows", type=int, default=180)
    parser.add_argument("--models", nargs="+", default=["DRLearnerGBM", "OrthogonalDMLGBM"])
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--max-iter", type=int, default=12)
    args = parser.parse_args()

    dataset, df = _load_dataset(args.dataset_id, Path(args.manifest))
    df = df.head(args.rows).copy()
    expected_methods = {"DRLearnerGBM": "dr", "OrthogonalDMLGBM": "r", "RLearnerGBM": "r"}

    rows = []
    failures: list[str] = []
    for model_name in args.models:
        cfg = UpliftConfig(
            treatment_col=dataset["treatment_col"],
            outcome_col=dataset["outcome_col"],
            feature_cols=dataset["feature_cols"],
            model_name=model_name,
            task=dataset.get("task", "classification"),
            valid_perc=None,
            max_rows=args.rows,
            run_name=f"nuisance-smoke-{model_name.lower()}",
            model_params={
                "cross_fit": True,
                "nuisance_folds": args.folds,
                "max_iter": args.max_iter,
                "random_state": 42,
            },
        )
        result = train_uplift_model(config=cfg, data_frame=df)
        diagnostics_path = Path(result.artifacts.get("nuisance_diagnostics", ""))
        model_failures = _validate_diagnostics(
            diagnostics_path,
            expected_method=expected_methods.get(model_name, "r"),
        )
        failures.extend(f"{model_name}: {item}" for item in model_failures)
        rows.append(
            {
                "model": model_name,
                "run_id": result.run_id,
                "run_dir": result.run_dir,
                "qini": result.eval_metrics.get("qini_score"),
                "auuc": result.eval_metrics.get("auuc_score"),
                "nuisance_diagnostics": str(diagnostics_path),
                "status": "failed" if model_failures else "ok",
                "failures": model_failures,
            }
        )

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    payload = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "dataset_id": args.dataset_id,
        "rows": int(len(df)),
        "models": args.models,
        "results": rows,
        "failures": failures,
        "status": "failed" if failures else "ok",
    }
    timestamped = reports_dir / f"nuisance_diagnostics_smoke_{time.strftime('%Y%m%d-%H%M%S')}.json"
    latest = reports_dir / "nuisance_diagnostics_smoke_latest.json"
    save_json(timestamped, payload)
    save_json(latest, payload)
    if failures:
        raise SystemExit(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps({"status": "ok", "path": str(latest), "results": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
