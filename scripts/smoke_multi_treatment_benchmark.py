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

from deepuplift.core import UpliftConfig
from deepuplift.core.artifacts import save_json
from deepuplift.core.multitreatment import train_multi_treatment_model


MODEL_FAMILIES = {
    "MultiTLearnerGBM": "gbm",
    "MultiDRLearnerGBM": "dr_gbm",
    "MultiTLearnerRF": "forest",
    "MultiDRLearnerRF": "dr_forest",
}


def _load_dataset(dataset_id: str, manifest_path: Path) -> tuple[dict, pd.DataFrame]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    for dataset in payload.get("datasets", []):
        if dataset.get("id") == dataset_id:
            return dataset, pd.read_csv(dataset["path"])
    raise ValueError(f"Dataset id not found in {manifest_path}: {dataset_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small multi-treatment benchmark leaderboard.")
    parser.add_argument("--dataset-id", default="synthetic_retail_multi_treatment_6k")
    parser.add_argument("--manifest", default="examples/datasets/manifest.json")
    parser.add_argument("--rows", type=int, default=700)
    parser.add_argument("--models", nargs="+", default=["MultiTLearnerGBM", "MultiDRLearnerGBM"])
    parser.add_argument("--bootstrap-samples", type=int, default=3)
    parser.add_argument("--max-iter", type=int, default=12)
    args = parser.parse_args()

    dataset, df = _load_dataset(args.dataset_id, Path(args.manifest))
    df = df.head(args.rows).copy()
    rows = []
    failures: list[dict[str, str]] = []
    runs = []

    for model_name in args.models:
        family = MODEL_FAMILIES.get(model_name)
        if family is None:
            failures.append({"model": model_name, "error": "unknown multi-treatment smoke model"})
            continue
        try:
            cfg = UpliftConfig(
                treatment_col=dataset["treatment_col"],
                outcome_col=dataset["outcome_col"],
                feature_cols=dataset["feature_cols"],
                categorical_cols=dataset.get("categorical_cols"),
                model_name=model_name,
                task=dataset.get("task", "classification"),
                max_rows=args.rows,
                bootstrap_samples=args.bootstrap_samples,
                run_name=f"multi-benchmark-{model_name.lower()}",
                model_params={"max_iter": args.max_iter, "random_state": 42},
            )
            result = train_multi_treatment_model(df, cfg, model_family=family)
            metrics = result["metrics"]
            row = {
                "dataset_id": args.dataset_id,
                "model": model_name,
                "strategy": metrics.get("strategy"),
                "run_id": result.get("run_id"),
                "run_dir": result.get("run_dir"),
                "incremental_policy_value_ipw": metrics.get("incremental_policy_value_ipw"),
                "recommended_policy_value_ipw": metrics.get("recommended_policy_value_ipw"),
                "control_policy_value_ipw": metrics.get("control_policy_value_ipw"),
                "action_count": len(metrics.get("treatment_values") or []),
                "top_action_rows": len(metrics.get("top_actions") or []),
                "propensity_rows": len(metrics.get("propensity_by_action") or []),
                "bootstrap_success": (metrics.get("policy_value_bootstrap") or {}).get("n_success"),
                "status": "ok",
            }
            if row["incremental_policy_value_ipw"] is None:
                raise ValueError("missing incremental_policy_value_ipw")
            if not metrics.get("top_actions") or not metrics.get("propensity_by_action"):
                raise ValueError("missing top action or propensity diagnostics")
            rows.append(row)
            runs.append({"model": model_name, "run_id": result.get("run_id"), "run_dir": result.get("run_dir"), "artifacts": result.get("artifacts")})
        except Exception as exc:
            failures.append({"model": model_name, "error": str(exc)})
            rows.append({"dataset_id": args.dataset_id, "model": model_name, "status": "failed", "error": str(exc)})

    leaderboard = sorted(
        [row for row in rows if row.get("status") == "ok"],
        key=lambda row: row.get("incremental_policy_value_ipw") or float("-inf"),
        reverse=True,
    )
    payload = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "dataset_id": args.dataset_id,
        "rows_used": int(len(df)),
        "models": args.models,
        "results": rows,
        "leaderboard": leaderboard,
        "runs": runs,
        "failures": failures,
        "status": "failed" if failures else "ok",
    }
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    timestamped = reports_dir / f"multi_treatment_benchmark_{time.strftime('%Y%m%d-%H%M%S')}.json"
    latest = reports_dir / "multi_treatment_benchmark_latest.json"
    latest_csv = reports_dir / "multi_treatment_benchmark_latest.csv"
    save_json(timestamped, payload)
    save_json(latest, payload)
    pd.DataFrame(rows).to_csv(latest_csv, index=False)
    if failures:
        raise SystemExit(json.dumps(payload, ensure_ascii=False, indent=2))
    print(json.dumps({"status": "ok", "path": str(latest), "leaderboard_rows": len(leaderboard)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
