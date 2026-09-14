#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from deepuplift.benchmarks.continuous import load_giks_continuous_benchmark, run_continuous_suite


def _json_default(value: Any):
    try:
        return value.item()
    except AttributeError:
        pass
    try:
        return value.tolist()
    except AttributeError:
        pass
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an offline continuous dose-response benchmark.")
    parser.add_argument("--model", action="append", dest="models", help="Model name; repeat to select several. Defaults to GBM, DRNet, VCNet, and both GIKS ablations.")
    parser.add_argument("--rows", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", choices=["synthetic", "ihdp", "news", "tcga"], default="synthetic")
    parser.add_argument("--data-path", type=Path, help="Local benchmark directory or normalized CSV; the runner never downloads raw data.")
    parser.add_argument("--split-id", type=int, default=0, help="Official GIKS split id for local IHDP/NEWS files.")
    parser.add_argument("--budget", type=float)
    parser.add_argument("--outcome-value", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=100, help="Factual training epochs for DRNet and VCNet.")
    parser.add_argument("--factual-epochs", type=int, default=80, help="GIKS factual-only stage epochs.")
    parser.add_argument("--giks-epochs", type=int, default=40, help="GIKS augmented stage epochs.")
    parser.add_argument("--out", type=Path, default=Path("runs/continuous"))
    args = parser.parse_args()

    models = args.models or ["DoseResponseGBM", "DRNet", "GIKS-DRNet", "VCNet", "GIKS-VCNet"]
    model_options = {
        "DRNet": {"epochs": args.epochs},
        "VCNet": {"epochs": args.epochs},
       "GIKS-VCNet": {"factual_epochs": args.factual_epochs, "giks_epochs": args.giks_epochs},
        "GIKS-DRNet": {"factual_epochs": args.factual_epochs, "giks_epochs": args.giks_epochs},
    }
    if args.dataset in {"ihdp", "news"}:
        if args.data_path is None:
            parser.error(f"--dataset {args.dataset} requires --data-path to user-provided official GIKS benchmark files.")
        data = load_giks_continuous_benchmark(args.dataset, args.data_path, split_id=args.split_id, outcome_value=args.outcome_value)
        result = run_continuous_suite(data=data, seed=args.split_id, model_names=models, outcome_value=args.outcome_value, budget=args.budget, model_options=model_options)
    elif args.dataset == "tcga":
        if args.data_path is None or not args.data_path.exists():
            print("TCGA: DATA_NOT_AVAILABLE (provide a local, licensed dataset path)")
            return 0
        parser.error("TCGA response-curve ground truth is not available in this benchmark runner; no metrics will be fabricated.")
    else:
        result = run_continuous_suite(
            rows=args.rows,
            seed=args.seed,
            model_names=models,
            outcome_value=args.outcome_value,
            budget=args.budget,
            model_options=model_options,
        )
    run_seed = args.split_id if args.dataset in {"ihdp", "news"} else args.seed
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"_seed{run_seed}"
    output = args.out / run_id
    output.mkdir(parents=True, exist_ok=False)

    artifacts = {
        "config.json": result["config"],
        "dataset.json": result["dataset"],
        "models.json": result["models"],
        "metrics.json": {row["model"]: row.get("metrics") for row in result["models"]},
        "policy.json": {row["model"]: row.get("policy") for row in result["models"]},
        "environment.json": result["environment"],
    }
    for filename, value in artifacts.items():
        (output / filename).write_text(json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    (output / "CONTINUOUS_BENCHMARK_REPORT.md").write_text(result["report"], encoding="utf-8")
    print(f"status={result['status']}")
    print(f"report={output / 'CONTINUOUS_BENCHMARK_REPORT.md'}")
    for row in result["models"]:
        metrics = row.get("metrics") or {}
        print(f"{row['model']}: {row['status']} MISE={metrics.get('mise')} economic_regret={metrics.get('economic_policy_regret')}")
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
