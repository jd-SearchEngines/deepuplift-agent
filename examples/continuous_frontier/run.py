#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from deepuplift.benchmarks.continuous import run_continuous_suite


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare continuous dose-response models on a small observational synthetic dataset.")
    parser.add_argument("--model", choices=["DoseResponseGBM", "DRNet", "VCNet", "GIKS-VCNet"], default="VCNet")
    parser.add_argument("--rows", type=int, default=240)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budget", type=float)
    parser.add_argument("--outcome-value", type=float, default=1.0)
    parser.add_argument("--out", type=Path, default=Path("continuous_frontier_result.json"))
    args = parser.parse_args()
    options = {
        "VCNet": {"epochs": 60},
        "DRNet": {"epochs": 60},
        "GIKS-VCNet": {"factual_epochs": 50, "giks_epochs": 25},
    }
    result = run_continuous_suite(
        rows=args.rows,
        seed=args.seed,
        model_names=[args.model],
        outcome_value=args.outcome_value,
        budget=args.budget,
        model_options=options,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True, default=lambda value: value.item() if hasattr(value, "item") else str(value)) + "\n", encoding="utf-8")
    print(result["report"])
    print(f"full curves and policy rows: {args.out}")
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
