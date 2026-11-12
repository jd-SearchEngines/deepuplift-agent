#!/usr/bin/env python3
"""Run Push-context semi-synthetic continuous-dose model benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deepuplift.benchmarks import run_push_continuous_suite


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", action="append", dest="models")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--factual-epochs", type=int, default=40)
    parser.add_argument("--giks-epochs", type=int, default=20)
    args = parser.parse_args()
    models = args.models or ["DoseResponseGBM", "DRNet", "VCNet", "GIKS-DRNet", "GIKS-VCNet"]
    options = {
        "DRNet": {"epochs": args.epochs},
        "VCNet": {"epochs": args.epochs},
        "GIKS-DRNet": {"factual_epochs": args.factual_epochs, "giks_epochs": args.giks_epochs},
        "GIKS-VCNet": {"factual_epochs": args.factual_epochs, "giks_epochs": args.giks_epochs},
    }
    report = run_push_continuous_suite(args.path, rows=args.rows, seed=args.seed, model_names=models, model_options=options)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, default=lambda value: value.tolist() if hasattr(value, "tolist") else str(value))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(f"status={report['status']}")
    print(f"output={args.output}")
    for shape in report["shapes"]:
        print(f"shape={shape['shape']} status={shape['status']}")
        for row in shape["models"]:
            print(f"  {row['model']}: {row['status']} mise={(row.get('metrics') or {}).get('mise')}")
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
