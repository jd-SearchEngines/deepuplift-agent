#!/usr/bin/env python3
"""Run Push-derived monotonicity and smoothness calibration comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deepuplift.benchmarks import run_push_calibration_benchmark


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = run_push_calibration_benchmark(args.path, rows=args.rows, seed=args.seed)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, default=lambda value: value.tolist() if hasattr(value, "tolist") else str(value))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(f"status={report['status']}")
    for shape in report["shapes"]:
        print(f"shape={shape['shape']}")
        for row in shape["results"]:
            print(f"  {row['method']}: violation={row['monotonic_violation_rate']:.6f} mise={row['metrics']['mise']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
