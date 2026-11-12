#!/usr/bin/env python3
"""Run the sparse multi-constraint optimizer benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deepuplift.decision.optimization import run_constrained_optimizer_suite


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scale", type=int, action="append", dest="scales")
    args = parser.parse_args()
    report = run_constrained_optimizer_suite(seed=args.seed, scales=tuple(args.scales or [1000, 10000]))
    rendered = json.dumps(_jsonable(report), ensure_ascii=False, indent=2, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(f"status={report['status']}")
    exact = report["small_instance"]["exact_reference"]["objective_net_value"]
    print(f"exact_small_objective={exact}")
    for row in report["small_instance"]["solvers"]:
        print(f"small {row['solver']}: status={row['status']} objective={row.get('objective_net_value')} gap={row.get('optimality_gap_vs_exact')} feasible={row.get('feasibility', {}).get('feasible')}")
    for scale in report["scales"]:
        print(f"scale={scale['size']} memory_bytes={scale['candidate_memory_bytes']} peak_rss_mb={scale['peak_rss_mb']:.2f}")
        for row in scale["solvers"]:
            print(f"  {row['solver']}: objective={row['objective_net_value']:.4f} runtime={row['runtime_seconds']:.4f} feasible={row['feasibility']['feasible']}")
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
