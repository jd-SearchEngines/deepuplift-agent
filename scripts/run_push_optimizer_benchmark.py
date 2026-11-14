#!/usr/bin/env python3
"""Run the sparse multi-constraint optimizer benchmark."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deepuplift.decision.optimization import run_constrained_optimizer_suite


def _jsonable(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if hasattr(value, "item"):
        return _jsonable(value.item())
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, action="append", dest="seeds")
    parser.add_argument("--scale", type=int, action="append", dest="scales")
    parser.add_argument("--memory-budget-mb", type=float, default=None)
    parser.add_argument("--isolated", action="store_true", help="run each scale solver in a separate process for isolated RSS")
    parser.add_argument("--allow-oversize", action="store_true", help="allow scales above 1M after the resource guard")
    args = parser.parse_args()
    seeds = args.seeds or [42]
    reports = [
        run_constrained_optimizer_suite(
            seed=seed,
            scales=tuple(args.scales or [1000, 10000]),
            isolate_scales=args.isolated,
            memory_budget_mb=args.memory_budget_mb,
            allow_oversize=args.allow_oversize,
        )
        for seed in seeds
    ]
    report = reports[0] if len(reports) == 1 else {"status": "PASS" if all(item["status"] == "PASS" for item in reports) else "FAIL", "seed_count": len(reports), "seeds": seeds, "runs": reports}
    rendered = json.dumps(_jsonable(report), ensure_ascii=False, indent=2, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(f"status={report['status']} seeds={seeds}")
    if len(reports) > 1:
        for index, item in enumerate(reports):
            print(f"run_seed={seeds[index]} status={item['status']} feasibility={item.get('feasibility_status')} quality={item.get('quality_status')} convergence={item.get('convergence_status')}")
        return 0 if report["status"] == "PASS" else 2
    exact = report["small_instance"]["exact_reference"]["objective_net_value"]
    print(f"exact_small_objective={exact}")
    for row in report["small_instance"]["solvers"]:
        print(f"small {row['solver']}: status={row['status']} objective={row.get('objective_net_value')} gap={row.get('optimality_gap_vs_exact')} feasible={row.get('feasibility', {}).get('feasible')}")
    for scale in report["scales"]:
        if scale["status"] == "RESOURCE_GUARD_SKIPPED":
            print(f"scale={scale['size']} status=RESOURCE_GUARD_SKIPPED reason={scale['resource_guard']['reason']}")
            continue
        print(f"scale={scale['size']} memory_bytes={scale['candidate_memory_bytes']} peak_rss_mb={scale['peak_rss_mb']:.2f}")
        for row in scale["solvers"]:
            print(f"  {row['solver']}: objective={row['objective_net_value']:.4f} gap={row.get('relative_heuristic_gap')} runtime={row['runtime_seconds']:.4f} rss={row.get('process_peak_rss_mb')} feasible={row['feasibility']['feasible']}")
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
