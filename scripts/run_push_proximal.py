#!/usr/bin/env python3
"""Run the participant-held-out proximal Push effect baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deepuplift.benchmarks import run_mrt_proximal_baseline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Local MRT source; raw data is not copied")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--outcome", default="activity_steps_60min", choices=["activity_steps_60min", "diet_low_salt_24hr"])
    parser.add_argument("--arm", default="activity", choices=["activity", "diet"])
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = run_mrt_proximal_baseline(
        args.path,
        outcome=args.outcome,
        arm=args.arm,
        n_splits=args.splits,
        seed=args.seed,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
