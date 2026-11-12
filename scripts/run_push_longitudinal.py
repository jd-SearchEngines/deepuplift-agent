#!/usr/bin/env python3
"""Run the known-truth longitudinal Push policy laboratory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deepuplift.benchmarks import run_push_longitudinal_suite


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Local MRT source; raw data is not written to the report")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--users", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scenarios", nargs="+", default=None)
    parser.add_argument("--policies", nargs="+", default=None)
    args = parser.parse_args()
    report = run_push_longitudinal_suite(
        args.path,
        scenarios=args.scenarios,
        policies=args.policies,
        users=args.users,
        horizon=args.horizon,
        seed=args.seed,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
