#!/usr/bin/env python3
"""Run the real-data Push MRT randomized baseline on a local source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deepuplift.benchmarks import run_push_mrt_baseline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Local Data_MRT.zip or extracted directory")
    parser.add_argument("--output", type=Path, required=True, help="Non-sensitive JSON report path")
    parser.add_argument("--bootstrap-repetitions", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = run_push_mrt_baseline(args.path, bootstrap_repetitions=args.bootstrap_repetitions, seed=args.seed)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
