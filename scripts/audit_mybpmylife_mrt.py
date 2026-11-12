#!/usr/bin/env python3
"""Audit a local myBPmyLife Push MRT source without downloading or copying it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deepuplift.data import audit_mrt_dataset, load_mybpmylife_mrt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Local MRT directory, CSV, or Data_MRT.zip")
    parser.add_argument("--output", type=Path, help="Optional JSON report path outside the repository")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    args = parser.parse_args()
    dataset = load_mybpmylife_mrt(args.path)
    report = audit_mrt_dataset(dataset, split_seed=args.split_seed, test_fraction=args.test_fraction)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["gate_status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
