#!/usr/bin/env python3
"""Build and audit canonical Push decision points from a local MRT source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deepuplift.data import build_mrt_decision_points


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Local Data_MRT.zip or extracted MRT directory")
    parser.add_argument("--outcome", default="activity_steps_60min", choices=["activity_steps_60min", "diet_low_salt_24hr"])
    parser.add_argument("--output", type=Path, required=True, help="Non-sensitive JSON report path")
    args = parser.parse_args()

    decision_points = build_mrt_decision_points(args.path, outcome=args.outcome)
    report = {
        "status": decision_points.validation["status"],
        "schema": decision_points.metadata,
        "validation": decision_points.validation,
        "analysis_ready_rows": int(len(decision_points.analysis_ready)),
        "raw_data_redistributed": False,
        "frame_exported": False,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
