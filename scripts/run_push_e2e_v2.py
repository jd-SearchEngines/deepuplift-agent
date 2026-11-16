#!/usr/bin/env python3
"""Run the Batch 3 known-truth Push decision-chain benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from deepuplift.benchmarks.push_e2e_v2 import render_push_e2e_report, run_push_e2e_v2


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return _jsonable(value.item())
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--users", type=int, default=300)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 52, 62, 72, 82])
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    report = run_push_e2e_v2(users=args.users, horizon=args.horizon, seeds=args.seeds)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(_jsonable(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    args.output_md.write_text(render_push_e2e_report(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "run_id": report["run_id"], "json": str(args.output_json), "markdown": str(args.output_md)}, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

