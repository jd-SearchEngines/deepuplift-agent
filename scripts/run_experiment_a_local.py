#!/usr/bin/env python3
"""Run the local Experiment A comparison with ignored JSON evidence output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deepuplift.benchmarks.experiment_a import frontier_status, run_binary_comparison, run_continuous_comparison, run_multi_comparison


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[5000, 20000])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 52, 62, 72, 82])
    parser.add_argument("--binary-only", action="store_true")
    parser.add_argument("--continuous-only", action="store_true")
    parser.add_argument("--continuous-epochs", type=int, default=10)
    parser.add_argument("--factual-epochs", type=int, default=8)
    parser.add_argument("--giks-epochs", type=int, default=4)
    parser.add_argument("--neural-device", choices=["cpu", "mps"], default="cpu")
    args = parser.parse_args()
    if args.binary_only and args.continuous_only:
        parser.error("--binary-only and --continuous-only are mutually exclusive")
    payload = {"config": {"sizes": args.sizes, "seeds": args.seeds}, "frontier_status": frontier_status()}
    if not args.continuous_only:
        print("[Experiment A] A1 binary")
        payload["a1_binary"] = run_binary_comparison(sizes=args.sizes, seeds=args.seeds)
        print("[Experiment A] A2 multi-treatment")
        payload["a2_multi"] = run_multi_comparison(sizes=args.sizes, seeds=args.seeds)
    if not args.binary_only:
        print("[Experiment A] A3 continuous")
        payload["a3_continuous"] = run_continuous_comparison(sizes=args.sizes, seeds=args.seeds, epochs=args.continuous_epochs, factual_epochs=args.factual_epochs, giks_epochs=args.giks_epochs, neural_device=args.neural_device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False, default=lambda value: value.tolist() if hasattr(value, "tolist") else str(value)) + "\n", encoding="utf-8")
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
