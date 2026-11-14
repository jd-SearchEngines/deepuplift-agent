#!/usr/bin/env python3
"""Run the local Experiment A comparison with ignored JSON evidence output."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deepuplift.benchmarks.experiment_a import (
    frontier_status,
    run_binary_comparison,
    run_binary_neural_comparison,
    run_continuous_comparison,
    run_multi_comparison,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[5000, 20000])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 52, 62, 72, 82])
    parser.add_argument("--binary-only", action="store_true")
    parser.add_argument("--continuous-only", action="store_true")
    parser.add_argument("--neural-only", action="store_true")
    parser.add_argument("--neural-epochs", type=int, default=8)
    parser.add_argument("--neural-device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--continuous-epochs", type=int, default=10)
    parser.add_argument("--factual-epochs", type=int, default=8)
    parser.add_argument("--giks-epochs", type=int, default=4)
    args = parser.parse_args()
    if sum(bool(value) for value in (args.binary_only, args.continuous_only, args.neural_only)) > 1:
        parser.error("--binary-only, --continuous-only and --neural-only are mutually exclusive")
    payload = {"config": {"sizes": args.sizes, "seeds": args.seeds}, "frontier_status": frontier_status()}
    if args.neural_only:
        print("[Experiment A] A1 neural adapters")
        payload["a1_neural"] = run_binary_neural_comparison(sizes=args.sizes, seeds=args.seeds, epochs=args.neural_epochs, neural_device=args.neural_device)
    elif not args.continuous_only:
        print("[Experiment A] A1 binary")
        payload["a1_binary"] = run_binary_comparison(sizes=args.sizes, seeds=args.seeds)
        print("[Experiment A] A2 multi-treatment")
        payload["a2_multi"] = run_multi_comparison(sizes=args.sizes, seeds=args.seeds)
    if not args.binary_only and not args.neural_only:
        print("[Experiment A] A3 continuous")
        payload["a3_continuous"] = run_continuous_comparison(sizes=args.sizes, seeds=args.seeds, epochs=args.continuous_epochs, factual_epochs=args.factual_epochs, giks_epochs=args.giks_epochs, neural_device=args.neural_device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False, default=lambda value: value.tolist() if hasattr(value, "tolist") else str(value)) + "\n", encoding="utf-8")
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
