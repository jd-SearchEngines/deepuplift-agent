import argparse
import json

from .config import ReleaseConfig
from .suite import run_release_benchmark


def main() -> int:
    parser = argparse.ArgumentParser(description="Run DeepUplift Release Benchmark v1")
    parser.add_argument("--output-dir", default="release_runs")
    parser.add_argument("--synthetic-rows", type=int, default=500)
    parser.add_argument("--scale-rows", nargs="+", type=int, default=[10_000, 100_000])
    parser.add_argument("--hillstrom-path")
    parser.add_argument("--criteo-path")
    args = parser.parse_args()
    report = run_release_benchmark(ReleaseConfig(output_dir=args.output_dir, synthetic_rows=args.synthetic_rows, scale_sizes=tuple(args.scale_rows), hillstrom_path=args.hillstrom_path, criteo_path=args.criteo_path))
    print(json.dumps({"run_id": report["run_id"], "commit": report["commit"], "overall": report["release_gate"]["overall"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
