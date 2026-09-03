import argparse
import json

from .config import ReleaseConfig
from .suite import run_release_benchmark


def main() -> int:
    parser = argparse.ArgumentParser(description="Run DeepUplift Release Benchmark v2")
    parser.add_argument("--output-dir", default="release_runs")
    parser.add_argument("--synthetic-rows", type=int, default=500)
    parser.add_argument("--scale-rows", "--scale", nargs="+", type=int, default=[10_000, 100_000])
    parser.add_argument("--real-data", action="store_true", help="Document that configured paths are upstream real datasets")
    parser.add_argument("--hillstrom-path")
    parser.add_argument("--criteo-path")
    parser.add_argument("--criteo-sample-rows", type=int, default=100_000)
    parser.add_argument("--packaging-python", help="Python executable used for build + fresh wheel smoke")
    parser.add_argument("--ci-status", default="EXTERNAL_CHECK", choices=["PASS", "FAIL", "EXTERNAL_CHECK"])
    parser.add_argument("--optional-report", help="JSON emitted by scripts/run_optional_backend_smoke.py")
    args = parser.parse_args()
    report = run_release_benchmark(ReleaseConfig(output_dir=args.output_dir, synthetic_rows=args.synthetic_rows, scale_sizes=tuple(args.scale_rows), hillstrom_path=args.hillstrom_path, criteo_path=args.criteo_path, criteo_sample_rows=args.criteo_sample_rows, packaging_python=args.packaging_python, ci_status=args.ci_status, optional_backend_report_path=args.optional_report))
    print(json.dumps({"run_id": report["run_id"], "commit": report["commit"], "overall": report["release_gate"]["overall"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
