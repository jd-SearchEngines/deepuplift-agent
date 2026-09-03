from deepuplift.benchmarks.benchmark_scalability import run_scalability_benchmark


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Measured DeepUplift scalability smoke benchmark")
    parser.add_argument("--rows", nargs="+", type=int, default=[10_000])
    args = parser.parse_args()
    print(json.dumps(run_scalability_benchmark(args.rows), indent=2))
