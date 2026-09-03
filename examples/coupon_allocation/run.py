from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepuplift.application import run_uplift_pipeline  # noqa: E402
from deepuplift.scenarios.coupon_allocation import generate_coupon_data  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the DeepUplift coupon allocation vertical slice.")
    parser.add_argument("--rows", type=int, default=10_000)
    parser.add_argument("--budget", type=float, default=50_000.0)
    parser.add_argument("--coupon-cost", type=float, default=10.0)
    parser.add_argument("--order-value", type=float, default=35.0)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/coupon_allocation"))
    args = parser.parse_args()

    frame = generate_coupon_data(args.rows)
    feature_cols = ["affinity", "price_sensitivity", "sessions_7d", "orders_30d", "spend_90d", "recency_days", "region"]
    result = run_uplift_pipeline(
        frame,
        feature_cols=feature_cols,
        treatment_col="treatment",
        outcome_col="outcome",
        id_column="user_id",
        treatment_cost=args.coupon_cost,
        outcome_value=args.order_value,
        budget=args.budget,
        task="classification",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payloads = {
        "diagnostics.json": result.diagnostics.to_dict(),
        "benchmark.json": result.benchmark.to_dict(),
        "policy.json": result.policy.to_dict(),
        "experiment_plan.json": result.experiment_plan,
    }
    for filename, payload in payloads.items():
        (args.output_dir / filename).write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    summary = result.policy.summary
    print("DeepUplift Coupon Allocation Demo")
    print("=================================")
    print(f"DATA       rows={result.diagnostics.sample_size} assignment={result.dataset.assignment_type.value} readiness={result.diagnostics.readiness}")
    print(f"MODEL      recommended={result.benchmark.recommended_model}")
    print(f"DECISION   targets={summary['target_count']} cost={summary['total_cost']:.2f} net_value={summary['expected_net_value']:.2f} iROAS={summary['iroas']}")
    print(f"EXPERIMENT  status={result.experiment_plan['status']} primary_metric={result.experiment_plan['primary_metric']}")
    print(f"ARTIFACTS   {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
