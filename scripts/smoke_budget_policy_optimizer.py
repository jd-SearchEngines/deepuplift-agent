from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepuplift.core.artifacts import save_json
from deepuplift.core.policy import budget_policy_optimizer, budget_policy_sensitivity


def main() -> None:
    df = pd.read_csv("examples/datasets/synthetic_llm_routing_uplift_8k.csv", nrows=600)
    df["uplift_score"] = pd.to_numeric(df["true_uplift"], errors="coerce")
    df["policy_cost"] = (
        pd.to_numeric(df["incremental_cost"], errors="coerce").fillna(0.0)
        + pd.to_numeric(df["latency_penalty"], errors="coerce").fillna(0.0)
    )
    df["risk_penalty"] = (
        0.08 * pd.to_numeric(df["safety_risk"], errors="coerce").fillna(0.0)
        + 0.04 * pd.to_numeric(df["prior_fail_rate"], errors="coerce").fillna(0.0)
    )
    budget = float(df["policy_cost"].quantile(0.75) * 80)
    result = budget_policy_optimizer(
        df,
        uplift_col="uplift_score",
        value_col="user_value",
        cost_col="policy_cost",
        risk_col="risk_penalty",
        group_col="task_type",
        budget=budget,
        max_contacts=120,
        min_net_value=0.0,
        risk_weight=1.0,
        rank_by="net_value",
    )
    sensitivity = budget_policy_sensitivity(
        df,
        budgets=[budget * 0.25, budget * 0.5, budget, budget * 1.5],
        uplift_col="uplift_score",
        value_col="user_value",
        cost_col="policy_cost",
        risk_col="risk_penalty",
        max_contacts=120,
        min_net_value=0.0,
        risk_weight=1.0,
        rank_by="net_value",
    )
    failures = []
    if result.get("selected_rows", 0) <= 0:
        failures.append("optimizer selected no rows")
    if (result.get("budget_used") or 0.0) > budget + 1e-9:
        failures.append("optimizer exceeded budget")
    if result.get("net_value") is None or result["net_value"] <= 0:
        failures.append("optimizer net value is not positive")
    if not result.get("group_summary"):
        failures.append("optimizer group summary is missing")
    if len(sensitivity) != 4:
        failures.append("budget sensitivity rows are incomplete")

    payload = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "dataset": "synthetic_llm_routing_uplift_8k",
        "rows": int(len(df)),
        "budget": budget,
        "result": result,
        "sensitivity": sensitivity,
        "failures": failures,
        "status": "failed" if failures else "ok",
    }
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    timestamped = reports_dir / f"budget_policy_optimizer_smoke_{time.strftime('%Y%m%d-%H%M%S')}.json"
    latest = reports_dir / "budget_policy_optimizer_smoke_latest.json"
    save_json(timestamped, payload)
    save_json(latest, payload)
    if failures:
        raise SystemExit(json.dumps(payload, ensure_ascii=False, indent=2))
    print(
        json.dumps(
            {
                "status": "ok",
                "path": str(latest),
                "selected_rows": result["selected_rows"],
                "budget_used": result["budget_used"],
                "net_value": result["net_value"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
