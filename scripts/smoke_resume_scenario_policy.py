from __future__ import annotations

import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepuplift.core.resume_scenario_policy import (  # noqa: E402
    RESUME_SCENARIO_IDS,
    didi_budget_allocation_plan,
    resume_scenario_policy_benchmarks,
)


def main() -> None:
    failures: list[str] = []
    rows = resume_scenario_policy_benchmarks()
    seen = {row["dataset_id"] for row in rows}
    missing = sorted(set(RESUME_SCENARIO_IDS) - seen)
    if missing:
        failures.append(f"Missing resume policy rows for: {missing}")
    for dataset_id in RESUME_SCENARIO_IDS:
        top10 = [row for row in rows if row["dataset_id"] == dataset_id and abs(float(row["top_fraction"]) - 0.10) < 1e-9]
        if not top10:
            failures.append(f"{dataset_id} missing Top 10% benchmark")
            continue
        if top10[0].get("policy_value_sum") is None:
            failures.append(f"{dataset_id} missing policy value sum")
    plan = didi_budget_allocation_plan(total_budget=100_000.0, max_cells=30)
    summary = plan["summary"]
    if summary["selected_cells"] <= 0:
        failures.append("DiDi budget allocation selected no cells")
    if summary["used_budget"] and summary["used_budget"] > summary["total_budget"] + 1e-6:
        failures.append("DiDi budget allocation exceeded total budget")
    if (summary.get("expected_policy_value") or 0.0) <= 0:
        failures.append("DiDi budget allocation expected policy value should be positive")

    output = ROOT / "reports" / "resume_scenario_policy_smoke_latest.json"
    output.parent.mkdir(exist_ok=True)
    payload = {
        "status": "fail" if failures else "ok",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "policy_rows": rows,
        "budget_plan": plan,
        "failures": failures,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "policy_rows": len(rows), "selected_cells": summary["selected_cells"], "failures": failures}, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
