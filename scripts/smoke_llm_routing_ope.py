from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepuplift.core.policy import (
    llm_routing_action_coverage_rows,
    llm_routing_bandit_drift_rows,
    llm_routing_ope_summary,
)


def main() -> None:
    summary = llm_routing_ope_summary()
    failures = []
    rows = summary.get("rows") or []
    if summary.get("status") != "ok":
        failures.append({"check": "status", "value": summary.get("status")})
    if len(rows) < 5:
        failures.append({"check": "policy_count", "value": len(rows)})
    for row in rows:
        if row.get("coverage_rate") is None or not (0.0 <= float(row["coverage_rate"]) <= 1.0):
            failures.append({"check": "coverage_rate", "policy": row.get("policy"), "value": row.get("coverage_rate")})
        if row.get("doubly_robust_value") is None:
            failures.append({"check": "doubly_robust_value", "policy": row.get("policy")})
        if row.get("launch_readiness") not in {"low", "medium", "high"}:
            failures.append({"check": "launch_readiness", "policy": row.get("policy"), "value": row.get("launch_readiness")})
    policy_names = {row.get("policy") for row in rows}
    for required in {"all_cheap", "all_strong", "positive_gain_router", "conservative_router", "oracle_best"}:
        if required not in policy_names:
            failures.append({"check": "required_policy", "missing": required})
    coverage_rows = llm_routing_action_coverage_rows()
    coverage_actions = {row.get("logged_action") for row in coverage_rows}
    for required_action in {"cheap_model", "strong_model", "rag", "tool", "human_review"}:
        if required_action not in coverage_actions:
            failures.append({"check": "coverage_action", "missing": required_action})
    for row in coverage_rows:
        if row.get("support_status") not in {"strong", "thin", "deficient"}:
            failures.append({"check": "support_status", "action": row.get("logged_action"), "value": row.get("support_status")})
    drift_rows = llm_routing_bandit_drift_rows()
    if len(drift_rows) < 5:
        failures.append({"check": "drift_rows", "value": len(drift_rows)})
    if not any(float(row.get("adaptive_minus_fixed") or 0.0) > 0 for row in drift_rows):
        failures.append({"check": "adaptive_improvement"})
    if not any("quality" in str(row.get("event", "")) for row in drift_rows):
        failures.append({"check": "quality_drift_event"})

    payload = {
        "status": "fail" if failures else "ok",
        "summary": summary,
        "coverage_rows": coverage_rows,
        "drift_rows": drift_rows,
        "failures": failures,
    }
    drift_csv = Path("reports/llm_routing_bandit_drift_latest.csv")
    if drift_rows:
        pd.DataFrame(drift_rows).to_csv(drift_csv, index=False)
        diagnostic_csv = Path("examples/diagnostic_cases/llm_routing_bandit_drift.csv")
        diagnostic_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(drift_rows).to_csv(diagnostic_csv, index=False)
        payload["bandit_drift_csv"] = str(drift_csv)
        payload["diagnostic_case_csv"] = str(diagnostic_csv)
    out = Path("reports/llm_routing_ope_smoke_latest.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
