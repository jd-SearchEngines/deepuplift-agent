from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepuplift.core.failure_modes import (
    diagnostic_case_rows,
    failure_mode_agent_reply,
    failure_mode_counts,
    model_failure_mode_rows,
    scenario_pitfall_rows,
)


REQUIRED_MODEL_FAMILIES = {
    "S-Learner",
    "T-Learner",
    "X-Learner",
    "DR / R Learner",
    "CausalForest / ForestDR",
    "Delayed Feedback / ECUP",
    "LLM Routing Uplift",
}
REQUIRED_CASES = {
    "weak_uplift_signal",
    "imbalanced_treatment",
    "extreme_propensity",
    "negative_uplift_sleeping_dog",
    "high_qini_low_roi",
    "delayed_feedback_trap",
    "full_funnel_click_trap",
    "geo_holdout_conflict",
    "llm_routing_cost_trap",
    "continuous_dose_saturation",
}


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def main() -> None:
    failures: list[str] = []
    counts = failure_mode_counts()
    model_rows = model_failure_mode_rows()
    pitfall_rows = scenario_pitfall_rows()
    case_rows = diagnostic_case_rows()
    require(counts["model_failure_modes"] >= 13, f"too few model failure modes: {counts}", failures)
    require(counts["scenario_pitfalls"] >= 9, f"too few scenario pitfalls: {counts}", failures)
    require(counts["diagnostic_cases"] >= 10, f"too few diagnostic cases: {counts}", failures)
    require(REQUIRED_MODEL_FAMILIES <= {row["family"] for row in model_rows}, "missing required model family rows", failures)
    require(REQUIRED_CASES <= {row["case_id"] for row in case_rows}, "missing required diagnostic cases", failures)
    for row in model_rows:
        for field in ["risk", "symptom", "diagnostic", "fix", "interview_line"]:
            require(bool(str(row.get(field, "")).strip()), f"blank {field} for {row.get('family')}", failures)
    for row in pitfall_rows:
        for field in ["scenario", "pitfall", "diagnostic", "fix", "interview_line"]:
            require(bool(str(row.get(field, "")).strip()), f"blank {field} for {row.get('scenario')}", failures)

    summary_path = ROOT / "reports" / "uplift_diagnostic_cases_latest.csv"
    require(summary_path.exists(), "missing reports/uplift_diagnostic_cases_latest.csv; run generator first", failures)
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        names = set(summary["case_id"].astype(str))
        require(REQUIRED_CASES <= names, f"diagnostic report missing cases: {sorted(REQUIRED_CASES - names)}", failures)
        high_qini = summary[summary["case_id"] == "high_qini_low_roi"]
        if not high_qini.empty:
            require(float(high_qini.iloc[0]["top10_policy_value"]) < 0, "high_qini_low_roi should have negative top10 policy value", failures)
        delayed = summary[summary["case_id"] == "delayed_feedback_trap"]
        if not delayed.empty:
            require(float(delayed.iloc[0]["d30_rate_top10"]) >= float(delayed.iloc[0]["d1_rate_top10"]), "delayed trap should show D30 >= D1", failures)
        funnel = summary[summary["case_id"] == "full_funnel_click_trap"]
        if not funnel.empty:
            require(
                float(funnel.iloc[0]["click_uplift_top10"]) > float(funnel.iloc[0]["conversion_uplift_top10"]),
                "full funnel trap should show click uplift > conversion uplift",
                failures,
            )
        overlap = summary[summary["case_id"] == "extreme_propensity"]
        if not overlap.empty:
            require(float(overlap.iloc[0]["weak_overlap_rate"]) > 0.6, "extreme propensity should have high weak overlap rate", failures)

    prompts = [
        ("S/T/X/DR/R learner 分别容易出什么问题？", ["S-Learner", "DR", "Evidence"]),
        ("为什么 QINI 高但 ROI 可能低？", ["QINI", "ROI", "policy"]),
        ("LLM routing 为什么需要随机探索？", ["LLM", "成本", "探索"]),
    ]
    prompt_rows = []
    for prompt, expected in prompts:
        reply = failure_mode_agent_reply(prompt)
        prompt_rows.append({"prompt": prompt, "answer_chars": len(reply or "")})
        require(reply is not None, f"no reply for prompt: {prompt}", failures)
        if reply:
            for token in expected:
                require(token.lower() in reply.lower(), f"reply for {prompt} missing {token}", failures)

    payload = {
        "status": "fail" if failures else "ok",
        "counts": counts,
        "prompt_rows": prompt_rows,
        "failures": failures,
    }
    out = ROOT / "reports" / "uplift_failure_modes_smoke_latest.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
