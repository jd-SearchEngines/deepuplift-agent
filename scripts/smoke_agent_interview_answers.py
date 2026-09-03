from __future__ import annotations

import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepuplift.core.agent_v2 import AGENT_V2_GOLDEN_QUESTIONS, agent_v2_regression_rows  # noqa: E402
from deepuplift.core.interview_agent import interview_agent_reply  # noqa: E402


def main() -> None:
    context = {
        "task": "classification",
        "model_name": "DRLearnerGBM",
        "treatment_col": "treatment",
        "outcome_col": "visit",
        "unit_col": "user_id",
        "time_window": "7-day conversion / quality window",
        "registered_models": 103,
        "ready_models": 47,
    }
    rows = agent_v2_regression_rows(context)
    smoke_rows = []
    failures = []

    for row in rows:
        prompt = row["prompt"]
        answer = interview_agent_reply(prompt, context) or ""
        smoke_rows.append(
            {
                "prompt": prompt,
                "answer_chars": len(answer),
                "ratio": row["ratio"],
                "checks": row["checks"],
                "expected": row["expected"],
                "missing_expected": row["missing_expected"],
            }
        )
        if len(answer) < 600:
            failures.append({"prompt": prompt, "reason": "answer_too_short", "answer_chars": len(answer)})
        if row["missing_expected"]:
            failures.append({"prompt": prompt, "reason": "missing_expected", "missing": row["missing_expected"]})
        failed_checks = [name for name, ok in row["checks"].items() if not ok]
        if failed_checks:
            failures.append({"prompt": prompt, "reason": "rubric_checks_failed", "checks": failed_checks})

    ratios = [float(row["ratio"]) for row in rows]
    avg_ratio = sum(ratios) / len(ratios) if ratios else 0.0
    if len(rows) < 20:
        failures.append({"reason": "too_few_golden_questions", "rows": len(rows)})
    if avg_ratio < 0.85:
        failures.append({"reason": "low_average_rubric_ratio", "avg_ratio": round(avg_ratio, 4)})

    payload = {
        "schema_version": 2,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "status": "fail" if failures else "ok",
        "golden_questions": len(AGENT_V2_GOLDEN_QUESTIONS),
        "avg_rubric_ratio": round(avg_ratio, 4),
        "min_rubric_ratio": round(min(ratios), 4) if ratios else 0.0,
        "rows": smoke_rows,
        "failures": failures,
        "rubric": {
            "causal_parse": "answer names treatment, outcome and unit",
            "risk_diagnosis": "answer covers causal risks such as selection bias, overlap, leakage, SUTVA, delayed feedback or metric mismatch",
            "model_recommendation": "answer recommends relevant uplift / CATE model families",
            "evaluation_online": "answer connects offline metrics to OPE, holdout, A/B or ramp-up",
            "evidence_grounding": "answer cites repo docs, reports or code paths",
            "interview_packaging": "answer includes 5/15/30 minute talk tracks",
            "answer_modes": "answer separates algorithm, architecture, ROI, rollout and interview views",
        },
    }
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    latest = reports_dir / "agent_v2_regression_latest.json"
    timestamped = reports_dir / f"agent_v2_regression_{time.strftime('%Y%m%d-%H%M%S')}.json"
    legacy_latest = reports_dir / "agent_interview_answers_smoke_latest.json"
    for path in [latest, timestamped, legacy_latest]:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "questions": len(rows), "avg_rubric_ratio": payload["avg_rubric_ratio"], "path": str(latest)}, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
