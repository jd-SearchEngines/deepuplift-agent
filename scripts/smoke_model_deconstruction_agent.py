from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepuplift.core.interview_agent import interview_agent_reply  # noqa: E402
from deepuplift.core.model_deconstruction import (  # noqa: E402
    MODEL_DECONSTRUCTION_GOLDEN_QUESTIONS,
    deconstruction_regression_rows,
)


def main() -> None:
    context = {
        "model_name": "CFRNet",
        "latest_run_dir": "reports/cfrnet_balance_training_runs/latest",
        "task": "classification",
    }
    rows = deconstruction_regression_rows(context)

    integration_failures = []
    integration_cases = [
        case
        for case in MODEL_DECONSTRUCTION_GOLDEN_QUESTIONS
        if any(term in case["prompt"].lower() or term in case["prompt"] for term in ("源码", "loss", "forward", "fit", "predict", "license", "训练评估", "Model Deconstruction".lower()))
    ][:6]
    for case in integration_cases:
        answer = interview_agent_reply(case["prompt"], context) or ""
        if "Model Deconstruction Agent 3.0" not in answer:
            integration_failures.append(f"interview_agent did not route deconstruction prompt: {case['prompt']}")

    failures = []
    if len(rows) < 20:
        failures.append(f"expected at least 20 model deconstruction golden questions; got {len(rows)}")
    for row in rows:
        if row["answer_chars"] < 1200:
            failures.append(f"short answer for {row['prompt']}: {row['answer_chars']} chars")
        if row.get("missing_terms"):
            failures.append(f"missing expected terms for {row['prompt']}: {row['missing_terms']}")
        failed_checks = [name for name, ok in (row.get("checks") or {}).items() if not ok]
        if failed_checks:
            failures.append(f"failed rubric checks for {row['prompt']}: {failed_checks}")
    failures.extend(integration_failures)

    avg_ratio = sum(float(row.get("rubric_ratio") or 0.0) for row in rows) / max(len(rows), 1)
    payload = {
        "schema_version": 1,
        "status": "fail" if failures else "ok",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "golden_questions": len(MODEL_DECONSTRUCTION_GOLDEN_QUESTIONS),
        "rows": rows,
        "avg_rubric_ratio": round(avg_ratio, 4),
        "failures": failures,
        "rubric": {
            "source_code": "answer cites fit/forward/predict/loss or repo code paths",
            "math": "answer includes tau/mu/pseudo/IPM/propensity formulas",
            "training": "answer explains train or fit flow",
            "evaluation": "answer links to QINI/AUUC/policy/metrics/evidence",
            "license": "answer names source/license risk",
            "interview": "answer includes 5/15/30 minute talk tracks",
        },
    }
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    latest = reports / "model_deconstruction_agent_latest.json"
    timestamped = reports / f"model_deconstruction_agent_{time.strftime('%Y%m%d-%H%M%S')}.json"
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    latest.write_text(text, encoding="utf-8")
    timestamped.write_text(text, encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "questions": len(rows),
                "avg_rubric_ratio": payload["avg_rubric_ratio"],
                "path": str(latest),
            },
            ensure_ascii=False,
        )
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
