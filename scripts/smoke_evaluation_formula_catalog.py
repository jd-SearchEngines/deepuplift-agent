from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepuplift.core.evaluation_formulas import (
    evaluation_formula_agent_reply,
    evaluation_formula_counts,
    evaluation_formula_rows,
    optimization_path_rows,
    policy_derivation_rows,
)


REQUIRED_METRICS = {
    "ITE / CATE",
    "uplift score",
    "QINI",
    "AUUC",
    "uplift@K / Top-K observed uplift",
    "policy value",
    "incremental profit / cost-aware gain",
    "iROAS",
    "DR pseudo-outcome",
    "R-learner objective",
    "calibration MAE",
    "bootstrap CI",
    "overlap / trimming sensitivity",
    "delayed feedback uplift",
    "full-funnel ECUP metric",
    "LLM routing cost-quality value",
}


def assert_reply(prompt: str, expected_terms: list[str]) -> None:
    reply = evaluation_formula_agent_reply(prompt)
    if not reply:
        raise SystemExit(f"empty formula agent reply for prompt: {prompt}")
    missing = [term for term in expected_terms if term not in reply]
    if missing:
        raise SystemExit(f"reply missing {missing} for prompt {prompt}: {reply[:300]}")


def main() -> None:
    rows = evaluation_formula_rows()
    names = {row["metric"] for row in rows}
    missing = sorted(REQUIRED_METRICS - names)
    if missing:
        raise SystemExit(f"missing formula rows: {missing}")
    for field in ["formula", "derivation_intuition", "business_meaning", "misleading_when", "code_source", "artifact_field"]:
        bad = [row["metric"] for row in rows if not str(row.get(field, "")).strip()]
        if bad:
            raise SystemExit(f"blank {field}: {bad}")
    counts = evaluation_formula_counts()
    if counts["metrics"] < 16 or len(policy_derivation_rows()) < 3 or len(optimization_path_rows()) < 8:
        raise SystemExit(f"formula catalog counts too small: {counts}")

    assert_reply("QINI / AUUC / uplift@K 有什么区别？", ["QINI", "AUUC", "uplift@K"])
    assert_reply("policy value 怎么推导，发券 ROI 怎么算？", ["K*", "发券", "iROAS"])
    assert_reply("DR learner / R learner 的目标函数怎么讲？", ["DR pseudo-outcome", "R-learner"])
    assert_reply("从0-1优化 uplift 项目怎么讲？", ["V0", "V8", "面试讲法"])

    out = Path("reports/evaluation_formula_catalog_smoke_latest.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"status": "pass", "counts": counts, "metrics": sorted(names)}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "pass", "counts": counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
