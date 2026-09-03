from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepuplift.core.industry_playbook import industry_agent_reply


PROMPT_EXPECTATIONS = [
    ("发券为什么不能只看转化率？", ["coupon", "sure thing", "gross profit"]),
    ("广告投放里 uplift 和 pCTR/pCVR 有什么区别？", ["pCTR", "pCVR", "incrementality"]),
    ("怎么把 uplift score 转成投放阈值和 ROI？", ["Top-K", "ROI", "policy"]),
    ("用户增长里怎么处理频控和疲劳？", ["delayed feedback", "fatigue", "opt-out"]),
    ("多 treatment、连续 treatment、长期 treatment 怎么做？", ["multi", "dose-response", "marketplace"]),
    ("工业界有哪些 uplift 落地案例？", ["Tencent", "Meituan", "DoorDash"]),
    ("发券如何处理补贴套利和羊毛党？", ["abuse", "gross_margin", "sure thing"]),
    ("预算有限时 uplift 阈值怎么定？", ["budget", "Top-K", "net value"]),
    ("Marketplace 干扰和 spillover 怎么处理？", ["SUTVA", "spillover", "multi-treatment"]),
    ("Push 频控和用户疲劳怎么建模？", ["fatigue", "opt-out", "delayed-feedback"]),
    ("Spotify 站内消息为什么要做 uplift？", ["Spotify", "holdout", "CATE"]),
    ("DoorDash ghost ads 怎么衡量广告增量？", ["DoorDash", "ghost", "iROAS"]),
    ("Airbnb incremental LTV 怎么接到补贴 ROI？", ["Airbnb", "incremental LTV", "cannibalization"]),
    ("阿里 X-Learner AB 实验异质性怎么讲？", ["X-Learner", "CATE", "SHAP"]),
    ("EdgeRec 请求价值增益模型怎么讲？", ["request", "Recommendation", "QPS"]),
    ("多个 CRM journey 重叠时 uplift 怎么评估？", ["pure lift", "global lift", "cannibalization"]),
    ("Uplift 和 LLM 怎么结合？", ["causal copilot", "Text/creative", "LLM routing"]),
        ("LLM routing 为什么是 uplift 问题？", ["strong model", "incremental quality", "budget"]),
        ("LLM 生成文案怎么做 uplift 实验？", ["text treatment", "embedding", "holdout"]),
        ("LLM routing 离线训练数据怎么采集？", ["随机探索", "overlap", "DR/R"]),
        ("什么时候用 uplift，什么时候升级到 bandit？", ["fixed threshold", "contextual bandit", "budget pacing"]),
    ]


def main() -> None:
    context = {
        "task": "classification",
        "model_name": "DRLearnerLightGBM",
        "treatment_col": "treatment",
        "outcome_col": "visit",
    }
    rows = []
    failures = []
    for prompt, expected_terms in PROMPT_EXPECTATIONS:
        answer = industry_agent_reply(prompt, context) or ""
        matched = {term: term.lower() in answer.lower() for term in expected_terms}
        rows.append({"prompt": prompt, "answer_chars": len(answer), "matched": matched})
        missing = [term for term, ok in matched.items() if not ok]
        if missing:
            failures.append({"prompt": prompt, "missing": missing})

    payload = {"status": "fail" if failures else "ok", "rows": rows, "failures": failures}
    out = Path("reports/industry_agent_answers_smoke_latest.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
