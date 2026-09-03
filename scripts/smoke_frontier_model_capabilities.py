from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepuplift.core.frontier_models import (
    capability_counts,
    frontier_agent_reply,
    frontier_model_capabilities,
    frontier_source_rows,
    scenario_model_ladders,
)


REQUIRED_CAPABILITIES = {
    "Meta-Learners + LightGBM/CatBoost",
    "DR/DML + Causal Forest",
    "Uplift Tree / Uplift Random Forest",
    "UTBoost",
    "EFIN",
    "DESCN / ESX",
    "UMLC",
    "ECUP / full-funnel uplift",
    "CFR-DF / delayed feedback uplift",
    "Continuous Treatment / Dose-Response",
    "Cost-aware / Incremental Profit",
    "LLM Routing Uplift",
}


def main() -> None:
    rows = frontier_model_capabilities()
    names = {row["capability"] for row in rows}
    missing = sorted(REQUIRED_CAPABILITIES - names)
    if missing:
        raise AssertionError(f"Missing frontier capabilities: {missing}")

    required_fields = {
        "priority",
        "capability",
        "layer",
        "integration_status",
        "training_status",
        "treatment_type",
        "business_scenarios",
        "adapter_path",
        "source_title",
        "source_url",
        "metrics",
        "risks",
        "recommended_action",
    }
    broken = []
    for row in rows:
        empty = [field for field in required_fields if not str(row.get(field, "")).strip()]
        if empty:
            broken.append({"capability": row.get("capability"), "empty_fields": empty})
    if broken:
        raise AssertionError(f"Frontier rows have empty fields: {broken}")

    counts = capability_counts()
    if counts["ready"] < 5:
        raise AssertionError(f"Expected at least 5 ready-linked capabilities, got {counts}")
    if counts["backlog"] < 3:
        raise AssertionError(f"Expected at least 3 backlog capabilities, got {counts}")

    prompts = {
        "latest": "为什么不要只追最新 uplift 模型？",
        "utboost": "UTBoost 怎么接？",
        "efin": "EFIN 和 DESCN 适合什么？",
        "delayed": "ECUP 和 CFR-DF delayed feedback uplift 怎么落地？",
        "llm": "LLM routing 为什么是 uplift 问题？",
        "scenario": "发券、广告、LLM routing 业务场景的模型路径怎么选？",
        "source_trace": "demo preset 的指标来源和演示证据怎么追踪？",
        "governance": "为什么不是所有模型都 ready，guarded optional 模型怎么治理？",
    }
    agent_answers = {}
    for key, prompt in prompts.items():
        answer = frontier_agent_reply(prompt)
        if not answer or len(answer) < 80:
            raise AssertionError(f"Frontier Agent answer is too weak for {key}: {answer}")
        if key == "source_trace":
            required_terms = ["Workflow", "Evaluation", "Evidence", "metrics.json", "smoke_demo_preset_source_trace.py"]
            missing_terms = [term for term in required_terms if term.lower() not in answer.lower()]
            if missing_terms:
                raise AssertionError(f"Source-trace answer missing {missing_terms}: {answer}")
        if key == "governance":
            required_terms = ["Model Card Governance", "Evidence", "probe_optional_uplift_backends.py", "smoke_utboost_guarded_adapter.py"]
            missing_terms = [term for term in required_terms if term.lower() not in answer.lower()]
            if missing_terms:
                raise AssertionError(f"Governance answer missing {missing_terms}: {answer}")
        agent_answers[key] = answer

    out = {
        "counts": counts,
        "capabilities": rows,
        "sources": frontier_source_rows(),
        "scenario_ladders": scenario_model_ladders(),
        "agent_answer_keys": sorted(agent_answers),
    }
    ladders = scenario_model_ladders()
    if len(ladders) < 6:
        raise AssertionError(f"Expected at least 6 scenario ladders, got {len(ladders)}")
    ladder_text = "\n".join(str(row) for row in ladders)
    for required in ["Coupon", "Ads", "Growth", "Recommendation", "Marketplace", "LLM routing"]:
        if required not in ladder_text:
            raise AssertionError(f"Scenario ladder missing {required}")
    Path("reports").mkdir(exist_ok=True)
    out_path = Path("reports/frontier_model_capabilities_latest.json")
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "pass", "counts": counts, "path": str(out_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
