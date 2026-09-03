from __future__ import annotations

import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepuplift.core.resume_scenario_workbench import (  # noqa: E402
    frontier_model_audit_rows_cn,
    industrial_extension_compare_rows_cn,
    industrial_extension_summary_rows_cn,
    llm_routing_action_coverage_rows_cn,
    llm_routing_bandit_drift_rows_cn,
    llm_routing_ope_scenario_rows_cn,
    model_limitations_rows_cn,
    research_evidence_gate_rows_cn,
    rollout_risk_rows_cn,
    resume_optimization_path_rows,
    resume_policy_rows_cn,
    resume_scenario_best_rows,
    resume_scenario_cards_cn,
    resume_scenario_compare_rows,
    resume_workbench_agent_reply,
)


def main() -> None:
    failures: list[str] = []
    cards = resume_scenario_cards_cn()
    compare = resume_scenario_compare_rows()
    best = resume_scenario_best_rows()
    policy = resume_policy_rows_cn()
    paths = resume_optimization_path_rows()
    audit = frontier_model_audit_rows_cn()
    limitations = model_limitations_rows_cn()
    extension_summary = industrial_extension_summary_rows_cn()
    extension_compare = industrial_extension_compare_rows_cn()
    rollout_risk = rollout_risk_rows_cn()
    llm_ope_rows = llm_routing_ope_scenario_rows_cn()
    llm_coverage_rows = llm_routing_action_coverage_rows_cn()
    llm_drift_rows = llm_routing_bandit_drift_rows_cn()
    research_gate_rows = research_evidence_gate_rows_cn()

    scenario_ids = {row["dataset_id"] for row in cards}
    compare_ids = {row["dataset_id"] for row in compare}
    best_ids = {row["dataset_id"] for row in best}

    if len(cards) != 5:
        failures.append(f"expected 5 resume scenarios, got {len(cards)}")
    missing_compare = sorted(scenario_ids - compare_ids)
    if missing_compare:
        failures.append(f"missing scenario compare evidence: {missing_compare}")
    missing_best = sorted(scenario_ids - best_ids)
    if missing_best:
        failures.append(f"missing best-model rows: {missing_best}")
    if len(policy) < 15:
        failures.append(f"expected at least 15 Top-K policy rows, got {len(policy)}")
    if len(paths) < len(cards) * 7:
        failures.append(f"expected V0-V6 optimization path per scenario, got {len(paths)} rows")
    for column in ["综合分", "上线判断", "排序理由"]:
        if compare and column not in compare[0]:
            failures.append(f"scenario compare rows missing column: {column}")
    if len(audit) < 10:
        failures.append(f"frontier model audit looks too small: {len(audit)}")
    if not limitations:
        failures.append("expected guarded/optional/research-backlog limitation rows")
    if len(extension_summary) < 9:
        failures.append(f"expected at least 9 industrial extension scenario rows, got {len(extension_summary)}")
    extension_ids = {row["dataset_id"] for row in extension_summary}
    extension_compare_ids = {row["dataset_id"] for row in extension_compare}
    missing_extension_compare = sorted(extension_ids - extension_compare_ids)
    if missing_extension_compare:
        failures.append(f"missing extension scenario compare evidence: {missing_extension_compare}")
    if len(rollout_risk) < len(cards) + len(extension_summary):
        failures.append(f"expected rollout risk rows for resume + extension scenarios, got {len(rollout_risk)}")
    if rollout_risk and "上线解释" not in rollout_risk[0]:
        failures.append("rollout risk rows missing 上线解释")
    if len(llm_ope_rows) < 5:
        failures.append(f"expected LLM OPE scenario rows, got {len(llm_ope_rows)}")
    if not any(row.get("策略") == "positive_gain_router" for row in llm_ope_rows):
        failures.append("LLM OPE scenario rows missing positive_gain_router")
    if len(llm_coverage_rows) < 5:
        failures.append(f"expected LLM action coverage rows, got {len(llm_coverage_rows)}")
    if len(llm_drift_rows) < 5:
        failures.append(f"expected LLM bandit drift rows, got {len(llm_drift_rows)}")
    if len(research_gate_rows) < 3:
        failures.append(f"expected research evidence gate rows, got {len(research_gate_rows)}")

    reply = resume_workbench_agent_reply("当前模型是不是业内先进？还有哪些依赖限制？每个场景怎么从 0-1 优化？") or ""
    for term in ["P0", "P1", "P2", "场景工作台", "scripts/smoke_industrial_scenario_compare.py"]:
        if term not in reply:
            failures.append(f"agent reply missing term: {term}")

    output = ROOT / "reports" / "resume_scenario_workbench_smoke_latest.json"
    output.parent.mkdir(exist_ok=True)
    payload = {
        "status": "fail" if failures else "ok",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "scenario_count": len(cards),
        "compare_rows": len(compare),
        "best_rows": len(best),
        "policy_rows": len(policy),
        "optimization_path_rows": len(paths),
        "frontier_audit_rows": len(audit),
        "limitation_rows": len(limitations),
        "extension_summary_rows": len(extension_summary),
        "extension_compare_rows": len(extension_compare),
        "rollout_risk_rows": len(rollout_risk),
        "llm_ope_rows": len(llm_ope_rows),
        "llm_coverage_rows": len(llm_coverage_rows),
        "llm_drift_rows": len(llm_drift_rows),
        "research_gate_rows": len(research_gate_rows),
        "agent_reply_preview": reply[:1200],
        "failures": failures,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
