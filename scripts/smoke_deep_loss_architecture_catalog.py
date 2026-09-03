from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepuplift.core.deep_uplift_designs import (
    deep_design_agent_reply,
    deep_design_counts,
    deep_loss_catalog,
    deep_source_rows,
    local_causal_report_rows,
    loss_metric_mapping_rows,
    network_architecture_catalog,
)


def main() -> None:
    losses = deep_loss_catalog()
    networks = network_architecture_catalog()
    required_losses = {
        "factual_outcome_bce_mse",
        "cfr_representation_balance_mmd",
        "dragonnet_propensity_targeted_regularization",
        "policy_value_incremental_profit_loss",
        "full_funnel_ecup_multitask_loss",
        "delayed_feedback_window_loss",
        "contrastive_uplift_representation_loss",
        "llm_routing_cost_quality_loss",
        "llm_multi_action_routing_loss",
    }
    required_networks = {
        "TARNet",
        "CFRNet",
        "DragonNet",
        "EFIN interaction network",
        "DESCN / ESX entire-space network",
        "ECUP full-funnel multi-task network",
        "LLM routing uplift network",
        "Multi-action LLM routing network",
    }
    loss_ids = {row["loss_id"] for row in losses}
    network_ids = {row["architecture"] for row in networks}
    failures = []
    if missing := sorted(required_losses - loss_ids):
        failures.append(f"missing losses: {missing}")
    if missing := sorted(required_networks - network_ids):
        failures.append(f"missing networks: {missing}")
    for name, rows in {"loss": losses, "network": networks}.items():
        for row in rows:
            required = ["status", "code_path", "source_url", "interview_line"]
            empty = [field for field in required if not str(row.get(field, "")).strip()]
            if empty:
                failures.append(f"{name} row {row}: empty {empty}")
    prompts = {
        "loss": "uplift deep loss 怎么设计，为什么不能只用 BCE？",
        "network": "TARNet CFRNet DragonNet EFIN DESCN 网络结构怎么讲？",
        "contrastive": "对比学习怎么用于 uplift，hard negative 怎么定义？",
        "llm": "LLM routing loss 和 cost-quality objective 怎么做？",
    }
    answers = {}
    for key, prompt in prompts.items():
        answer = deep_design_agent_reply(prompt)
        if not answer or len(answer) < 120:
            failures.append(f"weak agent answer for {key}: {answer}")
            continue
        answers[key] = answer
    if "representation_balance_loss" not in answers.get("loss", ""):
        failures.append("loss answer must cite representation_balance_loss")
    if "Network Architecture Catalog" not in answers.get("network", ""):
        failures.append("network answer must cite Network Architecture Catalog")
    if "Policy -> LLM Routing ROI Simulator" not in answers.get("llm", ""):
        failures.append("llm answer must cite policy simulator")

    out = {
        "status": "fail" if failures else "ok",
        "counts": deep_design_counts(),
        "losses": losses,
        "networks": networks,
        "loss_metric_map_rows": len(loss_metric_mapping_rows()),
        "sources": deep_source_rows(),
        "local_causal_report_rows": local_causal_report_rows(),
        "agent_answer_keys": sorted(answers),
        "failures": failures,
    }
    Path("reports").mkdir(exist_ok=True)
    out_path = Path("reports/deep_loss_architecture_catalog_latest.json")
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": out["status"], "path": str(out_path), "counts": out["counts"], "failures": failures}, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
