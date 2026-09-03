from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .model_cards import build_model_card


KB_PATH = Path(__file__).resolve().parents[1] / "knowledge" / "causal_agent_kb.json"


@lru_cache(maxsize=1)
def load_knowledge_base() -> Dict[str, Any]:
    with KB_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _get_number(payload: Dict[str, Any], key: str, default: float | None = None) -> float | None:
    value = payload.get(key, default)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _max_abs_balance(diagnostics: Dict[str, Any]) -> float:
    rows = diagnostics.get("feature_balance") or []
    if not rows:
        return 0.0
    return max(float(row.get("abs_imbalance") or 0.0) for row in rows)


def _treatment_rate(diagnostics: Dict[str, Any]) -> float | None:
    rows = diagnostics.get("treatment_distribution") or []
    total = sum(float(row.get("rows") or 0.0) for row in rows)
    if total <= 0:
        return None
    for row in rows:
        value = row.get("value")
        if str(value) in {"1", "1.0", "True", "true"}:
            return float(row.get("rows") or 0.0) / total
    return None


def _rule_matches(rule: Dict[str, Any], diagnostics: Dict[str, Any]) -> bool:
    trigger = rule.get("trigger") or {}
    design = diagnostics.get("design") or {}
    overlap = diagnostics.get("overlap") or {}
    treatment_type = design.get("treatment_type")
    if trigger.get("treatment_type") and trigger["treatment_type"] != treatment_type:
        return False

    propensity_auc = _get_number(overlap, "treatment_auc")
    weak_overlap_rate = _get_number(overlap, "weak_overlap_rate", 0.0) or 0.0
    max_abs_balance = _max_abs_balance(diagnostics)
    treatment_rate = _treatment_rate(diagnostics)

    if "min_propensity_auc" in trigger and (propensity_auc is None or propensity_auc < trigger["min_propensity_auc"]):
        return False
    if "max_propensity_auc" in trigger and (propensity_auc is not None and propensity_auc > trigger["max_propensity_auc"]):
        return False
    if "min_weak_overlap_rate" in trigger and weak_overlap_rate < trigger["min_weak_overlap_rate"]:
        return False
    if "min_abs_balance" in trigger and max_abs_balance < trigger["min_abs_balance"]:
        return False
    if "max_abs_balance" in trigger and max_abs_balance > trigger["max_abs_balance"]:
        return False
    if treatment_rate is not None:
        if "min_treatment_rate" in trigger and treatment_rate < trigger["min_treatment_rate"]:
            return False
        if "max_treatment_rate" in trigger and treatment_rate > trigger["max_treatment_rate"]:
            return False
    elif "min_treatment_rate" in trigger or "max_treatment_rate" in trigger:
        return False
    return True


def matched_knowledge(diagnostics: Dict[str, Any], task: str | None = None) -> List[Dict[str, Any]]:
    kb = load_knowledge_base()
    references = kb.get("references", {})
    matches = []
    for rule in kb.get("diagnostic_rules", []):
        if not _rule_matches(rule, diagnostics):
            continue
        item = dict(rule)
        item["reference_details"] = [references[ref] for ref in rule.get("references", []) if ref in references]
        matches.append(item)
    return matches


def model_card(model_name: str) -> Dict[str, Any]:
    knowledge_cards = load_knowledge_base().get("model_cards", {})
    return build_model_card(model_name, knowledge_card=knowledge_cards.get(model_name))


def knowledge_model_recommendations(
    diagnostics: Dict[str, Any],
    available: Iterable[str],
    limit: int = 8,
) -> List[str]:
    available_set = set(available)
    recommendations = []
    for item in matched_knowledge(diagnostics):
        for model in item.get("models", []):
            if model in available_set and model not in recommendations:
                recommendations.append(model)
    return recommendations[:limit]


def format_rule_evidence(matches: List[Dict[str, Any]], max_rules: int = 3, max_refs: int = 2) -> str:
    if not matches:
        return "知识库没有触发额外规则。"
    lines = []
    for item in matches[:max_rules]:
        lines.append(f"- {item['message']}")
        refs = item.get("reference_details", [])[:max_refs]
        for ref in refs:
            lines.append(f"  依据：{ref['title']} ({ref['url']})")
    return "\n".join(lines)


def format_model_card(model_name: str) -> str:
    card = model_card(model_name)
    if not card:
        return f"`{model_name}` 暂无知识库卡片。"
    best_for = "、".join(card.get("best_for", [])) or "当前任务"
    risks = "；".join(card.get("risks", [])) or "需要常规验证"
    assumptions = "；".join(card.get("assumptions", [])) or "需要常规因果假设"
    return (
        f"`{model_name}` 属于 `{card.get('source', 'unknown')}` / `{card.get('family', 'unknown')}`，状态 `{card.get('status', 'unknown')}`。\n"
        f"适合：{best_for}。\n"
        f"说明：{card.get('explanation') or card.get('description') or card.get('decision_use', '')}\n"
        f"关键假设：{assumptions}。\n"
        f"风险：{risks}。\n"
        f"使用建议：{card.get('decision_use', '')}"
    )


def research_frontier(limit: int = 8) -> List[Dict[str, Any]]:
    return load_knowledge_base().get("research_frontier", [])[:limit]


def format_research_frontier(limit: int = 6) -> str:
    rows = research_frontier(limit=limit)
    if not rows:
        return "知识库里还没有 research frontier 条目。"
    lines = []
    for row in rows:
        stage = row.get("status", "tracked")
        lines.append(
            f"- `{row.get('id')}` ({stage}): {row.get('business_signal')} "
            f"下一步：{row.get('agent_action')}"
        )
    return "\n".join(lines)


def knowledge_summary() -> Dict[str, Any]:
    kb = load_knowledge_base()
    return {
        "version": kb.get("version"),
        "principles": kb.get("principles", []),
        "diagnostic_rules": kb.get("diagnostic_rules", []),
        "evaluation_rules": kb.get("evaluation_rules", []),
        "model_cards": kb.get("model_cards", {}),
        "research_frontier": kb.get("research_frontier", []),
        "references": kb.get("references", {}),
    }
