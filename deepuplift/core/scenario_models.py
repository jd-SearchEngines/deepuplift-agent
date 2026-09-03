from __future__ import annotations

from typing import Any

from .industry_playbook import business_scenarios
from .model_cards import build_model_card, model_parameter_presets, model_source_and_family, recommended_model_preset
from .registry import MODEL_REGISTRY, missing_dependencies


def model_scenario_tags(model: str) -> list[str]:
    tags: list[str] = []
    lowered = model.lower()
    source, family = model_source_and_family(model)
    for scenario in business_scenarios():
        if model in scenario.get("ready_models", []):
            tags.append(str(scenario["name"]))

    if any(token in lowered for token in ["tlearner", "xlearner", "drlearner", "rlearner", "ggbm", "pav", "efin", "utboost", "mult"]):
        tags.extend(["Coupon / subsidy allocation", "CRM lifecycle intervention"])
    if any(token in lowered for token in ["drlearner", "rlearner", "dml", "causalforest", "forest", "domainadaptation"]):
        tags.extend(["Ads / audience targeting / bidding", "Recommendation intervention"])
    if any(token in lowered for token in ["dose", "multi", "grf", "forest"]):
        tags.extend(["Marketplace pricing / incentive"])
    if any(token in lowered for token in ["delayed", "dragon", "tarnet", "cfr", "desc", "euen", "eeuen"]):
        tags.extend(["Growth / recall / push / SMS", "Recommendation intervention"])
    if any(token in lowered for token in ["drlearner", "rlearner", "dml", "causalforest", "forest", "pav", "ggbm"]):
        tags.extend(["LLM routing / cost-quality escalation"])
    if source in {"EconML", "CausalML", "scikit-uplift", "UTBoost"}:
        tags.append("Open-source benchmark")
    if family in {"industrial_boosting", "industrial_uplift_gbdt", "official_cate", "official_uplift"}:
        tags.append("Industrial tabular")
    return list(dict.fromkeys(tags or ["General uplift modeling"]))


def scenario_model_matrix(descriptions: dict[str, str] | None = None) -> list[dict[str, Any]]:
    descriptions = descriptions or {}
    rows: list[dict[str, Any]] = []
    for model in sorted(MODEL_REGISTRY):
        spec = MODEL_REGISTRY[model]
        card = build_model_card(model, description=descriptions.get(model, ""))
        tags = model_scenario_tags(model)
        rows.append(
            {
                "model": model,
                "status": card.get("status", ""),
                "source": card.get("source", ""),
                "family": card.get("family", ""),
                "stage": card.get("stage", ""),
                "tasks": ", ".join(spec.supported_tasks),
                "scenario_tags": "; ".join(tags),
                "missing": ", ".join(missing_dependencies(model)),
                "recommended_preset": recommended_model_preset(model)[0],
                "preset_count": len(model_parameter_presets(model)),
                "best_for": "; ".join(card.get("best_for", [])),
                "assumptions": "; ".join(card.get("assumptions", [])),
                "risks": "; ".join(card.get("risks", [])),
                "decision_use": card.get("decision_use", ""),
            }
        )
    return rows
