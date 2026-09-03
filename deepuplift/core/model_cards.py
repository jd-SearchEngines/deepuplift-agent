from __future__ import annotations

from typing import Any

from .registry import MODEL_REGISTRY, missing_dependencies


SOURCE_LINKS = {
    "DeepUplift": "local://deepuplift/models",
    "LightGBM": "https://lightgbm.readthedocs.io/",
    "XGBoost": "https://xgboost.readthedocs.io/",
    "CatBoost": "https://catboost.ai/docs/",
    "EconML": "https://econml.azurewebsites.net/",
    "CausalML": "https://causalml.readthedocs.io/",
    "scikit-uplift": "https://www.uplift-modeling.com/",
    "UTBoost": "https://github.com/jd-opensource/UTBoost",
}

PRESET_PRIORITY = [
    "LightGBM stable",
    "CatBoost stable",
    "XGBoost guarded",
    "UTBoost guarded",
    "EconML balanced",
    "EconML small",
    "CausalML small",
    "Neural compact",
    "Interaction compact",
    "DESCN compact",
    "Balanced",
    "Fast smoke",
    "Custom / empty",
]


def model_source_and_family(model: str) -> tuple[str, str]:
    if model.startswith("EconML"):
        return "EconML", "official_cate"
    if model.startswith("CausalML"):
        return "CausalML", "official_uplift"
    if model.startswith("SkLift"):
        return "scikit-uplift", "official_uplift"
    if model.startswith("UTBoost"):
        return "UTBoost", "industrial_uplift_gbdt"
    if "CatBoost" in model:
        return "CatBoost", "industrial_boosting"
    if "XGBoost" in model or model.endswith("XGB"):
        return "XGBoost", "industrial_boosting"
    if "LightGBM" in model:
        return "LightGBM", "industrial_boosting"
    if model in {"TarNet", "CFRNet", "ContrastiveUpliftNet", "DragonNet", "DragonDeepFM", "EFIN", "DESCN", "ESX", "EUEN", "EEUEN"}:
        return "DeepUplift", "neural"
    if "PAV" in model or "GGBM" in model:
        return "DeepUplift", "calibrated_ranking"
    if "Forest" in model or model == "CausalForest":
        return "DeepUplift", "forest_style"
    return "DeepUplift", "meta_learner"


def model_readiness(model: str, missing: list[str] | None = None) -> tuple[str, str]:
    missing = missing_dependencies(model) if missing is None else missing
    if not missing:
        return "ready", "当前环境可直接训练、对比、打分和导出。"
    if any("DEEPUPLIFT_ENABLE_XGBOOST" in item or "experimental" in item for item in missing):
        return "guarded", "XGBoost 已注册但保持实验性禁用；验证本机稳定后用 DEEPUPLIFT_ENABLE_XGBOOST=1 启动。"
    if any("python>=3.11" in item for item in missing):
        return "needs python 3.11", "使用 scripts/setup_causalml_py311.sh 创建 Python 3.11 CausalML 环境。"
    if "catboost" in missing:
        return "needs dependency", "安装 catboost 后重启 Streamlit。"
    if "utboost" in missing:
        return "needs dependency", "安装 utboost 后重启 Streamlit；默认保持 guarded，先用 P0 LightGBM/DR/DML 对照。"
    if "sklift" in missing:
        return "needs dependency", "安装 scikit-uplift 后重启 Streamlit。"
    if "lightgbm" in missing:
        return "needs dependency", "安装 lightgbm 后重启 Streamlit。"
    return "needs dependency", "安装缺失依赖后重启 Streamlit。"


def _stage(model: str, family: str) -> str:
    if model.startswith("S") or "Solo" in model:
        return "baseline"
    if model.startswith("T") or "TwoModels" in model:
        return "baseline"
    if model.startswith("X") or "DomainAdaptation" in model:
        return "imbalance"
    if model.startswith("DR") or "DRLearner" in model:
        return "bias_adjusted"
    if model.startswith("R") or "DML" in model or "Orthogonal" in model:
        return "orthogonal"
    if "Forest" in model:
        return "nonlinear_hte"
    if "PAV" in model or "GGBM" in model:
        return "ranking_calibration"
    if family == "neural":
        return "representation"
    return family


def _best_for(model: str, source: str, family: str) -> list[str]:
    items: list[str] = []
    if "LightGBM" in model:
        items.extend(["large tabular data", "industrial batch scoring"])
    if "CatBoost" in model:
        items.extend(["category-heavy tables", "marketing/recommendation features"])
    if "UTBoost" in model:
        items.extend(
            [
                "uplift-oriented GBDT",
                "large tabular marketing data",
                "coupon / CRM / ads tables with strong treatment interactions",
                "P1 plugin after LightGBM DR/R/DML baselines",
            ]
        )
    if "XGBoost" in model:
        items.extend(["strong nonlinear tabular baselines", "guarded runtime experiments"])
    if model.startswith("S") or "Solo" in model:
        items.extend(["quick sanity baseline", "randomized or near-randomized treatment"])
    if model.startswith("T") or "TwoModels" in model:
        items.extend(["balanced treatment/control", "easy outcome-model comparison"])
    if model.startswith("X") or "DomainAdaptation" in model:
        items.extend(["imbalanced treatment/control", "counterfactual outcome modeling"])
    if "DR" in model:
        items.extend(["observational data", "selection-bias adjustment"])
    if model.startswith("R") or "DML" in model or "Orthogonal" in model:
        items.extend(["orthogonalized CATE", "propensity-driven assignment"])
    if "Forest" in model or "CausalForest" in model:
        items.extend(["nonlinear heterogeneous effects", "segment discovery"])
    if "PAV" in model or "GGBM" in model:
        items.extend(["Top-K targeting", "score calibration pressure test"])
    if family == "neural":
        items.extend(["representation learning", "deep uplift baselines"])
    if source in {"EconML", "CausalML", "scikit-uplift"}:
        items.append("open-source framework benchmark")
    return list(dict.fromkeys(items or ["general uplift modeling"]))


def _assumptions(model: str, family: str) -> list[str]:
    items = ["consistent treatment definition", "no post-treatment feature leakage"]
    if "UTBoost" in model:
        items.extend(
            [
                "binary treatment for the current adapter contract",
                "sufficient treatment/control support in each Top-K segment",
                "dependency import and guarded smoke validation before live compare",
            ]
        )
    if "DR" in model or "R" in model or "DML" in model or "CausalForest" in model or "DomainAdaptation" in model:
        items.extend(["conditional ignorability", "overlap/common support", "useful nuisance models"])
    elif model.startswith("S") or model.startswith("T") or model.startswith("X") or "TransformedOutcome" in model:
        items.extend(["randomized or observed-confounder-adjusted assignment", "enough treated and control support"])
    if "IPW" in model:
        items.append("well-calibrated propensity scores")
    if "PAV" in model or "GGBM" in model:
        items.append("holdout data for ranking calibration")
    if family == "neural":
        items.append("enough rows to fit representation layers")
    return list(dict.fromkeys(items))


def _risks(model: str, status: str) -> list[str]:
    items: list[str] = []
    if status != "ready":
        items.append("dependency/runtime is not ready in the current environment")
    if model.startswith("S") or "Solo" in model:
        items.append("can smooth away heterogeneous effects because treatment is only a feature")
    if model.startswith("T") or "TwoModels" in model:
        items.append("separate outcome models amplify noise when one arm is small")
    if model.startswith("X"):
        items.append("depends on reliable first-stage counterfactual outcome models")
    if "DR" in model or model.startswith("R") or "DML" in model:
        items.append("weak overlap can create unstable pseudo outcomes")
    if "IPW" in model:
        items.append("propensity near 0/1 can cause high-variance estimates")
    if "Forest" in model:
        items.append("slower compares and noisier small-leaf segment estimates")
    if "PAV" in model or "GGBM" in model:
        items.append("calibration must be checked on holdout data")
    if "XGBoost" in model:
        items.append("guarded on this machine because local runtime stability must be validated")
    if "CausalML" in model:
        items.append("requires the Python 3.11 CausalML helper environment")
    if "CatBoost" in model:
        items.append("optional native dependency must be installed before use")
    if "UTBoost" in model:
        items.append("guarded plugin; compare against LightGBM DR/R baselines before relying on it")
        items.append("native utboost dependency is not installed in the default demo environment")
        items.append("requires no-UI compare and screenshot/evidence gates before promotion to default-ready")
    return list(dict.fromkeys(items or ["validate QINI, AUUC, Top-K, calibration, and policy value before deployment"]))


def _decision_use(model: str, stage: str) -> str:
    if "UTBoost" in model:
        return (
            "Use it as a P1 guarded uplift-GBDT plugin for large marketing tables after P0 LightGBM "
            "S/T/X/R/DR and causal-forest baselines have passed overlap, calibration, Top-K and policy-value checks."
        )
    if stage == "baseline":
        return "Use it early to establish a fast baseline, then compare against DR/R/forest or calibrated ranking models."
    if stage == "imbalance":
        return "Use it when treatment/control support is uneven, and inspect overlap plus minority-arm Top-K support."
    if stage in {"bias_adjusted", "orthogonal"}:
        return "Use it as a main candidate for observational data, then require sensitivity, overlap, and CI checks."
    if stage == "ranking_calibration":
        return "Use it for Top-K targeting decisions where calibrated ranking and policy value matter more than raw AUUC."
    if stage == "nonlinear_hte":
        return "Use it to stress-test nonlinear heterogeneity and segment-level decisions."
    if stage == "representation":
        return "Use it as a deep representation baseline after simple learners and diagnostics have passed."
    return "Use it as part of model comparison, not as a standalone deployment decision."


def build_model_card(
    model_name: str,
    *,
    description: str = "",
    knowledge_card: dict[str, Any] | None = None,
) -> dict[str, Any]:
    spec = MODEL_REGISTRY.get(model_name)
    missing = missing_dependencies(model_name)
    status, setup_hint = model_readiness(model_name, missing)
    source, family = model_source_and_family(model_name)
    stage = _stage(model_name, family)
    base = {
        "model": model_name,
        "status": status,
        "source": source,
        "family": family,
        "stage": stage,
        "tasks": list(spec.supported_tasks) if spec else [],
        "dependencies": list(spec.dependencies) if spec else [],
        "missing": missing,
        "setup_hint": setup_hint,
        "source_url": SOURCE_LINKS.get(source, ""),
        "description": description,
        "parameter_presets": model_parameter_presets(model_name),
        "best_for": _best_for(model_name, source, family),
        "assumptions": _assumptions(model_name, family),
        "risks": _risks(model_name, status),
        "decision_use": _decision_use(model_name, stage),
    }
    if knowledge_card:
        for key in ["family", "best_for", "risks", "explanation"]:
            if knowledge_card.get(key):
                base[key] = knowledge_card[key]
        base["knowledge_overlay"] = True
    else:
        base["knowledge_overlay"] = False
    if "UTBoost" in model_name:
        base["integration_gate"] = (
            "reports/optional_backend_probe_latest.json; "
            "reports/utboost_guarded_adapter_smoke_latest.json; "
            "scripts/smoke_utboost_guarded_adapter.py; "
            "scripts/audit_model_catalog.py; "
            "scripts/smoke_compare_manifest.py --models UTBoostGBM TLearnerLightGBM DRLearnerLightGBM"
        )
        base["source_evidence"] = (
            "repo=https://github.com/jd-opensource/UTBoost; "
            "registry=deepuplift/core/registry.py:UTBoostGBM; "
            "adapter=deepuplift/core/external_models.py:_fit_utboost/_predict_utboost_outcomes; "
            "smoke=scripts/smoke_utboost_guarded_adapter.py; "
            "evidence=Evidence -> UTBoost Guarded Adapter Smoke"
        )
        base["promotion_rule"] = (
            "Promote from guarded to demo-ready only after utboost imports, tiny training smoke passes, "
            "catalog audit passes, and no-UI compare beats or complements P0 baselines under QINI/AUUC/Top-K/policy value."
        )
    if not base.get("description") and base.get("explanation"):
        base["description"] = str(base["explanation"])
    return base


def build_model_cards(descriptions: dict[str, str] | None = None, knowledge_cards: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    descriptions = descriptions or {}
    knowledge_cards = knowledge_cards or {}
    return [
        build_model_card(
            name,
            description=descriptions.get(name, ""),
            knowledge_card=knowledge_cards.get(name),
        )
        for name in sorted(MODEL_REGISTRY)
    ]


def model_parameter_presets(model_name: str) -> dict[str, dict[str, Any]]:
    """Small, conservative presets for local compare runs."""
    presets: dict[str, dict[str, Any]] = {
        "Custom / empty": {},
        "Fast smoke": {"max_iter": 30, "n_estimators": 40, "max_depth": 3},
        "Balanced": {"max_iter": 60, "n_estimators": 80, "max_depth": 4},
    }
    if "LightGBM" in model_name:
        presets["LightGBM stable"] = {
            "n_estimators": 80,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 20,
            "n_jobs": 1,
        }
    if "CatBoost" in model_name:
        presets["CatBoost stable"] = {
            "iterations": 80,
            "learning_rate": 0.05,
            "depth": 4,
            "verbose": False,
            "allow_writing_files": False,
        }
    if "UTBoost" in model_name:
        presets["UTBoost guarded"] = {
            "iterations": 80,
            "learning_rate": 0.05,
            "max_depth": 4,
            "criterion": "gbm",
            "ensemble_type": "boosting",
        }
    if "XGBoost" in model_name:
        presets["XGBoost guarded"] = {
            "n_estimators": 60,
            "max_depth": 3,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "n_jobs": 1,
        }
    if model_name.startswith("EconML"):
        presets["EconML small"] = {
            "n_estimators": 40,
            "min_samples_leaf": 10,
            "cv": 2,
            "max_depth": 6,
        }
        presets["EconML balanced"] = {
            "n_estimators": 80,
            "min_samples_leaf": 10,
            "cv": 3,
            "max_depth": 8,
        }
    if model_name.startswith("CausalML"):
        presets["CausalML small"] = {
            "n_estimators": 40,
            "max_depth": 4,
            "min_samples_leaf": 20,
        }
    if model_name in {"TarNet", "CFRNet", "ContrastiveUpliftNet", "DragonNet", "DragonDeepFM"}:
        presets["Neural compact"] = {
            "share_dim": 16,
            "share_hidden_dims": [64, 64],
            "base_hidden_dims": [64],
        }
    if model_name == "ContrastiveUpliftNet":
        presets["Contrastive compact"] = {
            "share_dim": 16,
            "share_hidden_dims": [64, 64],
            "base_hidden_dims": [64],
            "lambda_contrastive": 0.03,
            "temperature": 0.2,
            "propensity": 0.5,
        }
    if model_name == "CFRNet":
        presets["CFRNet differentiable balance"] = {
            "share_dim": 16,
            "share_hidden_dims": [64, 64],
            "base_hidden_dims": [64],
            "alpha": 0.2,
            "ipm_mode": "mmd_rbf",
        }
    if model_name in {"EFIN", "EUEN", "EEUEN"}:
        presets["Interaction compact"] = {
            "hc_dim": 32,
            "hu_dim": 16,
        }
    if model_name in {"DESCN", "ESX"}:
        presets["DESCN compact"] = {
            "share_dim": 12,
            "base_dim": 12,
        }
    return presets


def recommended_model_preset(model_name: str) -> tuple[str, dict[str, Any]]:
    presets = model_parameter_presets(model_name)
    preset_name = next((name for name in PRESET_PRIORITY if name in presets), next(iter(presets), "Custom / empty"))
    return preset_name, dict(presets.get(preset_name, {}))
