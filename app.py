from __future__ import annotations

import inspect
import json
import html
import io
import math
import os
import re
import subprocess
import time
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st


def install_streamlit_width_compat() -> None:
    try:
        from streamlit.delta_generator import DeltaGenerator
    except Exception:
        return

    def patch_method(method_name: str, *, image_width: bool = False) -> None:
        original = getattr(DeltaGenerator, method_name, None)
        if original is None:
            return
        if getattr(original, "_deepuplift_width_compat", False):
            if hasattr(st, "_main"):
                setattr(st, method_name, getattr(st._main, method_name))
            return
        supported_params = set(inspect.signature(original).parameters)

        def normalize_width(kwargs: dict) -> dict:
            if kwargs.get("width") == "stretch":
                kwargs.pop("width", None)
                if image_width and "use_container_width" in supported_params:
                    kwargs.setdefault("use_container_width", True)
                elif image_width and "use_column_width" in supported_params:
                    kwargs.setdefault("use_column_width", True)
                elif "use_container_width" in supported_params:
                    kwargs.setdefault("use_container_width", True)
            return kwargs

        def wrapper(self, *args, **kwargs):
            kwargs = normalize_width(kwargs)
            return original(self, *args, **kwargs)

        wrapper._deepuplift_width_compat = True
        setattr(DeltaGenerator, method_name, wrapper)
        if hasattr(st, "_main"):
            setattr(st, method_name, getattr(st._main, method_name))

    for name in ["dataframe", "line_chart", "bar_chart", "area_chart", "scatter_chart", "graphviz_chart"]:
        patch_method(name)
    patch_method("image", image_width=True)


install_streamlit_width_compat()

from deepuplift.core import UpliftConfig, score_uplift_run, train_uplift_model
from deepuplift.core.artifacts import save_json, save_text
from deepuplift.core.diagnostics import diagnose_uplift_data
from deepuplift.core.doseresponse import score_dose_response_run, train_dose_response_model
from deepuplift.core.evaluator import frontier_metric_summary, industrial_policy_metrics
from deepuplift.core.knowledge import (
    format_model_card,
    format_research_frontier,
    format_rule_evidence,
    knowledge_model_recommendations,
    knowledge_summary,
    matched_knowledge,
)
from deepuplift.core.interview_agent import (
    INTERVIEW_AGENT_QUICK_PROMPTS,
    interview_agent_reply,
    interview_agent_topics,
)
from deepuplift.core.industry_playbook import (
    INDUSTRY_INTERVIEW_PROMPTS,
    business_scenarios,
    industry_agent_reply,
    industry_references,
    llm_interview_qa,
    policy_templates,
    scenario_model_rows,
)
from deepuplift.core.open_source_frameworks import (
    framework_status_summary,
    open_source_framework_agent_reply,
    open_source_framework_counts,
    open_source_framework_rows,
)
from deepuplift.core.evaluation_formulas import (
    evaluation_formula_agent_reply,
    evaluation_formula_counts,
    evaluation_formula_rows,
    optimization_path_rows,
    policy_derivation_rows,
)
from deepuplift.core.failure_modes import (
    diagnostic_case_rows,
    diagnostic_case_summary_rows,
    failure_mode_agent_reply,
    failure_mode_counts,
    model_failure_mode_rows,
    scenario_pitfall_rows,
)
from deepuplift.core.frontier_models import (
    capability_counts,
    frontier_demo_trace_rows,
    frontier_agent_reply,
    frontier_model_capabilities,
    frontier_source_rows,
    scenario_model_ladders,
)
from deepuplift.core.glossary import glossary_context_names, glossary_rows
from deepuplift.core.deep_uplift_designs import (
    deep_design_agent_reply,
    deep_design_counts,
    deep_interview_drill_rows,
    deep_loss_catalog,
    deep_source_rows,
    local_causal_report_rows,
    loss_metric_mapping_rows,
    network_architecture_catalog,
)
from deepuplift.core.industrial_scenario_lab import (
    scenario_agent_reply,
    scenario_cards as industrial_scenario_cards,
    scenario_dataset_summary,
)
from deepuplift.core.resume_scenario_policy import didi_budget_allocation_plan, resume_scenario_policy_benchmarks
from deepuplift.core.resume_scenario_workbench import (
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
from deepuplift.core.multitreatment import score_multi_treatment_run, train_multi_treatment_model
from deepuplift.core.policy import (
    default_policy_fractions,
    exploration_ope_guardrail_rows,
    llm_routing_action_coverage_rows,
    llm_routing_bandit_drift_rows,
    llm_routing_ope_summary,
    policy_value_curve,
)
from deepuplift.core.predictor import score_top_fraction
from deepuplift.core.readiness import decision_readiness
from deepuplift.core.registry import MODEL_REGISTRY, available_models, is_model_available, missing_dependencies
from deepuplift.core.scenario_models import model_scenario_tags as core_model_scenario_tags
from deepuplift.core.model_cards import (
    build_model_card,
    model_parameter_presets,
    recommended_model_preset,
    model_readiness as catalog_model_readiness,
    model_source_and_family as catalog_source_and_family,
)
from deepuplift.core.model_deconstruction import (
    MODEL_DECONSTRUCTIONS,
    MODEL_DECONSTRUCTION_GOLDEN_QUESTIONS,
    detect_deconstruction_model,
    get_model_deconstruction,
    model_deconstruction_reply,
)
from deepuplift.core.frontend_research import (
    agent_prompt_groups,
    frontend_agent_reply,
    frontend_framework_audit_rows,
    frontend_research_counts,
    ui_inspiration_matrix_rows,
    visual_regression_gate_rows,
    visual_upgrade_rows,
)
from deepuplift.core.ml_training_platform_research import (
    ml_training_inspiration_matrix_rows,
    ml_training_platform_agent_reply,
    ml_training_platform_audit_rows,
    ml_training_platform_counts,
    ml_training_prompt_groups,
    training_workbench_stage_rows,
    uplift_application_expansion_rows,
)
from deepuplift.core.public_product_inspiration import (
    growth_control_plane_rows,
    launch_guardrail_rows,
    public_product_agent_reply,
    public_product_audit_rows,
    public_product_counts,
    public_product_inspiration_rows,
    public_product_prompt_groups,
)


SAMPLE_PATH = Path("deepuplift/dataset/criteo-uplift-v2.1-unbiased-sample50w.csv")
DATASET_MANIFEST = Path("examples/datasets/manifest.json")
RUN_EVIDENCE_FILES = [
    ("config.json", "training config"),
    ("metrics.json", "evaluation metrics"),
    ("readiness.json", "decision readiness"),
    ("predictions.csv", "scored holdout rows"),
    ("run_note.md", "human-readable report"),
    ("user_note.json", "tags and operator note"),
]

MODEL_DESCRIPTIONS = {
    "SLearnerGBM": "S-Learner：把 treatment 当作特征输入同一个 GBM outcome 模型，速度快，适合作为第一条非深度基线。",
    "SLearnerRF": "S-Learner 随机森林版：适合非线性强的数据，作为稳健但偏保守的 outcome baseline。",
    "SLearnerLightGBM": "S-Learner LightGBM：工业常用 boosting 基线，适合大规模表格数据快速建模。",
    "SLearnerXGBoost": "S-Learner XGBoost：工业常用 boosting 基线，适合表格数据和强非线性关系。",
    "TLearnerGBM": "T-Learner：分别训练 treatment/control 两个 GBM outcome 模型，直观稳健，适合随机实验数据。",
    "TLearnerRF": "T-Learner 随机森林版：分别训练 treatment/control 森林模型，适合非线性和特征交互明显的数据。",
    "TLearnerLightGBM": "T-Learner LightGBM：分别训练 treatment/control 的 LightGBM outcome 模型，适合大样本工业投放数据。",
    "TLearnerXGBoost": "T-Learner XGBoost：分别训练 treatment/control 的 XGBoost outcome 模型，适合强特征交互数据。",
    "XLearnerGBM": "X-Learner：先估计反事实 outcome，再学习 CATE，适合 treatment/control 样本量不均衡的场景。",
    "XLearnerForest": "X-Learner 森林版：用森林学习 imputed treatment effect，适合样本不均衡且异质性非线性的场景。",
    "XLearnerLightGBM": "X-Learner LightGBM：LightGBM 反事实 outcome + effect stage，适合样本不均衡的大规模二元 treatment。",
    "XLearnerXGBoost": "X-Learner XGBoost：XGBoost 反事实 outcome + effect stage，适合非线性且 treatment/control 不均衡的数据。",
    "DRLearnerGBM": "Doubly Robust learner：结合 outcome 模型和 propensity 模型，适合观测数据或存在分配偏差的数据。",
    "DRLearnerForest": "Doubly Robust learner 森林版：用森林学习 DR pseudo outcome，适合非线性异质效应。",
    "DRLearnerLightGBM": "Doubly Robust LightGBM：LightGBM 学习 DR pseudo outcome，适合有选择偏差的工业表格数据。",
    "DRLearnerXGBoost": "Doubly Robust XGBoost：XGBoost 学习 DR pseudo outcome，适合 selection bias 和非线性异质效应。",
    "DRForest": "Forest 版 DR learner：用随机森林学习 CATE，适合非线性强、需要更稳健排序的中小数据。",
    "RLearnerGBM": "R-Learner：用 outcome 与 propensity 残差学习 CATE，适合观测数据下的正交化估计。",
    "RLearnerForest": "R-Learner 森林版：残差化后用随机森林学习 CATE，适合非线性 selection bias。",
    "RLearnerLightGBM": "R-Learner LightGBM：残差化信号 + LightGBM final stage，适合高维稀疏和选择偏差场景。",
    "RLearnerXGBoost": "R-Learner XGBoost：残差化信号 + XGBoost final stage，适合强非线性观测数据。",
    "OrthogonalDMLGBM": "Orthogonal/DML 风格 learner：残差化 outcome 与 treatment 后学习 uplift，适合降低 nuisance 模型误差影响。",
    "OrthogonalRandomForest": "Orthogonal Random Forest 风格：R/DML 残差信号 + 森林 final stage，适合稳健排序验证。",
    "CausalForest": "CausalForest 风格 learner：用森林学习正交伪 outcome 的异质效应，适合非线性强且需要稳定排序的表格数据。",
    "TransformedOutcomeGBM": "Transformed Outcome learner：随机实验下快速学习 uplift pseudo outcome，适合 A/B 测试基线。",
    "TransformedOutcomeForest": "Transformed Outcome 森林版：用森林学习 transformed outcome，适合随机实验中的非线性 uplift。",
    "TransformedOutcomeLightGBM": "Transformed Outcome LightGBM：LightGBM 版 transformed outcome，适合随机实验的快速大样本 ranking baseline。",
    "TransformedOutcomeXGBoost": "Transformed Outcome XGBoost：XGBoost 版 transformed outcome，适合随机实验中的非线性 ranking baseline。",
    "ClassVariableTransformGBM": "Class-variable-transform uplift：把二元 outcome 转成 uplift 分类信号，适合随机/近似随机的 0/1 转化场景。",
    "ClassVariableTransformForest": "Class-variable-transform 森林版：非线性分类 uplift 基线，适合和 Transformed Outcome 做对照。",
    "PAVCalibratedClassVariableTransformGBM": "PAV-Calibrated class-variable-transform：CVT uplift + 保序校准，适合 Top-K 排序压力测试。",
    "IPWLearnerGBM": "IPW learner：用 propensity 加权 pseudo outcome 学习效应，适合做偏差校正基线。",
    "IPWForest": "IPW learner 森林版：propensity 加权 + 随机森林，适合非线性但需要注意高方差。",
    "IPWLightGBM": "IPW LightGBM：propensity 加权 pseudo outcome + LightGBM，适合做工业偏差校正对照。",
    "IPWXGBoost": "IPW XGBoost：propensity 加权 pseudo outcome + XGBoost，适合非线性偏差校正对照。",
    "DomainAdaptationLearner": "Domain Adaptation Learner：X-Learner 变体，用 propensity 权重修正 treatment/control 分布差异。",
    "DomainAdaptationLightGBM": "Domain Adaptation LightGBM：LightGBM outcome model + 分布适配权重，适合 treatment/control 明显不均衡的表格数据。",
    "DomainAdaptationXGBoost": "Domain Adaptation XGBoost：XGBoost outcome model + 分布适配权重，适合强非线性分布偏移。",
    "GGBMUpliftGBM": "GGBM 风格 uplift：DR pseudo effect + GBM final stage + PAV 保序校准，适合增长/营销 Top-K 排序。",
    "GGBMUpliftLightGBM": "GGBM LightGBM：DR pseudo effect + LightGBM effect stage + PAV 保序校准，适合工业级 Top-K 排序。",
    "GGBMUpliftXGBoost": "GGBM XGBoost：DR pseudo effect + XGBoost effect stage + PAV 保序校准，适合工业 Top-K 排序验证。",
    "PAVCalibratedDRLearnerGBM": "PAV-Calibrated DR learner：对 DR uplift 分数做保序校准，适合提升 Top-K 排序可靠性。",
    "PAVCalibratedDRLearnerLightGBM": "PAV-Calibrated DR LightGBM：DR LightGBM + 保序校准，适合投放名单排序稳定性验证。",
    "PAVCalibratedDRLearnerXGBoost": "PAV-Calibrated DR XGBoost：DR XGBoost + 保序校准，适合非线性 Top-K 排序稳定性验证。",
    "PAVCalibratedRLearnerGBM": "PAV-Calibrated R learner：对 R-Learner 残差化 uplift 做保序校准，适合观测数据排序验证。",
    "PAVCalibratedRLearnerLightGBM": "PAV-Calibrated R LightGBM：R-Learner LightGBM + 保序校准，适合选择偏差场景的 Top-K ranking。",
    "PAVCalibratedRLearnerXGBoost": "PAV-Calibrated R XGBoost：R-Learner XGBoost + 保序校准，适合强非线性观测数据。",
    "PAVCalibratedCausalForest": "PAV-Calibrated causal forest：森林 CATE + 保序校准，适合非线性异质性和分群策略。",
    "TarNet": "稳健基线模型，适合先跑通流程，也适合中小数据的快速对照。",
    "CFRNet": "在 TarNet 基础上加入分布平衡约束，适合 treatment/control 特征分布偏差明显的数据。",
    "ContrastiveUpliftNet": "TarNet + supervised contrastive uplift representation，适合探索 Top-K 排序和分群稳定性的深度 baseline。",
    "DragonNet": "同时学习 outcome 和 propensity，常作为二元 treatment uplift 的强基线。",
    "DragonDeepFM": "适合包含离散特征和连续特征交互的数据，但参数需要更认真配置。",
    "EFIN": "强调特征交互的 uplift 网络，适合营销/推荐类表格数据。",
    "DESCN": "偏 classification 的 uplift 模型，适合转化/访问这类 0/1 outcome。",
    "ESX": "DESCN 的实现别名，偏 classification。",
    "EUEN": "只支持 regression outcome，适合连续数值收益建模。",
    "EEUEN": "带 exposure 建模思路的扩展模型，适合有曝光偏差假设的场景。",
    "EconMLCausalForestDML": "EconML 官方 CausalForestDML：正交化 causal forest，适合观测数据和非线性 CATE，需要安装 econml。",
    "EconMLLinearDML": "EconML 官方 LinearDML：可解释的 DML/orthogonal learner，需要安装 econml。",
    "EconMLSparseLinearDML": "EconML 官方 SparseLinearDML：高维稀疏特征下的线性 DML，需要安装 econml。",
    "EconMLKernelDML": "EconML 官方 KernelDML：核特征 DML，适合非线性但样本量中等的 CATE 验证。",
    "EconMLNonParamDML": "EconML 官方 NonParamDML：通用非参数 final stage 的 DML，需要安装 econml。",
    "EconMLDRLearner": "EconML 官方 DRLearner：doubly robust CATE learner，需要安装 econml。",
    "EconMLLinearDRLearner": "EconML 官方 LinearDRLearner：可解释的 doubly robust CATE learner。",
    "EconMLSparseLinearDRLearner": "EconML 官方 SparseLinearDRLearner：高维稀疏场景的 doubly robust learner。",
    "EconMLForestDRLearner": "EconML 官方 ForestDRLearner：森林版 doubly robust learner，需要安装 econml。",
    "EconMLDMLOrthoForest": "EconML 官方 DMLOrthoForest：正交森林 CATE estimator，适合非线性异质效应验证。",
    "EconMLDROrthoForest": "EconML 官方 DROrthoForest：DR 正交森林，适合观测数据和稳健性对照。",
    "EconMLGRFCausalForest": "EconML 官方 GRF CausalForest：generalized random forest 风格 CATE estimator。",
    "CausalMLUpliftTree": "CausalML 官方 UpliftTreeClassifier：classification uplift tree，需要 Python>=3.11 与 causalml。",
    "CausalMLUpliftRandomForest": "CausalML 官方 UpliftRandomForestClassifier：classification uplift forest，需要 Python>=3.11 与 causalml。",
    "CausalMLCausalRandomForest": "CausalML 官方 CausalRandomForestRegressor：多 treatment causal forest 支持，需要 Python>=3.11 与 causalml。",
    "MultiTLearnerGBM": "Multi-treatment T-Learner：每个 treatment action 一个 GBM outcome 模型，输出 action-specific uplift 和推荐动作。",
    "MultiTLearnerRF": "Multi-treatment T-Learner 随机森林版：每个 action 一个森林 outcome 模型，适合非线性多动作策略。",
    "MultiDRLearnerGBM": "Multi-treatment DR-Learner：结合 action outcome models 与 multi-class propensity，学习 action-vs-control doubly robust uplift。",
    "MultiDRLearnerRF": "Multi-treatment DR-Learner 随机森林版：森林 effect stage 学习多动作 DR pseudo effect，适合非线性多动作策略。",
    "DoseResponseGBM": "连续 treatment 剂量响应模型：GBM outcome model + dose grid，输出推荐剂量和剂量响应曲线。",
    "DoseResponseRF": "连续 treatment 剂量响应森林版：适合非线性剂量响应和稳健策略探索。",
}

MODEL_DESCRIPTIONS.update(
    {
        "SLearnerCatBoost": "S-Learner CatBoost：适合类别特征较多的工业表格数据，安装 catboost 后可用。",
        "TLearnerCatBoost": "T-Learner CatBoost：分别训练 treatment/control 的 CatBoost outcome 模型，适合营销和推荐表格数据。",
        "XLearnerCatBoost": "X-Learner CatBoost：CatBoost 反事实 outcome + effect stage，适合样本不均衡与类别特征场景。",
        "DRLearnerCatBoost": "Doubly Robust CatBoost：CatBoost 学习 DR pseudo outcome，适合观测数据下的稳健 uplift 排序。",
        "RLearnerCatBoost": "R-Learner CatBoost：正交残差信号 + CatBoost final stage，适合 selection bias 和高维类别特征。",
        "TransformedOutcomeCatBoost": "Transformed Outcome CatBoost：随机实验下的 CatBoost uplift pseudo outcome baseline。",
        "IPWCatBoost": "IPW CatBoost：propensity 加权 pseudo outcome + CatBoost，适合偏差校正对照。",
        "DomainAdaptationCatBoost": "Domain Adaptation CatBoost：用 propensity 权重修正 treatment/control 分布偏移。",
        "GGBMUpliftCatBoost": "GGBM CatBoost：DR pseudo effect + CatBoost effect stage + PAV 保序校准，适合 Top-K 排序验证。",
        "PAVCalibratedDRLearnerCatBoost": "PAV-Calibrated DR CatBoost：DR CatBoost uplift 分数保序校准，适合投放名单稳定性验证。",
        "PAVCalibratedRLearnerCatBoost": "PAV-Calibrated R CatBoost：R-Learner CatBoost + 保序校准，适合观测数据 Top-K ranking。",
        "SkLiftSoloGBM": "scikit-uplift SoloModel：官方 S-Learner/dummy treatment 基线，使用 GBM outcome 模型。",
        "SkLiftSoloLightGBM": "scikit-uplift SoloModel + LightGBM：官方 S-Learner 接口，适合大样本快速 baseline。",
        "SkLiftSoloInteractionGBM": "scikit-uplift SoloModel interaction：显式 treatment-feature 交互的单模型 uplift baseline。",
        "SkLiftSoloInteractionLightGBM": "scikit-uplift SoloModel interaction + LightGBM：适合强 treatment-feature 交互的工业 baseline。",
        "SkLiftTwoModelsGBM": "scikit-uplift TwoModels：官方双模型 uplift baseline，分别拟合 treatment/control outcome。",
        "SkLiftTwoModelsLightGBM": "scikit-uplift TwoModels + LightGBM：LightGBM 双模型版本，适合大规模二元 treatment。",
        "SkLiftTwoModelsDDRControlGBM": "scikit-uplift DDR-control：dependent data representation，从 control 模型输出增强 treatment 模型。",
        "SkLiftTwoModelsDDRControlLightGBM": "scikit-uplift DDR-control + LightGBM：适合 control 表征对 treatment 预测有帮助的场景。",
        "SkLiftTwoModelsDDRTreatmentGBM": "scikit-uplift DDR-treatment：dependent data representation，从 treatment 模型输出增强 control 模型。",
        "SkLiftTwoModelsDDRTreatmentLightGBM": "scikit-uplift DDR-treatment + LightGBM：适合 treatment 表征能修正 control 预测的场景。",
        "SkLiftClassTransformationGBM": "scikit-uplift ClassTransformation：官方 class-variable transform，只支持二元 classification outcome。",
        "SkLiftClassTransformationLightGBM": "scikit-uplift ClassTransformation + LightGBM：LightGBM 版 CVT，适合随机实验中的转化 uplift。",
        "SkLiftTransformedOutcomeGBM": "scikit-uplift ClassTransformationReg：官方 transformed outcome/CATE transformation baseline。",
        "SkLiftTransformedOutcomeLightGBM": "scikit-uplift ClassTransformationReg + LightGBM：LightGBM 版 transformed outcome，适合快速 ranking baseline。",
        "CausalMLSLearnerLightGBM": "CausalML 官方 BaseS learner + LightGBM：Python 3.11 CausalML 环境中可运行。",
        "CausalMLTLearnerLightGBM": "CausalML 官方 BaseT learner + LightGBM：官方 meta-learner 双模型基线。",
        "CausalMLXLearnerLightGBM": "CausalML 官方 BaseX learner + LightGBM：适合 treatment/control 不均衡的官方 X-Learner 对照。",
        "CausalMLRLearnerLightGBM": "CausalML 官方 BaseR learner + LightGBM：官方 R-Learner/残差化 CATE 对照。",
        "UTBoostGBM": "UTBoost guarded plugin：uplift-oriented GBDT，适合大规模表格营销数据；安装 utboost 后可启用，默认先与 LightGBM DR/R/DML 基线对照。",
    }
)

MULTI_MODEL_FAMILIES = {
    "MultiTLearnerGBM": "gbm",
    "MultiTLearnerRF": "forest",
    "MultiDRLearnerGBM": "dr_gbm",
    "MultiDRLearnerRF": "dr_forest",
}


def model_catalog_audit_payload(catalog: pd.DataFrame) -> dict:
    registry_models = set(MODEL_REGISTRY)
    extra_described_models = set(MULTI_MODEL_FAMILIES) | {"DoseResponseGBM", "DoseResponseRF"}
    description_models = set(MODEL_DESCRIPTIONS)
    missing_descriptions = sorted(name for name in registry_models if not MODEL_DESCRIPTIONS.get(name, "").strip())
    missing_model_cards = []
    missing_model_presets = []
    for name in sorted(registry_models):
        card = build_model_card(name, description=MODEL_DESCRIPTIONS.get(name, ""))
        if not (card.get("best_for") and card.get("assumptions") and card.get("risks") and card.get("decision_use")):
            missing_model_cards.append(name)
        presets = model_parameter_presets(name)
        try:
            json.dumps(presets, ensure_ascii=False, allow_nan=False)
            preset_ok = bool(presets)
        except (TypeError, ValueError):
            preset_ok = False
        if not preset_ok:
            missing_model_presets.append(name)
    extra_descriptions = sorted(description_models - registry_models - extra_described_models)
    status_counts = catalog["status"].value_counts().to_dict() if not catalog.empty else {}
    source_counts = catalog["source"].value_counts().to_dict() if not catalog.empty else {}
    ready_source_counts = catalog[catalog["status"] == "ready"]["source"].value_counts().to_dict() if not catalog.empty else {}
    ready_family_counts = catalog[catalog["status"] == "ready"]["family"].value_counts().to_dict() if not catalog.empty else {}
    ready_models = int(status_counts.get("ready", 0))
    checks = [
        {
            "check": "Registered coverage",
            "status": "pass" if len(registry_models) == len(catalog) else "warn",
            "detail": f"{len(catalog)} catalog rows for {len(registry_models)} registry models",
        },
        {
            "check": "Descriptions",
            "status": "pass" if not missing_descriptions else "warn",
            "detail": "all registered models described"
            if not missing_descriptions
            else f"{len(missing_descriptions)} models need descriptions",
        },
        {
            "check": "Runnable backends",
            "status": "pass" if ready_models > 0 else "blocker",
            "detail": f"{ready_models} ready models in this environment",
        },
        {
            "check": "Model cards",
            "status": "pass" if not missing_model_cards else "warn",
            "detail": "all registered models have generated model cards"
            if not missing_model_cards
            else f"{len(missing_model_cards)} models need generated cards",
        },
        {
            "check": "Parameter presets",
            "status": "pass" if not missing_model_presets else "warn",
            "detail": "all registered models have serializable parameter presets"
            if not missing_model_presets
            else f"{len(missing_model_presets)} models need presets",
        },
        {
            "check": "Optional backend signals",
            "status": "pass" if any(status != "ready" for status in status_counts) else "pass",
            "detail": "guarded/optional models stay visible with setup hints",
        },
        {
            "check": "Stale descriptions",
            "status": "pass" if not extra_descriptions else "warn",
            "detail": "no unused registered-model descriptions"
            if not extra_descriptions
            else f"{len(extra_descriptions)} descriptions are outside model registries",
        },
    ]
    recommendation = "Catalog is audit-ready."
    if missing_descriptions:
        recommendation = "Add descriptions before shipping new registry models to users."
    elif ready_models == 0:
        recommendation = "Install at least one runnable backend before testing workflows."
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "registered_models": len(registry_models),
        "catalog_rows": len(catalog),
        "ready_models": ready_models,
        "status_counts": status_counts,
        "source_counts": source_counts,
        "ready_source_counts": ready_source_counts,
        "ready_family_counts": ready_family_counts,
        "missing_descriptions": missing_descriptions,
        "missing_model_cards": missing_model_cards,
        "missing_model_presets": missing_model_presets,
        "extra_descriptions": extra_descriptions,
        "checks": checks,
        "recommendation": recommendation,
    }


def latest_report_artifacts() -> dict:
    patterns = {
        "model_audit": "model_catalog_audit_*.json",
        "source_readiness_csv": "model_source_readiness_*.csv",
        "family_readiness_csv": "model_family_readiness_*.csv",
        "agent_regression": "agent_regression_*.json",
        "no_ui_compare": "no_ui_compare_manifest_*.json",
        "compare_bundle": "*compare_bundle_*.json",
        "interview_materials_audit": "interview_materials_audit*.json",
        "industry_playbook_audit": "industry_playbook_audit*.json",
        "industry_source_audit": "industry_source_audit*.json",
        "scenario_model_matrix": "scenario_model_matrix_latest.csv",
        "scenario_model_matrix_audit": "scenario_model_matrix_audit*.json",
        "github_framework_audit": "github_framework_audit_latest.json",
        "github_framework_audit_csv": "github_framework_audit_latest.csv",
        "llm_routing_dataset_smoke": "llm_routing_dataset_smoke_latest.json",
        "llm_routing_policy_smoke": "llm_routing_policy_smoke_latest.json",
        "llm_routing_ope_smoke": "llm_routing_ope_smoke_latest.json",
        "llm_routing_bandit_drift_csv": "llm_routing_bandit_drift_latest.csv",
        "continuous_treatment_policy_smoke": "continuous_treatment_policy_smoke_latest.json",
        "optional_backend_probe": "optional_backend_probe_latest.json",
        "utboost_guarded_adapter_smoke": "utboost_guarded_adapter_smoke_latest.json",
        "frontier_synthetic_datasets_smoke": "frontier_synthetic_datasets_smoke_latest.json",
        "frontier_evaluator_metrics_smoke": "frontier_evaluator_metrics_smoke_latest.json",
        "industrial_scenario_datasets_smoke": "industrial_scenario_datasets_smoke_latest.json",
        "industrial_scenario_training_smoke": "industrial_scenario_training_smoke_latest.json",
        "industrial_policy_metrics_smoke": "industrial_policy_metrics_smoke_latest.json",
        "resume_scenario_policy_smoke": "resume_scenario_policy_smoke_latest.json",
        "resume_scenario_workbench_smoke": "resume_scenario_workbench_smoke_latest.json",
        "industrial_scenario_compare_smoke": "industrial_scenario_compare_smoke_latest.json",
        "industrial_scenario_compare_csv": "industrial_scenario_compare_latest.csv",
        "demo_preset_source_trace": "demo_preset_source_trace_latest.json",
        "frontier_training_evidence": "frontier_training_evidence_latest.json",
        "deep_loss_architecture_catalog": "deep_loss_architecture_catalog_latest.json",
        "deep_loss_functions_smoke": "deep_loss_functions_smoke_latest.json",
        "cfrnet_differentiable_balance_smoke": "cfrnet_differentiable_balance_smoke_latest.json",
        "cfrnet_training_balance_smoke": "cfrnet_training_balance_smoke_latest.json",
        "contrastive_uplift_training_smoke": "contrastive_uplift_training_smoke_latest.json",
        "deep_neural_ablation": "deep_neural_ablation_latest.json",
        "deep_model_training_evidence": "deep_model_training_evidence_latest.json",
        "deep_model_training_leaderboard": "deep_model_training_leaderboard_latest.csv",
        "deep_model_tuned_benchmark": "deep_model_tuned_benchmark_latest.json",
        "deep_model_tuned_benchmark_leaderboard": "deep_model_tuned_benchmark_leaderboard_latest.csv",
        "deep_model_tuned_benchmark_battle_cards": "deep_model_tuned_benchmark_battle_cards_latest.csv",
        "deep_model_tuned_benchmark_manifest": "deep_model_tuned_benchmark_manifest_latest.json",
        "deep_model_tuned_benchmark_run_diff": "deep_model_tuned_benchmark_run_diff_latest.json",
        "deep_model_tuned_benchmark_doc": "deep_model_tuned_benchmark_latest.md",
        "deep_model_objective_diagnostics": "deep_model_objective_diagnostics_latest.json",
        "deep_model_objective_diagnostics_csv": "deep_model_objective_diagnostics_latest.csv",
        "algorithm_claim_ledger": "algorithm_claim_ledger_latest.json",
        "algorithm_claim_ledger_csv": "algorithm_claim_ledger_latest.csv",
        "paper_reproduction_gap_ledger": "paper_reproduction_gap_ledger_latest.json",
        "paper_reproduction_gap_ledger_csv": "paper_reproduction_gap_ledger_latest.csv",
        "model_evidence_promotion_matrix": "model_evidence_promotion_matrix_latest.json",
        "model_evidence_promotion_matrix_csv": "model_evidence_promotion_matrix_latest.csv",
        "model_upgrade_recipe_cards": "model_upgrade_recipe_cards_latest.json",
        "model_upgrade_recipe_cards_csv": "model_upgrade_recipe_cards_latest.csv",
        "deep_model_telemetry_gap_matrix": "deep_model_telemetry_gap_matrix_latest.json",
        "deep_model_telemetry_gap_matrix_csv": "deep_model_telemetry_gap_matrix_latest.csv",
        "deep_model_telemetry_smoke": "deep_model_telemetry_smoke_latest.json",
        "deep_model_telemetry_smoke_csv": "deep_model_telemetry_smoke_latest.csv",
        "deep_model_forward_hook_contracts": "deep_model_forward_hook_contracts_latest.json",
        "deep_model_forward_hook_contracts_csv": "deep_model_forward_hook_contracts_latest.csv",
        "deep_model_forward_hook_smoke": "deep_model_forward_hook_smoke_latest.json",
        "deep_model_forward_hook_smoke_csv": "deep_model_forward_hook_smoke_latest.csv",
        "deep_model_hook_training_bridge": "deep_model_hook_training_bridge_latest.json",
        "deep_model_hook_training_bridge_csv": "deep_model_hook_training_bridge_latest.csv",
        "deep_model_trainer_collector_smoke": "deep_model_trainer_collector_smoke_latest.json",
        "deep_model_trainer_collector_models_csv": "deep_model_trainer_collector_models_latest.csv",
        "deep_model_trainer_collector_epochs_csv": "deep_model_trainer_collector_epochs_latest.csv",
        "deep_model_telemetry_benchmark_linkage": "deep_model_telemetry_benchmark_linkage_latest.json",
        "deep_model_telemetry_benchmark_linkage_csv": "deep_model_telemetry_benchmark_linkage_latest.csv",
        "deep_model_telemetry_ablation_gate": "deep_model_telemetry_ablation_gate_latest.json",
        "deep_model_telemetry_ablation_gate_csv": "deep_model_telemetry_ablation_gate_latest.csv",
        "deep_model_ablation_interpretation": "deep_model_ablation_interpretation_latest.json",
        "deep_model_ablation_interpretation_csv": "deep_model_ablation_interpretation_latest.csv",
        "deep_model_ablation_promotion_matrix": "deep_model_ablation_promotion_matrix_latest.json",
        "deep_model_ablation_promotion_matrix_csv": "deep_model_ablation_promotion_matrix_latest.csv",
        "deep_model_ablation_next_experiment_plan": "deep_model_ablation_next_experiment_plan_latest.json",
        "deep_model_ablation_next_experiment_plan_csv": "deep_model_ablation_next_experiment_plan_latest.csv",
        "deep_model_ablation_command_contract": "deep_model_ablation_command_contract_latest.json",
        "deep_model_ablation_command_contract_csv": "deep_model_ablation_command_contract_latest.csv",
        "deep_model_ablation_command_contract_smoke": "deep_model_ablation_command_contract_smoke_latest.json",
        "deep_model_ablation_command_contract_smoke_csv": "deep_model_ablation_command_contract_smoke_latest.csv",
        "deep_model_ablation_recommended_command_smoke": "deep_model_ablation_recommended_command_smoke_latest.json",
        "deep_model_ablation_recommended_command_smoke_csv": "deep_model_ablation_recommended_command_smoke_latest.csv",
        "deep_model_ablation_command_smoke_coverage": "deep_model_ablation_command_smoke_coverage_latest.json",
        "deep_model_ablation_command_smoke_coverage_csv": "deep_model_ablation_command_smoke_coverage_latest.csv",
        "deep_model_ablation_command_smoke_coverage_diff": "deep_model_ablation_command_smoke_coverage_diff_latest.json",
        "deep_model_ablation_command_smoke_coverage_diff_csv": "deep_model_ablation_command_smoke_coverage_diff_latest.csv",
        "paper_benchmark": "paper_benchmark_latest.json",
        "paper_benchmark_leaderboard": "paper_benchmark_leaderboard_latest.csv",
        "paper_benchmark_run_diff": "paper_benchmark_run_diff_latest.json",
        "paper_benchmark_manifest": "paper_benchmark_manifest_latest.json",
        "paper_benchmark_interpretation": "paper_benchmark_interpretation_latest.json",
        "paper_benchmark_doc": "paper_benchmark_latest.md",
        "evidence_store": "evidence_store_latest.json",
        "evidence_store_csv": "evidence_store_latest.csv",
        "regression_warning_audit": "regression_warning_audit_latest.json",
        "regression_warning_audit_csv": "regression_warning_audit_latest.csv",
        "promotion_launch_cards": "promotion_launch_cards_latest.json",
        "promotion_launch_cards_csv": "promotion_launch_cards_latest.csv",
        "regression_warning_triage": "regression_warning_triage_latest.json",
        "regression_warning_triage_csv": "regression_warning_triage_latest.csv",
        "pilot_readiness_scorecard": "pilot_readiness_scorecard_latest.json",
        "pilot_readiness_scorecard_csv": "pilot_readiness_scorecard_latest.csv",
        "pilot_experiment_plan": "pilot_experiment_plan_latest.json",
        "pilot_experiment_plan_csv": "pilot_experiment_plan_latest.csv",
        "pilot_experiment_command_smoke": "pilot_experiment_command_smoke_latest.json",
        "pilot_experiment_command_smoke_csv": "pilot_experiment_command_smoke_latest.csv",
        "pilot_experiment_command_coverage": "pilot_experiment_command_coverage_latest.json",
        "pilot_experiment_command_coverage_csv": "pilot_experiment_command_coverage_latest.csv",
        "glossary": "glossary_latest.json",
        "model_deconstruction_catalog": "model_deconstruction_catalog_latest.json",
        "model_deconstruction_agent": "model_deconstruction_agent_latest.json",
        "uplift_diagnostic_cases": "uplift_diagnostic_cases_latest.json",
        "uplift_failure_modes_smoke": "uplift_failure_modes_smoke_latest.json",
        "interview_evidence_pack": "interview_evidence_pack_latest.json",
        "interview_zip": "deepuplift_agent_interview_pack_latest.zip",
    }
    reports_dir = Path("reports")
    artifacts = {}
    for name, pattern in patterns.items():
        files = sorted(reports_dir.glob(pattern), key=lambda item: item.stat().st_mtime, reverse=True)
        if files:
            path = files[0]
            artifacts[name] = {
                "path": str(path),
                "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime)),
                "bytes": path.stat().st_size,
            }
    screenshot_dirs = sorted(reports_dir.glob("ui_screenshots_*"), key=lambda item: item.stat().st_mtime, reverse=True)
    latest_dir = None
    screenshots: list[Path] = []
    for candidate in screenshot_dirs:
        candidate_screenshots = sorted(candidate.glob("*.png"))
        if candidate_screenshots:
            latest_dir = candidate
            screenshots = candidate_screenshots
            break
    if latest_dir is not None:
        artifacts["ui_screenshots"] = {
            "path": str(latest_dir),
            "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(latest_dir.stat().st_mtime)),
            "files": len(screenshots),
            "bytes": sum(path.stat().st_size for path in screenshots),
        }
    return artifacts


def artifact_mime(path: Path) -> str:
    if path.suffix == ".json":
        return "application/json"
    if path.suffix == ".csv":
        return "text/csv"
    if path.suffix == ".md":
        return "text/markdown"
    return "application/octet-stream"


def _run_git_lines(args: list[str]) -> list[str]:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return []
    if proc.returncode != 0:
        return []
    return [line for line in proc.stdout.splitlines() if line.strip()]


def _git_state_label(status_code: str) -> str:
    if status_code == "??":
        return "untracked"
    if "D" in status_code:
        return "deleted"
    if "R" in status_code:
        return "renamed"
    if "A" in status_code:
        return "added"
    if "M" in status_code:
        return "modified"
    return status_code.strip() or "changed"


def _classify_repo_change(path: str) -> dict[str, object]:
    if path == ".gitignore":
        return {
            "commit_order": 1,
            "commit_batch": "repo_hygiene",
            "lane": "Repo hygiene",
            "visual_page": "Workflow / Evidence",
            "pilot_relation": "keeps secrets, reports and local caches out of review",
            "risk": "low",
            "next_action": "stage with dependency metadata before larger feature batches",
        }
    if path.startswith("requirements") or path in {"pyproject.toml", "setup.py"}:
        return {
            "commit_order": 1,
            "commit_batch": "dependency_contract",
            "lane": "Repo hygiene",
            "visual_page": "Models / Evidence",
            "pilot_relation": "documents which backends are runnable versus optional or guarded",
            "risk": "medium",
            "next_action": "verify imports and optional backend gates after staging",
        }
    if path == "app.py" or path.startswith(".streamlit/"):
        return {
            "commit_order": 2,
            "commit_batch": "visualization_surface",
            "lane": "Visualization",
            "visual_page": "Workflow / Evidence / Story",
            "pilot_relation": "makes evidence visible; does not clear business launch blockers alone",
            "risk": "medium",
            "next_action": "run UI smoke after final page wiring",
        }
    if path.startswith("deepuplift/core/") or path.startswith("deepuplift/knowledge/") or path == "deepuplift/__init__.py":
        return {
            "commit_order": 3,
            "commit_batch": "agent_core_platform",
            "lane": "Agent core",
            "visual_page": "Workflow / Knowledge / Agent / Evidence",
            "pilot_relation": "feeds readiness, evidence, diagnostics and agent answers",
            "risk": "high",
            "next_action": "pair with smoke/regression scripts that cover the new contracts",
        }
    if path.startswith("deepuplift/models/") or path in {"deepuplift/main.py", "deepuplift/utils/evaluate.py"}:
        return {
            "commit_order": 4,
            "commit_batch": "model_training_backend",
            "lane": "Model backend",
            "visual_page": "Models / Model Deconstruction / Evidence",
            "pilot_relation": "supports model claims but needs benchmark and telemetry evidence",
            "risk": "high",
            "next_action": "run model smoke or focused unit checks before promotion claims",
        }
    if path.startswith("scripts/"):
        return {
            "commit_order": 5,
            "commit_batch": "evidence_automation",
            "lane": "Evidence automation",
            "visual_page": "Evidence / History",
            "pilot_relation": "turns readiness and pilot claims into reproducible commands",
            "risk": "medium",
            "next_action": "stage beside the docs or artifacts the script regenerates",
        }
    if path.startswith("examples/"):
        return {
            "commit_order": 6,
            "commit_batch": "scenario_datasets",
            "lane": "Scenario data",
            "visual_page": "Data / Scenario Workbench / Policy",
            "pilot_relation": "demo and benchmark support; synthetic data is not online pilot proof",
            "risk": "medium",
            "next_action": "keep manifest, generated data and scenario docs together",
        }
    if path.startswith("docs/") or path == "README.md":
        return {
            "commit_order": 7,
            "commit_batch": "docs_evidence_pack",
            "lane": "Docs and evidence",
            "visual_page": "Evidence / Story / Knowledge",
            "pilot_relation": "explains evidence state and remaining launch blockers",
            "risk": "low",
            "next_action": "refresh evidence index and story screenshots after final wording",
        }
    return {
        "commit_order": 99,
        "commit_batch": "misc_review",
        "lane": "Misc review",
        "visual_page": "Manual review",
        "pilot_relation": "needs owner classification before commit",
        "risk": "medium",
        "next_action": "inspect and assign to one of the named batches",
    }


def repo_change_snapshot() -> pd.DataFrame:
    rows = []
    for line in _run_git_lines(["status", "--short"]):
        status_code = line[:2]
        path = line[3:].strip()
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        classification = _classify_repo_change(path)
        rows.append(
            {
                "git_code": status_code,
                "git_state": _git_state_label(status_code),
                "path": path,
                **classification,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "git_code",
                "git_state",
                "path",
                "commit_order",
                "commit_batch",
                "lane",
                "visual_page",
                "pilot_relation",
                "risk",
                "next_action",
            ]
        )
    return pd.DataFrame(rows).sort_values(["commit_order", "path"]).reset_index(drop=True)


def render_repo_change_intake_dashboard(
    key_prefix: str = "repo_change_intake",
    *,
    compact: bool = False,
    artifacts: dict | None = None,
) -> None:
    change_rows = repo_change_snapshot()
    artifacts = artifacts or latest_report_artifacts()
    pilot_summary = _artifact_summary(artifacts, "pilot_readiness_scorecard")
    pilot_gate = pilot_summary.get("pilot_gate", "NA")
    blocked_rows = pilot_summary.get("blocked", "NA")
    untracked_count = int((change_rows["git_state"] == "untracked").sum()) if not change_rows.empty else 0
    modified_count = int((change_rows["git_state"] == "modified").sum()) if not change_rows.empty else 0
    batch_count = int(change_rows["commit_batch"].nunique()) if not change_rows.empty else 0

    st.subheader("Repo Change Intake Board")
    st.caption(
        "把当前未提交改动先按提交批次、业务含义和可视化落点整理清楚。"
        "这解决的是提交前归档和演示可解释性；pilot gate 是否放行仍取决于业务和上线证据。"
    )
    render_stat_grid(
        [
            ("Worktree Files", f"{len(change_rows):,}", False),
            ("Untracked", f"{untracked_count:,}", False),
            ("Modified", f"{modified_count:,}", False),
            ("Commit Batches", f"{batch_count:,}", False),
            ("Pilot Gate", pilot_gate, True),
            ("Blocked Rows", blocked_rows, True),
        ],
        columns=6 if not compact else 3,
    )
    if change_rows.empty:
        st.success("Working tree is clean. No pre-commit grouping is needed.")
        return

    batch_summary = (
        change_rows.groupby(
            ["commit_order", "commit_batch", "lane", "visual_page", "pilot_relation", "risk", "next_action"],
            as_index=False,
        )
        .agg(
            files=("path", "count"),
            states=("git_state", lambda values: ", ".join(sorted(set(str(value) for value in values)))),
            examples=("path", lambda values: "; ".join(list(values)[:6])),
        )
        .sort_values(["commit_order", "commit_batch"])
    )

    if compact:
        st.dataframe(
            batch_summary[["commit_batch", "lane", "files", "states", "visual_page", "risk", "next_action"]],
            width="stretch",
            hide_index=True,
        )
        return

    batch_tab, file_tab, gate_tab = st.tabs(["Commit Batches", "File Map", "Pilot Gate Link"])
    with batch_tab:
        st.dataframe(
            batch_summary[
                [
                    "commit_order",
                    "commit_batch",
                    "lane",
                    "files",
                    "states",
                    "visual_page",
                    "risk",
                    "next_action",
                    "examples",
                ]
            ],
            width="stretch",
            hide_index=True,
        )
        st.download_button(
            "Download repo change intake CSV",
            data=dataframe_csv_bytes(change_rows),
            file_name="deepuplift_repo_change_intake.csv",
            mime="text/csv",
            key=f"{key_prefix}_download_rows",
        )
    with file_tab:
        filters = st.columns([0.25, 0.25, 0.25, 0.25])
        batch_options = ["All"] + batch_summary["commit_batch"].tolist()
        state_options = ["All"] + sorted(change_rows["git_state"].dropna().astype(str).unique().tolist())
        risk_options = ["All"] + sorted(change_rows["risk"].dropna().astype(str).unique().tolist())
        selected_batch = filters[0].selectbox("Commit batch", batch_options, key=f"{key_prefix}_batch")
        selected_state = filters[1].selectbox("Git state", state_options, key=f"{key_prefix}_state")
        selected_risk = filters[2].selectbox("Risk", risk_options, key=f"{key_prefix}_risk")
        search_text = filters[3].text_input("Search path", value="", key=f"{key_prefix}_search")
        view = change_rows.copy()
        if selected_batch != "All":
            view = view[view["commit_batch"] == selected_batch]
        if selected_state != "All":
            view = view[view["git_state"] == selected_state]
        if selected_risk != "All":
            view = view[view["risk"] == selected_risk]
        if search_text:
            view = view[view["path"].astype(str).str.contains(search_text, case=False, na=False)]
        st.dataframe(
            view[
                [
                    "git_state",
                    "path",
                    "commit_batch",
                    "lane",
                    "visual_page",
                    "pilot_relation",
                    "risk",
                    "next_action",
                ]
            ],
            width="stretch",
            hide_index=True,
        )
    with gate_tab:
        gate_rows = [
            {
                "topic": "Current blocker",
                "status": f"pilot_gate={pilot_gate}; blocked={blocked_rows}",
                "meaning": "blocked 是业务/上线证据层 blocker，不是页面渲染问题。",
                "next_action": "补齐 support/value/refutation/shadow 或 A/B 前置证据。",
            },
            {
                "topic": "What this board solves",
                "status": "repo intake visible",
                "meaning": "把大量源码、文档、脚本和数据变成可审查的提交批次。",
                "next_action": "按 commit_order 分批 stage，避免一个超大提交吞掉所有上下文。",
            },
            {
                "topic": "What visualization solves",
                "status": "demo evidence surfaced",
                "meaning": "页面能展示证据链、剩余 blocker 和下一步命令。",
                "next_action": "UI smoke 只证明可视化可用，不能替代 pilot evidence。",
            },
            {
                "topic": "Safe next commit split",
                "status": f"{batch_count} suggested batches",
                "meaning": "先提交 hygiene/dependency/UI，再提交 core/model/scripts/data/docs。",
                "next_action": "每批提交后跑对应 smoke，最后刷新 Evidence/Story 截图。",
            },
        ]
        st.dataframe(pd.DataFrame(gate_rows), width="stretch", hide_index=True)


def source_readiness_trend(limit: int = 12) -> pd.DataFrame:
    reports_dir = Path("reports")
    files = sorted(reports_dir.glob("model_source_readiness_*.csv"), key=lambda item: item.stat().st_mtime, reverse=True)[:limit]
    rows = []
    for path in reversed(files):
        stamp = path.stem.replace("model_source_readiness_", "")
        try:
            frame = pd.read_csv(path)
        except Exception:
            continue
        for record in frame.to_dict(orient="records"):
            rows.append({"snapshot": stamp, **record})
    return pd.DataFrame(rows)


def render_regression_health() -> None:
    artifacts = latest_report_artifacts()
    audit_payload = {}
    audit_path = Path((artifacts.get("model_audit") or {}).get("path", ""))
    if audit_path.is_file():
        try:
            audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            audit_payload = {}
    regression_payload = {}
    regression_path = Path((artifacts.get("agent_regression") or {}).get("path", ""))
    if regression_path.is_file():
        try:
            regression_payload = json.loads(regression_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            regression_payload = {}
    latest_runs = regression_payload.get("results") or regression_payload.get("runs") or []
    if isinstance(regression_payload, list):
        latest_runs = regression_payload
    if not latest_runs and isinstance(regression_payload, dict):
        latest_runs = regression_payload.get("cases", [])
    failed_runs = [row for row in latest_runs if isinstance(row, dict) and row.get("status") not in {None, "ok"}]
    health = "pass" if artifacts and not failed_runs and not audit_payload.get("failures") else "warn"
    render_stat_grid(
        [
            ("Health", health, True),
            ("Registered", f"{audit_payload.get('registered_models', 'NA')}", False),
            ("Ready", f"{audit_payload.get('ready_models', 'NA')}", False),
            ("UI Shots", f"{(artifacts.get('ui_screenshots') or {}).get('files', 0)}", False),
        ]
    )
    rows = []
    for name, payload in artifacts.items():
        rows.append({"artifact": name, **payload})
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    else:
        st.info("No regression artifacts found yet. Run `scripts/full_regression_check.sh` to generate them.")
    ready_source_counts = audit_payload.get("ready_source_counts") or {}
    if ready_source_counts:
        ready_rows = pd.DataFrame(
            [{"source": source, "ready_models": count} for source, count in sorted(ready_source_counts.items())]
        )
        st.subheader("Ready Backends")
        st.bar_chart(ready_rows.set_index("source")["ready_models"])
        st.dataframe(ready_rows, width="stretch", hide_index=True)
    readiness_trend = source_readiness_trend()
    if not readiness_trend.empty:
        st.subheader("Source Readiness Trend")
        st.dataframe(readiness_trend, width="stretch", hide_index=True)
    if latest_runs:
        run_rows = []
        for row in latest_runs:
            if not isinstance(row, dict):
                continue
            warnings = row.get("diagnostic_warnings") or []
            sensitivity = row.get("sensitivity") or {}
            run_rows.append(
                {
                    "model": row.get("model"),
                    "status": row.get("status"),
                    "qini": row.get("qini"),
                    "auuc": row.get("auuc"),
                    "calibration_mae": row.get("calibration_mae"),
                    "sensitivity": sensitivity.get("verdict"),
                    "warnings": "; ".join(warnings[:3]),
                    "run_id": row.get("run_id"),
                }
            )
        if run_rows:
            st.subheader("Latest Regression Runs")
            st.dataframe(pd.DataFrame(run_rows), width="stretch", hide_index=True)
    for name, payload in artifacts.items():
        path = Path(payload.get("path", ""))
        if path.is_file():
            st.download_button(
                f"Download {name}",
                data=path.read_bytes(),
                file_name=path.name,
                mime=artifact_mime(path),
            )


def render_story_evidence_panel() -> None:
    artifacts = latest_report_artifacts()
    st.subheader("System Health Evidence")
    if not artifacts:
        st.info("No evidence artifacts found yet. Run `scripts/interview_demo_check.sh` to generate audit, regression, compare, and screenshot evidence.")
        return

    rows = []
    for name, payload in artifacts.items():
        rows.append(
            {
                "artifact": name,
                "path": payload.get("path"),
                "modified": payload.get("modified"),
                "files": str(payload.get("files", "")),
                "bytes": str(payload.get("bytes", "")),
            }
        )
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    download_cols = st.columns(4)
    downloadable = [(name, Path(str(payload.get("path", "")))) for name, payload in artifacts.items()]
    for index, (name, path) in enumerate(downloadable):
        with download_cols[index % len(download_cols)]:
            if path.is_file():
                st.download_button(
                    f"Download {name}",
                    data=path.read_bytes(),
                    file_name=path.name,
                    mime=artifact_mime(path),
                    key=f"story_artifact_download_{name}",
                )
            elif path.is_dir():
                screenshot_names = ", ".join(sorted(item.name for item in path.glob("*.png"))[:8])
                st.caption(f"{name}: {screenshot_names or path.name}")


def render_model_risk_matrix(key_prefix: str = "model_risk") -> None:
    st.subheader("模型风险矩阵")
    st.caption(
        "这里不是罗列模型，而是说明每类 uplift 模型在什么数据条件下会失效、哪些指标会误导、应该怎么诊断和修复。"
    )
    counts = failure_mode_counts()
    render_stat_grid(
        [
            ("模型 failure modes", f"{counts.get('model_failure_modes', 0):,}", False),
            ("场景 pitfall", f"{counts.get('scenario_pitfalls', 0):,}", False),
            ("mock 诊断案例", f"{counts.get('diagnostic_cases', 0):,}", False),
            ("上线闸门", "Overlap + CI + ROI", True),
        ],
        columns=4,
    )
    risk_df = pd.DataFrame(model_failure_mode_rows())
    if risk_df.empty:
        st.info("No model failure modes found.")
        return
    family_options = risk_df["family"].tolist()
    default_families = family_options[:8]
    selected_families = st.multiselect(
        "选择模型家族",
        family_options,
        default=default_families,
        key=f"{key_prefix}_family_filter",
    )
    risk_view = risk_df[risk_df["family"].isin(selected_families)] if selected_families else risk_df
    risk_display = risk_view.rename(
        columns={
            "family": "模型家族",
            "risk": "主要风险",
            "symptom": "现象",
            "root_cause": "根因",
            "diagnostic": "诊断方法",
            "misleading_metric": "容易误导的指标",
            "fix": "修复路径",
            "ui_evidence": "UI/Evidence",
            "interview_line": "面试讲法",
        }
    )
    st.dataframe(risk_display, width="stretch", hide_index=True)
    st.download_button(
        "下载模型风险矩阵 CSV",
        data=dataframe_csv_bytes(risk_df),
        file_name="deepuplift_model_failure_modes.csv",
        mime="text/csv",
        key=f"{key_prefix}_download_model_failure_modes",
    )


def render_scenario_pitfall_cards(key_prefix: str = "scenario_pitfalls") -> None:
    st.subheader("业务场景问题库")
    st.caption("同一个 uplift 算法放到不同业务里，最容易错的地方不同：发券看套利和毛利，广告看增量和 iROAS，B 端补贴看干扰和供需平衡。")
    pitfall_df = pd.DataFrame(scenario_pitfall_rows())
    if pitfall_df.empty:
        st.info("No scenario pitfall cards found.")
        return
    selected_scenarios = st.multiselect(
        "选择业务场景",
        pitfall_df["scenario"].tolist(),
        default=pitfall_df["scenario"].tolist()[:5],
        key=f"{key_prefix}_scenario_filter",
    )
    view = pitfall_df[pitfall_df["scenario"].isin(selected_scenarios)] if selected_scenarios else pitfall_df
    cards = []
    for row in view.to_dict(orient="records"):
        cards.append(
            "<div class='du-backend-card'>"
            f"<div class='du-backend-kicker'>{html.escape(row['scenario'])}</div>"
            f"<div class='du-backend-title'>{html.escape(row['pitfall'])}</div>"
            f"<div class='du-backend-meta'>{html.escape(row['interview_line'])}</div>"
            "</div>"
        )
    st.markdown(f"<div class='du-backend-grid'>{''.join(cards)}</div>", unsafe_allow_html=True)
    st.dataframe(
        view.rename(
            columns={
                "scenario": "场景",
                "pitfall": "坑位",
                "symptom": "现象",
                "diagnostic": "诊断",
                "fix": "修复",
                "interview_line": "面试讲法",
            }
        ),
        width="stretch",
        hide_index=True,
    )
    st.download_button(
        "下载业务场景 pitfall CSV",
        data=dataframe_csv_bytes(pitfall_df),
        file_name="deepuplift_scenario_pitfalls.csv",
        mime="text/csv",
        key=f"{key_prefix}_download_scenario_pitfalls",
    )


def render_failure_mode_playbook(key_prefix: str = "failure_modes") -> None:
    st.subheader("Failure Modes / 指标误导案例")
    st.caption(
        "这一块把“模型可能错在哪里”和“指标为什么可能误导”落成 mock 数据、case card、smoke 报告和修复路径，适合面试现场展开。"
    )
    case_df = pd.DataFrame(diagnostic_case_rows())
    summary_df = pd.DataFrame(diagnostic_case_summary_rows())
    if not case_df.empty:
        type_counts = case_df["problem_type"].value_counts().reset_index()
        type_counts.columns = ["问题类型", "案例数"]
        render_stat_grid(
            [
                ("诊断案例", f"{len(case_df):,}", False),
                ("生成数据", f"{len(summary_df):,}", False),
                ("核心陷阱", "QINI != ROI", True),
                ("修复主线", "诊断 -> 裁剪 -> 校准 -> Policy", True),
            ],
            columns=4,
        )
        if not type_counts.empty:
            st.bar_chart(type_counts.set_index("问题类型")["案例数"])
        case_view = case_df.rename(
            columns={
                "case_id": "case_id",
                "case_name": "案例",
                "problem_type": "问题类型",
                "mock_signal": "mock 信号",
                "affected_models": "受影响模型/指标",
                "misleading_metric": "误导指标",
                "diagnostic_method": "诊断方法",
                "fix_path": "修复路径",
                "interview_line": "面试讲法",
            }
        )
        st.dataframe(case_view, width="stretch", hide_index=True)
    if not summary_df.empty:
        st.subheader("Mock Diagnostic Case Summary")
        chart_cols = [col for col in ["case_id", "top10_policy_value"] if col in summary_df.columns]
        if len(chart_cols) == 2:
            chart_df = summary_df[chart_cols].copy()
            chart_df["top10_policy_value"] = pd.to_numeric(chart_df["top10_policy_value"], errors="coerce").fillna(0.0)
            st.bar_chart(chart_df.set_index("case_id")["top10_policy_value"])
        display_cols = [
            col
            for col in [
                "case_id",
                "rows",
                "treatment_rate",
                "weak_overlap_rate",
                "top10_true_uplift",
                "top10_policy_value",
                "d1_rate_top10",
                "d30_rate_top10",
                "click_uplift_top10",
                "conversion_uplift_top10",
                "mock_csv",
            ]
            if col in summary_df.columns
        ]
        st.dataframe(summary_df[display_cols], width="stretch", hide_index=True)
        st.download_button(
            "下载 mock diagnostic case summary",
            data=dataframe_csv_bytes(summary_df),
            file_name="deepuplift_diagnostic_case_summary.csv",
            mime="text/csv",
            key=f"{key_prefix}_download_diagnostic_case_summary",
        )
    else:
        st.info("运行 `scripts/generate_uplift_diagnostic_cases.py` 生成 mock diagnostic case 数据和 summary。")


def _read_json_file(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _artifact_payload(artifacts: dict, key: str) -> dict:
    return _read_json_file(Path((artifacts.get(key) or {}).get("path", "")))


def _artifact_summary(artifacts: dict, key: str) -> dict:
    return _artifact_payload(artifacts, key).get("summary") or {}


def _story_freshness_signal() -> tuple[str, str]:
    reports_dir = Path("reports")
    story_candidates = sorted(
        reports_dir.glob("ui_screenshots_*/story.png"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    references = [
        Path("docs/DEEPUplift_AGENT_ONEPAGER_CN.md"),
        Path("docs/DEEPUplift_AGENT_INTERVIEW_EVIDENCE_PACK.md"),
        Path("docs/DEEPUplift_AGENT_EVIDENCE_INDEX.md"),
        Path("reports/deepuplift_agent_interview_pack_latest.zip"),
    ]
    existing_refs = [path for path in references if path.exists()]
    if not story_candidates:
        return "missing", "run smoke_ui_screenshots for Story"
    if not existing_refs:
        return "unknown", "no generated reference artifact found"
    story = story_candidates[0]
    newest_ref = max(existing_refs, key=lambda path: path.stat().st_mtime)
    if story.stat().st_mtime < newest_ref.stat().st_mtime:
        return "stale", f"{story} older than {newest_ref}"
    return "fresh", str(story)


def optimization_closure_snapshot(artifacts: dict | None = None) -> tuple[pd.DataFrame, list[tuple[str, object, bool]]]:
    artifacts = artifacts or latest_report_artifacts()
    model_summary = _artifact_payload(artifacts, "model_audit")
    promotion_summary = _artifact_summary(artifacts, "model_evidence_promotion_matrix")
    telemetry_summary = _artifact_summary(artifacts, "deep_model_telemetry_gap_matrix")
    hook_summary = _artifact_summary(artifacts, "deep_model_forward_hook_smoke")
    bridge_summary = _artifact_summary(artifacts, "deep_model_trainer_collector_smoke")
    ablation_plan_summary = _artifact_summary(artifacts, "deep_model_ablation_next_experiment_plan")
    ablation_coverage_summary = _artifact_summary(artifacts, "deep_model_ablation_command_smoke_coverage")
    ablation_coverage_diff_summary = _artifact_summary(artifacts, "deep_model_ablation_command_smoke_coverage_diff")
    pilot_summary = _artifact_summary(artifacts, "pilot_readiness_scorecard")
    pilot_plan_summary = _artifact_summary(artifacts, "pilot_experiment_plan")
    pilot_smoke_summary = _artifact_summary(artifacts, "pilot_experiment_command_smoke")
    warning_triage_summary = _artifact_summary(artifacts, "regression_warning_triage")
    evidence_store_summary = _artifact_payload(artifacts, "evidence_store").get("summary") or {}

    gitignore_text = Path(".gitignore").read_text(encoding="utf-8") if Path(".gitignore").exists() else ""
    gitignore_lines = {line.strip() for line in gitignore_text.splitlines()}
    hygiene_status = "closed" if ".env" in gitignore_lines and "reports/" in gitignore_lines else "open"
    story_status, story_detail = _story_freshness_signal()
    story_ui_status = "ready" if story_status == "fresh" else "refresh"

    registered_models = model_summary.get("registered_models", model_summary.get("registered", "NA"))
    ready_models = model_summary.get("ready_models", model_summary.get("ready", "NA"))
    pilot_gate = pilot_summary.get("pilot_gate", "NA")
    blocked_pilots = pilot_summary.get("blocked", "NA")
    p0_telemetry = telemetry_summary.get("p0_rows", "NA")
    ablation_p0 = ablation_plan_summary.get("p0_rows", "NA")
    ablation_p1 = ablation_plan_summary.get("p1_rows", "NA")

    rows = [
        {
            "lane": "Repo hygiene",
            "capability": "Generated artifact guard",
            "status": hygiene_status,
            "signal": ".env and reports/ ignored" if hygiene_status == "closed" else "update .gitignore",
            "visual_page": "Workflow / Evidence / Story",
            "next_action": "commit source/docs/scripts without local secrets or bulk reports",
            "artifact": ".gitignore",
        },
        {
            "lane": "Model catalog",
            "capability": "Registered and runnable model inventory",
            "status": "ready",
            "signal": f"{registered_models} registered / {ready_models} runnable",
            "visual_page": "Models -> Ready Backends; Evidence -> Regression Gate Artifacts",
            "next_action": "keep guarded backends separate from ready models",
            "artifact": (artifacts.get("model_audit") or {}).get("path", ""),
        },
        {
            "lane": "Promotion governance",
            "capability": "Model evidence promotion matrix",
            "status": "ready",
            "signal": f"{promotion_summary.get('models', 'NA')} models; {len(promotion_summary.get('promotion_tier_counts') or {})} tiers",
            "visual_page": "Evidence / Model Deconstruction",
            "next_action": "promote only T0/T1 rows with launch guardrails",
            "artifact": (artifacts.get("model_evidence_promotion_matrix") or {}).get("path", ""),
        },
        {
            "lane": "Deep observability",
            "capability": "Telemetry gaps and forward hooks",
            "status": "watch",
            "signal": f"{p0_telemetry} P0 gaps; {hook_summary.get('ok_contracts', 'NA')} hook contracts pass",
            "visual_page": "Evidence -> Deep Model Telemetry Gap Matrix",
            "next_action": "turn phi/e_hat/attention/head telemetry into benchmark-linked attribution",
            "artifact": (artifacts.get("deep_model_telemetry_gap_matrix") or {}).get("path", ""),
        },
        {
            "lane": "Deep observability",
            "capability": "Trainer telemetry collectors",
            "status": "ready",
            "signal": f"{bridge_summary.get('ok_models', 'NA')} models; {bridge_summary.get('epoch_rows', 'NA')} epoch rows",
            "visual_page": "Evidence -> Deep Model Trainer Collector Smoke",
            "next_action": "wire collector signals into default benchmark dashboards",
            "artifact": (artifacts.get("deep_model_trainer_collector_smoke") or {}).get("path", ""),
        },
        {
            "lane": "Ablation",
            "capability": "Next experiment plan",
            "status": "watch",
            "signal": f"P0={ablation_p0}, P1={ablation_p1}; fallback={ablation_plan_summary.get('fallback_ready_rows', 'NA')}",
            "visual_page": "Evidence -> Deep Model Ablation Next Experiment Plan",
            "next_action": "run seed/baseline stress for P0 variants before stronger claims",
            "artifact": (artifacts.get("deep_model_ablation_next_experiment_plan") or {}).get("path", ""),
        },
        {
            "lane": "Ablation",
            "capability": "Command smoke coverage",
            "status": "ready" if ablation_coverage_summary.get("coverage_gate") == "ok" else "watch",
            "signal": f"{ablation_coverage_summary.get('any_smoked_rows', 'NA')} smoked; {ablation_coverage_diff_summary.get('coverage_gate_delta', 'NA')}",
            "visual_page": "Evidence -> Deep Model Ablation Command Smoke Coverage",
            "next_action": "keep runner-native command smoke green as plans change",
            "artifact": (artifacts.get("deep_model_ablation_command_smoke_coverage") or {}).get("path", ""),
        },
        {
            "lane": "Pilot readiness",
            "capability": "Pilot scorecard and launch blockers",
            "status": "blocked" if pilot_gate == "blocked" else "ready",
            "signal": f"gate={pilot_gate}; blocked={blocked_pilots}",
            "visual_page": "Evidence -> Pilot Readiness Scorecard",
            "next_action": "clear support/value/refutation blockers before shadow or A/B",
            "artifact": (artifacts.get("pilot_readiness_scorecard") or {}).get("path", ""),
        },
        {
            "lane": "Pilot readiness",
            "capability": "Pilot experiment commands",
            "status": "ready",
            "signal": f"P0={pilot_plan_summary.get('p0_rows', 'NA')}; smoke ok={pilot_smoke_summary.get('ok_rows', 'NA')}",
            "visual_page": "Evidence -> Pilot Experiment Plan",
            "next_action": "run one blocker-repair, one shadow, one paper-review command before launch review",
            "artifact": (artifacts.get("pilot_experiment_plan") or {}).get("path", ""),
        },
        {
            "lane": "Evidence",
            "capability": "Warning triage and evidence store",
            "status": "watch" if warning_triage_summary.get("triage_gate") else "ready",
            "signal": f"triage={warning_triage_summary.get('triage_gate', 'NA')}; records={evidence_store_summary.get('records', evidence_store_summary.get('manifest_count', 'NA'))}",
            "visual_page": "Evidence -> Regression Warning Triage",
            "next_action": "keep launch-blocking rows separated from smoke-only review rows",
            "artifact": (artifacts.get("regression_warning_triage") or {}).get("path", ""),
        },
        {
            "lane": "Visual proof",
            "capability": "Story screenshot freshness",
            "status": story_ui_status,
            "signal": story_detail,
            "visual_page": "Story / Evidence -> UI Screenshot Gallery",
            "next_action": "refresh Story screenshot after docs/evidence pack changes",
            "artifact": (artifacts.get("ui_screenshots") or {}).get("path", ""),
        },
    ]

    metrics = [
        ("Pilot Gate", pilot_gate, True),
        ("Blocked Rows", blocked_pilots, False),
        ("Telemetry P0", p0_telemetry, False),
        ("Ablation P0/P1", f"{ablation_p0}/{ablation_p1}", True),
        ("Command Coverage", ablation_coverage_summary.get("coverage_gate", "NA"), True),
        ("Story Freshness", story_status, True),
    ]
    return pd.DataFrame(rows), metrics


def render_optimization_closure_dashboard(
    key_prefix: str = "optimization_closure",
    *,
    compact: bool = False,
    artifacts: dict | None = None,
) -> None:
    rows, metrics = optimization_closure_snapshot(artifacts)
    st.subheader("Optimization Closure Board")
    st.caption("本轮新增能力的可视化总览：哪些已经可展示，哪些仍是 launch blocker，下一步应该在哪个页面继续钻取。")
    render_stat_grid(metrics, columns=6 if not compact else 3)
    if rows.empty:
        st.info("No optimization closure artifacts found yet.")
        return

    if compact:
        compact_cols = ["lane", "capability", "status", "signal", "visual_page"]
        st.dataframe(rows[compact_cols], width="stretch", hide_index=True)
        return

    overview_tab, open_tab, demo_tab = st.tabs(["Capability Map", "Open Tail", "Demo Path"])
    with overview_tab:
        status_counts = rows["status"].value_counts().rename_axis("status").reset_index(name="items")
        st.bar_chart(status_counts, x="status", y="items")
        st.dataframe(rows, width="stretch", hide_index=True)
        st.download_button(
            "Download optimization closure board",
            data=dataframe_csv_bytes(rows),
            file_name="deepuplift_optimization_closure_board.csv",
            mime="text/csv",
            key=f"{key_prefix}_download",
        )
    with open_tab:
        open_rows = rows[rows["status"].isin(["blocked", "watch", "refresh", "open"])]
        if open_rows.empty:
            st.success("All closure lanes are currently green.")
        else:
            st.dataframe(
                open_rows[["lane", "capability", "status", "signal", "next_action", "artifact"]],
                width="stretch",
                hide_index=True,
            )
    with demo_tab:
        demo_rows = [
            {
                "step": "1. Workflow",
                "what_to_show": "Optimization Closure Board plus current dataset/model gates",
                "proof": "shows repo hygiene, catalog, pilot and telemetry status before training",
            },
            {
                "step": "2. Evidence",
                "what_to_show": "Pilot Readiness, Telemetry Gap Matrix, Ablation Plan, Command Smoke Coverage",
                "proof": "separates runnable evidence from launch blockers",
            },
            {
                "step": "3. Model Deconstruction",
                "what_to_show": "selected model review card, forward hooks, ablation/promotion rows",
                "proof": "turns deep-model claims into source/loss/benchmark evidence",
            },
            {
                "step": "4. Story",
                "what_to_show": "resume-facing path with fresh visual proof",
                "proof": "keeps the interview story aligned with latest artifacts",
            },
        ]
        st.dataframe(pd.DataFrame(demo_rows), width="stretch", hide_index=True)


def render_term_glossary(context: str = "benchmark", *, expanded: bool = False, key_prefix: str = "glossary") -> None:
    rows = glossary_rows(context)
    if not rows:
        rows = glossary_rows()
    with st.expander("术语解释 / Glossary", expanded=expanded):
        st.caption("页面里的英文指标和因果术语都在这里解释：是什么、为什么重要、常见误区、证据在哪。")
        context_options = glossary_context_names()
        selected_context = st.selectbox(
            "术语场景",
            context_options,
            index=context_options.index(context) if context in context_options else 0,
            key=f"{key_prefix}_context",
        )
        context_rows = glossary_rows(selected_context)
        term_options = [row["term"] for row in context_rows]
        default_terms = term_options[: min(len(term_options), 8)]
        selected_terms = st.multiselect(
            "术语",
            term_options,
            default=default_terms,
            key=f"{key_prefix}_terms",
        )
        display_rows = [row for row in context_rows if not selected_terms or row["term"] in selected_terms]
        glossary_df = pd.DataFrame(display_rows)
        if not glossary_df.empty:
            st.dataframe(
                glossary_df[
                    [
                        "term",
                        "category",
                        "short_cn",
                        "formula",
                        "why_it_matters",
                        "common_pitfall",
                        "evidence",
                    ]
                ].rename(
                    columns={
                        "term": "术语",
                        "category": "类别",
                        "short_cn": "一句话解释",
                        "formula": "公式/定义",
                        "why_it_matters": "为什么重要",
                        "common_pitfall": "常见误区",
                        "evidence": "证据入口",
                    }
                ),
                width="stretch",
                hide_index=True,
            )
            st.download_button(
                "Download glossary CSV",
                data=dataframe_csv_bytes(glossary_df),
                file_name=f"deepuplift_glossary_{selected_context}.csv",
                mime="text/csv",
                key=f"{key_prefix}_download",
            )


def render_paper_benchmark_dashboard(
    paper_benchmark: dict,
    paper_benchmark_diff: dict | None = None,
    paper_benchmark_interpretation: dict | None = None,
    *,
    selected_model_id: str | None = None,
    key_prefix: str = "paper_benchmark_dashboard",
) -> None:
    st.subheader("Benchmark Dashboard")
    render_term_glossary("benchmark", expanded=False, key_prefix=f"{key_prefix}_glossary")
    if not paper_benchmark.get("leaderboard"):
        st.info("Run `scripts/run_paper_benchmark_suite.py --preset smoke` to generate PEHE/ATE/QINI/AUUC benchmark evidence.")
        return
    leaderboard = pd.DataFrame(paper_benchmark.get("leaderboard") or [])
    runs = paper_benchmark.get("runs") or []
    failure = paper_benchmark.get("failure_attribution") or {}
    seeds = paper_benchmark.get("seeds") or []
    render_stat_grid(
        [
            ("Tier", paper_benchmark.get("benchmark_tier", paper_benchmark.get("preset", "NA")), True),
            ("Datasets", f"{len(paper_benchmark.get('datasets', [])):,}", False),
            ("Models", f"{len(paper_benchmark.get('models', [])):,}", False),
            ("Seeds", f"{len(seeds):,}", False),
            ("Runs", f"{len(runs):,}", False),
            ("Leaderboard", f"{len(leaderboard):,}", False),
        ],
        columns=6,
    )
    filter_cols = st.columns([0.34, 0.33, 0.33])
    dataset_options = ["All"] + sorted(leaderboard["dataset_id"].dropna().astype(str).unique().tolist())
    model_options = ["All"] + sorted(leaderboard["model"].dropna().astype(str).unique().tolist())
    if selected_model_id in model_options:
        default_model_index = model_options.index(selected_model_id)
    else:
        default_model_index = 0
    selected_dataset = filter_cols[0].selectbox("Dataset", dataset_options, key=f"{key_prefix}_dataset")
    selected_model = filter_cols[1].selectbox("Model", model_options, index=default_model_index, key=f"{key_prefix}_model")
    ranking_metric = filter_cols[2].selectbox(
        "Ranking metric",
        ["PEHE ↓", "ATE error ↓", "QINI ↑", "AUUC ↑", "Policy oracle Top10 ↑"],
        key=f"{key_prefix}_metric",
    )
    view = leaderboard.copy()
    if selected_dataset != "All":
        view = view[view["dataset_id"].astype(str) == selected_dataset]
    if selected_model != "All":
        view = view[view["model"].astype(str) == selected_model]
    sort_map = {
        "PEHE ↓": ("pehe_mean", True),
        "ATE error ↓": ("ate_error_mean", True),
        "QINI ↑": ("qini_mean", False),
        "AUUC ↑": ("auuc_mean", False),
        "Policy oracle Top10 ↑": ("policy_top10_oracle_value_mean", False),
    }
    sort_col, ascending = sort_map[ranking_metric]
    if sort_col in view.columns:
        view = view.sort_values(sort_col, ascending=ascending, na_position="last")
    display_cols = [
        col
        for col in [
            "dataset_id",
            "model",
            "ok_runs",
            "pehe_mean",
            "pehe_ci_low",
            "pehe_ci_high",
            "ate_error_mean",
            "qini_mean",
            "auuc_mean",
            "policy_top10_oracle_value_mean",
            "oracle_top10_recall_mean",
            "evidence_manifest_count",
        ]
        if col in view.columns
    ]
    st.dataframe(view[display_cols], width="stretch", hide_index=True)
    st.caption(
        "读法：PEHE/ATE error 越低越好；QINI/AUUC/policy value 越高越好。单 seed 结果用于 regression smoke，多 seed 才适合严肃 paper-level 对标。"
    )
    interpretation = paper_benchmark_interpretation or {}
    if interpretation.get("model_scorecards"):
        with st.expander("Benchmark Interpretation / 面试讲法", expanded=False):
            tracks = interpretation.get("interview_tracks") or {}
            track_rows = [
                {"duration": "30s", "talk_track": tracks.get("30s", "")},
                {"duration": "5min", "talk_track": tracks.get("5min", "")},
                {"duration": "15min", "talk_track": tracks.get("15min", "")},
                {"duration": "30min", "talk_track": tracks.get("30min", "")},
            ]
            st.dataframe(pd.DataFrame(track_rows), width="stretch", hide_index=True)
            scorecard_df = pd.DataFrame(interpretation.get("model_scorecards") or [])
            scorecard_cols = [
                col
                for col in [
                    "model",
                    "family",
                    "verdict",
                    "datasets",
                    "ok_runs",
                    "pehe_mean_over_datasets",
                    "qini_mean_over_datasets",
                    "winner_counts",
                    "stability_counts",
                ]
                if col in scorecard_df.columns
            ]
            if scorecard_cols:
                st.dataframe(scorecard_df[scorecard_cols], width="stretch", hide_index=True)
            conflicts = interpretation.get("metric_conflicts") or []
            if conflicts:
                st.caption("Metric conflicts explain why PEHE, ranking quality and policy value must be reviewed together.")
                st.dataframe(pd.DataFrame(conflicts), width="stretch", hide_index=True)
            recommendations = interpretation.get("recommendations") or []
            if recommendations:
                st.caption("Recommended next actions")
                st.dataframe(pd.DataFrame(recommendations), width="stretch", hide_index=True)

    tabs = st.tabs(["Best By Dataset", "Rankings", "Stability", "Failure Attribution", "Run Diff"])
    with tabs[0]:
        best_rows = paper_benchmark.get("best_by_dataset") or []
        if best_rows:
            best_df = pd.DataFrame(best_rows)
            best_cols = [
                col
                for col in ["dataset_id", "model", "pehe_mean", "ate_error_mean", "qini_mean", "auuc_mean", "evidence_manifest_count"]
                if col in best_df.columns
            ]
            st.dataframe(best_df[best_cols], width="stretch", hide_index=True)
        else:
            st.info("No best-by-dataset rows yet.")
    with tabs[1]:
        rankings = paper_benchmark.get("rankings") or {}
        metric_key = ranking_metric.split()[0].lower().replace("policy", "policy_top10_oracle_value")
        if ranking_metric.startswith("ATE"):
            metric_key = "ate_error"
        elif ranking_metric.startswith("Policy"):
            metric_key = "policy_top10_oracle_value"
        ranking_rows = rankings.get(metric_key) or []
        if ranking_rows:
            st.dataframe(pd.DataFrame(ranking_rows).head(30), width="stretch", hide_index=True)
        else:
            st.info("Ranking rows will appear after rerunning the upgraded benchmark suite.")
    with tabs[2]:
        stability_rows = paper_benchmark.get("stability_summary") or []
        if stability_rows:
            st.dataframe(pd.DataFrame(stability_rows), width="stretch", hide_index=True)
        else:
            st.info("Stability rows will appear after rerunning the upgraded benchmark suite.")
    with tabs[3]:
        failure_rows = []
        for status, count in sorted((failure.get("status_counts") or {}).items()):
            failure_rows.append({"type": "status", "item": status, "count": count})
        for reason, count in sorted((failure.get("skipped") or {}).items()):
            failure_rows.append({"type": "skipped", "item": reason, "count": count})
        for reason, count in sorted((failure.get("errors") or {}).items()):
            failure_rows.append({"type": "error", "item": reason, "count": count})
        if failure_rows:
            st.dataframe(pd.DataFrame(failure_rows), width="stretch", hide_index=True)
        else:
            st.success("No skipped or failed rows in the latest paper benchmark.")
    with tabs[4]:
        diff_rows = (paper_benchmark_diff or {}).get("rows") or []
        st.caption((paper_benchmark_diff or {}).get("summary", "No run diff available."))
        if diff_rows:
            st.dataframe(pd.DataFrame(diff_rows), width="stretch", hide_index=True)


def render_deep_model_tuned_dashboard(
    tuned_benchmark: dict,
    tuned_diff: dict | None = None,
    *,
    selected_model_id: str | None = None,
    key_prefix: str = "deep_model_tuned_dashboard",
) -> None:
    st.subheader("Deep Model Battle Report")
    render_term_glossary("benchmark", expanded=False, key_prefix=f"{key_prefix}_glossary")
    if not tuned_benchmark.get("leaderboard"):
        st.info("Run `scripts/run_deep_model_tuned_benchmark.py --preset tuned-smoke` to generate tuned deep-model battle evidence.")
        return

    leaderboard = pd.DataFrame(tuned_benchmark.get("leaderboard") or [])
    battle_cards = pd.DataFrame(tuned_benchmark.get("battle_cards") or [])
    baseline = pd.DataFrame(tuned_benchmark.get("baseline_comparison") or [])
    ablation = pd.DataFrame(tuned_benchmark.get("ablation_comparison") or [])
    runs = tuned_benchmark.get("runs") or []
    seeds = tuned_benchmark.get("seeds") or []
    failure = tuned_benchmark.get("failure_attribution") or {}
    render_stat_grid(
        [
            ("Tier", tuned_benchmark.get("benchmark_tier", tuned_benchmark.get("preset", "NA")), True),
            ("Datasets", f"{len(tuned_benchmark.get('datasets', [])):,}", False),
            ("Variants", f"{len(tuned_benchmark.get('variants', [])):,}", False),
            ("Seeds", f"{len(seeds):,}", False),
            ("Runs", f"{len(runs):,}", False),
            ("Battle Cards", f"{len(battle_cards):,}", False),
        ],
        columns=6,
    )
    st.caption(
        "读法：这不是再做一个模型 zoo，而是把深度 uplift 模型和 T/DR 强基线放到同一 known-CATE benchmark 下，解释 PEHE、QINI、policy value 和 loss curve 的胜负原因。"
    )
    if tuned_benchmark.get("model_scorecards"):
        with st.expander("Battle Interpretation / 面试讲法", expanded=False):
            tracks = tuned_benchmark.get("interview_tracks") or {}
            track_rows = [
                {"duration": "30s", "talk_track": tracks.get("30s", "")},
                {"duration": "5min", "talk_track": tracks.get("5min", "")},
                {"duration": "15min", "talk_track": tracks.get("15min", "")},
                {"duration": "30min", "talk_track": tracks.get("30min", "")},
            ]
            st.dataframe(pd.DataFrame(track_rows), width="stretch", hide_index=True)
            scorecard_df = pd.DataFrame(tuned_benchmark.get("model_scorecards") or [])
            scorecard_cols = [
                col
                for col in [
                    "model",
                    "verdict",
                    "ok_runs",
                    "best_pehe_variant",
                    "best_pehe",
                    "best_qini_variant",
                    "best_qini",
                    "avg_delta_pehe_vs_baseline",
                    "avg_delta_qini_vs_baseline",
                    "watchout",
                    "interview_line",
                ]
                if col in scorecard_df.columns
            ]
            if scorecard_cols:
                st.dataframe(scorecard_df[scorecard_cols], width="stretch", hide_index=True)
            recommendations = tuned_benchmark.get("recommendations") or []
            if recommendations:
                st.caption("Recommended next actions")
                st.dataframe(pd.DataFrame(recommendations), width="stretch", hide_index=True)

    filter_cols = st.columns([0.34, 0.33, 0.33])
    dataset_options = ["All"] + sorted(leaderboard["dataset_id"].dropna().astype(str).unique().tolist())
    model_options = ["All"] + sorted(leaderboard["model"].dropna().astype(str).unique().tolist())
    default_model_index = model_options.index(selected_model_id) if selected_model_id in model_options else 0
    selected_dataset = filter_cols[0].selectbox("Dataset", dataset_options, key=f"{key_prefix}_dataset")
    selected_model = filter_cols[1].selectbox("Model", model_options, index=default_model_index, key=f"{key_prefix}_model")
    ranking_metric = filter_cols[2].selectbox(
        "Battle metric",
        ["PEHE ↓", "QINI ↑", "AUUC ↑", "Policy oracle Top10 ↑", "Train loss delta ↑"],
        key=f"{key_prefix}_metric",
    )
    view = leaderboard.copy()
    if selected_dataset != "All":
        view = view[view["dataset_id"].astype(str) == selected_dataset]
    if selected_model != "All":
        view = view[view["model"].astype(str) == selected_model]
    sort_map = {
        "PEHE ↓": ("pehe_mean", True),
        "QINI ↑": ("qini_mean", False),
        "AUUC ↑": ("auuc_mean", False),
        "Policy oracle Top10 ↑": ("policy_top10_oracle_value_mean", False),
        "Train loss delta ↑": ("train_loss_delta_mean", False),
    }
    sort_col, ascending = sort_map[ranking_metric]
    if sort_col in view.columns:
        view = view.sort_values(sort_col, ascending=ascending, na_position="last")
    display_cols = [
        col
        for col in [
            "dataset_id",
            "variant_id",
            "model",
            "family",
            "ok_runs",
            "pehe_mean",
            "ate_error_mean",
            "qini_mean",
            "auuc_mean",
            "policy_top10_oracle_value_mean",
            "oracle_top10_recall_mean",
            "train_loss_delta_mean",
            "evidence_manifest_count",
        ]
        if col in view.columns
    ]
    st.dataframe(view[display_cols], width="stretch", hide_index=True)

    tabs = st.tabs(["Battle Cards", "Baseline Delta", "Ablation", "Failure Attribution", "Run Diff"])
    with tabs[0]:
        if not battle_cards.empty:
            battle_view = battle_cards.copy()
            if selected_dataset != "All" and "dataset_id" in battle_view:
                battle_view = battle_view[battle_view["dataset_id"].astype(str) == selected_dataset]
            if selected_model != "All" and "model" in battle_view:
                battle_view = battle_view[battle_view["model"].astype(str) == selected_model]
            battle_cols = [
                col
                for col in [
                    "dataset_id",
                    "variant_id",
                    "model",
                    "verdict",
                    "reason",
                    "pehe_mean",
                    "qini_mean",
                    "policy_top10_oracle_value_mean",
                    "train_loss_delta_mean",
                    "interview_line",
                ]
                if col in battle_view.columns
            ]
            st.dataframe(battle_view[battle_cols], width="stretch", hide_index=True)
        else:
            st.info("Battle cards will appear after rerunning the tuned benchmark.")
    with tabs[1]:
        if not baseline.empty:
            baseline_cols = [
                col
                for col in [
                    "dataset_id",
                    "variant_id",
                    "best_baseline",
                    "delta_pehe_vs_baseline",
                    "delta_qini_vs_baseline",
                    "delta_auuc_vs_baseline",
                    "delta_policy_top10_oracle_value_vs_baseline",
                    "verdict",
                    "explanation",
                ]
                if col in baseline.columns
            ]
            st.dataframe(baseline[baseline_cols], width="stretch", hide_index=True)
        else:
            st.info("No baseline comparison rows yet.")
    with tabs[2]:
        if not ablation.empty:
            st.dataframe(ablation, width="stretch", hide_index=True)
        else:
            st.info("No ablation comparison rows yet.")
    with tabs[3]:
        rows = []
        for status, count in sorted((failure.get("status_counts") or {}).items()):
            rows.append({"type": "status", "item": status, "count": count})
        for verdict, count in sorted((failure.get("verdict_counts") or {}).items()):
            rows.append({"type": "verdict", "item": verdict, "count": count})
        for reason, count in sorted((failure.get("reason_counts") or {}).items()):
            rows.append({"type": "reason", "item": reason, "count": count})
        if rows:
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        readings = failure.get("default_failure_reading") or []
        if readings:
            st.caption(" / ".join(str(item) for item in readings[:4]))
    with tabs[4]:
        diff_rows = (tuned_diff or tuned_benchmark.get("run_diff") or {}).get("rows") or []
        st.caption((tuned_diff or tuned_benchmark.get("run_diff") or {}).get("summary", "No tuned benchmark run diff available."))
        if diff_rows:
            st.dataframe(pd.DataFrame(diff_rows), width="stretch", hide_index=True)


def render_evidence_dashboard() -> None:
    st.subheader("Evidence Dashboard")
    st.markdown(
        "这个页面把模型目录、工业来源、LLM routing、训练 smoke、UI 截图和面试证据包放在同一个回归视图里。"
        "目标是让每个简历 claim 都能追到本地 artifact、页面证据和验证脚本。"
    )

    artifacts = latest_report_artifacts()
    model_audit = _read_json_file(Path((artifacts.get("model_audit") or {}).get("path", "")))
    source_audit = _read_json_file(Path((artifacts.get("industry_source_audit") or {}).get("path", "")))
    github_framework_audit = _read_json_file(Path((artifacts.get("github_framework_audit") or {}).get("path", "")))
    regression = _read_json_file(Path((artifacts.get("agent_regression") or {}).get("path", "")))
    llm_dataset_smoke = _read_json_file(Path((artifacts.get("llm_routing_dataset_smoke") or {}).get("path", "")))
    optional_backend_probe = _read_json_file(Path((artifacts.get("optional_backend_probe") or {}).get("path", "")))
    utboost_smoke = _read_json_file(Path((artifacts.get("utboost_guarded_adapter_smoke") or {}).get("path", "")))
    frontier_dataset_smoke = _read_json_file(Path((artifacts.get("frontier_synthetic_datasets_smoke") or {}).get("path", "")))
    frontier_evaluator_smoke = _read_json_file(Path((artifacts.get("frontier_evaluator_metrics_smoke") or {}).get("path", "")))
    industrial_dataset_smoke = _read_json_file(Path((artifacts.get("industrial_scenario_datasets_smoke") or {}).get("path", "")))
    industrial_training_smoke = _read_json_file(Path((artifacts.get("industrial_scenario_training_smoke") or {}).get("path", "")))
    industrial_policy_smoke = _read_json_file(Path((artifacts.get("industrial_policy_metrics_smoke") or {}).get("path", "")))
    industrial_compare_smoke = _read_json_file(Path((artifacts.get("industrial_scenario_compare_smoke") or {}).get("path", "")))
    resume_policy_smoke = _read_json_file(Path((artifacts.get("resume_scenario_policy_smoke") or {}).get("path", "")))
    demo_trace_smoke = _read_json_file(Path((artifacts.get("demo_preset_source_trace") or {}).get("path", "")))
    frontier_training_evidence = _read_json_file(Path((artifacts.get("frontier_training_evidence") or {}).get("path", "")))
    deep_loss_architecture_smoke = _read_json_file(Path((artifacts.get("deep_loss_architecture_catalog") or {}).get("path", "")))
    deep_loss_functions_smoke = _read_json_file(Path((artifacts.get("deep_loss_functions_smoke") or {}).get("path", "")))
    cfrnet_balance_smoke = _read_json_file(Path((artifacts.get("cfrnet_differentiable_balance_smoke") or {}).get("path", "")))
    cfrnet_training_smoke = _read_json_file(Path((artifacts.get("cfrnet_training_balance_smoke") or {}).get("path", "")))
    contrastive_training_smoke = _read_json_file(Path((artifacts.get("contrastive_uplift_training_smoke") or {}).get("path", "")))
    deep_neural_ablation = _read_json_file(Path((artifacts.get("deep_neural_ablation") or {}).get("path", "")))
    deep_model_training_evidence = _read_json_file(Path((artifacts.get("deep_model_training_evidence") or {}).get("path", "")))
    deep_model_tuned_benchmark = _read_json_file(Path((artifacts.get("deep_model_tuned_benchmark") or {}).get("path", "")))
    deep_model_tuned_diff = _read_json_file(Path((artifacts.get("deep_model_tuned_benchmark_run_diff") or {}).get("path", "")))
    deep_model_objective_diagnostics = _read_json_file(Path((artifacts.get("deep_model_objective_diagnostics") or {}).get("path", "")))
    algorithm_claim_ledger = _read_json_file(Path((artifacts.get("algorithm_claim_ledger") or {}).get("path", "")))
    paper_reproduction_gap_ledger = _read_json_file(Path((artifacts.get("paper_reproduction_gap_ledger") or {}).get("path", "")))
    model_evidence_promotion_matrix = _read_json_file(Path((artifacts.get("model_evidence_promotion_matrix") or {}).get("path", "")))
    model_upgrade_recipe_cards = _read_json_file(Path((artifacts.get("model_upgrade_recipe_cards") or {}).get("path", "")))
    deep_model_telemetry_gap_matrix = _read_json_file(Path((artifacts.get("deep_model_telemetry_gap_matrix") or {}).get("path", "")))
    deep_model_telemetry_smoke = _read_json_file(Path((artifacts.get("deep_model_telemetry_smoke") or {}).get("path", "")))
    deep_model_forward_hook_contracts = _read_json_file(Path((artifacts.get("deep_model_forward_hook_contracts") or {}).get("path", "")))
    deep_model_forward_hook_smoke = _read_json_file(Path((artifacts.get("deep_model_forward_hook_smoke") or {}).get("path", "")))
    deep_model_hook_training_bridge = _read_json_file(Path((artifacts.get("deep_model_hook_training_bridge") or {}).get("path", "")))
    deep_model_trainer_collector_smoke = _read_json_file(Path((artifacts.get("deep_model_trainer_collector_smoke") or {}).get("path", "")))
    deep_model_telemetry_benchmark_linkage = _read_json_file(Path((artifacts.get("deep_model_telemetry_benchmark_linkage") or {}).get("path", "")))
    deep_model_telemetry_ablation_gate = _read_json_file(Path((artifacts.get("deep_model_telemetry_ablation_gate") or {}).get("path", "")))
    deep_model_ablation_interpretation = _read_json_file(Path((artifacts.get("deep_model_ablation_interpretation") or {}).get("path", "")))
    deep_model_ablation_promotion_matrix = _read_json_file(Path((artifacts.get("deep_model_ablation_promotion_matrix") or {}).get("path", "")))
    deep_model_ablation_next_experiment_plan = _read_json_file(Path((artifacts.get("deep_model_ablation_next_experiment_plan") or {}).get("path", "")))
    deep_model_ablation_command_contract = _read_json_file(Path((artifacts.get("deep_model_ablation_command_contract") or {}).get("path", "")))
    deep_model_ablation_command_contract_smoke = _read_json_file(Path((artifacts.get("deep_model_ablation_command_contract_smoke") or {}).get("path", "")))
    deep_model_ablation_recommended_command_smoke = _read_json_file(Path((artifacts.get("deep_model_ablation_recommended_command_smoke") or {}).get("path", "")))
    deep_model_ablation_command_smoke_coverage = _read_json_file(Path((artifacts.get("deep_model_ablation_command_smoke_coverage") or {}).get("path", "")))
    deep_model_ablation_command_smoke_coverage_diff = _read_json_file(Path((artifacts.get("deep_model_ablation_command_smoke_coverage_diff") or {}).get("path", "")))
    paper_benchmark = _read_json_file(Path((artifacts.get("paper_benchmark") or {}).get("path", "")))
    paper_benchmark_diff = _read_json_file(Path((artifacts.get("paper_benchmark_run_diff") or {}).get("path", "")))
    paper_benchmark_interpretation = _read_json_file(Path((artifacts.get("paper_benchmark_interpretation") or {}).get("path", "")))
    regression_warning_audit = _read_json_file(Path((artifacts.get("regression_warning_audit") or {}).get("path", "")))
    promotion_launch_cards = _read_json_file(Path((artifacts.get("promotion_launch_cards") or {}).get("path", "")))
    regression_warning_triage = _read_json_file(Path((artifacts.get("regression_warning_triage") or {}).get("path", "")))
    pilot_readiness_scorecard = _read_json_file(Path((artifacts.get("pilot_readiness_scorecard") or {}).get("path", "")))
    pilot_experiment_plan = _read_json_file(Path((artifacts.get("pilot_experiment_plan") or {}).get("path", "")))
    pilot_experiment_command_smoke = _read_json_file(Path((artifacts.get("pilot_experiment_command_smoke") or {}).get("path", "")))
    pilot_experiment_command_coverage = _read_json_file(Path((artifacts.get("pilot_experiment_command_coverage") or {}).get("path", "")))
    model_deconstruction_catalog = _read_json_file(Path((artifacts.get("model_deconstruction_catalog") or {}).get("path", "")))
    model_deconstruction_agent = _read_json_file(Path((artifacts.get("model_deconstruction_agent") or {}).get("path", "")))
    diagnostic_cases_artifact = _read_json_file(Path((artifacts.get("uplift_diagnostic_cases") or {}).get("path", "")))
    failure_modes_smoke = _read_json_file(Path((artifacts.get("uplift_failure_modes_smoke") or {}).get("path", "")))
    llm_smoke = _read_json_file(Path((artifacts.get("llm_routing_policy_smoke") or {}).get("path", "")))
    llm_ope_smoke = _read_json_file(Path((artifacts.get("llm_routing_ope_smoke") or {}).get("path", "")))
    continuous_policy_smoke = _read_json_file(Path((artifacts.get("continuous_treatment_policy_smoke") or {}).get("path", "")))
    screenshots = artifacts.get("ui_screenshots") or {}
    interview_zip_path = Path((artifacts.get("interview_zip") or {}).get("path", ""))
    llm_base = (llm_smoke.get("scenarios") or {}).get("base", {})
    quality_counts = source_audit.get("quality_counts") or {}

    render_optimization_closure_dashboard(key_prefix="evidence_closure", artifacts=artifacts)
    render_repo_change_intake_dashboard(key_prefix="evidence_repo_change_intake", artifacts=artifacts)
    render_visual_regression_evidence(screenshots)
    st.subheader("ML Training Platform Source Gate")
    st.caption("训练平台 UI/UX 参考只采用官方 GitHub repo / 官方 docs；GitHub API rate-limit 时使用官方页面与 commit feed 交叉验证。")
    platform_gate = pd.DataFrame(ml_training_platform_audit_rows())
    if not platform_gate.empty:
        st.dataframe(
            platform_gate[
                [
                    "platform",
                    "repo",
                    "stars",
                    "latest_update",
                    "activity",
                    "page_structure",
                    "status",
                    "github_url",
                    "official_url",
                ]
            ],
            width="stretch",
            hide_index=True,
        )
    st.subheader("Public Product Source Gate")
    st.caption("公开产品参考仅来自火山引擎官网、官方文档、开发者社区公开文章和官方 GitHub/OSS；只吸收产品结构，不复制品牌或私有 UI。")
    public_gate = pd.DataFrame(public_product_audit_rows())
    if not public_gate.empty:
        st.dataframe(
            public_gate[
                [
                    "product",
                    "area",
                    "source_type",
                    "activity",
                    "page_structure",
                    "status",
                    "official_url",
                ]
            ],
            width="stretch",
            hide_index=True,
        )

    render_stat_grid(
        [
            ("Models", f"{model_audit.get('registered_models', 'NA')} / {model_audit.get('ready_models', 'NA')}", False),
            ("Industry Sources", f"{source_audit.get('references', 'NA')}", False),
            ("Official Sources", f"{quality_counts.get('official', 0)}", False),
            ("LLM Dataset Rows", f"{llm_dataset_smoke.get('rows', 'NA')}", False),
            ("Optional Backends", f"{optional_backend_probe.get('status', 'NA')}", False),
            ("UTBoost Guard", f"{utboost_smoke.get('status', 'NA')}", False),
            ("Frontier Datasets", f"{len(frontier_dataset_smoke.get('datasets', [])) if frontier_dataset_smoke else 'NA'}", False),
            ("Frontier Metrics", f"{frontier_evaluator_smoke.get('status', 'NA')}", False),
            ("Scenario Datasets", f"{len(industrial_dataset_smoke.get('datasets', [])) if industrial_dataset_smoke else 'NA'}", False),
            ("Scenario Training", f"{industrial_training_smoke.get('status', 'NA')}", False),
            ("Scenario Policy", f"{industrial_policy_smoke.get('status', 'NA')}", False),
            ("Scenario Compare", f"{industrial_compare_smoke.get('status', 'NA')}", False),
            ("Resume Policy", f"{resume_policy_smoke.get('status', 'NA')}", False),
            ("Demo Trace", f"{demo_trace_smoke.get('status', 'NA')}", False),
            ("Frontier Training", f"{len(frontier_training_evidence.get('rows', [])) if frontier_training_evidence else 'NA'}", False),
            ("Deep Loss Catalog", f"{deep_loss_architecture_smoke.get('status', 'NA')}", False),
            ("Deep Loss Fn", f"{deep_loss_functions_smoke.get('status', 'NA')}", False),
            ("CFR Balance", f"{cfrnet_balance_smoke.get('status', 'NA')}", False),
            ("CFR Train", f"{cfrnet_training_smoke.get('status', 'NA')}", False),
            ("Contrastive Train", f"{contrastive_training_smoke.get('status', 'NA')}", False),
            ("Deep Ablation", f"{deep_neural_ablation.get('status', 'NA')}", False),
            ("Deep Model Evidence", f"{deep_model_training_evidence.get('ok_models', 'NA')}", False),
            ("Deep Battle", f"{len(deep_model_tuned_benchmark.get('battle_cards', [])) if deep_model_tuned_benchmark else 'NA'}", False),
            ("Deep Objective", f"{(deep_model_objective_diagnostics.get('summary') or {}).get('models', 'NA')}", False),
            ("Claim Ledger", f"{(algorithm_claim_ledger.get('summary') or {}).get('claims', 'NA')}", False),
            ("Paper Gaps", f"{(paper_reproduction_gap_ledger.get('summary') or {}).get('source_rows', 'NA')}", False),
            ("Promotion Matrix", f"{(model_evidence_promotion_matrix.get('summary') or {}).get('models', 'NA')}", False),
            ("Upgrade Recipes", f"{(model_upgrade_recipe_cards.get('summary') or {}).get('recipes', 'NA')}", False),
            ("Telemetry Gaps", f"{(deep_model_telemetry_gap_matrix.get('summary') or {}).get('telemetry_rows', 'NA')}", False),
            ("Telemetry Smoke", f"{(deep_model_telemetry_smoke.get('summary') or {}).get('basic_telemetry_pass', 'NA')}", False),
            ("Forward Hooks", f"{(deep_model_forward_hook_contracts.get('summary') or {}).get('contracts', 'NA')}", False),
            ("Hook Smoke", f"{(deep_model_forward_hook_smoke.get('summary') or {}).get('ok_contracts', 'NA')}", False),
            ("Hook Bridge", f"{(deep_model_hook_training_bridge.get('summary') or {}).get('bridge_ready', 'NA')}", False),
            ("Trainer Collectors", f"{(deep_model_trainer_collector_smoke.get('summary') or {}).get('ok_models', 'NA')}", False),
            ("Telemetry Links", f"{(deep_model_telemetry_benchmark_linkage.get('summary') or {}).get('link_ready', 'NA')}", False),
            ("Ablation Gates", f"{(deep_model_telemetry_ablation_gate.get('summary') or {}).get('gates', 'NA')}", False),
            ("Ablation Rows", f"{(deep_model_ablation_interpretation.get('summary') or {}).get('rows', 'NA')}", False),
            ("Ablation Promo", f"{(deep_model_ablation_promotion_matrix.get('summary') or {}).get('promotion_ready_rows', 'NA')}", True),
            ("Ablation Plan", f"{(deep_model_ablation_next_experiment_plan.get('summary') or {}).get('p0_rows', 'NA')}", True),
            ("Command Contract", f"{(deep_model_ablation_command_contract.get('summary') or {}).get('fallback_ready_rows', 'NA')}", True),
            ("Command Smoke", f"{(deep_model_ablation_command_contract_smoke.get('summary') or {}).get('ok_rows', 'NA')}", True),
            ("Recommended Cmd", f"{(deep_model_ablation_recommended_command_smoke.get('summary') or {}).get('ok_rows', 'NA')}", True),
            ("Smoke Coverage", f"{(deep_model_ablation_command_smoke_coverage.get('summary') or {}).get('any_smoked_rows', 'NA')}", True),
            ("Coverage Diff", f"{(deep_model_ablation_command_smoke_coverage_diff.get('summary') or {}).get('coverage_gate_delta', 'NA')}", True),
            ("Paper Bench", f"{len(paper_benchmark.get('leaderboard', [])) if paper_benchmark else 'NA'}", False),
            ("Warning Audit", f"{(regression_warning_audit.get('summary') or {}).get('total_warnings', 'NA')}", False),
            ("Launch Cards", f"{(promotion_launch_cards.get('summary') or {}).get('total_cards', 'NA')}", False),
            ("Warning Triage", f"{(regression_warning_triage.get('summary') or {}).get('triage_gate', 'NA')}", True),
            ("Pilot Gate", f"{(pilot_readiness_scorecard.get('summary') or {}).get('pilot_gate', 'NA')}", True),
            ("Pilot P0", f"{(pilot_experiment_plan.get('summary') or {}).get('p0_rows', 'NA')}", True),
            ("Pilot Cmd", f"{(pilot_experiment_command_smoke.get('summary') or {}).get('ok_rows', 'NA')}", True),
            ("Pilot Cvg", f"{(pilot_experiment_command_coverage.get('summary') or {}).get('sampled_smoked_ok_rows', 'NA')}", True),
            ("Model Decon", f"{(model_deconstruction_catalog.get('coverage') or {}).get('models', 'NA')}", False),
            ("Decon Agent", f"{model_deconstruction_agent.get('status', 'NA')}", False),
            ("Failure Cases", f"{len(diagnostic_cases_artifact.get('cases', [])) if diagnostic_cases_artifact else 'NA'}", False),
            ("Failure Smoke", f"{failure_modes_smoke.get('status', 'NA')}", False),
            ("LLM Routing Net", f"{llm_base.get('net_value', 'NA')}", False),
            ("LLM OPE", f"{llm_ope_smoke.get('status', 'NA')}", False),
            ("Screenshots", f"{screenshots.get('files', 0)}", False),
            ("Interview ZIP", f"{interview_zip_path.stat().st_size if interview_zip_path.is_file() else 'missing'}", True),
        ],
        columns=3,
    )

    st.subheader("Regression Gate Artifacts")
    artifact_rows = []
    for name, payload in artifacts.items():
        path = Path(str(payload.get("path", "")))
        artifact_rows.append(
            {
                "artifact": name,
                "path": str(path),
                "modified": payload.get("modified"),
                "files": str(payload.get("files", "")),
                "bytes": str(payload.get("bytes", path.stat().st_size if path.is_file() else "")),
            }
        )
    st.dataframe(pd.DataFrame(artifact_rows), width="stretch", hide_index=True)

    st.subheader("Regression Warning / Launch Guardrail Audit")
    warning_summary = regression_warning_audit.get("summary") or {}
    warning_rows = regression_warning_audit.get("warnings") or []
    if warning_rows:
        severity_counts = warning_summary.get("severity_counts") or {}
        render_stat_grid(
            [
                ("Launch Gate", warning_summary.get("launch_gate", "NA"), True),
                ("Warnings", f"{warning_summary.get('total_warnings', len(warning_rows))}", False),
                ("High", f"{severity_counts.get('high', 0)}", False),
                ("Medium", f"{severity_counts.get('medium', 0)}", False),
                ("Low", f"{severity_counts.get('low', 0)}", False),
            ],
            columns=5,
        )
        st.caption(
            "这张表把 smoke/benchmark/industrial compare/OPE/evidence store 的风险信号变成上线前 guardrail："
            "能跑不等于能上线，warning 需要能解释、能归因、能进入下一步实验。"
        )
        warning_df = pd.DataFrame(warning_rows)
        filter_cols = st.columns([0.25, 0.25, 0.5])
        severity_options = ["All"] + sorted(warning_df["severity"].dropna().astype(str).unique().tolist())
        domain_options = ["All"] + sorted(warning_df["domain"].dropna().astype(str).unique().tolist())
        selected_severity = filter_cols[0].selectbox("Severity", severity_options, key="evidence_warning_severity")
        selected_domain = filter_cols[1].selectbox("Domain", domain_options, key="evidence_warning_domain")
        search_text = filter_cols[2].text_input("Search warning/entity", value="", key="evidence_warning_search")
        warning_view = warning_df.copy()
        if selected_severity != "All":
            warning_view = warning_view[warning_view["severity"].astype(str) == selected_severity]
        if selected_domain != "All":
            warning_view = warning_view[warning_view["domain"].astype(str) == selected_domain]
        if search_text:
            haystack = warning_view.astype(str).agg(" ".join, axis=1)
            warning_view = warning_view[haystack.str.contains(search_text, case=False, na=False)]
        display_cols = [
            col
            for col in [
                "severity",
                "domain",
                "entity",
                "signal",
                "value",
                "promotion_decision",
                "owner_role",
                "next_experiment",
                "ui_route",
                "interview_priority",
                "why_it_matters",
                "recommended_action",
                "interview_line",
            ]
            if col in warning_view.columns
        ]
        warning_display = warning_view[display_cols].copy()
        if "value" in warning_display.columns:
            warning_display["value"] = warning_display["value"].map(
                lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True)
                if isinstance(value, (dict, list))
                else str(value)
            )
        st.dataframe(warning_display, width="stretch", hide_index=True)
        doc_path = Path("docs/DEEPUplift_REGRESSION_WARNING_AUDIT.md")
        if doc_path.exists():
            st.download_button(
                "Download regression warning audit",
                data=doc_path.read_bytes(),
                file_name=doc_path.name,
                mime="text/markdown",
                key="download_regression_warning_audit",
            )
    else:
        st.info("Run `scripts/generate_regression_warning_audit.py` to refresh launch guardrail warnings.")

    st.subheader("Promotion / Launch Review Cards")
    promotion_summary = promotion_launch_cards.get("summary") or {}
    promotion_cards = promotion_launch_cards.get("cards") or []
    if promotion_cards:
        decision_counts = promotion_summary.get("decision_counts") or {}
        render_stat_grid(
            [
                ("Launch Gate", promotion_summary.get("launch_gate", "NA"), True),
                ("Cards", promotion_summary.get("total_cards", len(promotion_cards)), False),
                ("Blocked", decision_counts.get("block_promotion", 0), False),
                ("Review", decision_counts.get("review_before_promotion", 0), False),
                ("Monitor", decision_counts.get("monitor", 0), False),
            ],
            columns=5,
        )
        st.caption(
            "把 warning rows 聚合成 review cards：每张卡说明是否能 promotion、谁负责、上线前还缺什么实验、哪些 claim 不能讲。"
        )
        cards_df = pd.DataFrame(promotion_cards)
        card_filter_cols = st.columns([0.3, 0.3, 0.4])
        decision_options = ["All"] + sorted(cards_df["overall_decision"].dropna().astype(str).unique().tolist())
        owner_options = ["All"] + sorted(cards_df["owner_role"].dropna().astype(str).unique().tolist())
        selected_decision = card_filter_cols[0].selectbox("Promotion decision", decision_options, key="promotion_card_decision")
        selected_owner = card_filter_cols[1].selectbox("Owner", owner_options, key="promotion_card_owner")
        card_search = card_filter_cols[2].text_input("Search card/entity", value="", key="promotion_card_search")
        card_view = cards_df.copy()
        if selected_decision != "All":
            card_view = card_view[card_view["overall_decision"].astype(str) == selected_decision]
        if selected_owner != "All":
            card_view = card_view[card_view["owner_role"].astype(str) == selected_owner]
        if card_search:
            haystack = card_view.astype(str).agg(" ".join, axis=1)
            card_view = card_view[haystack.str.contains(card_search, case=False, na=False)]
        display_cols = [
            col
            for col in [
                "card_id",
                "domain",
                "entity",
                "warning_count",
                "overall_decision",
                "promotion_stage",
                "owner_role",
                "primary_blocker",
                "primary_reason",
                "safe_claim",
                "unsafe_claim",
                "before_shadow",
                "before_ab",
                "before_ramp",
                "metrics_to_watch",
                "interview_line",
            ]
            if col in card_view.columns
        ]
        st.dataframe(card_view[display_cols], width="stretch", hide_index=True)
        doc_path = Path("docs/DEEPUplift_PROMOTION_LAUNCH_CARDS.md")
        if doc_path.exists():
            st.download_button(
                "Download promotion launch cards",
                data=doc_path.read_bytes(),
                file_name=doc_path.name,
                mime="text/markdown",
                key="download_promotion_launch_cards",
            )
    else:
        st.info("Run `scripts/generate_promotion_launch_cards.py` after warning audit to refresh launch review cards.")

    st.subheader("Regression Warning Triage")
    triage_summary = regression_warning_triage.get("summary") or {}
    triage_rows = regression_warning_triage.get("rows") or []
    if triage_rows:
        render_stat_grid(
            [
                ("Triage Gate", triage_summary.get("triage_gate", "NA"), True),
                ("Rows", triage_summary.get("warning_rows", len(triage_rows)), False),
                ("Smoke-only", triage_summary.get("smoke_only_rows", "NA"), True),
                ("Launch Blockers", triage_summary.get("launch_blocking_rows", "NA"), False),
                ("Review", triage_summary.get("review_rows", "NA"), False),
            ],
            columns=5,
        )
        st.caption(
            "这一层不消除 warning，而是把每条 warning 标成 claim boundary 和 action lane："
            "哪些只是小样本 smoke caveat，哪些会阻塞 shadow/A/B/pilot。"
        )
        triage_df = pd.DataFrame(triage_rows)
        triage_filter_cols = st.columns([0.25, 0.25, 0.25, 0.25])
        family_options = ["All"] + sorted(triage_df["risk_family"].dropna().astype(str).unique().tolist())
        boundary_options = ["All"] + sorted(triage_df["claim_boundary"].dropna().astype(str).unique().tolist())
        action_options = ["All"] + sorted(triage_df["action_lane"].dropna().astype(str).unique().tolist())
        selected_family = triage_filter_cols[0].selectbox("Risk family", family_options, key="warning_triage_family")
        selected_boundary = triage_filter_cols[1].selectbox("Claim boundary", boundary_options, key="warning_triage_boundary")
        selected_action = triage_filter_cols[2].selectbox("Action lane", action_options, key="warning_triage_action")
        triage_search = triage_filter_cols[3].text_input("Search triage", value="", key="warning_triage_search")
        triage_view = triage_df.copy()
        if selected_family != "All":
            triage_view = triage_view[triage_view["risk_family"].astype(str) == selected_family]
        if selected_boundary != "All":
            triage_view = triage_view[triage_view["claim_boundary"].astype(str) == selected_boundary]
        if selected_action != "All":
            triage_view = triage_view[triage_view["action_lane"].astype(str) == selected_action]
        if triage_search:
            haystack = triage_view.astype(str).agg(" ".join, axis=1)
            triage_view = triage_view[haystack.str.contains(triage_search, case=False, na=False)]
        triage_cols = [
            col
            for col in [
                "severity",
                "domain",
                "entity",
                "signal",
                "risk_family",
                "triage_status",
                "claim_boundary",
                "action_lane",
                "smoke_only",
                "launch_blocking",
                "owner_role",
                "safe_claim",
                "unsafe_claim",
                "next_experiment",
            ]
            if col in triage_view.columns
        ]
        st.dataframe(triage_view[triage_cols], width="stretch", hide_index=True)
        doc_path = Path("docs/DEEPUplift_REGRESSION_WARNING_TRIAGE.md")
        if doc_path.exists():
            st.download_button(
                "Download regression warning triage",
                data=doc_path.read_bytes(),
                file_name=doc_path.name,
                mime="text/markdown",
                key="download_regression_warning_triage",
            )
    else:
        st.info("Run `scripts/generate_regression_warning_triage.py` after warning audit and launch cards.")

    st.subheader("Pilot Readiness Scorecard")
    pilot_summary = pilot_readiness_scorecard.get("summary") or {}
    pilot_rows = pilot_readiness_scorecard.get("scorecards") or []
    if pilot_rows:
        render_stat_grid(
            [
                ("Pilot Gate", pilot_summary.get("pilot_gate", "NA"), True),
                ("Scorecards", pilot_summary.get("scorecards", len(pilot_rows)), False),
                ("Blocked", pilot_summary.get("blocked", "NA"), False),
                ("Paper Review", pilot_summary.get("paper_review_candidates", "NA"), False),
                ("Shadow", pilot_summary.get("shadow_candidates", "NA"), False),
                ("Monitor", pilot_summary.get("monitor", "NA"), False),
            ],
            columns=6,
        )
        st.caption(
            "这一层把 promotion cards 和 warning triage 汇总成 pilot gate："
            "哪些只能 offline，哪些适合 paper-level 评审，哪些可以进入 shadow scoring，以及进入 A/B 前还缺哪些 artifact。"
        )
        pilot_df = pd.DataFrame(pilot_rows)
        pilot_filter_cols = st.columns([0.25, 0.25, 0.25, 0.25])
        gate_options = ["All"] + sorted(pilot_df["pilot_gate"].dropna().astype(str).unique().tolist())
        grade_options = ["All"] + sorted(pilot_df["evidence_grade"].dropna().astype(str).unique().tolist())
        owner_options = ["All"] + sorted(pilot_df["owner_role"].dropna().astype(str).unique().tolist())
        selected_gate = pilot_filter_cols[0].selectbox("Pilot gate", gate_options, key="pilot_scorecard_gate")
        selected_grade = pilot_filter_cols[1].selectbox("Evidence grade", grade_options, key="pilot_scorecard_grade")
        selected_owner = pilot_filter_cols[2].selectbox("Pilot owner", owner_options, key="pilot_scorecard_owner")
        pilot_search = pilot_filter_cols[3].text_input("Search pilot scorecard", value="", key="pilot_scorecard_search")
        pilot_view = pilot_df.copy()
        if selected_gate != "All":
            pilot_view = pilot_view[pilot_view["pilot_gate"].astype(str) == selected_gate]
        if selected_grade != "All":
            pilot_view = pilot_view[pilot_view["evidence_grade"].astype(str) == selected_grade]
        if selected_owner != "All":
            pilot_view = pilot_view[pilot_view["owner_role"].astype(str) == selected_owner]
        if pilot_search:
            haystack = pilot_view.astype(str).agg(" ".join, axis=1)
            pilot_view = pilot_view[haystack.str.contains(pilot_search, case=False, na=False)]
        pilot_cols = [
            col
            for col in [
                "scorecard_id",
                "domain",
                "entity",
                "pilot_gate",
                "readiness_score",
                "evidence_grade",
                "owner_role",
                "launch_blockers",
                "review_rows",
                "primary_blocker",
                "next_gate",
                "pilot_path",
                "safe_claim",
                "unsafe_claim",
                "rollback_conditions",
                "required_artifacts",
                "interview_line",
            ]
            if col in pilot_view.columns
        ]
        st.dataframe(pilot_view[pilot_cols], width="stretch", hide_index=True)
        doc_path = Path("docs/DEEPUplift_PILOT_READINESS_SCORECARD.md")
        if doc_path.exists():
            st.download_button(
                "Download pilot readiness scorecard",
                data=doc_path.read_bytes(),
                file_name=doc_path.name,
                mime="text/markdown",
                key="download_pilot_readiness_scorecard",
            )
    else:
        st.info("Run `scripts/generate_pilot_readiness_scorecard.py` after warning triage.")

    st.subheader("Pilot Experiment Plan")
    experiment_summary = pilot_experiment_plan.get("summary") or {}
    experiment_rows = pilot_experiment_plan.get("rows") or []
    if experiment_rows:
        experiment_type_counts = experiment_summary.get("experiment_type_counts") or {}
        render_stat_grid(
            [
                ("Experiments", experiment_summary.get("experiments", len(experiment_rows)), False),
                ("P0", experiment_summary.get("p0_rows", "NA"), True),
                ("P1", experiment_summary.get("p1_rows", "NA"), False),
                ("P2", experiment_summary.get("p2_rows", "NA"), False),
                ("Shadow/Monitor", experiment_summary.get("shadow_or_monitor_rows", "NA"), True),
                ("Paper Review", experiment_summary.get("paper_review_rows", "NA"), False),
            ],
            columns=6,
        )
        st.caption(
            "这一层把 readiness gate 转成可执行实验：每一行都有推荐命令、需要观察的指标、预期 artifact、pass gate、fail action 和 claim upgrade 边界。"
        )
        experiment_df = pd.DataFrame(experiment_rows)
        experiment_filter_cols = st.columns([0.22, 0.26, 0.22, 0.30])
        priority_options = ["All"] + sorted(experiment_df["priority"].dropna().astype(str).unique().tolist())
        type_options = ["All"] + sorted(experiment_df["experiment_type"].dropna().astype(str).unique().tolist())
        gate_options = ["All"] + sorted(experiment_df["pilot_gate"].dropna().astype(str).unique().tolist())
        selected_priority = experiment_filter_cols[0].selectbox("Experiment priority", priority_options, key="pilot_experiment_priority")
        selected_type = experiment_filter_cols[1].selectbox("Experiment type", type_options, key="pilot_experiment_type")
        selected_gate = experiment_filter_cols[2].selectbox("Experiment gate", gate_options, key="pilot_experiment_gate")
        experiment_search = experiment_filter_cols[3].text_input("Search pilot experiment", value="", key="pilot_experiment_search")
        experiment_view = experiment_df.copy()
        if selected_priority != "All":
            experiment_view = experiment_view[experiment_view["priority"].astype(str) == selected_priority]
        if selected_type != "All":
            experiment_view = experiment_view[experiment_view["experiment_type"].astype(str) == selected_type]
        if selected_gate != "All":
            experiment_view = experiment_view[experiment_view["pilot_gate"].astype(str) == selected_gate]
        if experiment_search:
            haystack = experiment_view.astype(str).agg(" ".join, axis=1)
            experiment_view = experiment_view[haystack.str.contains(experiment_search, case=False, na=False)]
        experiment_cols = [
            col
            for col in [
                "experiment_id",
                "entity",
                "pilot_gate",
                "readiness_score",
                "priority",
                "experiment_type",
                "owner_role",
                "hypothesis",
                "recommended_command",
                "metrics_to_watch",
                "expected_artifacts",
                "pass_gate",
                "fail_action",
                "claim_upgrade",
                "interview_line",
            ]
            if col in experiment_view.columns
        ]
        st.dataframe(experiment_view[experiment_cols], width="stretch", hide_index=True)
        if experiment_type_counts:
            st.caption(f"Experiment type counts: `{experiment_type_counts}`")
        doc_path = Path("docs/DEEPUplift_PILOT_EXPERIMENT_PLAN.md")
        if doc_path.exists():
            st.download_button(
                "Download pilot experiment plan",
                data=doc_path.read_bytes(),
                file_name=doc_path.name,
                mime="text/markdown",
                key="download_pilot_experiment_plan",
            )
    else:
        st.info("Run `scripts/generate_pilot_experiment_plan.py` after pilot readiness scorecard.")

    st.subheader("Pilot Experiment Command Smoke")
    command_smoke_summary = pilot_experiment_command_smoke.get("summary") or {}
    command_smoke_rows = pilot_experiment_command_smoke.get("rows") or []
    if command_smoke_rows:
        render_stat_grid(
            [
                ("Smoke Rows", command_smoke_summary.get("smoke_rows", len(command_smoke_rows)), False),
                ("OK", command_smoke_summary.get("ok_rows", "NA"), True),
                ("Failed", command_smoke_summary.get("failed_rows", "NA"), True),
                ("Source Plan", command_smoke_summary.get("source_plan_rows", "NA"), False),
                ("Run All", command_smoke_summary.get("run_all", "NA"), False),
                ("Coverage", command_smoke_summary.get("coverage_note", "sampled"), False),
            ],
            columns=6,
        )
        st.caption(
            "这一层验证 Pilot Experiment Plan 里的命令是否真的能小样本执行。"
            "Smoke 使用隔离 output prefix 和 tiny-row override，不覆盖主 benchmark latest；它证明命令路径可跑，不等于完成线上试点。"
        )
        command_smoke_df = pd.DataFrame(command_smoke_rows)
        command_cols = [
            col
            for col in [
                "experiment_id",
                "entity",
                "priority",
                "pilot_gate",
                "experiment_type",
                "status",
                "elapsed_seconds",
                "runs",
                "ok_runs",
                "runner_json",
                "runner_manifest",
                "safe_smoke_command",
                "pass_gate",
                "fail_action",
                "claim_upgrade",
            ]
            if col in command_smoke_df.columns
        ]
        st.dataframe(command_smoke_df[command_cols], width="stretch", hide_index=True)
        doc_path = Path("docs/DEEPUplift_PILOT_EXPERIMENT_COMMAND_SMOKE.md")
        if doc_path.exists():
            st.download_button(
                "Download pilot experiment command smoke",
                data=doc_path.read_bytes(),
                file_name=doc_path.name,
                mime="text/markdown",
                key="download_pilot_experiment_command_smoke",
            )
    else:
        st.info("Run `scripts/smoke_pilot_experiment_commands.py --limit 3` after pilot experiment plan.")

    st.subheader("Pilot Experiment Command Coverage")
    command_coverage_summary = pilot_experiment_command_coverage.get("summary") or {}
    command_coverage_rows = pilot_experiment_command_coverage.get("rows") or []
    if command_coverage_rows:
        render_stat_grid(
            [
                ("Plan Rows", command_coverage_summary.get("plan_rows", len(command_coverage_rows)), False),
                ("Sampled OK", command_coverage_summary.get("sampled_smoked_ok_rows", "NA"), True),
                ("Backlog", command_coverage_summary.get("backlog_rows", "NA"), True),
                ("Coverage", command_coverage_summary.get("coverage_pct", "NA"), False),
                ("P0 Coverage", command_coverage_summary.get("p0_coverage_pct", "NA"), True),
                ("Full Gate", command_coverage_summary.get("full_coverage_gate", "NA"), True),
            ],
            columns=6,
        )
        st.caption(
            "这一层把 plan 和 smoke 关联起来：哪些试点实验只是 runner-supported backlog，哪些已经有 sampled command-smoke 证据。"
            "它防止把“有计划”误讲成“全量已验证”，也是上线评审里的 evidence coverage matrix。"
        )
        coverage_df = pd.DataFrame(command_coverage_rows)
        coverage_filter_cols = st.columns([0.22, 0.24, 0.24, 0.30])
        priority_options = ["All"] + sorted(coverage_df["priority"].dropna().astype(str).unique().tolist())
        status_options = ["All"] + sorted(coverage_df["coverage_status"].dropna().astype(str).unique().tolist())
        type_options = ["All"] + sorted(coverage_df["experiment_type"].dropna().astype(str).unique().tolist())
        selected_priority = coverage_filter_cols[0].selectbox("Coverage priority", priority_options, key="pilot_command_coverage_priority")
        selected_status = coverage_filter_cols[1].selectbox("Coverage status", status_options, key="pilot_command_coverage_status")
        selected_type = coverage_filter_cols[2].selectbox("Coverage type", type_options, key="pilot_command_coverage_type")
        coverage_search = coverage_filter_cols[3].text_input("Search command coverage", value="", key="pilot_command_coverage_search")
        coverage_view = coverage_df.copy()
        if selected_priority != "All":
            coverage_view = coverage_view[coverage_view["priority"].astype(str) == selected_priority]
        if selected_status != "All":
            coverage_view = coverage_view[coverage_view["coverage_status"].astype(str) == selected_status]
        if selected_type != "All":
            coverage_view = coverage_view[coverage_view["experiment_type"].astype(str) == selected_type]
        if coverage_search:
            haystack = coverage_view.astype(str).agg(" ".join, axis=1)
            coverage_view = coverage_view[haystack.str.contains(coverage_search, case=False, na=False)]
        coverage_cols = [
            col
            for col in [
                "priority",
                "experiment_id",
                "entity",
                "pilot_gate",
                "experiment_type",
                "coverage_status",
                "evidence_level",
                "sampled_smoked_ok",
                "runner_artifact",
                "claim_boundary",
                "next_action",
                "pass_gate",
                "fail_action",
            ]
            if col in coverage_view.columns
        ]
        st.dataframe(coverage_view[coverage_cols], width="stretch", hide_index=True)
        doc_path = Path("docs/DEEPUplift_PILOT_EXPERIMENT_COMMAND_COVERAGE.md")
        if doc_path.exists():
            st.download_button(
                "Download pilot experiment command coverage",
                data=doc_path.read_bytes(),
                file_name=doc_path.name,
                mime="text/markdown",
                key="download_pilot_experiment_command_coverage",
            )
    else:
        st.info("Run `scripts/generate_pilot_experiment_command_coverage.py` after command smoke.")

    st.subheader("Code / Model Provenance")
    st.dataframe(pd.DataFrame(model_code_provenance_rows()), width="stretch", hide_index=True)

    st.subheader("Frontier Model Source Gate")
    st.caption("前沿模型必须先有可信来源、adapter 计划、指标说明和回归验证，再进入默认训练菜单。")
    st.dataframe(pd.DataFrame(frontier_source_rows()), width="stretch", hide_index=True)

    st.subheader("Deep Loss / Network Source Gate")
    st.caption("Loss 和网络结构也要能追到代码、论文/开源来源、测试脚本和 UI 证据。")
    deep_source_df = pd.DataFrame(deep_source_rows())
    st.dataframe(deep_source_df, width="stretch", hide_index=True)
    st.download_button(
        "Download deep loss/network source gate",
        data=dataframe_csv_bytes(deep_source_df),
        file_name="deepuplift_deep_loss_network_source_gate.csv",
        mime="text/csv",
        key="download_deep_loss_network_source_gate",
    )
    local_report_df = pd.DataFrame(local_causal_report_rows())
    if not local_report_df.empty:
        with st.expander("Local causal-inference report signals", expanded=False):
            st.caption("Optional local research context loaded from `DEEPUPlIFT_RESEARCH_ARTICLES`; the file is not part of this repository.")
            st.dataframe(local_report_df, width="stretch", hide_index=True)

    st.subheader("Model Card Governance")
    st.caption("所有 guarded / optional / heavy backend 都必须说明缺什么、证据在哪、怎么从 catalog-only 晋级到可演示训练。")
    governance_rows = pd.DataFrame(model_card_governance_rows())
    if not governance_rows.empty:
        st.dataframe(governance_rows, width="stretch", hide_index=True)
        st.download_button(
            "Download model card governance CSV",
            data=dataframe_csv_bytes(governance_rows),
            file_name="deepuplift_model_card_governance.csv",
            mime="text/csv",
            key="download_model_card_governance_csv",
        )
    else:
        st.info("No guarded or optional model-card governance rows found.")

    st.subheader("Optional Backend Probe")
    optional_backend_rows = optional_backend_probe.get("rows") or []
    if optional_backend_rows:
        st.caption("Shows which heavy/guarded uplift backends are importable in the current environment without installing risky dependencies during the demo.")
        st.dataframe(pd.DataFrame(optional_backend_rows), width="stretch", hide_index=True)
    else:
        st.info("Run `scripts/probe_optional_uplift_backends.py` to refresh guarded/optional backend evidence.")

    st.subheader("UTBoost Guarded Adapter Smoke")
    if utboost_smoke:
        st.caption("This smoke passes when UTBoost is safely guarded without `utboost`; if the dependency is installed later, it runs a tiny training smoke.")
        st.dataframe(pd.DataFrame([utboost_smoke]), width="stretch", hide_index=True)
    else:
        st.info("Run `scripts/smoke_utboost_guarded_adapter.py` to refresh UTBoost guarded adapter evidence.")

    st.subheader("Deep Loss / Architecture Smoke")
    smoke_rows = []
    if deep_loss_architecture_smoke:
        smoke_rows.append(
            {
                "check": "deep_loss_architecture_catalog",
                "status": deep_loss_architecture_smoke.get("status"),
                "details": json.dumps(deep_loss_architecture_smoke.get("counts", {}), ensure_ascii=False),
            }
        )
    if cfrnet_balance_smoke:
        smoke_rows.append(
            {
                "check": "cfrnet_differentiable_balance",
                "status": cfrnet_balance_smoke.get("status"),
                "details": json.dumps(cfrnet_balance_smoke.get("results", {}), ensure_ascii=False),
            }
        )
    if deep_loss_functions_smoke:
        smoke_rows.append(
            {
                "check": "deep_loss_functions",
                "status": deep_loss_functions_smoke.get("status"),
                "details": json.dumps(deep_loss_functions_smoke.get("results", {}), ensure_ascii=False),
            }
        )
    if cfrnet_training_smoke:
        smoke_rows.append(
            {
                "check": "cfrnet_training_balance",
                "status": cfrnet_training_smoke.get("status"),
                "details": json.dumps(cfrnet_training_smoke.get("metrics", {}), ensure_ascii=False),
            }
        )
    if contrastive_training_smoke:
        smoke_rows.append(
            {
                "check": "contrastive_uplift_training",
                "status": contrastive_training_smoke.get("status"),
                "details": json.dumps(contrastive_training_smoke.get("metrics", {}), ensure_ascii=False),
            }
        )
    if deep_neural_ablation:
        smoke_rows.append(
            {
                "check": "deep_neural_ablation",
                "status": deep_neural_ablation.get("status"),
                "details": json.dumps(
                    {"best_model": deep_neural_ablation.get("best_model"), "rows": len(deep_neural_ablation.get("rows", []))},
                    ensure_ascii=False,
                ),
            }
        )
    if deep_model_training_evidence:
        smoke_rows.append(
            {
                "check": "deep_model_training_evidence",
                "status": deep_model_training_evidence.get("status"),
                "details": json.dumps(
                    {
                        "ok_models": deep_model_training_evidence.get("ok_models"),
                        "dataset_rows": (deep_model_training_evidence.get("dataset") or {}).get("rows"),
                    },
                    ensure_ascii=False,
                ),
            }
        )
    if deep_model_tuned_benchmark:
        smoke_rows.append(
            {
                "check": "deep_model_tuned_benchmark",
                "status": deep_model_tuned_benchmark.get("status"),
                "details": json.dumps(
                    {
                        "preset": deep_model_tuned_benchmark.get("preset"),
                        "runs": len(deep_model_tuned_benchmark.get("runs", [])),
                        "battle_cards": len(deep_model_tuned_benchmark.get("battle_cards", [])),
                    },
                    ensure_ascii=False,
                ),
            }
        )
    if paper_benchmark:
        smoke_rows.append(
            {
                "check": "paper_level_benchmark",
                "status": paper_benchmark.get("status"),
                "details": json.dumps(
                    {
                        "datasets": len(paper_benchmark.get("datasets", [])),
                        "runs": len(paper_benchmark.get("runs", [])),
                        "leaderboard": len(paper_benchmark.get("leaderboard", [])),
                    },
                    ensure_ascii=False,
                ),
            }
        )
    if smoke_rows:
        st.dataframe(pd.DataFrame(smoke_rows), width="stretch", hide_index=True)
    else:
        st.info("Run deep-loss smoke scripts to refresh loss catalog, gradients and CFRNet training evidence.")

    st.subheader("Uplift Failure Mode Diagnostic Cases")
    if failure_modes_smoke:
        st.caption("验证模型风险矩阵、业务 pitfall、mock case 和 Agent 回答是否完整。")
        st.dataframe(pd.DataFrame([failure_modes_smoke]), width="stretch", hide_index=True)
    diagnostic_summary = pd.DataFrame(diagnostic_case_summary_rows())
    if not diagnostic_summary.empty:
        st.caption("这些 mock CSV 用于演示 weak uplift、poor overlap、QINI 高但 ROI 低、延迟反馈、全链路广告和 LLM routing 成本陷阱。")
        display_cols = [
            col
            for col in [
                "case_id",
                "rows",
                "treatment_rate",
                "weak_overlap_rate",
                "top10_true_uplift",
                "top10_policy_value",
                "d1_rate_top10",
                "d30_rate_top10",
                "click_uplift_top10",
                "conversion_uplift_top10",
                "mock_csv",
            ]
            if col in diagnostic_summary.columns
        ]
        st.dataframe(diagnostic_summary[display_cols], width="stretch", hide_index=True)
    else:
        st.info("Run `scripts/generate_uplift_diagnostic_cases.py` and `scripts/smoke_uplift_failure_modes.py` to refresh failure-mode evidence.")

    if deep_neural_ablation.get("rows"):
        st.subheader("Deep Neural Ablation")
        st.caption("Same synthetic uplift design, same trainer/evaluator, comparing factual TarNet, CFR balance and contrastive representation.")
        st.dataframe(pd.DataFrame(deep_neural_ablation["rows"]), width="stretch", hide_index=True)

    if deep_model_training_evidence.get("leaderboard"):
        st.subheader("Deep Model Training Evidence")
        st.caption("CFRNet / DragonNet / EFIN / DESCN share one synthetic uplift design, trainer, evaluator, manifest and interview evidence contract.")
        deep_training_rows = deep_model_training_evidence.get("leaderboard") or []
        deep_training_df = pd.DataFrame(deep_training_rows)
        display_cols = [
            col
            for col in [
                "model",
                "status",
                "qini_score",
                "auuc_score",
                "oracle_top10_recall",
                "train_loss_last",
                "aux_loss_last",
                "run_id",
                "evidence_manifest",
                "interview_line",
            ]
            if col in deep_training_df.columns
        ]
        st.dataframe(deep_training_df[display_cols], width="stretch", hide_index=True)
        model_options = [str(row.get("model")) for row in deep_training_rows if row.get("history")]
        if model_options:
            chosen_training_model = st.selectbox(
                "Inspect deep model loss curve",
                model_options,
                key="evidence_deep_training_model",
            )
            chosen_training_row = next((row for row in deep_training_rows if row.get("model") == chosen_training_model), {})
            history_df = pd.DataFrame(chosen_training_row.get("history") or [])
            curve_cols = [
                col
                for col in ["train_loss", "valid_loss", "train_outcome_loss", "train_treatment_loss"]
                if col in history_df.columns
            ]
            if curve_cols:
                st.line_chart(history_df.set_index("epoch")[curve_cols], height=240, width="stretch")
            st.caption(chosen_training_row.get("interview_line", ""))

    if deep_model_objective_diagnostics.get("cards"):
        st.subheader("Deep Model Objective Diagnostics")
        st.caption("把深度模型的 objective 拆成 loss term、公式、可观测诊断、failure signal 和 safe/unsafe claim。")
        objective_cards = deep_model_objective_diagnostics.get("cards") or []
        objective_rows = []
        for card in objective_cards:
            diagnostics = card.get("diagnostics") or {}
            objective_rows.append(
                {
                    "model": card.get("model"),
                    "loss_terms": len(card.get("loss_terms") or []),
                    "training_status": diagnostics.get("training_status"),
                    "tuned_verdict": diagnostics.get("tuned_verdict"),
                    "best_pehe": diagnostics.get("best_pehe"),
                    "best_qini": diagnostics.get("best_qini"),
                    "safe_claim": card.get("safe_claim"),
                    "unsafe_claim": card.get("unsafe_claim"),
                    "interview_line": card.get("interview_line"),
                }
            )
        st.dataframe(pd.DataFrame(objective_rows), width="stretch", hide_index=True)
    else:
        st.info("Run `scripts/generate_deep_model_objective_diagnostics.py` to refresh objective/loss diagnostics.")

    if algorithm_claim_ledger.get("rows"):
        st.subheader("Algorithm Claim Ledger")
        st.caption("把模型 claim 绑定到公式、源码路径、benchmark verdict、license gate、safe/unsafe claim 和 UI 路径，防止过度包装。")
        claim_summary = algorithm_claim_ledger.get("summary") or {}
        render_stat_grid(
            [
                ("Models", claim_summary.get("models", "NA"), False),
                ("Claim Rows", claim_summary.get("claims", "NA"), False),
                ("Types", len(claim_summary.get("claim_type_counts") or {}), False),
                ("Evidence Levels", len(claim_summary.get("evidence_level_counts") or {}), False),
            ],
            columns=4,
        )
        claim_rows = pd.DataFrame(algorithm_claim_ledger.get("rows") or [])
        claim_filter_cols = st.columns([0.25, 0.25, 0.5])
        model_options = ["All"] + sorted(claim_rows["model"].dropna().astype(str).unique().tolist())
        claim_type_options = ["All"] + sorted(claim_rows["claim_type"].dropna().astype(str).unique().tolist())
        selected_claim_model = claim_filter_cols[0].selectbox("Claim model", model_options, key="evidence_claim_model")
        selected_claim_type = claim_filter_cols[1].selectbox("Claim type", claim_type_options, key="evidence_claim_type")
        claim_search = claim_filter_cols[2].text_input("Search claim/evidence", value="", key="evidence_claim_search")
        claim_view = claim_rows.copy()
        if selected_claim_model != "All":
            claim_view = claim_view[claim_view["model"].astype(str) == selected_claim_model]
        if selected_claim_type != "All":
            claim_view = claim_view[claim_view["claim_type"].astype(str) == selected_claim_type]
        if claim_search:
            haystack = claim_view.astype(str).agg(" ".join, axis=1)
            claim_view = claim_view[haystack.str.contains(claim_search, case=False, na=False)]
        display_cols = [
            col
            for col in [
                "claim_id",
                "model",
                "claim_type",
                "evidence_level",
                "benchmark_verdict",
                "promotion_decision",
                "formula_or_test",
                "safe_claim",
                "unsafe_claim",
                "ui_route",
                "next_action",
            ]
            if col in claim_view.columns
        ]
        st.dataframe(claim_view[display_cols], width="stretch", hide_index=True)
        doc_path = Path("docs/DEEPUplift_ALGORITHM_CLAIM_LEDGER.md")
        if doc_path.exists():
            st.download_button(
                "Download algorithm claim ledger",
                data=doc_path.read_bytes(),
                file_name=doc_path.name,
                mime="text/markdown",
                key="download_algorithm_claim_ledger",
            )
    else:
        st.info("Run `scripts/generate_algorithm_claim_ledger.py` to refresh model-claim evidence governance.")

    if paper_reproduction_gap_ledger.get("rows"):
        st.subheader("Paper Reproduction Gap Ledger")
        st.caption("把论文/开源项目、license 风险、本仓库复现程度、benchmark 证据和剩余 gap 放到同一张评审表里。")
        gap_summary = paper_reproduction_gap_ledger.get("summary") or {}
        render_stat_grid(
            [
                ("Models", gap_summary.get("models", "NA"), False),
                ("Source Rows", gap_summary.get("source_rows", "NA"), False),
                ("Proof Levels", len(gap_summary.get("proof_level_counts") or {}), False),
                ("Adoption Decisions", len(gap_summary.get("adoption_decision_counts") or {}), False),
            ],
            columns=4,
        )
        gap_df = pd.DataFrame(paper_reproduction_gap_ledger.get("rows") or [])
        gap_filter_cols = st.columns([0.25, 0.25, 0.5])
        gap_model_options = ["All"] + sorted(gap_df["model"].dropna().astype(str).unique().tolist())
        proof_options = ["All"] + sorted(gap_df["proof_level"].dropna().astype(str).unique().tolist())
        selected_gap_model = gap_filter_cols[0].selectbox("Reproduction model", gap_model_options, key="evidence_gap_model")
        selected_proof = gap_filter_cols[1].selectbox("Proof level", proof_options, key="evidence_gap_proof")
        gap_search = gap_filter_cols[2].text_input("Search paper/source/license", value="", key="evidence_gap_search")
        gap_view = gap_df.copy()
        if selected_gap_model != "All":
            gap_view = gap_view[gap_view["model"].astype(str) == selected_gap_model]
        if selected_proof != "All":
            gap_view = gap_view[gap_view["proof_level"].astype(str) == selected_proof]
        if gap_search:
            haystack = gap_view.astype(str).agg(" ".join, axis=1)
            gap_view = gap_view[haystack.str.contains(gap_search, case=False, na=False)]
        gap_cols = [
            col
            for col in [
                "model",
                "source_name",
                "source_type",
                "source_license",
                "adoption_decision",
                "proof_level",
                "benchmark_verdict",
                "safe_claim",
                "unsafe_claim",
                "validation_needed",
            ]
            if col in gap_view.columns
        ]
        st.dataframe(gap_view[gap_cols], width="stretch", hide_index=True)
        doc_path = Path("docs/DEEPUplift_PAPER_REPRODUCTION_GAP_LEDGER.md")
        if doc_path.exists():
            st.download_button(
                "Download paper reproduction gap ledger",
                data=doc_path.read_bytes(),
                file_name=doc_path.name,
                mime="text/markdown",
                key="download_paper_reproduction_gap_ledger",
            )
    else:
        st.info("Run `scripts/generate_paper_reproduction_gap_ledger.py` to refresh paper/repo reproduction gaps.")

    if model_evidence_promotion_matrix.get("rows"):
        st.subheader("Model Evidence Promotion Matrix")
        st.caption("把源码拆解、论文复现、benchmark verdict、objective diagnostics、license gate、safe/blocked claim 和 launch decision 聚合成一张模型评审矩阵。")
        matrix_summary = model_evidence_promotion_matrix.get("summary") or {}
        render_stat_grid(
            [
                ("Models", matrix_summary.get("models", "NA"), False),
                ("Stages", len(matrix_summary.get("stage_counts") or {}), False),
                ("Promotion Tiers", len(matrix_summary.get("promotion_tier_counts") or {}), False),
                ("License Gates", len(matrix_summary.get("license_gate_counts") or {}), False),
            ],
            columns=4,
        )
        matrix_df = pd.DataFrame(model_evidence_promotion_matrix.get("rows") or [])
        matrix_filter_cols = st.columns([0.25, 0.25, 0.5])
        matrix_stage_options = ["All"] + sorted(matrix_df["evidence_stage"].dropna().astype(str).unique().tolist())
        matrix_tier_options = ["All"] + sorted(matrix_df["promotion_tier"].dropna().astype(str).unique().tolist())
        selected_matrix_stage = matrix_filter_cols[0].selectbox("Evidence stage", matrix_stage_options, key="evidence_matrix_stage")
        selected_matrix_tier = matrix_filter_cols[1].selectbox("Promotion tier", matrix_tier_options, key="evidence_matrix_tier")
        matrix_search = matrix_filter_cols[2].text_input("Search model/claim/next step", value="", key="evidence_matrix_search")
        matrix_view = matrix_df.copy()
        if selected_matrix_stage != "All":
            matrix_view = matrix_view[matrix_view["evidence_stage"].astype(str) == selected_matrix_stage]
        if selected_matrix_tier != "All":
            matrix_view = matrix_view[matrix_view["promotion_tier"].astype(str) == selected_matrix_tier]
        if matrix_search:
            haystack = matrix_view.astype(str).agg(" ".join, axis=1)
            matrix_view = matrix_view[haystack.str.contains(matrix_search, case=False, na=False)]
        matrix_cols = [
            col
            for col in [
                "model",
                "evidence_stage",
                "promotion_tier",
                "evidence_score",
                "paper_verdict",
                "tuned_verdict",
                "license_gate",
                "launch_decision",
                "safe_claim",
                "blocked_claim",
                "next_evidence_step",
            ]
            if col in matrix_view.columns
        ]
        st.dataframe(matrix_view[matrix_cols], width="stretch", hide_index=True)
        doc_path = Path("docs/DEEPUplift_MODEL_EVIDENCE_PROMOTION_MATRIX.md")
        if doc_path.exists():
            st.download_button(
                "Download model evidence promotion matrix",
                data=doc_path.read_bytes(),
                file_name=doc_path.name,
                mime="text/markdown",
                key="download_model_evidence_promotion_matrix",
            )
    else:
        st.info("Run `scripts/generate_model_evidence_promotion_matrix.py` to refresh model evidence promotion tiers.")

    if model_upgrade_recipe_cards.get("rows"):
        st.subheader("Model Upgrade Recipe Cards")
        st.caption("把 promotion matrix 的下一步变成可执行实验卡：hypothesis、命令、指标、pass gate、fail action 和 expected artifacts。")
        recipe_summary = model_upgrade_recipe_cards.get("summary") or {}
        render_stat_grid(
            [
                ("Models", recipe_summary.get("models", "NA"), False),
                ("Recipes", recipe_summary.get("recipes", "NA"), False),
                ("Upgrade Goals", len(recipe_summary.get("upgrade_goal_counts") or {}), False),
                ("Owner Roles", len(recipe_summary.get("owner_role_counts") or {}), False),
            ],
            columns=4,
        )
        recipe_df = pd.DataFrame(model_upgrade_recipe_cards.get("rows") or [])
        recipe_filter_cols = st.columns([0.25, 0.25, 0.5])
        recipe_model_options = ["All"] + sorted(recipe_df["model"].dropna().astype(str).unique().tolist())
        recipe_owner_options = ["All"] + sorted(recipe_df["owner_role"].dropna().astype(str).unique().tolist())
        selected_recipe_model = recipe_filter_cols[0].selectbox("Recipe model", recipe_model_options, key="evidence_recipe_model")
        selected_recipe_owner = recipe_filter_cols[1].selectbox("Owner role", recipe_owner_options, key="evidence_recipe_owner")
        recipe_search = recipe_filter_cols[2].text_input("Search recipe/pass gate", value="", key="evidence_recipe_search")
        recipe_view = recipe_df.copy()
        if selected_recipe_model != "All":
            recipe_view = recipe_view[recipe_view["model"].astype(str) == selected_recipe_model]
        if selected_recipe_owner != "All":
            recipe_view = recipe_view[recipe_view["owner_role"].astype(str) == selected_recipe_owner]
        if recipe_search:
            haystack = recipe_view.astype(str).agg(" ".join, axis=1)
            recipe_view = recipe_view[haystack.str.contains(recipe_search, case=False, na=False)]
        recipe_cols = [
            col
            for col in [
                "recipe_id",
                "model",
                "upgrade_goal",
                "promotion_tier",
                "hypothesis",
                "command",
                "metrics_to_watch",
                "pass_gate",
                "fail_action",
                "owner_role",
                "ui_route",
            ]
            if col in recipe_view.columns
        ]
        st.dataframe(recipe_view[recipe_cols], width="stretch", hide_index=True)
        doc_path = Path("docs/DEEPUplift_MODEL_UPGRADE_RECIPE_CARDS.md")
        if doc_path.exists():
            st.download_button(
                "Download model upgrade recipe cards",
                data=doc_path.read_bytes(),
                file_name=doc_path.name,
                mime="text/markdown",
                key="download_model_upgrade_recipe_cards",
            )
    else:
        st.info("Run `scripts/generate_model_upgrade_recipe_cards.py` to refresh executable model-upgrade recipes.")

    if deep_model_telemetry_smoke.get("rows"):
        st.subheader("Deep Model Telemetry Smoke")
        st.caption("读取最新 deep model run artifacts，检查 loss history、prediction spread、Top-K known-CATE signal、metrics 和 evidence manifest 是否可观测。")
        telemetry_smoke_summary = deep_model_telemetry_smoke.get("summary") or {}
        render_stat_grid(
            [
                ("Models", telemetry_smoke_summary.get("models", "NA"), False),
                ("Rows", telemetry_smoke_summary.get("rows", "NA"), False),
                ("Basic Pass", telemetry_smoke_summary.get("basic_telemetry_pass", "NA"), False),
                ("Advanced Hook Gaps", telemetry_smoke_summary.get("advanced_hook_gaps", "NA"), False),
            ],
            columns=4,
        )
        telemetry_smoke_df = pd.DataFrame(deep_model_telemetry_smoke.get("rows") or [])
        telemetry_smoke_cols = [
            col
            for col in [
                "model",
                "status",
                "epochs",
                "train_loss_delta",
                "uplift_score_std",
                "top10_true_uplift_gain",
                "manifest_files",
                "advanced_hook_status",
                "advanced_hook_needed",
                "pass_gate",
                "interview_line",
            ]
            if col in telemetry_smoke_df.columns
        ]
        st.dataframe(telemetry_smoke_df[telemetry_smoke_cols], width="stretch", hide_index=True)
        doc_path = Path("docs/DEEPUplift_DEEP_MODEL_TELEMETRY_SMOKE.md")
        if doc_path.exists():
            st.download_button(
                "Download deep model telemetry smoke",
                data=doc_path.read_bytes(),
                file_name=doc_path.name,
                mime="text/markdown",
                key="download_deep_model_telemetry_smoke",
            )
    else:
        st.info("Run `scripts/smoke_deep_model_telemetry.py` to refresh deep model telemetry smoke.")

    if deep_model_forward_hook_contracts.get("rows"):
        st.subheader("Deep Model Forward Hook Contracts")
        st.caption("把高级 telemetry 改造前置成 contract：tensor keys、artifact files、smoke test、pass gate、fallback 和 backward compatibility。")
        hook_summary = deep_model_forward_hook_contracts.get("summary") or {}
        render_stat_grid(
            [
                ("Models", hook_summary.get("models", "NA"), False),
                ("Contracts", hook_summary.get("contracts", "NA"), False),
                ("P0 Contracts", hook_summary.get("p0_contracts", "NA"), False),
                ("Guarded", hook_summary.get("guarded_contracts", "NA"), False),
            ],
            columns=4,
        )
        hook_df = pd.DataFrame(deep_model_forward_hook_contracts.get("rows") or [])
        hook_filter_cols = st.columns([0.25, 0.25, 0.5])
        hook_model_options = ["All"] + sorted(hook_df["model"].dropna().astype(str).unique().tolist())
        hook_priority_options = ["All"] + sorted(hook_df["priority"].dropna().astype(str).unique().tolist())
        selected_hook_model = hook_filter_cols[0].selectbox("Hook model", hook_model_options, key="evidence_hook_model")
        selected_hook_priority = hook_filter_cols[1].selectbox("Hook priority", hook_priority_options, key="evidence_hook_priority")
        hook_search = hook_filter_cols[2].text_input("Search hook/tensor/artifact/pass gate", value="", key="evidence_hook_search")
        hook_view = hook_df.copy()
        if selected_hook_model != "All":
            hook_view = hook_view[hook_view["model"].astype(str) == selected_hook_model]
        if selected_hook_priority != "All":
            hook_view = hook_view[hook_view["priority"].astype(str) == selected_hook_priority]
        if hook_search:
            haystack = hook_view.astype(str).agg(" ".join, axis=1)
            hook_view = hook_view[haystack.str.contains(hook_search, case=False, na=False)]
        hook_cols = [
            col
            for col in [
                "model",
                "hook_name",
                "contract_type",
                "priority",
                "expected_tensor_keys",
                "artifact_files",
                "source_code_entry",
                "pass_gate",
                "fallback_behavior",
                "backward_compat",
                "interview_line",
            ]
            if col in hook_view.columns
        ]
        hook_table = hook_view[hook_cols].copy()
        for col in ["expected_tensor_keys", "artifact_files"]:
            if col in hook_table.columns:
                hook_table[col] = hook_table[col].apply(
                    lambda value: ", ".join(map(str, value)) if isinstance(value, list) else str(value)
                )
        st.dataframe(hook_table, width="stretch", hide_index=True)
        doc_path = Path("docs/DEEPUplift_DEEP_MODEL_FORWARD_HOOK_CONTRACTS.md")
        if doc_path.exists():
            st.download_button(
                "Download deep model forward hook contracts",
                data=doc_path.read_bytes(),
                file_name=doc_path.name,
                mime="text/markdown",
                key="download_deep_model_forward_hook_contracts",
            )
    else:
        st.info("Run `scripts/generate_deep_model_forward_hook_contracts.py` to refresh guarded forward-hook contracts.")

    if deep_model_forward_hook_smoke.get("rows"):
        st.subheader("Deep Model Forward Hook Smoke")
        st.caption("真正执行 guarded forward instrumentation，导出 phi_x、e_hat/tarreg、attention/path、head/constraint sidecar artifacts。")
        hook_smoke_summary = deep_model_forward_hook_smoke.get("summary") or {}
        render_stat_grid(
            [
                ("Models", hook_smoke_summary.get("models", "NA"), False),
                ("Contracts", hook_smoke_summary.get("contracts", "NA"), False),
                ("OK Contracts", hook_smoke_summary.get("ok_contracts", "NA"), True),
                ("Artifact Files", hook_smoke_summary.get("artifact_files", "NA"), False),
            ],
            columns=4,
        )
        st.caption(f"Artifact dir: `{deep_model_forward_hook_smoke.get('artifact_dir', 'NA')}`")
        hook_smoke_df = pd.DataFrame(deep_model_forward_hook_smoke.get("rows") or [])
        hook_smoke_cols = [
            col
            for col in [
                "model",
                "hook_name",
                "status",
                "tensor_keys_exported",
                "artifact_files",
                "metric_summary",
                "pass_gate",
                "remaining_gap",
                "interview_line",
            ]
            if col in hook_smoke_df.columns
        ]
        hook_smoke_table = hook_smoke_df[hook_smoke_cols].copy()
        for col in ["tensor_keys_exported", "artifact_files", "metric_summary"]:
            if col in hook_smoke_table.columns:
                hook_smoke_table[col] = hook_smoke_table[col].apply(
                    lambda value: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
                )
        st.dataframe(hook_smoke_table, width="stretch", hide_index=True)
        doc_path = Path("docs/DEEPUplift_DEEP_MODEL_FORWARD_HOOK_SMOKE.md")
        if doc_path.exists():
            st.download_button(
                "Download deep model forward hook smoke",
                data=doc_path.read_bytes(),
                file_name=doc_path.name,
                mime="text/markdown",
                key="download_deep_model_forward_hook_smoke",
            )
    else:
        st.info("Run `scripts/smoke_deep_model_forward_hooks.py` to refresh guarded forward-hook smoke artifacts.")

    if deep_model_hook_training_bridge.get("rows"):
        st.subheader("Deep Model Hook Training Bridge")
        st.caption("把 forward hook sidecar 和训练 run/manifest 串起来，定义下一步 trainer-level per-epoch telemetry、pass gate 和 fail action。")
        bridge_summary = deep_model_hook_training_bridge.get("summary") or {}
        render_stat_grid(
            [
                ("Models", bridge_summary.get("models", "NA"), False),
                ("Bridges", bridge_summary.get("bridges", "NA"), False),
                ("Bridge Ready", bridge_summary.get("bridge_ready", "NA"), True),
                ("Epoch Columns", bridge_summary.get("proposed_epoch_columns", "NA"), False),
            ],
            columns=4,
        )
        bridge_df = pd.DataFrame(deep_model_hook_training_bridge.get("rows") or [])
        bridge_cols = [
            col
            for col in [
                "model",
                "contract_id",
                "priority",
                "bridge_status",
                "proposed_epoch_columns",
                "proposed_epoch_artifacts",
                "pass_gate",
                "fail_action",
                "interview_line",
            ]
            if col in bridge_df.columns
        ]
        bridge_table = bridge_df[bridge_cols].copy()
        for col in ["proposed_epoch_columns", "proposed_epoch_artifacts"]:
            if col in bridge_table.columns:
                bridge_table[col] = bridge_table[col].apply(
                    lambda value: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
                )
        st.dataframe(bridge_table, width="stretch", hide_index=True)
        doc_path = Path("docs/DEEPUplift_DEEP_MODEL_HOOK_TRAINING_BRIDGE.md")
        if doc_path.exists():
            st.download_button(
                "Download deep model hook training bridge",
                data=doc_path.read_bytes(),
                file_name=doc_path.name,
                mime="text/markdown",
                key="download_deep_model_hook_training_bridge",
            )
    else:
        st.info("Run `scripts/generate_deep_model_hook_training_bridge.py` to refresh trainer-level hook bridge artifacts.")

    if deep_model_trainer_collector_smoke.get("model_rows"):
        st.subheader("Deep Model Trainer Collector Smoke")
        st.caption("真实跑 tiny training loop，按 epoch 导出 CFRNet/DragonNet/EFIN/DESCN 的高级 telemetry sidecar。")
        collector_summary = deep_model_trainer_collector_smoke.get("summary") or {}
        render_stat_grid(
            [
                ("Models", collector_summary.get("models", "NA"), False),
                ("OK Models", collector_summary.get("ok_models", "NA"), True),
                ("Epoch Rows", collector_summary.get("epoch_rows", "NA"), False),
                ("Contracts", collector_summary.get("contracts_covered", "NA"), False),
            ],
            columns=4,
        )
        st.caption(f"Artifact dir: `{deep_model_trainer_collector_smoke.get('artifact_dir', 'NA')}`")
        collector_df = pd.DataFrame(deep_model_trainer_collector_smoke.get("model_rows") or [])
        collector_cols = [
            col
            for col in [
                "model",
                "status",
                "epochs",
                "contracts_covered",
                "collected_epoch_columns",
                "metric_last",
                "pass_gate",
                "interview_line",
            ]
            if col in collector_df.columns
        ]
        collector_table = collector_df[collector_cols].copy()
        for col in ["contracts_covered", "collected_epoch_columns", "metric_last"]:
            if col in collector_table.columns:
                collector_table[col] = collector_table[col].apply(
                    lambda value: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
                )
        st.dataframe(collector_table, width="stretch", hide_index=True)
        doc_path = Path("docs/DEEPUplift_DEEP_MODEL_TRAINER_COLLECTOR_SMOKE.md")
        if doc_path.exists():
            st.download_button(
                "Download deep model trainer collector smoke",
                data=doc_path.read_bytes(),
                file_name=doc_path.name,
                mime="text/markdown",
                key="download_deep_model_trainer_collector_smoke",
            )
    else:
        st.info("Run `scripts/smoke_deep_model_trainer_collectors.py` to refresh per-epoch trainer collector smoke artifacts.")

    if deep_model_telemetry_benchmark_linkage.get("rows"):
        st.subheader("Deep Model Telemetry-Benchmark Linkage")
        st.caption("把 per-epoch 内部 telemetry 和 paper/tuned benchmark 胜负、baseline verdict、failure hypothesis、next ablation 串成可评审证据。")
        linkage_summary = deep_model_telemetry_benchmark_linkage.get("summary") or {}
        render_stat_grid(
            [
                ("Models", linkage_summary.get("models", "NA"), False),
                ("Linkage Rows", linkage_summary.get("linkage_rows", "NA"), False),
                ("Link Ready", linkage_summary.get("link_ready", "NA"), True),
                ("Families", len(linkage_summary.get("telemetry_families") or {}), False),
            ],
            columns=4,
        )
        linkage_df = pd.DataFrame(deep_model_telemetry_benchmark_linkage.get("rows") or [])
        linkage_cols = [
            col
            for col in [
                "model",
                "telemetry_family",
                "linkage_status",
                "paper_verdict",
                "tuned_verdict",
                "tuned_baseline_verdict",
                "telemetry_last",
                "failure_hypothesis",
                "next_ablation",
                "interview_line",
            ]
            if col in linkage_df.columns
        ]
        linkage_table = linkage_df[linkage_cols].copy()
        for col in ["telemetry_last"]:
            if col in linkage_table.columns:
                linkage_table[col] = linkage_table[col].apply(
                    lambda value: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
                )
        st.dataframe(linkage_table, width="stretch", hide_index=True)
        doc_path = Path("docs/DEEPUplift_DEEP_MODEL_TELEMETRY_BENCHMARK_LINKAGE.md")
        if doc_path.exists():
            st.download_button(
                "Download deep model telemetry-benchmark linkage",
                data=doc_path.read_bytes(),
                file_name=doc_path.name,
                mime="text/markdown",
                key="download_deep_model_telemetry_benchmark_linkage",
            )
    else:
        st.info("Run `scripts/generate_deep_model_telemetry_benchmark_linkage.py` to refresh telemetry-to-benchmark linkage artifacts.")

    if deep_model_telemetry_ablation_gate.get("rows"):
        st.subheader("Deep Model Telemetry Ablation Gate")
        st.caption("把 telemetry-benchmark linkage 的 next ablation 变成可执行门禁：需要哪些 variant、看哪些指标、什么条件能升级 claim、失败后怎么处理。")
        ablation_summary = deep_model_telemetry_ablation_gate.get("summary") or {}
        render_stat_grid(
            [
                ("Models", ablation_summary.get("models", "NA"), False),
                ("Gates", ablation_summary.get("gates", "NA"), False),
                ("Missing Variants", ablation_summary.get("missing_variant_rows", "NA"), True),
                ("Decisions", len(ablation_summary.get("decision_counts") or {}), False),
            ],
            columns=4,
        )
        ablation_df = pd.DataFrame(deep_model_telemetry_ablation_gate.get("rows") or [])
        ablation_cols = [
            col
            for col in [
                "model",
                "ablation_family",
                "gate_decision",
                "variant_coverage_status",
                "missing_variants",
                "pass_gate",
                "fail_action",
                "interview_line",
            ]
            if col in ablation_df.columns
        ]
        ablation_table = ablation_df[ablation_cols].copy()
        for col in ["missing_variants"]:
            if col in ablation_table.columns:
                ablation_table[col] = ablation_table[col].apply(
                    lambda value: ", ".join(str(item) for item in value)
                    if isinstance(value, list)
                    else str(value)
                )
        st.dataframe(ablation_table, width="stretch", hide_index=True)
        doc_path = Path("docs/DEEPUplift_DEEP_MODEL_TELEMETRY_ABLATION_GATE.md")
        if doc_path.exists():
            st.download_button(
                "Download deep model telemetry ablation gate",
                data=doc_path.read_bytes(),
                file_name=doc_path.name,
                mime="text/markdown",
                key="download_deep_model_telemetry_ablation_gate",
            )
    else:
        st.info("Run `scripts/generate_deep_model_telemetry_ablation_gate.py` to refresh telemetry ablation gates.")

    if deep_model_ablation_interpretation.get("rows"):
        st.subheader("Deep Model Ablation Interpretation")
        st.caption("把已覆盖的消融 variant 进一步解释成 default-vs-variant 指标移动、safe claim、blocked claim 和面试话术。")
        ablation_interp_summary = deep_model_ablation_interpretation.get("summary") or {}
        render_stat_grid(
            [
                ("Rows", ablation_interp_summary.get("rows", "NA"), False),
                ("Variants", ablation_interp_summary.get("variants", "NA"), False),
                ("Support Rows", ablation_interp_summary.get("support_rows", "NA"), True),
                ("Tradeoffs", ablation_interp_summary.get("tradeoff_rows", "NA"), True),
                ("Blocked", ablation_interp_summary.get("blocked_rows", "NA"), True),
            ],
            columns=5,
        )
        tracks = deep_model_ablation_interpretation.get("interview_tracks") or {}
        with st.expander("Ablation Interpretation / 面试讲法", expanded=False):
            st.dataframe(
                pd.DataFrame(
                    [
                        {"duration": "30s", "talk_track": tracks.get("30s", "")},
                        {"duration": "5min", "talk_track": tracks.get("5min", "")},
                        {"duration": "15min", "talk_track": tracks.get("15min", "")},
                        {"duration": "30min", "talk_track": tracks.get("30min", "")},
                    ]
                ),
                width="stretch",
                hide_index=True,
            )
            scorecard_df = pd.DataFrame(deep_model_ablation_interpretation.get("variant_scorecards") or [])
            if not scorecard_df.empty:
                scorecard_cols = [
                    col
                    for col in [
                        "model",
                        "variant_id",
                        "mechanism",
                        "datasets",
                        "pehe_improved_datasets",
                        "qini_improved_datasets",
                        "policy_improved_datasets",
                        "best_pehe_delta",
                        "best_qini_delta",
                        "interview_line",
                    ]
                    if col in scorecard_df.columns
                ]
                st.dataframe(scorecard_df[scorecard_cols], width="stretch", hide_index=True)
        ablation_interp_df = pd.DataFrame(deep_model_ablation_interpretation.get("rows") or [])
        ablation_interp_cols = [
            col
            for col in [
                "dataset_id",
                "model",
                "variant_id",
                "default_variant",
                "mechanism",
                "verdict",
                "delta_pehe_mean_vs_default",
                "delta_qini_mean_vs_default",
                "delta_policy_top10_oracle_value_mean_vs_default",
                "baseline_verdict",
                "safe_claim",
                "blocked_claim",
            ]
            if col in ablation_interp_df.columns
        ]
        st.dataframe(ablation_interp_df[ablation_interp_cols], width="stretch", hide_index=True)
        doc_path = Path("docs/DEEPUplift_DEEP_MODEL_ABLATION_INTERPRETATION.md")
        if doc_path.exists():
            st.download_button(
                "Download deep model ablation interpretation",
                data=doc_path.read_bytes(),
                file_name=doc_path.name,
                mime="text/markdown",
                key="download_deep_model_ablation_interpretation",
            )
    else:
        st.info("Run `scripts/generate_deep_model_ablation_interpretation.py` to refresh deep model ablation interpretation.")

    if deep_model_ablation_promotion_matrix.get("rows"):
        st.subheader("Deep Model Ablation Promotion Matrix")
        st.caption("把消融解读进一步升级成 claim tier、promotion decision、default-or-variant 推荐和 blocked claim。")
        ablation_promo_summary = deep_model_ablation_promotion_matrix.get("summary") or {}
        render_stat_grid(
            [
                ("Variants", ablation_promo_summary.get("variants", "NA"), False),
                ("Ready Rows", ablation_promo_summary.get("promotion_ready_rows", "NA"), True),
                ("Tradeoff Rows", ablation_promo_summary.get("tradeoff_rows", "NA"), True),
                ("Blocked/Missing", ablation_promo_summary.get("blocked_or_missing_rows", "NA"), True),
            ],
            columns=4,
        )
        promo_tracks = deep_model_ablation_promotion_matrix.get("interview_tracks") or {}
        with st.expander("Ablation Promotion / 面试讲法", expanded=False):
            st.dataframe(
                pd.DataFrame(
                    [
                        {"duration": "30s", "talk_track": promo_tracks.get("30s", "")},
                        {"duration": "5min", "talk_track": promo_tracks.get("5min", "")},
                        {"duration": "15min", "talk_track": promo_tracks.get("15min", "")},
                        {"duration": "30min", "talk_track": promo_tracks.get("30min", "")},
                    ]
                ),
                width="stretch",
                hide_index=True,
            )
        promo_df = pd.DataFrame(deep_model_ablation_promotion_matrix.get("rows") or [])
        promo_cols = [
            col
            for col in [
                "model",
                "variant_id",
                "claim_tier",
                "promotion_decision",
                "promotion_score",
                "support_rows",
                "tradeoff_rows",
                "baseline_still_stronger_rows",
                "recommended_claim_boundary",
                "safe_claim",
                "blocked_claim",
            ]
            if col in promo_df.columns
        ]
        st.dataframe(promo_df[promo_cols], width="stretch", hide_index=True)
        doc_path = Path("docs/DEEPUplift_DEEP_MODEL_ABLATION_PROMOTION_MATRIX.md")
        if doc_path.exists():
            st.download_button(
                "Download deep model ablation promotion matrix",
                data=doc_path.read_bytes(),
                file_name=doc_path.name,
                mime="text/markdown",
                key="download_deep_model_ablation_promotion_matrix",
            )
    else:
        st.info("Run `scripts/generate_deep_model_ablation_promotion_matrix.py` to refresh deep model ablation promotion matrix.")

    if deep_model_ablation_next_experiment_plan.get("rows"):
        st.subheader("Deep Model Ablation Next Experiment Plan")
        st.caption("把每个消融 variant 的 claim tier 翻译成下一步实验命令、指标、pass gate、fail action 和 owner。")
        plan_summary = deep_model_ablation_next_experiment_plan.get("summary") or {}
        render_stat_grid(
            [
                ("Variants", plan_summary.get("variants", "NA"), False),
                ("P0 Plans", plan_summary.get("p0_rows", "NA"), True),
                ("P1 Plans", plan_summary.get("p1_rows", "NA"), True),
                ("Experiment Types", len(plan_summary.get("experiment_type_counts") or {}), False),
            ],
            columns=4,
        )
        plan_tracks = deep_model_ablation_next_experiment_plan.get("interview_tracks") or {}
        with st.expander("Ablation Next Experiments / 面试追问", expanded=False):
            st.dataframe(
                pd.DataFrame(
                    [
                        {"duration": "30s", "talk_track": plan_tracks.get("30s", "")},
                        {"duration": "5min", "talk_track": plan_tracks.get("5min", "")},
                        {"duration": "15min", "talk_track": plan_tracks.get("15min", "")},
                        {"duration": "30min", "talk_track": plan_tracks.get("30min", "")},
                    ]
                ),
                width="stretch",
                hide_index=True,
            )
        plan_df = pd.DataFrame(deep_model_ablation_next_experiment_plan.get("rows") or [])
        plan_cols = [
            col
            for col in [
                "priority",
                "model",
                "variant_id",
                "experiment_type",
                "claim_tier",
                "promotion_decision",
                "recommended_command",
                "pass_gate",
                "fail_action",
                "owner_role",
            ]
            if col in plan_df.columns
        ]
        st.dataframe(plan_df[plan_cols], width="stretch", hide_index=True)
        doc_path = Path("docs/DEEPUplift_DEEP_MODEL_ABLATION_NEXT_EXPERIMENT_PLAN.md")
        if doc_path.exists():
            st.download_button(
                "Download deep model ablation next-experiment plan",
                data=doc_path.read_bytes(),
                file_name=doc_path.name,
                mime="text/markdown",
                key="download_deep_model_ablation_next_experiment_plan",
            )
    else:
        st.info("Run `scripts/generate_deep_model_ablation_next_experiment_plan.py` to refresh deep model ablation next-experiment plans.")

    if deep_model_ablation_command_contract.get("rows"):
        st.subheader("Deep Model Ablation Command Contract")
        st.caption("审计 next-experiment 命令是否匹配当前 benchmark runner CLI：规划命令可以是路线图，可执行 fallback 必须被当前 runner 支持。")
        contract_summary = deep_model_ablation_command_contract.get("summary") or {}
        render_stat_grid(
            [
                ("Variants", contract_summary.get("variants", "NA"), False),
                ("Fallback Ready", contract_summary.get("fallback_ready_rows", "NA"), True),
                ("Planner Adapter Rows", contract_summary.get("planner_adapter_rows", "NA"), False),
                ("Status Types", len(contract_summary.get("status_counts") or {}), False),
            ],
            columns=4,
        )
        contract_tracks = deep_model_ablation_command_contract.get("interview_tracks") or {}
        with st.expander("Command Contract / 命令可执行性证据", expanded=False):
            st.dataframe(
                pd.DataFrame(
                    [
                        {"duration": "30s", "talk_track": contract_tracks.get("30s", "")},
                        {"duration": "5min", "talk_track": contract_tracks.get("5min", "")},
                        {"duration": "15min", "talk_track": contract_tracks.get("15min", "")},
                        {"duration": "30min", "talk_track": contract_tracks.get("30min", "")},
                    ]
                ),
                width="stretch",
                hide_index=True,
            )
        contract_df = pd.DataFrame(deep_model_ablation_command_contract.get("rows") or [])
        contract_cols = [
            col
            for col in [
                "priority",
                "model",
                "variant_id",
                "contract_status",
                "fallback_ready",
                "recommended_unsupported_flags",
                "runnable_fallback_command",
                "contract_interview_line",
            ]
            if col in contract_df.columns
        ]
        st.dataframe(contract_df[contract_cols], width="stretch", hide_index=True)
        doc_path = Path("docs/DEEPUplift_DEEP_MODEL_ABLATION_COMMAND_CONTRACT.md")
        if doc_path.exists():
            st.download_button(
                "Download deep model ablation command contract",
                data=doc_path.read_bytes(),
                file_name=doc_path.name,
                mime="text/markdown",
                key="download_deep_model_ablation_command_contract",
            )
    else:
        st.info("Run `scripts/generate_deep_model_ablation_command_contract.py` to refresh deep model ablation command contracts.")

    if deep_model_ablation_command_contract_smoke.get("rows"):
        st.subheader("Deep Model Ablation Command Contract Smoke")
        st.caption("抽样执行 fallback command，并使用隔离 output prefix / no-update-latest，证明命令能跑且不污染主 benchmark latest。")
        smoke_summary = deep_model_ablation_command_contract_smoke.get("summary") or {}
        render_stat_grid(
            [
                ("Rows", smoke_summary.get("rows", "NA"), False),
                ("OK Rows", smoke_summary.get("ok_rows", "NA"), True),
                ("Failed Rows", smoke_summary.get("failed_rows", "NA"), True),
                ("Source Contract Rows", smoke_summary.get("source_contract_rows", "NA"), False),
            ],
            columns=4,
        )
        smoke_df = pd.DataFrame(deep_model_ablation_command_contract_smoke.get("rows") or [])
        smoke_cols = [
            col
            for col in [
                "variant_id",
                "model",
                "status",
                "runs",
                "ok_runs",
                "runner_manifest",
                "runner_json",
                "command",
            ]
            if col in smoke_df.columns
        ]
        st.dataframe(smoke_df[smoke_cols], width="stretch", hide_index=True)
        doc_path = Path("docs/DEEPUplift_DEEP_MODEL_ABLATION_COMMAND_CONTRACT_SMOKE.md")
        if doc_path.exists():
            st.download_button(
                "Download deep model ablation command contract smoke",
                data=doc_path.read_bytes(),
                file_name=doc_path.name,
                mime="text/markdown",
                key="download_deep_model_ablation_command_contract_smoke",
            )
    else:
        st.info("Run `scripts/smoke_deep_model_ablation_command_contract.py` to execute a small fallback-command smoke.")

    if deep_model_ablation_recommended_command_smoke.get("rows"):
        st.subheader("Deep Model Ablation Recommended Command Smoke")
        st.caption("抽样执行 runner-native recommended command，并用安全覆盖参数隔离输出；证明 --models / --include-variants / --emit-battle-cards 路径真实可跑。")
        recommended_smoke_summary = deep_model_ablation_recommended_command_smoke.get("summary") or {}
        render_stat_grid(
            [
                ("Rows", recommended_smoke_summary.get("rows", "NA"), False),
                ("OK Rows", recommended_smoke_summary.get("ok_rows", "NA"), True),
                ("Models Alias Rows", recommended_smoke_summary.get("models_alias_rows", "NA"), True),
                ("Emit Battle Rows", recommended_smoke_summary.get("emit_battle_cards_alias_rows", "NA"), True),
            ],
            columns=4,
        )
        recommended_smoke_df = pd.DataFrame(deep_model_ablation_recommended_command_smoke.get("rows") or [])
        recommended_smoke_cols = [
            col
            for col in [
                "priority",
                "variant_id",
                "model",
                "status",
                "uses_models_alias",
                "uses_include_variants_alias",
                "uses_emit_battle_cards_alias",
                "runs",
                "ok_runs",
                "runner_manifest",
                "safe_smoke_command",
            ]
            if col in recommended_smoke_df.columns
        ]
        st.dataframe(recommended_smoke_df[recommended_smoke_cols], width="stretch", hide_index=True)
        doc_path = Path("docs/DEEPUplift_DEEP_MODEL_ABLATION_RECOMMENDED_COMMAND_SMOKE.md")
        if doc_path.exists():
            st.download_button(
                "Download deep model ablation recommended command smoke",
                data=doc_path.read_bytes(),
                file_name=doc_path.name,
                mime="text/markdown",
                key="download_deep_model_ablation_recommended_command_smoke",
            )
    else:
        st.info("Run `scripts/smoke_deep_model_ablation_recommended_commands.py` to execute runner-native recommended command smoke rows.")

    if deep_model_ablation_command_smoke_coverage.get("rows"):
        st.subheader("Deep Model Ablation Command Smoke Coverage")
        st.caption("把 command contract、fallback smoke、runner-native recommended smoke 合并成覆盖率视图：哪些变体已经真实执行，哪些还只是 contract-ready backlog。")
        coverage_summary = deep_model_ablation_command_smoke_coverage.get("summary") or {}
        render_stat_grid(
            [
                ("Contract Rows", coverage_summary.get("contract_rows", "NA"), False),
                ("Any Smoked", coverage_summary.get("any_smoked_rows", "NA"), True),
                ("Recommended", coverage_summary.get("recommended_smoked_rows", "NA"), True),
                ("Fallback", coverage_summary.get("fallback_smoked_rows", "NA"), True),
                ("Unsmoked", coverage_summary.get("unsmoked_rows", "NA"), False),
                ("Gate", coverage_summary.get("coverage_gate", "NA"), True),
            ],
            columns=6,
        )
        coverage_df = pd.DataFrame(deep_model_ablation_command_smoke_coverage.get("rows") or [])
        coverage_cols = [
            col
            for col in [
                "priority",
                "model",
                "variant_id",
                "contract_status",
                "fallback_smoked",
                "recommended_smoked",
                "coverage_status",
                "recommended_runner_aliases",
                "next_action",
            ]
            if col in coverage_df.columns
        ]
        st.dataframe(coverage_df[coverage_cols], width="stretch", hide_index=True)
        doc_path = Path("docs/DEEPUplift_DEEP_MODEL_ABLATION_COMMAND_SMOKE_COVERAGE.md")
        if doc_path.exists():
            st.download_button(
                "Download deep model ablation command smoke coverage",
                data=doc_path.read_bytes(),
                file_name=doc_path.name,
                mime="text/markdown",
                key="download_deep_model_ablation_command_smoke_coverage",
            )
    else:
        st.info("Run `scripts/generate_deep_model_ablation_command_smoke_coverage.py` to refresh command smoke coverage.")

    if deep_model_ablation_command_smoke_coverage_diff.get("rows"):
        st.subheader("Deep Model Ablation Command Smoke Coverage Diff")
        st.caption("把 fallback-only smoke 到 runner-native full smoke 的变化讲清楚：哪些变体新增了真实执行证据，以及这个 gate 为什么从 review_required 升到 ok。")
        diff_summary = deep_model_ablation_command_smoke_coverage_diff.get("summary") or {}
        render_stat_grid(
            [
                ("Baseline", f"{diff_summary.get('baseline_fallback_smoked_rows', 'NA')}/{diff_summary.get('contract_rows', 'NA')}", False),
                ("Current", f"{diff_summary.get('current_any_smoked_rows', 'NA')}/{diff_summary.get('contract_rows', 'NA')}", True),
                ("Runner Native", f"{diff_summary.get('runner_native_recommended_smoked_rows', 'NA')}/{diff_summary.get('contract_rows', 'NA')}", True),
                ("Gain Rows", diff_summary.get("smoke_gain_rows", "NA"), True),
                ("Gate Delta", diff_summary.get("coverage_gate_delta", "NA"), True),
            ],
            columns=5,
        )
        diff_df = pd.DataFrame(deep_model_ablation_command_smoke_coverage_diff.get("rows") or [])
        diff_cols = [
            col
            for col in [
                "priority",
                "model",
                "variant_id",
                "baseline_fallback_smoked",
                "current_any_smoked",
                "runner_native_recommended_smoked",
                "transition",
                "next_action",
            ]
            if col in diff_df.columns
        ]
        st.dataframe(diff_df[diff_cols], width="stretch", hide_index=True)
        doc_path = Path("docs/DEEPUplift_DEEP_MODEL_ABLATION_COMMAND_SMOKE_COVERAGE_DIFF.md")
        if doc_path.exists():
            st.download_button(
                "Download deep model ablation command smoke coverage diff",
                data=doc_path.read_bytes(),
                file_name=doc_path.name,
                mime="text/markdown",
                key="download_deep_model_ablation_command_smoke_coverage_diff",
            )
    else:
        st.info("Run `scripts/generate_deep_model_ablation_command_smoke_coverage_diff.py` to refresh command smoke coverage diff.")

    if deep_model_telemetry_gap_matrix.get("rows"):
        st.subheader("Deep Model Telemetry Gap Matrix")
        st.caption("把 CFRNet / DragonNet / EFIN / DESCN 的 loss term 映射到缺失 telemetry、实现 hook、pass gate 和 failure action。")
        telemetry_summary = deep_model_telemetry_gap_matrix.get("summary") or {}
        render_stat_grid(
            [
                ("Models", telemetry_summary.get("models", "NA"), False),
                ("Telemetry Rows", telemetry_summary.get("telemetry_rows", "NA"), False),
                ("P0 Gaps", telemetry_summary.get("p0_rows", "NA"), False),
                ("Families", len(telemetry_summary.get("telemetry_family_counts") or {}), False),
            ],
            columns=4,
        )
        telemetry_df = pd.DataFrame(deep_model_telemetry_gap_matrix.get("rows") or [])
        telemetry_filter_cols = st.columns([0.25, 0.25, 0.5])
        telemetry_model_options = ["All"] + sorted(telemetry_df["model"].dropna().astype(str).unique().tolist())
        telemetry_priority_options = ["All"] + sorted(telemetry_df["priority"].dropna().astype(str).unique().tolist())
        selected_telemetry_model = telemetry_filter_cols[0].selectbox("Telemetry model", telemetry_model_options, key="evidence_telemetry_model")
        selected_telemetry_priority = telemetry_filter_cols[1].selectbox("Telemetry priority", telemetry_priority_options, key="evidence_telemetry_priority")
        telemetry_search = telemetry_filter_cols[2].text_input("Search telemetry/hook/pass gate", value="", key="evidence_telemetry_search")
        telemetry_view = telemetry_df.copy()
        if selected_telemetry_model != "All":
            telemetry_view = telemetry_view[telemetry_view["model"].astype(str) == selected_telemetry_model]
        if selected_telemetry_priority != "All":
            telemetry_view = telemetry_view[telemetry_view["priority"].astype(str) == selected_telemetry_priority]
        if telemetry_search:
            haystack = telemetry_view.astype(str).agg(" ".join, axis=1)
            telemetry_view = telemetry_view[haystack.str.contains(telemetry_search, case=False, na=False)]
        telemetry_cols = [
            col
            for col in [
                "model",
                "loss_term",
                "telemetry_family",
                "priority",
                "current_signal",
                "missing_telemetry",
                "implementation_hook",
                "pass_gate",
                "fail_action",
                "expected_metric_link",
                "interview_line",
            ]
            if col in telemetry_view.columns
        ]
        st.dataframe(telemetry_view[telemetry_cols], width="stretch", hide_index=True)
        doc_path = Path("docs/DEEPUplift_DEEP_MODEL_TELEMETRY_GAP_MATRIX.md")
        if doc_path.exists():
            st.download_button(
                "Download deep model telemetry gap matrix",
                data=doc_path.read_bytes(),
                file_name=doc_path.name,
                mime="text/markdown",
                key="download_deep_model_telemetry_gap_matrix",
            )
    else:
        st.info("Run `scripts/generate_deep_model_telemetry_gap_matrix.py` to refresh deep model telemetry gaps.")

    if deep_model_tuned_benchmark.get("leaderboard"):
        st.subheader("Deep Model Battle Report")
        st.caption(
            "CFRNet / DragonNet / EFIN / DESCN 的 tuned preset 和 ablation 与 T-Learner/DRLearner 强基线同台比较；"
            "这里专门解释深度模型为什么赢、为什么输，以及哪些 claim 只能作为 ranking/policy candidate。"
        )
        render_deep_model_tuned_dashboard(
            deep_model_tuned_benchmark,
            deep_model_tuned_diff,
            key_prefix="evidence_deep_tuned_benchmark",
        )
    else:
        st.info("Run `scripts/run_deep_model_tuned_benchmark.py` to refresh tuned deep-model battle evidence.")

    if paper_benchmark.get("leaderboard"):
        st.subheader("Paper-Level Benchmark Evidence")
        st.caption(
            "IHDP、ACIC-style synthetic 和 known-CATE synthetic 被同一 suite 跑通；每个 ok run 都保留 PEHE/ATE/QINI/AUUC/"
            "policy value、bootstrap CI、run diff 和 evidence manifest。"
        )
        render_paper_benchmark_dashboard(
            paper_benchmark,
            paper_benchmark_diff,
            paper_benchmark_interpretation,
            key_prefix="evidence_paper_benchmark",
        )
    else:
        st.info("Run `scripts/run_paper_benchmark_suite.py` to refresh IHDP / ACIC-style / known-CATE paper-level benchmark evidence.")

    st.subheader("Model Deconstruction Evidence")
    deconstruction_coverage = model_deconstruction_catalog.get("coverage") or {}
    deconstruction_rows = model_deconstruction_agent.get("rows") or []
    if deconstruction_coverage or deconstruction_rows:
        render_stat_grid(
            [
                ("Models", f"{deconstruction_coverage.get('models', 'NA')}", False),
                ("Golden Q", f"{deconstruction_coverage.get('golden_questions', model_deconstruction_agent.get('golden_questions', 'NA'))}", False),
                ("Agent Status", f"{model_deconstruction_agent.get('status', 'NA')}", True),
                (
                    "Passed",
                    f"{sum(1 for row in deconstruction_rows if isinstance(row, dict) and row.get('passed'))}/{len(deconstruction_rows) or 'NA'}",
                    True,
                ),
            ],
            columns=4,
        )
        if deconstruction_rows:
            st.dataframe(
                pd.DataFrame(deconstruction_rows)[["prompt", "model", "rubric_ratio", "passed", "missing_terms"]],
                width="stretch",
                hide_index=True,
            )
        st.caption(
            "This is the UI evidence that Model Deconstruction Agent 3.0 is grounded in docs, source paths, smoke reports and license gates."
        )
    else:
        st.info("Run `scripts/generate_model_deconstruction_docs.py` and `scripts/smoke_model_deconstruction_agent.py` to refresh model deconstruction evidence.")

    st.subheader("Scenario Model Decision Ladder")
    st.caption("把发券、广告、增长、推荐、marketplace 和 LLM routing 的模型升级路线固定成可审计证据。")
    st.dataframe(pd.DataFrame(scenario_model_ladders()), width="stretch", hide_index=True)

    st.subheader("Demo Preset Source Trace")
    st.caption("把演示入口、数据集、evaluator、metrics.json、Compare/History 字段和验证脚本放在同一张证据表里。")
    st.dataframe(pd.DataFrame(frontier_demo_trace_rows()), width="stretch", hide_index=True)

    st.subheader("Source Quality Gate")
    quality_rows = [{"quality": key, "sources": value} for key, value in sorted(quality_counts.items())]
    if quality_rows:
        st.dataframe(pd.DataFrame(quality_rows), width="stretch", hide_index=True)
    warnings = source_audit.get("warnings") or []
    if warnings:
        st.caption("Warnings: " + " | ".join(warnings[:3]))

    st.subheader("Research Evidence Gate")
    st.caption("参考 deep research / GPT Researcher 类框架，但只吸收可验证的研究流程：来源质量、claim-to-artifact、UI proof、smoke proof 和证据包。")
    research_gate = pd.DataFrame(research_evidence_gate_rows_cn())
    if not research_gate.empty:
        st.dataframe(research_gate, width="stretch", hide_index=True)

    st.subheader("GitHub Source Evidence Gate")
    github_rows = github_framework_audit.get("rows") or []
    if github_rows:
        github_view = pd.DataFrame(github_rows)
        visible_cols = [
            col
            for col in [
                "framework",
                "repo",
                "stargazers_count",
                "updated_at",
                "pushed_at",
                "archived",
                "activity",
                "api_status",
                "adoption_status",
                "integration_path",
                "next_step",
            ]
            if col in github_view.columns
        ]
        st.caption("GitHub API metadata is used as source evidence only; dependency promotion still requires local import/training/smoke gates.")
        st.dataframe(github_view[visible_cols], width="stretch", hide_index=True)
    else:
        st.info("Run `scripts/refresh_github_framework_audit.py` to refresh GitHub repo metadata evidence.")

    st.subheader("Latest Training Smoke")
    regression_rows = regression.get("results") or regression.get("runs") or []
    if regression_rows:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "model": row.get("model"),
                        "status": row.get("status"),
                        "qini": row.get("qini"),
                        "auuc": row.get("auuc"),
                        "sensitivity": (row.get("sensitivity") or {}).get("verdict"),
                        "run_id": row.get("run_id"),
                    }
                    for row in regression_rows
                    if isinstance(row, dict)
                ]
            ),
            width="stretch",
            hide_index=True,
        )

    st.subheader("LLM Routing Smoke")
    llm_rows = []
    for name, row in (llm_smoke.get("scenarios") or {}).items():
        llm_rows.append({"scenario": name, **row})
    if llm_rows:
        st.dataframe(pd.DataFrame(llm_rows), width="stretch", hide_index=True)
    else:
        st.info("Run `scripts/smoke_llm_routing_policy.py` to refresh deterministic LLM routing policy evidence.")

    st.subheader("LLM Routing OPE Smoke")
    if llm_ope_smoke.get("status") == "ok":
        ope_rows = (llm_ope_smoke.get("summary") or {}).get("rows") or []
        coverage_rows = llm_ope_smoke.get("coverage_rows") or []
        drift_rows = llm_ope_smoke.get("drift_rows") or []
        st.caption("Validates DM / IPS / SNIPS / DR OPE, action coverage and fixed-threshold-vs-bandit drift examples.")
        if ope_rows:
            st.dataframe(pd.DataFrame(ope_rows), width="stretch", hide_index=True)
        if coverage_rows:
            with st.expander("OPE action coverage", expanded=False):
                st.dataframe(pd.DataFrame(coverage_rows), width="stretch", hide_index=True)
        if drift_rows:
            with st.expander("Routing drift lab", expanded=False):
                st.dataframe(pd.DataFrame(drift_rows), width="stretch", hide_index=True)
    else:
        st.info("Run `scripts/smoke_llm_routing_ope.py` to refresh OPE / exploration / bandit drift evidence.")

    st.subheader("Continuous Treatment Policy Smoke")
    if continuous_policy_smoke.get("status") == "ok":
        st.caption("Validates the DRNet/VCNet plugin path with a local dose-response baseline, cost-aware dose optimizer and unsafe-extrapolation guardrail.")
        summary_rows = continuous_policy_smoke.get("policy_rows") or []
        if summary_rows:
            st.dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)
        model_rows = continuous_policy_smoke.get("model_gate_rows") or []
        if model_rows:
            with st.expander("DRNet / VCNet guarded adapter gate", expanded=False):
                st.dataframe(pd.DataFrame(model_rows), width="stretch", hide_index=True)
    else:
        st.info("Run `scripts/smoke_continuous_treatment_policy.py` to refresh continuous-treatment optimizer evidence.")

    st.subheader("Frontier Synthetic Dataset Smoke")
    frontier_dataset_rows = frontier_dataset_smoke.get("datasets") or []
    if frontier_dataset_rows:
        st.dataframe(pd.DataFrame(frontier_dataset_rows), width="stretch", hide_index=True)
    else:
        st.info("Run `scripts/generate_frontier_synthetic_datasets.py` and `scripts/smoke_frontier_synthetic_datasets.py` to refresh ECUP/delayed-feedback dataset evidence.")

    st.subheader("Industrial Scenario Dataset Smoke")
    industrial_dataset_rows = industrial_dataset_smoke.get("datasets") or []
    if industrial_dataset_rows:
        st.caption("Validates that coupon, growth, ads, recommendation, marketplace and LLM-routing datasets exist, avoid oracle leakage and expose policy-value columns.")
        st.dataframe(pd.DataFrame(industrial_dataset_rows), width="stretch", hide_index=True)
    else:
        st.info("Run `scripts/generate_industrial_scenario_datasets.py` and `scripts/smoke_industrial_scenario_datasets.py` to refresh scenario dataset evidence.")

    st.subheader("Industrial Scenario Training Smoke")
    industrial_training_rows = industrial_training_smoke.get("runs") or []
    if industrial_training_rows:
        st.caption("One lightweight runnable baseline is trained per scenario, proving the data, trainer, evaluator, artifacts and policy metrics are wired together.")
        st.dataframe(pd.DataFrame(industrial_training_rows), width="stretch", hide_index=True)
    else:
        st.info("Run `scripts/smoke_industrial_scenario_training.py` to refresh scenario training/evaluation evidence.")

    st.subheader("Industrial Policy Metrics Smoke")
    industrial_policy_rows = industrial_policy_smoke.get("checks") or []
    if industrial_policy_rows:
        st.caption("Validates evaluator-level scenario ROI metrics for coupon profit, recommendation intervention, marketplace subsidy and LLM routing.")
        st.dataframe(pd.DataFrame(industrial_policy_rows), width="stretch", hide_index=True)
    else:
        st.info("Run `scripts/smoke_industrial_policy_metrics.py` to refresh evaluator-level industrial policy metric evidence.")

    st.subheader("Resume Scenario Policy Smoke")
    resume_policy_rows = resume_policy_smoke.get("policy_rows") or []
    if resume_policy_rows:
        st.caption("Validates Baidu Waimai, DiDi and Shopee top-K policy benchmarks plus the DiDi city/time budget allocation simulator.")
        st.dataframe(pd.DataFrame(resume_policy_rows), width="stretch", hide_index=True)
        budget_summary = (resume_policy_smoke.get("budget_plan") or {}).get("summary") or {}
        if budget_summary:
            st.json(budget_summary)
    else:
        st.info("Run `scripts/smoke_resume_scenario_policy.py` to refresh resume-specific policy benchmark evidence.")

    st.subheader("Industrial Scenario Compare Smoke")
    industrial_compare_best = industrial_compare_smoke.get("best_by_dataset") or []
    industrial_compare_runs = industrial_compare_smoke.get("runs") or []
    if industrial_compare_best:
        st.caption("Scenario-aware multi-model compare: ranks runnable P0 candidates per business setting using readiness, uplift and scenario policy metrics.")
        st.dataframe(pd.DataFrame(industrial_compare_best), width="stretch", hide_index=True)
        with st.expander("All scenario compare runs", expanded=False):
            st.dataframe(pd.DataFrame(industrial_compare_runs), width="stretch", hide_index=True)
    else:
        st.info("Run `scripts/smoke_industrial_scenario_compare.py` to refresh scenario-aware multi-model compare evidence.")

    st.subheader("Frontier Evaluator Metrics Smoke")
    frontier_metric_rows = frontier_evaluator_smoke.get("checks") or []
    if frontier_metric_rows:
        st.caption("Validates that ECUP full-funnel and delayed-feedback metrics are emitted by the shared evaluator, not only described in model cards.")
        st.dataframe(pd.DataFrame(frontier_metric_rows), width="stretch", hide_index=True)
    else:
        st.info("Run `scripts/smoke_frontier_evaluator_metrics.py` to refresh evaluator-level ECUP/delayed-feedback evidence.")

    st.subheader("Frontier Training Evidence")
    frontier_training_rows = frontier_training_evidence.get("rows") or []
    if frontier_training_rows:
        st.dataframe(pd.DataFrame(frontier_training_rows), width="stretch", hide_index=True)
    else:
        st.info("Run frontier no-UI compare smokes and `scripts/summarize_frontier_training_evidence.py` to refresh training evidence.")

    st.subheader("Download Evidence Pack")
    evidence_docs = [
        ("Evidence index", Path("docs/DEEPUplift_AGENT_EVIDENCE_INDEX.md")),
        ("Interview evidence pack", Path("docs/DEEPUplift_AGENT_INTERVIEW_EVIDENCE_PACK.md")),
        ("Model capability architecture", Path("docs/UPLIFT_MODEL_CAPABILITY_ARCHITECTURE.md")),
        ("Model deconstruction index", Path("docs/DEEPUplift_MODEL_DECONSTRUCTION_INDEX.md")),
        ("Model deconstruction catalog", Path((artifacts.get("model_deconstruction_catalog") or {}).get("path", ""))),
        ("Model deconstruction agent report", Path((artifacts.get("model_deconstruction_agent") or {}).get("path", ""))),
        ("Frontier training evidence", Path("docs/FRONTIER_DATASET_TRAINING_EVIDENCE.md")),
        ("Industrial scenario lab", Path("docs/UPLIFT_INDUSTRIAL_SCENARIO_LAB.md")),
        ("Model failure modes", Path("docs/UPLIFT_MODEL_FAILURE_MODES.md")),
        ("Scenario pitfalls", Path("docs/UPLIFT_SCENARIO_PITFALLS.md")),
        ("Diagnostic cases", Path("docs/UPLIFT_DIAGNOSTIC_CASES.md")),
        ("Metric misleading cases", Path("docs/UPLIFT_METRIC_MISLEADING_CASES.md")),
        ("LLM interview Q&A", Path("docs/UPLIFT_LLM_INTERVIEW_QA.md")),
        ("Industry references", Path("docs/UPLIFT_INDUSTRY_REFERENCES.md")),
        ("Interview ZIP", interview_zip_path),
    ]
    cols = st.columns(3)
    for index, (label, path) in enumerate(evidence_docs):
        with cols[index % len(cols)]:
            if path.is_file():
                st.download_button(
                    f"Download {label}",
                    data=path.read_bytes(),
                    file_name=path.name,
                    mime=artifact_mime(path),
                    key=f"evidence_download_{path.name}",
                )


def inject_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            color-scheme: dark;
            --du-bg: #060b13;
            --du-bg-soft: #0a101b;
            --du-panel: #0f1724;
            --du-panel-2: #121d2d;
            --du-border: #263244;
            --du-border-strong: #364256;
            --du-text: #e7edf6;
            --du-muted: #9aa7b8;
            --du-faint: #66758a;
            --du-accent: #22d3ee;
            --du-accent-strong: #14b8a6;
            --du-success: #34d399;
            --du-warning: #f59e0b;
            --du-danger: #fb7185;
            --du-info: #60a5fa;
        }
        html, body, .stApp, [data-testid="stAppViewContainer"] {
            color: var(--du-text);
            background:
                radial-gradient(circle at 22% 0%, rgba(34, 211, 238, 0.09), transparent 28rem),
                radial-gradient(circle at 82% 8%, rgba(96, 165, 250, 0.07), transparent 30rem),
                linear-gradient(180deg, #070b13 0%, #060b13 44%, #08101b 100%);
        }
        [data-testid="stAppViewContainer"] {
            background:
                linear-gradient(rgba(148, 163, 184, 0.026) 1px, transparent 1px),
                linear-gradient(90deg, rgba(148, 163, 184, 0.022) 1px, transparent 1px),
                radial-gradient(circle at 18% 0%, rgba(34, 211, 238, 0.07), transparent 34rem),
                linear-gradient(180deg, #070b13 0%, #060b13 100%);
            background-size: 32px 32px, 32px 32px, auto, auto;
        }
        [data-testid="stHeader"] {
            background: rgba(7, 17, 31, 0.88);
            color: var(--du-text);
            border-bottom: 1px solid rgba(45, 212, 191, 0.13);
            backdrop-filter: blur(10px);
        }
        [data-testid="stHeader"] * {
            color: var(--du-text) !important;
        }
        .block-container {
            padding-top: 3.4rem;
            padding-bottom: 2.5rem;
            max-width: 1500px;
        }
        [data-testid="stSidebar"] {
            background:
                linear-gradient(180deg, rgba(15, 23, 42, 0.96) 0%, rgba(8, 17, 31, 0.98) 100%);
            border-right: 1px solid rgba(51, 65, 85, 0.72);
            box-shadow: 12px 0 34px rgba(0, 0, 0, 0.22);
        }
        [data-testid="stSidebar"] *:not(button):not(svg):not(path),
        .block-container *:not(button):not(svg):not(path),
        [data-testid="stMarkdownContainer"],
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        label,
        h1, h2, h3, h4, h5, h6 {
            color: var(--du-text);
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stWidgetLabel"] p {
            color: #cbd5e1;
            font-weight: 620;
        }
        .du-hero {
            position: relative;
            background:
                linear-gradient(135deg, rgba(34, 211, 238, 0.12) 0%, rgba(17, 24, 39, 0) 42%),
                linear-gradient(90deg, #0f1724 0%, #141f2f 54%, #182232 100%);
            border: 1px solid rgba(34, 211, 238, 0.22);
            border-radius: 8px;
            padding: 20px 24px 22px;
            margin-bottom: 22px;
            box-shadow: 0 22px 46px rgba(15, 23, 42, 0.22);
            overflow: hidden;
        }
        .du-hero:before {
            content: "";
            position: absolute;
            inset: 0;
            pointer-events: none;
            background-image:
                linear-gradient(rgba(255,255,255,0.055) 1px, transparent 1px),
                linear-gradient(90deg, rgba(255,255,255,0.045) 1px, transparent 1px);
            background-size: 28px 28px;
            opacity: 0.32;
        }
        .du-hero > * {
            position: relative;
            z-index: 1;
        }
        .du-hero,
        .du-hero * {
            color: #f8fafc !important;
        }
        .du-hero-kicker {
            color: #67e8f9 !important;
            font-size: 0.74rem;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 8px;
        }
        .du-title {
            color: #f8fafc !important;
            font-size: clamp(1.4rem, 1.8vw, 1.8rem);
            font-weight: 760;
            line-height: 1.15;
            margin: 0 0 6px 0;
            letter-spacing: 0;
        }
        .du-subtitle {
            color: #cbd5e1 !important;
            font-size: 0.98rem;
            margin: 0 0 12px 0;
        }
        .du-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }
        .du-chip {
            border: 1px solid rgba(148, 163, 184, 0.46);
            background: rgba(15, 23, 42, 0.45);
            border-radius: 999px;
            color: #e5e7eb !important;
            padding: 4px 10px;
            font-size: 0.78rem;
            white-space: nowrap;
            backdrop-filter: blur(8px);
        }
        .du-chip strong {
            color: #67e8f9 !important;
        }
        .du-status-row {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 10px;
            margin-top: 16px;
            max-width: 900px;
        }
        .du-status {
            border: 1px solid rgba(148, 163, 184, 0.32);
            background: rgba(255, 255, 255, 0.06);
            border-radius: 8px;
            padding: 9px 11px;
        }
        .du-status-label {
            color: #94a3b8 !important;
            font-size: 0.68rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .du-status-value {
            color: #f8fafc !important;
            font-size: 0.95rem;
            font-weight: 740;
            margin-top: 3px;
        }
        .du-stat-grid {
            display: grid;
            grid-template-columns: repeat(var(--du-stat-cols, 4), minmax(0, 1fr));
            gap: 14px;
            margin: 12px 0 18px;
        }
        .du-stat-card {
            background: linear-gradient(180deg, rgba(17, 28, 49, 0.98) 0%, rgba(15, 23, 42, 0.98) 100%);
            border: 1px solid rgba(51, 65, 85, 0.86);
            border-radius: 8px;
            padding: 14px 16px;
            min-height: 92px;
            box-shadow: 0 12px 28px rgba(0, 0, 0, 0.18);
        }
        .du-stat-label {
            color: var(--du-muted);
            font-size: 0.78rem;
            font-weight: 700;
            line-height: 1.2;
            margin-bottom: 9px;
        }
        .du-stat-value {
            color: var(--du-text);
            font-size: clamp(1.45rem, 2vw, 2.05rem);
            font-weight: 720;
            line-height: 1.08;
            letter-spacing: 0;
            overflow-wrap: anywhere;
        }
        .du-stat-value.is-text {
            font-size: clamp(1.15rem, 1.45vw, 1.55rem);
            line-height: 1.18;
        }
        .du-backend-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 12px;
            margin: 12px 0 16px;
        }
        .du-backend-card {
            border: 1px solid rgba(51, 65, 85, 0.86);
            background: linear-gradient(180deg, rgba(17, 28, 49, 0.98) 0%, rgba(15, 23, 42, 0.98) 100%);
            border-radius: 8px;
            padding: 13px 14px;
            min-height: 104px;
            box-shadow: 0 12px 28px rgba(0, 0, 0, 0.18);
        }
        .du-backend-kicker {
            color: var(--du-accent);
            font-size: 0.68rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .du-backend-title {
            color: var(--du-text);
            font-size: 1.03rem;
            font-weight: 760;
            line-height: 1.18;
            margin-top: 5px;
        }
        .du-backend-meta {
            color: var(--du-muted);
            font-size: 0.78rem;
            line-height: 1.35;
            margin-top: 8px;
        }
        div[data-testid="stMetric"] {
            background: linear-gradient(180deg, rgba(17, 28, 49, 0.98) 0%, rgba(15, 23, 42, 0.98) 100%);
            border: 1px solid rgba(51, 65, 85, 0.86);
            border-radius: 8px;
            padding: 13px 16px;
            box-shadow: 0 12px 28px rgba(0, 0, 0, 0.18);
            min-height: 104px;
        }
        div[data-testid="stMetricLabel"] p {
            color: var(--du-muted);
            font-weight: 650;
            font-size: 0.83rem;
            line-height: 1.2;
        }
        div[data-testid="stMetricValue"] {
            color: var(--du-text);
            font-size: clamp(1.45rem, 2.1vw, 2.15rem);
            line-height: 1.08;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 6px;
            border-bottom: 1px solid rgba(51, 65, 85, 0.8);
            overflow-x: auto;
            overflow-y: hidden;
            scrollbar-width: thin;
            padding-bottom: 1px;
        }
        .stTabs [data-baseweb="tab"] {
            border: 1px solid rgba(51, 65, 85, 0.78);
            border-bottom: 0;
            border-radius: 8px 8px 0 0;
            background: rgba(15, 23, 42, 0.86);
            padding: 7px 11px;
            min-width: fit-content;
            white-space: nowrap;
            flex: 0 0 auto;
        }
        .stTabs [data-baseweb="tab"] p {
            font-size: 0.9rem;
            white-space: nowrap;
            color: #dbeafe !important;
        }
        .stTabs [aria-selected="true"] {
            background: #0e1b2d;
            color: #ffffff;
            border-color: rgba(45, 212, 191, 0.64);
        }
        .stTabs [data-baseweb="tab-highlight"] {
            background-color: var(--du-accent);
            height: 3px;
        }
        .stTabs [aria-selected="true"] *,
        .stButton button[kind="primary"] * {
            color: #ffffff !important;
        }
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        div[data-baseweb="textarea"] > div {
            background: rgba(15, 23, 42, 0.98);
            border-color: rgba(71, 85, 105, 0.92);
            color: var(--du-text);
        }
        div[data-baseweb="select"] > div:hover,
        div[data-baseweb="input"] > div:hover,
        div[data-baseweb="textarea"] > div:hover {
            border-color: rgba(45, 212, 191, 0.82);
        }
        div[data-baseweb="select"] span,
        div[data-baseweb="select"] div,
        div[data-baseweb="input"] input,
        div[data-baseweb="textarea"] textarea {
            color: var(--du-text) !important;
        }
        [data-testid="stSidebar"] div[data-baseweb="select"] > div,
        [data-testid="stSidebar"] div[data-baseweb="input"] > div,
        [data-testid="stSidebar"] div[data-baseweb="textarea"] > div {
            background: rgba(8, 17, 31, 0.92);
            border-color: rgba(71, 85, 105, 0.86);
        }
        [data-testid="stSidebar"] div[data-baseweb="select"] span,
        [data-testid="stSidebar"] div[data-baseweb="select"] div,
        [data-testid="stSidebar"] div[data-baseweb="input"] input,
        [data-testid="stSidebar"] div[data-baseweb="textarea"] textarea {
            color: var(--du-text) !important;
        }
        [data-testid="stSidebar"] div[data-baseweb="select"] svg,
        [data-testid="stSidebar"] div[data-baseweb="input"] svg {
            color: #cbd5e1 !important;
            fill: #cbd5e1 !important;
        }
        .stButton button, .stDownloadButton button, a[data-testid="stLinkButton"] {
            border-radius: 8px;
            border-color: rgba(71, 85, 105, 0.9);
            background: rgba(15, 23, 42, 0.96);
            color: var(--du-text) !important;
            min-height: 38px;
            font-weight: 620;
        }
        .stButton button:hover, .stDownloadButton button:hover, a[data-testid="stLinkButton"]:hover {
            border-color: rgba(45, 212, 191, 0.85);
            color: #ffffff !important;
            background: rgba(20, 184, 166, 0.16);
        }
        .stDataFrame {
            border: 1px solid rgba(51, 65, 85, 0.86);
            border-radius: 8px;
            overflow: hidden;
            background: rgba(15, 23, 42, 0.72);
        }
        [data-testid="stVerticalBlock"] > [style*="flex-direction: column"] {
            gap: 0.85rem;
        }
        [data-testid="stCaptionContainer"],
        .stCaptionContainer,
        small {
            color: var(--du-muted) !important;
        }
        div[data-testid="stExpander"] {
            background: rgba(15, 23, 42, 0.72);
            border: 1px solid rgba(51, 65, 85, 0.86);
            border-radius: 8px;
        }
        div[data-testid="stExpander"] summary,
        div[data-testid="stExpander"] summary * {
            color: var(--du-text) !important;
        }
        [data-testid="stChatMessage"] {
            background: rgba(15, 23, 42, 0.72);
            border: 1px solid rgba(51, 65, 85, 0.72);
            border-radius: 8px;
            padding: 10px 12px;
        }
        [data-testid="stChatInput"] {
            background: rgba(15, 23, 42, 0.94);
            border-top: 1px solid rgba(51, 65, 85, 0.86);
        }
        [data-testid="stChatInput"] textarea {
            color: var(--du-text) !important;
        }
        div[data-testid="stAlert"] {
            background: rgba(15, 23, 42, 0.94);
            border: 1px solid rgba(45, 212, 191, 0.22);
            color: var(--du-text);
        }
        div[data-baseweb="tag"] {
            background: rgba(20, 184, 166, 0.78) !important;
            color: #04111f !important;
        }
        div[data-baseweb="tag"] span,
        div[data-baseweb="tag"] svg {
            color: #04111f !important;
            fill: #04111f !important;
        }
        [role="radiogroup"] label,
        [data-baseweb="radio"] {
            color: var(--du-text) !important;
        }
        section[data-testid="stSidebar"] .stRadio, section[data-testid="stSidebar"] .stSelectbox,
        section[data-testid="stSidebar"] .stMultiSelect, section[data-testid="stSidebar"] .stNumberInput,
        section[data-testid="stSidebar"] .stSlider, section[data-testid="stSidebar"] .stTextArea {
            margin-bottom: 0.55rem;
        }
        .decision-board {
            border: 1px solid rgba(51, 65, 85, 0.86);
            border-radius: 8px;
            background: linear-gradient(180deg, rgba(17, 28, 49, 0.92) 0%, rgba(8, 17, 31, 0.92) 100%);
            padding: 10px 12px;
            margin: 8px 0 18px;
        }
        .decision-board-head,
        .decision-board-row {
            display: grid;
            grid-template-columns: minmax(150px, 0.24fr) minmax(0, 0.76fr);
            gap: 14px;
            align-items: center;
        }
        .decision-board-head {
            color: var(--du-muted);
            font-size: 0.74rem;
            font-weight: 760;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            padding: 2px 4px 8px;
        }
        .decision-board-row {
            border-top: 1px solid rgba(51, 65, 85, 0.64);
            padding: 10px 4px;
        }
        .decision-board-model {
            color: var(--du-text);
            font-weight: 700;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .decision-board-bars {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 10px;
        }
        .decision-mini-bar {
            display: grid;
            grid-template-columns: 66px minmax(0, 1fr) 32px;
            gap: 8px;
            align-items: center;
            min-width: 0;
        }
        .decision-mini-bar span {
            color: var(--du-muted);
            font-size: 0.76rem;
            font-weight: 650;
            white-space: nowrap;
        }
        .decision-mini-track {
            height: 8px;
            border-radius: 999px;
            background: rgba(51, 65, 85, 0.88);
            overflow: hidden;
            box-shadow: inset 0 0 0 1px rgba(148, 163, 184, 0.08);
        }
        .decision-mini-track i {
            display: block;
            height: 100%;
            border-radius: 999px;
            box-shadow: 0 0 12px rgba(45, 212, 191, 0.28);
        }
        .decision-mini-bar b {
            color: var(--du-text);
            font-size: 0.76rem;
            text-align: right;
            font-weight: 700;
        }
        .rationale-panel {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 10px;
            margin: 8px 0 18px;
        }
        .rationale-card {
            border: 1px solid rgba(51, 65, 85, 0.86);
            background: linear-gradient(180deg, rgba(17, 28, 49, 0.96) 0%, rgba(15, 23, 42, 0.96) 100%);
            border-radius: 8px;
            padding: 11px 12px;
            min-height: 88px;
            box-shadow: 0 12px 24px rgba(0, 0, 0, 0.15);
        }
        .rationale-card span {
            display: block;
            color: var(--du-muted);
            font-size: 0.68rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 7px;
        }
        .rationale-card b {
            display: block;
            color: var(--du-text);
            font-size: 0.9rem;
            line-height: 1.3;
            font-weight: 700;
            overflow-wrap: anywhere;
        }
        .run-status-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 10px;
            margin: 10px 0 12px;
        }
        .run-status-card {
            border: 1px solid rgba(51, 65, 85, 0.86);
            background: linear-gradient(180deg, rgba(17, 28, 49, 0.96) 0%, rgba(15, 23, 42, 0.96) 100%);
            border-radius: 8px;
            padding: 10px 12px;
            min-height: 72px;
        }
        .run-status-card span {
            display: block;
            color: var(--du-muted);
            font-size: 0.68rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 6px;
        }
        .run-status-card b {
            display: block;
            color: var(--du-text);
            font-size: 1.05rem;
            line-height: 1.2;
            overflow-wrap: anywhere;
        }
        .du-badge {
            display: inline-flex;
            align-items: center;
            border: 1px solid rgba(148, 163, 184, 0.35);
            border-radius: 999px;
            padding: 2px 8px;
            font-size: 0.72rem;
            font-weight: 780;
            line-height: 1.3;
            color: #e5edf7 !important;
            background: rgba(15, 23, 42, 0.86);
            white-space: nowrap;
        }
        .du-badge-ready {
            color: #a7f3d0 !important;
            border-color: rgba(52, 211, 153, 0.46);
            background: rgba(6, 78, 59, 0.28);
        }
        .du-badge-inspired {
            color: #bae6fd !important;
            border-color: rgba(56, 189, 248, 0.42);
            background: rgba(12, 74, 110, 0.28);
        }
        .du-badge-future-react {
            color: #fde68a !important;
            border-color: rgba(245, 158, 11, 0.42);
            background: rgba(120, 53, 15, 0.28);
        }
        .du-badge-backlog {
            color: #fecdd3 !important;
            border-color: rgba(244, 63, 94, 0.38);
            background: rgba(136, 19, 55, 0.24);
        }
        .du-ux-grid,
        .du-prompt-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px;
            margin: 12px 0 18px;
        }
        .du-ux-card,
        .du-prompt-card {
            border: 1px solid rgba(51, 65, 85, 0.86);
            background:
                linear-gradient(180deg, rgba(17, 28, 49, 0.97) 0%, rgba(15, 23, 42, 0.97) 100%);
            border-radius: 8px;
            padding: 13px 14px;
            min-height: 156px;
            box-shadow: 0 12px 26px rgba(0, 0, 0, 0.17);
        }
        .du-ux-kicker {
            color: #fbbf24 !important;
            font-size: 0.68rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 8px;
        }
        .du-ux-title {
            color: #fecdd3 !important;
            font-size: 0.76rem;
            font-weight: 800;
            text-transform: uppercase;
            margin-top: 4px;
        }
        .du-ux-title-after {
            color: #86efac !important;
            margin-top: 10px;
        }
        .du-ux-card p,
        .du-prompt-card li {
            color: #cbd5e1 !important;
            font-size: 0.82rem;
            line-height: 1.42;
            margin: 4px 0;
        }
        .du-ux-proof {
            margin-top: 10px;
            color: #93c5fd !important;
            font-size: 0.76rem;
            font-weight: 700;
            border-top: 1px solid rgba(51, 65, 85, 0.7);
            padding-top: 8px;
        }
        .du-prompt-card ul {
            margin: 10px 0 0 0;
            padding-left: 17px;
        }
        .du-workbench-rail {
            display: grid;
            grid-template-columns: repeat(7, minmax(0, 1fr));
            gap: 10px;
            margin: 12px 0 18px;
        }
        .du-stage-card,
        .du-decision-card {
            border: 1px solid rgba(54, 66, 86, 0.88);
            background:
                linear-gradient(180deg, rgba(18, 29, 45, 0.96) 0%, rgba(12, 18, 30, 0.98) 100%);
            border-radius: 8px;
            padding: 12px 13px;
            box-shadow: 0 12px 24px rgba(0, 0, 0, 0.16);
            min-width: 0;
        }
        .du-stage-card {
            min-height: 138px;
        }
        .du-stage-name,
        .du-decision-kicker {
            color: var(--du-muted) !important;
            font-size: 0.66rem;
            font-weight: 820;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .du-stage-status {
            display: inline-flex;
            margin-top: 8px;
            border-radius: 999px;
            padding: 2px 8px;
            color: #06101d !important;
            background: var(--du-info);
            font-size: 0.68rem;
            font-weight: 800;
            max-width: 100%;
            overflow-wrap: anywhere;
        }
        .du-stage-status.is-ready,
        .du-stage-status.is-active,
        .du-stage-status.is-audited,
        .du-stage-status.is-governed,
        .du-stage-status.is-compared {
            background: var(--du-success);
        }
        .du-stage-status.is-needs-compare,
        .du-stage-status.is-pending,
        .du-stage-status.is-decision {
            background: var(--du-warning);
        }
        .du-stage-signal {
            color: var(--du-text) !important;
            font-size: 0.92rem;
            font-weight: 750;
            line-height: 1.25;
            margin-top: 10px;
            overflow-wrap: anywhere;
        }
        .du-stage-meta {
            color: var(--du-muted) !important;
            font-size: 0.74rem;
            line-height: 1.35;
            margin-top: 7px;
        }
        .du-decision-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 12px;
            margin: 12px 0 18px;
        }
        .du-decision-title {
            color: var(--du-text) !important;
            font-size: 0.98rem;
            font-weight: 760;
            line-height: 1.25;
            margin-top: 6px;
        }
        .du-decision-meta {
            color: var(--du-muted) !important;
            font-size: 0.78rem;
            line-height: 1.38;
            margin-top: 8px;
        }
        .du-control-plane {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px;
            margin: 12px 0 18px;
        }
        .du-control-card {
            border: 1px solid rgba(54, 66, 86, 0.88);
            background:
                linear-gradient(180deg, rgba(17, 27, 43, 0.98) 0%, rgba(9, 14, 24, 0.99) 100%);
            border-radius: 8px;
            padding: 13px 14px;
            min-height: 132px;
            box-shadow: 0 12px 24px rgba(0, 0, 0, 0.16);
        }
        .du-control-kicker {
            color: var(--du-muted) !important;
            font-size: 0.66rem;
            font-weight: 820;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .du-control-signal {
            color: var(--du-text) !important;
            font-size: 0.98rem;
            font-weight: 780;
            line-height: 1.26;
            margin-top: 8px;
        }
        .du-control-meta {
            color: var(--du-muted) !important;
            font-size: 0.78rem;
            line-height: 1.4;
            margin-top: 8px;
        }
        .du-control-status {
            display: inline-flex;
            margin-top: 9px;
            border-radius: 999px;
            padding: 2px 8px;
            color: #06101d !important;
            background: var(--du-info);
            font-size: 0.68rem;
            font-weight: 820;
        }
        .du-control-status.is-ready {
            background: var(--du-success);
        }
        .du-control-status.is-guarded,
        .du-control-status.is-decision {
            background: var(--du-warning);
        }
        .du-token-strip {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin: 8px 0 14px;
        }
        .du-token {
            display: inline-flex;
            border: 1px solid rgba(154, 167, 184, 0.35);
            border-radius: 999px;
            padding: 3px 9px;
            color: #dbeafe !important;
            background: rgba(15, 23, 36, 0.78);
            font-size: 0.74rem;
            font-weight: 720;
        }
        .du-flow-stack {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 10px;
            margin: 12px 0 18px;
        }
        .du-flow-step {
            border: 1px solid rgba(54, 66, 86, 0.88);
            background: linear-gradient(180deg, rgba(17, 28, 49, 0.97) 0%, rgba(8, 17, 31, 0.98) 100%);
            border-radius: 8px;
            padding: 12px 13px;
            min-height: 118px;
            box-shadow: 0 12px 24px rgba(0, 0, 0, 0.16);
        }
        .du-flow-step span {
            display: block;
            color: var(--du-accent) !important;
            font-size: 0.66rem;
            font-weight: 820;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 8px;
        }
        .du-flow-step b {
            display: block;
            color: var(--du-text) !important;
            font-size: 0.92rem;
            line-height: 1.28;
            overflow-wrap: anywhere;
        }
        .du-flow-step small {
            display: block;
            color: var(--du-muted) !important;
            font-size: 0.74rem;
            line-height: 1.35;
            margin-top: 8px;
        }
        @media (max-width: 900px) {
            .block-container {
                padding-top: 4.2rem;
            }
            .du-stat-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            .du-status-row {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            .du-backend-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            .decision-board-head,
            .decision-board-row {
                grid-template-columns: 1fr;
            }
            .decision-board-bars {
                grid-template-columns: 1fr;
            }
            .rationale-panel {
                grid-template-columns: 1fr;
            }
            .run-status-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            .du-ux-grid,
            .du-prompt-grid {
                grid-template-columns: 1fr;
            }
            .du-workbench-rail,
            .du-decision-grid,
            .du-control-plane,
            .du-flow-stack {
                grid-template-columns: 1fr;
            }
            .du-hero {
                padding: 15px 16px;
            }
            div[data-testid="stMetric"] {
                min-height: 90px;
            }
        }
        @media (max-width: 520px) {
            .du-stat-grid {
                grid-template-columns: 1fr;
            }
            .du-status-row {
                grid-template-columns: 1fr;
            }
            .du-backend-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    total_models = len(available_models())
    ready_models = len(available_models(only_available=True))
    st.markdown(
        f"""
        <div class="du-hero">
          <div class="du-hero-kicker">因果决策 OS</div>
          <div class="du-title">DeepUplift Agent Workbench</div>
          <div class="du-subtitle">按工业 uplift 建模链路组织：业务设计、数据诊断、模型选择、评估指标、ROI 策略、打分导出和证据复现。</div>
          <div class="du-chip-row">
            <span class="du-chip"><strong>{total_models}</strong> 个注册模型</span>
            <span class="du-chip"><strong>{ready_models}</strong> 个当前可训练</span>
            <span class="du-chip">流程 + 指标 + 证据</span>
            <span class="du-chip">本地可复现训练</span>
          </div>
          <div class="du-status-row">
            <div class="du-status"><div class="du-status-label">工作流</div><div class="du-status-value">设计 → 诊断 → 决策</div></div>
            <div class="du-status"><div class="du-status-label">训练后端</div><div class="du-status-value">Local + EconML + LightGBM</div></div>
            <div class="du-status"><div class="du-status-label">可信护栏</div><div class="du-status-value">Overlap + CI + Drift</div></div>
            <div class="du-status"><div class="du-status-label">上线决策</div><div class="du-status-value">Top-K + Policy Value</div></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stat_grid(items: list[tuple[str, object, bool]], columns: int = 4) -> None:
    cards = []
    for label, value, is_text in items:
        value_text = "" if value is None else str(value)
        text_class = " is-text" if is_text else ""
        cards.append(
            "<div class=\"du-stat-card\">"
            f"<div class=\"du-stat-label\">{html.escape(str(label))}</div>"
            f"<div class=\"du-stat-value{text_class}\">{html.escape(value_text)}</div>"
            "</div>"
        )
    st.markdown(
        f"<div class=\"du-stat-grid\" style=\"--du-stat-cols:{columns}\">{''.join(cards)}</div>",
        unsafe_allow_html=True,
    )


def render_backend_cards(catalog: pd.DataFrame) -> None:
    if catalog.empty:
        return
    status_counts = catalog["status"].value_counts().to_dict()
    source_counts = catalog.groupby("source").size().to_dict()
    cards = [
        ("Core Ready", "Local + EconML + LightGBM + scikit-uplift", status_counts.get("ready", 0), "可直接训练、对比、打分、导出。"),
        ("Guarded", "XGBoost", status_counts.get("guarded", 0), "已注册，需显式开关，当前机器保持安全禁用。"),
        ("Optional", "CatBoost", source_counts.get("CatBoost", 0), "适合类别特征重的工业表格，安装后自动 ready。"),
        ("Py 3.11", "CausalML", source_counts.get("CausalML", 0), "官方 uplift tree/forest/meta learner，走独立环境。"),
    ]
    html_cards = []
    for kicker, title, count, meta in cards:
        html_cards.append(
            "<div class=\"du-backend-card\">"
            f"<div class=\"du-backend-kicker\">{html.escape(kicker)}</div>"
            f"<div class=\"du-backend-title\">{html.escape(str(count))} · {html.escape(title)}</div>"
            f"<div class=\"du-backend-meta\">{html.escape(meta)}</div>"
            "</div>"
        )
    st.markdown(f"<div class=\"du-backend-grid\">{''.join(html_cards)}</div>", unsafe_allow_html=True)


def render_du_badge(value: str) -> str:
    status = str(value).lower().replace("_", "-")
    return f"<span class='du-badge du-badge-{html.escape(status)}'>{html.escape(str(value))}</span>"


def render_frontend_framework_audit(key_prefix: str = "frontend") -> None:
    st.subheader("Frontend Framework Audit / UI Inspiration Matrix")
    st.caption(
        "本轮只采用官方 GitHub repo、官方文档和成熟开源 dashboard / agent workbench 作为 UI 参考。"
        "目标不是重写 React，而是把最适合工业 uplift workbench 的信息架构和组件模式先吸收到当前 Streamlit。"
    )
    rows = pd.DataFrame(frontend_framework_audit_rows())
    counts = frontend_research_counts()
    render_stat_grid(
        [
            ("Audited", f"{counts['total']:,}", False),
            ("Ready Patterns", f"{counts['ready']:,}", False),
            ("Inspired", f"{counts['inspired']:,}", False),
            ("Future React", f"{counts['future_react']:,}", False),
            ("Backlog", f"{counts['backlog']:,}", False),
        ],
        columns=5,
    )
    if rows.empty:
        st.info("No frontend framework audit rows found.")
        return
    c1, c2, c3 = st.columns([0.22, 0.30, 0.48])
    with c1:
        status_filter = st.multiselect(
            "吸收状态",
            sorted(rows["status"].unique().tolist()),
            default=sorted(rows["status"].unique().tolist()),
            key=f"{key_prefix}_frontend_status",
        )
    with c2:
        category_filter = st.multiselect(
            "类别",
            sorted(rows["category"].unique().tolist()),
            default=sorted(rows["category"].unique().tolist()),
            key=f"{key_prefix}_frontend_category",
        )
    with c3:
        search_text = st.text_input("搜索框架 / 页面 / pattern", value="", key=f"{key_prefix}_frontend_search")
    view = rows[rows["status"].isin(status_filter) & rows["category"].isin(category_filter)].copy()
    if search_text.strip():
        pattern = re.escape(search_text.strip())
        mask = view.astype(str).apply(lambda col: col.str.contains(pattern, case=False, regex=True, na=False)).any(axis=1)
        view = view[mask]
    visible_cols = [
        "framework",
        "status",
        "category",
        "repo",
        "stars",
        "latest_update",
        "activity",
        "deepuplift_page",
        "ui_pattern",
        "streamlit_direct",
        "future_react_fit",
        "dependency_risk",
        "migration_cost",
        "interview_line",
        "github_url",
        "official_url",
    ]
    st.dataframe(view[visible_cols], width="stretch", hide_index=True)
    matrix = pd.DataFrame(ui_inspiration_matrix_rows())
    if not matrix.empty:
        st.subheader("UI Inspiration Matrix")
        matrix_view = matrix[["pattern", "borrowed_from", "absorbed_now", "deepuplift_value", "future_react", "status"]]
        st.dataframe(matrix_view, width="stretch", hide_index=True)
    st.download_button(
        "下载前端框架审计 CSV",
        data=dataframe_csv_bytes(rows),
        file_name="deepuplift_frontend_framework_audit.csv",
        mime="text/csv",
        key=f"{key_prefix}_frontend_audit_download",
    )


def render_ml_training_platform_audit(key_prefix: str = "ml_training") -> None:
    st.subheader("ML Training Platform Inspiration Matrix")
    st.caption(
        "参考 MLflow、W&B、ClearML、Kubeflow、DVC、Dagster、Ray、Aim 等官方 GitHub / docs，"
        "把训练平台的 Experiments / Runs / Models / Artifacts / Lineage / Gates 模式吸收到当前 Streamlit。"
    )
    rows = pd.DataFrame(ml_training_platform_audit_rows())
    counts = ml_training_platform_counts()
    render_stat_grid(
        [
            ("Training Platforms", f"{counts['total']:,}", False),
            ("Ready Patterns", f"{counts['ready']:,}", False),
            ("Inspired", f"{counts['inspired']:,}", False),
            ("Future React", f"{counts['future_react']:,}", False),
            ("Backlog", f"{counts['backlog']:,}", False),
        ],
        columns=5,
    )
    st.markdown(
        "<div class='du-token-strip'>"
        "<span class='du-token'>Experiments</span>"
        "<span class='du-token'>Runs</span>"
        "<span class='du-token'>Model Registry</span>"
        "<span class='du-token'>Artifacts</span>"
        "<span class='du-token'>Lineage</span>"
        "<span class='du-token'>Evidence Gates</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    if rows.empty:
        st.info("No ML training platform audit rows found.")
        return
    c1, c2, c3 = st.columns([0.22, 0.30, 0.48])
    with c1:
        status_filter = st.multiselect(
            "吸收状态",
            sorted(rows["status"].unique().tolist()),
            default=sorted(rows["status"].unique().tolist()),
            key=f"{key_prefix}_status",
        )
    with c2:
        category_filter = st.multiselect(
            "平台类别",
            sorted(rows["category"].unique().tolist()),
            default=sorted(rows["category"].unique().tolist()),
            key=f"{key_prefix}_category",
        )
    with c3:
        search_text = st.text_input("搜索平台 / 页面结构 / 训练能力", value="", key=f"{key_prefix}_search")
    view = rows[rows["status"].isin(status_filter) & rows["category"].isin(category_filter)].copy()
    if search_text.strip():
        pattern = re.escape(search_text.strip())
        mask = view.astype(str).apply(lambda col: col.str.contains(pattern, case=False, regex=True, na=False)).any(axis=1)
        view = view[mask]
    visible_cols = [
        "platform",
        "status",
        "category",
        "repo",
        "stars",
        "latest_update",
        "activity",
        "page_structure",
        "core_capability",
        "ui_pattern",
        "visual_style",
        "deepuplift_absorption",
        "streamlit_direct",
        "future_react_fit",
        "dependency_risk",
        "migration_cost",
        "interview_line",
        "github_url",
        "official_url",
    ]
    st.dataframe(view[visible_cols], width="stretch", hide_index=True)

    matrix = pd.DataFrame(ml_training_inspiration_matrix_rows())
    if not matrix.empty:
        st.subheader("Training Platform Pattern Adoption")
        st.dataframe(
            matrix[["pattern", "borrowed_from", "absorbed_now", "deepuplift_value", "future_react", "status"]],
            width="stretch",
            hide_index=True,
        )
    st.download_button(
        "下载训练平台审计 CSV",
        data=dataframe_csv_bytes(rows),
        file_name="deepuplift_ml_training_platform_audit.csv",
        mime="text/csv",
        key=f"{key_prefix}_download",
    )


def render_training_platform_workbench_overview(context: dict[str, str]) -> None:
    st.subheader("ML Training Workbench Overview")
    st.caption("参考 MLflow / W&B / ClearML / DVC / Dagster 的训练平台骨架，把当前 uplift 项目状态压缩成 project cockpit。")
    rows = training_workbench_stage_rows(context)
    cards = []
    for row in rows:
        status = str(row["status"]).lower().replace(" ", "-")
        cards.append(
            "<div class='du-stage-card'>"
            f"<div class='du-stage-name'>{html.escape(row['stage'])}</div>"
            f"<div class='du-stage-status is-{html.escape(status)}'>{html.escape(row['status'])}</div>"
            f"<div class='du-stage-signal'>{html.escape(row['signal'])}</div>"
            f"<div class='du-stage-meta'>{html.escape(row['deepuplift_role'])}</div>"
            "</div>"
        )
    st.markdown(f"<div class='du-workbench-rail'>{''.join(cards)}</div>", unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def render_uplift_application_expansion_matrix(key_prefix: str = "uplift_app") -> None:
    st.subheader("Uplift Application Expansion Matrix")
    st.caption("把开源训练平台的应用目录/registry 思路转成 uplift 应用矩阵：每个业务场景都有 treatment、决策单元、policy metric 和 evidence gate。")
    rows = pd.DataFrame(uplift_application_expansion_rows())
    if rows.empty:
        st.info("No uplift application expansion rows found.")
        return
    status_options = sorted(rows["status"].unique().tolist())
    status_filter = st.multiselect(
        "应用状态",
        status_options,
        default=status_options,
        key=f"{key_prefix}_status",
    )
    view = rows[rows["status"].isin(status_filter)].copy()
    st.dataframe(view, width="stretch", hide_index=True)
    featured = view.head(4).to_dict(orient="records")
    cards = []
    for row in featured:
        cards.append(
            "<div class='du-decision-card'>"
            f"<div class='du-decision-kicker'>{html.escape(row['status'])}</div>"
            f"<div class='du-decision-title'>{html.escape(row['application'])}</div>"
            f"<div class='du-decision-meta'>{html.escape(row['policy_metric'])}</div>"
            f"<div class='du-ux-proof'>{html.escape(row['evidence_gate'])}</div>"
            "</div>"
        )
    if cards:
        st.markdown(f"<div class='du-decision-grid'>{''.join(cards)}</div>", unsafe_allow_html=True)
    st.download_button(
        "下载 uplift 应用扩展矩阵 CSV",
        data=dataframe_csv_bytes(rows),
        file_name="deepuplift_application_expansion_matrix.csv",
        mime="text/csv",
        key=f"{key_prefix}_download",
    )


def render_public_product_inspiration_matrix(key_prefix: str = "public_product") -> None:
    st.subheader("ByteDance-style Public Product Inspiration Matrix")
    st.caption(
        "只参考火山引擎 / 字节公开可访问的官方文档与公开文章，吸收信息架构和工作台模式，"
        "不复制私有 UI、品牌或内部交互。"
    )
    rows = pd.DataFrame(public_product_audit_rows())
    counts = public_product_counts()
    render_stat_grid(
        [
            ("Public Products", f"{counts['total']:,}", False),
            ("Ready Patterns", f"{counts['ready']:,}", False),
            ("Inspired", f"{counts['inspired']:,}", False),
            ("Future React", f"{counts['future_react']:,}", False),
        ],
        columns=4,
    )
    st.markdown(
        "<div class='du-token-strip'>"
        "<span class='du-token'>Data Governance</span>"
        "<span class='du-token'>Growth Metrics</span>"
        "<span class='du-token'>A/B Testing</span>"
        "<span class='du-token'>Realtime Analytics</span>"
        "<span class='du-token'>Agent-to-Action</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    if rows.empty:
        st.info("No public product inspiration rows found.")
        return
    c1, c2, c3 = st.columns([0.22, 0.30, 0.48])
    with c1:
        status_filter = st.multiselect(
            "吸收状态",
            sorted(rows["status"].unique().tolist()),
            default=sorted(rows["status"].unique().tolist()),
            key=f"{key_prefix}_status",
        )
    with c2:
        area_filter = st.multiselect(
            "产品方向",
            sorted(rows["area"].unique().tolist()),
            default=sorted(rows["area"].unique().tolist()),
            key=f"{key_prefix}_area",
        )
    with c3:
        search_text = st.text_input("搜索产品 / pattern / 页面结构", value="", key=f"{key_prefix}_search")
    view = rows[rows["status"].isin(status_filter) & rows["area"].isin(area_filter)].copy()
    if search_text.strip():
        pattern = re.escape(search_text.strip())
        mask = view.astype(str).apply(lambda col: col.str.contains(pattern, case=False, regex=True, na=False)).any(axis=1)
        view = view[mask]
    visible_cols = [
        "product",
        "status",
        "area",
        "stars_or_scale",
        "latest_update",
        "activity",
        "page_structure",
        "core_capability",
        "ui_pattern",
        "visual_pattern",
        "deepuplift_absorption",
        "streamlit_direct",
        "future_react_fit",
        "dependency_risk",
        "migration_cost",
        "interview_line",
        "official_url",
    ]
    st.dataframe(view[visible_cols], width="stretch", hide_index=True)

    matrix = pd.DataFrame(public_product_inspiration_rows())
    if not matrix.empty:
        st.subheader("Public Product Pattern Adoption")
        st.dataframe(
            matrix[["pattern", "borrowed_from", "absorbed_now", "deepuplift_value", "future_react", "status"]],
            width="stretch",
            hide_index=True,
        )
    st.download_button(
        "下载公开产品灵感审计 CSV",
        data=dataframe_csv_bytes(rows),
        file_name="deepuplift_public_product_inspiration.csv",
        mime="text/csv",
        key=f"{key_prefix}_download",
    )


def render_growth_decision_control_plane() -> None:
    st.subheader("Growth Decision Control Plane")
    st.caption(
        "参考 DataLeap / DataFinder / DataTester / ByteHouse / Data Agent 的公开产品结构，"
        "把 uplift 项目从模型页提升成增长决策控制台。"
    )
    rows = growth_control_plane_rows()
    cards = []
    for row in rows:
        status = str(row["status"]).lower().replace(" ", "-")
        cards.append(
            "<div class='du-control-card'>"
            f"<div class='du-control-kicker'>{html.escape(row['stage'])}</div>"
            f"<div class='du-control-signal'>{html.escape(row['signal'])}</div>"
            f"<div class='du-control-meta'>{html.escape(row['deepuplift_gate'])}</div>"
            f"<div class='du-control-status is-{html.escape(status)}'>{html.escape(row['status'])}</div>"
            "</div>"
        )
    st.markdown(f"<div class='du-control-plane'>{''.join(cards)}</div>", unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def render_launch_guardrail_board() -> None:
    st.subheader("Launch Guardrail Board")
    st.caption("把 A/B 实验准备、增长指标、OPE 支持度和 evidence pack 合并成上线前检查，而不是只看离线曲线。")
    rows = pd.DataFrame(launch_guardrail_rows())
    st.dataframe(rows, width="stretch", hide_index=True)
    featured = rows.head(4).to_dict(orient="records")
    cards = []
    for row in featured:
        status = str(row["status"]).lower().replace(" ", "-")
        cards.append(
            "<div class='du-decision-card'>"
            f"<div class='du-decision-kicker'>{html.escape(row['gate'])}</div>"
            f"<div class='du-decision-title'>{html.escape(row['check'])}</div>"
            f"<div class='du-decision-meta'>{html.escape(row['why'])}</div>"
            f"<div class='du-control-status is-{html.escape(status)}'>{html.escape(row['status'])}</div>"
            "</div>"
        )
    st.markdown(f"<div class='du-decision-grid'>{''.join(cards)}</div>", unsafe_allow_html=True)


def render_visual_upgrade_story() -> None:
    st.subheader("Visual Upgrade Before / After")
    rows = visual_upgrade_rows()
    cards = []
    for row in rows:
        cards.append(
            "<div class='du-ux-card'>"
            f"<div class='du-ux-kicker'>{html.escape(row['area'])}</div>"
            f"<div class='du-ux-title'>Before</div><p>{html.escape(row['before'])}</p>"
            f"<div class='du-ux-title du-ux-title-after'>After</div><p>{html.escape(row['after'])}</p>"
            f"<div class='du-ux-proof'>{html.escape(row['proof'])}</div>"
            "</div>"
        )
    st.markdown(f"<div class='du-ux-grid'>{''.join(cards)}</div>", unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def render_visual_regression_evidence(screenshots: dict | None = None) -> None:
    st.subheader("UI Screenshot Gallery / Visual Regression Evidence")
    st.caption("截图不是装饰材料，而是证明 Story、Models、Evidence、Agent 等页面在视觉升级后仍能打开、渲染且非空。")
    st.dataframe(pd.DataFrame(visual_regression_gate_rows()), width="stretch", hide_index=True)
    screenshot_path = Path(str((screenshots or {}).get("path", "")))
    if not screenshot_path.is_dir():
        artifacts = latest_report_artifacts()
        screenshot_path = Path(str((artifacts.get("ui_screenshots") or {}).get("path", "")))
    if not screenshot_path.is_dir():
        st.info("No screenshot directory found yet. Run `scripts/smoke_ui_screenshots.py` after starting Streamlit.")
        return
    shots = sorted(screenshot_path.glob("*.png"))
    shot_rows = [{"screenshot": item.name, "path": str(item), "bytes": item.stat().st_size} for item in shots]
    st.dataframe(pd.DataFrame(shot_rows), width="stretch", hide_index=True)
    preferred = ["workflow.png", "scenario.png", "models.png", "model_deconstruction.png", "evidence.png", "agent.png", "story.png", "policy.png"]
    ordered = [item for name in preferred for item in shots if item.name == name]
    if not ordered:
        ordered = shots[:6]
    cols = st.columns(2)
    for index, item in enumerate(ordered[:6]):
        with cols[index % 2]:
            st.image(str(item), caption=f"{item.name} · {item.stat().st_size:,} bytes", width="stretch")


def render_agent_prompt_group_panel() -> None:
    st.subheader("Quick Prompt Groups")
    st.caption("把 Agent 入口从按钮墙整理成面试工作台：模型选择、评估策略、工业场景、LLM/OPE、前端设计各有明确问题组。")
    model_deconstruction_group = {
        "group": "模型源码拆解",
        "intent": "从 0 到 1 讲清模型源码、公式、训练评估和 license 风险",
        "prompts": [
            "从源码拆解 TLearnerGBM 的 fit 和 predict 调用链",
            "DRLearnerGBM 的 pseudo outcome 公式和源码在哪里？",
            "CFRNet 的 loss 怎么从 TARNet 推出来？",
            "DragonNet 的 propensity head 和 targeted regularization 源码怎么讲？",
            "EFIN 官方代码 license 风险和 DeepUplift 借鉴策略是什么？",
            "MultiDRLearnerGBM 如何把二元 uplift 扩展成多动作推荐？",
        ],
    }
    groups = [model_deconstruction_group] + agent_prompt_groups() + ml_training_prompt_groups() + public_product_prompt_groups()
    group_cards = []
    for group in groups:
        prompts = group.get("prompts") or []
        prompt_html = "".join(f"<li>{html.escape(str(prompt))}</li>" for prompt in prompts)
        group_cards.append(
            "<div class='du-prompt-card'>"
            f"<div class='du-backend-kicker'>{html.escape(str(group.get('group', '')))}</div>"
            f"<div class='du-backend-meta'>{html.escape(str(group.get('intent', '')))}</div>"
            f"<ul>{prompt_html}</ul>"
            "</div>"
        )
    st.markdown(f"<div class='du-prompt-grid'>{''.join(group_cards)}</div>", unsafe_allow_html=True)


def render_workflow_page(
    selected_dataset: dict | None,
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    feature_cols: list[str],
    task: str,
    model_name: str,
    contact_cost: float,
    conversion_value: float,
) -> None:
    st.subheader("Industrial Uplift Workflow")
    dataset_label = selected_dataset.get("name") if selected_dataset else "Uploaded / current dataset"
    diagnostics = diagnose_uplift_data(df, treatment_col, outcome_col, feature_cols)
    design = diagnostics.get("design") or {}
    warnings = diagnostics.get("warnings") or []
    overlap = diagnostics.get("overlap") or {}
    compare_ranking = st.session_state.get("compare_ranking")
    result = st.session_state.get("uplift_result")
    scoring_result = st.session_state.get("scoring_result")

    if result is None:
        next_step = "Train one baseline model, then run Compare with 2-4 candidates."
    elif compare_ranking is None or getattr(compare_ranking, "empty", True):
        next_step = "Run Compare to avoid choosing from a single model."
    elif scoring_result is None:
        next_step = "Use Predict to score a new audience and export Top-K lists."
    else:
        next_step = "Use Policy and Evidence to lock the threshold, ROI and reproducibility pack."

    render_stat_grid(
        [
            ("Business Problem", dataset_label, True),
            ("Treatment Design", design.get("treatment_type", "unknown"), True),
            ("Outcome", f"{outcome_col} · {design.get('outcome_type', task)}", True),
            ("Cost / Value", f"{float(contact_cost):.4f} / {float(conversion_value):.4f}", True),
        ],
        columns=4,
    )
    render_stat_grid(
        [
            ("Rows", f"{len(df):,}", False),
            ("Features", f"{len(feature_cols):,}", False),
            ("Current Model", model_name, True),
            ("Next Action", next_step, True),
        ],
        columns=4,
    )
    render_growth_decision_control_plane()
    render_training_platform_workbench_overview(
        {
            "dataset": dataset_label,
            "model": model_name,
            "model_counts": f"{len(available_models()):,} registered / {len(available_models(only_available=True)):,} ready",
            "compare_ready": "yes" if compare_ranking is not None and not compare_ranking.empty else "no",
            "eval_status": "ready" if result is not None else "pending",
        }
    )
    artifacts = latest_report_artifacts()
    render_optimization_closure_dashboard(key_prefix="workflow_closure", compact=True, artifacts=artifacts)
    render_repo_change_intake_dashboard(key_prefix="workflow_repo_change_intake", compact=True, artifacts=artifacts)
    evidence_rows = [
        {
            "evidence_gate": "data loaded",
            "status": "pass" if len(df) > 0 else "blocker",
            "where_to_check": "Data / Diagnostics",
            "artifact_or_signal": f"{len(df):,} rows, {len(feature_cols):,} features",
        },
        {
            "evidence_gate": "diagnostics available",
            "status": "pass" if diagnostics else "blocker",
            "where_to_check": "Diagnostics",
            "artifact_or_signal": f"{len(warnings)} warning(s)",
        },
        {
            "evidence_gate": "model catalog audited",
            "status": "pass" if (artifacts.get("model_audit") or {}).get("path") else "warn",
            "where_to_check": "Models / Evidence",
            "artifact_or_signal": (artifacts.get("model_audit") or {}).get("path", "run audit_model_catalog"),
        },
        {
            "evidence_gate": "UI screenshots captured",
            "status": "pass" if (artifacts.get("ui_screenshots") or {}).get("files", 0) else "warn",
            "where_to_check": "Evidence",
            "artifact_or_signal": (artifacts.get("ui_screenshots") or {}).get("path", "run smoke_ui_screenshots"),
        },
        {
            "evidence_gate": "interview evidence ZIP",
            "status": "pass" if Path((artifacts.get("interview_zip") or {}).get("path", "")).is_file() else "warn",
            "where_to_check": "Evidence / Story",
            "artifact_or_signal": (artifacts.get("interview_zip") or {}).get("path", "run build_interview_pack_zip"),
        },
    ]
    st.subheader("Current Session Evidence Status")
    st.dataframe(pd.DataFrame(evidence_rows), width="stretch", hide_index=True)

    st.subheader("Frontier Demo Presets")
    st.caption("Use the sidebar `Demo preset` selector to jump directly into interview-friendly industrial scenarios.")
    demo_rows = [
        {
            "preset": "Coupon ROI",
            "dataset_id": "synthetic_coupon_profit_uplift_8k",
            "scenario": "coupon / subsidy / incremental profit",
            "what_to_show": "Policy -> Industrial Scenario Dataset ROI Benchmarks; Evidence -> Scenario Training Smoke",
        },
        {
            "preset": "Baidu Waimai red packet",
            "dataset_id": "synthetic_baidu_waimai_red_packet_8k",
            "scenario": "food delivery / red packet / GMV and margin uplift",
            "what_to_show": "Data -> red-packet schema; Policy -> coupon ROI; Agent -> 天降红包 interview answer",
        },
        {
            "preset": "Didi passenger subsidy",
            "dataset_id": "synthetic_didi_passenger_subsidy_8k",
            "scenario": "C-side ride subsidy / completion uplift / retention",
            "what_to_show": "Models -> scenario recommendations; Policy -> subsidy ROI; Agent -> C/B subsidy differences",
        },
        {
            "preset": "Didi driver supply",
            "dataset_id": "synthetic_didi_driver_supply_subsidy_8k",
            "scenario": "B-side supply incentive / wait-time value / spillover risk",
            "what_to_show": "Policy -> marketplace guardrails; Evidence -> scenario training smoke",
        },
        {
            "preset": "Didi budget allocation",
            "dataset_id": "synthetic_didi_budget_allocation_5k",
            "scenario": "city-time budget split / marginal ROI / constrained allocation",
            "what_to_show": "Policy -> budget value; Agent -> why not sort by historical ROI",
        },
        {
            "preset": "Shopee ads iROAS",
            "dataset_id": "synthetic_shopee_ads_full_funnel_8k",
            "scenario": "ads targeting / full funnel / incremental GMV / iROAS",
            "what_to_show": "Evaluation -> full-funnel metrics; Policy -> iROAS; Agent -> pCTR vs uplift",
        },
        {
            "preset": "ECUP full-funnel",
            "dataset_id": "synthetic_ads_full_funnel_ecup_7k",
            "scenario": "ads / recommendation / coupon impression-click-conversion",
            "what_to_show": "Evaluation -> Full-Funnel ECUP Metrics; Compare/History -> full_funnel_top10_conversion_uplift",
        },
        {
            "preset": "Delayed feedback",
            "dataset_id": "synthetic_growth_delayed_feedback_7k",
            "scenario": "growth / CRM / ads conversion windows",
            "what_to_show": "Evaluation -> Delayed Feedback Uplift Metrics; Compare/History -> delayed_d30_top10_uplift",
        },
        {
            "preset": "Recommendation intervention",
            "dataset_id": "synthetic_recommendation_intervention_7k",
            "scenario": "feed / creative / request uplift",
            "what_to_show": "Models -> Scenario Data-To-Model Matrix; Evidence -> Scenario Dataset Smoke",
        },
        {
            "preset": "Marketplace subsidy",
            "dataset_id": "synthetic_marketplace_subsidy_uplift_7k",
            "scenario": "supply-demand subsidy / spillover risk",
            "what_to_show": "Policy -> Scenario ROI Lab; Agent -> marketplace interference answer",
        },
        {
            "preset": "LLM routing",
            "dataset_id": "synthetic_llm_routing_uplift_8k",
            "scenario": "strong-model escalation / cost-quality policy",
            "what_to_show": "Policy -> LLM Routing ROI; Knowledge/Agent -> Uplift + LLM interview Q&A",
        },
        {
            "preset": "LLM multi-action routing",
            "dataset_id": "synthetic_llm_multi_action_routing_9k",
            "scenario": "strong/RAG/tool/human escalation / evidence-risk policy",
            "what_to_show": "Policy -> multi-action router; Agent -> LLM routing 多动作面试回答",
        },
        {
            "preset": "Game ops gift",
            "dataset_id": "synthetic_game_ops_gift_uplift_6k",
            "scenario": "game retention / gift cost / backlash risk",
            "what_to_show": "Scenario -> new scenario data; Agent -> 游戏礼包 uplift answer",
        },
        {
            "preset": "FinTech credit line",
            "dataset_id": "synthetic_fintech_credit_line_uplift_6k",
            "scenario": "credit offer / risk-adjusted revenue uplift",
            "what_to_show": "Policy -> risk-adjusted policy value; Agent -> 金融授信 uplift answer",
        },
        {
            "preset": "Customer service escalation",
            "dataset_id": "synthetic_customer_service_escalation_uplift_6k",
            "scenario": "human escalation / CSAT uplift / queue cost",
            "what_to_show": "Policy -> escalation ROI; Agent -> 客服升级人工 uplift answer",
        },
        {
            "preset": "Healthcare follow-up",
            "dataset_id": "synthetic_healthcare_followup_uplift_6k",
            "scenario": "patient outreach / adherence uplift / safety guardrail",
            "what_to_show": "Models -> interpretable scenario mapping; Agent -> 医疗随访 uplift answer",
        },
        {
            "preset": "SaaS retention",
            "dataset_id": "synthetic_saas_retention_offer_uplift_6k",
            "scenario": "renewal offer / ARR-weighted policy value",
            "what_to_show": "Policy -> ARR-weighted net value; Agent -> SaaS 续费 uplift answer",
        },
    ]
    st.dataframe(pd.DataFrame(demo_rows), width="stretch", hide_index=True)
    st.subheader("Scenario Data Lab")
    st.caption(
        "Industrial and resume-specific scenarios are backed by runnable synthetic datasets, policy-value columns and smoke checks, "
        "so the demo can move from business design to training/evaluation/ROI instead of only showing model names."
    )
    st.markdown(
        "**Resume scenario rail:** Baidu Waimai red packet -> DiDi passenger subsidy -> DiDi driver supply subsidy -> "
        "DiDi city-time budget allocation -> Shopee ads iROAS."
    )
    scenario_summary = pd.DataFrame(scenario_dataset_summary())
    if not scenario_summary.empty:
        st.dataframe(
            scenario_summary[
                [
                    "scenario",
                    "dataset_id",
                    "status",
                    "rows",
                    "features",
                    "treatment_rate",
                    "outcome_rate",
                    "policy_value_mean",
                    "primary_metrics",
                    "recommended_models",
                ]
            ],
            width="stretch",
            hide_index=True,
        )
    with st.expander("Scenario Cards: business objective, metrics and interview hooks", expanded=False):
        st.dataframe(pd.DataFrame(industrial_scenario_cards()), width="stretch", hide_index=True)
    trace_rows = pd.DataFrame(frontier_demo_trace_rows())
    with st.expander("Demo Preset Source Trace", expanded=False):
        st.caption("Trace each demo from dataset manifest to evaluator contract, saved metrics, ranking columns and smoke validation.")
        st.dataframe(trace_rows, width="stretch", hide_index=True)
        st.download_button(
            "Download demo preset source trace",
            data=dataframe_csv_bytes(trace_rows),
            file_name="deepuplift_demo_preset_source_trace.csv",
            mime="text/csv",
            key="workflow_demo_preset_source_trace_download",
        )

    workflow_rows = [
        {
            "stage": "1. Business Problem / Treatment Design",
            "decision": "Define coupon, ads, push, CRM, recommendation, marketplace or LLM-routing treatment.",
            "current_signal": f"treatment={treatment_col}; outcome={outcome_col}; task={task}",
            "primary_page": "Workflow + sidebar",
            "evidence": "Business value, contact cost, treatment/control contract",
        },
        {
            "stage": "2. Data & Diagnostics",
            "decision": "Check schema, treatment/control ratio, outcome rate, missingness, balance and overlap.",
            "current_signal": f"warnings={len(warnings)}; overlap={overlap.get('verdict', 'computed after binary diagnostics')}",
            "primary_page": "Data / Diagnostics",
            "evidence": "feature balance, propensity overlap, SSB/leakage warnings",
        },
        {
            "stage": "3. Model Selection",
            "decision": "Choose learner family by design: randomized, observational, imbalanced, multi-action or dose.",
            "current_signal": f"selected={model_name}",
            "primary_page": "Models",
            "evidence": "model card, backend readiness, source/code provenance",
        },
        {
            "stage": "4. Training & Compare",
            "decision": "Train baselines and stress-test alternatives with a unified y0/y1/uplift contract.",
            "current_signal": "compare ranking available" if compare_ranking is not None and not compare_ranking.empty else "compare not run",
            "primary_page": "Train / Compare",
            "evidence": "run artifacts, compare ranking, readiness JSON",
        },
        {
            "stage": "5. Evaluation Metrics",
            "decision": "Read ranking quality, Top-K business lift, uncertainty, calibration and support together.",
            "current_signal": "latest run available" if result is not None else "no latest run yet",
            "primary_page": "Evaluation",
            "evidence": "QINI, AUUC, uplift@K, policy value, calibration, bootstrap CI",
        },
        {
            "stage": "6. Policy / ROI Decision",
            "decision": "Turn uplift scores into a cost-aware targeting threshold under budget and fatigue constraints.",
            "current_signal": f"value={float(conversion_value):.4f}; cost={float(contact_cost):.4f}",
            "primary_page": "Policy",
            "evidence": "Top-K threshold, predicted/observed net value, scenario ROI lab",
        },
        {
            "stage": "7. Predict / Scoring",
            "decision": "Score new data with the saved preprocessor and model, then export ranked audiences.",
            "current_signal": "scoring result available" if scoring_result is not None else "no scoring result yet",
            "primary_page": "Predict",
            "evidence": "y0_pred, y1_pred, uplift_score, Top 5/10/20 downloads",
        },
        {
            "stage": "8. Evidence / Reproducibility",
            "decision": "Preserve code paths, model sources, screenshots, metrics and run artifacts for review.",
            "current_signal": "evidence dashboard available",
            "primary_page": "Evidence",
            "evidence": "metrics.json, predictions.csv, screenshots, regression gate, evidence pack",
        },
    ]
    workflow_cards = []
    for row in workflow_rows:
        workflow_cards.append(
            "<div class='du-backend-card'>"
            f"<div class='du-backend-kicker'>{html.escape(row['primary_page'])}</div>"
            f"<div class='du-backend-title'>{html.escape(row['stage'])}</div>"
            f"<div class='du-backend-meta'>{html.escape(row['decision'])}</div>"
            "</div>"
        )
    st.markdown(f"<div class='du-backend-grid'>{''.join(workflow_cards)}</div>", unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(workflow_rows), width="stretch", hide_index=True)

    st.subheader("Industrial Gate Checklist")
    gate_rows = [
        {
            "gate": "Design gate",
            "pass_signal": "treatment/control definition is explicit; outcome and unit of decision are stable",
            "blocker": "mixed treatment semantics, post-treatment leakage, unclear cost/value",
        },
        {
            "gate": "Support gate",
            "pass_signal": "reasonable overlap and no severe treatment/control imbalance in key features",
            "blocker": "weak common support, high SSB proxy, rare treatment cells",
        },
        {
            "gate": "Metric gate",
            "pass_signal": "positive QINI/AUUC, positive Top-K uplift, usable calibration and CI lower bound",
            "blocker": "single metric win, negative Top-K, noisy bootstrap, unstable sensitivity checks",
        },
        {
            "gate": "Business gate",
            "pass_signal": "policy value remains positive after contact cost, budget and fatigue penalties",
            "blocker": "raw uplift improves but net value is negative or threshold violates budget",
        },
        {
            "gate": "Evidence gate",
            "pass_signal": "run artifacts, model card, code/source path and screenshot smoke are available",
            "blocker": "cannot reproduce run or explain estimator/backend origin",
        },
    ]
    st.dataframe(pd.DataFrame(gate_rows), width="stretch", hide_index=True)


def markdown_section(markdown_text: str, heading: str) -> str:
    pattern = rf"(?ms)^## {re.escape(heading)}\n(.*?)(?=^## |\Z)"
    match = re.search(pattern, markdown_text)
    return match.group(1).strip() if match else ""


def render_resume_scenario_workbench() -> None:
    st.subheader("履历业务场景工作台")
    st.caption(
        "这一页把百度外卖、滴滴和 Shopee 的 uplift 项目按工业落地流程串起来：场景设计、模型对比、Top-K ROI、优化路径、模型先进性和证据来源。"
    )
    scenario_cards = pd.DataFrame(resume_scenario_cards_cn())
    compare_rows = pd.DataFrame(resume_scenario_compare_rows())
    best_rows = pd.DataFrame(resume_scenario_best_rows())
    policy_rows = pd.DataFrame(resume_policy_rows_cn())
    audit_rows = pd.DataFrame(frontier_model_audit_rows_cn())
    limitation_rows = pd.DataFrame(model_limitations_rows_cn())

    render_stat_grid(
        [
            ("履历场景", f"{len(scenario_cards):,}", False),
            ("Compare 运行", f"{len(compare_rows):,}", False),
            ("前沿能力", f"{len(audit_rows):,}", False),
            ("受限/待接入", f"{len(limitation_rows):,}", False),
        ],
        columns=4,
    )

    scenario_names = scenario_cards["场景"].tolist() if not scenario_cards.empty else []
    selected_name = st.selectbox("选择具体业务场景", scenario_names, key="resume_scenario_workbench_selector") if scenario_names else ""
    selected = scenario_cards[scenario_cards["场景"] == selected_name].iloc[0].to_dict() if selected_name else {}
    selected_dataset_id = selected.get("dataset_id")

    if selected:
        st.markdown(f"### {selected['场景']}")
        render_stat_grid(
            [
                ("业务问题", selected["业务问题"], True),
                ("Treatment", selected["Treatment"], True),
                ("Outcome", selected["Outcome"], True),
                ("上线目标", selected["上线目标"], True),
            ],
            columns=2,
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Control": selected["Control"],
                        "Cost / Value": selected["Cost / Value"],
                        "推荐模型": selected["推荐模型"],
                        "面试讲法": selected["面试讲法"],
                        "Evidence": selected["证据"],
                    }
                ]
            ),
            width="stretch",
            hide_index=True,
        )

    render_scenario_pitfall_cards("resume_scenario_workbench")

    st.subheader("场景内 Model Compare")
    st.caption("排序逻辑优先看 Top-K policy value / ROI，再看 QINI、AUUC 和 readiness；这能解释为什么离线指标高不一定适合上线。")
    if not compare_rows.empty:
        view = compare_rows[compare_rows["dataset_id"] == selected_dataset_id] if selected_dataset_id else compare_rows
        if "综合分" in view.columns and "模型" in view.columns:
            chart_view = view[["模型", "综合分"]].copy()
            chart_view["综合分"] = pd.to_numeric(chart_view["综合分"], errors="coerce").fillna(0.0)
            st.caption("综合分 = Top-K policy value 35% + ROI/iROAS 20% + QINI 20% + Top10 uplift 15% + readiness 10%。")
            st.bar_chart(chart_view.set_index("模型")["综合分"])
        st.dataframe(view, width="stretch", hide_index=True)
        st.download_button(
            "下载当前场景模型对比 CSV",
            data=dataframe_csv_bytes(view),
            file_name=f"{selected_dataset_id or 'resume_scenario'}_model_compare.csv",
            mime="text/csv",
            key="resume_scenario_compare_download",
        )
    else:
        st.info("还没有场景 compare 结果。运行 `scripts/smoke_industrial_scenario_compare.py --rows 700` 刷新。")

    if not best_rows.empty:
        st.subheader("每个履历场景当前最优可跑模型")
        st.dataframe(best_rows, width="stretch", hide_index=True)

    st.subheader("上线解释 / 为什么暂不建议直接上线")
    rollout_risk = pd.DataFrame(rollout_risk_rows_cn())
    if not rollout_risk.empty:
        st.caption("这张表把 compare 的最佳候选转成上线评审语言：即使 QINI/AUUC 高，也要看 policy value、calibration、overlap 和 readiness。")
        st.dataframe(rollout_risk, width="stretch", hide_index=True)

    st.subheader("Top-K policy value / ROI / 风险")
    if not policy_rows.empty:
        policy_view = policy_rows[policy_rows["场景"] == selected_name] if selected_name else policy_rows
        st.dataframe(policy_view, width="stretch", hide_index=True)
    else:
        st.info("运行 `scripts/smoke_resume_scenario_policy.py` 刷新履历场景 Policy benchmark。")

    with st.expander("滴滴城市/时段预算分配 simulator", expanded=selected_dataset_id == "synthetic_didi_budget_allocation_5k"):
        budget_total = st.number_input(
            "总预算",
            min_value=10_000.0,
            max_value=2_000_000.0,
            value=100_000.0,
            step=10_000.0,
            key="resume_workbench_didi_budget",
        )
        budget_plan = didi_budget_allocation_plan(total_budget=float(budget_total), max_cells=30)
        render_stat_grid(
            [
                ("选择城市/时段", f"{budget_plan['summary']['selected_cells']:,}", False),
                ("已用预算", f"{budget_plan['summary']['used_budget']:.2f}", False),
                ("预期 policy value", f"{budget_plan['summary']['expected_policy_value']:.2f}", False),
                ("平均 spillover", f"{budget_plan['summary']['mean_spillover_risk']:.4f}", False),
            ],
            columns=4,
        )
        allocation = pd.DataFrame(budget_plan.get("allocations") or [])
        if not allocation.empty:
            st.dataframe(allocation, width="stretch", hide_index=True)

    st.subheader("从 0-1 到相对最优的优化路径")
    path_rows = pd.DataFrame(resume_optimization_path_rows(selected_dataset_id))
    st.dataframe(path_rows, width="stretch", hide_index=True)

    st.subheader("当前模型是否业内先进")
    st.caption("结论：P0/P1/P2 覆盖已经接近工业先进工作台，但不是所有论文模型都应该伪装成 ready；受依赖、运行时稳定性和论文代码质量影响的模型必须 guarded 或 backlog。")
    if not audit_rows.empty:
        priority_filter = st.multiselect(
            "模型层级",
            sorted(audit_rows["优先级"].dropna().unique().tolist()),
            default=sorted(audit_rows["优先级"].dropna().unique().tolist()),
            key="resume_frontier_priority_filter",
        )
        audit_view = audit_rows[audit_rows["优先级"].isin(priority_filter)] if priority_filter else audit_rows
        st.dataframe(audit_view, width="stretch", hide_index=True)
        st.download_button(
            "下载前沿模型审计 CSV",
            data=dataframe_csv_bytes(audit_view),
            file_name="deepuplift_frontier_model_audit_cn.csv",
            mime="text/csv",
            key="frontier_model_audit_cn_download",
        )

    st.subheader("模型限制与依赖")
    st.caption("这些能力已经在模型卡或 Evidence 中可见，但因依赖、运行时、Python 版本或插件成熟度限制，默认不作为 fully runnable。")
    if not limitation_rows.empty:
        st.dataframe(limitation_rows, width="stretch", hide_index=True)

    st.subheader("工业扩展场景雷达")
    st.caption("除了履历场景，工作台还沉淀了站内消息、ghost ads、商家补贴、LLM routing、推荐干预和 marketplace subsidy 等可训练仿真场景。")
    extension_view = pd.DataFrame(industrial_extension_summary_rows_cn())
    if not extension_view.empty:
        st.dataframe(extension_view, width="stretch", hide_index=True)

    extension_compare = pd.DataFrame(industrial_extension_compare_rows_cn())
    st.subheader("工业扩展场景 Model Compare")
    st.caption("这些场景来自工业论文/技术博客的启发，但都落成了可生成数据、可训练模型、可评估 policy value 的本地证据。排序同样不只看 QINI，而是把 Top-K 业务价值和 readiness 放在前面。")
    if not extension_compare.empty:
        extension_names = extension_compare["场景"].dropna().unique().tolist()
        selected_extension = st.selectbox(
            "选择工业扩展场景",
            ["全部"] + extension_names,
            key="industrial_extension_compare_selector",
        )
        extension_compare_view = (
            extension_compare if selected_extension == "全部" else extension_compare[extension_compare["场景"] == selected_extension]
        )
        if "综合分" in extension_compare_view.columns and "模型" in extension_compare_view.columns:
            chart_view = extension_compare_view[["场景", "模型", "综合分"]].copy()
            chart_view["标签"] = chart_view["场景"].astype(str) + " / " + chart_view["模型"].astype(str)
            chart_view["综合分"] = pd.to_numeric(chart_view["综合分"], errors="coerce").fillna(0.0)
            st.bar_chart(chart_view.set_index("标签")["综合分"])
        st.dataframe(extension_compare_view, width="stretch", hide_index=True)
        st.download_button(
            "下载工业扩展场景模型对比 CSV",
            data=dataframe_csv_bytes(extension_compare_view),
            file_name="industrial_extension_model_compare_cn.csv",
            mime="text/csv",
            key="industrial_extension_compare_download",
        )
    else:
        st.info("运行 `scripts/smoke_industrial_scenario_compare.py --all --rows 500` 刷新工业扩展场景 compare 证据。")

    st.subheader("LLM 多动作 Routing / OPE 场景证据")
    st.caption("把 cheap / strong / RAG / tool / human review 多动作 routing 当作 uplift policy：先看历史动作覆盖，再用 DM/IPS/SNIPS/DR 做 OPE，最后用漂移诊断决定是否升级到 bandit。")
    llm_ope_view = pd.DataFrame(llm_routing_ope_scenario_rows_cn())
    if not llm_ope_view.empty:
        st.dataframe(llm_ope_view, width="stretch", hide_index=True)
        st.download_button(
            "下载 LLM routing OPE 策略评估 CSV",
            data=dataframe_csv_bytes(llm_ope_view),
            file_name="llm_routing_ope_scenario_rows_cn.csv",
            mime="text/csv",
            key="llm_routing_ope_scenario_download",
        )
    coverage_view = pd.DataFrame(llm_routing_action_coverage_rows_cn())
    drift_view = pd.DataFrame(llm_routing_bandit_drift_rows_cn())
    left, right = st.columns(2)
    with left:
        st.markdown("#### 动作覆盖 / exploration 诊断")
        if not coverage_view.empty:
            st.dataframe(coverage_view, width="stretch", hide_index=True)
    with right:
        st.markdown("#### Bandit drift mock")
        if not drift_view.empty:
            st.dataframe(drift_view, width="stretch", hide_index=True)


def render_project_story() -> None:
    st.subheader("Project Story")
    resume_path = Path("docs/RESUME_DEEPUplift_AGENT.md")
    resume_text = resume_path.read_text(encoding="utf-8") if resume_path.exists() else ""
    catalog = model_catalog_dataframe()
    catalog_audit = model_catalog_audit_payload(catalog)
    ready_count = int((catalog["status"] == "ready").sum()) if not catalog.empty else 0
    artifacts = latest_report_artifacts()
    regression_state = "pass" if artifacts and not catalog_audit["missing_descriptions"] and not catalog_audit["missing_model_cards"] else "warn"

    render_stat_grid(
        [
            ("Positioning", "Causal Decision Workbench", True),
            ("Registered Models", f"{len(catalog):,}", False),
            ("Runnable Models", f"{ready_count:,}", False),
            ("Regression Gate", regression_state, True),
        ]
    )
    st.markdown(
        """
        **DeepUplift Agent** 面向增长、营销、CRM 和推荐场景，把 uplift modeling 从离线脚本升级为可演示、可验证、可导出证据的 causal decision workbench。

        面试时可以按这条主线讲：**为什么不是预测转化率，而是预测增量效果；为什么不是只看 QINI，而是把 overlap、calibration、bootstrap CI、policy value 和 evidence bundle 放进同一个决策闭环。**
        """
    )
    render_optimization_closure_dashboard(key_prefix="story_closure", compact=True, artifacts=artifacts)
    render_visual_upgrade_story()
    st.subheader("UI Inspiration Matrix")
    st.caption("这次视觉升级把 dashboard、agent workbench、ML observability 和 design system 的设计模式压缩成当前 Streamlit 可用的工作台组件。")
    st.dataframe(pd.DataFrame(ui_inspiration_matrix_rows()), width="stretch", hide_index=True)
    st.subheader("ML Training Platform Inspiration Matrix")
    st.caption("新增训练平台视角：把 Experiments、Runs、Model Registry、Artifacts、Lineage 和 Evidence Gates 融入 DeepUplift。")
    st.dataframe(pd.DataFrame(ml_training_inspiration_matrix_rows()), width="stretch", hide_index=True)
    st.subheader("ByteDance-style Public Product Inspiration Matrix")
    st.caption("只采用火山引擎 / 字节公开资料，把数据治理、增长分析、A/B 实验、实时分析和 Data Agent 的产品结构转成 DeepUplift 可讲的工作台模式。")
    st.dataframe(pd.DataFrame(public_product_inspiration_rows()), width="stretch", hide_index=True)
    st.subheader("Model Strategy Interview Card")
    strategy_rows = [
        {
            "layer": "P0 industrial baseline",
            "models": "S/T/X/R/DR learner + LightGBM, DML/DR, CausalForestDML",
            "why_first": "fast, stable, auditable, strong on tabular marketing data, easy to stress-test with overlap/CI/policy value",
            "interview_line": "I do not start from the newest neural model; I first establish a high-quality causal baseline and decision metric gate.",
        },
        {
            "layer": "P1 advanced plugin",
            "models": "UTBoost, EFIN, DESCN/ESX, TARNet/CFRNet/DragonNet",
            "why_first": "use after P0 to test treatment interaction, biased logs or representation learning gains",
            "interview_line": "P1 models must beat or complement P0 under QINI/AUUC/Top-K/policy value, not just look advanced.",
        },
        {
            "layer": "P2 frontier extension",
            "models": "ECUP full-funnel, delayed feedback CFR-DF, continuous treatment, LLM routing, causal bandit",
            "why_first": "activated by business design: funnel logs, delayed labels, treatment dose, budget or cost-quality routing",
            "interview_line": "P2 is not a model bucket; it changes schema, metrics, policy constraints and online experiment design.",
        },
        {
            "layer": "governance gate",
            "models": "registered / ready / guarded / optional",
            "why_first": "keeps broad coverage without overclaiming local runtime readiness",
            "interview_line": "Every optional backend needs source evidence, integration gate and promotion rule before it becomes demo-ready.",
        },
    ]
    st.dataframe(pd.DataFrame(strategy_rows), width="stretch", hide_index=True)
    st.subheader("Deep Uplift Architecture Interview Card")
    deep_counts = deep_design_counts()
    render_stat_grid(
        [
            ("Loss Families", f"{deep_counts.get('losses', 0)}", False),
            ("Architectures", f"{deep_counts.get('architectures', 0)}", False),
            ("Ready Loss/Net", f"{deep_counts.get('ready', 0)}", False),
            ("Loss Library Ready", f"{deep_counts.get('loss_library_ready', 0)}", False),
        ],
        columns=4,
    )
    deep_arch_rows = [
        {
            "interview_topic": "Why BCE/MSE is not enough",
            "technical_answer": "factual loss fits observed arms; uplift needs counterfactual robustness, ranking and policy-value gates",
            "ui_evidence": "Models: Deep Uplift Loss Catalog; Evaluation: Loss-to-Metric Map",
        },
        {
            "interview_topic": "Representation balance",
            "technical_answer": "CFRNet now uses differentiable torch MMD / energy / mean-L2 balance so the IPM term has gradients",
            "ui_evidence": "Models: Network Architecture Catalog; Evidence: CFRNet differentiable balance smoke",
        },
        {
            "interview_topic": "Contrastive uplift",
            "technical_answer": "contrastive positives should come from randomized true uplift or DR/R pseudo-effect buckets, not raw conversion labels",
            "ui_evidence": "Models: Deep Uplift Loss Catalog; docs/UPLIFT_CONTRASTIVE_AND_LLM_UPLIFT.md",
        },
        {
            "interview_topic": "LLM routing",
            "technical_answer": "strong-model escalation is treatment; optimize incremental quality minus model cost and latency",
            "ui_evidence": "Policy: LLM Routing ROI Simulator; Synthetic LLM Routing Uplift 8k",
        },
    ]
    st.dataframe(pd.DataFrame(deep_arch_rows), width="stretch", hide_index=True)

    st.subheader("从问题诊断到修复优化")
    failure_counts = failure_mode_counts()
    render_stat_grid(
        [
            ("模型风险", f"{failure_counts.get('model_failure_modes', 0)}", False),
            ("业务坑位", f"{failure_counts.get('scenario_pitfalls', 0)}", False),
            ("mock 案例", f"{failure_counts.get('diagnostic_cases', 0)}", False),
            ("面试主线", "Failure -> Diagnosis -> Fix", True),
        ],
        columns=4,
    )
    diagnostic_story_rows = pd.DataFrame(diagnostic_case_rows())[
        ["case_name", "problem_type", "misleading_metric", "fix_path", "interview_line"]
    ]
    st.caption("这部分用可生成 mock 数据把 uplift 模型最容易出错的地方讲清楚：不是只说指标，而是展示错误现象、误导指标和修复路径。")
    st.dataframe(diagnostic_story_rows, width="stretch", hide_index=True)

    st.subheader("Industrial Demo Path")
    concise_path_rows = [
        {
            "step": "1. Workflow",
            "what_to_show": "industrial uplift chain and decision gates",
            "interview_line": "I start from treatment design and causal support, not from a model dropdown.",
        },
        {
            "step": "2. Diagnostics",
            "what_to_show": "treatment/control balance, overlap, SSB and leakage warnings",
            "interview_line": "A model is only credible if the treatment/control comparison is credible.",
        },
        {
            "step": "3. Models",
            "what_to_show": "scenario model matrix, ready/guarded status, code provenance and model cards",
            "interview_line": "Model choice is driven by data design, dependency readiness and estimator assumptions.",
        },
        {
            "step": "4. Evaluation",
            "what_to_show": "QINI, AUUC, Top-K, policy value, calibration, bootstrap CI and overlap gates",
            "interview_line": "I use a metric system, not a single leaderboard score.",
        },
        {
            "step": "5. Policy",
            "what_to_show": "cost/value threshold, budget cap, scenario ROI and LLM routing simulator",
            "interview_line": "The model matters only after it becomes a constrained business decision.",
        },
        {
            "step": "6. Evidence",
            "what_to_show": "artifacts, screenshots, audits, source quality and evidence ZIP",
            "interview_line": "Every resume claim can be traced to code, metrics, docs and screenshots.",
        },
    ]
    st.dataframe(pd.DataFrame(concise_path_rows), width="stretch", hide_index=True)

    st.subheader("Resume / Interview Evidence")
    evidence_short_rows = [
        {
            "asset": "Architecture and data flow",
            "path": "docs/DEEPUplift_AGENT_ARCHITECTURE.md",
            "use": "explain how UI, trainer, evaluator, policy and artifacts fit together",
        },
        {
            "asset": "Interview difficulty deep dive",
            "path": "docs/INTERVIEW_DEEPUplift_AGENT_DEEP_DIVE.md",
            "use": "answer hard questions about causal assumptions, metrics and registry design",
        },
        {
            "asset": "Model selection playbook",
            "path": "docs/UPLIFT_MODEL_SELECTION_PLAYBOOK.md",
            "use": "map business scenario and treatment type to estimator family",
        },
        {
            "asset": "Evidence pack",
            "path": "reports/deepuplift_agent_interview_pack_latest.zip",
            "use": "handoff bundle for review or interview demo proof",
        },
    ]
    st.dataframe(pd.DataFrame(evidence_short_rows), width="stretch", hide_index=True)
    short_downloads = [
        ("Resume project brief", Path("docs/RESUME_DEEPUplift_AGENT.md")),
        ("Architecture", Path("docs/DEEPUplift_AGENT_ARCHITECTURE.md")),
        ("Interview deep dive", Path("docs/INTERVIEW_DEEPUplift_AGENT_DEEP_DIVE.md")),
        ("Evidence ZIP", Path("reports/deepuplift_agent_interview_pack_latest.zip")),
    ]
    cols = st.columns(4)
    for index, (label, path) in enumerate(short_downloads):
        with cols[index % len(cols)]:
            if path.exists() and path.is_file():
                st.download_button(
                    f"Download {label}",
                    data=path.read_bytes(),
                    file_name=path.name,
                    mime=artifact_mime(path),
                    key=f"story_concise_download_{path.name}",
                )

    st.subheader("Production Migration Snapshot")
    production_rows = [
        {
            "layer": "UI",
            "current_mvp": "Streamlit workbench",
            "production_path": "React + typed API client",
            "interview_point": "MVP optimizes iteration speed; production separates long-running jobs and review surfaces.",
        },
        {
            "layer": "API",
            "current_mvp": "in-process trainer calls",
            "production_path": "FastAPI endpoints for datasets, runs, metrics, predictions and artifacts",
            "interview_point": "Training should not block web requests; API contracts mirror current run artifacts.",
        },
        {
            "layer": "Worker",
            "current_mvp": "local deterministic training",
            "production_path": "queue-backed workers with run status, retry and resource limits",
            "interview_point": "Compare jobs become asynchronous and reproducible, not browser-session dependent.",
        },
        {
            "layer": "Governance",
            "current_mvp": "Evidence tab + ZIP + screenshots",
            "production_path": "artifact registry, regression gate, model approval and online experiment handoff",
            "interview_point": "Causal decisions need auditability because offline uplift is only pre-launch evidence.",
        },
    ]
    st.dataframe(pd.DataFrame(production_rows), width="stretch", hide_index=True)

    if not st.checkbox("Show full evidence library", value=False, key="story_show_full_evidence_library"):
        return

    st.subheader("Interview Runbook")
    runbook_rows = [
        {"step": "1", "page": "Story", "show": "Project Story / Industry Mode / Evidence Pack", "message": "这是 causal decision workbench，不是模型脚本。"},
        {"step": "2", "page": "Models", "show": "Scenario Model Matrix / Scenario Tag / Model Card", "message": "模型按业务 treatment design 和 readiness 选择。"},
        {"step": "3", "page": "Agent", "show": "Industrial hard questions", "message": "Agent 能解释发券 ROI、广告增量、偏差、疲劳和 marketplace 干扰。"},
        {"step": "4", "page": "Policy", "show": "Scenario ROI Helper / constrained Top-K", "message": "离线 uplift 转成成本感知投放阈值。"},
        {"step": "5", "page": "Evidence", "show": "ZIP / audits / screenshots", "message": "项目可复现，并有回归守卫。"},
    ]
    st.dataframe(pd.DataFrame(runbook_rows), width="stretch", hide_index=True)
    st.subheader("Industry Mode")
    st.markdown("**Coupon / Subsidy · Ads / Bidding · Growth / CRM · Marketplace Subsidy**")
    industry_mode_rows = [
        {
            "scenario": "Coupon / Subsidy",
            "interview_point": "发券不是找高 CVR 用户，而是找增量毛利为正且不会被补贴浪费的人。",
            "models": "T/X -> DR/R -> Multi-treatment -> Dose-response",
            "proof": "Policy ROI templates + coupon case studies",
        },
        {
            "scenario": "Ads / Bidding",
            "interview_point": "pCTR/pCVR 是相关性预测，incrementality 要用 holdout/geo/ghost-ad 证明曝光增量。",
            "models": "DR/R, causal forest, geo/holdout design",
            "proof": "Tencent Ads + Google geo experiments",
        },
        {
            "scenario": "Growth / CRM",
            "interview_point": "触达要同时看激活增量、退订/疲劳和长期留存，不能只看当天打开。",
            "models": "DR/R, delayed-feedback uplift, multi-channel treatment",
            "proof": "Industry Q&A + online incrementality playbook",
        },
        {
            "scenario": "Marketplace Subsidy",
            "interview_point": "平台补贴会影响供需两侧和竞争关系，需要考虑 interference、spillover 和连续 treatment。",
            "models": "Multi-treatment, dose-response, causal forest",
            "proof": "DiDi + Amazon/JD marketplace experiment notes",
        },
        {
            "scenario": "LLM Routing",
            "interview_point": "强模型调用不是越多越好，要估计相对便宜模型的增量质量是否覆盖成本、延迟和预算风险。",
            "models": "DR/R, calibrated uplift, contextual bandit",
            "proof": "LLM Bandit + Microsoft cost-aware routing + Policy simulator",
        },
    ]
    st.dataframe(pd.DataFrame(industry_mode_rows), width="stretch", hide_index=True)
    st.subheader("Fresh Industrial Additions")
    fresh_ids = [
        "SPOTIFY_IN_APP_MESSAGING_UPLIFT",
        "SPOTIFY_PERSONALIZATION_EXPERIMENT_SEPARATION",
        "AIRBNB_INCREMENTAL_LTV",
        "ALIBABA_ABTEST_HETEROGENEITY_XLEARNER",
        "ALIBABA_EDGEREC_REQUEST_UPLIFT",
        "HIERARCHICAL_CAUSAL_LIFT_JOURNEYS",
        "CAUSAL_COPILOT_LLM_AGENT",
        "CATE_B_LLM_COPILOT",
        "LLM_BANDIT_ROUTING",
        "MICROSOFT_COST_AWARE_LLM_SELECTION",
        "FRUGALGPT_LLM_CASCADE",
        "ROUTELLM_OPEN_SOURCE_ROUTER",
        "BEST_ROUTE_TEST_TIME_COMPUTE",
        "UNIVERSAL_MODEL_ROUTING",
    ]
    fresh_refs = pd.DataFrame([ref for ref in industry_references() if ref.get("id") in fresh_ids])
    if not fresh_refs.empty:
        st.markdown(
            "这一轮新增的工业证据已经进入可视化：Spotify 站内消息 CATE/uplift、Spotify 个性化与实验平台分层、Airbnb incremental LTV、阿里 X-Learner/EdgeRec、多 CRM journey 重叠增量，以及 LLM causal copilot / LLM routing / RouteLLM / BEST-Route。"
        )
        st.dataframe(
            fresh_refs[["company", "scenario", "source_type", "title", "lesson", "interview_hook", "url"]],
            width="stretch",
            hide_index=True,
        )
    st.subheader("China / Global Industry Mini-Board")
    mini_board_rows = [
        {
            "board": "China growth and marketing",
            "companies": "Tencent Ads, Meituan, Alibaba, DiDi, ByteDance, Kuaishou, JD.com",
            "what_to_say": "国内业务重点讲发券/补贴/广告/推荐干预，突出随机实验、X-Learner/DR/R、全链路 ROI、延迟反馈和多 treatment。",
            "ui_proof": "Story Fresh Additions + Knowledge Industrial Knowledge Map + Models Scenario Matrix",
        },
        {
            "board": "Global experimentation",
            "companies": "Uber, Google, Amazon, Airbnb, Spotify",
            "what_to_say": "海外材料重点讲 A/B holdout、geo/marketplace experiment、incremental LTV、个性化和实验平台分层。",
            "ui_proof": "Policy Scenario ROI Lab + Knowledge Source Adoption Drilldown + Evidence Pack",
        },
        {
            "board": "Research frontier",
            "companies": "PMLR, arXiv, journal/conference papers",
            "what_to_say": "前沿问题包括 delayed feedback、multi-treatment、continuous treatment、combinatorial treatment、budget-constrained causal bandit 和 journey overlap。",
            "ui_proof": "Models Scenario Tag + Agent industrial hard questions + docs/UPLIFT_MODEL_SELECTION_PLAYBOOK.md",
        },
    ]
    st.dataframe(pd.DataFrame(mini_board_rows), width="stretch", hide_index=True)
    st.subheader("LLM + Uplift Frontier")
    st.markdown(
        "Trainable LLM routing dataset: `Synthetic LLM Routing Uplift 8k` lets the demo train a real uplift model where strong-model escalation is the treatment."
    )
    llm_frontier_rows = [
        {
            "frontier": "LLM as causal copilot",
            "causal_question": "如何把业务问题转成 treatment / outcome / confounders / estimator？",
            "agent_capability": "Source-grounded workflow orchestration, evidence card, regression gate",
            "interview_line": "LLM 负责因果工作流编排，不替代 DR/R/DML/causal forest 等估计器。",
        },
        {
            "frontier": "Text / creative treatment",
            "causal_question": "广告文案、push 内容、offer 描述是否带来增量？",
            "agent_capability": "Treat text/creative embedding as structured treatment and route to multi-treatment uplift",
            "interview_line": "LLM 生成内容后仍需随机实验、holdout 和 text-treatment CATE。",
        },
        {
            "frontier": "LLM routing uplift",
            "causal_question": "哪些请求值得从便宜模型升级到强模型？",
            "agent_capability": "Policy page cost-quality simulator plus scenario model matrix",
            "interview_line": "强模型调用是 treatment，核心是增量质量是否超过增量成本。",
        },
        {
            "frontier": "Trainable LLM routing dataset",
            "causal_question": "如何把 LLM routing 从概念变成可训练、可评估、可演示的 uplift 闭环？",
            "agent_capability": "Synthetic LLM Routing Uplift 8k with true_uplift, cost, latency and oracle policy value",
            "interview_line": "我为 LLM routing 做了专门的合成训练集，能现场演示 strong-model escalation 的 CATE/Top-K/policy value。",
        },
        {
            "frontier": "Multi-action routing + OPE",
            "causal_question": "如何评估 strong/RAG/tool/人工审核多动作 router，且不上来就全量试错？",
            "agent_capability": "Synthetic LLM Multi-Action Routing 9k + DM/IPS/SNIPS/DR OPE + action coverage + drift lab",
            "interview_line": "先用随机探索和 DR OPE 验证固定 policy，再在价格、质量或 action 池漂移时升级到 bandit。",
        },
    ]
    st.dataframe(pd.DataFrame(llm_frontier_rows), width="stretch", hide_index=True)
    st.subheader("Fixed Threshold -> OPE -> Bandit Evolution")
    bandit_story_rows = [
        {
            "stage": "V1 固定 uplift 阈值",
            "when": "第一版上线、业务要可解释、流量和成本相对稳定",
            "evidence": "Policy value / Top-K / calibration / bootstrap / overlap",
            "risk": "价格、质量、流量或 action 池漂移后阈值会变旧",
        },
        {
            "stage": "V2 随机探索 + OPE",
            "when": "要离线评估一个新 router / subsidy / ad policy",
            "evidence": "logged action, propensity, reward, coverage, effective sample size, DM/IPS/SNIPS/DR",
            "risk": "coverage 低时 OPE 是外推，需要先补探索桶",
        },
        {
            "stage": "V3 Budget pacing",
            "when": "预算、强模型成本、人工审核队列或广告 spend 有硬约束",
            "evidence": "maximize gain under spend/capacity constraint",
            "risk": "平均收益为正也可能提前烧完预算或过载稀缺 action",
        },
        {
            "stage": "V4 Contextual bandit",
            "when": "模型价格、质量、prompt 分布或业务价值持续漂移",
            "evidence": "online holdout, guardrail, OPE sanity, drift monitor",
            "risk": "没有稳定 logging/propensity/guardrail 时，bandit 只会快速优化噪声",
        },
    ]
    st.dataframe(pd.DataFrame(bandit_story_rows), width="stretch", hide_index=True)
    st.subheader("LLM Interview Q&A Matrix")
    st.markdown(
        "这块用于面试时展开 `Uplift + LLM`：从 causal copilot、text treatment、LLM routing 到 budget-constrained bandit，每个问题都有工程设计、证据和一句话讲法。"
    )
    st.dataframe(pd.DataFrame(llm_interview_qa()), width="stretch", hide_index=True)

    workflow_rows = [
        {"stage": "1. Diagnose", "what_to_show": "Data / Diagnostics", "proof": "treatment balance, overlap, missingness, leakage checks"},
        {"stage": "2. Recommend", "what_to_show": "Agent / Models", "proof": "knowledge rules, model cards, presets, dependency readiness"},
        {"stage": "3. Compare", "what_to_show": "Compare", "proof": "QINI, AUUC, Top-K uplift, bootstrap CI, readiness score"},
        {"stage": "4. Decide", "what_to_show": "Policy / Predict", "proof": "cost-value simulation, optimal targeting threshold, scored export"},
        {"stage": "5. Govern", "what_to_show": "History / Knowledge", "proof": "run notes, tags, evidence ZIP, regression screenshots"},
    ]
    st.subheader("Demo Narrative")
    st.dataframe(pd.DataFrame(workflow_rows), width="stretch", hide_index=True)
    st.subheader("Architecture Flow")
    st.graphviz_chart(
        """
        digraph {
          graph [rankdir=LR, bgcolor="transparent", pad="0.2", nodesep="0.35", ranksep="0.45"];
          node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12, color="#2dd4bf", fillcolor="#111827", fontcolor="#f8fafc"];
          edge [color="#38bdf8", fontname="Helvetica", fontsize=10, fontcolor="#94a3b8"];
          data [label="Data\\nCSV / samples"];
          diagnostics [label="Diagnostics\\nBalance / Overlap / SSB"];
          agent [label="Agent\\nWorkflow orchestration"];
          registry [label="Model Registry\\nCards / presets / guards"];
          train [label="Train + Compare\\n103 catalog / 73 ready"];
          eval [label="Evaluation\\nQINI / AUUC / Top-K / CI"];
          policy [label="Policy\\nCost-value threshold"];
          evidence [label="Evidence\\nRun notes / ZIP / screenshots"];
          regression [label="Regression Gate\\nsmoke + audits + freshness"];
          data -> diagnostics -> agent -> registry -> train -> eval -> policy -> evidence -> regression;
          diagnostics -> registry [label="recommend"];
          eval -> agent [label="explain"];
          regression -> evidence [label="refresh"];
        }
        """,
        width="stretch",
    )
    st.subheader("Interview Demo Checklist")
    checklist_rows = [
        {"duration": "5 min", "path": "Story -> Diagnostics -> Models -> Policy -> Evidence", "goal": "deliver the core causal decision story quickly"},
        {"duration": "10 min", "path": "Story -> Data -> Diagnostics -> Models -> Agent -> Compare -> Policy", "goal": "show the end-to-end workflow and model governance"},
        {"duration": "20 min", "path": "Story -> Diagnostics -> Models -> Compare -> Predict -> History -> Production", "goal": "deep dive on assumptions, metrics, artifacts, and productionization"},
    ]
    st.dataframe(pd.DataFrame(checklist_rows), width="stretch", hide_index=True)
    st.subheader("Agent Interview Coach")
    st.markdown(
        "Agent 页现在可以直接回答面试追问：为什么不用普通转化率模型、观测数据有偏怎么办、模型 registry 难在哪里、offline 指标怎么上线验证，并支持下载 Agent transcript 作为现场问答证据。"
    )
    st.dataframe(pd.DataFrame(interview_agent_topics()), width="stretch", hide_index=True)
    st.subheader("Industrial Case Studies")
    industry_refs = pd.DataFrame(industry_references())
    if not industry_refs.empty:
        st.dataframe(
            industry_refs[["company", "scenario", "source_type", "title", "lesson", "url"]],
            width="stretch",
            hide_index=True,
        )
    st.subheader("Industry Source Quality Audit")
    st.markdown(
        "工业案例知识库按 `official / peer-reviewed / research preprint / secondary context` 分层审计，"
        "二手材料只作为补充背景，面试主证据优先引用官方文档、论文、开源仓库和实验设计材料。"
    )
    source_audit_path = Path("docs/INDUSTRY_SOURCE_QUALITY_AUDIT.md")
    if source_audit_path.exists():
        source_audit_text = source_audit_path.read_text(encoding="utf-8")
        source_audit_preview = markdown_section(source_audit_text, "Quality Mix")
        if source_audit_preview:
            st.markdown(source_audit_preview)
    st.subheader("Industry Source Adoption Backlog")
    st.markdown(
        "不是所有搜到的大厂文章都会进入核心知识库；backlog 记录 `adopted / partial / backlog`，"
        "用于说明为什么优先采用官方、论文、开源和实验设计证据。"
    )
    source_backlog_path = Path("docs/UPLIFT_INDUSTRY_SOURCE_ADOPTION_BACKLOG.md")
    if source_backlog_path.exists():
        with st.expander("Open source adoption and backlog notes", expanded=False):
            st.markdown(source_backlog_path.read_text(encoding="utf-8"))
    st.subheader("Business Scenarios")
    scenario_df = pd.DataFrame(scenario_model_rows())
    if not scenario_df.empty:
        st.dataframe(scenario_df, width="stretch", hide_index=True)
    lab_df = pd.DataFrame(scenario_dataset_summary())
    if not lab_df.empty:
        st.subheader("Scenario Data + Training Lab")
        st.caption("This is the resume/interview proof that coupon, growth, ads, recommendation, marketplace and LLM-routing stories are runnable workflows, not slideware.")
        st.dataframe(
            lab_df[
                [
                    "scenario",
                    "dataset_id",
                    "status",
                    "rows",
                    "primary_metrics",
                    "policy_objective",
                    "ui_pages",
                ]
            ],
            width="stretch",
            hide_index=True,
        )
    st.subheader("Interview Hard Questions")
    st.markdown("\n".join(f"- {prompt}" for prompt in INDUSTRY_INTERVIEW_PROMPTS))
    industry_docs = [
        ("Industrial case studies", Path("docs/INDUSTRIAL_UPLIFT_CASE_STUDIES.md")),
        ("Business scenarios", Path("docs/UPLIFT_BUSINESS_SCENARIOS.md")),
        ("Industry interview deep dive", Path("docs/UPLIFT_INTERVIEW_INDUSTRY_DEEP_DIVE.md")),
        ("Model selection playbook", Path("docs/UPLIFT_MODEL_SELECTION_PLAYBOOK.md")),
        ("Coupon ads growth playbook", Path("docs/UPLIFT_COUPON_ADS_GROWTH_PLAYBOOK.md")),
        ("LLM causal frontier", Path("docs/UPLIFT_LLM_CAUSAL_FRONTIER.md")),
        ("LLM routing policy", Path("docs/LLM_ROUTING_POLICY_PLAYBOOK.md")),
        ("LLM interview Q&A", Path("docs/UPLIFT_LLM_INTERVIEW_QA.md")),
        ("Online incrementality", Path("docs/UPLIFT_ONLINE_EXPERIMENT_AND_INCREMENTALITY.md")),
        ("Industry references", Path("docs/UPLIFT_INDUSTRY_REFERENCES.md")),
        ("Industry source quality audit", Path("docs/INDUSTRY_SOURCE_QUALITY_AUDIT.md")),
        ("Industry source adoption backlog", Path("docs/UPLIFT_INDUSTRY_SOURCE_ADOPTION_BACKLOG.md")),
        ("Scenario model matrix", Path("docs/UPLIFT_SCENARIO_MODEL_MATRIX.md")),
        ("Scenario model matrix interview Q&A", Path("docs/UPLIFT_SCENARIO_MODEL_MATRIX_INTERVIEW_QA.md")),
        ("Interview runbook", Path("docs/UPLIFT_AGENT_INTERVIEW_RUNBOOK.md")),
    ]
    with st.expander("Industrial Uplift Playbook Downloads", expanded=False):
        for label, path in industry_docs:
            if path.exists():
                text = path.read_text(encoding="utf-8")
                st.download_button(
                    f"Download {label}",
                    data=text.encode("utf-8"),
                    file_name=path.name,
                    mime="text/markdown",
                    key=f"story_industry_{path.stem}",
                )
    if not catalog.empty:
        st.subheader("Model Family Coverage")
        ready_catalog = catalog[catalog["status"] == "ready"].copy()
        family_counts = (
            ready_catalog.groupby("family", as_index=False)
            .size()
            .rename(columns={"size": "ready_models"})
            .sort_values("ready_models", ascending=False)
        )
        source_family_counts = (
            catalog.groupby(["source", "family", "status"], as_index=False)
            .size()
            .rename(columns={"size": "models"})
            .sort_values(["source", "family", "status"])
        )
        left, right = st.columns([0.38, 0.62])
        with left:
            if not family_counts.empty:
                st.bar_chart(family_counts.set_index("family")["ready_models"])
        with right:
            st.dataframe(source_family_counts, width="stretch", hide_index=True)
    readiness_trend_csv = Path("reports/model_readiness_trend_latest.csv")
    if readiness_trend_csv.exists():
        try:
            readiness_trend = pd.read_csv(readiness_trend_csv)
        except Exception:
            readiness_trend = pd.DataFrame()
        if not readiness_trend.empty and {"generated_at", "registered_models", "ready_models"}.issubset(readiness_trend.columns):
            st.subheader("Model Readiness Trend")
            trend_view = readiness_trend[["generated_at", "registered_models", "ready_models"]].tail(12)
            st.line_chart(trend_view.set_index("generated_at")[["registered_models", "ready_models"]])
            st.dataframe(trend_view, width="stretch", hide_index=True)
    interview_pack_json = Path("reports/interview_evidence_pack_latest.json")
    if interview_pack_json.exists():
        try:
            interview_pack = json.loads(interview_pack_json.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            interview_pack = {}
        difficulties = interview_pack.get("difficulties") or []
        if difficulties:
            st.subheader("Interview Difficulty Proof Map")
            proof_map = pd.DataFrame(difficulties)[["id", "topic", "tabs", "proof", "artifact"]]
            st.dataframe(proof_map, width="stretch", hide_index=True)
    render_story_evidence_panel()
    evidence_index_path = Path("docs/DEEPUplift_AGENT_EVIDENCE_INDEX.md")
    if evidence_index_path.exists():
        evidence_index_text = evidence_index_path.read_text(encoding="utf-8")
        with st.expander("Evidence Index", expanded=False):
            st.markdown(evidence_index_text)
            st.download_button(
                "Download evidence index",
                data=evidence_index_text.encode("utf-8"),
                file_name="DEEPUplift_AGENT_EVIDENCE_INDEX.md",
                mime="text/markdown",
            )
    release_note_path = Path("docs/DEEPUplift_AGENT_RELEASE_NOTE.md")
    if release_note_path.exists():
        release_note_text = release_note_path.read_text(encoding="utf-8")
        with st.expander("Release Note", expanded=False):
            st.markdown(release_note_text)
            st.download_button(
                "Download release note",
                data=release_note_text.encode("utf-8"),
                file_name="DEEPUplift_AGENT_RELEASE_NOTE.md",
                mime="text/markdown",
            )
    interview_difficulties_path = Path("docs/INTERVIEW_DIFFICULTIES_AND_SOLUTIONS.md")
    if interview_difficulties_path.exists():
        interview_difficulties_text = interview_difficulties_path.read_text(encoding="utf-8")
        with st.expander("Interview Difficulties & Solutions", expanded=False):
            st.markdown(interview_difficulties_text)
            st.download_button(
                "Download interview difficulties",
                data=interview_difficulties_text.encode("utf-8"),
                file_name="INTERVIEW_DIFFICULTIES_AND_SOLUTIONS.md",
                mime="text/markdown",
            )
    interview_deep_dive_path = Path("docs/INTERVIEW_DEEPUplift_AGENT_DEEP_DIVE.md")
    if interview_deep_dive_path.exists():
        interview_deep_dive_text = interview_deep_dive_path.read_text(encoding="utf-8")
        with st.expander("Interview Deep Dive", expanded=False):
            st.markdown(interview_deep_dive_text)
            st.download_button(
                "Download interview deep dive",
                data=interview_deep_dive_text.encode("utf-8"),
                file_name="INTERVIEW_DEEPUplift_AGENT_DEEP_DIVE.md",
                mime="text/markdown",
            )
    agent_transcript_sample_path = Path("docs/AGENT_INTERVIEW_TRANSCRIPT_SAMPLE.md")
    if agent_transcript_sample_path.exists():
        agent_transcript_sample_text = agent_transcript_sample_path.read_text(encoding="utf-8")
        with st.expander("Agent Interview Transcript Sample", expanded=False):
            st.markdown(agent_transcript_sample_text)
            st.download_button(
                "Download Agent interview transcript sample",
                data=agent_transcript_sample_text.encode("utf-8"),
                file_name="AGENT_INTERVIEW_TRANSCRIPT_SAMPLE.md",
                mime="text/markdown",
            )
    interview_evidence_pack_path = Path("docs/DEEPUplift_AGENT_INTERVIEW_EVIDENCE_PACK.md")
    if interview_evidence_pack_path.exists():
        interview_evidence_pack_text = interview_evidence_pack_path.read_text(encoding="utf-8")
        with st.expander("Interview Evidence Pack", expanded=False):
            st.markdown(interview_evidence_pack_text)
            st.download_button(
                "Download interview evidence pack",
                data=interview_evidence_pack_text.encode("utf-8"),
                file_name="DEEPUplift_AGENT_INTERVIEW_EVIDENCE_PACK.md",
                mime="text/markdown",
            )
    interview_pack_zip_path = Path("reports/deepuplift_agent_interview_pack_latest.zip")
    if interview_pack_zip_path.exists():
        st.download_button(
            "Download complete interview pack ZIP",
            data=interview_pack_zip_path.read_bytes(),
            file_name="deepuplift_agent_interview_pack_latest.zip",
            mime="application/zip",
        )
    interview_qa_path = Path("docs/INTERVIEW_QA_REHEARSAL_CN.md")
    if interview_qa_path.exists():
        interview_qa_text = interview_qa_path.read_text(encoding="utf-8")
        with st.expander("Chinese Interview Q&A Rehearsal", expanded=False):
            st.markdown(interview_qa_text)
            st.download_button(
                "Download Chinese interview Q&A",
                data=interview_qa_text.encode("utf-8"),
                file_name="INTERVIEW_QA_REHEARSAL_CN.md",
                mime="text/markdown",
            )
    interview_checklist_path = Path("docs/INTERVIEW_DEMO_CHECKLIST_CN.md")
    if interview_checklist_path.exists():
        interview_checklist_text = interview_checklist_path.read_text(encoding="utf-8")
        with st.expander("Chinese Interview Demo Checklist", expanded=False):
            st.markdown(interview_checklist_text)
            st.download_button(
                "Download Chinese demo checklist",
                data=interview_checklist_text.encode("utf-8"),
                file_name="INTERVIEW_DEMO_CHECKLIST_CN.md",
                mime="text/markdown",
            )
    snapshot_path = Path("docs/RESUME_DEEPUplift_AGENT_SNAPSHOT.md")
    if snapshot_path.exists():
        snapshot_text = snapshot_path.read_text(encoding="utf-8")
        with st.expander("Live Resume Snapshot", expanded=False):
            st.markdown(snapshot_text)
            st.download_button(
                "Download live resume snapshot",
                data=snapshot_text.encode("utf-8"),
                file_name="RESUME_DEEPUplift_AGENT_SNAPSHOT.md",
                mime="text/markdown",
            )
    walkthrough_path = Path("docs/DEEPUplift_AGENT_DEMO_WALKTHROUGH.md")
    if walkthrough_path.exists():
        walkthrough_text = walkthrough_path.read_text(encoding="utf-8")
        with st.expander("Interview Demo Walkthrough", expanded=False):
            st.markdown(walkthrough_text)
            st.download_button(
                "Download demo walkthrough",
                data=walkthrough_text.encode("utf-8"),
                file_name="DEEPUplift_AGENT_DEMO_WALKTHROUGH.md",
                mime="text/markdown",
            )
    risk_path = Path("docs/DEEPUplift_AGENT_RISK_REGISTER.md")
    if risk_path.exists():
        risk_text = risk_path.read_text(encoding="utf-8")
        with st.expander("Risk Register", expanded=False):
            st.markdown(risk_text)
            st.download_button(
                "Download risk register",
                data=risk_text.encode("utf-8"),
                file_name="DEEPUplift_AGENT_RISK_REGISTER.md",
                mime="text/markdown",
            )
    resume_cn_path = Path("docs/RESUME_BULLETS_CN.md")
    if resume_cn_path.exists():
        resume_cn_text = resume_cn_path.read_text(encoding="utf-8")
        with st.expander("Chinese Resume Bullets", expanded=False):
            st.markdown(resume_cn_text)
            st.download_button(
                "Download Chinese resume bullets",
                data=resume_cn_text.encode("utf-8"),
                file_name="RESUME_BULLETS_CN.md",
                mime="text/markdown",
            )
    cn_onepager_path = Path("docs/DEEPUplift_AGENT_ONEPAGER_CN.md")
    if cn_onepager_path.exists():
        cn_onepager_text = cn_onepager_path.read_text(encoding="utf-8")
        with st.expander("Chinese Project One-Pager", expanded=False):
            st.markdown(cn_onepager_text)
            st.download_button(
                "Download Chinese project one-pager",
                data=cn_onepager_text.encode("utf-8"),
                file_name="DEEPUplift_AGENT_ONEPAGER_CN.md",
                mime="text/markdown",
            )

    resume_bullets = markdown_section(resume_text, "Resume Bullets")
    if resume_bullets:
        st.subheader("Resume Bullets")
        st.markdown(resume_bullets)

    left, right = st.columns([0.52, 0.48])
    with left:
        architecture = markdown_section(resume_text, "Architecture")
        if architecture:
            st.subheader("Architecture")
            st.markdown(architecture)
    with right:
        st.subheader("Interview Angle")
        st.markdown(
            """
            - **业务问题**：触达谁才能带来增量，而不是谁本来就会转化。
            - **工程问题**：把 103 个模型统一成 registry、model card、preset、artifact contract。
            - **因果问题**：在推荐模型前先检查随机化、overlap、calibration 和置信区间。
            - **产品问题**：把 Compare、Policy、Predict、History 串成可复现决策流。
            """
        )

    st.subheader("Production & Backend Tradeoffs")
    production_rows = [
        {"area": "UI", "mvp": "Streamlit workbench", "production": "React + API-driven state"},
        {"area": "API", "mvp": "local core calls", "production": "FastAPI contracts for datasets/runs/artifacts"},
        {"area": "Training", "mvp": "single-process smoke runs", "production": "async worker queue for long-running compare jobs"},
        {"area": "Storage", "mvp": "local runs/ artifacts", "production": "object storage + metadata DB + lineage"},
        {"area": "Governance", "mvp": "evidence ZIP + regression gate", "production": "RBAC, audit logs, model promotion gates"},
    ]
    backend_rows = [
        {"backend": "DeepUplift", "status": "ready", "reason": "native models and benchmark neural baselines"},
        {"backend": "LightGBM", "status": "ready", "reason": "industrial table-data meta-learners"},
        {"backend": "EconML", "status": "ready", "reason": "orthogonal, DR, DML and causal-forest style learners"},
        {"backend": "scikit-uplift", "status": "ready", "reason": "official uplift tree/forest/meta-learner adapters"},
        {"backend": "XGBoost", "status": "guarded", "reason": "cataloged but explicit flag required for local runtime stability"},
        {"backend": "CatBoost", "status": "optional", "reason": "valuable categorical backend, not a default dependency"},
        {"backend": "CausalML", "status": "optional", "reason": "Python-version-specific helper environment"},
    ]
    left, right = st.columns([0.52, 0.48])
    with left:
        st.dataframe(pd.DataFrame(production_rows), width="stretch", hide_index=True)
    with right:
        st.dataframe(pd.DataFrame(backend_rows), width="stretch", hide_index=True)

    architecture_doc_path = Path("docs/DEEPUplift_AGENT_ARCHITECTURE.md")
    if architecture_doc_path.exists():
        architecture_doc = architecture_doc_path.read_text(encoding="utf-8")
        with st.expander("Architecture Diagram & Data Flow", expanded=False):
            st.markdown(architecture_doc)
            st.download_button(
                "Download architecture notes",
                data=architecture_doc.encode("utf-8"),
                file_name="DEEPUplift_AGENT_ARCHITECTURE.md",
                mime="text/markdown",
            )
    production_doc_path = Path("docs/DEEPUplift_AGENT_PRODUCTION_ROADMAP.md")
    if production_doc_path.exists():
        production_doc = production_doc_path.read_text(encoding="utf-8")
        with st.expander("Productionization Roadmap", expanded=False):
            st.markdown(production_doc)
            st.download_button(
                "Download productionization roadmap",
                data=production_doc.encode("utf-8"),
                file_name="DEEPUplift_AGENT_PRODUCTION_ROADMAP.md",
                mime="text/markdown",
            )
    backend_strategy_path = Path("docs/DEEPUplift_AGENT_MODEL_BACKEND_STRATEGY.md")
    if backend_strategy_path.exists():
        backend_strategy = backend_strategy_path.read_text(encoding="utf-8")
        with st.expander("Model Backend Strategy", expanded=False):
            st.markdown(backend_strategy)
            st.download_button(
                "Download model backend strategy",
                data=backend_strategy.encode("utf-8"),
                file_name="DEEPUplift_AGENT_MODEL_BACKEND_STRATEGY.md",
                mime="text/markdown",
            )
    offline_online_path = Path("docs/DEEPUplift_AGENT_OFFLINE_ONLINE_PLAYBOOK.md")
    if offline_online_path.exists():
        offline_online_doc = offline_online_path.read_text(encoding="utf-8")
        with st.expander("Offline-To-Online Validation Playbook", expanded=False):
            st.markdown(offline_online_doc)
            st.download_button(
                "Download offline-online playbook",
                data=offline_online_doc.encode("utf-8"),
                file_name="DEEPUplift_AGENT_OFFLINE_ONLINE_PLAYBOOK.md",
                mime="text/markdown",
                key="story_offline_online_playbook_download",
            )
    readiness_trend_path = Path("docs/DEEPUplift_AGENT_MODEL_READINESS_TREND.md")
    if readiness_trend_path.exists():
        readiness_trend_doc = readiness_trend_path.read_text(encoding="utf-8")
        with st.expander("Model Readiness Trend", expanded=False):
            st.markdown(readiness_trend_doc)
            st.download_button(
                "Download model readiness trend",
                data=readiness_trend_doc.encode("utf-8"),
                file_name="DEEPUplift_AGENT_MODEL_READINESS_TREND.md",
                mime="text/markdown",
            )

    best_resume = markdown_section(resume_text, "Best Resume Version")
    if best_resume:
        st.subheader("Best Resume Version")
        st.markdown(best_resume)

    command_proof = markdown_section(resume_text, "Command-Line Proof")
    if command_proof:
        with st.expander("Command-Line Proof", expanded=False):
            st.markdown(command_proof)

    if resume_text:
        with st.expander("Full Resume Project Brief", expanded=False):
            st.markdown(resume_text)
        st.download_button(
            "Download resume project brief",
            data=resume_text.encode("utf-8"),
            file_name="RESUME_DEEPUplift_AGENT.md",
            mime="text/markdown",
        )
    else:
        st.warning("Missing docs/RESUME_DEEPUplift_AGENT.md. Run the resume documentation step before demo.")


def render_policy_roi_helper(default_conversion_value: float, default_contact_cost: float) -> dict[str, Any]:
    templates = policy_templates()
    template_lookup = {row["scenario"]: row for row in templates}
    scenario = st.selectbox(
        "Business scenario",
        list(template_lookup),
        index=0,
        key="policy_roi_scenario",
        help="Pick the business setting first; the helper turns scenario assumptions into value/cost inputs for the policy curve.",
    )
    template = template_lookup[scenario]
    st.caption(f"Guardrails: {template['extra_guardrails']}")

    if scenario == "Coupon":
        c1, c2, c3, c4 = st.columns(4)
        gross_margin = c1.number_input("Gross margin / order", min_value=0.0, value=max(float(default_conversion_value), 20.0), step=1.0)
        coupon_face = c2.number_input("Coupon face value", min_value=0.0, value=max(float(default_contact_cost), 5.0), step=1.0)
        redemption_prob = c3.number_input("Redemption prob.", min_value=0.0, max_value=1.0, value=0.35, step=0.05)
        fatigue_penalty = c4.number_input("Fatigue / abuse penalty", min_value=0.0, value=0.10, step=0.05)
        conversion = gross_margin
        cost = coupon_face * redemption_prob + fatigue_penalty
        assumptions = {
            "gross_margin": gross_margin,
            "coupon_face_value": coupon_face,
            "redemption_probability": redemption_prob,
            "fatigue_abuse_penalty": fatigue_penalty,
        }
    elif scenario == "Ads":
        c1, c2, c3, c4 = st.columns(4)
        conversion = c1.number_input("Incremental conversion value", min_value=0.0, value=max(float(default_conversion_value), 50.0), step=5.0)
        cpm = c2.number_input("CPM", min_value=0.0, value=20.0, step=1.0)
        exposures = c3.number_input("Exposures / user", min_value=0.0, value=1.0, step=0.25)
        measurement_buffer = c4.number_input("Attribution / spillover buffer", min_value=0.0, value=0.0, step=0.05)
        cost = cpm / 1000.0 * exposures + measurement_buffer
        assumptions = {
            "incremental_conversion_value": conversion,
            "cpm": cpm,
            "exposures_per_user": exposures,
            "measurement_buffer": measurement_buffer,
        }
    elif scenario == "Growth Push":
        c1, c2, c3, c4 = st.columns(4)
        conversion = c1.number_input("Activation / LTV value", min_value=0.0, value=max(float(default_conversion_value), 15.0), step=1.0)
        message_cost = c2.number_input("Message cost", min_value=0.0, value=0.03, step=0.01, format="%.4f")
        fatigue_penalty = c3.number_input("Fatigue penalty", min_value=0.0, value=0.08, step=0.01, format="%.4f")
        opt_out_penalty = c4.number_input("Opt-out risk penalty", min_value=0.0, value=0.10, step=0.01, format="%.4f")
        cost = message_cost + fatigue_penalty + opt_out_penalty
        assumptions = {
            "activation_ltv_value": conversion,
            "message_cost": message_cost,
            "fatigue_penalty": fatigue_penalty,
            "opt_out_risk_penalty": opt_out_penalty,
        }
    else:
        c1, c2, c3, c4 = st.columns(4)
        conversion = c1.number_input("Balance / gross profit value", min_value=0.0, value=max(float(default_conversion_value), 30.0), step=2.0)
        subsidy = c2.number_input("Subsidy payout", min_value=0.0, value=max(float(default_contact_cost), 4.0), step=1.0)
        ops_cost = c3.number_input("Ops cost", min_value=0.0, value=0.20, step=0.05)
        spillover_buffer = c4.number_input("Spillover buffer", min_value=0.0, value=0.50, step=0.10)
        cost = subsidy + ops_cost + spillover_buffer
        assumptions = {
            "balance_gross_profit_value": conversion,
            "subsidy_payout": subsidy,
            "ops_cost": ops_cost,
            "spillover_buffer": spillover_buffer,
        }

    use_helper = st.checkbox("Use scenario assumptions for this Policy page", value=True, key="policy_use_roi_helper")
    render_stat_grid(
        [
            ("Scenario", scenario, True),
            ("Value / conversion", f"{conversion:.4f}", False),
            ("Cost / contact", f"{cost:.4f}", False),
            ("Decision rule", "uplift * value - cost", True),
        ],
        columns=4,
    )
    return {
        "scenario": scenario,
        "template": template,
        "conversion_value": float(conversion),
        "contact_cost": float(cost),
        "use_helper": bool(use_helper),
        "assumptions": assumptions,
    }


@st.cache_data(show_spinner=False)
def load_csv_from_path(path: str, max_rows: int | None = None) -> pd.DataFrame:
    return pd.read_csv(path, nrows=max_rows)


@st.cache_data(show_spinner=False)
def load_dataset_manifest(path: str) -> list[dict]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        return []
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return payload.get("datasets", [])


def infer_default_column(columns: list[str], candidates: list[str], fallback_index: int = 0) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return columns[min(fallback_index, len(columns) - 1)]


def infer_task(df: pd.DataFrame, outcome_col: str) -> str:
    return "classification" if df[outcome_col].dropna().nunique() <= 2 else "regression"


def recommend_models(
    df: pd.DataFrame,
    task: str,
    treatment_col: str,
    feature_cols: list[str],
    selected_dataset: dict | None,
) -> list[str]:
    candidates = available_models(task, only_available=True)
    dataset_recommended = []
    if selected_dataset and selected_dataset.get("recommended_models"):
        dataset_recommended = [model for model in selected_dataset["recommended_models"] if model in candidates]
    categorical_count = sum(not pd.api.types.is_numeric_dtype(df[col]) for col in feature_cols if col in df.columns)
    treatment_ratio = df[treatment_col].mean() if pd.api.types.is_numeric_dtype(df[treatment_col]) else None
    recommended = []

    if task == "classification":
        recommended.extend(["TLearnerLightGBM", "DRLearnerLightGBM", "SkLiftTwoModelsLightGBM", "GGBMUpliftLightGBM", "SLearnerLightGBM", "TarNet", "DragonNet"])
        if categorical_count:
            recommended.extend(["XLearnerLightGBM", "EFIN"])
        if treatment_ratio is not None and (treatment_ratio < 0.35 or treatment_ratio > 0.65):
            recommended.extend(["XLearnerLightGBM", "CFRNet", "DRForest"])
        recommended.extend(["RLearnerLightGBM", "OrthogonalDMLGBM", "CausalForest", "DESCN"])
        recommended.extend(["GGBMUpliftLightGBM", "PAVCalibratedDRLearnerLightGBM", "PAVCalibratedRLearnerLightGBM"])
        recommended.extend(["ClassVariableTransformGBM", "PAVCalibratedClassVariableTransformGBM"])
        recommended.extend(["SkLiftTwoModelsLightGBM", "SkLiftSoloInteractionLightGBM", "SkLiftClassTransformationLightGBM", "SkLiftTransformedOutcomeLightGBM"])
    else:
        recommended.extend(["SLearnerLightGBM", "TLearnerLightGBM", "XLearnerLightGBM", "DRLearnerLightGBM", "RLearnerLightGBM", "GGBMUpliftLightGBM", "SkLiftTwoModelsLightGBM", "SkLiftTransformedOutcomeLightGBM", "PAVCalibratedDRLearnerLightGBM", "PAVCalibratedCausalForest", "OrthogonalDMLGBM", "CausalForest", "DRForest", "TarNet", "CFRNet", "DragonNet", "EUEN"])

    recommended.extend(dataset_recommended)
    return [model for model in dict.fromkeys(recommended) if model in candidates]


def compare_model_presets(
    df: pd.DataFrame,
    task: str,
    treatment_col: str,
    feature_cols: list[str],
    selected_dataset: dict | None,
) -> dict[str, list[str]]:
    ready = set(available_models(task, only_available=True))
    recommended = recommend_models(df, task, treatment_col, feature_cols, selected_dataset)
    treatment_ratio = df[treatment_col].mean() if pd.api.types.is_numeric_dtype(df[treatment_col]) else None
    imbalanced_binary = treatment_ratio is not None and (treatment_ratio < 0.35 or treatment_ratio > 0.65)

    def keep(models: list[str], limit: int | None = None) -> list[str]:
        picked = [model for model in dict.fromkeys(models) if model in ready]
        return picked[:limit] if limit else picked

    industrial = keep(
        [
            "TLearnerLightGBM",
            "DRLearnerLightGBM",
            "RLearnerLightGBM",
            "GGBMUpliftLightGBM",
            "SkLiftTwoModelsLightGBM",
            "SkLiftTransformedOutcomeLightGBM",
        ]
    )
    observational = keep(
        [
            "DRLearnerLightGBM",
            "RLearnerLightGBM",
            "DomainAdaptationLightGBM",
            "EconMLCausalForestDML",
            "EconMLDRLearner",
            "EconMLGRFCausalForest",
        ]
    )
    official = keep(
        [
            "SkLiftSoloLightGBM",
            "SkLiftTwoModelsLightGBM",
            "SkLiftTwoModelsDDRControlLightGBM",
            "SkLiftTwoModelsDDRTreatmentLightGBM",
            "SkLiftClassTransformationLightGBM",
            "SkLiftTransformedOutcomeLightGBM",
            "EconMLLinearDRLearner",
            "EconMLCausalForestDML",
        ]
    )
    neural = keep(["TarNet", "CFRNet", "DragonNet", "EFIN", "DESCN", "EUEN", "EEUEN"])
    calibrated = keep(
        [
            "GGBMUpliftLightGBM",
            "PAVCalibratedDRLearnerLightGBM",
            "PAVCalibratedRLearnerLightGBM",
            "PAVCalibratedCausalForest",
            "PAVCalibratedClassVariableTransformGBM",
        ]
    )
    dynamic_label = "Recommended: observational / imbalanced" if imbalanced_binary else "Recommended: randomized / A-B"
    dynamic_models = keep(observational if imbalanced_binary else recommended, limit=4)
    if not dynamic_models:
        dynamic_models = keep(recommended, limit=4)
    return {
        dynamic_label: dynamic_models,
        "Recommended short list": keep(recommended, limit=4),
        "Industrial tabular": industrial,
        "Observational robust": observational,
        "Official open-source adapters": official,
        "Calibration / Top-K ranking": calibrated,
        "Neural baselines": neural,
    }


def format_model_choices(models: list[str]) -> str:
    rows = []
    for model in models:
        missing = missing_dependencies(model)
        status = f" 依赖未就绪：{', '.join(missing)}。" if missing else ""
        rows.append(f"- `{model}`: {MODEL_DESCRIPTIONS.get(model, '可用于当前任务。')}{status}")
    return "\n".join(rows)


def format_model_option(model: str) -> str:
    missing = missing_dependencies(model)
    if missing:
        return f"{model}  · needs {', '.join(missing)}"
    return model


def extract_model_mentions(prompt: str) -> list[str]:
    text = prompt.lower()
    mentions = []
    for model in sorted(MODEL_REGISTRY, key=len, reverse=True):
        if model.lower() in text:
            mentions.append(model)
    return list(dict.fromkeys(mentions))


def model_source_and_family(model: str) -> tuple[str, str]:
    if model.startswith("Multi"):
        return "DeepUplift", "multi_treatment"
    if model.startswith("DoseResponse"):
        return "DeepUplift", "continuous_treatment"
    return catalog_source_and_family(model)


def model_readiness(model: str, missing: list[str]) -> tuple[str, str]:
    return catalog_model_readiness(model, missing)


def model_scenario_tags(model: str) -> list[str]:
    return core_model_scenario_tags(model)


def model_treatment_type(model: str, family: str, scenario_tags: list[str]) -> str:
    family_lower = str(family).lower()
    tag_text = " ".join(scenario_tags).lower()
    if model.startswith("Multi") or "multi_treatment" in family_lower:
        return "multi-treatment / action"
    if model.startswith("DoseResponse") or "continuous_treatment" in family_lower:
        return "continuous / dose-response"
    if "llm routing" in tag_text:
        return "binary treatment / LLM routing"
    if "marketplace" in tag_text:
        return "binary or multi-action marketplace"
    return "binary treatment"


def model_catalog_dataframe(task: str | None = None) -> pd.DataFrame:
    rows = []
    for name in available_models(task):
        spec = MODEL_REGISTRY[name]
        missing = missing_dependencies(name)
        card = build_model_card(name, description=MODEL_DESCRIPTIONS.get(name, ""))
        preset_name, _preset_params = recommended_model_preset(name)
        presets = model_parameter_presets(name)
        scenario_tags = model_scenario_tags(name)
        rows.append(
            {
                "model": name,
                "status": card["status"],
                "source": card["source"],
                "family": card["family"],
                "treatment_type": model_treatment_type(name, card["family"], scenario_tags),
                "stage": card["stage"],
                "tasks": ", ".join(spec.supported_tasks),
                "missing": ", ".join(missing),
                "setup_hint": card["setup_hint"],
                "description": card["description"],
                "best_for": "; ".join(card["best_for"]),
                "scenario_tags": "; ".join(scenario_tags),
                "decision_use": card["decision_use"],
                "preset_count": len(presets),
                "recommended_preset": preset_name,
            }
        )
    return pd.DataFrame(rows)


def model_code_provenance_rows() -> list[dict[str, str]]:
    return [
        {
            "area": "DeepUplift neural models",
            "code_path": "deepuplift/models/*.py",
            "source_or_backend": "local DeepUplift implementations",
            "coverage": "TarNet, CFRNet, DragonNet, DESCN/ESX, EFIN, EUEN/EEUEN, DragonDeepFM",
            "why_it_matters": "keeps the original deep uplift family usable behind a unified adapter contract",
        },
        {
            "area": "Meta learners and calibrated ranking",
            "code_path": "deepuplift/core/sklearn_models.py",
            "source_or_backend": "scikit-learn, LightGBM, optional XGBoost/CatBoost",
            "coverage": "S/T/X/DR/R learners, IPW, transformed outcome, domain adaptation, PAV-calibrated variants",
            "why_it_matters": "industrial tabular uplift baselines are fast, interpretable and easy to compare",
        },
        {
            "area": "External causal estimators",
            "code_path": "deepuplift/core/external_models.py",
            "source_or_backend": "EconML, scikit-uplift, optional CausalML, guarded UTBoost",
            "coverage": "DML, DRLearner, causal forests, uplift tree/forest, official meta-learner wrappers, UTBoostGBM adapter path",
            "why_it_matters": "lets the catalog distinguish ready, guarded and optional backends instead of hiding dependency risk",
        },
        {
            "area": "Multi-treatment workflow",
            "code_path": "deepuplift/core/multitreatment.py",
            "source_or_backend": "local action-specific learner adapters",
            "coverage": "MultiTLearnerGBM/RF, MultiDRLearnerGBM/RF, action propensity and IPW policy value",
            "why_it_matters": "supports coupon type, channel, offer and action recommendation beyond binary treatment",
        },
        {
            "area": "Dose-response workflow",
            "code_path": "deepuplift/core/doseresponse.py",
            "source_or_backend": "local continuous-treatment baseline",
            "coverage": "DoseResponseGBM/RF, dose grid, recommended dose and gain-vs-baseline scoring",
            "why_it_matters": "covers discount depth, subsidy amount, frequency and price/intensity decisions",
        },
        {
            "area": "Evaluation contract",
            "code_path": "deepuplift/core/evaluator.py; deepuplift/core/readiness.py",
            "source_or_backend": "local evaluator and decision-readiness gate",
            "coverage": "QINI, AUUC, Top-K, calibration, bootstrap CI, overlap trimming, policy value, oracle recall",
            "why_it_matters": "model outputs become comparable decision evidence instead of isolated score columns",
        },
        {
            "area": "Training, scoring and artifacts",
            "code_path": "deepuplift/core/trainer.py; deepuplift/core/predictor.py; deepuplift/core/artifacts.py",
            "source_or_backend": "local orchestration layer",
            "coverage": "preprocessing, fit/predict, predictions.csv, metrics.json, run_note.md, evidence ZIP",
            "why_it_matters": "keeps the workflow reproducible from training through Top-K audience export",
        },
        {
            "area": "Model cards and industrial playbooks",
            "code_path": "deepuplift/core/model_cards.py; deepuplift/core/industry_playbook.py",
            "source_or_backend": "local knowledge base plus official/paper references",
            "coverage": "scenario tags, assumptions, risks, presets, guarded backend guidance, interview evidence",
            "why_it_matters": "connects estimator choice to treatment design, business scenario and evidence quality",
        },
    ]


def model_card_governance_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for name in sorted(MODEL_REGISTRY):
        card = build_model_card(name, description=MODEL_DESCRIPTIONS.get(name, ""))
        status = str(card.get("status", ""))
        source = str(card.get("source", ""))
        integration_gate = str(card.get("integration_gate", ""))
        source_evidence = str(card.get("source_evidence", ""))
        promotion_rule = str(card.get("promotion_rule", ""))
        missing = "; ".join(str(item) for item in card.get("missing", []))
        if status == "ready" and not integration_gate and source not in {"XGBoost", "CatBoost", "CausalML", "UTBoost"}:
            continue
        rows.append(
            {
                "model": name,
                "status": status,
                "source": source,
                "family": str(card.get("family", "")),
                "missing_or_guard": missing,
                "setup_hint": str(card.get("setup_hint", "")),
                "integration_gate": integration_gate or "standard catalog audit + compare smoke",
                "source_evidence": source_evidence or str(card.get("source_url", "")),
                "promotion_rule": promotion_rule or "promote only after dependency/import, smoke, compare and evidence checks pass",
            }
        )
    return rows


def render_model_source_provenance() -> None:
    st.subheader("Model Source & Code Provenance")
    st.caption("模型目录不是只列名字：每个 family 都要能追到本地代码、外部后端、适用 treatment 类型和验证证据。")
    st.dataframe(pd.DataFrame(model_code_provenance_rows()), width="stretch", hide_index=True)


def render_open_source_framework_audit(key_prefix: str = "models") -> None:
    st.subheader("开源框架审计")
    st.caption(
        "这里不是简单列库名，而是把每个 uplift / CATE / causal ML 框架映射到：可支持模型、treatment 类型、"
        "本地 ready/guarded 状态、adapter 路径、依赖限制和下一步接入路线。"
    )
    rows = pd.DataFrame(open_source_framework_rows())
    counts = open_source_framework_counts()
    render_stat_grid(
        [
            ("框架总数", f"{counts['total']:,}", False),
            ("Ready", f"{counts['ready']:,}", True),
            ("Guarded", f"{counts['guarded']:,}", True),
            ("Optional", f"{counts['optional']:,}", True),
            ("Backlog", f"{counts['backlog']:,}", True),
        ],
        columns=5,
    )
    status_df = pd.DataFrame(framework_status_summary())
    if not status_df.empty:
        st.dataframe(
            status_df.rename(
                columns={
                    "status": "状态",
                    "frameworks": "框架数",
                    "examples": "示例",
                    "how_to_read": "怎么理解",
                }
            ),
            width="stretch",
            hide_index=True,
        )
    if rows.empty:
        return
    c1, c2, c3 = st.columns([0.24, 0.30, 0.46])
    with c1:
        status_filter = st.multiselect(
            "接入状态",
            sorted(rows["local_status"].unique().tolist()),
            default=sorted(rows["local_status"].unique().tolist()),
            key=f"{key_prefix}_open_source_framework_status",
        )
    with c2:
        source_filter = st.multiselect(
            "来源类型",
            sorted(rows["source_type"].unique().tolist()),
            default=sorted(rows["source_type"].unique().tolist()),
            key=f"{key_prefix}_open_source_framework_source_type",
        )
    with c3:
        framework_filter = st.multiselect(
            "框架",
            rows["framework"].tolist(),
            default=rows["framework"].tolist(),
            key=f"{key_prefix}_open_source_framework_name_filter",
        )
    view = rows[
        rows["local_status"].isin(status_filter)
        & rows["source_type"].isin(source_filter)
        & rows["framework"].isin(framework_filter)
    ].copy()
    display = view.rename(
        columns={
            "framework": "框架",
            "owner": "维护方",
            "source_type": "来源",
            "github_repo": "GitHub repo",
            "github_stars": "Stars",
            "github_updated_at": "GitHub updated",
            "github_activity": "活跃度",
            "github_api_status": "API 状态",
            "supported_models": "支持模型",
            "treatment_support": "Treatment 支持",
            "scenario_fit": "适合场景",
            "local_status": "本地状态",
            "integration_path": "接入路径",
            "limitations": "限制",
            "next_step": "下一步",
            "interview_line": "面试讲法",
            "url": "来源链接",
        }
    )
    st.dataframe(
        display[
            [
                "框架",
                "本地状态",
                "GitHub repo",
                "Stars",
                "GitHub updated",
                "活跃度",
                "支持模型",
                "Treatment 支持",
                "适合场景",
                "接入路径",
                "限制",
                "下一步",
                "来源链接",
            ]
        ],
        width="stretch",
        hide_index=True,
    )
    st.download_button(
        "下载开源框架审计 CSV",
        data=dataframe_csv_bytes(rows),
        file_name="deepuplift_open_source_framework_audit.csv",
        mime="text/csv",
        key=f"{key_prefix}_download_open_source_framework_audit",
    )


def render_frontier_model_architecture() -> None:
    st.subheader("Industrial Model Layer Architecture")
    st.caption(
        "按工业落地优先级组织模型能力：P0 是可训练强基线，P1 是高级工业插件，P2 是前沿/业务扩展。"
        "每个能力都绑定 adapter、指标、业务场景、依赖状态和 source gate。"
    )
    rows = pd.DataFrame(frontier_model_capabilities())
    counts = capability_counts()
    render_stat_grid(
        [
            ("Capabilities", f"{counts['total']:,}", False),
            ("Ready-linked", f"{counts['ready']:,}", True),
            ("Guarded", f"{counts['guarded']:,}", True),
            ("Optional", f"{counts['optional']:,}", True),
            ("Backlog", f"{counts['backlog']:,}", True),
        ],
        columns=5,
    )
    scenario_lab_df = pd.DataFrame(scenario_dataset_summary())
    if not scenario_lab_df.empty:
        st.subheader("Scenario Data-To-Model Matrix")
        st.caption("Every industrial scenario points to a runnable dataset, policy objective, metrics and recommended model path.")
        st.dataframe(
            scenario_lab_df[
                [
                    "scenario",
                    "dataset_id",
                    "status",
                    "rows",
                    "primary_metrics",
                    "recommended_models",
                    "policy_objective",
                ]
            ],
            width="stretch",
            hide_index=True,
        )
    if rows.empty:
        return
    st.markdown(
        "<div class='du-chip-row'>"
        "<span class='du-chip'>P0 Meta-Learners</span>"
        "<span class='du-chip'>Causal Forest / DML</span>"
        "<span class='du-chip'>Uplift Tree / Forest</span>"
        "<span class='du-chip'>UTBoost</span>"
        "<span class='du-chip'>EFIN</span>"
        "<span class='du-chip'>DESCN / ESX</span>"
        "<span class='du-chip'>UMLC</span>"
        "<span class='du-chip'>ECUP / full-funnel uplift</span>"
        "<span class='du-chip'>CFR-DF / delayed feedback</span>"
        "<span class='du-chip'>Continuous Treatment</span>"
        "<span class='du-chip'>LLM Routing Uplift</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    priority_options = sorted(rows["priority"].unique().tolist())
    layer_options = sorted(rows["layer"].unique().tolist())
    status_options = sorted(rows["integration_status"].unique().tolist())
    scenario_options = sorted(
        {
            scenario.strip()
            for value in rows["business_scenarios"].tolist()
            for scenario in str(value).split(",")
            if scenario.strip()
        }
    )
    c1, c2, c3, c4 = st.columns([0.16, 0.22, 0.26, 0.36])
    with c1:
        priority_filter = st.multiselect("Priority", priority_options, default=priority_options, key="frontier_priority")
    with c2:
        layer_filter = st.multiselect("Capability Layer", layer_options, default=layer_options, key="frontier_layer")
    with c3:
        status_filter = st.multiselect("Integration Status", status_options, default=status_options, key="frontier_status")
    with c4:
        scenario_filter = st.multiselect("Business Scenario", scenario_options, default=scenario_options, key="frontier_scenario")
    view = rows[
        rows["priority"].isin(priority_filter)
        & rows["layer"].isin(layer_filter)
        & rows["integration_status"].isin(status_filter)
    ].copy()
    if scenario_filter:
        view = view[
            view["business_scenarios"].apply(
                lambda value: any(scenario in str(value) for scenario in scenario_filter)
            )
        ]
    st.dataframe(
        view[
            [
                "priority",
                "capability",
                "layer",
                "integration_status",
                "training_status",
                "treatment_type",
                "business_scenarios",
                "adapter_path",
                "model_examples",
                "metrics",
                "risks",
                "recommended_action",
            ]
        ],
        width="stretch",
        hide_index=True,
    )
    source_df = pd.DataFrame(frontier_source_rows())
    st.subheader("Frontier Plugin Backlog")
    st.caption("P1/P2 能力先经过 source gate、adapter gate、metric gate、smoke gate，稳定后再进入默认训练菜单。")
    st.dataframe(source_df, width="stretch", hide_index=True)
    st.download_button(
        "Download frontier model capability matrix",
        data=dataframe_csv_bytes(rows),
        file_name="deepuplift_frontier_model_capabilities.csv",
        mime="text/csv",
    )


def render_deep_loss_network_catalog() -> None:
    st.subheader("深度 Uplift Loss Catalog")
    st.caption(
        "Loss 按优化目标分组：factual fit、因果稳健性、排序/决策收益、full-funnel/delayed label、"
        "contrastive representation 和 LLM routing cost-quality。"
    )
    loss_df = pd.DataFrame(deep_loss_catalog())
    network_df = pd.DataFrame(network_architecture_catalog())
    drill_df = pd.DataFrame(deep_interview_drill_rows())
    counts = deep_design_counts()
    render_stat_grid(
        [
            ("Loss Rows", f"{counts.get('losses', 0)}", False),
            ("Network Rows", f"{counts.get('architectures', 0)}", False),
            ("Ready", f"{counts.get('ready', 0)}", False),
            ("Loss Library Ready", f"{counts.get('loss_library_ready', 0)}", False),
        ],
        columns=4,
    )
    if not loss_df.empty:
        status_filter = st.multiselect(
            "Loss status",
            sorted(loss_df["status"].unique().tolist()),
            default=sorted(loss_df["status"].unique().tolist()),
            key="deep_loss_status_filter",
        )
        loss_view = loss_df[loss_df["status"].isin(status_filter)]
        st.dataframe(
            loss_view[
                [
                    "loss_id",
                    "family",
                    "status",
                    "objective",
                    "formula",
                    "business_use",
                    "metric_link",
                    "code_path",
                    "source_url",
                    "risk",
                    "interview_line",
                ]
            ],
            width="stretch",
            hide_index=True,
        )
        st.download_button(
            "Download deep uplift loss catalog",
            data=dataframe_csv_bytes(loss_df),
            file_name="deepuplift_loss_catalog.csv",
            mime="text/csv",
            key="download_deep_loss_catalog_models",
        )

    st.subheader("深度网络结构 Catalog")
    st.caption("每个 architecture 都映射到核心模块、loss stack、适用业务、代码路径和面试讲法。")
    if not network_df.empty:
        st.dataframe(
            network_df[
                [
                    "architecture",
                    "status",
                    "core_blocks",
                    "loss_stack",
                    "best_for",
                    "code_path",
                    "source_url",
                    "interview_line",
                ]
            ],
            width="stretch",
            hide_index=True,
        )
        st.download_button(
            "Download network architecture catalog",
            data=dataframe_csv_bytes(network_df),
            file_name="deepuplift_network_architecture_catalog.csv",
            mime="text/csv",
            key="download_network_architecture_catalog_models",
        )

    st.subheader("深度算法面试追问 Drill")
    st.caption(
        "把论文和工程实现转成面试可讲的问题链：为什么难、公式是什么、项目里哪里能证明、面试官追问时怎么展开。"
    )
    if not drill_df.empty:
        st.dataframe(drill_df, width="stretch", hide_index=True)
        st.download_button(
            "Download deep algorithm interview drill",
            data=dataframe_csv_bytes(drill_df),
            file_name="deepuplift_deep_algorithm_interview_drill.csv",
            mime="text/csv",
            key="download_deep_algorithm_interview_drill",
        )


def _graphviz_label(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def model_deconstruction_catalog_rows() -> list[dict[str, str]]:
    rows = []
    for model in MODEL_DECONSTRUCTIONS.values():
        rows.append(
            {
                "model": model.model_id,
                "display_name": model.display_name,
                "family": model.family,
                "role": model.role,
                "status": model.current_status,
                "main_code": model.code_paths[0] if model.code_paths else "",
                "primary_metric": ", ".join(model.eval_metrics[:4]),
                "license_gate": model.license_policy,
                "next_task": model.next_tasks[0] if model.next_tasks else "",
                "interview_line": model.interview_line,
            }
        )
    return rows


def render_model_deconstruction_workbench(current_model: str | None = None, key_prefix: str = "model_deconstruction") -> None:
    st.subheader("Model Deconstruction Workbench")
    st.caption(
        "把新增的 Model Deconstruction Agent 3.0 做成可视化工作台：模型源码入口、公式、loss/forward/fit/predict、"
        "训练评估证据、外部来源和 license gate 都能在一个页面里检查。"
    )

    artifacts = latest_report_artifacts()
    catalog_payload = _read_json_file(Path((artifacts.get("model_deconstruction_catalog") or {}).get("path", "")))
    agent_payload = _read_json_file(Path((artifacts.get("model_deconstruction_agent") or {}).get("path", "")))
    training_payload = _read_json_file(Path((artifacts.get("deep_model_training_evidence") or {}).get("path", "")))
    tuned_payload = _read_json_file(Path((artifacts.get("deep_model_tuned_benchmark") or {}).get("path", "")))
    tuned_diff_payload = _read_json_file(Path((artifacts.get("deep_model_tuned_benchmark_run_diff") or {}).get("path", "")))
    objective_payload = _read_json_file(Path((artifacts.get("deep_model_objective_diagnostics") or {}).get("path", "")))
    claim_ledger_payload = _read_json_file(Path((artifacts.get("algorithm_claim_ledger") or {}).get("path", "")))
    reproduction_gap_payload = _read_json_file(Path((artifacts.get("paper_reproduction_gap_ledger") or {}).get("path", "")))
    promotion_matrix_payload = _read_json_file(Path((artifacts.get("model_evidence_promotion_matrix") or {}).get("path", "")))
    upgrade_recipe_payload = _read_json_file(Path((artifacts.get("model_upgrade_recipe_cards") or {}).get("path", "")))
    telemetry_gap_payload = _read_json_file(Path((artifacts.get("deep_model_telemetry_gap_matrix") or {}).get("path", "")))
    telemetry_smoke_payload = _read_json_file(Path((artifacts.get("deep_model_telemetry_smoke") or {}).get("path", "")))
    forward_hook_payload = _read_json_file(Path((artifacts.get("deep_model_forward_hook_contracts") or {}).get("path", "")))
    forward_hook_smoke_payload = _read_json_file(Path((artifacts.get("deep_model_forward_hook_smoke") or {}).get("path", "")))
    hook_training_bridge_payload = _read_json_file(Path((artifacts.get("deep_model_hook_training_bridge") or {}).get("path", "")))
    trainer_collector_payload = _read_json_file(Path((artifacts.get("deep_model_trainer_collector_smoke") or {}).get("path", "")))
    telemetry_benchmark_linkage_payload = _read_json_file(Path((artifacts.get("deep_model_telemetry_benchmark_linkage") or {}).get("path", "")))
    telemetry_ablation_gate_payload = _read_json_file(Path((artifacts.get("deep_model_telemetry_ablation_gate") or {}).get("path", "")))
    ablation_interpretation_payload = _read_json_file(Path((artifacts.get("deep_model_ablation_interpretation") or {}).get("path", "")))
    ablation_promotion_payload = _read_json_file(Path((artifacts.get("deep_model_ablation_promotion_matrix") or {}).get("path", "")))
    ablation_next_plan_payload = _read_json_file(Path((artifacts.get("deep_model_ablation_next_experiment_plan") or {}).get("path", "")))
    ablation_command_contract_payload = _read_json_file(Path((artifacts.get("deep_model_ablation_command_contract") or {}).get("path", "")))
    ablation_command_contract_smoke_payload = _read_json_file(Path((artifacts.get("deep_model_ablation_command_contract_smoke") or {}).get("path", "")))
    ablation_recommended_command_smoke_payload = _read_json_file(Path((artifacts.get("deep_model_ablation_recommended_command_smoke") or {}).get("path", "")))
    ablation_command_smoke_coverage_payload = _read_json_file(Path((artifacts.get("deep_model_ablation_command_smoke_coverage") or {}).get("path", "")))
    ablation_command_smoke_coverage_diff_payload = _read_json_file(Path((artifacts.get("deep_model_ablation_command_smoke_coverage_diff") or {}).get("path", "")))
    paper_payload = _read_json_file(Path((artifacts.get("paper_benchmark") or {}).get("path", "")))
    paper_diff_payload = _read_json_file(Path((artifacts.get("paper_benchmark_run_diff") or {}).get("path", "")))
    paper_interpretation_payload = _read_json_file(Path((artifacts.get("paper_benchmark_interpretation") or {}).get("path", "")))
    agent_rows = agent_payload.get("rows") or []
    training_rows = training_payload.get("leaderboard") or []
    tuned_rows = tuned_payload.get("leaderboard") or []
    paper_rows = paper_payload.get("leaderboard") or []
    avg_ratio = 0.0
    if agent_rows:
        ratios = [float(row.get("rubric_ratio", 0.0) or 0.0) for row in agent_rows if isinstance(row, dict)]
        avg_ratio = sum(ratios) / max(len(ratios), 1)
    passed_questions = sum(1 for row in agent_rows if isinstance(row, dict) and row.get("passed"))
    catalog_rows = model_deconstruction_catalog_rows()
    license_gate_count = sum(
        1
        for model in MODEL_DECONSTRUCTIONS.values()
        if any(
            risk in model.license_policy.lower()
            for risk in ("gpl", "unknown", "guarded", "reference-only", "do not copy")
        )
    )

    render_stat_grid(
        [
            ("Deconstructed Models", f"{len(catalog_rows):,}", False),
            ("Golden Questions", f"{len(MODEL_DECONSTRUCTION_GOLDEN_QUESTIONS):,}", False),
            ("Agent Pass", f"{passed_questions}/{len(agent_rows) or 'NA'}", True),
            ("Avg Rubric", f"{avg_ratio:.2f}" if agent_rows else "NA", True),
            ("Training Evidence", f"{training_payload.get('ok_models', 'NA')}", False),
            ("Deep Battle", f"{len(tuned_payload.get('battle_cards', [])) if tuned_payload else 'NA'}", False),
            ("Objective Cards", f"{(objective_payload.get('summary') or {}).get('models', 'NA')}", False),
            ("Claim Rows", f"{(claim_ledger_payload.get('summary') or {}).get('claims', 'NA')}", False),
            ("Paper Gaps", f"{(reproduction_gap_payload.get('summary') or {}).get('source_rows', 'NA')}", False),
            ("Promotion Matrix", f"{(promotion_matrix_payload.get('summary') or {}).get('models', 'NA')}", False),
            ("Upgrade Recipes", f"{(upgrade_recipe_payload.get('summary') or {}).get('recipes', 'NA')}", False),
            ("Telemetry Gaps", f"{(telemetry_gap_payload.get('summary') or {}).get('telemetry_rows', 'NA')}", False),
            ("Telemetry Smoke", f"{(telemetry_smoke_payload.get('summary') or {}).get('basic_telemetry_pass', 'NA')}", False),
            ("Forward Hooks", f"{(forward_hook_payload.get('summary') or {}).get('contracts', 'NA')}", False),
            ("Hook Smoke", f"{(forward_hook_smoke_payload.get('summary') or {}).get('ok_contracts', 'NA')}", False),
            ("Telemetry Links", f"{(telemetry_benchmark_linkage_payload.get('summary') or {}).get('link_ready', 'NA')}", False),
            ("Ablation Gates", f"{(telemetry_ablation_gate_payload.get('summary') or {}).get('gates', 'NA')}", False),
            ("Ablation Rows", f"{(ablation_interpretation_payload.get('summary') or {}).get('rows', 'NA')}", False),
            ("Ablation Promo", f"{(ablation_promotion_payload.get('summary') or {}).get('promotion_ready_rows', 'NA')}", True),
            ("Ablation Plan", f"{(ablation_next_plan_payload.get('summary') or {}).get('p0_rows', 'NA')}", True),
            ("Command Contract", f"{(ablation_command_contract_payload.get('summary') or {}).get('fallback_ready_rows', 'NA')}", True),
            ("Command Smoke", f"{(ablation_command_contract_smoke_payload.get('summary') or {}).get('ok_rows', 'NA')}", True),
            ("Recommended Cmd", f"{(ablation_recommended_command_smoke_payload.get('summary') or {}).get('ok_rows', 'NA')}", True),
            ("Smoke Coverage", f"{(ablation_command_smoke_coverage_payload.get('summary') or {}).get('any_smoked_rows', 'NA')}", True),
            ("Coverage Diff", f"{(ablation_command_smoke_coverage_diff_payload.get('summary') or {}).get('smoke_gain_rows', 'NA')}", True),
            ("Paper Bench", f"{len(paper_rows):,}" if paper_rows else "NA", False),
            ("License Gates", f"{license_gate_count:,}", False),
        ],
        columns=6,
    )
    st.markdown(
        "<div class='du-token-strip'>"
        "<span class='du-token'>Source map</span>"
        "<span class='du-token'>Formula / objective</span>"
        "<span class='du-token'>fit / forward / predict</span>"
        "<span class='du-token'>Training smoke</span>"
        "<span class='du-token'>PEHE / ATE benchmark</span>"
        "<span class='du-token'>Objective diagnostics</span>"
        "<span class='du-token'>Claim ledger</span>"
        "<span class='du-token'>Paper reproduction gaps</span>"
        "<span class='du-token'>Promotion matrix</span>"
        "<span class='du-token'>Upgrade recipes</span>"
        "<span class='du-token'>Telemetry gaps</span>"
        "<span class='du-token'>Telemetry smoke</span>"
        "<span class='du-token'>Telemetry-benchmark linkage</span>"
        "<span class='du-token'>Telemetry ablation gate</span>"
        "<span class='du-token'>Ablation interpretation</span>"
        "<span class='du-token'>Ablation promotion matrix</span>"
        "<span class='du-token'>Ablation next experiments</span>"
        "<span class='du-token'>Loss curve</span>"
        "<span class='du-token'>License gate</span>"
        "<span class='du-token'>Interview talk track</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    catalog_df = pd.DataFrame(catalog_rows)
    st.subheader("Deconstruction Coverage")
    st.dataframe(catalog_df, width="stretch", hide_index=True)
    st.download_button(
        "Download model deconstruction catalog CSV",
        data=dataframe_csv_bytes(catalog_df),
        file_name="deepuplift_model_deconstruction_catalog.csv",
        mime="text/csv",
        key=f"{key_prefix}_download_catalog_csv",
    )

    model_ids = [row["model"] for row in catalog_rows]
    detected_model = detect_deconstruction_model(current_model or "") if current_model else None
    default_model = detected_model if detected_model in model_ids else "CFRNet"
    selected_model_id = st.selectbox(
        "Open source-level model deconstruction",
        model_ids,
        index=model_ids.index(default_model) if default_model in model_ids else 0,
        key=f"{key_prefix}_selected_model",
    )
    model = get_model_deconstruction(selected_model_id)

    st.subheader("Model Review Card")
    tuned_scorecard = next(
        (row for row in tuned_payload.get("model_scorecards", []) if row.get("model") == selected_model_id),
        {},
    )
    paper_scorecard = next(
        (row for row in paper_interpretation_payload.get("model_scorecards", []) if row.get("model") == selected_model_id),
        {},
    )
    battle_rows_for_model = [
        row
        for row in tuned_payload.get("battle_cards", [])
        if row.get("model") == selected_model_id or str(row.get("variant_id") or "").startswith(selected_model_id)
    ]
    review_rows = [
        {
            "review_item": "Architecture claim",
            "status": model.current_status,
            "evidence": model.code_paths[0] if model.code_paths else "",
            "safe_talk_track": model.interview_line,
        },
        {
            "review_item": "Paper-level verdict",
            "status": paper_scorecard.get("verdict", "not_in_latest_paper_scorecard"),
            "evidence": "reports/paper_benchmark_interpretation_latest.json",
            "safe_talk_track": paper_scorecard.get("interview_line", "Use this model's architecture role and guarded benchmark boundary."),
        },
        {
            "review_item": "Deep tuned battle verdict",
            "status": tuned_scorecard.get("verdict", "not_in_latest_tuned_scorecard"),
            "evidence": "reports/deep_model_tuned_benchmark_latest.json",
            "safe_talk_track": tuned_scorecard.get("interview_line", "Use smoke/model-card evidence until tuned battle rows exist."),
        },
        {
            "review_item": "Failure boundary",
            "status": "explicit",
            "evidence": ", ".join(model.failure_modes[:2]),
            "safe_talk_track": tuned_scorecard.get("watchout", model.failure_modes[0] if model.failure_modes else ""),
        },
    ]
    selected_linkage_rows = [
        row for row in telemetry_benchmark_linkage_payload.get("rows") or [] if row.get("model") == selected_model_id
    ]
    if selected_linkage_rows:
        review_rows.append(
            {
                "review_item": "Telemetry to benchmark linkage",
                "status": "link_ready",
                "evidence": "reports/deep_model_telemetry_benchmark_linkage_latest.json",
                "safe_talk_track": selected_linkage_rows[0].get("interview_line", ""),
            }
        )
    selected_ablation_gate_rows = [
        row for row in telemetry_ablation_gate_payload.get("rows") or [] if row.get("model") == selected_model_id
    ]
    selected_ablation_interpretation_rows = [
        row for row in ablation_interpretation_payload.get("rows") or [] if row.get("model") == selected_model_id
    ]
    selected_ablation_promotion_rows = [
        row for row in ablation_promotion_payload.get("rows") or [] if row.get("model") == selected_model_id
    ]
    selected_ablation_next_plan_rows = [
        row for row in ablation_next_plan_payload.get("rows") or [] if row.get("model") == selected_model_id
    ]
    selected_ablation_command_contract_rows = [
        row for row in ablation_command_contract_payload.get("rows") or [] if row.get("model") == selected_model_id
    ]
    selected_ablation_command_contract_smoke_rows = [
        row for row in ablation_command_contract_smoke_payload.get("rows") or [] if row.get("model") == selected_model_id
    ]
    selected_ablation_recommended_command_smoke_rows = [
        row for row in ablation_recommended_command_smoke_payload.get("rows") or [] if row.get("model") == selected_model_id
    ]
    selected_ablation_command_smoke_coverage_rows = [
        row for row in ablation_command_smoke_coverage_payload.get("rows") or [] if row.get("model") == selected_model_id
    ]
    selected_ablation_command_smoke_coverage_diff_rows = [
        row for row in ablation_command_smoke_coverage_diff_payload.get("rows") or [] if row.get("model") == selected_model_id
    ]
    if selected_ablation_gate_rows:
        review_rows.append(
            {
                "review_item": "Telemetry ablation gate",
                "status": selected_ablation_gate_rows[0].get("gate_decision", "review_required"),
                "evidence": "reports/deep_model_telemetry_ablation_gate_latest.json",
                "safe_talk_track": selected_ablation_gate_rows[0].get("interview_line", ""),
            }
        )
    if selected_ablation_interpretation_rows:
        support_rows = sum(
            1 for row in selected_ablation_interpretation_rows if str(row.get("verdict") or "").startswith("ablation_supports")
        )
        review_rows.append(
            {
                "review_item": "Ablation interpretation",
                "status": f"{support_rows}/{len(selected_ablation_interpretation_rows)} support rows",
                "evidence": "reports/deep_model_ablation_interpretation_latest.json",
                "safe_talk_track": selected_ablation_interpretation_rows[0].get("interview_line", ""),
            }
        )
    if selected_ablation_promotion_rows:
        best_promo = sorted(
            selected_ablation_promotion_rows,
            key=lambda row: int(row.get("promotion_score") or 0),
            reverse=True,
        )[0]
        review_rows.append(
            {
                "review_item": "Ablation promotion matrix",
                "status": f"{best_promo.get('claim_tier')} / {best_promo.get('promotion_decision')}",
                "evidence": "reports/deep_model_ablation_promotion_matrix_latest.json",
                "safe_talk_track": best_promo.get("interview_line", ""),
            }
        )
    if selected_ablation_next_plan_rows:
        first_plan = sorted(
            selected_ablation_next_plan_rows,
            key=lambda row: (str(row.get("priority") or "P9"), str(row.get("variant_id") or "")),
        )[0]
        review_rows.append(
            {
                "review_item": "Ablation next experiment",
                "status": f"{first_plan.get('priority')} / {first_plan.get('experiment_type')}",
                "evidence": "reports/deep_model_ablation_next_experiment_plan_latest.json",
                "safe_talk_track": first_plan.get("safe_interview_line", ""),
            }
        )
    if selected_ablation_command_contract_rows:
        fallback_ready = sum(1 for row in selected_ablation_command_contract_rows if row.get("fallback_ready"))
        first_contract = selected_ablation_command_contract_rows[0]
        review_rows.append(
            {
                "review_item": "Ablation command contract",
                "status": f"{fallback_ready}/{len(selected_ablation_command_contract_rows)} fallback ready",
                "evidence": "reports/deep_model_ablation_command_contract_latest.json",
                "safe_talk_track": first_contract.get("contract_interview_line", ""),
            }
        )
    if selected_ablation_recommended_command_smoke_rows:
        ok_rows = sum(1 for row in selected_ablation_recommended_command_smoke_rows if row.get("status") == "ok")
        first_recommended_smoke = selected_ablation_recommended_command_smoke_rows[0]
        review_rows.append(
            {
                "review_item": "Recommended command smoke",
                "status": f"{ok_rows}/{len(selected_ablation_recommended_command_smoke_rows)} runner-native rows ok",
                "evidence": "reports/deep_model_ablation_recommended_command_smoke_latest.json",
                "safe_talk_track": (
                    "Recommended commands now exercise native --models / --include-variants aliases with isolated output; "
                    f"sample manifest={first_recommended_smoke.get('runner_manifest', '')}."
                ),
            }
        )
    if selected_ablation_command_smoke_coverage_rows:
        covered_rows = sum(1 for row in selected_ablation_command_smoke_coverage_rows if row.get("any_smoked"))
        first_coverage = selected_ablation_command_smoke_coverage_rows[0]
        review_rows.append(
            {
                "review_item": "Command smoke coverage",
                "status": f"{covered_rows}/{len(selected_ablation_command_smoke_coverage_rows)} variants smoke-covered",
                "evidence": "reports/deep_model_ablation_command_smoke_coverage_latest.json",
                "safe_talk_track": first_coverage.get("next_action", ""),
            }
        )
    if selected_ablation_command_smoke_coverage_diff_rows:
        gain_rows = sum(1 for row in selected_ablation_command_smoke_coverage_diff_rows if int(row.get("smoke_gain") or 0) > 0)
        first_diff = selected_ablation_command_smoke_coverage_diff_rows[0]
        review_rows.append(
            {
                "review_item": "Command smoke coverage diff",
                "status": f"{gain_rows}/{len(selected_ablation_command_smoke_coverage_diff_rows)} variants gained smoke evidence",
                "evidence": "reports/deep_model_ablation_command_smoke_coverage_diff_latest.json",
                "safe_talk_track": first_diff.get("interview_claim", ""),
            }
        )
    def metric_text(value) -> str:
        if value is None:
            return "NA"
        try:
            return f"{float(value):.4g}"
        except (TypeError, ValueError):
            return str(value)

    review_metrics = [
        ("Paper Verdict", paper_scorecard.get("verdict", "NA"), True),
        ("Tuned Verdict", tuned_scorecard.get("verdict", "NA"), True),
        ("Best PEHE", metric_text(tuned_scorecard.get("best_pehe")), False),
        ("Best QINI", metric_text(tuned_scorecard.get("best_qini")), False),
        ("Battle Cards", f"{len(battle_rows_for_model):,}", False),
        ("Ablation Gates", f"{len(selected_ablation_gate_rows):,}", False),
        ("Ablation Rows", f"{len(selected_ablation_interpretation_rows):,}", False),
        ("Ablation Promo", f"{len(selected_ablation_promotion_rows):,}", True),
        ("Ablation Plan", f"{len(selected_ablation_next_plan_rows):,}", True),
        ("Command Contract", f"{len(selected_ablation_command_contract_rows):,}", True),
        ("Command Smoke", f"{len(selected_ablation_command_contract_smoke_rows):,}", True),
        ("Recommended Cmd", f"{len(selected_ablation_recommended_command_smoke_rows):,}", True),
        ("Smoke Coverage", f"{len(selected_ablation_command_smoke_coverage_rows):,}", True),
        ("Coverage Diff", f"{len(selected_ablation_command_smoke_coverage_diff_rows):,}", True),
        ("License Gate", "guarded" if any(key in model.license_policy.lower() for key in ("gpl", "unknown", "guarded", "do not copy")) else "clear", True),
    ]
    render_stat_grid(review_metrics, columns=6)
    st.dataframe(pd.DataFrame(review_rows), width="stretch", hide_index=True)
    if battle_rows_for_model:
        with st.expander("Selected Model Battle Cards", expanded=False):
            battle_df = pd.DataFrame(battle_rows_for_model)
            battle_cols = [
                col
                for col in [
                    "dataset_id",
                    "variant_id",
                    "verdict",
                    "pehe_mean",
                    "qini_mean",
                    "policy_top10_oracle_value_mean",
                    "train_loss_delta_mean",
                    "reason",
                    "interview_line",
                ]
                if col in battle_df.columns
            ]
            st.dataframe(battle_df[battle_cols], width="stretch", hide_index=True)

    st.subheader("Deep Model Objective Diagnostics")
    objective_cards = objective_payload.get("cards") or []
    selected_objective_card = next((row for row in objective_cards if row.get("model") == selected_model_id), None)
    if selected_objective_card:
        objective_rows = []
        for item in selected_objective_card.get("loss_terms") or []:
            objective_rows.append(
                {
                    "loss_term": item.get("term"),
                    "formula": item.get("formula"),
                    "observable_diagnostic": item.get("diagnostic"),
                    "failure_signal": item.get("failure_signal"),
                }
            )
        st.caption(
            "把公式/loss 拆成可观察诊断：factual loss 看拟合，IPM/propensity/interaction/cross-head 看因果结构，PEHE/QINI/policy value 看决策收益。"
        )
        render_stat_grid(
            [
                ("Loss Terms", f"{len(selected_objective_card.get('loss_terms') or [])}", False),
                ("Training", (selected_objective_card.get("diagnostics") or {}).get("training_status", "NA"), True),
                ("Tuned", (selected_objective_card.get("diagnostics") or {}).get("tuned_verdict", "NA"), True),
                ("Best PEHE", metric_text((selected_objective_card.get("diagnostics") or {}).get("best_pehe")), False),
                ("Best QINI", metric_text((selected_objective_card.get("diagnostics") or {}).get("best_qini")), False),
            ],
            columns=5,
        )
        st.dataframe(pd.DataFrame(objective_rows), width="stretch", hide_index=True)
        claim_df = pd.DataFrame(
            [
                {
                    "safe_claim": selected_objective_card.get("safe_claim"),
                    "unsafe_claim": selected_objective_card.get("unsafe_claim"),
                    "next_tasks": "; ".join(selected_objective_card.get("next_tasks") or []),
                    "observability_gaps": "; ".join(selected_objective_card.get("observability_gaps") or []),
                }
            ]
        )
        st.dataframe(claim_df, width="stretch", hide_index=True)
        doc_path = Path("docs/DEEPUplift_DEEP_MODEL_OBJECTIVE_DIAGNOSTICS.md")
        if doc_path.exists():
            st.download_button(
                "Download deep objective diagnostics",
                data=doc_path.read_bytes(),
                file_name=doc_path.name,
                mime="text/markdown",
                key=f"{key_prefix}_download_objective_diagnostics",
            )
    else:
        st.info("Run `scripts/generate_deep_model_objective_diagnostics.py` to refresh loss-term objective diagnostics.")

    st.subheader("Algorithm Claim Ledger")
    claim_rows_for_model = [
        row for row in claim_ledger_payload.get("rows") or [] if row.get("model") == selected_model_id
    ]
    if claim_rows_for_model:
        st.caption("每个可讲的模型结论都绑定到公式、代码、benchmark、license gate 和 safe/unsafe claim。")
        claim_df = pd.DataFrame(claim_rows_for_model)
        claim_cols = [
            col
            for col in [
                "claim_type",
                "evidence_level",
                "benchmark_verdict",
                "promotion_decision",
                "formula_or_test",
                "safe_claim",
                "unsafe_claim",
                "ui_route",
                "next_action",
            ]
            if col in claim_df.columns
        ]
        st.dataframe(claim_df[claim_cols], width="stretch", hide_index=True)
        doc_path = Path("docs/DEEPUplift_ALGORITHM_CLAIM_LEDGER.md")
        if doc_path.exists():
            st.download_button(
                "Download algorithm claim ledger",
                data=doc_path.read_bytes(),
                file_name=doc_path.name,
                mime="text/markdown",
                key=f"{key_prefix}_download_algorithm_claim_ledger",
            )
    else:
        st.info("Run `scripts/generate_algorithm_claim_ledger.py` to refresh model claim rows.")

    st.subheader("Paper Reproduction Gap Ledger")
    gap_rows_for_model = [
        row for row in reproduction_gap_payload.get("rows") or [] if row.get("model") == selected_model_id
    ]
    if gap_rows_for_model:
        st.caption("区分论文引用、开源参考、本地复现、benchmark 证据和 license 风险；避免把 citation 说成 full reproduction。")
        gap_df = pd.DataFrame(gap_rows_for_model)
        gap_cols = [
            col
            for col in [
                "source_name",
                "source_type",
                "source_license",
                "adoption_decision",
                "proof_level",
                "benchmark_verdict",
                "safe_claim",
                "unsafe_claim",
                "validation_needed",
            ]
            if col in gap_df.columns
        ]
        st.dataframe(gap_df[gap_cols], width="stretch", hide_index=True)
        doc_path = Path("docs/DEEPUplift_PAPER_REPRODUCTION_GAP_LEDGER.md")
        if doc_path.exists():
            st.download_button(
                "Download paper reproduction gap ledger",
                data=doc_path.read_bytes(),
                file_name=doc_path.name,
                mime="text/markdown",
                key=f"{key_prefix}_download_paper_reproduction_gap_ledger",
            )
    else:
        st.info("Run `scripts/generate_paper_reproduction_gap_ledger.py` to refresh paper/source gap rows.")

    st.subheader("Model Evidence Promotion Matrix")
    matrix_rows_for_model = [
        row for row in promotion_matrix_payload.get("rows") or [] if row.get("model") == selected_model_id
    ]
    if matrix_rows_for_model:
        st.caption("把这个模型当前能讲什么、不能讲什么、下一步补什么证据，压缩成一张评审卡。")
        matrix_df = pd.DataFrame(matrix_rows_for_model)
        matrix_cols = [
            col
            for col in [
                "evidence_stage",
                "promotion_tier",
                "evidence_score",
                "paper_proof_level",
                "paper_verdict",
                "tuned_verdict",
                "objective_status",
                "training_status",
                "license_gate",
                "launch_decision",
                "safe_claim",
                "blocked_claim",
                "next_evidence_step",
            ]
            if col in matrix_df.columns
        ]
        st.dataframe(matrix_df[matrix_cols], width="stretch", hide_index=True)
        doc_path = Path("docs/DEEPUplift_MODEL_EVIDENCE_PROMOTION_MATRIX.md")
        if doc_path.exists():
            st.download_button(
                "Download model evidence promotion matrix",
                data=doc_path.read_bytes(),
                file_name=doc_path.name,
                mime="text/markdown",
                key=f"{key_prefix}_download_model_evidence_promotion_matrix",
            )
    else:
        st.info("Run `scripts/generate_model_evidence_promotion_matrix.py` to refresh model promotion tiers.")

    st.subheader("Model Upgrade Recipe Cards")
    recipe_rows_for_model = [
        row for row in upgrade_recipe_payload.get("rows") or [] if row.get("model") == selected_model_id
    ]
    if recipe_rows_for_model:
        st.caption("这一行回答：下一步到底跑什么实验、看什么指标、什么条件算升级成功。")
        recipe_df = pd.DataFrame(recipe_rows_for_model)
        recipe_cols = [
            col
            for col in [
                "recipe_id",
                "upgrade_goal",
                "hypothesis",
                "command",
                "metrics_to_watch",
                "pass_gate",
                "fail_action",
                "expected_artifacts",
                "owner_role",
                "interview_line",
            ]
            if col in recipe_df.columns
        ]
        st.dataframe(recipe_df[recipe_cols], width="stretch", hide_index=True)
        doc_path = Path("docs/DEEPUplift_MODEL_UPGRADE_RECIPE_CARDS.md")
        if doc_path.exists():
            st.download_button(
                "Download model upgrade recipe cards",
                data=doc_path.read_bytes(),
                file_name=doc_path.name,
                mime="text/markdown",
                key=f"{key_prefix}_download_model_upgrade_recipe_cards",
            )
    else:
        st.info("Run `scripts/generate_model_upgrade_recipe_cards.py` to refresh executable model-upgrade recipes.")

    st.subheader("Deep Model Telemetry Smoke")
    telemetry_smoke_rows_for_model = [
        row for row in telemetry_smoke_payload.get("rows") or [] if row.get("model") == selected_model_id
    ]
    if telemetry_smoke_rows_for_model:
        st.caption("这一块回答：这个深度模型的基础 telemetry 是否真的能从 run artifacts 里读出来。")
        smoke_df = pd.DataFrame(telemetry_smoke_rows_for_model)
        smoke_cols = [
            col
            for col in [
                "status",
                "epochs",
                "train_loss_delta",
                "valid_loss_delta",
                "uplift_score_std",
                "top10_true_uplift_gain",
                "manifest_files",
                "advanced_hook_status",
                "advanced_hook_needed",
                "pass_gate",
                "fail_action",
                "interview_line",
            ]
            if col in smoke_df.columns
        ]
        st.dataframe(smoke_df[smoke_cols], width="stretch", hide_index=True)
        doc_path = Path("docs/DEEPUplift_DEEP_MODEL_TELEMETRY_SMOKE.md")
        if doc_path.exists():
            st.download_button(
                "Download deep model telemetry smoke",
                data=doc_path.read_bytes(),
                file_name=doc_path.name,
                mime="text/markdown",
                key=f"{key_prefix}_download_deep_model_telemetry_smoke",
            )
    else:
        st.info("Run `scripts/smoke_deep_model_telemetry.py` to refresh basic telemetry smoke for deep models.")

    st.subheader("Deep Model Forward Hook Contracts")
    hook_rows_for_model = [
        row for row in forward_hook_payload.get("rows") or [] if row.get("model") == selected_model_id
    ]
    if hook_rows_for_model:
        st.caption("这一块回答：下一步要改源码时，应该导出哪些 tensor、写哪些 artifact、怎么保持兼容、怎么验收。")
        hook_df = pd.DataFrame(hook_rows_for_model)
        hook_cols = [
            col
            for col in [
                "hook_name",
                "contract_type",
                "priority",
                "expected_tensor_keys",
                "artifact_files",
                "source_code_entry",
                "pass_gate",
                "fallback_behavior",
                "backward_compat",
                "interview_line",
            ]
            if col in hook_df.columns
        ]
        hook_table = hook_df[hook_cols].copy()
        for col in ["expected_tensor_keys", "artifact_files"]:
            if col in hook_table.columns:
                hook_table[col] = hook_table[col].apply(
                    lambda value: ", ".join(map(str, value)) if isinstance(value, list) else str(value)
                )
        st.dataframe(hook_table, width="stretch", hide_index=True)
        doc_path = Path("docs/DEEPUplift_DEEP_MODEL_FORWARD_HOOK_CONTRACTS.md")
        if doc_path.exists():
            st.download_button(
                "Download deep model forward hook contracts",
                data=doc_path.read_bytes(),
                file_name=doc_path.name,
                mime="text/markdown",
                key=f"{key_prefix}_download_deep_model_forward_hook_contracts",
            )
    else:
        st.info("Run `scripts/generate_deep_model_forward_hook_contracts.py` to refresh guarded forward-hook contracts.")

    st.subheader("Deep Model Forward Hook Smoke")
    hook_smoke_rows_for_model = [
        row for row in forward_hook_smoke_payload.get("rows") or [] if row.get("model") == selected_model_id
    ]
    if hook_smoke_rows_for_model:
        st.caption("这一块回答：契约里的高级 hook 是否已经能在小批 forward 上实际导出 sidecar artifacts。")
        st.caption(f"Artifact dir: `{forward_hook_smoke_payload.get('artifact_dir', 'NA')}`")
        hook_smoke_df = pd.DataFrame(hook_smoke_rows_for_model)
        hook_smoke_cols = [
            col
            for col in [
                "hook_name",
                "status",
                "tensor_keys_exported",
                "artifact_files",
                "metric_summary",
                "pass_gate",
                "remaining_gap",
                "interview_line",
            ]
            if col in hook_smoke_df.columns
        ]
        hook_smoke_table = hook_smoke_df[hook_smoke_cols].copy()
        for col in ["tensor_keys_exported", "artifact_files", "metric_summary"]:
            if col in hook_smoke_table.columns:
                hook_smoke_table[col] = hook_smoke_table[col].apply(
                    lambda value: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
                )
        st.dataframe(hook_smoke_table, width="stretch", hide_index=True)
        doc_path = Path("docs/DEEPUplift_DEEP_MODEL_FORWARD_HOOK_SMOKE.md")
        if doc_path.exists():
            st.download_button(
                "Download deep model forward hook smoke",
                data=doc_path.read_bytes(),
                file_name=doc_path.name,
                mime="text/markdown",
                key=f"{key_prefix}_download_deep_model_forward_hook_smoke",
            )
    else:
        st.info("Run `scripts/smoke_deep_model_forward_hooks.py` to refresh guarded forward-hook smoke artifacts.")

    st.subheader("Deep Model Hook Training Bridge")
    bridge_rows_for_model = [
        row for row in hook_training_bridge_payload.get("rows") or [] if row.get("model") == selected_model_id
    ]
    if bridge_rows_for_model:
        st.caption("这一块回答：已导出的内部 tensor 如何安全升级成 trainer-level / per-epoch evidence，而不破坏默认模型 API。")
        bridge_df = pd.DataFrame(bridge_rows_for_model)
        bridge_cols = [
            col
            for col in [
                "contract_id",
                "priority",
                "bridge_status",
                "proposed_epoch_columns",
                "proposed_epoch_artifacts",
                "pass_gate",
                "fail_action",
                "interview_line",
            ]
            if col in bridge_df.columns
        ]
        bridge_table = bridge_df[bridge_cols].copy()
        for col in ["proposed_epoch_columns", "proposed_epoch_artifacts"]:
            if col in bridge_table.columns:
                bridge_table[col] = bridge_table[col].apply(
                    lambda value: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
                )
        st.dataframe(bridge_table, width="stretch", hide_index=True)
        doc_path = Path("docs/DEEPUplift_DEEP_MODEL_HOOK_TRAINING_BRIDGE.md")
        if doc_path.exists():
            st.download_button(
                "Download deep model hook training bridge",
                data=doc_path.read_bytes(),
                file_name=doc_path.name,
                mime="text/markdown",
                key=f"{key_prefix}_download_deep_model_hook_training_bridge",
            )
    else:
        st.info("Run `scripts/generate_deep_model_hook_training_bridge.py` to refresh trainer-level hook bridge artifacts.")

    st.subheader("Deep Model Trainer Collector Smoke")
    collector_rows_for_model = [
        row for row in trainer_collector_payload.get("model_rows") or [] if row.get("model") == selected_model_id
    ]
    if collector_rows_for_model:
        st.caption("这一块回答：这个模型的高级 hook 是否已经能在训练循环里按 epoch 产出 telemetry sidecar。")
        st.caption(f"Artifact dir: `{trainer_collector_payload.get('artifact_dir', 'NA')}`")
        collector_df = pd.DataFrame(collector_rows_for_model)
        collector_cols = [
            col
            for col in [
                "status",
                "epochs",
                "contracts_covered",
                "collected_epoch_columns",
                "metric_last",
                "pass_gate",
                "blocked_claim",
                "interview_line",
            ]
            if col in collector_df.columns
        ]
        collector_table = collector_df[collector_cols].copy()
        for col in ["contracts_covered", "collected_epoch_columns", "metric_last"]:
            if col in collector_table.columns:
                collector_table[col] = collector_table[col].apply(
                    lambda value: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
                )
        st.dataframe(collector_table, width="stretch", hide_index=True)
        doc_path = Path("docs/DEEPUplift_DEEP_MODEL_TRAINER_COLLECTOR_SMOKE.md")
        if doc_path.exists():
            st.download_button(
                "Download deep model trainer collector smoke",
                data=doc_path.read_bytes(),
                file_name=doc_path.name,
                mime="text/markdown",
                key=f"{key_prefix}_download_deep_model_trainer_collector_smoke",
            )
    else:
        st.info("Run `scripts/smoke_deep_model_trainer_collectors.py` to refresh per-epoch trainer collector smoke artifacts.")

    st.subheader("Deep Model Telemetry-Benchmark Linkage")
    linkage_rows_for_model = [
        row for row in telemetry_benchmark_linkage_payload.get("rows") or [] if row.get("model") == selected_model_id
    ]
    if linkage_rows_for_model:
        st.caption("这一块回答：内部 telemetry 如何和 paper/tuned benchmark 胜负、baseline verdict、failure hypothesis、next ablation 连起来。")
        linkage_df = pd.DataFrame(linkage_rows_for_model)
        linkage_cols = [
            col
            for col in [
                "telemetry_family",
                "linkage_status",
                "paper_verdict",
                "tuned_verdict",
                "tuned_baseline_verdict",
                "telemetry_last",
                "failure_hypothesis",
                "next_ablation",
                "safe_claim",
                "blocked_claim",
                "interview_line",
            ]
            if col in linkage_df.columns
        ]
        linkage_table = linkage_df[linkage_cols].copy()
        for col in ["telemetry_last"]:
            if col in linkage_table.columns:
                linkage_table[col] = linkage_table[col].apply(
                    lambda value: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
                )
        st.dataframe(linkage_table, width="stretch", hide_index=True)
        doc_path = Path("docs/DEEPUplift_DEEP_MODEL_TELEMETRY_BENCHMARK_LINKAGE.md")
        if doc_path.exists():
            st.download_button(
                "Download deep model telemetry-benchmark linkage",
                data=doc_path.read_bytes(),
                file_name=doc_path.name,
                mime="text/markdown",
                key=f"{key_prefix}_download_deep_model_telemetry_benchmark_linkage",
            )
    else:
        st.info("Run `scripts/generate_deep_model_telemetry_benchmark_linkage.py` to refresh telemetry-to-benchmark linkage artifacts.")

    st.subheader("Deep Model Telemetry Ablation Gate")
    ablation_gate_rows_for_model = [
        row for row in telemetry_ablation_gate_payload.get("rows") or [] if row.get("model") == selected_model_id
    ]
    if ablation_gate_rows_for_model:
        st.caption("这一块回答：要把某个 loss/telemetry claim 从可观察升级成可证明，需要哪些消融 variant、pass gate 和 fail action。")
        gate_df = pd.DataFrame(ablation_gate_rows_for_model)
        gate_cols = [
            col
            for col in [
                "ablation_family",
                "gate_decision",
                "variant_coverage_status",
                "missing_variants",
                "metrics_to_watch",
                "pass_gate",
                "fail_action",
                "safe_claim",
                "blocked_claim",
                "interview_line",
            ]
            if col in gate_df.columns
        ]
        gate_table = gate_df[gate_cols].copy()
        for col in ["missing_variants", "metrics_to_watch"]:
            if col in gate_table.columns:
                gate_table[col] = gate_table[col].apply(
                    lambda value: ", ".join(str(item) for item in value)
                    if isinstance(value, list)
                    else str(value)
                )
        st.dataframe(gate_table, width="stretch", hide_index=True)
        doc_path = Path("docs/DEEPUplift_DEEP_MODEL_TELEMETRY_ABLATION_GATE.md")
        if doc_path.exists():
            st.download_button(
                "Download deep model telemetry ablation gate",
                data=doc_path.read_bytes(),
                file_name=doc_path.name,
                mime="text/markdown",
                key=f"{key_prefix}_download_deep_model_telemetry_ablation_gate",
            )
    else:
        st.info("Run `scripts/generate_deep_model_telemetry_ablation_gate.py` to refresh telemetry ablation gates.")

    st.subheader("Deep Model Ablation Interpretation")
    ablation_interpretation_rows_for_model = [
        row for row in ablation_interpretation_payload.get("rows") or [] if row.get("model") == selected_model_id
    ]
    tracks = ablation_interpretation_payload.get("interview_tracks") or {}
    if tracks:
        with st.expander("Ablation Interpretation / 面试讲法", expanded=False):
            st.dataframe(
                pd.DataFrame(
                    [
                        {"duration": "30s", "talk_track": tracks.get("30s", "")},
                        {"duration": "5min", "talk_track": tracks.get("5min", "")},
                        {"duration": "15min", "talk_track": tracks.get("15min", "")},
                        {"duration": "30min", "talk_track": tracks.get("30min", "")},
                    ]
                ),
                width="stretch",
                hide_index=True,
            )
    if ablation_interpretation_rows_for_model:
        st.caption("这一块回答：消融 variant 已经跑了之后，到底是 PEHE、QINI、policy 还是 loss 支持了机制 claim，哪些话仍然不能讲。")
        interp_df = pd.DataFrame(ablation_interpretation_rows_for_model)
        interp_cols = [
            col
            for col in [
                "dataset_id",
                "variant_id",
                "default_variant",
                "mechanism",
                "verdict",
                "delta_pehe_mean_vs_default",
                "delta_qini_mean_vs_default",
                "delta_policy_top10_oracle_value_mean_vs_default",
                "safe_claim",
                "blocked_claim",
                "interview_line",
            ]
            if col in interp_df.columns
        ]
        st.dataframe(interp_df[interp_cols], width="stretch", hide_index=True)
        doc_path = Path("docs/DEEPUplift_DEEP_MODEL_ABLATION_INTERPRETATION.md")
        if doc_path.exists():
            st.download_button(
                "Download deep model ablation interpretation",
                data=doc_path.read_bytes(),
                file_name=doc_path.name,
                mime="text/markdown",
                key=f"{key_prefix}_download_deep_model_ablation_interpretation",
            )
    else:
        st.info("Run `scripts/generate_deep_model_ablation_interpretation.py` to refresh deep-model ablation interpretation.")

    st.subheader("Deep Model Ablation Promotion Matrix")
    ablation_promotion_rows_for_model = [
        row for row in ablation_promotion_payload.get("rows") or [] if row.get("model") == selected_model_id
    ]
    promo_tracks = ablation_promotion_payload.get("interview_tracks") or {}
    if promo_tracks:
        with st.expander("Ablation Promotion / 面试讲法", expanded=False):
            st.dataframe(
                pd.DataFrame(
                    [
                        {"duration": "30s", "talk_track": promo_tracks.get("30s", "")},
                        {"duration": "5min", "talk_track": promo_tracks.get("5min", "")},
                        {"duration": "15min", "talk_track": promo_tracks.get("15min", "")},
                        {"duration": "30min", "talk_track": promo_tracks.get("30min", "")},
                    ]
                ),
                width="stretch",
                hide_index=True,
            )
    if ablation_promotion_rows_for_model:
        st.caption("这一块回答：消融结果到底能不能升级成面试 claim，variant 该不该替代 default，哪些话仍然必须 blocked。")
        promo_df = pd.DataFrame(ablation_promotion_rows_for_model)
        promo_cols = [
            col
            for col in [
                "variant_id",
                "claim_tier",
                "promotion_decision",
                "promotion_score",
                "support_rows",
                "tradeoff_rows",
                "baseline_still_stronger_rows",
                "recommended_claim_boundary",
                "safe_claim",
                "blocked_claim",
                "interview_line",
            ]
            if col in promo_df.columns
        ]
        st.dataframe(promo_df[promo_cols], width="stretch", hide_index=True)
        doc_path = Path("docs/DEEPUplift_DEEP_MODEL_ABLATION_PROMOTION_MATRIX.md")
        if doc_path.exists():
            st.download_button(
                "Download deep model ablation promotion matrix",
                data=doc_path.read_bytes(),
                file_name=doc_path.name,
                mime="text/markdown",
                key=f"{key_prefix}_download_deep_model_ablation_promotion_matrix",
            )
    else:
        st.info("Run `scripts/generate_deep_model_ablation_promotion_matrix.py` to refresh deep-model ablation promotion matrix.")

    st.subheader("Deep Model Ablation Next Experiment Plan")
    ablation_next_plan_rows_for_model = [
        row for row in ablation_next_plan_payload.get("rows") or [] if row.get("model") == selected_model_id
    ]
    next_plan_tracks = ablation_next_plan_payload.get("interview_tracks") or {}
    if next_plan_tracks:
        with st.expander("Ablation Next Experiments / 面试追问", expanded=False):
            st.dataframe(
                pd.DataFrame(
                    [
                        {"duration": "30s", "talk_track": next_plan_tracks.get("30s", "")},
                        {"duration": "5min", "talk_track": next_plan_tracks.get("5min", "")},
                        {"duration": "15min", "talk_track": next_plan_tracks.get("15min", "")},
                        {"duration": "30min", "talk_track": next_plan_tracks.get("30min", "")},
                    ]
                ),
                width="stretch",
                hide_index=True,
            )
    if ablation_next_plan_rows_for_model:
        st.caption("这一块回答：面试官追问下一步怎么优化时，具体跑什么、看什么、怎么通过和失败后怎么处理。")
        next_plan_df = pd.DataFrame(ablation_next_plan_rows_for_model)
        next_plan_cols = [
            col
            for col in [
                "priority",
                "variant_id",
                "experiment_type",
                "claim_tier",
                "recommended_command",
                "runnable_fallback_command",
                "command_contract_status",
                "metrics_to_watch",
                "pass_gate",
                "fail_action",
                "safe_interview_line",
            ]
            if col in next_plan_df.columns
        ]
        st.dataframe(next_plan_df[next_plan_cols], width="stretch", hide_index=True)
        doc_path = Path("docs/DEEPUplift_DEEP_MODEL_ABLATION_NEXT_EXPERIMENT_PLAN.md")
        if doc_path.exists():
            st.download_button(
                "Download deep model ablation next-experiment plan",
                data=doc_path.read_bytes(),
                file_name=doc_path.name,
                mime="text/markdown",
                key=f"{key_prefix}_download_deep_model_ablation_next_experiment_plan",
            )
    else:
        st.info("Run `scripts/generate_deep_model_ablation_next_experiment_plan.py` to refresh deep-model ablation next experiments.")

    st.subheader("Deep Model Ablation Command Contract")
    command_contract_tracks = ablation_command_contract_payload.get("interview_tracks") or {}
    if command_contract_tracks:
        with st.expander("Command Contract / 可执行命令边界", expanded=False):
            st.dataframe(
                pd.DataFrame(
                    [
                        {"duration": "30s", "talk_track": command_contract_tracks.get("30s", "")},
                        {"duration": "5min", "talk_track": command_contract_tracks.get("5min", "")},
                        {"duration": "15min", "talk_track": command_contract_tracks.get("15min", "")},
                        {"duration": "30min", "talk_track": command_contract_tracks.get("30min", "")},
                    ]
                ),
                width="stretch",
                hide_index=True,
            )
    if selected_ablation_command_contract_rows:
        st.caption("这一块回答：实验计划里的命令哪些是当前 runner 真能跑，哪些是未来 runner adapter backlog。")
        command_contract_df = pd.DataFrame(selected_ablation_command_contract_rows)
        command_contract_cols = [
            col
            for col in [
                "priority",
                "variant_id",
                "contract_status",
                "fallback_ready",
                "recommended_unsupported_flags",
                "runnable_fallback_command",
                "contract_interview_line",
            ]
            if col in command_contract_df.columns
        ]
        st.dataframe(command_contract_df[command_contract_cols], width="stretch", hide_index=True)
        doc_path = Path("docs/DEEPUplift_DEEP_MODEL_ABLATION_COMMAND_CONTRACT.md")
        if doc_path.exists():
            st.download_button(
                "Download deep model ablation command contract",
                data=doc_path.read_bytes(),
                file_name=doc_path.name,
                mime="text/markdown",
                key=f"{key_prefix}_download_deep_model_ablation_command_contract",
            )
    else:
        st.info("Run `scripts/generate_deep_model_ablation_command_contract.py` to refresh deep-model ablation command contracts.")

    st.subheader("Deep Model Ablation Command Contract Smoke")
    if selected_ablation_command_contract_smoke_rows:
        st.caption("这一块回答：fallback command 是否真的端到端执行过，以及是否隔离输出不覆盖主 benchmark latest。")
        command_smoke_df = pd.DataFrame(selected_ablation_command_contract_smoke_rows)
        command_smoke_cols = [
            col
            for col in [
                "variant_id",
                "status",
                "runs",
                "ok_runs",
                "runner_manifest",
                "runner_json",
                "command",
            ]
            if col in command_smoke_df.columns
        ]
        st.dataframe(command_smoke_df[command_smoke_cols], width="stretch", hide_index=True)
        doc_path = Path("docs/DEEPUplift_DEEP_MODEL_ABLATION_COMMAND_CONTRACT_SMOKE.md")
        if doc_path.exists():
            st.download_button(
                "Download deep model ablation command contract smoke",
                data=doc_path.read_bytes(),
                file_name=doc_path.name,
                mime="text/markdown",
                key=f"{key_prefix}_download_deep_model_ablation_command_contract_smoke",
            )
    else:
        st.info("Run `scripts/smoke_deep_model_ablation_command_contract.py` to execute fallback command smoke rows.")

    st.subheader("Deep Model Ablation Recommended Command Smoke")
    if selected_ablation_recommended_command_smoke_rows:
        st.caption("这一块回答：recommended command 是否已经变成 runner-native，而不是只靠 fallback `--variants` 命令兜底。")
        recommended_command_smoke_df = pd.DataFrame(selected_ablation_recommended_command_smoke_rows)
        recommended_command_smoke_cols = [
            col
            for col in [
                "priority",
                "variant_id",
                "status",
                "uses_models_alias",
                "uses_include_variants_alias",
                "uses_emit_battle_cards_alias",
                "runs",
                "ok_runs",
                "runner_manifest",
                "safe_smoke_command",
            ]
            if col in recommended_command_smoke_df.columns
        ]
        st.dataframe(recommended_command_smoke_df[recommended_command_smoke_cols], width="stretch", hide_index=True)
        doc_path = Path("docs/DEEPUplift_DEEP_MODEL_ABLATION_RECOMMENDED_COMMAND_SMOKE.md")
        if doc_path.exists():
            st.download_button(
                "Download deep model ablation recommended command smoke",
                data=doc_path.read_bytes(),
                file_name=doc_path.name,
                mime="text/markdown",
                key=f"{key_prefix}_download_deep_model_ablation_recommended_command_smoke",
            )
    else:
        st.info("Run `scripts/smoke_deep_model_ablation_recommended_commands.py` to execute recommended command smoke rows.")

    st.subheader("Deep Model Ablation Command Smoke Coverage")
    if selected_ablation_command_smoke_coverage_rows:
        st.caption("这一块回答：当前模型的 ablation command 里哪些已经 smoke 执行、哪些只是 runner-supported backlog，避免把计划讲成结果。")
        coverage_df = pd.DataFrame(selected_ablation_command_smoke_coverage_rows)
        coverage_cols = [
            col
            for col in [
                "priority",
                "variant_id",
                "contract_status",
                "fallback_smoked",
                "recommended_smoked",
                "coverage_status",
                "recommended_runner_aliases",
                "recommended_smoke_aliases",
                "next_action",
            ]
            if col in coverage_df.columns
        ]
        st.dataframe(coverage_df[coverage_cols], width="stretch", hide_index=True)
        doc_path = Path("docs/DEEPUplift_DEEP_MODEL_ABLATION_COMMAND_SMOKE_COVERAGE.md")
        if doc_path.exists():
            st.download_button(
                "Download deep model ablation command smoke coverage",
                data=doc_path.read_bytes(),
                file_name=doc_path.name,
                mime="text/markdown",
                key=f"{key_prefix}_download_deep_model_ablation_command_smoke_coverage",
            )
    else:
        st.info("Run `scripts/generate_deep_model_ablation_command_smoke_coverage.py` to refresh command smoke coverage.")

    st.subheader("Deep Model Ablation Command Smoke Coverage Diff")
    if selected_ablation_command_smoke_coverage_diff_rows:
        st.caption("这一块回答：这个模型的 ablation 命令覆盖率在本轮优化中从哪里提升到哪里，哪些 claim 能讲、哪些还要等 benchmark。")
        diff_df = pd.DataFrame(selected_ablation_command_smoke_coverage_diff_rows)
        diff_cols = [
            col
            for col in [
                "priority",
                "variant_id",
                "baseline_fallback_smoked",
                "current_any_smoked",
                "runner_native_recommended_smoked",
                "transition",
                "interview_claim",
                "next_action",
            ]
            if col in diff_df.columns
        ]
        st.dataframe(diff_df[diff_cols], width="stretch", hide_index=True)
        doc_path = Path("docs/DEEPUplift_DEEP_MODEL_ABLATION_COMMAND_SMOKE_COVERAGE_DIFF.md")
        if doc_path.exists():
            st.download_button(
                "Download deep model ablation command smoke coverage diff",
                data=doc_path.read_bytes(),
                file_name=doc_path.name,
                mime="text/markdown",
                key=f"{key_prefix}_download_deep_model_ablation_command_smoke_coverage_diff",
            )
    else:
        st.info("Run `scripts/generate_deep_model_ablation_command_smoke_coverage_diff.py` to refresh command smoke coverage diff.")

    st.subheader("Deep Model Telemetry Gap Matrix")
    telemetry_rows_for_model = [
        row for row in telemetry_gap_payload.get("rows") or [] if row.get("model") == selected_model_id
    ]
    if telemetry_rows_for_model:
        st.caption("这一块回答：深度模型的 loss term 现在能观测到什么，还缺什么 telemetry，应该在哪个 hook 上补。")
        telemetry_df = pd.DataFrame(telemetry_rows_for_model)
        telemetry_cols = [
            col
            for col in [
                "loss_term",
                "telemetry_family",
                "priority",
                "current_signal",
                "missing_telemetry",
                "implementation_hook",
                "pass_gate",
                "fail_action",
                "expected_metric_link",
                "interview_line",
            ]
            if col in telemetry_df.columns
        ]
        st.dataframe(telemetry_df[telemetry_cols], width="stretch", hide_index=True)
        doc_path = Path("docs/DEEPUplift_DEEP_MODEL_TELEMETRY_GAP_MATRIX.md")
        if doc_path.exists():
            st.download_button(
                "Download deep model telemetry gap matrix",
                data=doc_path.read_bytes(),
                file_name=doc_path.name,
                mime="text/markdown",
                key=f"{key_prefix}_download_deep_model_telemetry_gap_matrix",
            )
    else:
        st.info("Run `scripts/generate_deep_model_telemetry_gap_matrix.py` to refresh deep-model loss telemetry gaps.")

    flow_cards = [
        ("Source", model.code_paths[0] if model.code_paths else "", "local file / function entry"),
        ("Objective", model.formulas[0] if model.formulas else "", "causal quantity and loss signal"),
        ("Train", model.train_flow[0] if model.train_flow else "", "fit or forward evidence path"),
        ("Evaluate", ", ".join(model.eval_metrics[:3]), "shared uplift evaluator contract"),
        ("Evidence", model.smoke_evidence[0] if model.smoke_evidence else "", "smoke/report proof"),
    ]
    st.markdown(
        "<div class='du-flow-stack'>"
        + "".join(
            "<div class='du-flow-step'>"
            f"<span>{html.escape(kicker)}</span>"
            f"<b>{html.escape(title)}</b>"
            f"<small>{html.escape(meta)}</small>"
            "</div>"
            for kicker, title, meta in flow_cards
        )
        + "</div>",
        unsafe_allow_html=True,
    )

    st.subheader("Deep Model Training Evidence")
    if training_rows:
        training_df = pd.DataFrame(training_rows)
        training_cols = [
            col
            for col in [
                "model",
                "status",
                "qini_score",
                "auuc_score",
                "oracle_top10_recall",
                "train_loss_last",
                "aux_loss_last",
                "run_id",
                "evidence_manifest",
                "interview_line",
            ]
            if col in training_df.columns
        ]
        st.caption("代表性深度 uplift 模型现在有同一训练数据、同一 evaluator、同一 artifact manifest 下的 smoke evidence。")
        st.dataframe(training_df[training_cols], width="stretch", hide_index=True)
        selected_training_row = next((row for row in training_rows if row.get("model") == selected_model_id), None)
        if selected_training_row:
            metric_cols = st.columns(4)
            metric_items = [
                ("QINI", selected_training_row.get("qini_score")),
                ("AUUC", selected_training_row.get("auuc_score")),
                ("Top10 Oracle", selected_training_row.get("oracle_top10_recall")),
                ("Aux/IPM Loss", selected_training_row.get("aux_loss_last")),
            ]
            for col, (label, value) in zip(metric_cols, metric_items):
                col.metric(label, "NA" if value is None else f"{float(value):.4g}")
            history_df = pd.DataFrame(selected_training_row.get("history") or [])
            curve_cols = [
                col
                for col in ["train_loss", "valid_loss", "train_outcome_loss", "train_treatment_loss"]
                if col in history_df.columns
            ]
            if curve_cols:
                st.line_chart(history_df.set_index("epoch")[curve_cols], height=260, width="stretch")
            st.caption(selected_training_row.get("interview_line", ""))
        else:
            st.info("Deep training evidence currently covers CFRNet, DragonNet, EFIN and DESCN; meta learner and multi-treatment rows remain in benchmark/evidence artifacts.")
    else:
        st.info("Run `scripts/smoke_deep_model_training_evidence.py` to refresh deep model training leaderboard and loss curves.")

    st.subheader("Deep Model Battle Report")
    if tuned_rows:
        st.caption(
            "把 smoke evidence 再升级一层：同一 known-CATE 数据上比较 baseline、tuned preset 和 ablation，并把失败原因做成可面试讲解的 battle card。"
        )
        render_deep_model_tuned_dashboard(
            tuned_payload,
            tuned_diff_payload,
            selected_model_id=selected_model_id,
            key_prefix=f"{key_prefix}_deep_tuned_benchmark",
        )
    else:
        st.info("Run `scripts/run_deep_model_tuned_benchmark.py` to generate tuned deep-model battle evidence.")

    st.subheader("Paper-Level Benchmark Evidence")
    if paper_rows:
        st.caption(
            "把 smoke evidence 升级到 paper-level：IHDP + ACIC-style + synthetic known-CATE，用真实 CATE 计算 PEHE/ATE，"
            "同时保留 QINI/AUUC/policy value 与每次训练的 evidence manifest。"
        )
        render_paper_benchmark_dashboard(
            paper_payload,
            paper_diff_payload,
            paper_interpretation_payload,
            selected_model_id=selected_model_id,
            key_prefix=f"{key_prefix}_paper_benchmark",
        )
    else:
        st.info("Run `scripts/run_paper_benchmark_suite.py` to generate PEHE/ATE/QINI/AUUC leaderboard evidence.")

    st.subheader("Source Call Chain")
    graph_nodes = [
        ("data", "Data / features"),
        ("preprocess", "preprocess"),
        ("fit", "fit / forward"),
        ("loss", "loss or pseudo outcome"),
        ("predict", "predict uplift"),
        ("eval", "QINI / AUUC / policy"),
        ("evidence", "artifacts / docs"),
    ]
    graph_edges = [("data", "preprocess"), ("preprocess", "fit"), ("fit", "loss"), ("loss", "predict"), ("predict", "eval"), ("eval", "evidence")]
    graph_lines = [
        'digraph { graph [rankdir=LR, bgcolor="transparent", pad="0.2", nodesep="0.35", ranksep="0.45"];',
        'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, color="#2dd4bf", fillcolor="#111827", fontcolor="#f8fafc"];',
        'edge [color="#38bdf8", fontname="Helvetica", fontsize=9, fontcolor="#94a3b8"];',
    ]
    for node_id, label in graph_nodes:
        graph_lines.append(f'{node_id} [label="{_graphviz_label(label)}"];')
    for left, right in graph_edges:
        graph_lines.append(f"{left} -> {right};")
    graph_lines.append("}")
    st.graphviz_chart("\n".join(graph_lines), width="stretch")

    c1, c2 = st.columns([0.52, 0.48])
    with c1:
        st.subheader("Code / Math / Flow")
        source_flow_rows = []
        for stage, values in [
            ("source_code", model.code_paths),
            ("formula", model.formulas),
            ("call_chain", model.call_chain),
            ("data_flow", model.data_flow),
            ("train_flow", model.train_flow),
            ("predict_flow", model.predict_flow),
        ]:
            for item in values:
                source_flow_rows.append({"stage": stage, "detail": item})
        st.dataframe(pd.DataFrame(source_flow_rows), width="stretch", hide_index=True)
    with c2:
        st.subheader("Evidence / Risk / Next")
        evidence_rows = []
        for item in model.smoke_evidence:
            evidence_rows.append({"type": "smoke_or_report", "detail": item})
        for item in model.failure_modes:
            evidence_rows.append({"type": "failure_mode", "detail": item})
        for item in model.next_tasks:
            evidence_rows.append({"type": "next_task", "detail": item})
        st.dataframe(pd.DataFrame(evidence_rows), width="stretch", hide_index=True)

    st.subheader("External Sources And License Gate")
    source_df = pd.DataFrame(
        [
            {
                "source": source.name,
                "type": source.source_type,
                "license": source.license,
                "license_risk": source.license_risk,
                "adoption": source.adoption,
                "url": source.url,
            }
            for source in model.external_sources
        ]
    )
    st.dataframe(source_df, width="stretch", hide_index=True)
    st.caption(model.license_policy)

    st.subheader("Golden Question Regression")
    if agent_rows:
        regression_df = pd.DataFrame(agent_rows)
        selected_rows = regression_df[regression_df["model"] == selected_model_id] if "model" in regression_df.columns else regression_df
        st.dataframe(selected_rows, width="stretch", hide_index=True)
    else:
        st.info("Run `scripts/smoke_model_deconstruction_agent.py` to refresh source-level golden question evidence.")

    st.subheader("Model Deconstruction Agent")
    default_prompt = f"从源码拆解 {selected_model_id} 的 loss / fit / predict，并说明怎么训练评估"
    prompt = st.text_area(
        "Ask a source-level model question",
        value=default_prompt,
        height=88,
        key=f"{key_prefix}_agent_prompt",
    )
    if st.button("Ask Model Deconstruction Agent", key=f"{key_prefix}_ask_agent"):
        reply = model_deconstruction_reply(
            prompt,
            {
                "model_name": current_model or selected_model_id,
                "latest_run_dir": (st.session_state.get("uplift_result") or None).run_dir
                if st.session_state.get("uplift_result") is not None
                else "",
            },
        )
        st.session_state[f"{key_prefix}_last_reply"] = reply or "Model Deconstruction Agent did not match this prompt."
    if st.session_state.get(f"{key_prefix}_last_reply"):
        st.markdown(st.session_state[f"{key_prefix}_last_reply"])

    doc_path = Path("docs/model_deconstruction") / f"{selected_model_id}.md"
    index_path = Path("docs/DEEPUplift_MODEL_DECONSTRUCTION_INDEX.md")
    download_cols = st.columns(4)
    for index, (label, path) in enumerate(
        [
            ("Selected doc", doc_path),
            ("Index doc", index_path),
            ("Catalog JSON", Path((artifacts.get("model_deconstruction_catalog") or {}).get("path", ""))),
            ("Agent report", Path((artifacts.get("model_deconstruction_agent") or {}).get("path", ""))),
            ("Training evidence", Path((artifacts.get("deep_model_training_evidence") or {}).get("path", ""))),
            ("Deep tuned benchmark", Path((artifacts.get("deep_model_tuned_benchmark") or {}).get("path", ""))),
            ("Deep tuned leaderboard", Path((artifacts.get("deep_model_tuned_benchmark_leaderboard") or {}).get("path", ""))),
            ("Deep tuned battle cards", Path((artifacts.get("deep_model_tuned_benchmark_battle_cards") or {}).get("path", ""))),
            ("Deep tuned manifest", Path((artifacts.get("deep_model_tuned_benchmark_manifest") or {}).get("path", ""))),
            ("Ablation interpretation", Path((artifacts.get("deep_model_ablation_interpretation") or {}).get("path", ""))),
            ("Ablation interpretation doc", Path("docs/DEEPUplift_DEEP_MODEL_ABLATION_INTERPRETATION.md")),
            ("Ablation promotion", Path((artifacts.get("deep_model_ablation_promotion_matrix") or {}).get("path", ""))),
            ("Ablation promotion doc", Path("docs/DEEPUplift_DEEP_MODEL_ABLATION_PROMOTION_MATRIX.md")),
            ("Ablation next plan", Path((artifacts.get("deep_model_ablation_next_experiment_plan") or {}).get("path", ""))),
            ("Ablation next plan doc", Path("docs/DEEPUplift_DEEP_MODEL_ABLATION_NEXT_EXPERIMENT_PLAN.md")),
            ("Ablation command contract", Path((artifacts.get("deep_model_ablation_command_contract") or {}).get("path", ""))),
            ("Ablation command contract doc", Path("docs/DEEPUplift_DEEP_MODEL_ABLATION_COMMAND_CONTRACT.md")),
            ("Ablation command contract smoke", Path((artifacts.get("deep_model_ablation_command_contract_smoke") or {}).get("path", ""))),
            ("Ablation command contract smoke doc", Path("docs/DEEPUplift_DEEP_MODEL_ABLATION_COMMAND_CONTRACT_SMOKE.md")),
            ("Ablation recommended command smoke", Path((artifacts.get("deep_model_ablation_recommended_command_smoke") or {}).get("path", ""))),
            ("Ablation recommended command smoke doc", Path("docs/DEEPUplift_DEEP_MODEL_ABLATION_RECOMMENDED_COMMAND_SMOKE.md")),
            ("Ablation command smoke coverage", Path((artifacts.get("deep_model_ablation_command_smoke_coverage") or {}).get("path", ""))),
            ("Ablation command smoke coverage doc", Path("docs/DEEPUplift_DEEP_MODEL_ABLATION_COMMAND_SMOKE_COVERAGE.md")),
            ("Ablation command smoke coverage diff", Path((artifacts.get("deep_model_ablation_command_smoke_coverage_diff") or {}).get("path", ""))),
            ("Ablation command smoke coverage diff doc", Path("docs/DEEPUplift_DEEP_MODEL_ABLATION_COMMAND_SMOKE_COVERAGE_DIFF.md")),
            ("Paper benchmark", Path((artifacts.get("paper_benchmark") or {}).get("path", ""))),
            ("Paper leaderboard", Path((artifacts.get("paper_benchmark_leaderboard") or {}).get("path", ""))),
            ("Paper manifest", Path((artifacts.get("paper_benchmark_manifest") or {}).get("path", ""))),
            ("Benchmark interpretation", Path((artifacts.get("paper_benchmark_interpretation") or {}).get("path", ""))),
            ("Benchmark interpretation doc", Path("docs/DEEPUplift_BENCHMARK_V3_INTERPRETATION.md")),
            ("Deep tuned benchmark doc", Path("docs/DEEPUplift_DEEP_MODEL_TUNING_BENCHMARK.md")),
        ]
    ):
        with download_cols[index % len(download_cols)]:
            if path.is_file():
                st.download_button(
                    f"Download {label}",
                    data=path.read_bytes(),
                    file_name=path.name,
                    mime=artifact_mime(path),
                    key=f"{key_prefix}_download_{path.name}",
                )
    if doc_path.exists():
        with st.expander("Selected Markdown Deconstruction", expanded=False):
            doc_text = doc_path.read_text(encoding="utf-8")
            st.markdown(doc_text[:14000])
            if len(doc_text) > 14000:
                st.caption("Preview truncated in UI; use the download button for the full Markdown.")
    if catalog_payload.get("coverage"):
        st.caption("Catalog coverage: " + json.dumps(catalog_payload.get("coverage"), ensure_ascii=False))


def render_scenario_model_decision_ladder() -> None:
    st.subheader("Scenario-to-Model Decision Ladder")
    st.caption(
        "工业落地不是从模型名字开始，而是从 treatment design、业务成本、反馈窗口和实验风险开始。"
        "每个场景都按 P0 baseline -> P1 plugin -> P2 extension 给出升级路线。"
    )
    rows = pd.DataFrame(scenario_model_ladders())
    if rows.empty:
        return
    render_stat_grid(
        [
            ("Scenarios", f"{len(rows):,}", False),
            ("P0 First", "Meta/DR/DML", True),
            ("P1 Plugins", "UTBoost + Deep", True),
            ("P2 Extensions", "Delayed + Funnel + Bandit", True),
        ],
        columns=4,
    )
    st.dataframe(
        rows[
            [
                "scenario",
                "treatment_design",
                "p0_first",
                "p1_candidate",
                "p2_extension",
                "evaluation_focus",
                "policy_focus",
                "go_no_go_gate",
                "ui_evidence",
            ]
        ],
        width="stretch",
        hide_index=True,
    )
    st.download_button(
        "Download scenario-to-model decision ladder",
        data=dataframe_csv_bytes(rows),
        file_name="deepuplift_scenario_model_decision_ladder.csv",
        mime="text/csv",
    )


def frontier_metric_reading_rows() -> list[dict[str, str]]:
    return [
        {
            "field": "full_funnel_top10_conversion_uplift",
            "scenario": "ECUP / ads / recommendation / coupon funnel",
            "how_to_read": "Top 10% audience conversion-stage observed uplift; read it together with click and impression stage uplift.",
            "deployment_implication": "If conversion uplift is positive but click/exposure uplift is weak, check attribution, stage selection bias and delayed conversion before scaling.",
        },
        {
            "field": "full_funnel_top10_click_uplift",
            "scenario": "ECUP / creative and exposure optimization",
            "how_to_read": "Top 10% click-stage uplift; shows whether the policy changes intent before final conversion.",
            "deployment_implication": "Useful for ads/recommendation debugging; do not optimize clicks alone if conversion or profit is negative.",
        },
        {
            "field": "delayed_d30_top10_uplift",
            "scenario": "CRM / push / ads delayed conversion",
            "how_to_read": "Top 10% D30 observed uplift; compares mature-window impact against fast-window labels.",
            "deployment_implication": "If D1/D7 is weak but D30 is positive, avoid prematurely rejecting slow-response campaigns.",
        },
        {
            "field": "delayed_censored_rate",
            "scenario": "Delayed feedback / lifecycle decisions",
            "how_to_read": "Share of holdout rows whose conversion outcome is not mature in the chosen window.",
            "deployment_implication": "High censoring means QINI/AUUC and policy value should be treated as exploratory until labels mature or censoring is modeled.",
        },
    ]


def evaluation_metric_rows() -> list[dict[str, str]]:
    return [
        {
            "metric": "QINI",
            "definition": "Sort users by predicted uplift and compare cumulative incremental gain against a random targeting baseline.",
            "business_meaning": "answers whether the ranking finds persuadable users earlier than random targeting",
            "when_to_use": "primary ranking signal for binary treatment uplift compare",
            "risk": "can look good while Top-K ROI is poor; unstable when treatment/control support is weak",
            "ui_artifact": "Evaluation: QINI curve; Compare: qini / qini_ci_low",
            "artifact_source": "runs/{run_id}/metrics.json:qini_score; runs/{run_id}/curves.json:qini",
            "code_source": "deepuplift/core/evaluator.py",
        },
        {
            "metric": "AUUC",
            "definition": "Area under the uplift curve over targeting fractions.",
            "business_meaning": "summarizes cumulative incremental response across different audience sizes",
            "when_to_use": "compare broad ranking quality and validate that lift is not limited to one tiny bucket",
            "risk": "less cost-aware than policy value and can hide bad early Top-K behavior",
            "ui_artifact": "Evaluation: AUUC curve; Compare: auuc",
            "artifact_source": "runs/{run_id}/metrics.json:auuc_score; runs/{run_id}/curves.json:auuc",
            "code_source": "deepuplift/core/evaluator.py",
        },
        {
            "metric": "uplift@K / Top-K observed uplift",
            "definition": "Observed treatment-control outcome difference inside the top scored fraction.",
            "business_meaning": "approximates the audience bucket a growth, coupon or ads team would actually target",
            "when_to_use": "choose Top 5/10/20 percent launch buckets and debug ranking monotonicity",
            "risk": "high variance in small buckets; needs bootstrap and minimum treated/control counts",
            "ui_artifact": "Evaluation: Top-K table; Policy: threshold curve",
            "artifact_source": "runs/{run_id}/metrics.json:top_k; runs/{run_id}/curves.json:bins",
            "code_source": "deepuplift/core/evaluator.py",
        },
        {
            "metric": "policy value",
            "definition": "Expected or observed net value after applying uplift score, conversion value and contact cost.",
            "business_meaning": "turns uplift into coupon ROI, ad iROAS, push fatigue-adjusted value or LLM routing net value",
            "when_to_use": "final threshold and budget decision after model ranking is acceptable",
            "risk": "depends on business value/cost assumptions; should be validated with holdout or online experiment",
            "ui_artifact": "Evaluation: Policy Value; Policy: Scenario ROI Lab",
            "artifact_source": "runs/{run_id}/metrics.json:policy_best; runs/{run_id}/curves.json:policy_value",
            "code_source": "deepuplift/core/policy.py",
        },
        {
            "metric": "incremental profit / cost-aware gain",
            "definition": "Estimate incremental outcome value minus treatment, subsidy, ad, messaging, or LLM escalation cost.",
            "business_meaning": "connects uplift ranking to gross profit, iROAS, fatigue-adjusted CRM value and cost-quality routing",
            "when_to_use": "coupon, ads, growth and LLM routing decisions where each treatment has a real marginal cost",
            "risk": "wrong margin, cost, cannibalization or fatigue assumptions can reverse the recommended Top-K threshold",
            "ui_artifact": "Policy: Scenario ROI Lab; Evaluation: metric map; Evidence: policy artifacts",
            "artifact_source": "runs/{run_id}/metrics.json:policy_best; policy scenario inputs",
            "code_source": "deepuplift/core/policy.py; app.py:Policy",
        },
        {
            "metric": "delayed feedback uplift",
            "definition": "Evaluate uplift over explicit conversion windows such as D1/D7/D14/D30 and track censoring risk.",
            "business_meaning": "prevents fast-window metrics from undercounting slow responders in ads, retention and lifecycle CRM",
            "when_to_use": "delayed purchase, ad attribution, renewal, retention or any outcome that matures after treatment",
            "risk": "future leakage, right censoring and inconsistent label windows can make offline uplift look falsely strong",
            "ui_artifact": "Evaluation: metric map; Models: CFR-DF / delayed feedback capability",
            "artifact_source": "runs/{run_id}/metrics.json:delayed_feedback; runs/{run_id}/curves.json:delayed_feedback_top_k",
            "code_source": "deepuplift/core/evaluator.py:delayed_feedback_uplift_metrics; deepuplift/core/frontier_models.py",
        },
        {
            "metric": "full-funnel ECUP metric",
            "definition": "Track treatment effect through impression, click and conversion stages instead of a single terminal label.",
            "business_meaning": "ads and recommendation teams can see whether uplift comes from exposure, click intent or downstream conversion",
            "when_to_use": "impression-click-conversion logs, ad ranking, recommendation exposure and coupon funnel optimization",
            "risk": "stage selection bias and delayed conversion can hide where the policy actually creates incremental value",
            "ui_artifact": "Evaluation: metric map; Models: ECUP / full-funnel uplift capability",
            "artifact_source": "runs/{run_id}/metrics.json:full_funnel; runs/{run_id}/curves.json:full_funnel_top_k",
            "code_source": "deepuplift/core/evaluator.py:full_funnel_uplift_metrics; deepuplift/core/frontier_models.py",
        },
        {
            "metric": "calibration",
            "definition": "Compare predicted uplift buckets with observed uplift buckets.",
            "business_meaning": "checks whether higher model scores correspond to higher actual incremental response",
            "when_to_use": "before trusting thresholds, segment explanations or budget pacing",
            "risk": "bucket estimates are noisy; good calibration does not remove selection bias",
            "ui_artifact": "Evaluation: Uplift Calibration; readiness: calibration check",
            "artifact_source": "runs/{run_id}/metrics.json:calibration_mae; runs/{run_id}/curves.json:calibration",
            "code_source": "deepuplift/core/evaluator.py; deepuplift/core/readiness.py",
        },
        {
            "metric": "bootstrap CI",
            "definition": "Repeatedly resample holdout rows and recompute metrics to estimate confidence intervals.",
            "business_meaning": "shows whether a model win is stable enough to explain or deploy",
            "when_to_use": "model compare, Top-K launch decision and policy value stability checks",
            "risk": "wide or negative lower bounds should downgrade decisions even when mean metric is high",
            "ui_artifact": "Evaluation: bootstrap caption; Compare: qini_ci_low/high",
            "artifact_source": "runs/{run_id}/metrics.json:bootstrap",
            "code_source": "deepuplift/core/evaluator.py; deepuplift/core/readiness.py",
        },
        {
            "metric": "overlap / propensity",
            "definition": "Estimate treatment assignment probability and check common support between treatment/control.",
            "business_meaning": "tests whether treatment and control users are comparable enough for causal claims",
            "when_to_use": "before model selection and again after training via overlap trimming",
            "risk": "weak overlap means policy may extrapolate into unsupported users",
            "ui_artifact": "Diagnostics: overlap; Evaluation: Overlap Trimming",
            "artifact_source": "runs/{run_id}/metrics.json:overlap_trim",
            "code_source": "deepuplift/core/diagnostics.py; deepuplift/core/evaluator.py",
        },
        {
            "metric": "sensitivity / placebo",
            "definition": "Compare observed metrics to random-score or permuted-treatment null distributions.",
            "business_meaning": "guards against metric artifacts and accidental leakage-driven ranking",
            "when_to_use": "before trusting a surprising compare winner or interview demo result",
            "risk": "not a proof of causality; it is a regression guard and sanity check",
            "ui_artifact": "Evaluation: Sensitivity Checks; Evidence: Latest Training Smoke",
            "artifact_source": "runs/{run_id}/metrics.json:sensitivity; reports/agent_regression_*.json",
            "code_source": "deepuplift/core/evaluator.py",
        },
        {
            "metric": "oracle top-k recall",
            "definition": "When true_uplift / ite / cate exists, measure whether predicted Top-K captures true high-effect users.",
            "business_meaning": "useful for synthetic data, simulation and benchmark validation where ground-truth uplift is known",
            "when_to_use": "testing datasets, LLM routing synthetic benchmark and regression checks",
            "risk": "not available in real observational business logs; do not expect it in production data",
            "ui_artifact": "Evaluation: Oracle Top-K Recall; Compare: oracle_top10_recall",
            "artifact_source": "runs/{run_id}/metrics.json:oracle_top_k",
            "code_source": "deepuplift/core/trainer.py; deepuplift/core/evaluator.py",
        },
    ]


def render_evaluation_formula_playbook() -> None:
    st.subheader("指标公式与推导")
    st.caption(
        "这一块把面试最容易被追问的公式、推导直觉、业务含义、误导风险、代码路径和 run artifact 字段放在一起，"
        "用于解释 uplift 模型到底怎么评估。"
    )
    counts = evaluation_formula_counts()
    render_stat_grid(
        [
            ("指标公式", f"{counts['metrics']:,}", False),
            ("Policy 推导", f"{counts['policy_derivations']:,}", True),
            ("优化阶段", f"{counts['optimization_stages']:,}", True),
            ("核心口径", "Ranking + ROI + Trust", True),
        ],
        columns=4,
    )
    formulas = pd.DataFrame(evaluation_formula_rows())
    if not formulas.empty:
        metric_options = formulas["metric"].tolist()
        default_metrics = [
            metric
            for metric in [
                "QINI",
                "AUUC",
                "uplift@K / Top-K observed uplift",
                "policy value",
                "incremental profit / cost-aware gain",
                "DR pseudo-outcome",
                "R-learner objective",
                "bootstrap CI",
                "overlap / trimming sensitivity",
            ]
            if metric in metric_options
        ]
        selected_metrics = st.multiselect(
            "选择指标公式",
            metric_options,
            default=default_metrics or metric_options[:8],
            key="evaluation_formula_metric_filter",
        )
        formula_view = formulas[formulas["metric"].isin(selected_metrics)] if selected_metrics else formulas
        formula_display = formula_view.rename(
            columns={
                "metric": "指标",
                "formula": "公式",
                "derivation_intuition": "推导直觉",
                "business_meaning": "业务含义",
                "when_to_use": "什么时候看",
                "misleading_when": "什么时候会误导",
                "code_source": "代码路径",
                "artifact_field": "产物字段",
                "interview_line": "面试讲法",
            }
        )
        st.dataframe(
            formula_display[
                [
                    "指标",
                    "公式",
                    "推导直觉",
                    "业务含义",
                    "什么时候看",
                    "什么时候会误导",
                    "代码路径",
                    "产物字段",
                    "面试讲法",
                ]
            ],
            width="stretch",
            hide_index=True,
        )
        st.download_button(
            "下载指标公式 CSV",
            data=dataframe_csv_bytes(formulas),
            file_name="deepuplift_evaluation_metric_formulas.csv",
            mime="text/csv",
            key="download_evaluation_metric_formulas",
        )

    policy_rows = pd.DataFrame(policy_derivation_rows())
    if not policy_rows.empty:
        with st.expander("Policy Value / ROI 推导卡", expanded=True):
            st.dataframe(
                policy_rows.rename(
                    columns={
                        "topic": "主题",
                        "steps": "推导步骤",
                        "formula": "公式",
                        "business_example": "业务例子",
                        "ui_location": "UI 位置",
                        "code_source": "代码路径",
                    }
                ),
                width="stretch",
                hide_index=True,
            )

    path_rows = pd.DataFrame(optimization_path_rows())
    if not path_rows.empty:
        with st.expander("从 0-1 到相对最优的优化路径", expanded=True):
            st.dataframe(
                path_rows.rename(
                    columns={
                        "stage": "阶段",
                        "why": "为什么做",
                        "what_changes": "做了什么优化",
                        "metric_effect": "指标/业务作用",
                        "risk": "风险",
                        "evidence": "证据入口",
                    }
                ),
                width="stretch",
                hide_index=True,
            )


def render_evaluation_metric_guide(result) -> None:
    st.subheader("Evaluation Metrics")
    st.caption("工业 uplift 决策不能只看一个分数：这里把 ranking、Top-K、ROI、稳定性、校准和 common support 放在同一页解释。")
    render_term_glossary("evaluation", expanded=False, key_prefix="evaluation_glossary")
    render_stat_grid(
        [
            ("Ranking", "QINI / AUUC", True),
            ("Launch Bucket", "uplift@K / Top-K", True),
            ("Business Value", "Policy Value / ROI", True),
            ("Trust Gates", "Calibration / CI / Overlap", True),
        ],
        columns=4,
    )
    render_evaluation_formula_playbook()
    render_failure_mode_playbook("evaluation")
    risk_rows = [
        {
            "signal": "high trust",
            "pattern": "QINI/AUUC positive, Top-K uplift positive, QINI CI lower bound positive, calibration usable, overlap stable",
            "decision": "candidate can move to Policy thresholding and online holdout design",
        },
        {
            "signal": "medium trust",
            "pattern": "ranking metrics positive but CI is wide, calibration is noisy, or overlap trimming changes the result",
            "decision": "use smaller Top-K, require bootstrap/segment review, and label as exploratory",
        },
        {
            "signal": "low trust",
            "pattern": "negative Top-K uplift, non-positive QINI, weak overlap, failed sensitivity, or policy value below cost",
            "decision": "do not deploy; revisit data design, features, treatment assignment or model family",
        },
    ]
    st.subheader("Metric Risk Legend")
    st.dataframe(pd.DataFrame(risk_rows), width="stretch", hide_index=True)
    metric_cards = []
    for row in evaluation_metric_rows():
        metric_cards.append(
            "<div class='du-backend-card'>"
            f"<div class='du-backend-kicker'>{html.escape(row['when_to_use'])}</div>"
            f"<div class='du-backend-title'>{html.escape(row['metric'])}</div>"
            f"<div class='du-backend-meta'>{html.escape(row['business_meaning'])}</div>"
            "</div>"
        )
    st.markdown(f"<div class='du-backend-grid'>{''.join(metric_cards)}</div>", unsafe_allow_html=True)
    metric_map = pd.DataFrame(evaluation_metric_rows())
    st.dataframe(metric_map, width="stretch", hide_index=True)
    st.download_button(
        "Download metric-to-artifact map",
        data=dataframe_csv_bytes(metric_map),
        file_name="deepuplift_metric_artifact_map.csv",
        mime="text/csv",
    )
    st.subheader("Loss-to-Metric Map")
    st.caption("This connects train-time objectives to the offline metrics and business decisions they are supposed to improve.")
    loss_metric_df = pd.DataFrame(loss_metric_mapping_rows())
    st.dataframe(loss_metric_df, width="stretch", hide_index=True)
    st.download_button(
        "Download loss-to-metric map",
        data=dataframe_csv_bytes(loss_metric_df),
        file_name="deepuplift_loss_to_metric_map.csv",
        mime="text/csv",
        key="download_loss_to_metric_map_evaluation",
    )
    trace_map = pd.DataFrame(frontier_demo_trace_rows())
    with st.expander("Frontier Metric Source Trace", expanded=False):
        st.caption(
            "For ECUP, delayed feedback and LLM routing, this trace connects demo preset, dataset schema, evaluator contract, saved metrics and Compare/History fields."
        )
        st.dataframe(trace_map, width="stretch", hide_index=True)
        st.download_button(
            "Download frontier metric source trace",
            data=dataframe_csv_bytes(trace_map),
            file_name="deepuplift_frontier_metric_source_trace.csv",
            mime="text/csv",
            key="evaluation_frontier_metric_source_trace_download",
        )
    if result is not None:
        metrics = result.eval_metrics
        top10 = _metric_top_fraction(metrics, 0.1)
        bootstrap = (metrics.get("bootstrap") or {}).get("metrics") or {}
        qini_ci = bootstrap.get("qini_score") or {}
        policy_best = metrics.get("policy_best") or {}
        overlap_propensity = ((metrics.get("overlap_trim") or {}).get("propensity") or {})
        latest_rows = [
            {"metric": "QINI", "latest_value": metrics.get("qini_score"), "decision_note": "positive and preferably positive lower CI"},
            {"metric": "AUUC", "latest_value": metrics.get("auuc_score"), "decision_note": "should agree directionally with QINI"},
            {"metric": "Top 10% observed uplift", "latest_value": top10.get("observed_uplift"), "decision_note": "launch bucket must be positive and sufficiently supported"},
            {"metric": "Calibration MAE", "latest_value": metrics.get("calibration_mae"), "decision_note": "lower is better; high values weaken threshold trust"},
            {"metric": "QINI CI low", "latest_value": qini_ci.get("low"), "decision_note": "negative lower bound means ranking may not be stable"},
            {"metric": "Best policy top fraction", "latest_value": policy_best.get("top_fraction"), "decision_note": "cost-aware recommended targeting size"},
            {"metric": "Best policy net value", "latest_value": policy_best.get("observed_net_value", policy_best.get("predicted_net_value")), "decision_note": "closer to ROI than AUUC alone"},
            {"metric": "Weak overlap rate", "latest_value": overlap_propensity.get("weak_overlap_rate"), "decision_note": "high values cap causal confidence"},
        ]
        st.subheader("Latest Run Metric Snapshot")
        st.dataframe(pd.DataFrame(latest_rows), width="stretch", hide_index=True)


def dataframe_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def build_run_evidence_zip(run_dir: Path, run_id: str) -> tuple[bytes, list[str]]:
    buffer = io.BytesIO()
    added: list[str] = []
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, _label in RUN_EVIDENCE_FILES:
            path = run_dir / name
            if path.exists() and path.is_file():
                zf.write(path, arcname=f"{run_id}/{name}")
                added.append(name)
        manifest = {
            "run_id": run_id,
            "run_dir": str(run_dir),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
            "files": added,
        }
        zf.writestr(f"{run_id}/evidence_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        added.append("evidence_manifest.json")
    return buffer.getvalue(), added


def run_evidence_summary(run_dir: Path) -> pd.DataFrame:
    rows = []
    for name, label in RUN_EVIDENCE_FILES:
        path = run_dir / name
        rows.append(
            {
                "file": name,
                "role": label,
                "present": path.exists() and path.is_file(),
                "size_kb": round(path.stat().st_size / 1024, 2) if path.exists() and path.is_file() else 0,
            }
        )
    return pd.DataFrame(rows)


def dataframe_markdown(df: pd.DataFrame, columns: list[str]) -> str:
    present = [col for col in columns if col in df.columns]
    if not present or df.empty:
        return "No ranking rows available."
    view = df[present].copy().fillna("")
    header = "| " + " | ".join(present) + " |"
    separator = "| " + " | ".join("---" for _ in present) + " |"
    rows = []
    for record in view.to_dict(orient="records"):
        values = [str(record.get(col, "")).replace("|", "\\|") for col in present]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def model_card_markdown(model_name: str) -> str:
    card = build_model_card(model_name, description=MODEL_DESCRIPTIONS.get(model_name, ""))
    scenario_tags = model_scenario_tags(model_name)
    presets = model_parameter_presets(model_name)
    recommended_preset, recommended_params = recommended_model_preset(model_name)
    rows = [
        ("Status", card.get("status")),
        ("Backend", f"{card.get('source')} / {card.get('family')}"),
        ("Stage", card.get("stage")),
        ("Tasks", ", ".join(card.get("tasks", []))),
        ("Scenario Tags", "; ".join(scenario_tags)),
        ("Parameter presets", "; ".join(model_parameter_presets(model_name).keys())),
        ("Recommended preset JSON", f"{recommended_preset}: `{json.dumps(recommended_params, ensure_ascii=False)}`"),
        ("Best for", "; ".join(card.get("best_for", []))),
        ("Assumptions", "; ".join(card.get("assumptions", []))),
        ("Risks", "; ".join(card.get("risks", []))),
        ("Decision use", card.get("decision_use")),
        ("Setup", card.get("setup_hint")),
        ("Source", card.get("source_url")),
    ]
    lines = [f"### `{model_name}`", ""]
    lines.extend(f"- {label}: {value}" for label, value in rows if value)
    return "\n".join(lines)


def json_clean(value):
    if isinstance(value, dict):
        return {str(key): json_clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_clean(item) for item in value]
    if isinstance(value, tuple):
        return [json_clean(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def model_decision_guide() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "situation": "Randomized / A-B-like binary treatment",
                "start_with": "TLearnerLightGBM, SkLiftTwoModelsLightGBM, TransformedOutcomeLightGBM",
                "stress_test": "ClassVariableTransform, TarNet/DragonNet, calibration buckets",
                "watch_out": "Do not deploy from QINI alone; inspect Top-K observed uplift and policy value.",
            },
            {
                "situation": "Observational treatment assignment / selection bias",
                "start_with": "DRLearnerLightGBM, RLearnerLightGBM, EconMLDRLearner",
                "stress_test": "EconMLCausalForestDML, EconMLGRFCausalForest, overlap trimming",
                "watch_out": "Requires conditional ignorability and common support; weak overlap should cap targeting.",
            },
            {
                "situation": "Treatment/control imbalance",
                "start_with": "XLearnerLightGBM, DomainAdaptationLightGBM",
                "stress_test": "CFRNet, DR/R learners, bootstrap lower bounds",
                "watch_out": "Small minority cells make Top-K and segment metrics noisy.",
            },
            {
                "situation": "Category-heavy industrial tables",
                "start_with": "LightGBM meta-learners",
                "stress_test": "CatBoost optional family after installing catboost",
                "watch_out": "Validate local runtime stability before adding heavy native backends to large compares.",
            },
            {
                "situation": "Top-K growth deployment",
                "start_with": "GGBMUpliftLightGBM, PAVCalibratedDRLearnerLightGBM",
                "stress_test": "Policy Value, bootstrap CI, calibration, business-score alignment",
                "watch_out": "Choose by stable net value and calibrated ranking, not only AUUC.",
            },
            {
                "situation": "Multiple treatments / actions",
                "start_with": "MultiTLearnerGBM, MultiDRLearnerGBM",
                "stress_test": "Action propensity overlap, action mix, IPW policy value",
                "watch_out": "Rare actions need support checks before recommending them.",
            },
            {
                "situation": "Continuous treatment / dose",
                "start_with": "DoseResponseGBM, DoseResponseRF",
                "stress_test": "Dose grid curves, recommended dose distribution, gain vs baseline",
                "watch_out": "Binary QINI/AUUC is not enough for dose-response decisions.",
            },
        ]
    )


def build_compare_report(
    ranking: pd.DataFrame,
    dataset_label: str,
    diagnostics: dict,
    candidate_models: list[str],
) -> str:
    best = ranking.iloc[0].to_dict() if ranking is not None and not ranking.empty else {}
    warnings = diagnostics.get("warnings", [])
    design = diagnostics.get("design") or {}
    overlap = diagnostics.get("overlap") or {}
    lines = [
        "# DeepUplift Model Compare Report",
        "",
        "## Context",
        "",
        f"- Dataset: {dataset_label}",
        f"- Treatment design: `{design.get('treatment_type')}`",
        f"- Outcome type: `{design.get('outcome_type')}`",
        f"- Candidate models: {', '.join(candidate_models)}",
        "",
        "## Recommendation",
        "",
    ]
    if best:
        lines.extend(
            [
                f"- Recommended model: `{best.get('model')}`",
                f"- Run id: `{best.get('run_id')}`",
                f"- Decision readiness: {best.get('readiness_score')} / 100 ({best.get('readiness_level')})",
                f"- QINI: {best.get('qini')}",
                f"- AUUC: {best.get('auuc')}",
                f"- Sensitivity verdict: `{best.get('sensitivity_verdict')}`",
                f"- Weak overlap rate: {best.get('weak_overlap_rate')}",
                "",
                "Primary sort order is QINI CI lower bound when available, otherwise Decision Readiness and QINI.",
            ]
        )
    else:
        lines.append("- No successful model run is available.")

    if warnings:
        lines.extend(["", "## Diagnostic Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)

    if overlap:
        lines.extend(
            [
                "",
                "## Pre-Train Overlap Summary",
                "",
                f"- Treatment AUC: {overlap.get('treatment_auc')}",
                f"- Weak overlap rate: {overlap.get('weak_overlap_rate')}",
                f"- Propensity p05/p95: {overlap.get('propensity_p05')} / {overlap.get('propensity_p95')}",
            ]
        )

    lines.extend(
        [
            "",
            "## Ranking",
            "",
            dataframe_markdown(
                ranking,
                [
                    "model",
                    "preset",
                    "run_id",
                    "readiness_score",
                    "readiness_level",
                    "qini",
                    "qini_ci_low",
                    "auuc",
                    "calibration_mae",
                    "top10_observed_uplift",
                    "oracle_top10_recall",
                    "business_top10_overlap",
                    "business_top10_capture",
                    "frontier_metric_schema",
                    "full_funnel_top10_conversion_uplift",
                    "delayed_d30_top10_uplift",
                    "delayed_censored_rate",
                    "policy_best_fraction",
                    "policy_best_net",
                    "sensitivity_verdict",
                    "weak_overlap_rate",
                ],
            ),
            "",
            "## Decision Guardrails",
            "",
            "- Prefer models with high Decision Readiness, stable QINI lower bounds, positive Top-K uplift, acceptable calibration, and sensitivity checks that do not fail.",
            "- If weak overlap is high, restrict deployment to well-overlapped users or gather more randomized data.",
            "- Treat segment findings as hypotheses unless segment support is large and stable.",
            "",
        ]
    )
    card_models = []
    if best.get("model"):
        card_models.append(str(best.get("model")))
    card_models.extend(candidate_models)
    card_models = list(dict.fromkeys(card_models))
    if card_models:
        lines.extend(["", "## Candidate Model Cards", ""])
        for model in card_models[:12]:
            lines.extend([model_card_markdown(model), ""])
    return "\n".join(lines)


def save_compare_report(
    ranking: pd.DataFrame,
    dataset_label: str,
    diagnostics: dict,
    candidate_models: list[str],
    prefix: str = "compare",
) -> str:
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    path = reports_dir / f"{prefix}_report_{timestamp}.md"
    return save_text(path, build_compare_report(ranking, dataset_label, diagnostics, candidate_models))


def save_compare_readiness_json(
    ranking: pd.DataFrame,
    dataset_label: str,
    diagnostics: dict,
    candidate_models: list[str],
    prefix: str = "compare",
) -> str:
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    best = json_clean(ranking.iloc[0].to_dict()) if ranking is not None and not ranking.empty else {}
    payload = {
        "dataset": dataset_label,
        "candidate_models": candidate_models,
        "best": best,
        "ranking": [] if ranking is None or ranking.empty else json_clean(ranking.to_dict(orient="records")),
        "diagnostic_warnings": diagnostics.get("warnings", []),
        "design": diagnostics.get("design", {}),
        "overlap": diagnostics.get("overlap", {}),
        "selection_rule": "qini_ci_low when available; otherwise readiness_score and qini",
    }
    path = reports_dir / f"{prefix}_readiness_{timestamp}.json"
    return save_json(path, json_clean(payload))


def save_compare_bundle_manifest(
    ranking: pd.DataFrame,
    dataset_label: str,
    report_path: str,
    readiness_path: str,
    compare_results: list,
    prefix: str = "compare",
) -> str:
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    runs = []
    for item in compare_results:
        artifacts = getattr(item, "artifacts", {}) or {}
        runs.append(
            {
                "run_id": getattr(item, "run_id", None),
                "model": (getattr(item, "config", {}) or {}).get("model_name"),
                "run_dir": getattr(item, "run_dir", None),
                "metrics": artifacts.get("metrics"),
                "readiness": artifacts.get("readiness"),
                "predictions": artifacts.get("predictions"),
                "run_note": artifacts.get("run_note"),
            }
        )
    payload = {
        "schema_version": 2,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "dataset": dataset_label,
        "candidate_models": sorted({row.get("model") for row in runs if row.get("model")}),
        "report": report_path,
        "compare_readiness": readiness_path,
        "ranking": [] if ranking is None or ranking.empty else json_clean(ranking.to_dict(orient="records")),
        "runs": runs,
        "model_cards": [
            build_model_card(model, description=MODEL_DESCRIPTIONS.get(model, ""))
            for model in sorted({row.get("model") for row in runs if row.get("model")})
        ],
    }
    path = reports_dir / f"{prefix}_bundle_{timestamp}.json"
    return save_json(path, json_clean(payload))


def build_multi_compare_report(ranking: pd.DataFrame, dataset_label: str, candidate_models: list[str]) -> str:
    best = ranking.iloc[0].to_dict() if ranking is not None and not ranking.empty else {}
    lines = [
        "# DeepUplift Multi-Treatment Compare Report",
        "",
        "## Context",
        "",
        f"- Dataset: {dataset_label}",
        f"- Candidate models: {', '.join(candidate_models)}",
        "",
        "## Recommendation",
        "",
    ]
    if best:
        lines.extend(
            [
                f"- Recommended model: `{best.get('model')}`",
                f"- Strategy: `{best.get('strategy')}`",
                f"- Run id: `{best.get('run_id')}`",
                f"- Incremental IPW value: {best.get('incremental_policy_value_ipw')}",
                f"- Incremental IPW CI low: {best.get('incremental_ci_low')}",
                f"- Max low-propensity rate: {best.get('max_low_propensity_rate')}",
                "",
                "Primary sort order is incremental policy-value CI lower bound when available, otherwise incremental IPW value.",
            ]
        )
    else:
        lines.append("- No successful multi-treatment run is available.")

    lines.extend(
        [
            "",
            "## Ranking",
            "",
            dataframe_markdown(
                ranking,
                [
                    "model",
                    "strategy",
                    "run_id",
                    "incremental_policy_value_ipw",
                    "incremental_ci_low",
                    "incremental_ci_high",
                    "recommended_policy_value_ipw",
                    "control_policy_value_ipw",
                    "max_low_propensity_rate",
                    "top_recommended_action",
                ],
            ),
            "",
            "## Guardrails",
            "",
            "- Prefer models whose incremental policy-value lower bound is positive or least fragile.",
            "- If an action has high low-propensity rate, avoid broad deployment for that action without more randomized data.",
            "- Inspect recommended action mix before exporting a policy, because degenerate one-action recommendations can hide instability.",
            "",
        ]
    )
    return "\n".join(lines)


def save_multi_compare_report(
    ranking: pd.DataFrame,
    dataset_label: str,
    candidate_models: list[str],
    prefix: str = "multi_compare",
) -> str:
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    path = reports_dir / f"{prefix}_report_{timestamp}.md"
    return save_text(path, build_multi_compare_report(ranking, dataset_label, candidate_models))


def bootstrap_text(metrics: dict, key: str) -> str:
    bootstrap = (metrics.get("bootstrap") or {}).get("metrics", {})
    item = bootstrap.get(key)
    if not item:
        return "CI not computed"
    low, high = item.get("low"), item.get("high")
    if low is None or high is None:
        return "CI not computed"
    return f"95% CI [{low:.4f}, {high:.4f}]"


def sensitivity_summary_text(metrics: dict) -> str:
    sensitivity = metrics.get("sensitivity") or {}
    if not sensitivity:
        return "Sensitivity checks not computed."
    return f"{sensitivity.get('verdict', 'unknown')}: {sensitivity.get('message', '')}"


def _as_float(value) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def render_decision_readiness(metrics: dict) -> dict:
    readiness = decision_readiness(metrics)
    render_stat_grid(
        [
            ("Decision Readiness", f"{readiness['score']}/100", False),
            ("Level", readiness["level"], True),
            ("Blockers", len(readiness["blockers"]), False),
            ("Recommendation", readiness["recommendation"], True),
        ],
        columns=4,
    )
    checks = pd.DataFrame(readiness["checks"])
    if not checks.empty:
        st.dataframe(checks, width="stretch", hide_index=True)
    return readiness


def _metric_top_fraction(metrics: dict, fraction: float) -> dict:
    for row in metrics.get("top_k", []) or []:
        if abs((row.get("top_fraction") or 0) - fraction) < 1e-6:
            return row
    return {}


def render_model_rationale(model_name: str, metrics: dict | None = None, ranking_row: dict | None = None) -> None:
    description = MODEL_DESCRIPTIONS.get(model_name, model_name)
    missing = missing_dependencies(model_name) if model_name in MODEL_REGISTRY else []
    source, family = model_source_and_family(model_name)
    status, setup_hint = model_readiness(model_name, missing)
    if metrics:
        readiness = decision_readiness(metrics)
        top10 = _metric_top_fraction(metrics, 0.1)
        evidence = (
            f"QINI {_as_float(metrics.get('qini_score')) or 0:.3f}; "
            f"AUUC {_as_float(metrics.get('auuc_score')) or 0:.3f}; "
            f"Top 10% uplift {_as_float(top10.get('observed_uplift')) if top10 else 'NA'}"
        )
        risk = "; ".join(readiness.get("blockers") or []) or sensitivity_summary_text(metrics)
        decision = f"{readiness.get('score')}/100 · {readiness.get('level')}"
    else:
        row = ranking_row or {}
        evidence = (
            f"QINI {row.get('qini', 'NA')}; "
            f"QINI low {row.get('qini_ci_low', 'NA')}; "
            f"Policy net {row.get('policy_best_net', 'NA')}"
        )
        risk_bits = [
            f"sensitivity={row.get('sensitivity_verdict')}" if row.get("sensitivity_verdict") else "",
            f"weak overlap={row.get('weak_overlap_rate')}" if row.get("weak_overlap_rate") is not None else "",
        ]
        risk = "; ".join(bit for bit in risk_bits if bit) or "No blocking risk available in ranking row."
        decision = f"{row.get('readiness_score', 'NA')}/100 · {row.get('readiness_level', 'unknown')}"

    items = [
        ("Model", model_name),
        ("Backend", f"{source} · {family} · {status}"),
        ("Setup", setup_hint),
        ("Why this model", description),
        ("Decision", decision),
        ("Evidence", evidence),
        ("Risk / next check", risk),
    ]
    cards = "".join(
        "<div class='rationale-card'>"
        f"<span>{html.escape(label)}</span>"
        f"<b>{html.escape(str(value))}</b>"
        "</div>"
        for label, value in items
    )
    st.markdown(f"<div class='rationale-panel'>{cards}</div>", unsafe_allow_html=True)


def render_model_card_panel(card: dict) -> None:
    presets = card.get("parameter_presets") or model_parameter_presets(str(card.get("model", "")))
    recommended_preset, recommended_params = recommended_model_preset(str(card.get("model", "")))
    scenario_tags = card.get("scenario_tags") or model_scenario_tags(str(card.get("model", "")))
    items = [
        ("Model", card.get("model", "")),
        ("Status", card.get("status", "")),
        ("Backend", f"{card.get('source', '')} · {card.get('family', '')}"),
        ("Stage", card.get("stage", "")),
        ("Tasks", ", ".join(card.get("tasks", []))),
        ("Scenario Tags", "; ".join(scenario_tags)),
        ("Setup", card.get("setup_hint", "")),
        ("Best For", "; ".join(card.get("best_for", []))),
        ("Parameter Presets", "; ".join(presets.keys())),
        ("Recommended Preset", f"{recommended_preset}: {json.dumps(recommended_params, ensure_ascii=False)}"),
        ("Assumptions", "; ".join(card.get("assumptions", []))),
        ("Risks", "; ".join(card.get("risks", []))),
        ("Decision Use", card.get("decision_use", "")),
        ("Integration Gate", card.get("integration_gate", "")),
        ("Source Evidence", card.get("source_evidence", "")),
        ("Promotion Rule", card.get("promotion_rule", "")),
        ("Source", card.get("source_url", "")),
    ]
    cards = "".join(
        "<div class='rationale-card'>"
        f"<span>{html.escape(label)}</span>"
        f"<b>{html.escape(str(value))}</b>"
        "</div>"
        for label, value in items
    )
    st.markdown(f"<div class='rationale-panel'>{cards}</div>", unsafe_allow_html=True)


def _scale_index(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == 0:
        return numeric
    low = numeric.min()
    high = numeric.max()
    if pd.isna(low) or pd.isna(high):
        return numeric
    if abs(high - low) < 1e-12:
        return pd.Series([100.0 if pd.notna(value) else None for value in numeric], index=series.index)
    return ((numeric - low) / (high - low) * 100).round(2)


def render_compare_decision_board(ranking: pd.DataFrame, *, caption: str | None = None) -> None:
    if ranking is None or ranking.empty or "model" not in ranking.columns:
        return
    board = ranking.copy()
    chart_cols = []
    if "readiness_score" in board.columns:
        board["readiness"] = pd.to_numeric(board["readiness_score"], errors="coerce")
        chart_cols.append("readiness")
    if "qini" in board.columns:
        board["qini_index"] = _scale_index(board["qini"])
        chart_cols.append("qini_index")
    if "policy_best_net" in board.columns:
        board["policy_net_index"] = _scale_index(board["policy_best_net"])
        chart_cols.append("policy_net_index")
    if not chart_cols:
        return
    chart = board[["model", *chart_cols]].dropna(how="all", subset=chart_cols).head(12)
    if chart.empty:
        return
    if caption:
        st.caption(caption)
    labels = {
        "readiness": ("Readiness", "#2dd4bf"),
        "qini_index": ("QINI", "#60a5fa"),
        "policy_net_index": ("Policy", "#f59e0b"),
    }
    rows_html = []
    for row in chart.to_dict("records"):
        bar_html = []
        for col in chart_cols:
            value = _as_float(row.get(col))
            width = 0 if value is None else max(0.0, min(100.0, value))
            label, color = labels.get(col, (col, "#94a3b8"))
            value_text = "NA" if value is None else f"{width:.0f}"
            bar_html.append(
                "<div class='decision-mini-bar'>"
                f"<span>{html.escape(label)}</span>"
                "<div class='decision-mini-track'>"
                f"<i style='width:{width:.2f}%; background:{color};'></i>"
                "</div>"
                f"<b>{value_text}</b>"
                "</div>"
            )
        rows_html.append(
            "<div class='decision-board-row'>"
            f"<div class='decision-board-model'>{html.escape(str(row.get('model', '')))}</div>"
            f"<div class='decision-board-bars'>{''.join(bar_html)}</div>"
            "</div>"
        )
    st.markdown(
        "<div class='decision-board'>"
        "<div class='decision-board-head'><span>Model</span><span>Decision signal</span></div>"
        f"{''.join(rows_html)}"
        "</div>",
        unsafe_allow_html=True,
    )


def compare_status_html(current: int, total: int, model: str, succeeded: int, failed: int) -> str:
    items = [
        ("Progress", f"{current}/{total}"),
        ("Current Model", model),
        ("Succeeded", str(succeeded)),
        ("Failed", str(failed)),
    ]
    cards = "".join(
        "<div class='run-status-card'>"
        f"<span>{html.escape(label)}</span>"
        f"<b>{html.escape(str(value))}</b>"
        "</div>"
        for label, value in items
    )
    return f"<div class='run-status-grid'>{cards}</div>"


def model_capability_summary(task: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    catalog = model_catalog_dataframe(task)
    if catalog.empty:
        return catalog, catalog
    source_summary = (
        catalog.groupby(["source", "status"], as_index=False)
        .size()
        .rename(columns={"size": "models"})
        .sort_values(["status", "models"], ascending=[True, False])
    )
    ready_catalog = catalog[catalog["status"] == "ready"]
    family_summary = (
        ready_catalog.groupby(["family"], as_index=False)
        .size()
        .rename(columns={"size": "ready_models"})
        .sort_values("ready_models", ascending=False)
    )
    return source_summary, family_summary


def policy_metric_disagreement_rows() -> list[dict[str, str | float]]:
    return [
        {
            "case": "High QINI, low margin coupon",
            "offline_signal": "QINI/AUUC positive because ranking finds persuadable users",
            "business_assumption": "gross margin 4.0, expected coupon payout + fatigue cost 5.2",
            "policy_value": -1.20,
            "decision": "do not launch; reduce coupon value or target a smaller supported bucket",
        },
        {
            "case": "Moderate QINI, high LTV retention",
            "offline_signal": "QINI is lower, but Top-K is stable and users have high incremental LTV",
            "business_assumption": "retention value 35.0, push/fatigue cost 0.21",
            "policy_value": 2.60,
            "decision": "launch controlled Top-K with holdout and delayed-feedback monitoring",
        },
        {
            "case": "Ads high pCVR, weak incrementality",
            "offline_signal": "pCVR segment looks attractive but uplift@K is near zero",
            "business_assumption": "media cost consumes attribution-only conversion value",
            "policy_value": -0.35,
            "decision": "keep geo/ghost-ad holdout; do not optimize bidding on pCVR alone",
        },
        {
            "case": "LLM routing high hard-query uplift",
            "offline_signal": "QINI/AUUC positive only for hard prompts; easy prompts have low uplift",
            "business_assumption": "quality value 2.0, extra model + latency cost 0.006",
            "policy_value": 0.18,
            "decision": "route only high uplift prompts; enforce budget and latency guardrails",
        },
    ]


def agent_reply(
    prompt: str,
    df: pd.DataFrame,
    task: str,
    treatment_col: str,
    outcome_col: str,
    feature_cols: list[str],
    model_name: str,
    result,
    selected_dataset: dict | None,
    compare_ranking: pd.DataFrame | None = None,
    policy_context: dict | None = None,
) -> str:
    text = prompt.strip().lower()
    mentioned_models = extract_model_mentions(prompt)
    models = recommend_models(df, task, treatment_col, feature_cols, selected_dataset)
    treatment_dist = df[treatment_col].value_counts(normalize=True, dropna=True).to_dict()
    categorical_cols = [col for col in feature_cols if col in df.columns and not pd.api.types.is_numeric_dtype(df[col])]
    public_product_reply = public_product_agent_reply(prompt)
    if public_product_reply:
        return public_product_reply
    training_platform_reply = ml_training_platform_agent_reply(prompt)
    if training_platform_reply:
        return training_platform_reply
    ui_reply = frontend_agent_reply(prompt)
    if ui_reply:
        return ui_reply
    framework_reply = open_source_framework_agent_reply(prompt)
    if framework_reply:
        return framework_reply
    deconstruction_reply = model_deconstruction_reply(
        prompt,
        {
            "model_name": model_name,
            "latest_run_dir": result.run_dir if result is not None else "",
        },
    )
    if deconstruction_reply:
        return deconstruction_reply
    failure_reply = failure_mode_agent_reply(prompt)
    if failure_reply:
        return failure_reply
    formula_reply = evaluation_formula_agent_reply(prompt)
    if formula_reply:
        return formula_reply
    frontier_reply = frontier_agent_reply(prompt)
    if frontier_reply:
        return frontier_reply
    resume_workbench_reply = resume_workbench_agent_reply(prompt)
    if resume_workbench_reply:
        return resume_workbench_reply
    deep_design_reply = deep_design_agent_reply(prompt)
    if deep_design_reply:
        return deep_design_reply
    scenario_reply = scenario_agent_reply(prompt)
    if scenario_reply:
        return scenario_reply
    industry_reply = industry_agent_reply(
        prompt,
        {
            "task": task,
            "model_name": model_name,
            "treatment_col": treatment_col,
            "outcome_col": outcome_col,
            "registered_models": len(MODEL_REGISTRY),
            "ready_models": len(available_models(task, only_available=True)),
            "rows": len(df),
            "features": len(feature_cols),
        },
    )
    if industry_reply:
        return industry_reply
    interview_reply = interview_agent_reply(
        prompt,
        {
            "task": task,
            "model_name": model_name,
            "treatment_col": treatment_col,
            "outcome_col": outcome_col,
            "registered_models": len(MODEL_REGISTRY),
            "ready_models": len(available_models(task, only_available=True)),
            "rows": len(df),
            "features": len(feature_cols),
        },
    )
    if interview_reply:
        return interview_reply

    if any(key in text for key in ["5分钟", "五分钟", "怎么演示", "演示路径", "runbook", "现场演示", "面试演示"]):
        return (
            "### 5 分钟面试演示路径\n\n"
            "1. **Story**：先讲定位，DeepUplift Agent 是 causal decision workbench，不是单模型脚本。\n"
            "2. **Workflow**：在 sidebar 的 `Demo preset` 里切到 `ECUP full-funnel` 或 `Delayed feedback`，展示工业场景不是空讲。\n"
            "3. **Models**：打开 `Scenario Model Matrix` 和 `Scenario Tag`，说明模型按发券、广告、增长、推荐、平台补贴选择。\n"
            "4. **Agent**：问一个硬问题，比如 `发券如何处理补贴套利和羊毛党？` 或 `ECUP 全链路 uplift 怎么评估？`。\n"
            "5. **Evaluation / Compare / History**：展示 full-funnel 或 delayed-feedback 指标已经进入 evaluator、ranking 和 history。\n"
            "6. **Policy / Evidence**：用 `Scenario ROI Helper` 解释阈值，再下载 evidence pack 证明模型目录、来源质量、截图、回归和运行 artifact 都可复现。\n\n"
            "本地证据：`Workflow -> Frontier Demo Presets`、`docs/UPLIFT_AGENT_INTERVIEW_RUNBOOK.md`、`docs/UPLIFT_SCENARIO_MODEL_MATRIX.md`、`docs/FRONTIER_DATASET_TRAINING_EVIDENCE.md`。"
        )

    if any(key in text for key in ["demo preset", "source trace", "演示入口", "演示证据", "指标来源", "字段来源"]):
        rows = frontier_demo_trace_rows()
        return (
            "Demo Preset 的证据链已经固定在 `Workflow -> Demo Preset Source Trace` 和 `Evidence -> Demo Preset Source Trace`。\n\n"
            f"核心链路：`{rows}`\n\n"
            "面试讲法：我不是只放几个示例 CSV，而是把 demo preset 绑定到 manifest、evaluator contract、metrics.json、Compare/History 摘要字段和 smoke 脚本，确保前沿场景能被复现。"
        )

    if any(key in text for key in ["模型卡", "model card", "介绍模型", "解释模型", "模型解释", "这个模型"]):
        targets = mentioned_models or [model_name]
        cards = "\n\n".join(format_model_card(model) for model in targets[:5])
        return (
            f"下面是模型卡片解释，共 `{len(targets[:5])}` 个模型。\n\n"
            f"{cards}\n\n"
            "你也可以到 `Models` 页打开完整结构化 Model Card，并下载 JSON。"
        )

    if any(key in text for key in ["诊断", "overlap", "平衡", "bias", "偏差", "可比"]):
        diagnostics = diagnose_uplift_data(df, treatment_col, outcome_col, feature_cols)
        design = diagnostics["design"]
        overlap = diagnostics.get("overlap") or {}
        selection_bias = diagnostics.get("selection_bias") or {}
        warnings = diagnostics.get("warnings", [])
        knowledge = matched_knowledge(diagnostics, task)
        balance = pd.DataFrame(diagnostics.get("feature_balance", []))
        balance_text = ""
        if not balance.empty:
            top_balance = balance.head(5)[["feature", "metric", "imbalance"]].to_dict(orient="records")
            balance_text = f"\n\nTop imbalance features: `{top_balance}`"
        warning_text = "\n".join(f"- {warning}" for warning in warnings) or "- 暂未发现明显诊断告警。"
        return (
            f"当前数据设计：treatment 是 `{design['treatment_type']}`，outcome 是 `{design['outcome_type']}`。\n\n"
            f"Propensity overlap 摘要：`{overlap or '当前 treatment 不是二值，暂不计算 overlap'}`\n\n"
            f"Selection bias / SSB proxy：`{selection_bias}`\n\n"
            f"诊断提醒：\n{warning_text}{balance_text}\n\n"
            f"知识库依据：\n{format_rule_evidence(knowledge)}"
        )

    if any(key in text for key in ["裁剪", "trim", "common support", "positivity", "重叠区间"]):
        if result is None:
            return "还没有训练结果。训练完成后我会在 `Evaluation` 页显示 Overlap Trimming，比较全量 holdout 和 common-support 子样本上的 QINI/Top-K。"
        overlap_trim = result.eval_metrics.get("overlap_trim") or {}
        if not overlap_trim:
            return "这次 run 没有 overlap trimming 结果，通常是因为没有可用于 propensity 的特征列，或 treatment 不是二值。"
        propensity = overlap_trim.get("propensity") or {}
        rows = overlap_trim.get("rows") or []
        worst_drop = ""
        if rows:
            worst_drop = f"\n\nTrim 摘要：`{rows}`"
        return (
            f"Overlap 摘要：p05 `{propensity.get('p05')}`，median `{propensity.get('median')}`，p95 `{propensity.get('p95')}`，"
            f"weak overlap rate `{propensity.get('weak_overlap_rate')}`。{worst_drop}\n\n"
            "如果裁剪后 QINI/Top10 uplift 明显下降，我会把模型建议降级为探索性；如果裁剪后仍稳定，说明排序更可信。"
        )

    if any(key in text for key in ["multi", "多 treatment", "多处理", "多动作", "多值"]):
        values = sorted(df[treatment_col].dropna().unique().tolist(), key=lambda item: str(item))
        if len(values) <= 2:
            return (
                f"当前 treatment 只有 `{values}`，还是二元 uplift 场景。多 treatment 页会在 treatment 取值超过 2 个时启用。\n\n"
                "如果你有多个触达动作，比如 control/email/push/coupon，可以上传包含多值 treatment 的 CSV，然后训练 `MultiTLearnerGBM` 或 `MultiTLearnerRF`。"
            )
        return (
            f"当前检测到多 treatment：`{values}`。\n\n"
            "建议先在 `Multi-Treatment` 页训练 `MultiTLearnerGBM` 快速建立基线，再用 `MultiDRLearnerGBM` 做偏差校正对照。\n\n"
            "重点看三类结果：\n"
            "1. `Action Propensity Overlap`：每个 action 是否有足够 common support。\n"
            "2. `Recommended Action Mix`：模型是否过度集中推荐某一个 action。\n"
            "3. `Incremental IPW Value` 和 bootstrap CI：推荐策略相对 control 是否稳定增益。\n\n"
            "训练后可以直接用 `Score With Saved Multi-Treatment Run` 对新数据输出 `recommended_treatment` 和 Top-K 推荐名单。"
        )

    if any(key in text for key in ["连续", "dose", "剂量", "强度", "频次", "discount", "vcnet", "drnet"]):
        if pd.api.types.is_numeric_dtype(df[treatment_col]) and df[treatment_col].dropna().nunique() > 5:
            return (
                f"当前 treatment `{treatment_col}` 看起来是连续 treatment，取值范围约为 "
                f"`{df[treatment_col].min()}` 到 `{df[treatment_col].max()}`。\n\n"
                "建议到 `Dose-Response` 页先训练 `DoseResponseGBM`，查看平均剂量响应曲线、推荐剂量分布和 `predicted_gain_vs_baseline`。\n\n"
                "这条链路是 DRNet/VCNet/VC-Transformer 的接口底座：先把连续 treatment 的训练、评估、打分、导出闭环跑稳，再逐步替换更强的神经剂量响应模型。"
            )
        return "当前 treatment 不像连续数值 treatment。若要做 dose-response，请选择折扣率、补贴强度、触达频次、价格调整幅度这类连续列。"

    if any(key in text for key in ["模型", "model", "选", "推荐", "choose"]):
        diagnostics = diagnose_uplift_data(df, treatment_col, outcome_col, feature_cols)
        design = diagnostics.get("design") or {}
        treatment_ratio_now = df[treatment_col].mean() if pd.api.types.is_numeric_dtype(df[treatment_col]) else None
        if treatment_ratio_now is not None and (treatment_ratio_now < 0.35 or treatment_ratio_now > 0.65):
            design_note = (
                f"数据更像观测/选择偏差场景：treatment rate `{treatment_ratio_now:.1%}`，"
                "优先用 DR/R/Domain Adaptation/Causal Forest，并重点看 overlap、sensitivity 和 bootstrap。"
            )
        else:
            design_note = (
                f"数据更像随机实验或近似均衡二元 treatment：design=`{design.get('treatment_type')}`，"
                "可以先用 T/S/X learner 和 scikit-uplift baseline 快速比较，再用 DR/R 做稳健性检查。"
            )
        runnable = available_models(task, only_available=True)
        knowledge_models = knowledge_model_recommendations(diagnostics, runnable)
        design_first = []
        if treatment_ratio_now is not None and (treatment_ratio_now < 0.35 or treatment_ratio_now > 0.65):
            design_first = [
                "DRLearnerLightGBM",
                "RLearnerLightGBM",
                "DomainAdaptationLightGBM",
                "EconMLCausalForestDML",
                "EconMLDRLearner",
                "CausalForest",
                "DRForest",
                "PAVCalibratedDRLearnerLightGBM",
            ]
        blended = [model for model in design_first + knowledge_models + models if model in runnable]
        blended = list(dict.fromkeys(blended))
        frontier_note = ""
        if any(key in text for key in ["最新", "前沿", "扩展", "llm", "论文", "research"]):
            frontier_note = f"\n\n前沿扩展队列：\n{format_research_frontier(limit=6)}"
        return (
            f"当前任务是 `{task}`，页面左侧的 `Model` 下拉框可以直接切换模型。\n\n"
            f"数据设计判断：{design_note}\n\n"
            f"我建议优先试这些：\n{format_model_choices(blended[:8])}\n\n"
            f"当前已选择 `{model_name}`。\n\n"
            f"模型卡片：\n{format_model_card(model_name)}"
            "\n\n完整选择逻辑可以在 `Models` 页展开 `Model Family Decision Guide` 查看。"
            f"{frontier_note}"
        )

    if any(key in text for key in ["对比", "compare", "排名", "ranking", "哪个最好"]):
        if compare_ranking is None or compare_ranking.empty:
            return "还没有模型对比结果。到 `Compare` 页选择多个模型并点击 `Train comparison`，我会按 QINI、AUUC、Calibration MAE 和 Top 10% uplift 帮你解释排名。"
        view = compare_ranking.head(5).to_dict(orient="records")
        best = compare_ranking.iloc[0].to_dict()
        oracle_note = ""
        if "oracle_top10_recall" in compare_ranking.columns and compare_ranking["oracle_top10_recall"].notna().any():
            oracle_note = "\n\n当前数据含 oracle uplift，建议同时看 `oracle_top10_recall`，它衡量模型 Top 10% 是否抓住真实高 uplift 人群。"
        frontier_note = ""
        if "frontier_metric_schema" in compare_ranking.columns and compare_ranking["frontier_metric_schema"].fillna("").astype(str).str.len().gt(0).any():
            frontier_cols = [
                "frontier_metric_schema",
                "full_funnel_top10_conversion_uplift",
                "delayed_d30_top10_uplift",
                "delayed_censored_rate",
            ]
            frontier_view = compare_ranking[[col for col in frontier_cols if col in compare_ranking.columns]].head(5).to_dict(orient="records")
            frontier_note = (
                "\n\n本轮含 P2 工业指标摘要："
                f"`{frontier_view}`。Full-funnel 看 conversion/click stage uplift，delayed feedback 看 D30 Top-K 和 censoring。"
            )
        return (
            f"当前排名第一的是 `{best.get('model')}`，QINI 为 `{best.get('qini')}`，AUUC 为 `{best.get('auuc')}`。\n\n"
            f"前 5 名摘要：`{view}`\n\n"
            "选择模型时不要只看 QINI：如果 Top 10% observed uplift 更稳、Calibration MAE 更低，通常更适合真实投放。"
            f"{oracle_note}{frontier_note}\n\n"
            f"第一名模型卡片：\n{format_model_card(str(best.get('model')))}"
        )

    if any(key in text for key in ["qini", "auuc", "增量利润", "incremental profit", "cost-aware", "成本收益", "离线指标", "roi"]):
        rows = policy_metric_disagreement_rows()
        examples = "\n".join(
            f"- {row['case']}: {row['offline_signal']}；{row['business_assumption']}；policy value `{row['policy_value']}`，结论：{row['decision']}"
            for row in rows
        )
        return (
            "### 为什么 QINI/AUUC 不等于可上线收益\n\n"
            "QINI/AUUC 衡量 ranking 是否能更早找到增量用户，但上线是一个成本收益阈值问题。"
            "同一个 uplift score，在发券、广告、push、LLM routing 下的 value/cost 完全不同，所以最终要看 policy value 或 incremental profit。\n\n"
            f"{examples}\n\n"
            "UI 证据：`Evaluation -> metric map` 解释 QINI/AUUC/uplift@K/policy value，`Policy -> Metric Disagreement Lab` 展示离线指标和增量利润可能不一致。"
        )

    if any(key in text for key in ["打分", "predict", "scoring", "名单", "导出"]):
        return (
            "现在可以在 `Predict` 页选择历史 run，然后上传新 CSV 或使用当前数据打分。\n\n"
            "我会复用该 run 的 `preprocessor.pkl` 和 `model.pt/model.pkl`，输出 `y0_pred`、`y1_pred`、`uplift_score`、排序名次，并提供 Top 5%/10%/20% 人群下载。"
        )

    if any(key in text for key in ["收益", "policy", "投放", "成本", "value", "阈值"]):
        if policy_context:
            best = policy_context.get("best") or {}
            constrained = policy_context.get("constrained_best") or {}
            ranking_col = policy_context.get("ranking_col", "net_value")
            if constrained:
                scenario_note = ""
                if policy_context.get("scenario"):
                    scenario_note = (
                        f"- 业务场景：`{policy_context.get('scenario')}`，"
                        f"使用场景化 ROI 假设：`{policy_context.get('scenario_helper_enabled')}`。\n"
                    )
                return (
                    "当前 Policy 页已经完成约束投放模拟。\n\n"
                    f"{scenario_note}"
                    f"- 无约束最优：Top `{best.get('top_fraction', 0):.0%}`，`{ranking_col}` = `{best.get(ranking_col)}`。\n"
                    f"- 约束后推荐：Top `{constrained.get('top_fraction', 0):.0%}`，触达 `{int(constrained.get('rows', 0))}` 人，"
                    f"`{ranking_col}` = `{constrained.get(ranking_col)}`。\n"
                    f"- 预算上限：`{policy_context.get('budget_cap')}`，最大触达：`{policy_context.get('max_contacts')}`，"
                    f"单人成本：`{policy_context.get('contact_cost')}`。\n\n"
                    "解释：如果约束后 Top-K 小于无约束最优，说明预算或触达上限正在截断投放规模；此时应导出 `Download constrained audience`，"
                    "并保留 holdout/灰度监控，而不是盲目扩大到无约束阈值。"
                )
            return (
                "Policy 页已经计算了净收益曲线，但当前预算/触达约束下没有可用阈值。"
                "建议放宽 `Budget cap` 或 `Max contacts`，或重新检查单次触达成本是否过高。"
            )
        return (
            "到 `Policy` 页输入单次触达成本和单次转化收益后，我会按 Top-K 人群计算净收益曲线。\n\n"
            "如果数据里有真实 `treatment/outcome`，优先用 observed net value 找阈值；如果是新打分数据，则用 predicted net value 给出推荐投放比例。"
        )

    if any(key in text for key in ["分群", "segment", "subgroup", "规则", "distill", "蒸馏", "cdt"]):
        if result is None:
            return (
                "还没有训练结果。训练任意二元 uplift 模型后，`Evaluation` 页会给出两类分群：普通特征分箱 Segment Analysis，"
                "以及 Causal Distillation Segments，把黑盒 uplift 排名蒸馏成浅层树规则。"
            )
        distilled = pd.DataFrame(result.curves.get("distilled_segments", []))
        if distilled.empty:
            return "这次 run 没有生成可用的蒸馏分群，通常是样本量太小、特征不足，或 uplift score 几乎没有变化。"
        top_rules = distilled.head(5)[["rule", "rows", "mean_predicted_uplift", "observed_uplift"]].to_dict(orient="records")
        return (
            "我已经在 `Evaluation` 页生成 Causal Distillation Segments。\n\n"
            f"Top 规则摘要：`{top_rules}`\n\n"
            "这些规则适合解释和业务复盘，但上线前要看 support、treated/control 覆盖、observed uplift，以及 bootstrap/holdout 稳定性。"
        )

    if any(key in text for key in ["置信", "bootstrap", "ci", "稳定", "敏感", "sensitivity", "refutation", "placebo", "稳健"]):
        if result is not None and (result.eval_metrics.get("sensitivity") or {}):
            sensitivity = result.eval_metrics.get("sensitivity") or {}
            tests = sensitivity.get("tests") or {}
            random_qini = ((tests.get("random_score") or {}).get("qini_score") or {})
            perm_qini = ((tests.get("permuted_treatment") or {}).get("qini_score") or {})
            return (
                f"最近一次 sensitivity/refutation 结论：`{sensitivity.get('verdict')}`。\n\n"
                f"{sensitivity.get('message')}\n\n"
                f"- Random score QINI null p95: `{random_qini.get('null_p95')}`，excess: `{random_qini.get('excess_over_p95')}`\n"
                f"- Permuted treatment QINI null p95: `{perm_qini.get('null_p95')}`，excess: `{perm_qini.get('excess_over_p95')}`\n\n"
                "如果没有同时超过这些 null checks，我会把结果标记为探索性，不建议直接上线。"
            )
        return (
            "侧边栏的 `Bootstrap samples` 可以打开置信区间评估，`Sensitivity samples` 可以打开随机排序和 treatment permutation 的 refutation checks。\n\n"
            "建议快速试验：Bootstrap 30，Sensitivity 20；正式对比：Bootstrap 100-300，Sensitivity 100。"
        )

    if any(key in text for key in ["数据", "dataset", "列", "特征", "样本"]):
        return (
            f"当前数据有 `{len(df):,}` 行、`{df.shape[1]}` 列。\n\n"
            f"- treatment: `{treatment_col}`，分布约为 `{treatment_dist}`\n"
            f"- outcome: `{outcome_col}`，唯一值数量 `{df[outcome_col].dropna().nunique()}`\n"
            f"- features: `{len(feature_cols)}` 个\n"
            f"- categorical features: `{categorical_cols or []}`\n\n"
            "如果 treatment/control 比例很偏，建议把 `CFRNet` 纳入对比。"
        )

    if any(key in text for key in ["训练", "train", "参数", "epoch", "batch", "learning"]):
        presets = model_parameter_presets(model_name)
        preset_hint = "\n".join(
            f"- `{name}`: `{json.dumps(params, ensure_ascii=False)}`" for name, params in list(presets.items())[:4]
        )
        return (
            f"当前训练配置：模型 `{model_name}`，epochs `{epochs}`，batch size `{batch_size}`，learning rate `{learning_rate}`。\n\n"
            f"可用参数预设：\n{preset_hint}\n\n"
            "建议顺序：\n"
            "1. 先用 1-2 epochs 跑通流程。\n"
            "2. 确认 QINI/AUUC 曲线不是明显坏掉。\n"
            "3. 再提高到 5-20 epochs 做模型对比。\n"
            "4. 对大数据先限制 `Load rows`，确认配置没问题后再放大。"
        )

    if any(key in text for key in ["结果", "评估", "qini", "auuc", "top", "interpret", "解释"]):
        if result is None:
            return "还没有训练结果。先到 `Train` 页点 `Start training`，训练完成后我可以帮你解释 QINI、AUUC、Top-K 和分箱结果。"
        metrics = result.eval_metrics
        top_k = metrics.get("top_k", [])
        top10 = next((row for row in top_k if abs(row.get("top_fraction", 0) - 0.1) < 1e-6), None)
        top_text = ""
        if top10:
            top_text = f"\nTop 10% 观测 uplift 约为 `{top10.get('observed_uplift')}`，平均预测 uplift 约为 `{top10.get('mean_predicted_uplift')}`。"
        oracle_rows = (metrics.get("oracle_top_k") or {}).get("rows") or []
        oracle_top10 = next((row for row in oracle_rows if abs(row.get("top_fraction", 0) - 0.1) < 1e-6), None)
        oracle_text = ""
        if oracle_top10:
            oracle_text = f"\nOracle Top 10% recall 为 `{oracle_top10.get('topk_recall')}`，gain capture 为 `{oracle_top10.get('oracle_gain_capture')}`。"
        business_rows = (metrics.get("business_score_alignment") or {}).get("rows") or []
        business_top10 = next((row for row in business_rows if abs(row.get("top_fraction", 0) - 0.1) < 1e-6), None)
        business_text = ""
        if business_top10:
            business_text = (
                f"\nBusiness score Top 10% overlap recall 为 `{business_top10.get('topk_overlap_recall')}`，"
                f"score capture 为 `{business_top10.get('business_score_capture')}`。"
            )
        full_funnel_rows = (metrics.get("full_funnel") or {}).get("top_k") or []
        ff_top10 = next(
            (
                row
                for row in full_funnel_rows
                if abs(row.get("top_fraction", 0) - 0.1) < 1e-6 and row.get("stage") == "conversion"
            ),
            None,
        )
        full_funnel_text = ""
        if ff_top10:
            full_funnel_text = f"\nFull-funnel Top 10% conversion-stage uplift 为 `{ff_top10.get('observed_uplift')}`。"
        delayed_rows = (metrics.get("delayed_feedback") or {}).get("top_k") or []
        delayed_top10 = next(
            (
                row
                for row in delayed_rows
                if abs(row.get("top_fraction", 0) - 0.1) < 1e-6
                and row.get("window_col") in {"converted_d30", "converted_d14"}
            ),
            None,
        )
        delayed_text = ""
        if delayed_top10:
            delayed_text = f"\nDelayed-feedback Top 10% {delayed_top10.get('window')} uplift 为 `{delayed_top10.get('observed_uplift')}`。"
        readiness = decision_readiness(metrics)
        blocker_text = ""
        if readiness["blockers"]:
            blocker_text = "\n\n关键阻塞：\n" + "\n".join(f"- {item}" for item in readiness["blockers"][:3])
        return (
            f"最近一次结果：QINI `{metrics.get('qini_score')}`，AUUC `{metrics.get('auuc_score')}`。"
            f"{top_text}{oracle_text}{business_text}{full_funnel_text}{delayed_text}\n\n"
            f"Decision Readiness：`{readiness['score']}/100`，级别 `{readiness['level']}`。{blocker_text}\n\n"
            "解读时优先看：模型 QINI 是否高于 random、Top-K 人群 uplift 是否为正、bin 表是否呈现越靠前 uplift 越高。\n\n"
            "知识库提醒：Calibration、Bootstrap CI、Policy Value、Full-Funnel ECUP、Delayed Feedback 和 Segment Analysis 要一起看，单个 QINI 胜出不等于可以上线。\n\n"
            "证据入口：`Evaluation` 页展示指标表，`runs/{run_id}/metrics.json` 保存 full_funnel / delayed_feedback，代码在 `deepuplift/core/evaluator.py`。"
        )

    if any(key in text for key in ["下一步", "next", "功能", "todo", "加"]):
        return (
            "下一步我建议按这个顺序加：\n"
            "1. `Multi-treatment`：继续扩展 DR/Forest 官方实现和 action-specific policy value。\n"
            "2. `Continuous treatment`：支持剂量响应曲线和最优 treatment 强度。\n"
            "3. `Sensitivity Analysis`：加入 DoWhy 风格 placebo/refuter 和 hidden confounding bounds。\n"
            "4. `Knowledge Refresh`：把本地 causal-inference 报告自动增量写入知识库。"
        )

    if any(key in text for key in ["测试", "test", "qa", "验收", "检查清单", "checklist"]):
        return (
            "当前平台的最小验收清单：\n\n"
            "- `Data`：切换示例数据，确认 treatment/outcome/features 自动识别，图表不空白。\n"
            "- `Diagnostics`：检查 treatment design、overlap、feature balance 是否给出告警。\n"
            "- `Train`：用 `TLearnerLightGBM` 跑 300-1000 行，确认生成 run、predictions、run note、readiness JSON。\n"
            "- `Compare`：跑 2-4 个模型，确认 ranking、status cards、compare report、compare readiness JSON、bundle manifest 都可下载。\n"
            "- `Evaluation`：确认指标解释、Model Rationale、Decision Readiness、Top-K、Policy、Sensitivity、Overlap Trimming 可读。\n"
            "- `Predict`：选择历史 run 给当前数据打分，确认 Top 5%/10%/20% 可导出。\n"
            "- `Policy`：设置 `Budget cap` 或 `Max contacts`，确认 constrained recommendation 和 constrained audience 下载。\n"
            "- `Agent`：问 `推荐哪个模型？`、`投放阈值怎么解释？`、`测试清单`，确认回答带上下文。\n\n"
            "命令行 smoke：`PYTHON_BIN=python3 scripts/smoke_test_agent.sh`。完整文档在 `Knowledge` 页的 `End-to-End Test Guide`。"
        )

    return (
        "你可以问我这些：\n"
        "- 推荐哪个模型？\n"
        "- 当前数据怎么样？\n"
        "- 诊断一下数据？\n"
        "- 模型对比结果怎么看？\n"
        "- 训练参数怎么设？\n"
        "- 解释一下结果。\n"
        "- 下一步加什么功能？"
    )


AGENT_EVIDENCE_CARDS = [
    {
        "keywords": ["字节", "火山", "volcengine", "dataleap", "bytehouse", "datafinder", "datatester", "datawind", "vedi", "公开产品", "增长分析", "ab测试", "a/b"],
        "claim": "The public-product UI references are source-gated against public Volcengine/ByteDance docs and articles, then translated into DeepUplift workbench patterns without copying private UI.",
        "ui_pages": "Workflow: Growth Decision Control Plane; Models: ByteDance-style Public Product Inspiration Matrix; Policy: Launch Guardrail Board; Evidence: Public Product Source Gate; Agent: 公开产品参考",
        "docs": "docs/PUBLIC_PRODUCT_UI_RESEARCH.md; docs/UI_UX_IMPROVEMENT_LOG.md; docs/DEEPUplift_AGENT_VISUAL_DESIGN_SYSTEM.md",
        "code_paths": "deepuplift/core/public_product_inspiration.py; app.py; scripts/smoke_ui_screenshots.py",
        "source_quality": "official Volcengine docs, official developer articles and public product pages only; no private UI copying",
        "validation": "py_compile plus UI screenshot smoke across Workflow/Scenario/Models/Evidence/Agent/Story/Policy.",
    },
    {
        "keywords": ["训练平台", "mlflow", "wandb", "w&b", "clearml", "kubeflow", "aim", "mlops", "experiment tracking", "实验追踪", "training workbench"],
        "claim": "The workbench structure is source-gated against mature ML training platforms: experiments, runs, registry, artifacts, lineage and gates are visible in Streamlit.",
        "ui_pages": "Workflow: ML Training Workbench Overview; Models: ML Training Platform Inspiration Matrix; Policy: Uplift Application Expansion Matrix; Evidence: ML Training Platform Source Gate",
        "docs": "docs/ML_TRAINING_PLATFORM_UI_RESEARCH.md; docs/UI_UX_IMPROVEMENT_LOG.md; docs/DEEPUplift_AGENT_VISUAL_DESIGN_SYSTEM.md",
        "code_paths": "deepuplift/core/ml_training_platform_research.py; app.py; scripts/smoke_ui_screenshots.py",
        "source_quality": "official GitHub repos, official documentation, and official commit feeds when GitHub API is rate-limited",
        "validation": "py_compile plus UI screenshot smoke across Workflow/Scenario/Models/Evidence/Agent/Story/Policy.",
    },
    {
        "keywords": ["frontend", "前端", "ui/ux", "streamlit", "react", "fastapi", "shadcn", "tremor", "echarts", "visual", "视觉", "框架审计"],
        "claim": "The UI upgrade is source-gated: official GitHub repos and official docs are audited before a pattern enters Streamlit or the future React backlog.",
        "ui_pages": "Models: Frontend Framework Audit / UI Inspiration Matrix; Story: Visual Upgrade Before / After; Evidence: UI Screenshot Gallery / Visual Regression Evidence; Agent: Quick Prompt Groups",
        "docs": "docs/FRONTEND_FRAMEWORK_RESEARCH.md; docs/UI_UX_IMPROVEMENT_LOG.md; docs/DEEPUplift_AGENT_VISUAL_DESIGN_SYSTEM.md",
        "code_paths": "deepuplift/core/frontend_research.py; app.py; scripts/smoke_ui_screenshots.py",
        "source_quality": "official GitHub repos, official documentation, mature OSS dashboard and agent workbench projects",
        "validation": "py_compile plus UI screenshot smoke across Workflow/Scenario/Models/Evidence/Agent/Story/Policy.",
    },
    {
        "keywords": ["qini", "auuc", "top-k", "top k", "policy value", "calibration", "bootstrap", "ci", "评估指标", "指标", "结果解释"],
        "claim": "Evaluation is a metric system: QINI/AUUC for ranking, Top-K for launch buckets, policy value for ROI, calibration/CI/overlap for trust.",
        "ui_pages": "Evaluation: metric guide and latest run snapshot; Compare: ranking; Policy: threshold and ROI; Evidence: metrics artifacts",
        "docs": "docs/INTERVIEW_DIFFICULTIES_AND_SOLUTIONS.md; docs/UPLIFT_MODEL_SELECTION_PLAYBOOK.md; docs/DEEPUplift_AGENT_OFFLINE_ONLINE_PLAYBOOK.md",
        "code_paths": "deepuplift/core/evaluator.py; deepuplift/core/policy.py; deepuplift/core/readiness.py; deepuplift/core/trainer.py",
        "source_quality": "local code + generated run artifacts + regression smoke",
        "validation": "scripts/smoke_ui_screenshots.py checks Evaluation; scripts/full_regression_check.sh validates training, compare and evidence artifacts.",
    },
    {
        "keywords": ["utboost", "efin", "descn", "umlc", "ecup", "cfr-df", "delayed feedback", "frontier", "最新模型", "模型能力", "causal forest", "dml", "dr learner"],
        "claim": "The model catalog is layered into P0 industrial baselines, P1 advanced plugins and P2 frontier business extensions instead of a flat model-name list.",
        "ui_pages": "Models: Industrial Model Layer Architecture and Frontier Plugin Backlog; Evidence: Frontier Model Source Gate; Evaluation: delayed/full-funnel metrics",
        "docs": "docs/UPLIFT_MODEL_SELECTION_PLAYBOOK.md; docs/INDUSTRIAL_UPLIFT_CASE_STUDIES.md",
        "code_paths": "deepuplift/core/frontier_models.py; deepuplift/core/external_models.py; deepuplift/core/sklearn_models.py; deepuplift/core/registry.py",
        "source_quality": "official/open-source docs and papers are source-gated before optional adapters enter default training",
        "validation": "scripts/smoke_frontier_model_capabilities.py and UI screenshot smoke check the frontier matrix.",
    },
    {
        "keywords": ["spotify", "in-app messaging", "站内消息", "应用内消息"],
        "claim": "Growth messaging needs CATE/uplift plus offline policy evaluation and online holdout validation.",
        "ui_pages": "Story: Fresh Industrial Additions; Knowledge: Industrial Knowledge Map; Agent: this Q&A",
        "docs": "docs/UPLIFT_BUSINESS_SCENARIOS.md; docs/INDUSTRIAL_UPLIFT_CASE_STUDIES.md",
        "validation": "scripts/smoke_industry_agent_answers.py checks Spotify holdout/CATE wording.",
    },
    {
        "keywords": ["airbnb", "incremental ltv", "lifetime value", "cannibalization", "长期价值"],
        "claim": "Marketplace ROI should use incremental LTV, cannibalization checks and long-window value, not only short-term conversion.",
        "ui_pages": "Story: Fresh Industrial Additions; Policy: Scenario ROI Helper; Knowledge: Industrial Knowledge Map",
        "docs": "docs/UPLIFT_COUPON_ADS_GROWTH_PLAYBOOK.md; docs/UPLIFT_ONLINE_EXPERIMENT_AND_INCREMENTALITY.md",
        "validation": "scripts/smoke_industry_agent_answers.py checks Airbnb incremental LTV/cannibalization wording.",
    },
    {
        "keywords": ["阿里", "alibaba", "x-learner", "xlearner", "异质性", "edgerec", "请求价值"],
        "claim": "AB-test heterogeneity and request uplift turn uplift modeling into subsidy, recommendation and system-cost decisions.",
        "ui_pages": "Story: Fresh Industrial Additions; Knowledge: Industrial Knowledge Map; Models: Scenario Model Matrix",
        "docs": "docs/INDUSTRIAL_UPLIFT_CASE_STUDIES.md; docs/UPLIFT_BUSINESS_SCENARIOS.md",
        "validation": "industry source audit accepts official developer articles and Agent smoke checks Alibaba/EdgeRec answers.",
    },
    {
        "keywords": ["journey", "customer journey", "营销旅程", "旅程重叠", "多旅程", "协同", "蚕食"],
        "claim": "Overlapping CRM journeys need pure/global lift separation, cannibalization checks and combinatorial treatment thinking.",
        "ui_pages": "Knowledge: Industrial Knowledge Map; Models: CRM lifecycle intervention tag; Policy: Growth Push ROI template",
        "docs": "docs/UPLIFT_ONLINE_EXPERIMENT_AND_INCREMENTALITY.md; docs/UPLIFT_BUSINESS_SCENARIOS.md",
        "validation": "industry Agent smoke checks journey overlap answer terms.",
    },
    {
        "keywords": ["llm", "大模型", "模型路由", "llm routing", "强模型", "小模型", "文案", "creative", "prompt"],
        "claim": "LLM decisions become causal decisions when the platform asks whether escalation, retrieval, tool use, or generated creative creates incremental value.",
        "ui_pages": "Story: LLM + Uplift Frontier; Knowledge: LLM + Causal Frontier Map; Policy: LLM Routing ROI Simulator",
        "docs": "docs/UPLIFT_LLM_CAUSAL_FRONTIER.md; docs/LLM_ROUTING_POLICY_PLAYBOOK.md; docs/UPLIFT_LLM_INTERVIEW_QA.md",
        "validation": "industry Agent smoke checks LLM routing/text-treatment answers and source audit covers the frontier references.",
    },
    {
        "keywords": ["发券", "coupon", "补贴", "羊毛党", "套利", "sure thing"],
        "claim": "Coupon decisions need incremental gross profit, subsidy cost and abuse/fatigue guards instead of raw conversion rate.",
        "ui_pages": "Policy: Scenario ROI Helper; Models: Scenario Model Matrix; Story: Coupon / Subsidy",
        "docs": "docs/UPLIFT_COUPON_ADS_GROWTH_PLAYBOOK.md; docs/UPLIFT_MODEL_SELECTION_PLAYBOOK.md",
        "validation": "source quality audit + industry Agent smoke cover coupon ROI answers.",
    },
    {
        "keywords": ["广告", "pctr", "pcvr", "incrementality", "geo", "roas", "iroas"],
        "claim": "Ads uplift estimates causal incremental value beyond attribution-driven pCTR/pCVR predictions.",
        "ui_pages": "Story: Industrial Case Studies; Policy: Ads ROI template; Models: Ads scenario tag",
        "docs": "docs/UPLIFT_ONLINE_EXPERIMENT_AND_INCREMENTALITY.md; docs/UPLIFT_BUSINESS_SCENARIOS.md",
        "validation": "industry source audit requires Tencent/Google/JD/Amazon evidence coverage.",
    },
    {
        "keywords": ["marketplace", "spillover", "sutva", "interference", "供需", "干扰"],
        "claim": "Marketplace treatments can violate SUTVA, so policy evaluation needs spillover-aware experiment design.",
        "ui_pages": "Story: Marketplace Subsidy; Models: Marketplace pricing / incentive tag; Policy: Marketplace ROI template",
        "docs": "docs/UPLIFT_SCENARIO_MODEL_MATRIX.md; docs/UPLIFT_ONLINE_EXPERIMENT_AND_INCREMENTALITY.md",
        "validation": "industry Agent smoke checks SUTVA/spillover/multi-treatment answer terms.",
    },
    {
        "keywords": ["selection bias", "ssb", "overlap", "观测数据", "有偏", "偏差"],
        "claim": "Observational uplift needs overlap, propensity and selection-bias diagnostics before model ranking is trusted.",
        "ui_pages": "Diagnostics: Overlap and SSB Proxy; Evaluation: trimming/sensitivity; Story: Interview Difficulty Proof Map",
        "docs": "docs/INTERVIEW_DIFFICULTIES_AND_SOLUTIONS.md; docs/UPLIFT_MODEL_SELECTION_PLAYBOOK.md",
        "validation": "full_regression_check runs diagnostic, sensitivity and interview-material audits.",
    },
]


def agent_evidence_card(content: str) -> dict[str, str]:
    lowered = content.lower()
    for card in AGENT_EVIDENCE_CARDS:
        if any(keyword.lower() in lowered for keyword in card["keywords"]):
            payload = {key: value for key, value in card.items() if key != "keywords"}
            payload.setdefault(
                "code_paths",
                "app.py; deepuplift/core/trainer.py; deepuplift/core/evaluator.py; deepuplift/core/model_cards.py; deepuplift/core/industry_playbook.py",
            )
            payload.setdefault("source_quality", "curated local docs + official/paper references where applicable")
            return payload
    return {
        "claim": "Agent answers should be backed by visible UI sections, generated docs and smoke/regression artifacts.",
        "ui_pages": "Story: Evidence Index; Knowledge: Latest Regression Health; Models: Model Card; Policy: Policy Value",
        "docs": "docs/DEEPUplift_AGENT_EVIDENCE_INDEX.md; docs/UPLIFT_AGENT_INTERVIEW_RUNBOOK.md",
        "code_paths": "app.py; deepuplift/core/trainer.py; deepuplift/core/evaluator.py; deepuplift/core/predictor.py",
        "source_quality": "local UI + generated evidence pack + smoke/regression artifacts",
        "validation": "demo_lite_check and smoke_ui_screenshots validate the demo-critical pages.",
    }


def render_agent_evidence_card(content: str) -> None:
    card = agent_evidence_card(content)
    rows = [{"field": key.replace("_", " ").title(), "evidence": value} for key, value in card.items()]
    st.caption("Agent Evidence Card")
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def format_agent_transcript(
    messages: list[dict[str, str]],
    *,
    dataset_label: str,
    model_name: str,
    task: str,
    treatment_col: str,
    outcome_col: str,
    feature_cols: list[str],
) -> str:
    lines = [
        "# DeepUplift Agent Transcript",
        "",
        f"- generated_at: `{time.strftime('%Y-%m-%d %H:%M:%S %z')}`",
        f"- dataset: `{dataset_label}`",
        f"- task: `{task}`",
        f"- model: `{model_name}`",
        f"- treatment_col: `{treatment_col}`",
        f"- outcome_col: `{outcome_col}`",
        f"- feature_count: `{len(feature_cols)}`",
        "",
        "## Messages",
        "",
    ]
    for idx, message in enumerate(messages, start=1):
        role = message.get("role", "unknown")
        content = message.get("content", "").strip()
        lines.extend([f"### {idx}. {role}", "", content or "_empty_", ""])
        if role == "assistant":
            lines.extend(["#### Agent Evidence Card", ""])
            for key, value in agent_evidence_card(content).items():
                lines.append(f"- **{key.replace('_', ' ').title()}**: {value}")
            lines.append("")
    return "\n".join(lines).strip() + "\n"


def render_data_diagnostics(df: pd.DataFrame, treatment_col: str, outcome_col: str, feature_cols: list[str]) -> None:
    render_stat_grid(
        [
            ("Rows", f"{len(df):,}", False),
            ("Columns", f"{df.shape[1]:,}", False),
            ("Features", f"{len(feature_cols):,}", False),
            ("Outcome Values", f"{df[outcome_col].dropna().nunique():,}", False),
        ]
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Treatment")
        treatment_counts = df[treatment_col].value_counts(dropna=False).rename_axis("value").reset_index(name="rows")
        st.bar_chart(treatment_counts, x="value", y="rows")
    with right:
        st.subheader("Outcome")
        if df[outcome_col].dropna().nunique() <= 20:
            outcome_counts = df[outcome_col].value_counts(dropna=False).rename_axis("value").reset_index(name="rows")
            st.bar_chart(outcome_counts, x="value", y="rows")
        else:
            st.line_chart(df[outcome_col].dropna().reset_index(drop=True))

    st.subheader("Missing Rate")
    missing = df[feature_cols + [treatment_col, outcome_col]].isna().mean().sort_values(ascending=False)
    st.dataframe(missing.rename("missing_rate").reset_index().rename(columns={"index": "column"}), width="stretch")


def render_result(result) -> None:
    metrics = result.eval_metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("QINI", f"{metrics.get('qini_score') or 0:.4f}")
    col2.metric("AUUC", f"{metrics.get('auuc_score') or 0:.4f}")
    col3.metric("Calibration MAE", f"{metrics.get('calibration_mae') or 0:.4f}")
    response = metrics.get("response", {})
    col4.metric("AUC Y0", f"{response.get('auc_y0') or 0:.4f}")
    col5.metric("AUC Y1", f"{response.get('auc_y1') or 0:.4f}")

    st.subheader("Model Rationale")
    render_model_rationale(result.config.get("model_name", "model"), metrics=metrics)

    st.subheader("Decision Readiness")
    render_decision_readiness(metrics)

    bootstrap = metrics.get("bootstrap") or {}
    if bootstrap:
        st.caption(
            f"Bootstrap samples: {bootstrap.get('n_success', 0)}/{bootstrap.get('n_bootstrap', 0)} | "
            f"QINI {bootstrap_text(metrics, 'qini_score')} | AUUC {bootstrap_text(metrics, 'auuc_score')}"
        )

    sensitivity = metrics.get("sensitivity") or {}
    if sensitivity:
        st.subheader("Sensitivity Checks")
        s1, s2, s3 = st.columns(3)
        s1.metric("Verdict", str(sensitivity.get("verdict", "unknown")).upper())
        s2.metric("Permutations", f"{sensitivity.get('n_permutations', 0):,}")
        observed = sensitivity.get("observed") or {}
        s3.metric("Observed QINI", f"{observed.get('qini_score') or 0:.4f}")
        st.caption(sensitivity.get("message", ""))
        sensitivity_rows = []
        for test_name, payload in (sensitivity.get("tests") or {}).items():
            for metric_name, summary in payload.items():
                sensitivity_rows.append({"test": test_name, "metric": metric_name, **summary})
        if sensitivity_rows:
            st.dataframe(pd.DataFrame(sensitivity_rows), width="stretch")

    overlap_trim = metrics.get("overlap_trim") or {}
    if overlap_trim:
        st.subheader("Overlap Trimming")
        if overlap_trim.get("error"):
            st.warning(overlap_trim["error"])
        else:
            propensity = overlap_trim.get("propensity") or {}
            o1, o2, o3, o4 = st.columns(4)
            o1.metric("Propensity p05", f"{propensity.get('p05') or 0:.4f}")
            o2.metric("Propensity Median", f"{propensity.get('median') or 0:.4f}")
            o3.metric("Propensity p95", f"{propensity.get('p95') or 0:.4f}")
            o4.metric("Weak Overlap", f"{propensity.get('weak_overlap_rate') or 0:.1%}")
            trim_rows = pd.DataFrame(overlap_trim.get("rows", []))
            if not trim_rows.empty:
                st.dataframe(trim_rows, width="stretch")

    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.subheader("QINI Curve")
        st.caption("Ranking curve: cumulative incremental gain versus random targeting. Use it to check whether high-score users are enriched for treatment effect.")
        qini = pd.DataFrame(result.curves["qini"])
        if not qini.empty:
            st.line_chart(qini.set_index("sample"))
    with chart_right:
        st.subheader("AUUC Curve")
        st.caption("Cumulative uplift curve: useful for comparing ranking quality across targeting fractions, but not a replacement for ROI.")
        auuc = pd.DataFrame(result.curves["auuc"])
        if not auuc.empty:
            st.line_chart(auuc.set_index("sample"))

    st.subheader("Top-K Targeting Metrics")
    st.caption("Top-K observed uplift is the bucket-level signal closest to a real launch list. Treat small treated/control cells as noisy.")
    st.dataframe(pd.DataFrame(metrics.get("top_k", [])), width="stretch")

    oracle_top_k = metrics.get("oracle_top_k") or {}
    oracle_rows = pd.DataFrame(oracle_top_k.get("rows", []))
    if not oracle_rows.empty:
        st.subheader(f"Oracle Top-K Recall ({oracle_top_k.get('oracle_col')})")
        st.dataframe(oracle_rows, width="stretch")

    business_alignment = metrics.get("business_score_alignment") or {}
    business_alignment_rows = pd.DataFrame(business_alignment.get("rows", []))
    if not business_alignment_rows.empty:
        st.subheader(f"Business Score Top-K Alignment ({business_alignment.get('business_col')})")
        st.caption("Compares uplift Top-K with an existing business, delivery, online, or DM score column when present.")
        st.dataframe(business_alignment_rows, width="stretch")

    full_funnel = metrics.get("full_funnel") or {}
    if full_funnel.get("top_k") or full_funnel.get("overall"):
        st.subheader("Full-Funnel ECUP Metrics")
        st.caption(
            "For impression-click-conversion logs, this breaks uplift into funnel stages so ads, coupon and recommendation teams can see where incremental value is created."
        )
        ff1, ff2, ff3 = st.columns(3)
        ff1.metric("Stages", ", ".join(full_funnel.get("stage_cols") or []) or "NA")
        ff2.metric("Mean Media Cost", f"{full_funnel.get('media_cost_mean') or 0:.4f}")
        ff3.metric("Mean Order Value", f"{full_funnel.get('order_value_mean') or 0:.4f}")
        overall = pd.DataFrame(full_funnel.get("overall", []))
        if not overall.empty:
            st.dataframe(overall, width="stretch", hide_index=True)
        top_k = pd.DataFrame(full_funnel.get("top_k", []))
        if not top_k.empty:
            st.caption("Top-K stage uplift: the same score ranking evaluated at exposure, click and conversion stages.")
            st.dataframe(top_k, width="stretch", hide_index=True)
        oracle_stage = pd.DataFrame(full_funnel.get("oracle_top_k", []))
        if not oracle_stage.empty:
            with st.expander("Stage oracle capture"):
                st.dataframe(oracle_stage, width="stretch", hide_index=True)
        policy_stage = pd.DataFrame(full_funnel.get("policy_top_k", []))
        if not policy_stage.empty:
            with st.expander("Full-funnel policy value capture"):
                st.dataframe(policy_stage, width="stretch", hide_index=True)

    delayed_feedback = metrics.get("delayed_feedback") or {}
    if delayed_feedback.get("top_k") or delayed_feedback.get("overall"):
        st.subheader("Delayed Feedback Uplift Metrics")
        st.caption(
            "For ads, lifecycle CRM and retention, this evaluates D1/D7/D14/D30 windows and surfaces censoring risk before a fast label drives the wrong policy."
        )
        summary = delayed_feedback.get("summary") or {}
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Windows", ", ".join(delayed_feedback.get("window_cols") or []) or "NA")
        d2.metric("Censored Rate", f"{summary.get('censored_rate') or 0:.1%}")
        d3.metric("Observed Delay p50", f"{summary.get('observed_conversion_delay_days_p50') or 0:.1f}")
        d4.metric("Observed Delay p90", f"{summary.get('observed_conversion_delay_days_p90') or 0:.1f}")
        delayed_overall = pd.DataFrame(delayed_feedback.get("overall", []))
        if not delayed_overall.empty:
            st.dataframe(delayed_overall, width="stretch", hide_index=True)
        delayed_top_k = pd.DataFrame(delayed_feedback.get("top_k", []))
        if not delayed_top_k.empty:
            st.caption("Top-K window uplift: compare short-window and mature-window treatment effects for the same launch bucket.")
            st.dataframe(delayed_top_k, width="stretch", hide_index=True)
        delayed_oracle = pd.DataFrame(delayed_feedback.get("oracle_top_k", []))
        if not delayed_oracle.empty:
            with st.expander("Delayed-window oracle capture"):
                st.dataframe(delayed_oracle, width="stretch", hide_index=True)
        delayed_policy = pd.DataFrame(delayed_feedback.get("policy_top_k", []))
        if not delayed_policy.empty:
            with st.expander("Delayed-window policy value capture"):
                st.dataframe(delayed_policy, width="stretch", hide_index=True)

    industrial_policy = metrics.get("industrial_policy") or {}
    industrial_policy_rows = pd.DataFrame(industrial_policy.get("top_k", []))
    if not industrial_policy_rows.empty:
        st.subheader("Industrial Scenario Policy Metrics")
        st.caption(
            "Scenario-specific ROI signals: coupon incremental profit, recommendation negative-uplift risk, "
            "marketplace spillover/subsidy risk, and LLM routing cost-quality value."
        )
        top10_rows = pd.DataFrame(industrial_policy.get("top10", []))
        if not top10_rows.empty:
            st.dataframe(top10_rows, width="stretch", hide_index=True)
        with st.expander("All scenario policy Top-K buckets", expanded=False):
            st.dataframe(industrial_policy_rows, width="stretch", hide_index=True)

    st.subheader("Uplift by Score Bin")
    st.caption("A healthy ranking should usually show stronger observed uplift in higher score bins. Non-monotonic bins are a calibration and support warning.")
    st.dataframe(pd.DataFrame(result.curves["bins"]), width="stretch")

    st.subheader("Uplift Calibration")
    st.caption("Calibration compares predicted uplift buckets with observed uplift. It answers whether score thresholds are trustworthy, not just whether ranking is non-random.")
    calibration = pd.DataFrame(result.curves.get("calibration", []))
    if not calibration.empty:
        st.dataframe(calibration, width="stretch")

    st.subheader("Segment Analysis")
    segments = pd.DataFrame(result.curves.get("segments", []))
    if not segments.empty:
        st.dataframe(segments.head(50), width="stretch")
    else:
        st.info("No segment analysis available for this run.")

    st.subheader("Causal Distillation Segments")
    distilled_segments = pd.DataFrame(result.curves.get("distilled_segments", []))
    if not distilled_segments.empty:
        st.caption("A shallow tree distills the model's uplift ranking into readable subgroup rules. Treat these as hypotheses unless support and holdout uplift are stable.")
        st.dataframe(distilled_segments.head(20), width="stretch")
    else:
        st.info("No distilled segment tree available for this run.")

    st.subheader("Policy Value")
    st.caption("Policy value converts uplift into net business value using value and contact cost. This is the metric closest to coupon ROI, ad iROAS, push fatigue and LLM routing cost-quality decisions.")
    policy_value = pd.DataFrame(result.curves.get("policy_value", []))
    if not policy_value.empty:
        best_policy = metrics.get("policy_best") or {}
        if best_policy:
            st.caption(
                f"Best threshold: Top {best_policy.get('top_fraction', 0):.0%}, "
                f"net value: {best_policy.get('observed_net_value', best_policy.get('predicted_net_value'))}"
            )
        st.line_chart(policy_value.set_index("top_fraction")[["predicted_net_value"]])
        st.dataframe(policy_value, width="stretch")

    preview = pd.DataFrame(result.preview)
    st.subheader("Prediction Preview")
    st.dataframe(preview, width="stretch")
    st.download_button(
        "Download predictions",
        data=Path(result.artifacts["predictions"]).read_bytes(),
        file_name="uplift_predictions.csv",
        mime="text/csv",
    )
    run_note = result.artifacts.get("run_note")
    if run_note and Path(run_note).exists():
        st.download_button(
            "Download run note",
            data=Path(run_note).read_bytes(),
            file_name=f"{result.run_id}_run_note.md",
            mime="text/markdown",
        )
    readiness_path = result.artifacts.get("readiness")
    if readiness_path and Path(readiness_path).exists():
        st.download_button(
            "Download readiness JSON",
            data=Path(readiness_path).read_bytes(),
            file_name=f"{result.run_id}_readiness.json",
            mime="application/json",
        )


def load_run_history(base_dir: Path = Path("runs")) -> pd.DataFrame:
    rows = []
    if not base_dir.exists():
        return pd.DataFrame()

    run_dirs = sorted([path for path in base_dir.iterdir() if path.is_dir()], key=lambda path: path.stat().st_mtime, reverse=True)
    for run_dir in run_dirs:
        metrics_path = run_dir / "metrics.json"
        config_path = run_dir / "config.json"
        if not metrics_path.exists() or not config_path.exists():
            continue
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        treatment_values = config.get("treatment_values") or metrics.get("treatment_values") or []
        if len(treatment_values) > 2 or "recommended_policy_value_ipw" in metrics:
            continue
        if str(config.get("strategy", metrics.get("strategy", ""))).startswith("dose_response"):
            continue
        bootstrap_metrics = (metrics.get("bootstrap") or {}).get("metrics", {})
        qini_ci = bootstrap_metrics.get("qini_score") or {}
        policy_best = metrics.get("policy_best") or {}
        sensitivity = metrics.get("sensitivity") or {}
        overlap_trim = metrics.get("overlap_trim") or {}
        overlap_propensity = overlap_trim.get("propensity") or {}
        oracle_rows = (metrics.get("oracle_top_k") or {}).get("rows") or []
        oracle_top10 = next((row for row in oracle_rows if abs(row.get("top_fraction", 0) - 0.1) < 1e-6), {})
        business_rows = (metrics.get("business_score_alignment") or {}).get("rows") or []
        business_top10 = next((row for row in business_rows if abs(row.get("top_fraction", 0) - 0.1) < 1e-6), {})
        readiness = decision_readiness(metrics)
        user_note_path = run_dir / "user_note.json"
        user_note = {}
        if user_note_path.exists():
            try:
                user_note = json.loads(user_note_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                user_note = {}
        rows.append(
            {
                "run_id": run_dir.name,
                "run_dir": str(run_dir),
                "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(run_dir.stat().st_mtime)),
                "tags": ", ".join(user_note.get("tags", [])),
                "user_note": user_note.get("note", ""),
                "model": config.get("model_name"),
                "strategy": config.get("strategy", metrics.get("strategy")),
                "task": config.get("task"),
                "rows": config.get("max_rows"),
                "readiness_score": readiness.get("score"),
                "readiness_level": readiness.get("level"),
                "readiness_recommendation": readiness.get("recommendation"),
                "readiness_blockers": "; ".join(readiness.get("blockers") or []),
                "qini": metrics.get("qini_score"),
                "qini_ci_low": qini_ci.get("low"),
                "auuc": metrics.get("auuc_score"),
                "calibration_mae": metrics.get("calibration_mae"),
                "oracle_top10_recall": oracle_top10.get("topk_recall"),
                "business_top10_overlap": business_top10.get("topk_overlap_recall"),
                "business_top10_capture": business_top10.get("business_score_capture"),
                **frontier_metric_summary(metrics),
                "policy_best_fraction": policy_best.get("top_fraction"),
                "policy_best_net": policy_best.get("observed_net_value", policy_best.get("predicted_net_value")),
                "sensitivity_verdict": sensitivity.get("verdict"),
                "weak_overlap_rate": overlap_propensity.get("weak_overlap_rate"),
                "predictions": str(run_dir / "predictions.csv"),
                "readiness": str(run_dir / "readiness.json"),
                "run_note": str(run_dir / "run_note.md"),
            }
        )
    return pd.DataFrame(rows)


def load_multi_run_history(base_dir: Path = Path("runs")) -> pd.DataFrame:
    rows = []
    if not base_dir.exists():
        return pd.DataFrame()

    run_dirs = sorted([path for path in base_dir.iterdir() if path.is_dir()], key=lambda path: path.stat().st_mtime, reverse=True)
    for run_dir in run_dirs:
        metrics_path = run_dir / "metrics.json"
        config_path = run_dir / "config.json"
        if not metrics_path.exists() or not config_path.exists():
            continue
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        treatment_values = config.get("treatment_values") or metrics.get("treatment_values") or []
        is_multi = len(treatment_values) > 2 or "recommended_policy_value_ipw" in metrics
        if not is_multi:
            continue
        incremental_ci = (((metrics.get("policy_value_bootstrap") or {}).get("metrics") or {}).get("incremental_policy_value_ipw") or {})
        rows.append(
            {
                "run_id": run_dir.name,
                "run_dir": str(run_dir),
                "model": config.get("model_name"),
                "strategy": config.get("strategy", metrics.get("strategy", "multi_treatment")),
                "task": config.get("task"),
                "control_treatment": config.get("control_treatment", metrics.get("control_treatment")),
                "treatment_values": ", ".join(str(value) for value in treatment_values),
                "recommended_policy_value_ipw": metrics.get("recommended_policy_value_ipw"),
                "control_policy_value_ipw": metrics.get("control_policy_value_ipw"),
                "incremental_policy_value_ipw": metrics.get("incremental_policy_value_ipw"),
                "incremental_ci_low": incremental_ci.get("low"),
                "incremental_ci_high": incremental_ci.get("high"),
                "predictions": str(run_dir / "predictions.csv"),
                "run_note": str(run_dir / "run_note.md"),
            }
        )
    return pd.DataFrame(rows)


def load_dose_run_history(base_dir: Path = Path("runs")) -> pd.DataFrame:
    rows = []
    if not base_dir.exists():
        return pd.DataFrame()

    run_dirs = sorted([path for path in base_dir.iterdir() if path.is_dir()], key=lambda path: path.stat().st_mtime, reverse=True)
    for run_dir in run_dirs:
        metrics_path = run_dir / "metrics.json"
        config_path = run_dir / "config.json"
        if not metrics_path.exists() or not config_path.exists():
            continue
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        strategy = str(config.get("strategy", metrics.get("strategy", "")))
        if not strategy.startswith("dose_response"):
            continue
        best = metrics.get("average_best_dose") or {}
        rows.append(
            {
                "run_id": run_dir.name,
                "run_dir": str(run_dir),
                "model": config.get("model_name"),
                "task": config.get("task"),
                "baseline_dose": metrics.get("baseline_dose"),
                "mean_recommended_dose": best.get("mean_recommended_dose"),
                "mean_predicted_gain_vs_baseline": best.get("mean_predicted_gain_vs_baseline"),
                "predictions": str(run_dir / "predictions.csv"),
                "run_note": str(run_dir / "run_note.md"),
            }
        )
    return pd.DataFrame(rows)


st.set_page_config(page_title="DeepUplift Agent", layout="wide")
inject_theme()
render_header()

sample_datasets = load_dataset_manifest(str(DATASET_MANIFEST))
sample_names = [dataset["name"] for dataset in sample_datasets]
dataset_id_to_name = {dataset.get("id"): dataset.get("name") for dataset in sample_datasets}
frontier_demo_dataset_ids = {
    "Manual dataset": None,
    "Coupon ROI": "synthetic_coupon_profit_uplift_8k",
    "Baidu Waimai red packet": "synthetic_baidu_waimai_red_packet_8k",
    "Didi passenger subsidy": "synthetic_didi_passenger_subsidy_8k",
    "Didi driver supply": "synthetic_didi_driver_supply_subsidy_8k",
    "Didi budget allocation": "synthetic_didi_budget_allocation_5k",
    "Shopee ads iROAS": "synthetic_shopee_ads_full_funnel_8k",
    "ECUP full-funnel": "synthetic_ads_full_funnel_ecup_7k",
    "Delayed feedback": "synthetic_growth_delayed_feedback_7k",
    "Recommendation intervention": "synthetic_recommendation_intervention_7k",
    "Marketplace subsidy": "synthetic_marketplace_subsidy_uplift_7k",
    "LLM routing": "synthetic_llm_routing_uplift_8k",
    "LLM multi-action routing": "synthetic_llm_multi_action_routing_9k",
    "Spotify message diet": "synthetic_spotify_inapp_message_uplift_6k",
    "DoorDash ghost ads": "synthetic_doordash_ghost_ads_lift_6k",
    "Merchant subsidy": "synthetic_merchant_subsidy_margin_uplift_6k",
    "Game ops gift": "synthetic_game_ops_gift_uplift_6k",
    "FinTech credit line": "synthetic_fintech_credit_line_uplift_6k",
    "Customer service escalation": "synthetic_customer_service_escalation_uplift_6k",
    "Healthcare follow-up": "synthetic_healthcare_followup_uplift_6k",
    "SaaS retention": "synthetic_saas_retention_offer_uplift_6k",
}
selected_dataset = None

with st.sidebar:
    st.header("数据")
    source_options = ["Sample dataset", "Upload CSV", "Full local Criteo"]
    source_labels = {
        "Sample dataset": "示例数据集",
        "Upload CSV": "上传 CSV",
        "Full local Criteo": "本地完整 Criteo",
    }
    demo_preset_labels = {
        "Manual dataset": "手动选择数据",
        "Coupon ROI": "通用发券 ROI",
        "Baidu Waimai red packet": "百度外卖天降红包",
        "Didi passenger subsidy": "滴滴 C 端乘客补贴",
        "Didi driver supply": "滴滴 B 端司机补贴",
        "Didi budget allocation": "滴滴城市/时段预算分配",
        "Shopee ads iROAS": "Shopee 广告 iROAS",
        "ECUP full-funnel": "ECUP 全链路广告",
        "Delayed feedback": "延迟反馈增长",
        "Recommendation intervention": "推荐干预",
        "Marketplace subsidy": "Marketplace 补贴",
        "LLM routing": "LLM routing",
        "LLM multi-action routing": "LLM 多动作 routing",
        "Spotify message diet": "Spotify 站内消息",
        "DoorDash ghost ads": "DoorDash ghost ads",
        "Merchant subsidy": "商家补贴",
        "Game ops gift": "游戏运营礼包",
        "FinTech credit line": "金融授信/利率优惠",
        "Customer service escalation": "客服升级人工",
        "Healthcare follow-up": "医疗随访",
        "SaaS retention": "SaaS 续费挽留",
    }
    source = st.radio("数据来源", source_options, horizontal=False, format_func=lambda item: source_labels.get(item, item))
    if source == "Sample dataset" and sample_datasets:
        available_demo_options = [
            label
            for label, dataset_id in frontier_demo_dataset_ids.items()
            if dataset_id is None or dataset_id in dataset_id_to_name
        ]
        demo_preset = st.selectbox(
            "演示场景",
            available_demo_options,
            index=0,
            key="frontier_demo_preset",
            format_func=lambda item: demo_preset_labels.get(item, item),
            help="快速切换到面试友好的发券、百度外卖红包、滴滴补贴/预算、Shopee 广告、ECUP、延迟反馈、Marketplace 或 LLM routing 数据。",
        )
        preset_dataset_id = frontier_demo_dataset_ids.get(demo_preset)
        preset_name = dataset_id_to_name.get(preset_dataset_id)
        dataset_index = sample_names.index(preset_name) if preset_name in sample_names else 0
        dataset_name = st.selectbox("数据集", sample_names, index=dataset_index, key=f"sample_dataset_name_{demo_preset}")
        selected_dataset = sample_datasets[sample_names.index(dataset_name)]
        if preset_dataset_id and selected_dataset.get("id") == preset_dataset_id:
            st.success(f"演示数据已就绪：{demo_preset_labels.get(demo_preset, demo_preset)}")
        st.caption(selected_dataset.get("description", ""))
        if selected_dataset.get("source_url"):
            st.link_button("数据来源", selected_dataset["source_url"])
    elif source == "Sample dataset":
        st.warning("未找到示例数据清单。请运行 `scripts/prepare_sample_datasets.py`。")

    default_rows = 50000
    if selected_dataset:
        default_rows = int(selected_dataset.get("quick_train", {}).get("max_rows", selected_dataset.get("rows", 5000)))
    max_preview_rows = st.number_input("加载行数", min_value=100, max_value=1_000_000, value=default_rows, step=1_000)

uploaded_file = None
df = None

if source == "Sample dataset" and selected_dataset:
    df = load_csv_from_path(selected_dataset["path"], max_rows=int(max_preview_rows))
elif source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("CSV", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, nrows=int(max_preview_rows))
elif source == "Full local Criteo" and SAMPLE_PATH.exists():
    df = load_csv_from_path(str(SAMPLE_PATH), max_rows=int(max_preview_rows))
else:
    st.warning("Dataset was not found. Upload a CSV or prepare sample datasets to start.")

if df is None:
    st.stop()

columns = df.columns.tolist()
default_treatment = selected_dataset.get("treatment_col") if selected_dataset else None
if default_treatment not in columns:
    default_treatment = infer_default_column(columns, ["treatment", "T", "w", "is_treat"], 0)
default_outcome = selected_dataset.get("outcome_col") if selected_dataset else None
if default_outcome not in columns:
    default_outcome = infer_default_column(columns, ["visit", "outcome", "Y", "y", "label"], 1)
default_features = selected_dataset.get("feature_cols", []) if selected_dataset else []
default_features = [col for col in default_features if col in columns]
if not default_features:
    default_features = [col for col in columns if col not in {default_treatment, default_outcome}]
dataset_key = selected_dataset["id"] if selected_dataset else source.lower().replace(" ", "_")

with st.sidebar:
    st.header("字段")
    treatment_col = st.selectbox("Treatment 处理组字段", columns, index=columns.index(default_treatment), key=f"treatment_{dataset_key}")
    outcome_col = st.selectbox("Outcome 结果字段", columns, index=columns.index(default_outcome), key=f"outcome_{dataset_key}")
    feature_cols = st.multiselect("特征字段", columns, default=default_features, key=f"features_{dataset_key}")

    inferred_task = infer_task(df, outcome_col)
    task_default = selected_dataset.get("task", inferred_task) if selected_dataset else inferred_task
    task = st.selectbox("任务类型", ["auto", "classification", "regression"], index=["auto", "classification", "regression"].index(task_default), key=f"task_{dataset_key}")
    model_task = inferred_task if task == "auto" else task
    model_options = available_models(model_task)
    runnable_model_options = available_models(model_task, only_available=True)
    recommended = recommend_models(df, model_task, treatment_col, feature_cols, selected_dataset)
    default_model = next((model for model in recommended if model in runnable_model_options), runnable_model_options[0] if runnable_model_options else model_options[0])
    model_name = st.selectbox(
        "模型",
        model_options,
        index=model_options.index(default_model),
        format_func=format_model_option,
        key=f"model_{dataset_key}",
    )
    st.caption(MODEL_DESCRIPTIONS.get(model_name, "当前模型可用于所选任务。"))
    missing = missing_dependencies(model_name)
    if missing:
        st.warning(f"当前环境还不能训练 `{model_name}`，缺少：{', '.join(missing)}。")

    st.header("训练参数")
    quick_train = selected_dataset.get("quick_train", {}) if selected_dataset else {}
    epochs = st.number_input("训练轮数 Epochs", min_value=1, max_value=200, value=int(quick_train.get("epochs", 5)), step=1, key=f"epochs_{dataset_key}")
    batch_size = st.number_input("Batch size", min_value=8, max_value=4096, value=int(quick_train.get("batch_size", 128)), step=8, key=f"batch_{dataset_key}")
    learning_rate = st.number_input("学习率", min_value=1e-7, max_value=1e-1, value=float(quick_train.get("learning_rate", 1e-4)), format="%.7f", key=f"lr_{dataset_key}")
    test_size = st.slider("测试集比例", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
    valid_perc = st.slider("验证集比例", min_value=0.0, max_value=0.5, value=0.2, step=0.05)
    param_presets = model_parameter_presets(model_name)
    preset_names = list(param_presets)
    default_preset, _ = recommended_model_preset(model_name)
    preset_name = st.selectbox(
        "参数预设",
        preset_names,
        index=preset_names.index(default_preset) if default_preset in preset_names else 0,
        key=f"param_preset_{dataset_key}_{model_name}",
    )
    preset_params = param_presets[preset_name]
    model_params_text = st.text_area(
        "模型参数 JSON",
        value=json.dumps(preset_params, ensure_ascii=False, indent=2),
        height=150,
        key=f"model_params_{dataset_key}_{model_name}_{preset_name}",
    )

    st.header("决策参数")
    contact_cost = st.number_input("触达/干预成本", min_value=0.0, value=0.0, step=0.01, format="%.4f")
    conversion_value = st.number_input("单次转化价值", min_value=0.0, value=1.0, step=0.1, format="%.4f")
    bootstrap_samples = st.number_input("Bootstrap 样本数", min_value=0, max_value=500, value=0, step=10)
    sensitivity_samples = st.number_input("敏感性检验样本数", min_value=0, max_value=500, value=20, step=10)

workflow_tab, resume_scenario_tab, data_tab, diagnostics_tab, models_tab, deconstruction_tab, train_tab, compare_tab, result_tab, policy_tab, predict_tab, evidence_tab, knowledge_tab, agent_tab, multi_tab, dose_tab, history_tab, story_tab = st.tabs(
    [
        "流程总览",
        "场景工作台",
        "数据",
        "诊断",
        "模型",
        "模型拆解",
        "训练",
        "模型对比",
        "指标评估",
        "Policy/ROI",
        "打分导出",
        "证据",
        "知识库",
        "Agent",
        "多 Treatment",
        "剂量响应",
        "历史",
        "履历 Story",
    ]
)

with workflow_tab:
    render_workflow_page(
        selected_dataset,
        df,
        treatment_col,
        outcome_col,
        feature_cols,
        task,
        model_name,
        float(contact_cost),
        float(conversion_value),
    )

with resume_scenario_tab:
    render_resume_scenario_workbench()

with data_tab:
    if selected_dataset:
        st.subheader("Dataset")
        render_stat_grid(
            [
                ("Dataset", selected_dataset["name"], True),
                ("Rows", f"{selected_dataset.get('rows', len(df)):,}", False),
                ("Kind", selected_dataset.get("kind", ""), True),
                ("Task", selected_dataset.get("task", inferred_task), True),
            ]
        )
    render_data_diagnostics(df, treatment_col, outcome_col, feature_cols)
    st.subheader("Preview")
    st.dataframe(df.head(100), width="stretch")

with deconstruction_tab:
    render_model_deconstruction_workbench(model_name)

with train_tab:
    st.subheader("Training Contract")
    render_stat_grid(
        [
            ("Estimator", model_name, True),
            ("Task", model_task, True),
            ("Holdout", f"{float(test_size):.0%}", False),
            ("Decision Cost / Value", f"{float(contact_cost):.4f} / {float(conversion_value):.4f}", True),
        ],
        columns=4,
    )
    training_contract_rows = [
        {
            "gate": "pre-train data gate",
            "check": "treatment/outcome/features are fixed before fit",
            "why": "prevents target leakage and accidental post-treatment features",
        },
        {
            "gate": "baseline gate",
            "check": "run one fast ready model before expensive compare",
            "why": "establishes artifact contract and catches schema or dependency issues early",
        },
        {
            "gate": "evaluation gate",
            "check": "write predictions, metrics, readiness and run note",
            "why": "every model can be inspected on Evaluation, Policy, Predict and Evidence pages",
        },
    ]
    st.dataframe(pd.DataFrame(training_contract_rows), width="stretch", hide_index=True)
    st.subheader("Run")
    try:
        model_params = json.loads(model_params_text or "{}")
    except json.JSONDecodeError as exc:
        st.error(f"Invalid model params JSON: {exc}")
        st.stop()

    can_train = bool(feature_cols) and treatment_col != outcome_col and not missing
    if not can_train:
        if missing:
            st.error("当前模型依赖未就绪，请切换 ready 模型或安装对应后端。")
        else:
            st.error("Select at least one feature and keep treatment/outcome different.")

    if st.button("Start training", type="primary", disabled=not can_train):
        progress = st.progress(0)
        history_box = st.empty()
        status = st.empty()
        local_history = []

        def on_epoch(metrics):
            local_history.append(metrics)
            progress.progress(min(1.0, (metrics["epoch"] + 1) / max(int(epochs), 1)))
            history_box.dataframe(pd.DataFrame(local_history), width="stretch")
            status.write(f"Epoch {metrics['epoch'] + 1}/{int(epochs)}")

        cfg = UpliftConfig(
            treatment_col=treatment_col,
            outcome_col=outcome_col,
            feature_cols=feature_cols,
            model_name=model_name,
            task=task,
            test_size=float(test_size),
            valid_perc=float(valid_perc) if valid_perc > 0 else None,
            epochs=int(epochs),
            batch_size=int(batch_size),
            learning_rate=float(learning_rate),
            model_params=model_params,
            max_rows=int(max_preview_rows),
            bootstrap_samples=int(bootstrap_samples),
            sensitivity_samples=int(sensitivity_samples),
            policy_contact_cost=float(contact_cost),
            policy_conversion_value=float(conversion_value),
        )

        try:
            with st.spinner("Training uplift model"):
                result = train_uplift_model(config=cfg, data_frame=df, callback=on_epoch)
        except Exception as exc:
            st.error(str(exc))
        else:
            st.session_state["uplift_result"] = result
            st.success(f"Run saved: {result.run_id}")

with diagnostics_tab:
    st.subheader("Data Design")
    diagnostics = diagnose_uplift_data(df, treatment_col, outcome_col, feature_cols)
    design = diagnostics["design"]
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Treatment", design["treatment_type"])
    d2.metric("Outcome", design["outcome_type"])
    d3.metric("Treatment Values", design["treatment_unique"])
    d4.metric("Outcome Values", design["outcome_unique"])

    for warning in diagnostics.get("warnings", []):
        st.warning(warning)

    left, right = st.columns(2)
    with left:
        st.subheader("Treatment Distribution")
        st.dataframe(pd.DataFrame(diagnostics["treatment_distribution"]), width="stretch")
        st.subheader("Outcome By Treatment")
        st.dataframe(pd.DataFrame(diagnostics["outcome_by_treatment"]), width="stretch")
    with right:
        st.subheader("Propensity Overlap")
        overlap = diagnostics.get("overlap") or {}
        if overlap:
            st.dataframe(pd.DataFrame([overlap]), width="stretch")
        else:
            st.info("Overlap diagnostics are available for binary treatment.")
        st.subheader("Missingness")
        st.dataframe(pd.DataFrame(diagnostics["missingness"]).head(20), width="stretch")

    st.subheader("Selection Bias / SSB Proxy")
    selection_bias = diagnostics.get("selection_bias") or {}
    if selection_bias:
        render_stat_grid(
            [
                ("Risk", selection_bias.get("risk_level", "unknown"), True),
                ("Signals", len(selection_bias.get("signals") or []), False),
                ("Recommended", ", ".join((selection_bias.get("recommended_models") or [])[:3]), True),
                ("Use", "diagnostic gate", True),
            ]
        )
        signals = pd.DataFrame(selection_bias.get("signals") or [])
        if not signals.empty:
            st.dataframe(signals, width="stretch")
        st.caption(selection_bias.get("interview_note", ""))

    st.subheader("Feature Balance")
    balance = pd.DataFrame(diagnostics["feature_balance"])
    if not balance.empty:
        st.dataframe(balance.head(30), width="stretch")
    else:
        st.info("No feature balance diagnostics available.")

with compare_tab:
    st.subheader("Model Compare")
    render_stat_grid(
        [
            ("Compare Role", "stress-test estimator choice", True),
            ("Primary Sort", "QINI CI low / readiness", True),
            ("Business Check", "Top-K + policy value", True),
            ("Guardrail", "overlap + calibration", True),
        ],
        columns=4,
    )
    compare_decision_rows = [
        {
            "step": "candidate design",
            "what_to_do": "compare a fast baseline, a bias-aware learner, and a calibrated/ranking learner",
            "risk_if_skipped": "a single model win may only reflect estimator variance or dependency quirks",
        },
        {
            "step": "metric reading",
            "what_to_do": "sort by QINI lower bound when bootstrap exists, otherwise readiness and QINI",
            "risk_if_skipped": "mean QINI can overstate a noisy or unsupported model",
        },
        {
            "step": "business sanity",
            "what_to_do": "confirm Top-K observed uplift and policy value agree with ranking metrics",
            "risk_if_skipped": "a good curve can still produce a bad launch bucket or negative ROI",
        },
    ]
    st.dataframe(pd.DataFrame(compare_decision_rows), width="stretch", hide_index=True)
    with st.expander("Frontier Metric Reading Guide", expanded=False):
        st.caption("When ECUP or delayed-feedback runs appear in Compare, use these fields as industrial decision signals, not just extra columns.")
        st.dataframe(pd.DataFrame(frontier_metric_reading_rows()), width="stretch", hide_index=True)
    presets = compare_model_presets(df, model_task, treatment_col, feature_cols, selected_dataset)
    preset_name = st.selectbox("Candidate set", list(presets.keys()), key=f"compare_preset_{dataset_key}")
    if "observational" in preset_name:
        st.caption("This preset is selected first because treatment allocation looks imbalanced; it emphasizes DR/R/orthogonal/forest-style learners.")
    elif "randomized" in preset_name:
        st.caption("This preset is selected first for randomized-like binary uplift; it emphasizes fast industrial baselines and Top-K ranking checks.")
    compare_defaults = presets[preset_name] or recommend_models(df, model_task, treatment_col, feature_cols, selected_dataset)[:4]
    compare_models = st.multiselect("Models", model_options, default=compare_defaults, format_func=format_model_option)
    use_compare_presets = st.checkbox(
        "Use model-specific recommended presets",
        value=True,
        key=f"compare_use_model_presets_{dataset_key}",
        help="Each compare candidate uses the recommended preset from its model card.",
    )
    apply_compare_manual_overrides = st.checkbox(
        "Apply manual JSON overrides to every candidate",
        value=False,
        key=f"compare_apply_manual_overrides_{dataset_key}",
        help="When enabled, the sidebar Model params JSON overrides each candidate preset.",
    )
    missing_compare = {model: missing_dependencies(model) for model in compare_models if missing_dependencies(model)}
    if missing_compare:
        st.warning("有些候选模型依赖未就绪：" + "; ".join(f"{model}: {', '.join(deps)}" for model, deps in missing_compare.items()))
    runnable_compare_models = [model for model in compare_models if not missing_dependencies(model)]
    if compare_models and not runnable_compare_models:
        st.error("所选候选模型都不可运行，请选择 ready 模型。")
    if runnable_compare_models:
        st.caption("Compare Candidate Presets")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "model": compare_model,
                        "preset": recommended_model_preset(compare_model)[0] if use_compare_presets else "Manual JSON",
                        "params": json.dumps(recommended_model_preset(compare_model)[1] if use_compare_presets else {}, ensure_ascii=False),
                    }
                    for compare_model in runnable_compare_models
                ]
            ),
            width="stretch",
            hide_index=True,
        )
    can_compare = bool(runnable_compare_models) and bool(feature_cols) and treatment_col != outcome_col
    if st.button("Train comparison", type="primary", disabled=not can_compare):
        compare_model_params = {}
        if apply_compare_manual_overrides or not use_compare_presets:
            try:
                compare_model_params = json.loads(model_params_text or "{}")
            except json.JSONDecodeError as exc:
                st.error(f"Invalid model params JSON: {exc}")
                st.stop()
        compare_rows = []
        compare_results = []
        progress = st.progress(0)
        status_panel = st.empty()
        live = st.empty()
        succeeded = 0
        failed = 0
        for idx, compare_model in enumerate(runnable_compare_models, start=1):
            progress.progress((idx - 1) / len(runnable_compare_models))
            status_panel.markdown(
                compare_status_html(idx, len(runnable_compare_models), compare_model, succeeded, failed),
                unsafe_allow_html=True,
            )
            compare_preset = "Manual JSON"
            candidate_params = dict(compare_model_params)
            if use_compare_presets:
                compare_preset, candidate_params = recommended_model_preset(compare_model)
                if apply_compare_manual_overrides:
                    candidate_params.update(compare_model_params)
            cfg = UpliftConfig(
                treatment_col=treatment_col,
                outcome_col=outcome_col,
                feature_cols=feature_cols,
                model_name=compare_model,
                task=task,
                test_size=float(test_size),
                valid_perc=float(valid_perc) if valid_perc > 0 else None,
                epochs=int(epochs),
                batch_size=int(batch_size),
                learning_rate=float(learning_rate),
                model_params=candidate_params,
                max_rows=int(max_preview_rows),
                bootstrap_samples=int(bootstrap_samples),
                sensitivity_samples=int(sensitivity_samples),
                policy_contact_cost=float(contact_cost),
                policy_conversion_value=float(conversion_value),
                run_name=f"compare-{compare_model.lower()}",
            )
            try:
                compare_result = train_uplift_model(config=cfg, data_frame=df)
            except Exception as exc:
                compare_rows.append({"model": compare_model, "preset": compare_preset, "error": str(exc)})
                failed += 1
                status_panel.markdown(
                    compare_status_html(idx, len(runnable_compare_models), compare_model, succeeded, failed),
                    unsafe_allow_html=True,
                )
                continue
            succeeded += 1
            metrics = compare_result.eval_metrics
            top_k = metrics.get("top_k", [])
            top10 = next((row for row in top_k if abs(row.get("top_fraction", 0) - 0.1) < 1e-6), {})
            oracle_rows = (metrics.get("oracle_top_k") or {}).get("rows") or []
            oracle_top10 = next((row for row in oracle_rows if abs(row.get("top_fraction", 0) - 0.1) < 1e-6), {})
            business_rows = (metrics.get("business_score_alignment") or {}).get("rows") or []
            business_top10 = next((row for row in business_rows if abs(row.get("top_fraction", 0) - 0.1) < 1e-6), {})
            bootstrap_metric = ((metrics.get("bootstrap") or {}).get("metrics") or {}).get("qini_score") or {}
            policy_best = metrics.get("policy_best") or {}
            sensitivity = metrics.get("sensitivity") or {}
            overlap_trim = metrics.get("overlap_trim") or {}
            overlap_propensity = overlap_trim.get("propensity") or {}
            readiness = decision_readiness(metrics)
            compare_rows.append(
                {
                    "model": compare_model,
                    "preset": compare_preset,
                    "run_id": compare_result.run_id,
                    "readiness_score": readiness.get("score"),
                    "readiness_level": readiness.get("level"),
                    "qini": metrics.get("qini_score"),
                    "qini_ci_low": bootstrap_metric.get("low"),
                    "qini_ci_high": bootstrap_metric.get("high"),
                    "auuc": metrics.get("auuc_score"),
                    "calibration_mae": metrics.get("calibration_mae"),
                    "top10_observed_uplift": top10.get("observed_uplift"),
                    "oracle_top10_recall": oracle_top10.get("topk_recall"),
                    "business_top10_overlap": business_top10.get("topk_overlap_recall"),
                    "business_top10_capture": business_top10.get("business_score_capture"),
                    **frontier_metric_summary(metrics),
                    "policy_best_fraction": policy_best.get("top_fraction"),
                    "policy_best_net": policy_best.get("observed_net_value", policy_best.get("predicted_net_value")),
                    "sensitivity_verdict": sensitivity.get("verdict"),
                    "weak_overlap_rate": overlap_propensity.get("weak_overlap_rate"),
                    "predictions": compare_result.artifacts["predictions"],
                }
            )
            compare_results.append(compare_result)
            live.dataframe(pd.DataFrame(compare_rows), width="stretch")
            progress.progress(idx / len(runnable_compare_models))
        status_panel.markdown(
            compare_status_html(len(runnable_compare_models), len(runnable_compare_models), "finished", succeeded, failed),
            unsafe_allow_html=True,
        )
        ranking = pd.DataFrame(compare_rows)
        if "qini_ci_low" in ranking.columns and ranking["qini_ci_low"].notna().any():
            ranking = ranking.sort_values("qini_ci_low", ascending=False, na_position="last")
        elif "readiness_score" in ranking.columns and ranking["readiness_score"].notna().any():
            ranking = ranking.sort_values(["readiness_score", "qini"], ascending=[False, False], na_position="last")
        elif "qini" in ranking.columns:
            ranking = ranking.sort_values("qini", ascending=False, na_position="last")
        diagnostics_for_report = diagnose_uplift_data(df, treatment_col, outcome_col, feature_cols)
        dataset_label = selected_dataset["name"] if selected_dataset else source
        report_path = save_compare_report(ranking, dataset_label, diagnostics_for_report, runnable_compare_models)
        readiness_path = save_compare_readiness_json(ranking, dataset_label, diagnostics_for_report, runnable_compare_models)
        bundle_path = save_compare_bundle_manifest(ranking, dataset_label, report_path, readiness_path, compare_results)
        st.session_state["compare_ranking"] = ranking
        st.session_state["compare_results"] = compare_results
        st.session_state["compare_report_path"] = report_path
        st.session_state["compare_readiness_path"] = readiness_path
        st.session_state["compare_bundle_path"] = bundle_path
        st.success("Comparison finished.")

    ranking = st.session_state.get("compare_ranking")
    if ranking is not None and not ranking.empty:
        st.subheader("Ranking")
        render_compare_decision_board(
            ranking,
            caption="Readiness is absolute. QINI and policy net value are scaled to 0-100 within this compare run for quick visual triage.",
        )
        best_row = ranking.iloc[0].to_dict()
        st.subheader("Best Model Rationale")
        render_model_rationale(str(best_row.get("model", "")), ranking_row=best_row)
        st.dataframe(ranking, width="stretch")
        if "frontier_metric_schema" in ranking.columns and ranking["frontier_metric_schema"].fillna("").astype(str).str.len().gt(0).any():
            st.subheader("Frontier Metric Summary")
            frontier_cols = [
                col
                for col in [
                    "model",
                    "frontier_metric_schema",
                    "full_funnel_top10_impression_uplift",
                    "full_funnel_top10_click_uplift",
                    "full_funnel_top10_conversion_uplift",
                    "delayed_d7_top10_uplift",
                    "delayed_d14_top10_uplift",
                    "delayed_d30_top10_uplift",
                    "delayed_censored_rate",
                    "delayed_observed_delay_p90",
                ]
                if col in ranking.columns
            ]
            st.caption("These columns make ECUP and delayed-feedback metrics visible at model-selection time, not only in the Evaluation detail page.")
            st.dataframe(ranking[frontier_cols], width="stretch", hide_index=True)
        compare_report_path = st.session_state.get("compare_report_path")
        if compare_report_path and Path(compare_report_path).exists():
            st.download_button(
                "Download compare report",
                data=Path(compare_report_path).read_bytes(),
                file_name=Path(compare_report_path).name,
                mime="text/markdown",
            )
        compare_readiness_path = st.session_state.get("compare_readiness_path")
        if compare_readiness_path and Path(compare_readiness_path).exists():
            st.download_button(
                "Download compare readiness JSON",
                data=Path(compare_readiness_path).read_bytes(),
                file_name=Path(compare_readiness_path).name,
                mime="application/json",
            )
        compare_bundle_path = st.session_state.get("compare_bundle_path")
        if compare_bundle_path and Path(compare_bundle_path).exists():
            st.download_button(
                "Download compare bundle manifest",
                data=Path(compare_bundle_path).read_bytes(),
                file_name=Path(compare_bundle_path).name,
                mime="application/json",
            )

with multi_tab:
    st.subheader("Multi-Treatment Policy")
    treatment_values = sorted(df[treatment_col].dropna().unique().tolist(), key=lambda item: str(item))
    if len(treatment_values) <= 2:
        st.info("当前 treatment 是二值。多 treatment 工作流会在 treatment 取值超过 2 个时启用。")
    else:
        st.caption(f"Detected treatment values: {treatment_values}")
        control_value = st.selectbox("Control treatment", treatment_values, index=0, key=f"multi_control_{dataset_key}")
        multi_model = st.selectbox(
            "Multi-treatment model",
            ["MultiTLearnerGBM", "MultiTLearnerRF", "MultiDRLearnerGBM", "MultiDRLearnerRF"],
            format_func=lambda name: MODEL_DESCRIPTIONS.get(name, name),
            key=f"multi_model_{dataset_key}",
        )
        try:
            model_params = json.loads(model_params_text or "{}")
        except json.JSONDecodeError as exc:
            st.error(f"Invalid model params JSON: {exc}")
            st.stop()
        can_train_multi = bool(feature_cols) and treatment_col != outcome_col
        if st.button("Train multi-treatment policy", type="primary", disabled=not can_train_multi):
            cfg = UpliftConfig(
                treatment_col=treatment_col,
                outcome_col=outcome_col,
                feature_cols=feature_cols,
                model_name=multi_model,
                task=task,
                test_size=float(test_size),
                valid_perc=0.0,
                epochs=1,
                batch_size=int(batch_size),
                learning_rate=float(learning_rate),
                model_params=model_params,
                max_rows=int(max_preview_rows),
                bootstrap_samples=int(bootstrap_samples),
                control_value=control_value,
                policy_contact_cost=float(contact_cost),
                policy_conversion_value=float(conversion_value),
                run_name=f"multi-{multi_model.lower()}",
            )
            try:
                with st.spinner("Training multi-treatment learner"):
                    multi_result = train_multi_treatment_model(
                        data_frame=df,
                        config=cfg,
                        model_family=MULTI_MODEL_FAMILIES[multi_model],
                    )
            except Exception as exc:
                st.error(str(exc))
            else:
                st.session_state["multi_treatment_result"] = multi_result
                st.success(f"Multi-treatment run saved: {multi_result['run_id']}")

        multi_result = st.session_state.get("multi_treatment_result")
        if multi_result is not None:
            metrics = multi_result["metrics"]
            st.caption(f"Strategy: `{metrics.get('strategy', 't_learner')}`")
            m1, m2, m3 = st.columns(3)
            m1.metric("Recommended IPW Value", f"{metrics.get('recommended_policy_value_ipw') or 0:.4f}")
            m2.metric("Control IPW Value", f"{metrics.get('control_policy_value_ipw') or 0:.4f}")
            m3.metric("Incremental IPW Value", f"{metrics.get('incremental_policy_value_ipw') or 0:.4f}")
            bootstrap = metrics.get("policy_value_bootstrap") or {}
            bootstrap_metrics = bootstrap.get("metrics") or {}
            incremental_ci = bootstrap_metrics.get("incremental_policy_value_ipw") or {}
            if incremental_ci:
                ci_low = incremental_ci.get("low")
                ci_high = incremental_ci.get("high")
                st.caption(
                    f"Bootstrap samples: {bootstrap.get('n_success', 0)}/{bootstrap.get('n_bootstrap', 0)} | "
                    f"Incremental IPW 95% CI [{ci_low or 0:.4f}, {ci_high or 0:.4f}]"
                )
            st.subheader("Recommended Action Mix")
            st.dataframe(pd.DataFrame(metrics.get("recommended_action_counts", [])), width="stretch")
            st.subheader("Action Propensity Overlap")
            propensity_by_action = pd.DataFrame(metrics.get("propensity_by_action", []))
            if not propensity_by_action.empty:
                st.dataframe(propensity_by_action, width="stretch")
            else:
                st.info("No action propensity diagnostics available for this run.")
            st.subheader("Observed Outcome By Actual Treatment")
            st.dataframe(pd.DataFrame(metrics.get("observed_by_treatment", [])), width="stretch")
            st.subheader("Top Action Uplift")
            st.dataframe(pd.DataFrame(metrics.get("top_actions", [])), width="stretch")
            st.subheader("Prediction Preview")
            st.dataframe(pd.DataFrame(multi_result.get("preview", [])), width="stretch")
            predictions_path = Path(multi_result["artifacts"]["predictions"])
            note_path = Path(multi_result["artifacts"]["run_note"])
            st.download_button(
                "Download multi-treatment predictions",
                data=predictions_path.read_bytes(),
                file_name=f"{multi_result['run_id']}_predictions.csv",
                mime="text/csv",
            )
            st.download_button(
                "Download multi-treatment run note",
                data=note_path.read_bytes(),
                file_name=f"{multi_result['run_id']}_run_note.md",
                mime="text/markdown",
            )

        st.divider()
        st.subheader("Compare Multi-Treatment Models")
        multi_compare_defaults = [model for model in ["MultiTLearnerGBM", "MultiDRLearnerGBM"] if model in MULTI_MODEL_FAMILIES]
        multi_compare_models = st.multiselect(
            "Multi-treatment compare models",
            list(MULTI_MODEL_FAMILIES.keys()),
            default=multi_compare_defaults,
            format_func=lambda name: MODEL_DESCRIPTIONS.get(name, name),
            key=f"multi_compare_models_{dataset_key}",
        )
        can_compare_multi = bool(multi_compare_models) and bool(feature_cols) and treatment_col != outcome_col
        if st.button("Train multi-treatment comparison", type="primary", disabled=not can_compare_multi):
            compare_rows = []
            progress = st.progress(0)
            live = st.empty()
            for idx, compare_model in enumerate(multi_compare_models, start=1):
                live.write(f"Training {compare_model} ({idx}/{len(multi_compare_models)})")
                cfg = UpliftConfig(
                    treatment_col=treatment_col,
                    outcome_col=outcome_col,
                    feature_cols=feature_cols,
                    model_name=compare_model,
                    task=task,
                    test_size=float(test_size),
                    valid_perc=0.0,
                    epochs=1,
                    batch_size=int(batch_size),
                    learning_rate=float(learning_rate),
                    model_params=model_params,
                    max_rows=int(max_preview_rows),
                    bootstrap_samples=int(bootstrap_samples),
                    control_value=control_value,
                    policy_contact_cost=float(contact_cost),
                    policy_conversion_value=float(conversion_value),
                    run_name=f"multi-compare-{compare_model.lower()}",
                )
                try:
                    compare_result = train_multi_treatment_model(
                        data_frame=df,
                        config=cfg,
                        model_family=MULTI_MODEL_FAMILIES[compare_model],
                    )
                except Exception as exc:
                    compare_rows.append({"model": compare_model, "error": str(exc)})
                    progress.progress(idx / len(multi_compare_models))
                    live.dataframe(pd.DataFrame(compare_rows), width="stretch")
                    continue

                metrics = compare_result["metrics"]
                incremental_ci = (((metrics.get("policy_value_bootstrap") or {}).get("metrics") or {}).get("incremental_policy_value_ipw") or {})
                propensity_rows = metrics.get("propensity_by_action", [])
                action_counts = metrics.get("recommended_action_counts", [])
                top_action = action_counts[0].get("treatment") if action_counts else None
                compare_rows.append(
                    {
                        "model": compare_model,
                        "strategy": metrics.get("strategy"),
                        "run_id": compare_result["run_id"],
                        "incremental_policy_value_ipw": metrics.get("incremental_policy_value_ipw"),
                        "incremental_ci_low": incremental_ci.get("low"),
                        "incremental_ci_high": incremental_ci.get("high"),
                        "recommended_policy_value_ipw": metrics.get("recommended_policy_value_ipw"),
                        "control_policy_value_ipw": metrics.get("control_policy_value_ipw"),
                        "max_low_propensity_rate": max((row.get("low_propensity_rate", 0) for row in propensity_rows), default=None),
                        "top_recommended_action": top_action,
                        "predictions": compare_result["artifacts"]["predictions"],
                    }
                )
                progress.progress(idx / len(multi_compare_models))
                live.dataframe(pd.DataFrame(compare_rows), width="stretch")

            ranking = pd.DataFrame(compare_rows)
            if not ranking.empty:
                if "incremental_ci_low" in ranking.columns and ranking["incremental_ci_low"].notna().any():
                    ranking = ranking.sort_values("incremental_ci_low", ascending=False, na_position="last")
                elif "incremental_policy_value_ipw" in ranking.columns:
                    ranking = ranking.sort_values("incremental_policy_value_ipw", ascending=False, na_position="last")
            dataset_label = selected_dataset["name"] if selected_dataset else source
            report_path = save_multi_compare_report(ranking, dataset_label, multi_compare_models)
            st.session_state["multi_compare_ranking"] = ranking
            st.session_state["multi_compare_report_path"] = report_path
            st.success("Multi-treatment comparison finished.")

        multi_compare_ranking = st.session_state.get("multi_compare_ranking")
        if multi_compare_ranking is not None and not multi_compare_ranking.empty:
            st.subheader("Multi-Treatment Ranking")
            st.dataframe(multi_compare_ranking, width="stretch")
            multi_compare_report_path = st.session_state.get("multi_compare_report_path")
            if multi_compare_report_path and Path(multi_compare_report_path).exists():
                st.download_button(
                    "Download multi-treatment compare report",
                    data=Path(multi_compare_report_path).read_bytes(),
                    file_name=Path(multi_compare_report_path).name,
                    mime="text/markdown",
                )

    st.divider()
    st.subheader("Score With Saved Multi-Treatment Run")
    multi_history = load_multi_run_history()
    if multi_history.empty:
        st.info("还没有可用于多 treatment 打分的历史 run。先在本页训练一个 MultiTLearner。")
    else:
        multi_display_cols = [
            col
            for col in [
                "run_id",
                "model",
                "strategy",
                "control_treatment",
                "treatment_values",
                "incremental_policy_value_ipw",
                "incremental_ci_low",
                "incremental_ci_high",
            ]
            if col in multi_history.columns
        ]
        st.dataframe(
            multi_history[multi_display_cols],
            width="stretch",
        )
        multi_score_run_id = st.selectbox("Multi-treatment run", multi_history["run_id"].tolist(), key="multi_score_run")
        multi_run_row = multi_history[multi_history["run_id"] == multi_score_run_id].iloc[0]
        multi_scoring_source = st.radio("Multi scoring data", ["Current loaded data", "Upload CSV"], horizontal=True, key="multi_scoring_source")
        multi_scoring_df = None
        if multi_scoring_source == "Current loaded data":
            multi_scoring_df = df.copy()
            st.caption("Uses the dataset currently loaded in the sidebar.")
        else:
            multi_scoring_file = st.file_uploader("Multi scoring CSV", type=["csv"], key="multi_scoring_csv")
            if multi_scoring_file is not None:
                multi_scoring_df = pd.read_csv(multi_scoring_file)

        if multi_scoring_df is not None:
            st.dataframe(multi_scoring_df.head(50), width="stretch")
        if st.button("Score multi-treatment data", type="primary", disabled=multi_scoring_df is None):
            try:
                with st.spinner("Scoring multi-treatment policy"):
                    multi_scoring_result = score_multi_treatment_run(multi_run_row["run_dir"], multi_scoring_df)
            except Exception as exc:
                st.error(str(exc))
            else:
                st.session_state["multi_scoring_result"] = multi_scoring_result
                st.success(f"Scored {len(multi_scoring_result['predictions']):,} rows with {multi_scoring_result['model_name']}.")

        multi_scoring_result = st.session_state.get("multi_scoring_result")
        if multi_scoring_result is not None:
            st.subheader("Multi-Treatment Scoring Result")
            st.dataframe(pd.DataFrame(multi_scoring_result.get("recommended_action_counts", [])), width="stretch")
            top_summary = pd.DataFrame(
                [
                    {
                        key: value
                        for key, value in row.items()
                        if key != "recommended_action_counts"
                    }
                    for row in multi_scoring_result.get("top_counts", [])
                ]
            )
            if not top_summary.empty:
                st.dataframe(top_summary, width="stretch")
            scored_predictions = multi_scoring_result["predictions"]
            st.dataframe(scored_predictions.head(100), width="stretch")
            st.download_button(
                "Download all multi-treatment scored rows",
                data=dataframe_csv_bytes(scored_predictions),
                file_name=f"{multi_scoring_result['run_id']}_multi_scored.csv",
                mime="text/csv",
            )
            top_cols = st.columns(3)
            for col, fraction in zip(top_cols, [0.05, 0.10, 0.20]):
                top_df = score_top_fraction(scored_predictions, fraction)
                col.download_button(
                    f"Download Multi Top {fraction:.0%}",
                    data=dataframe_csv_bytes(top_df),
                    file_name=f"{multi_scoring_result['run_id']}_multi_top_{int(fraction * 100)}.csv",
                    mime="text/csv",
                )

with dose_tab:
    st.subheader("Continuous Treatment Dose-Response")
    is_numeric_treatment = pd.api.types.is_numeric_dtype(df[treatment_col])
    unique_treatment = df[treatment_col].dropna().nunique()
    if not is_numeric_treatment or unique_treatment <= 5:
        st.info("当前 treatment 不是连续数值列。Dose-Response 工作流适合折扣率、触达强度、频次、补贴比例这类连续 treatment。")
    else:
        dose_model = st.selectbox(
            "Dose-response model",
            ["DoseResponseGBM", "DoseResponseRF"],
            format_func=lambda name: MODEL_DESCRIPTIONS.get(name, name),
            key=f"dose_model_{dataset_key}",
        )
        dose_grid_size = st.slider("Dose grid size", min_value=5, max_value=51, value=21, step=2, key=f"dose_grid_{dataset_key}")
        baseline_dose = st.number_input(
            "Baseline dose",
            min_value=float(pd.to_numeric(df[treatment_col], errors="coerce").min()),
            max_value=float(pd.to_numeric(df[treatment_col], errors="coerce").max()),
            value=float(pd.to_numeric(df[treatment_col], errors="coerce").quantile(0.01)),
            step=0.01,
            format="%.4f",
            key=f"dose_baseline_{dataset_key}",
        )
        try:
            model_params = json.loads(model_params_text or "{}")
        except json.JSONDecodeError as exc:
            st.error(f"Invalid model params JSON: {exc}")
            st.stop()

        if st.button("Train dose-response model", type="primary", disabled=not feature_cols):
            cfg = UpliftConfig(
                treatment_col=treatment_col,
                outcome_col=outcome_col,
                feature_cols=feature_cols,
                model_name=dose_model,
                task=task,
                test_size=float(test_size),
                valid_perc=0.0,
                epochs=1,
                batch_size=int(batch_size),
                learning_rate=float(learning_rate),
                model_params=model_params,
                max_rows=int(max_preview_rows),
                control_value=float(baseline_dose),
                run_name=f"dose-{dose_model.lower()}",
            )
            try:
                with st.spinner("Training dose-response model"):
                    dose_result = train_dose_response_model(
                        data_frame=df,
                        config=cfg,
                        model_family="forest" if dose_model == "DoseResponseRF" else "gbm",
                        grid_size=int(dose_grid_size),
                    )
            except Exception as exc:
                st.error(str(exc))
            else:
                st.session_state["dose_response_result"] = dose_result
                st.success(f"Dose-response run saved: {dose_result['run_id']}")

        dose_result = st.session_state.get("dose_response_result")
        if dose_result is not None:
            metrics = dose_result["metrics"]
            best = metrics.get("average_best_dose") or {}
            d1, d2, d3 = st.columns(3)
            d1.metric("Baseline Dose", f"{metrics.get('baseline_dose') or 0:.4f}")
            d2.metric("Mean Recommended Dose", f"{best.get('mean_recommended_dose') or 0:.4f}")
            d3.metric("Mean Gain", f"{best.get('mean_predicted_gain_vs_baseline') or 0:.4f}")
            curve = pd.DataFrame(metrics.get("dose_response_curve", []))
            if not curve.empty:
                st.subheader("Average Dose-Response Curve")
                st.line_chart(curve.set_index("dose")[["mean_predicted_outcome"]])
                st.dataframe(curve, width="stretch")
            optimizer = metrics.get("policy_optimizer") or {}
            if optimizer.get("rows"):
                st.subheader("Cost-Aware Dose Policy Optimizer")
                opt_best = optimizer.get("best") or {}
                c1, c2, c3 = st.columns(3)
                c1.metric("Best Top-K", f"{float(opt_best.get('top_fraction') or 0):.0%}")
                c2.metric("Net Value", f"{float(opt_best.get('net_value') or 0):.2f}")
                c3.metric("Unsafe Extrapolation", f"{float(opt_best.get('unsafe_extrapolation_rate') or 0):.2%}")
                st.caption("连续 treatment 不应该只选最大 dose；这里按 predicted gain * value - incremental dose cost 选择预算内最优强度，并提示是否外推到 observed support 之外。")
                st.dataframe(pd.DataFrame(optimizer.get("rows") or []), width="stretch", hide_index=True)
            st.subheader("Observed Outcome By Dose Bin")
            st.dataframe(pd.DataFrame(metrics.get("observed_by_dose_bin", [])), width="stretch")
            st.subheader("Dose Policy Preview")
            st.dataframe(pd.DataFrame(dose_result.get("preview", [])), width="stretch")
            predictions_path = Path(dose_result["artifacts"]["predictions"])
            note_path = Path(dose_result["artifacts"]["run_note"])
            st.download_button(
                "Download dose-response predictions",
                data=predictions_path.read_bytes(),
                file_name=f"{dose_result['run_id']}_predictions.csv",
                mime="text/csv",
            )
            st.download_button(
                "Download dose-response run note",
                data=note_path.read_bytes(),
                file_name=f"{dose_result['run_id']}_run_note.md",
                mime="text/markdown",
            )

    st.divider()
    st.subheader("Score With Saved Dose-Response Run")
    dose_history = load_dose_run_history()
    if dose_history.empty:
        st.info("还没有可用于连续 treatment 打分的历史 run。")
    else:
        st.dataframe(dose_history, width="stretch")
        dose_score_run_id = st.selectbox("Dose-response run", dose_history["run_id"].tolist(), key="dose_score_run")
        dose_run_row = dose_history[dose_history["run_id"] == dose_score_run_id].iloc[0]
        dose_scoring_source = st.radio("Dose scoring data", ["Current loaded data", "Upload CSV"], horizontal=True, key="dose_scoring_source")
        dose_scoring_df = None
        if dose_scoring_source == "Current loaded data":
            dose_scoring_df = df.copy()
        else:
            dose_scoring_file = st.file_uploader("Dose scoring CSV", type=["csv"], key="dose_scoring_csv")
            if dose_scoring_file is not None:
                dose_scoring_df = pd.read_csv(dose_scoring_file)
        if dose_scoring_df is not None:
            st.dataframe(dose_scoring_df.head(50), width="stretch")
        if st.button("Score dose-response data", type="primary", disabled=dose_scoring_df is None):
            try:
                with st.spinner("Scoring dose-response policy"):
                    dose_scoring_result = score_dose_response_run(dose_run_row["run_dir"], dose_scoring_df)
            except Exception as exc:
                st.error(str(exc))
            else:
                st.session_state["dose_scoring_result"] = dose_scoring_result
                st.success(f"Scored {len(dose_scoring_result['predictions']):,} rows with {dose_scoring_result['model_name']}.")

        dose_scoring_result = st.session_state.get("dose_scoring_result")
        if dose_scoring_result is not None:
            scored_predictions = dose_scoring_result["predictions"]
            st.subheader("Dose-Response Scoring Result")
            st.dataframe(scored_predictions.head(100), width="stretch")
            st.download_button(
                "Download all dose-response scored rows",
                data=dataframe_csv_bytes(scored_predictions),
                file_name=f"{dose_scoring_result['run_id']}_dose_scored.csv",
                mime="text/csv",
            )

with result_tab:
    result = st.session_state.get("uplift_result")
    render_evaluation_metric_guide(result)
    if result is None:
        st.info("No run yet.")
    else:
        render_result(result)

with predict_tab:
    st.subheader("Score New Data")
    history = load_run_history()
    if history.empty:
        st.info("Train at least one model before scoring new data.")
    else:
        run_id = st.selectbox("Run", history["run_id"].tolist(), key="predict_run")
        run_row = history[history["run_id"] == run_id].iloc[0]
        scoring_source = st.radio("Scoring data", ["Current loaded data", "Upload CSV"], horizontal=True)
        scoring_df = None
        if scoring_source == "Current loaded data":
            scoring_df = df.copy()
            st.caption("Uses the dataset currently loaded in the sidebar.")
        else:
            scoring_file = st.file_uploader("Scoring CSV", type=["csv"], key="scoring_csv")
            if scoring_file is not None:
                scoring_df = pd.read_csv(scoring_file)

        if scoring_df is not None:
            st.dataframe(scoring_df.head(50), width="stretch")
        if st.button("Score data", type="primary", disabled=scoring_df is None):
            try:
                with st.spinner("Scoring uplift"):
                    scoring_result = score_uplift_run(run_row["run_dir"], scoring_df)
            except Exception as exc:
                st.error(str(exc))
            else:
                st.session_state["scoring_result"] = scoring_result
                st.success(f"Scored {len(scoring_result.predictions):,} rows with {scoring_result.model_name}.")

        scoring_result = st.session_state.get("scoring_result")
        if scoring_result is not None:
            st.subheader("Scoring Result")
            shift = scoring_result.shift_diagnostics or {}
            if shift:
                verdict = str(shift.get("verdict", "unknown")).upper()
                if shift.get("verdict") == "high":
                    st.error(f"Training-vs-scoring shift: {verdict}. {shift.get('message')}")
                elif shift.get("verdict") == "medium":
                    st.warning(f"Training-vs-scoring shift: {verdict}. {shift.get('message')}")
                else:
                    st.success(f"Training-vs-scoring shift: {verdict}. {shift.get('message')}")
                shift_rows = pd.DataFrame(shift.get("rows", []))
                if not shift_rows.empty:
                    st.dataframe(
                        shift_rows[["feature", "type", "metric", "shift", "reference_missing_rate", "scoring_missing_rate"]].head(30),
                        width="stretch",
                    )
            st.dataframe(pd.DataFrame(scoring_result.top_counts), width="stretch")
            st.dataframe(scoring_result.predictions.head(100), width="stretch")
            st.download_button(
                "Download all scored rows",
                data=dataframe_csv_bytes(scoring_result.predictions),
                file_name=f"{scoring_result.run_id}_scored.csv",
                mime="text/csv",
            )
            top_cols = st.columns(3)
            for col, fraction in zip(top_cols, [0.05, 0.10, 0.20]):
                top_df = score_top_fraction(scoring_result.predictions, fraction)
                col.download_button(
                    f"Download Top {fraction:.0%}",
                    data=dataframe_csv_bytes(top_df),
                    file_name=f"{scoring_result.run_id}_top_{int(fraction * 100)}.csv",
                    mime="text/csv",
                )

with policy_tab:
    st.subheader("Policy Value")
    st.subheader("Industrial Threshold Rule")
    threshold_rule_rows = [
        {
            "scenario": "Coupon / Subsidy",
            "threshold_logic": "target only when uplift * gross_margin exceeds expected coupon payout, abuse risk and fatigue cost",
            "do_not_target": "sure things, subsidy abusers, users with negative incremental margin",
        },
        {
            "scenario": "Ads / Audience",
            "threshold_logic": "target when incremental conversion value exceeds impression/bid cost and iROAS guardrail",
            "do_not_target": "high attribution but low incrementality users; unsupported geo/audience cells",
        },
        {
            "scenario": "Growth Push / CRM",
            "threshold_logic": "target when activation or retention uplift exceeds message cost, opt-out risk and fatigue penalty",
            "do_not_target": "sleeping dogs, high opt-out risk users, over-frequency cohorts",
        },
        {
            "scenario": "LLM Routing",
            "threshold_logic": "route to strong model when incremental quality value exceeds extra model cost, latency and budget pacing risk",
            "do_not_target": "easy requests, low-value tasks, prompts with evidence/hallucination guard failures",
        },
    ]
    st.dataframe(pd.DataFrame(threshold_rule_rows), width="stretch", hide_index=True)
    render_launch_guardrail_board()
    render_uplift_application_expansion_matrix("policy_uplift_app")
    st.subheader("Cost-Aware Loss To Policy")
    policy_loss_rows = [
        row
        for row in deep_loss_catalog()
        if any(token in row["loss_id"] for token in ["policy", "llm", "full_funnel", "delayed", "continuous"])
    ]
    st.caption("These loss families are closest to deployment objectives: incremental profit, full-funnel value, delayed labels and LLM cost-quality routing.")
    policy_loss_df = pd.DataFrame(policy_loss_rows)
    if not policy_loss_df.empty:
        st.dataframe(
            policy_loss_df[
                ["loss_id", "status", "objective", "formula", "business_use", "metric_link", "code_path", "interview_line"]
            ],
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("No cost-aware policy loss rows are currently registered.")
    with st.expander("Coupon / Ads / Growth ROI Templates", expanded=True):
        template_df = pd.DataFrame(policy_templates())
        st.dataframe(template_df, width="stretch", hide_index=True)
        st.caption(
            "Use these templates to translate uplift scores into scenario-specific ROI: coupon cost, ad spend, fatigue penalty, marketplace subsidy, and holdout guardrails."
        )
    st.subheader("Scenario ROI Helper")
    roi_helper = render_policy_roi_helper(float(conversion_value), float(contact_cost))
    effective_conversion_value = roi_helper["conversion_value"] if roi_helper["use_helper"] else float(conversion_value)
    effective_contact_cost = roi_helper["contact_cost"] if roi_helper["use_helper"] else float(contact_cost)
    st.subheader("Scenario ROI Lab")
    st.caption(
        "Use the same scored uplift list and compare coupon, ads, growth-push and marketplace subsidy value/cost assumptions side by side."
    )
    scenario_dataset_view = pd.DataFrame(scenario_dataset_summary())
    if not scenario_dataset_view.empty:
        st.subheader("Industrial Scenario Dataset ROI Benchmarks")
        st.caption("These are generated data-level policy signals before model training; after training, Evidence shows per-scenario model smoke metrics.")
        st.dataframe(
            scenario_dataset_view[
                [
                    "scenario",
                    "dataset_id",
                    "rows",
                    "treatment_rate",
                    "outcome_rate",
                    "true_uplift_mean",
                    "policy_value_mean",
                    "policy_value_p90",
                    "policy_objective",
                ]
            ],
            width="stretch",
            hide_index=True,
        )
    st.subheader("Resume Scenario Top-K Policy Benchmarks")
    st.caption("Baidu Waimai, DiDi and Shopee scenarios are ranked by oracle policy value to show the business target each model should approximate.")
    resume_policy_rows = pd.DataFrame(resume_scenario_policy_benchmarks())
    if not resume_policy_rows.empty:
        st.dataframe(resume_policy_rows, width="stretch", hide_index=True)
    with st.expander("DiDi City-Time Budget Allocation Simulator", expanded=True):
        budget_total = st.number_input(
            "Total DiDi subsidy budget",
            min_value=10_000.0,
            max_value=2_000_000.0,
            value=100_000.0,
            step=10_000.0,
            key="didi_budget_allocation_total",
        )
        budget_plan = didi_budget_allocation_plan(total_budget=float(budget_total), max_cells=30)
        render_stat_grid(
            [
                ("Selected Cells", f"{budget_plan['summary']['selected_cells']:,}", False),
                ("Used Budget", f"{budget_plan['summary']['used_budget']:.2f}", False),
                ("Policy Value", f"{budget_plan['summary']['expected_policy_value']:.2f}", False),
                ("Mean Spillover", f"{budget_plan['summary']['mean_spillover_risk']:.4f}", False),
            ],
            columns=4,
        )
        allocation_rows = pd.DataFrame(budget_plan.get("allocations") or [])
        if not allocation_rows.empty:
            st.dataframe(allocation_rows, width="stretch", hide_index=True)
    st.subheader("Metric Disagreement Lab")
    st.caption("QINI/AUUC answer ranking quality; incremental profit and policy value decide whether a bucket is worth treating.")
    disagreement = pd.DataFrame(policy_metric_disagreement_rows())
    st.dataframe(disagreement, width="stretch", hide_index=True)
    st.bar_chart(disagreement.set_index("case")["policy_value"])
    st.subheader("LLM Routing ROI Simulator")
    with st.expander("Cost-quality escalation policy", expanded=True):
        llm_c1, llm_c2, llm_c3, llm_c4 = st.columns(4)
        request_volume = llm_c1.number_input("Requests / day", min_value=100, value=10000, step=500, key="llm_route_requests")
        strong_lift = llm_c2.number_input("Quality uplift / escalated request", min_value=0.0, value=0.08, step=0.01, format="%.4f", key="llm_route_quality_lift")
        value_per_quality = llm_c3.number_input("Value / quality point", min_value=0.0, value=2.0, step=0.25, key="llm_route_quality_value")
        route_fraction = llm_c4.slider("Escalation fraction", min_value=0.01, max_value=1.0, value=0.20, step=0.01, key="llm_route_fraction")
        llm_c5, llm_c6, llm_c7, llm_c8 = st.columns(4)
        small_cost = llm_c5.number_input("Small model cost / 1k", min_value=0.0, value=0.20, step=0.05, key="llm_route_small_cost")
        strong_cost = llm_c6.number_input("Strong model cost / 1k", min_value=0.0, value=4.00, step=0.25, key="llm_route_strong_cost")
        latency_penalty = llm_c7.number_input("Latency penalty / escalated request", min_value=0.0, value=0.002, step=0.001, format="%.4f", key="llm_route_latency_penalty")
        router_precision = llm_c8.slider("Router precision", min_value=0.10, max_value=1.0, value=0.75, step=0.05, key="llm_route_precision")
        routed_requests = int(float(request_volume) * float(route_fraction))
        incremental_cost = routed_requests * ((float(strong_cost) - float(small_cost)) / 1000.0 + float(latency_penalty))
        incremental_value = routed_requests * float(strong_lift) * float(value_per_quality) * float(router_precision)
        net_value = incremental_value - incremental_cost
        all_strong_cost = int(request_volume) * ((float(strong_cost) - float(small_cost)) / 1000.0 + float(latency_penalty))
        all_strong_value = int(request_volume) * float(strong_lift) * float(value_per_quality)
        all_strong_net = all_strong_value - all_strong_cost
        render_stat_grid(
            [
                ("Escalated Requests", f"{routed_requests:,}", False),
                ("Incremental Value", f"{incremental_value:.4f}", False),
                ("Incremental Cost", f"{incremental_cost:.4f}", False),
                ("Net Value", f"{net_value:.4f}", False),
            ],
            columns=4,
        )
        llm_routing_rows = pd.DataFrame(
            [
                {
                    "policy": "All small model",
                    "escalation_fraction": 0.0,
                    "incremental_value": 0.0,
                    "incremental_cost": 0.0,
                    "net_value": 0.0,
                    "guardrail": "lowest cost, may miss hard-query quality",
                },
                {
                    "policy": "All strong model",
                    "escalation_fraction": 1.0,
                    "incremental_value": all_strong_value,
                    "incremental_cost": all_strong_cost,
                    "net_value": all_strong_net,
                    "guardrail": "highest cost, budget and latency risk",
                },
                {
                    "policy": "Uplift router",
                    "escalation_fraction": float(route_fraction),
                    "incremental_value": incremental_value,
                    "incremental_cost": incremental_cost,
                    "net_value": net_value,
                    "guardrail": "route only high incremental-quality queries; validate with holdout",
                },
            ]
        )
        st.dataframe(llm_routing_rows, width="stretch", hide_index=True)
        st.markdown("#### 多动作路由策略样例")
        multi_action_rows = pd.DataFrame(
            [
                {
                    "action": "strong_model",
                    "use_when": "复杂推理、代码、数学、混合语言问题",
                    "value_signal": "ambiguity / complex task uplift",
                    "cost_risk": "模型成本和延迟较高",
                    "policy_rule": "quality uplift * value > model cost + latency penalty",
                },
                {
                    "action": "rag",
                    "use_when": "需要证据、引用、政策/知识库 grounding",
                    "value_signal": "retrieval_need 和 evidence_requirement 高",
                    "cost_risk": "检索延迟、证据失败、上下文污染",
                    "policy_rule": "evidence quality uplift > retrieval cost + evidence failure risk",
                },
                {
                    "action": "tool",
                    "use_when": "计算、数据分析、代码执行、数据库查询",
                    "value_signal": "tool_need 高且 cheap model 容易幻觉",
                    "cost_risk": "工具调用成本、执行失败、SLO 风险",
                    "policy_rule": "tool success uplift > tool cost + latency risk",
                },
                {
                    "action": "human_review",
                    "use_when": "高安全风险、高客诉价值、合规/医疗/金融人工复核",
                    "value_signal": "safety_risk / user_value / prior_fail_rate 高",
                    "cost_risk": "人工成本、队列等待和吞吐限制",
                    "policy_rule": "human review uplift > labor cost + queue penalty",
                },
            ]
        )
        st.dataframe(multi_action_rows, width="stretch", hide_index=True)
        st.caption("新增 `Synthetic LLM Multi-Action Routing 9k` 将 strong model、RAG、tool、human review 作为候选 action；训练仍使用统一 uplift 输出，Policy 用 cost-quality-risk 目标做上线阈值。")
        st.caption("Decision rule: route to the stronger LLM only when expected quality uplift times value exceeds incremental model cost, latency penalty and budget pacing risk.")
    st.subheader("LLM Routing Drift Lab: 固定阈值 vs 自适应 Bandit")
    st.caption(
        "当模型价格、质量、证据风险或 action 池变化时，固定 uplift 阈值可能失效；这张表展示什么时候需要升级到带 guardrail 的 contextual bandit。"
    )
    drift_df = pd.DataFrame(llm_routing_bandit_drift_rows())
    if not drift_df.empty:
        st.dataframe(drift_df, width="stretch", hide_index=True)
        st.line_chart(drift_df.set_index("period")[["fixed_threshold_net", "adaptive_bandit_net", "all_strong_net"]])
    st.subheader("从离线 uplift 到在线 bandit 的上线门禁")
    st.caption(
        "固定阈值适合可解释、可复现的第一版策略；当预算、流量、模型池或反馈持续变化时，再在有随机探索和 OPE 证据后升级到 contextual bandit。"
    )
    exploration_df = pd.DataFrame(exploration_ope_guardrail_rows())
    st.dataframe(exploration_df, width="stretch", hide_index=True)
    st.download_button(
        "Download exploration / OPE / bandit guardrails",
        data=dataframe_csv_bytes(exploration_df),
        file_name="deepuplift_exploration_ope_bandit_guardrails.csv",
        mime="text/csv",
        key="download_exploration_ope_bandit_guardrails",
    )
    st.subheader("LLM 多动作 Routing OPE Mini-Lab")
    st.caption(
        "用合成 logged routing 数据演示 DM / IPS / SNIPS / DR off-policy evaluation：coverage 和 effective sample size 太低时，先补随机探索，而不是直接相信新 router。"
    )
    ope_summary = llm_routing_ope_summary()
    if ope_summary.get("status") == "ok":
        ope_df = pd.DataFrame(ope_summary.get("rows") or [])
        render_stat_grid(
            [
                ("OPE Rows", f"{ope_summary.get('rows_used', 0):,}", False),
                ("Policies", f"{len(ope_df):,}", False),
                ("Formula", "DR OPE", True),
                ("Dataset", "LLM Multi-Action", True),
            ],
            columns=4,
        )
        st.code(ope_summary.get("formula", ""), language="text")
        st.dataframe(ope_df, width="stretch", hide_index=True)
        coverage_df = pd.DataFrame(llm_routing_action_coverage_rows())
        if not coverage_df.empty:
            st.markdown("#### Action-level exploration coverage")
            st.caption("每个 action arm 都要有足够 logged support；coverage deficient 的动作需要先补探索流量，再谈 OPE 或 bandit。")
            st.dataframe(coverage_df, width="stretch", hide_index=True)
            st.bar_chart(coverage_df.set_index("logged_action")["share"])
        st.download_button(
            "Download LLM routing OPE mini-lab",
            data=dataframe_csv_bytes(ope_df),
            file_name="deepuplift_llm_routing_ope_mini_lab.csv",
            mime="text/csv",
            key="download_llm_routing_ope_mini_lab",
        )
    else:
        st.info(f"OPE mini-lab unavailable: {ope_summary}")
    offline_online_path = Path("docs/DEEPUplift_AGENT_OFFLINE_ONLINE_PLAYBOOK.md")
    if offline_online_path.exists():
        with st.expander("Offline-To-Online Validation Playbook", expanded=False):
            offline_online_doc = offline_online_path.read_text(encoding="utf-8")
            st.markdown(offline_online_doc)
            st.download_button(
                "Download offline-online playbook",
                data=offline_online_doc.encode("utf-8"),
                file_name="DEEPUplift_AGENT_OFFLINE_ONLINE_PLAYBOOK.md",
                mime="text/markdown",
                key="policy_offline_online_playbook_download",
            )
    history = load_run_history()
    policy_sources = ["Latest training result", "History run", "Latest scoring result"]
    policy_source = st.radio("Source", policy_sources, horizontal=True)
    policy_df = None
    observed_cols = ("outcome", "treatment")

    if policy_source == "Latest training result":
        result = st.session_state.get("uplift_result")
        if result is not None and Path(result.artifacts["predictions"]).exists():
            policy_df = pd.read_csv(result.artifacts["predictions"])
        else:
            st.info("No latest training predictions available.")
    elif policy_source == "History run":
        if history.empty:
            st.info("No saved runs yet.")
        else:
            run_id = st.selectbox("Run", history["run_id"].tolist(), key="policy_run")
            run_row = history[history["run_id"] == run_id].iloc[0]
            predictions_path = Path(run_row["predictions"])
            if predictions_path.exists():
                policy_df = pd.read_csv(predictions_path)
    else:
        scoring_result = st.session_state.get("scoring_result")
        if scoring_result is not None:
            policy_df = scoring_result.predictions
        else:
            st.info("No scoring result yet. Use the Predict page first.")

    if policy_df is None:
        st.session_state.pop("policy_context", None)

    if policy_df is not None:
        policy_controls = st.columns(3)
        with policy_controls[0]:
            max_fraction = st.slider("Max targeting fraction", min_value=0.05, max_value=1.0, value=1.0, step=0.05)
        with policy_controls[1]:
            budget_cap = st.number_input(
                "Budget cap",
                min_value=0.0,
                value=0.0,
                step=100.0,
                help="0 means unlimited budget. Uses the Scenario ROI Helper when enabled, otherwise the sidebar contact cost.",
            )
        with policy_controls[2]:
            max_contacts = st.number_input(
                "Max contacts",
                min_value=0,
                max_value=max(len(policy_df), 1),
                value=0,
                step=max(1, len(policy_df) // 20),
                help="0 means unlimited contacts.",
            )
        fractions = default_policy_fractions(max_fraction=max_fraction, step=0.05)
        has_observed = all(col in policy_df.columns for col in observed_cols)
        policy = policy_value_curve(
            policy_df,
            uplift_col="uplift_score",
            outcome_col="outcome" if has_observed else None,
            treatment_col="treatment" if has_observed else None,
            conversion_value=float(effective_conversion_value),
            contact_cost=float(effective_contact_cost),
            fractions=fractions,
        )
        policy_rows = pd.DataFrame(policy["rows"])
        scenario_specs = [
            {
                "scenario": "Coupon",
                "value": 30.0,
                "cost": 5.0 * 0.35 + 0.10,
                "value_basis": "gross margin per order",
                "cost_basis": "expected coupon payout + abuse/fatigue penalty",
                "guardrail": "exclude sure things, cap subsidy, monitor redemption abuse",
            },
            {
                "scenario": "Ads",
                "value": 50.0,
                "cost": 20.0 / 1000.0,
                "value_basis": "incremental conversion value",
                "cost_basis": "CPM-implied exposure cost",
                "guardrail": "holdout/geo lift, iROAS lower bound, attribution sanity check",
            },
            {
                "scenario": "Growth Push",
                "value": 15.0,
                "cost": 0.03 + 0.08 + 0.10,
                "value_basis": "activation or LTV proxy",
                "cost_basis": "message cost + fatigue + opt-out penalty",
                "guardrail": "frequency cap, opt-out uplift, delayed feedback window",
            },
            {
                "scenario": "Marketplace Subsidy",
                "value": 30.0,
                "cost": 4.0 + 0.20 + 0.50,
                "value_basis": "gross profit / balance value",
                "cost_basis": "subsidy + ops cost + spillover buffer",
                "guardrail": "budget cap, supply-demand balance, spillover-aware experiment",
            },
            {
                "scenario": "LLM Routing",
                "value": float(value_per_quality),
                "cost": (float(strong_cost) - float(small_cost)) / 1000.0 + float(latency_penalty),
                "value_basis": "quality point value or task-success value",
                "cost_basis": "strong-minus-small model cost + latency penalty",
                "guardrail": "budget pacing, judge calibration, hallucination/evidence failures",
            },
        ]
        scenario_lab_rows = []
        for spec in scenario_specs:
            scenario_policy = policy_value_curve(
                policy_df,
                uplift_col="uplift_score",
                outcome_col="outcome" if has_observed else None,
                treatment_col="treatment" if has_observed else None,
                conversion_value=float(spec["value"]),
                contact_cost=float(spec["cost"]),
                fractions=fractions,
            )
            scenario_best = scenario_policy.get("best") or {}
            scenario_ranking_col = scenario_policy.get("ranking_col", "predicted_net_value")
            scenario_lab_rows.append(
                {
                    "scenario": spec["scenario"],
                    "value": spec["value"],
                    "cost": spec["cost"],
                    "best_top_fraction": scenario_best.get("top_fraction"),
                    "best_predicted_net_value": scenario_best.get("predicted_net_value"),
                    "best_observed_net_value": scenario_best.get("observed_net_value"),
                    "ranking_col": scenario_ranking_col,
                    "value_basis": spec["value_basis"],
                    "cost_basis": spec["cost_basis"],
                    "guardrail": spec["guardrail"],
                }
            )
        scenario_lab = pd.DataFrame(scenario_lab_rows)
        st.dataframe(scenario_lab, width="stretch", hide_index=True)
        if "best_predicted_net_value" in scenario_lab.columns:
            chart_lab = scenario_lab.set_index("scenario")[["best_predicted_net_value"]]
            if "best_observed_net_value" in scenario_lab.columns and scenario_lab["best_observed_net_value"].notna().any():
                chart_lab["best_observed_net_value"] = scenario_lab.set_index("scenario")["best_observed_net_value"]
            st.bar_chart(chart_lab)
        scenario_policy_metrics = industrial_policy_metrics(policy_df, uplift_col="uplift_score") if "uplift_score" in policy_df.columns else {}
        scenario_policy_rows = pd.DataFrame(scenario_policy_metrics.get("top_k", []))
        if not scenario_policy_rows.empty:
            st.subheader("Run-Level Industrial Policy Metrics")
            st.caption(
                "Uses the selected run/scoring table directly. If scenario columns are present, this shows Top-K net policy value, "
                "ROI proxy, and risk means using the run's own uplift ranking."
            )
            top10_policy_rows = pd.DataFrame(scenario_policy_metrics.get("top10", []))
            if not top10_policy_rows.empty:
                st.dataframe(top10_policy_rows, width="stretch", hide_index=True)
            with st.expander("All run-level scenario policy buckets", expanded=False):
                st.dataframe(scenario_policy_rows, width="stretch", hide_index=True)
        if not policy_rows.empty:
            budget_limited = float(budget_cap) > 0
            contact_limited = int(max_contacts) > 0
            policy_rows["within_budget"] = True if not budget_limited else policy_rows["contact_cost"] <= float(budget_cap)
            policy_rows["within_contact_cap"] = True if not contact_limited else policy_rows["rows"] <= int(max_contacts)
            policy_rows["within_constraints"] = policy_rows["within_budget"] & policy_rows["within_contact_cap"]
        best = policy["best"] or {}
        if best:
            metric_name = "observed net value" if policy["has_observed"] else "predicted net value"
            best_value = best.get(policy["ranking_col"])
            st.success(f"Recommended threshold: Top {best.get('top_fraction', 0):.0%}; {metric_name}: {best_value:.4f}")
        constrained_best = {}
        if not policy_rows.empty:
            eligible = policy_rows[policy_rows["within_constraints"]]
            ranking_col = policy["ranking_col"]
            eligible = eligible[eligible[ranking_col].notna()] if ranking_col in eligible.columns else eligible
            if eligible.empty:
                st.warning("No threshold satisfies the current budget/contact constraints.")
            else:
                constrained_best = eligible.sort_values(ranking_col, ascending=False).iloc[0].to_dict()
                constrained_value = constrained_best.get(ranking_col)
                st.info(
                    f"Constrained recommendation: Top {constrained_best.get('top_fraction', 0):.0%}; "
                    f"rows {int(constrained_best.get('rows', 0)):,}; "
                    f"{ranking_col}: {constrained_value:.4f}"
                )
                remaining_budget = None if float(budget_cap) <= 0 else float(budget_cap) - float(constrained_best.get("contact_cost", 0) or 0)
                render_stat_grid(
                    [
                        ("Constrained Top-K", f"Top {constrained_best.get('top_fraction', 0):.0%}", False),
                        ("Contacts", f"{int(constrained_best.get('rows', 0)):,}", False),
                        ("Cost", f"{float(constrained_best.get('contact_cost', 0) or 0):.4f}", False),
                        ("Remaining Budget", "unlimited" if remaining_budget is None else f"{remaining_budget:.4f}", True),
                    ],
                    columns=4,
                )
                constrained_top = policy_df.sort_values("uplift_score", ascending=False).head(int(constrained_best.get("rows", 0)))
                st.download_button(
                    "Download constrained audience",
                    data=dataframe_csv_bytes(constrained_top),
                    file_name="policy_constrained_audience.csv",
                    mime="text/csv",
                )
        st.session_state["policy_context"] = {
            "source": policy_source,
            "rows": len(policy_df),
            "best": best,
            "constrained_best": constrained_best,
            "budget_cap": float(budget_cap),
            "max_contacts": int(max_contacts),
            "contact_cost": float(effective_contact_cost),
            "conversion_value": float(effective_conversion_value),
            "scenario": roi_helper.get("scenario"),
            "scenario_assumptions": roi_helper.get("assumptions"),
            "scenario_helper_enabled": bool(roi_helper.get("use_helper")),
            "ranking_col": policy.get("ranking_col"),
            "has_observed": policy.get("has_observed"),
        }
        chart_cols = ["predicted_net_value"]
        if "observed_net_value" in policy_rows.columns:
            chart_cols.append("observed_net_value")
        st.line_chart(policy_rows.set_index("top_fraction")[chart_cols])
        st.dataframe(policy_rows, width="stretch")
        st.download_button(
            "Download policy curve",
            data=dataframe_csv_bytes(policy_rows),
            file_name="policy_value_curve.csv",
            mime="text/csv",
        )

with history_tab:
    history = load_run_history()
    if history.empty:
        st.info("No saved runs yet.")
    else:
        st.subheader("Run History")
        with st.expander("Frontier Metric Reading Guide", expanded=False):
            st.caption("History keeps ECUP/delayed-feedback summary fields so old runs remain explainable after the live session ends.")
            st.dataframe(pd.DataFrame(frontier_metric_reading_rows()), width="stretch", hide_index=True)
        f1, f2, f3, f4 = st.columns([0.24, 0.26, 0.24, 0.26])
        with f1:
            level_options = ["all", *sorted(history["readiness_level"].dropna().unique().tolist())]
            selected_level = st.selectbox("Readiness level", level_options, key="history_readiness_level")
        with f2:
            model_filter = st.multiselect(
                "Models",
                sorted(history["model"].dropna().unique().tolist()),
                default=sorted(history["model"].dropna().unique().tolist()),
                key="history_model_filter",
            )
        with f3:
            verdict_options = ["all", *sorted(history["sensitivity_verdict"].dropna().unique().tolist())]
            selected_verdict = st.selectbox("Sensitivity", verdict_options, key="history_sensitivity")
        with f4:
            blockers_only = st.checkbox("Only blockers", value=False, key="history_blockers_only")
        all_tags = sorted(
            {
                tag.strip()
                for tags in history.get("tags", pd.Series(dtype=str)).fillna("").astype(str)
                for tag in tags.split(",")
                if tag.strip()
            }
        )
        selected_tags = st.multiselect("Tags", all_tags, default=[], key="history_tags")
        min_readiness = st.slider(
            "Minimum readiness score",
            min_value=0,
            max_value=100,
            value=0,
            step=5,
            key="history_min_readiness",
        )
        filtered_history = history.copy()
        if selected_level != "all":
            filtered_history = filtered_history[filtered_history["readiness_level"] == selected_level]
        if model_filter:
            filtered_history = filtered_history[filtered_history["model"].isin(model_filter)]
        if selected_verdict != "all":
            filtered_history = filtered_history[filtered_history["sensitivity_verdict"] == selected_verdict]
        filtered_history = filtered_history[pd.to_numeric(filtered_history["readiness_score"], errors="coerce").fillna(0) >= min_readiness]
        if blockers_only:
            filtered_history = filtered_history[filtered_history["readiness_blockers"].fillna("").astype(str).str.len() > 0]
        if selected_tags:
            tag_pattern = "|".join(re.escape(tag) for tag in selected_tags)
            filtered_history = filtered_history[filtered_history["tags"].fillna("").astype(str).str.contains(tag_pattern, case=False, regex=True)]
        render_stat_grid(
            [
                ("Runs", f"{len(filtered_history):,}/{len(history):,}", False),
                ("Ready Avg", f"{pd.to_numeric(filtered_history['readiness_score'], errors='coerce').mean() if not filtered_history.empty else 0:.1f}", False),
                ("Blockers", f"{int((filtered_history['readiness_blockers'].fillna('').astype(str).str.len() > 0).sum()) if not filtered_history.empty else 0}", False),
                ("Models", f"{filtered_history['model'].nunique() if not filtered_history.empty else 0}", False),
                (
                    "Frontier Runs",
                    f"{int(filtered_history.get('frontier_metric_schema', pd.Series(dtype=str)).fillna('').astype(str).str.len().gt(0).sum()) if not filtered_history.empty else 0}",
                    True,
                ),
            ],
            columns=5,
        )
        if filtered_history.empty:
            st.warning("No runs match the current filters.")
        else:
            render_compare_decision_board(
                filtered_history,
                caption="Latest saved runs: readiness is absolute; QINI and policy net are scaled inside this history table.",
            )
        history_display_cols = [
            col
            for col in [
                "run_id",
                "tags",
                "model",
                "readiness_score",
                "readiness_level",
                "qini_ci_low",
                "qini",
                "auuc",
                "calibration_mae",
                "policy_best_fraction",
                "policy_best_net",
                "sensitivity_verdict",
                "weak_overlap_rate",
                "oracle_top10_recall",
                "business_top10_overlap",
                "frontier_metric_schema",
                "full_funnel_top10_conversion_uplift",
                "full_funnel_top10_click_uplift",
                "delayed_d30_top10_uplift",
                "delayed_d14_top10_uplift",
                "delayed_censored_rate",
                "user_note",
                "readiness_blockers",
            ]
            if col in history.columns
        ]
        st.dataframe(filtered_history[history_display_cols], width="stretch")
        st.download_button(
            "Download filtered run history CSV",
            data=filtered_history.to_csv(index=False).encode("utf-8"),
            file_name="deepuplift_filtered_run_history.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download full run history CSV",
            data=history.to_csv(index=False).encode("utf-8"),
            file_name="deepuplift_run_history.csv",
            mime="text/csv",
        )
        detail_history = filtered_history if not filtered_history.empty else history
        selected_run = st.selectbox("Open run", detail_history["run_id"].tolist())
        run_row = detail_history[detail_history["run_id"] == selected_run].iloc[0]
        run_dir = Path(run_row["run_dir"])
        user_note_path = run_dir / "user_note.json"
        existing_user_note = {}
        if user_note_path.exists():
            try:
                existing_user_note = json.loads(user_note_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                existing_user_note = {}
        with st.expander("Run Tags / Notes", expanded=False):
            tag_text = st.text_input(
                "Tags",
                value=", ".join(existing_user_note.get("tags", [])),
                placeholder="candidate, needs-review, production-check",
                key=f"tags_{selected_run}",
            )
            note_text = st.text_area(
                "Note",
                value=existing_user_note.get("note", ""),
                height=120,
                key=f"user_note_{selected_run}",
            )
            quick_tags = [
                ("Mark candidate", "candidate"),
                ("Needs review", "needs-review"),
                ("Production check", "production-check"),
                ("Archive", "archive"),
            ]
            quick_cols = st.columns(len(quick_tags))
            for quick_col, (label, quick_tag) in zip(quick_cols, quick_tags):
                with quick_col:
                    if st.button(label, key=f"quick_tag_{quick_tag}_{selected_run}"):
                        tags = [tag.strip() for tag in tag_text.split(",") if tag.strip()]
                        if quick_tag not in tags:
                            tags.append(quick_tag)
                        save_json(
                            user_note_path,
                            {
                                "run_id": selected_run,
                                "tags": tags,
                                "note": note_text,
                                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
                            },
                        )
                        st.success(f"Tagged {selected_run} as {quick_tag}.")
                        st.rerun()
            if st.button("Save run note", key=f"save_note_{selected_run}"):
                tags = [tag.strip() for tag in tag_text.split(",") if tag.strip()]
                save_json(
                    user_note_path,
                    {
                        "run_id": selected_run,
                        "tags": tags,
                        "note": note_text,
                        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
                    },
                )
                st.success(f"Saved note for {selected_run}.")
        metrics_path = Path(run_row["run_dir"]) / "metrics.json"
        if metrics_path.exists():
            with st.expander("Selected run decision readiness", expanded=True):
                try:
                    selected_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                    render_decision_readiness(selected_metrics)
                except json.JSONDecodeError:
                    st.warning("Selected run metrics file is not valid JSON.")
        predictions_path = Path(run_row["predictions"])
        if predictions_path.exists():
            st.download_button(
                "Download selected predictions",
                data=predictions_path.read_bytes(),
                file_name=f"{selected_run}_predictions.csv",
                mime="text/csv",
            )
        readiness_path = Path(run_row.get("readiness", ""))
        if readiness_path.exists():
            st.download_button(
                "Download selected readiness JSON",
                data=readiness_path.read_bytes(),
                file_name=f"{selected_run}_readiness.json",
                mime="application/json",
            )
        note_path = Path(run_row.get("run_note", ""))
        if note_path.exists():
            st.download_button(
                "Download selected run note",
                data=note_path.read_bytes(),
                file_name=f"{selected_run}_run_note.md",
                mime="text/markdown",
            )
        st.subheader("Evidence Summary")
        st.dataframe(run_evidence_summary(run_dir), width="stretch", hide_index=True)
        evidence_zip, evidence_files = build_run_evidence_zip(run_dir, selected_run)
        st.download_button(
            "Download selected run evidence ZIP",
            data=evidence_zip,
            file_name=f"{selected_run}_evidence.zip",
            mime="application/zip",
        )
        st.caption("Evidence files: " + ", ".join(evidence_files))

with models_tab:
    st.subheader("Model Catalog")
    st.caption("Scenario Model Matrix exports map every registered model to business scenario tags and readiness metadata.")
    catalog_task = st.radio("Task filter", ["all", "classification", "regression"], horizontal=True, key="catalog_task")
    catalog = model_catalog_dataframe(None if catalog_task == "all" else catalog_task)
    render_backend_cards(catalog)
    render_frontend_framework_audit("models")
    render_ml_training_platform_audit("models_ml_training")
    render_public_product_inspiration_matrix("models_public_product")
    render_deep_loss_network_catalog()
    render_model_risk_matrix("models")
    render_open_source_framework_audit("models")
    render_model_source_provenance()
    render_frontier_model_architecture()
    render_scenario_model_decision_ladder()
    with st.expander("Business Scenario Model Recommendation", expanded=True):
        scenario_rows_df = pd.DataFrame(scenario_model_rows())
        if not scenario_rows_df.empty:
            scenario_label = st.selectbox(
                "Scenario",
                scenario_rows_df["scenario"].tolist(),
                key="business_scenario_model_recommendation",
            )
            selected_scenario = scenario_rows_df[scenario_rows_df["scenario"] == scenario_label].iloc[0].to_dict()
            render_stat_grid(
                [
                    ("Scenario", selected_scenario["scenario"], True),
                    ("Primary Metrics", selected_scenario["primary_metrics"], True),
                    ("Recommended Models", selected_scenario["recommended_models"], True),
                    ("Major Risks", selected_scenario["major_risks"], True),
                ],
                columns=2,
            )
            st.dataframe(scenario_rows_df, width="stretch", hide_index=True)
            st.caption("This guide maps business treatment design to runnable models and interview evidence.")
    if not catalog.empty:
        scenario_options = sorted(
            {
                tag.strip()
                for value in catalog["scenario_tags"].dropna().tolist()
                for tag in str(value).split(";")
                if tag.strip()
            }
        )
        st.caption("Scenario Tags connect each model to coupon, ads, growth, recommendation, marketplace, and benchmark use cases.")
        f1, f2, f3, f4, f5 = st.columns([0.20, 0.20, 0.22, 0.14, 0.24])
        with f1:
            source_filter = st.multiselect("Source", sorted(catalog["source"].unique().tolist()), default=sorted(catalog["source"].unique().tolist()))
        with f2:
            family_filter = st.multiselect("Family", sorted(catalog["family"].unique().tolist()), default=sorted(catalog["family"].unique().tolist()))
        with f3:
            treatment_filter = st.multiselect(
                "Treatment Type",
                sorted(catalog["treatment_type"].unique().tolist()),
                default=sorted(catalog["treatment_type"].unique().tolist()),
            )
        with f4:
            status_filter = st.radio("Status", ["all", *sorted(catalog["status"].unique().tolist())], key="catalog_status")
        with f5:
            scenario_filter = st.multiselect("Scenario Tag", scenario_options, default=scenario_options)
        catalog_view = catalog[
            catalog["source"].isin(source_filter)
            & catalog["family"].isin(family_filter)
            & catalog["treatment_type"].isin(treatment_filter)
        ].copy()
        if status_filter != "all":
            catalog_view = catalog_view[catalog_view["status"] == status_filter]
        if scenario_filter:
            catalog_view = catalog_view[
                catalog_view["scenario_tags"].apply(
                    lambda value: any(tag in str(value) for tag in scenario_filter)
                )
            ]
    else:
        catalog_view = catalog
    ready_count = int((catalog["status"] == "ready").sum()) if not catalog.empty else 0
    render_stat_grid(
        [
            ("Registered", f"{len(catalog):,}", False),
            ("Runnable", f"{ready_count:,}", False),
            ("Frameworks", f"{catalog['source'].nunique() if not catalog.empty else 0:,}", False),
            ("Families", f"{catalog['family'].nunique() if not catalog.empty else 0:,}", False),
        ]
    )
    catalog_audit = model_catalog_audit_payload(catalog)
    with st.expander("Catalog Audit", expanded=True):
        render_stat_grid(
            [
                ("Audit", "pass" if not catalog_audit["missing_descriptions"] else "warn", True),
                ("Registry Rows", f"{catalog_audit['catalog_rows']:,}", False),
                ("Ready Models", f"{catalog_audit['ready_models']:,}", False),
                ("Sources", f"{len(catalog_audit['source_counts']):,}", False),
            ]
        )
        st.caption(catalog_audit["recommendation"])
        st.dataframe(pd.DataFrame(catalog_audit["checks"]), width="stretch", hide_index=True)
        ready_sources = pd.DataFrame(
            [
                {"source": source, "ready_models": count}
                for source, count in sorted(catalog_audit["ready_source_counts"].items())
            ]
        )
        if not ready_sources.empty:
            st.subheader("Ready Backends")
            st.bar_chart(ready_sources.set_index("source")["ready_models"])
            st.dataframe(ready_sources, width="stretch", hide_index=True)
        if catalog_audit["missing_descriptions"]:
            st.warning("Missing descriptions: " + ", ".join(catalog_audit["missing_descriptions"][:20]))
        if catalog_audit["missing_model_presets"]:
            st.warning("Missing parameter presets: " + ", ".join(catalog_audit["missing_model_presets"][:20]))
        st.download_button(
            "Download catalog audit JSON",
            data=json.dumps(json_clean(catalog_audit), ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="deepuplift_model_catalog_audit.json",
            mime="application/json",
        )
        scenario_matrix_csv = Path("reports/scenario_model_matrix_latest.csv")
        scenario_matrix_json = Path("reports/scenario_model_matrix_latest.json")
        if scenario_matrix_csv.exists():
            st.download_button(
                "Download scenario model matrix CSV",
                data=scenario_matrix_csv.read_bytes(),
                file_name=scenario_matrix_csv.name,
                mime="text/csv",
                key="download_scenario_model_matrix_csv",
            )
        if scenario_matrix_json.exists():
            st.download_button(
                "Download scenario model matrix JSON",
                data=scenario_matrix_json.read_bytes(),
                file_name=scenario_matrix_json.name,
                mime="application/json",
                key="download_scenario_model_matrix_json",
            )
    if not catalog.empty:
        treatment_counts = (
            catalog_view.groupby(["treatment_type", "status"], as_index=False)
            .size()
            .rename(columns={"size": "models"})
            .sort_values(["treatment_type", "status"])
        )
        family_counts = catalog_view.groupby(["source", "family", "stage", "status"], as_index=False).size().rename(columns={"size": "models"})
        preset_coverage = (
            catalog_view.groupby(["source", "status"], as_index=False)
            .agg(
                models=("model", "count"),
                avg_presets=("preset_count", "mean"),
                min_presets=("preset_count", "min"),
            )
            .sort_values(["source", "status"])
        )
        left, right = st.columns([0.42, 0.58])
        with left:
            st.subheader("Treatment Type Matrix")
            if not treatment_counts.empty:
                st.dataframe(treatment_counts, width="stretch", hide_index=True)
            st.subheader("Coverage")
            st.dataframe(family_counts, width="stretch")
            st.subheader("Preset Coverage")
            if not preset_coverage.empty:
                st.bar_chart(preset_coverage.set_index("source")["avg_presets"])
                st.dataframe(preset_coverage, width="stretch", hide_index=True)
        with right:
            st.subheader("Models")
            st.dataframe(
                catalog_view[
                    [
                        "model",
                        "status",
                        "source",
                        "family",
                        "treatment_type",
                        "stage",
                        "tasks",
                        "recommended_preset",
                        "preset_count",
                        "scenario_tags",
                        "setup_hint",
                        "best_for",
                        "description",
                    ]
                ],
                width="stretch",
            )
        st.subheader("Model Card")
        model_card_options = catalog_view["model"].tolist() if not catalog_view.empty else catalog["model"].tolist()
        selected_model_card = st.selectbox("Open model card", model_card_options, key="model_card_open")
        card = build_model_card(selected_model_card, description=MODEL_DESCRIPTIONS.get(selected_model_card, ""))
        card["scenario_tags"] = model_scenario_tags(selected_model_card)
        render_model_card_panel(card)
        st.download_button(
            "Download selected model card JSON",
            data=json.dumps(json_clean(card), ensure_ascii=False, indent=2).encode("utf-8"),
            file_name=f"{selected_model_card}_model_card.json",
            mime="application/json",
        )
        with st.expander("Model Family Decision Guide", expanded=False):
            st.dataframe(model_decision_guide(), width="stretch", hide_index=True)
        with st.expander("Catalog Sources", expanded=False):
            st.markdown(
                """
                - [EconML DR/DML/forest estimators](https://econml.azurewebsites.net/spec/estimation/dr.html)
                - [CausalML uplift tree, uplift forest, and meta-learner documentation](https://causalml.readthedocs.io/en/latest/methodology.html)
                - [scikit-uplift SoloModel, TwoModels, and transformation models](https://www.uplift-modeling.com/en/latest/api/models/TwoModels.html)
                - [UpliftBench 2026 marketing uplift comparison](https://arxiv.org/abs/2604.06123)
                """
            )
        with st.expander("Guarded Backend Runtime Guide", expanded=False):
            guarded_rows = [
                {
                    "backend": "XGBoost",
                    "catalog_status": "guarded",
                    "why_guarded": "local macOS smoke training can enter a low-level unstable state",
                    "enable_path": "set DEEPUPLIFT_ENABLE_XGBOOST=1 before starting Streamlit",
                },
                {
                    "backend": "CatBoost",
                    "catalog_status": "optional dependency",
                    "why_guarded": "heavy categorical boosting backend, not required for default demo",
                    "enable_path": "install catboost into the conda env, then rerun model audit",
                },
                {
                    "backend": "CausalML",
                    "catalog_status": "optional Python 3.11 path",
                    "why_guarded": "package ecosystem is more stable in a Python 3.11 helper env",
                    "enable_path": "run scripts/setup_causalml_py311.sh and use the helper launcher",
                },
                {
                    "backend": "UTBoost",
                    "catalog_status": "optional guarded plugin",
                    "why_guarded": "new uplift GBDT dependency; should be compared against stable P0 LightGBM DR/R/DML baselines before default use",
                    "enable_path": "install utboost into the conda env, restart Streamlit, then run catalog audit and a no-UI compare smoke",
                },
            ]
            st.dataframe(pd.DataFrame(guarded_rows), width="stretch", hide_index=True)
            backend_strategy_path = Path("docs/DEEPUplift_AGENT_MODEL_BACKEND_STRATEGY.md")
            if backend_strategy_path.exists():
                backend_strategy = backend_strategy_path.read_text(encoding="utf-8")
                st.download_button(
                    "Download guarded backend strategy",
                    data=backend_strategy.encode("utf-8"),
                    file_name="DEEPUplift_AGENT_MODEL_BACKEND_STRATEGY.md",
                    mime="text/markdown",
                )
        with st.expander("Model Card Governance Matrix", expanded=False):
            governance_df = pd.DataFrame(model_card_governance_rows())
            st.caption("Shows why optional/guarded backends are not silently treated as ready and what evidence is required for promotion.")
            if not governance_df.empty:
                st.dataframe(governance_df, width="stretch", hide_index=True)
                st.download_button(
                    "Download governance matrix CSV",
                    data=dataframe_csv_bytes(governance_df),
                    file_name="deepuplift_model_card_governance.csv",
                    mime="text/csv",
                    key="download_model_card_governance_matrix_models",
                )
            else:
                st.info("No guarded or optional model-card governance rows found.")
        missing_catalog = catalog_view[catalog_view["status"] != "ready"]
        if not missing_catalog.empty:
            st.subheader("Optional Backends")
            st.caption("CausalML 官方 uplift tree/forest 需要 Python 3.11：`scripts/setup_causalml_py311.sh`。XGBoost 已注册但默认实验性禁用：设置 `DEEPUPLIFT_ENABLE_XGBOOST=1` 后再启动 Streamlit。CatBoost 是可选工业后端：安装 `catboost` 后自动变为 ready。UTBoost 是 guarded uplift-GBDT 插件：安装 `utboost` 后先跑 audit 和 no-UI compare。")
            st.dataframe(missing_catalog[["model", "source", "family", "tasks", "missing", "setup_hint", "description"]], width="stretch")
        st.download_button(
            "Download model catalog",
            data=dataframe_csv_bytes(catalog),
            file_name="deepuplift_model_catalog.csv",
            mime="text/csv",
        )

with knowledge_tab:
    st.subheader("Causal Knowledge Base")
    st.subheader("术语速查 / Glossary")
    render_term_glossary("evaluation", expanded=True, key_prefix="knowledge_glossary")
    with st.expander("Latest Regression Health", expanded=True):
        render_regression_health()
    test_guide_path = Path("docs/UPLIFT_AGENT_TEST_GUIDE.md")
    if test_guide_path.exists():
        with st.expander("End-to-End Test Guide", expanded=False):
            guide_text = test_guide_path.read_text(encoding="utf-8")
            st.markdown(guide_text[:12000])
            if len(guide_text) > 12000:
                st.caption("Guide preview truncated in-app; download the full markdown below.")
            st.download_button(
                "Download full test guide",
                data=guide_text.encode("utf-8"),
                file_name="UPLIFT_AGENT_TEST_GUIDE.md",
                mime="text/markdown",
            )
    iteration_log_path = Path("docs/UPLIFT_AGENT_ITERATION_LOG.md")
    if iteration_log_path.exists():
        with st.expander("Recent Optimization Log", expanded=False):
            log_text = iteration_log_path.read_text(encoding="utf-8")
            matches = re.findall(r"(?ms)^## (2026-05-16\.(\d+))\n(.*?)(?=^## |\Z)", log_text)
            sorted_sections = sorted(matches, key=lambda item: int(item[1]))
            latest_sections = sorted_sections[-8:]
            preview = "# Uplift Agent Iteration Log\n\n"
            preview += "\n\n".join(f"## {title}\n{body.strip()}" for title, _, body in latest_sections)
            st.markdown(preview)
            st.download_button(
                "Download full iteration log",
                data=log_text.encode("utf-8"),
                file_name="UPLIFT_AGENT_ITERATION_LOG.md",
                mime="text/markdown",
            )
    kb = knowledge_summary()
    st.caption(f"Knowledge version: {kb.get('version')}")
    st.subheader("Industrial Knowledge Map")
    industry_map = pd.DataFrame(industry_references())
    if industry_map.empty:
        st.info("No industrial references have been registered yet.")
    else:
        left, middle, right = st.columns(3)
        company_filter = left.multiselect(
            "Company",
            sorted(industry_map["company"].dropna().unique().tolist()),
            default=[],
            key="knowledge_industry_company_filter",
        )
        scenario_filter = middle.multiselect(
            "Scenario",
            sorted(industry_map["scenario"].dropna().unique().tolist()),
            default=[],
            key="knowledge_industry_scenario_filter",
        )
        source_filter = right.multiselect(
            "Source quality",
            sorted(industry_map["source_type"].dropna().unique().tolist()),
            default=[],
            key="knowledge_industry_source_filter",
        )
        industry_view = industry_map.copy()
        if company_filter:
            industry_view = industry_view[industry_view["company"].isin(company_filter)]
        if scenario_filter:
            industry_view = industry_view[industry_view["scenario"].isin(scenario_filter)]
        if source_filter:
            industry_view = industry_view[industry_view["source_type"].isin(source_filter)]
        official_count = int(industry_view["source_type"].str.startswith("official", na=False).sum())
        paper_count = int(industry_view["source_type"].str.contains("paper|arxiv|journal|review|working", case=False, na=False).sum())
        render_stat_grid(
            [
                ("References", f"{len(industry_view):,}", False),
                ("Companies", f"{industry_view['company'].nunique():,}", False),
                ("Official Sources", f"{official_count:,}", False),
                ("Paper/Research", f"{paper_count:,}", False),
            ],
            columns=4,
        )
        st.dataframe(
            industry_view[["company", "scenario", "source_type", "title", "lesson", "interview_hook", "url"]],
            width="stretch",
            hide_index=True,
        )
        st.caption("这张表是 Agent 回答工业落地问题、Story 页面案例和业务场景模型推荐的统一证据源。")
    st.subheader("Source Adoption Drilldown")
    backlog_json_path = Path("reports/industry_source_backlog_latest.json")
    if backlog_json_path.exists():
        try:
            backlog_payload = json.loads(backlog_json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            backlog_payload = {}
        backlog_rows = pd.DataFrame(backlog_payload.get("backlog_rows") or [])
        if not backlog_rows.empty:
            status_counts = backlog_rows["status"].value_counts().to_dict()
            render_stat_grid(
                [
                    ("Adopted Sources", f"{int(backlog_payload.get('adopted_sources', 0)):,}", False),
                    ("Adopted Targets", f"{status_counts.get('adopted', 0):,}", False),
                    ("Partial Targets", f"{status_counts.get('partial', 0):,}", False),
                    ("Backlog Targets", f"{status_counts.get('backlog', 0):,}", False),
                ],
                columns=4,
            )
            st.dataframe(backlog_rows, width="stretch", hide_index=True)
            st.caption(
                "这个 drilldown 用来解释为什么 Baidu/Xiaohongshu/Ctrip 等搜索目标没有被直接写入核心知识库：没有足够强的官方或论文级证据就留在 backlog。"
            )
    else:
        st.info("Run `scripts/generate_industry_source_backlog.py` to refresh the source adoption drilldown.")
    render_open_source_framework_audit("knowledge")
    st.subheader("LLM + Causal Frontier Map")
    if industry_map.empty:
        st.info("No LLM frontier references have been registered yet.")
    else:
        llm_mask = (
            industry_map["scenario"].astype(str).str.contains("llm|text|cost_aware", case=False, na=False)
            | industry_map["title"].astype(str).str.contains("LLM|Textual|Structured Treatments", case=False, na=False)
        )
        llm_frontier = industry_map[llm_mask].copy()
        if llm_frontier.empty:
            st.info("No LLM frontier references matched the current playbook.")
        else:
            render_stat_grid(
                [
                    ("LLM Frontier Sources", f"{len(llm_frontier):,}", False),
                    ("Copilot / Agent", f"{int(llm_frontier['scenario'].astype(str).str.contains('copilot|survey', case=False, na=False).sum()):,}", False),
                    ("Routing / Bandit", f"{int(llm_frontier['scenario'].astype(str).str.contains('routing|serving|selection', case=False, na=False).sum()):,}", False),
                    ("Text Treatment", f"{int(llm_frontier['scenario'].astype(str).str.contains('text', case=False, na=False).sum()):,}", False),
                ],
                columns=4,
            )
            st.dataframe(
                llm_frontier[["company", "scenario", "source_type", "title", "lesson", "interview_hook", "url"]],
                width="stretch",
                hide_index=True,
            )
    st.subheader("LLM Interview Q&A")
    st.dataframe(pd.DataFrame(llm_interview_qa()), width="stretch", hide_index=True)
    llm_qa_doc_path = Path("docs/UPLIFT_LLM_INTERVIEW_QA.md")
    if llm_qa_doc_path.exists():
        st.download_button(
            "Download Uplift + LLM Interview Q&A",
            data=llm_qa_doc_path.read_text(encoding="utf-8").encode("utf-8"),
            file_name=llm_qa_doc_path.name,
            mime="text/markdown",
            key="knowledge_llm_interview_qa_download",
        )

    st.subheader("Principles")
    st.dataframe(pd.DataFrame(kb.get("principles", [])), width="stretch")

    st.subheader("Model Availability")
    availability_rows = []
    for model in available_models():
        missing = missing_dependencies(model)
        status, setup_hint = model_readiness(model, missing)
        availability_rows.append(
            {
                "model": model,
                "status": status,
                "missing": ", ".join(missing),
                "setup_hint": setup_hint,
                "description": MODEL_DESCRIPTIONS.get(model, ""),
            }
        )
    st.dataframe(pd.DataFrame(availability_rows), width="stretch")

    st.subheader("Diagnostic Rules")
    rule_rows = []
    for rule in kb.get("diagnostic_rules", []):
        rule_rows.append(
            {
                "id": rule.get("id"),
                "message": rule.get("message"),
                "models": ", ".join(rule.get("models", [])),
                "assumptions": "; ".join(rule.get("assumptions", [])),
                "references": ", ".join(rule.get("references", [])),
            }
        )
    st.dataframe(pd.DataFrame(rule_rows), width="stretch")

    st.subheader("Model Cards")
    model_card_rows = []
    for name, card in kb.get("model_cards", {}).items():
        model_card_rows.append(
            {
                "model": name,
                "family": card.get("family"),
                "best_for": "; ".join(card.get("best_for", [])),
                "risks": "; ".join(card.get("risks", [])),
                "explanation": card.get("explanation"),
            }
        )
    st.dataframe(pd.DataFrame(model_card_rows), width="stretch")

    st.subheader("References")
    references = []
    for ref_id, ref in kb.get("references", {}).items():
        references.append({"id": ref_id, **ref})
    st.dataframe(pd.DataFrame(references), width="stretch")

    st.subheader("Research Frontier")
    frontier = kb.get("research_frontier", [])
    if frontier:
        st.dataframe(pd.DataFrame(frontier), width="stretch")
    else:
        st.info("No research frontier entries yet.")

with story_tab:
    render_project_story()

with evidence_tab:
    render_evidence_dashboard()

with agent_tab:
    result = st.session_state.get("uplift_result")
    st.subheader("Agent Chat")
    render_agent_prompt_group_panel()
    with st.expander("Run Causal Workflow", expanded=False):
        st.write("自动执行：Diagnostics → 模型候选推荐 → Model Compare → 排名解释。")
        workflow_model_count = st.slider("Candidate models", min_value=2, max_value=min(8, len(model_options)), value=min(4, len(model_options)))
        use_candidate_presets = st.checkbox(
            "Use model-specific recommended presets",
            value=True,
            help="Each workflow candidate uses the preset from its model card instead of reusing the currently selected model JSON.",
        )
        apply_manual_overrides = st.checkbox(
            "Apply manual JSON overrides to every candidate",
            value=False,
            help="When enabled, the sidebar Model params JSON overrides each candidate preset.",
        )
        preview_diagnostics = diagnose_uplift_data(df, treatment_col, outcome_col, feature_cols)
        preview_runnable = available_models(model_task, only_available=True)
        preview_knowledge = knowledge_model_recommendations(preview_diagnostics, preview_runnable, limit=workflow_model_count)
        preview_base = recommend_models(df, model_task, treatment_col, feature_cols, selected_dataset)
        preview_candidates = [model for model in preview_knowledge + preview_base if model in preview_runnable]
        preview_candidates = list(dict.fromkeys(preview_candidates))[:workflow_model_count]
        if preview_candidates:
            preview_rows = []
            for candidate in preview_candidates:
                card = build_model_card(candidate, description=MODEL_DESCRIPTIONS.get(candidate, ""))
                recommended_preset, recommended_params = recommended_model_preset(candidate)
                preview_rows.append(
                    {
                        "candidate": candidate,
                        "status": card.get("status"),
                        "family": card.get("family"),
                        "stage": card.get("stage"),
                        "preset": recommended_preset if use_candidate_presets else "Manual JSON",
                        "preset_params": json.dumps(recommended_params if use_candidate_presets else {}, ensure_ascii=False),
                        "best_for": "; ".join(card.get("best_for", [])[:3]),
                    }
                )
            st.caption("Workflow Candidate Preview: " + ", ".join(preview_candidates))
            st.dataframe(pd.DataFrame(preview_rows), width="stretch", hide_index=True)
        if st.button("Run workflow", type="primary"):
            workflow_params = {}
            if apply_manual_overrides or not use_candidate_presets:
                try:
                    workflow_params = json.loads(model_params_text or "{}")
                except json.JSONDecodeError as exc:
                    st.error(f"Invalid model params JSON: {exc}")
                    workflow_params = None

            if workflow_params is not None:
                diagnostics = diagnose_uplift_data(df, treatment_col, outcome_col, feature_cols)
                knowledge_matches = matched_knowledge(diagnostics, model_task)
                runnable_workflow_models = available_models(model_task, only_available=True)
                knowledge_candidates = knowledge_model_recommendations(diagnostics, runnable_workflow_models, limit=workflow_model_count)
                base_candidates = recommend_models(df, model_task, treatment_col, feature_cols, selected_dataset)
                candidates = [model for model in knowledge_candidates + base_candidates if model in runnable_workflow_models]
                candidates = list(dict.fromkeys(candidates))[:workflow_model_count]
                if not candidates:
                    st.error("当前没有可运行的 workflow 候选模型，请先安装依赖或切换任务类型。")
                    st.stop()
                workflow_rows = []
                workflow_results = []
                progress = st.progress(0)
                live = st.empty()
                for idx, candidate in enumerate(candidates, start=1):
                    live.write(f"Workflow training {candidate} ({idx}/{len(candidates)})")
                    candidate_preset = "Manual JSON"
                    candidate_params = dict(workflow_params)
                    if use_candidate_presets:
                        candidate_preset, candidate_params = recommended_model_preset(candidate)
                        if apply_manual_overrides:
                            candidate_params.update(workflow_params)
                    cfg = UpliftConfig(
                        treatment_col=treatment_col,
                        outcome_col=outcome_col,
                        feature_cols=feature_cols,
                        model_name=candidate,
                        task=task,
                        test_size=float(test_size),
                        valid_perc=float(valid_perc) if valid_perc > 0 else None,
                        epochs=int(epochs),
                        batch_size=int(batch_size),
                        learning_rate=float(learning_rate),
                        model_params=candidate_params,
                        max_rows=int(max_preview_rows),
                        bootstrap_samples=int(bootstrap_samples),
                        sensitivity_samples=int(sensitivity_samples),
                        policy_contact_cost=float(contact_cost),
                        policy_conversion_value=float(conversion_value),
                        run_name=f"agent-{candidate.lower()}",
                    )
                    try:
                        workflow_result = train_uplift_model(config=cfg, data_frame=df)
                    except Exception as exc:
                        workflow_rows.append({"model": candidate, "preset": candidate_preset, "error": str(exc)})
                        progress.progress(idx / len(candidates))
                        continue
                    metrics = workflow_result.eval_metrics
                    qini_ci = (((metrics.get("bootstrap") or {}).get("metrics") or {}).get("qini_score") or {})
                    policy_best = metrics.get("policy_best") or {}
                    sensitivity = metrics.get("sensitivity") or {}
                    overlap_trim = metrics.get("overlap_trim") or {}
                    overlap_propensity = overlap_trim.get("propensity") or {}
                    top10 = next((row for row in metrics.get("top_k", []) if abs(row.get("top_fraction", 0) - 0.1) < 1e-6), {})
                    oracle_rows = (metrics.get("oracle_top_k") or {}).get("rows") or []
                    oracle_top10 = next((row for row in oracle_rows if abs(row.get("top_fraction", 0) - 0.1) < 1e-6), {})
                    readiness = decision_readiness(metrics)
                    workflow_rows.append(
                        {
                            "model": candidate,
                            "preset": candidate_preset,
                            "run_id": workflow_result.run_id,
                            "readiness_score": readiness.get("score"),
                            "readiness_level": readiness.get("level"),
                            "qini": metrics.get("qini_score"),
                            "qini_ci_low": qini_ci.get("low"),
                            "auuc": metrics.get("auuc_score"),
                            "calibration_mae": metrics.get("calibration_mae"),
                            "top10_observed_uplift": top10.get("observed_uplift"),
                            "oracle_top10_recall": oracle_top10.get("topk_recall"),
                            "policy_best_fraction": policy_best.get("top_fraction"),
                            "policy_best_net": policy_best.get("observed_net_value", policy_best.get("predicted_net_value")),
                            "sensitivity_verdict": sensitivity.get("verdict"),
                            "weak_overlap_rate": overlap_propensity.get("weak_overlap_rate"),
                        }
                    )
                    workflow_results.append(workflow_result)
                    live.dataframe(pd.DataFrame(workflow_rows), width="stretch")
                    progress.progress(idx / len(candidates))

                ranking = pd.DataFrame(workflow_rows)
                if not ranking.empty:
                    if "qini_ci_low" in ranking.columns and ranking["qini_ci_low"].notna().any():
                        ranking = ranking.sort_values("qini_ci_low", ascending=False, na_position="last")
                    elif "readiness_score" in ranking.columns and ranking["readiness_score"].notna().any():
                        ranking = ranking.sort_values(["readiness_score", "qini"], ascending=[False, False], na_position="last")
                    elif "qini" in ranking.columns:
                        ranking = ranking.sort_values("qini", ascending=False, na_position="last")
                best_model = ranking.iloc[0]["model"] if not ranking.empty and "model" in ranking.columns else None
                report_lines = [
                    f"数据设计：treatment `{diagnostics['design']['treatment_type']}`，outcome `{diagnostics['design']['outcome_type']}`。",
                    f"候选模型：{', '.join(candidates)}。",
                    "参数策略：按候选模型卡推荐预设训练。" if use_candidate_presets else "参数策略：使用手工 JSON 训练所有候选模型。",
                    f"推荐模型：`{best_model}`。" if best_model else "暂无可推荐模型。",
                ]
                if apply_manual_overrides:
                    report_lines.append("手工 JSON 已作为覆盖参数应用到所有候选模型。")
                if knowledge_matches:
                    report_lines.append("知识库依据：\n" + format_rule_evidence(knowledge_matches))
                if best_model:
                    report_lines.append("推荐模型卡片：\n" + format_model_card(str(best_model)))
                if diagnostics.get("warnings"):
                    report_lines.append("诊断风险：" + "；".join(diagnostics["warnings"]))
                if workflow_results and best_model:
                    best_result = next((item for item in workflow_results if item.config.get("model_name") == best_model), workflow_results[0])
                    report_path = Path(best_result.run_dir) / "agent_workflow_report.md"
                    save_text(report_path, "\n\n".join(report_lines))
                    report_lines.append(f"工作流报告已保存：`{report_path}`")
                if not ranking.empty:
                    dataset_label = selected_dataset["name"] if selected_dataset else source
                    compare_report_path = save_compare_report(
                        ranking,
                        dataset_label,
                        diagnostics,
                        candidates,
                        prefix="agent_compare",
                    )
                    workflow_readiness_path = save_compare_readiness_json(
                        ranking,
                        dataset_label,
                        diagnostics,
                        candidates,
                        prefix="agent_compare",
                    )
                    workflow_bundle_path = save_compare_bundle_manifest(
                        ranking,
                        dataset_label,
                        compare_report_path,
                        workflow_readiness_path,
                        workflow_results,
                        prefix="agent_compare",
                    )
                    st.session_state["compare_report_path"] = compare_report_path
                    st.session_state["agent_workflow_readiness_path"] = workflow_readiness_path
                    st.session_state["agent_workflow_bundle_path"] = workflow_bundle_path
                st.session_state["compare_ranking"] = ranking
                st.session_state["agent_workflow_report"] = "\n\n".join(report_lines)
                st.success("Workflow finished.")

        workflow_report = st.session_state.get("agent_workflow_report")
        if workflow_report:
            st.markdown(workflow_report)
            ranking = st.session_state.get("compare_ranking")
            if ranking is not None and not ranking.empty:
                st.dataframe(ranking, width="stretch")
                compare_report_path = st.session_state.get("compare_report_path")
                if compare_report_path and Path(compare_report_path).exists():
                    st.download_button(
                        "Download workflow compare report",
                        data=Path(compare_report_path).read_bytes(),
                        file_name=Path(compare_report_path).name,
                        mime="text/markdown",
                    )
                workflow_readiness_path = st.session_state.get("agent_workflow_readiness_path")
                if workflow_readiness_path and Path(workflow_readiness_path).exists():
                    st.download_button(
                        "Download workflow readiness JSON",
                        data=Path(workflow_readiness_path).read_bytes(),
                        file_name=Path(workflow_readiness_path).name,
                        mime="application/json",
                    )
                workflow_bundle_path = st.session_state.get("agent_workflow_bundle_path")
                if workflow_bundle_path and Path(workflow_bundle_path).exists():
                    st.download_button(
                        "Download workflow bundle manifest",
                        data=Path(workflow_bundle_path).read_bytes(),
                        file_name=Path(workflow_bundle_path).name,
                        mime="application/json",
                    )

    agent_context_key = f"{dataset_key}_{model_name}_{outcome_col}_{treatment_col}"
    if st.session_state.get("agent_context_key") != agent_context_key:
        st.session_state["agent_context_key"] = agent_context_key
        st.session_state["agent_messages"] = [
            {
                "role": "assistant",
                "content": "我已经读取当前页面配置。你可以问我：推荐哪个模型、解释当前模型卡、诊断数据、模型对比、训练参数、投放阈值，也可以直接问面试难点，比如为什么不用普通转化率模型、观测数据有偏怎么办。",
            }
        ]

    quick_prompts = [
        "推荐哪个模型？",
        "诊断一下数据？",
        "解释当前模型卡？",
        "对比结果怎么看？",
        "训练参数怎么设？",
        "投放阈值怎么解释？",
        "现在有哪些业务场景模拟数据和训练闭环？",
        "5分钟怎么演示？",
        *INTERVIEW_AGENT_QUICK_PROMPTS,
        "发券为什么不能只看转化率？",
        "广告投放里 uplift 和 pCTR/pCVR 有什么区别？",
        "怎么把 uplift score 转成投放阈值和 ROI？",
        "发券如何处理补贴套利和羊毛党？",
        "预算有限时 uplift 阈值怎么定？",
        "Marketplace 干扰和 spillover 怎么处理？",
        "Push 频控和用户疲劳怎么建模？",
        "Spotify 站内消息为什么要做 uplift？",
        "DoorDash ghost ads 怎么衡量广告增量？",
        "Airbnb incremental LTV 怎么接到补贴 ROI？",
        "阿里 X-Learner AB 实验异质性怎么讲？",
        "EdgeRec 请求价值增益模型怎么讲？",
        "多个 CRM journey 重叠时 uplift 怎么评估？",
        "为什么不要只追最新 uplift 模型？",
        "为什么 S-Learner + LightGBM 有时会打败复杂模型？",
        "UTBoost 怎么接到这个框架？",
        "EFIN、DESCN、UTBoost 分别适合什么？",
        "ECUP 全链路 uplift 怎么评估？",
        "delayed feedback uplift 怎么做？",
        "发券、广告、LLM routing 业务场景的模型路径怎么选？",
        "为什么 QINI/AUUC 不等于增量利润？",
        "uplift deep loss 怎么设计，为什么不能只用 BCE？",
        "从源码拆解 TLearnerGBM 的 fit 和 predict 调用链",
        "DRLearnerGBM 的 pseudo outcome 公式和源码在哪里？",
        "TARNet、CFRNet、DragonNet 网络结构怎么讲？",
        "CFRNet 的 loss 怎么从 TARNet 推出来？",
        "DragonNet 的 propensity head 和 targeted regularization 源码怎么讲？",
        "EFIN 官方代码 license 风险和 DeepUplift 借鉴策略是什么？",
        "DESCN/ESX 的 propensity、mu0、mu1、tau head 怎么协同？",
        "对比学习怎么用于 uplift？",
        "LLM routing loss 和 cost-quality objective 怎么做？",
        "Uplift 和 LLM 怎么结合？",
        "LLM routing 为什么是 uplift 问题？",
        "LLM 生成文案怎么做 uplift 实验？",
        "当前平台用了哪些开源框架，哪些 ready/guarded？",
        "uplift 模型到底怎么评估？公式怎么讲？",
        "QINI / AUUC / uplift@K 有什么区别？",
        "DR learner / R learner 的目标函数怎么讲？",
        "从0-1优化 uplift 项目怎么讲？",
        "参考了哪些 GitHub 前端框架？",
        "为什么当前阶段还是 Streamlit，而不是直接 React？",
        "哪些设计模式被吸收到当前页面？",
        "哪些适合未来 FastAPI + React 版本？",
        "如何设计工业级 uplift workbench 的信息架构？",
        "可视化如何服务模型评估和业务决策？",
        "面试中怎么讲 UI/UX 和算法平台结合？",
        "参考了哪些开源模型训练平台？",
        "哪些训练平台设计被吸收到当前页面？",
        "DeepUplift 未来 FastAPI + React 怎么演进？",
        "如何设计工业级 uplift training workbench？",
        "参考了哪些字节/火山引擎公开数据产品？",
        "哪些公开产品设计被吸收到当前页面？",
        "为什么只参考公开资料，不复制私有 UI？",
        "DeepUplift 怎么像增长分析和 A/B 实验平台？",
    ]
    quick_prompts = list(dict.fromkeys(quick_prompts))

    def submit_quick_prompt(quick_prompt: str) -> None:
        st.session_state["agent_messages"].append({"role": "user", "content": quick_prompt})
        st.session_state["agent_messages"].append(
            {
                "role": "assistant",
                "content": agent_reply(
                    quick_prompt,
                    df,
                    model_task,
                    treatment_col,
                    outcome_col,
                    feature_cols,
                    model_name,
                    result,
                    selected_dataset,
                    st.session_state.get("compare_ranking"),
                    st.session_state.get("policy_context"),
                ),
            }
        )

    grouped_prompts: list[str] = []
    model_deconstruction_prompt_group = {
        "group": "模型源码拆解",
        "intent": "公式、loss、forward、fit、predict、训练评估和 license gate",
        "prompts": [
            "从源码拆解 TLearnerGBM 的 fit 和 predict 调用链",
            "DRLearnerGBM 的 pseudo outcome 公式和源码在哪里？",
            "CFRNet 的 differentiable balance smoke 证明了什么？",
            "DragonNet 的训练证据和 failure mode 怎么讲给面试官？",
            "EFIN 的 treatment-aware interaction attention 源码怎么讲？",
            "DESCN/ESX 的 propensity、mu0、mu1、tau head 怎么协同？",
            "MultiDRLearnerGBM 如何把二元 uplift 扩展成多动作推荐？",
        ],
    }
    agent_groups = [model_deconstruction_prompt_group] + agent_prompt_groups() + ml_training_prompt_groups() + public_product_prompt_groups()
    for group_idx, prompt_group in enumerate(agent_groups):
        prompts = [str(prompt) for prompt in (prompt_group.get("prompts") or [])]
        grouped_prompts.extend(prompts)
        with st.expander(f"{prompt_group.get('group')} · {prompt_group.get('intent')}", expanded=group_idx == 0):
            for start in range(0, len(prompts), 4):
                quick_cols = st.columns(len(prompts[start : start + 4]))
                for prompt_idx, (col, quick_prompt) in enumerate(zip(quick_cols, prompts[start : start + 4])):
                    if col.button(quick_prompt, key=f"agent_group_prompt_{group_idx}_{start}_{prompt_idx}"):
                        submit_quick_prompt(quick_prompt)

    extra_prompts = [prompt for prompt in quick_prompts if prompt not in set(grouped_prompts)]
    with st.expander("更多行业 / 算法 / 证据追问", expanded=False):
        for start in range(0, len(extra_prompts), 4):
            quick_cols = st.columns(len(extra_prompts[start : start + 4]))
            for prompt_idx, (col, quick_prompt) in enumerate(zip(quick_cols, extra_prompts[start : start + 4])):
                if col.button(quick_prompt, key=f"agent_extra_prompt_{start}_{prompt_idx}"):
                    submit_quick_prompt(quick_prompt)

    for message in st.session_state["agent_messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_agent_evidence_card(message["content"])

    transcript_dataset_label = selected_dataset["name"] if selected_dataset else source
    agent_transcript = format_agent_transcript(
        st.session_state["agent_messages"],
        dataset_label=transcript_dataset_label,
        model_name=model_name,
        task=model_task,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        feature_cols=feature_cols,
    )
    st.download_button(
        "Download Agent transcript",
        data=agent_transcript.encode("utf-8"),
        file_name=f"deepuplift_agent_transcript_{time.strftime('%Y%m%d-%H%M%S')}.md",
        mime="text/markdown",
        help="Export the current Agent Q&A as interview/demo evidence.",
    )

    prompt = st.chat_input("问我模型选择、数据诊断、训练参数、结果解释或面试难点")
    if prompt:
        st.session_state["agent_messages"].append({"role": "user", "content": prompt})
        reply = agent_reply(
            prompt,
            df,
            model_task,
            treatment_col,
            outcome_col,
            feature_cols,
            model_name,
            result,
            selected_dataset,
            st.session_state.get("compare_ranking"),
            st.session_state.get("policy_context"),
        )
        st.session_state["agent_messages"].append({"role": "assistant", "content": reply})
        st.rerun()

    st.subheader("Recommendations")
    if result is None:
        treatment_rate = df[treatment_col].value_counts(normalize=True, dropna=True).to_dict()
        ready_models = available_models(inferred_task, only_available=True)
        recommended_now = recommend_models(df, inferred_task, treatment_col, feature_cols, selected_dataset)[:8]
        render_stat_grid(
            [
                ("Detected Task", inferred_task, True),
                ("Ready Models", f"{len(ready_models):,}", False),
                ("Treatment Mix", str(treatment_rate), True),
                ("Recommended", ", ".join(recommended_now[:4]), True),
            ]
        )
        source_summary, family_summary = model_capability_summary(inferred_task)
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Model backends by status")
            st.dataframe(source_summary, width="stretch", hide_index=True)
        with c2:
            st.caption("Ready model families")
            st.dataframe(family_summary, width="stretch", hide_index=True)
        with st.expander("Runnable model list", expanded=False):
            st.write(", ".join(ready_models))
    else:
        for item in result.recommendations:
            st.write(f"- {item}")
        st.write(f"Artifacts: `{result.run_dir}`")

    st.subheader("Next Features")
    st.write("- Multi-treatment: add DR/Forest official adapters and action-specific confidence intervals.")
    st.write("- Continuous treatment: estimate dose-response curves and optimal intensity.")
    st.write("- Sensitivity analysis: add hidden confounding bounds and DoWhy-style refuters.")
