from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"


GITHUB_REPOS: dict[str, str] = {
    "EconML": "py-why/EconML",
    "CausalML": "uber/causalml",
    "scikit-uplift": "maks-sh/scikit-uplift",
    "UTBoost": "jd-opensource/UTBoost",
    "pylift": "Wayfair/pylift",
    "grf": "grf-labs/grf",
    "DoWhy / PyWhy": "py-why/dowhy",
    "DoubleML": "DoubleML/doubleml-for-py",
    "YLearn": "DataCanvasIO/YLearn",
    "causallib": "IBM/causallib",
    "CausalTune": "py-why/causaltune",
    "GeoLift": "facebookincubator/GeoLift",
    "UpliftML": "bookingcom/upliftml",
    "CausalLift": "Minyus/causallift",
    "causal-learn": "py-why/causal-learn",
    "LightGBM": "microsoft/LightGBM",
    "XGBoost / CatBoost": "dmlc/xgboost; catboost/catboost",
    "RouteLLM": "lm-sys/RouteLLM",
    "Open Bandit Pipeline": "st-tech/zr-obp",
    "Vowpal Wabbit": "VowpalWabbit/vowpal_wabbit",
    "LangGraph Open Deep Research": "langchain-ai/open_deep_research",
    "GPT Researcher": "assafelovic/gpt-researcher",
    "Local Deep Researcher": "langchain-ai/local-deep-researcher",
    "Deep uplift paper code family": "local/DeepUplift",
}


OPEN_SOURCE_FRAMEWORKS: list[dict[str, str]] = [
    {
        "framework": "EconML",
        "owner": "PyWhy / Microsoft",
        "source_type": "official_docs",
        "url": "https://www.pywhy.org/EconML/",
        "supported_models": "DML, LinearDML, CausalForestDML, DRLearner, ForestDRLearner, OrthoForest, meta-learners, inference APIs",
        "treatment_support": "binary / categorical / continuous depending on estimator",
        "scenario_fit": "观测数据、异质性解释、广告/发券/补贴里的稳健 CATE baseline",
        "local_status": "ready",
        "integration_path": "deepuplift/core/external_models.py; deepuplift/core/registry.py",
        "ui_evidence": "Models -> Open Source Framework Audit; Models -> Industrial Model Layer Architecture",
        "limitations": "需要 overlap 和 nuisance model 质量；forest inference 不能替代实验设计",
        "next_step": "继续把 EconML policy interpreter / DRTester 思路接到 Evaluation 和 Evidence",
        "interview_line": "EconML 是 P0 稳健因果层，适合讲 DML/DR/causal forest 和置信区间。",
    },
    {
        "framework": "CausalML",
        "owner": "Uber",
        "source_type": "official_open_source",
        "url": "https://github.com/uber/causalml",
        "supported_models": "S/T/X/R learners, uplift tree, uplift random forest, causal random forest, metrics and examples",
        "treatment_support": "binary / multi-treatment classification-oriented uplift",
        "scenario_fit": "营销分群、规则解释、传统 uplift tree/forest、P0/P1 对比后端",
        "local_status": "optional_py311",
        "integration_path": "scripts/setup_causalml_py311.sh; deepuplift/core/external_models.py guarded adapters",
        "ui_evidence": "Models -> Guarded Backend Runtime Guide; Evidence -> optional backend probe",
        "limitations": "当前主 conda 环境不默认启用，Python/依赖组合更适合独立 helper env",
        "next_step": "在 helper env 跑 uplift tree/forest no-UI smoke 后再晋级 demo-ready",
        "interview_line": "CausalML 很适合说明为什么可解释 uplift tree/forest 要 guarded 接入。",
    },
    {
        "framework": "scikit-uplift",
        "owner": "scikit-uplift community",
        "source_type": "official_docs",
        "url": "https://www.uplift-modeling.com/",
        "supported_models": "SoloModel, TwoModels, ClassTransformation, metrics such as uplift curve and qini curve",
        "treatment_support": "binary treatment",
        "scenario_fit": "快速 baseline、教育式演示、QINI/AUUC 指标校验",
        "local_status": "ready",
        "integration_path": "deepuplift/core/external_models.py; deepuplift/core/registry.py",
        "ui_evidence": "Models -> Ready Backends; Evaluation -> metric map",
        "limitations": "模型覆盖偏基础；复杂观测偏差和多 treatment 需要 DR/DML/自定义扩展",
        "next_step": "继续把 scikit-uplift metric 对照加入 regression gate",
        "interview_line": "scikit-uplift 是轻量基线层，用来保证指标和模型接口先跑稳。",
    },
    {
        "framework": "UTBoost",
        "owner": "JD open source",
        "source_type": "official_open_source / paper",
        "url": "https://github.com/jd-opensource/UTBoost",
        "supported_models": "uplift gradient boosted decision trees with treatment-effect split criteria",
        "treatment_support": "binary / multi-treatment uplift",
        "scenario_fit": "大规模表格发券、CRM、广告人群排序、工业 GBDT uplift 插件",
        "local_status": "guarded",
        "integration_path": "deepuplift/core/external_models.py:_fit_utboost; scripts/smoke_utboost_guarded_adapter.py",
        "ui_evidence": "Models -> Frontier Plugin Backlog; Evidence -> UTBoost guarded adapter smoke",
        "limitations": "依赖未默认安装；需要和 LightGBM DR/R baseline 做 no-UI compare 后才能默认启用",
        "next_step": "安装 utboost 后跑 tiny train + scenario compare，再补 feature importance evidence",
        "interview_line": "UTBoost 代表工业 uplift GBDT，但平台没有把未验证依赖伪装成 ready。",
    },
    {
        "framework": "pylift",
        "owner": "Wayfair / maintained fork",
        "source_type": "official_docs / company_blog",
        "url": "https://docs.pylift.org/",
        "supported_models": "TransformedOutcome, UpliftEval, uplift evaluation and visualization helpers",
        "treatment_support": "binary treatment",
        "scenario_fit": "display remarketing、营销 uplift 评估、早期 uplift baseline 复现",
        "local_status": "backlog",
        "integration_path": "planned: deepuplift/core/frontier_plugins/pylift_adapter.py",
        "ui_evidence": "Models -> Open Source Framework Audit",
        "limitations": "生态相对早期；价值更多在评估/历史方法对照，不适合作为默认训练后端",
        "next_step": "把 UpliftEval 思路转成本地 metric cross-check，而不是直接依赖老 package",
        "interview_line": "pylift 用来讲 uplift 评估演进和为什么现代平台要统一自有 metrics。",
    },
    {
        "framework": "grf",
        "owner": "GRF Labs",
        "source_type": "official_docs / R_package",
        "url": "https://grf-labs.github.io/grf/",
        "supported_models": "causal_forest, regression_forest, instrumental_forest, survival_forest, local linear forest",
        "treatment_support": "binary / continuous treatment in R ecosystem",
        "scenario_fit": "需要 honest forest、置信区间、异质性解释的因果森林基准",
        "local_status": "optional_r_bridge",
        "integration_path": "planned: R bridge or offline benchmark import",
        "ui_evidence": "Models -> Open Source Framework Audit; Knowledge -> framework audit",
        "limitations": "R 包，不应强塞进 Python Streamlit 主训练链路；可通过离线 benchmark 或 rpy2 guarded 接入",
        "next_step": "先导入 grf benchmark artifacts，再决定是否做 R bridge",
        "interview_line": "grf 是 causal forest 重要参考，实现上用 optional bridge 控制工程风险。",
    },
    {
        "framework": "DoWhy / PyWhy",
        "owner": "PyWhy",
        "source_type": "official_docs",
        "url": "https://www.pywhy.org/dowhy/",
        "supported_models": "causal graph, identification, estimation wrappers, refuters, placebo/data-subset/bootstrap sensitivity",
        "treatment_support": "design/identification layer; not primarily an uplift trainer",
        "scenario_fit": "因果假设声明、backdoor/adjustment、placebo/refuter、Evidence/可信度页面",
        "local_status": "optional_diagnostic",
        "integration_path": "planned: diagnostics/refuter adapter; current local sensitivity in deepuplift/core/evaluator.py",
        "ui_evidence": "Evaluation -> Trust Gates; Evidence -> regression and sensitivity artifacts",
        "limitations": "它解决因果识别和 refutation，不直接替代 uplift ranking 模型",
        "next_step": "把 placebo treatment/data subset refuter 做成 guarded Evidence check",
        "interview_line": "DoWhy 放在诊断和 refutation 层，说明平台不是只训练模型，而是管理因果假设。",
    },
    {
        "framework": "DoubleML",
        "owner": "DoubleML community",
        "source_type": "official_docs",
        "url": "https://docs.doubleml.org/",
        "supported_models": "PLR, IRM, IIVM, DID and double/debiased machine learning workflows",
        "treatment_support": "binary / continuous depending on causal model",
        "scenario_fit": "正交化估计、DML 教学、政策效应和观测数据稳健性",
        "local_status": "optional",
        "integration_path": "planned: deepuplift/core/frontier_plugins/doubleml_adapter.py",
        "ui_evidence": "Models -> Open Source Framework Audit; Evaluation -> DR/R/DML formula cards",
        "limitations": "更偏 ATE/score-based causal models；个体 uplift ranking 仍需 adapter 层统一输出",
        "next_step": "优先接 IRM/PLR smoke，并导出 pseudo-outcome / score diagnostics",
        "interview_line": "DoubleML 适合讲 Neyman orthogonality，作为 DR/R learner 的理论补强。",
    },
    {
        "framework": "YLearn",
        "owner": "DataCanvas / community",
        "source_type": "official_open_source",
        "url": "https://github.com/DataCanvasIO/YLearn",
        "supported_models": "causal estimators, meta-learners, doubly robust estimators, policy interpretation and what-if tools",
        "treatment_support": "binary / multi-treatment / continuous depending on estimator",
        "scenario_fit": "中文生态可讲的因果推断工具，适合补充 treatment effect、policy interpretation 和业务 what-if",
        "local_status": "optional",
        "integration_path": "planned: deepuplift/core/frontier_plugins/ylearn_adapter.py",
        "ui_evidence": "Models -> Open Source Framework Audit; Knowledge -> optional framework backlog",
        "limitations": "与当前 registry 的 y0/y1/uplift 输出还需要 adapter；先作为可选中文生态参考",
        "next_step": "做 estimator import probe 和 tiny CATE smoke，再判断是否晋级 guarded",
        "interview_line": "YLearn 可以说明我不只看海外库，也审计中文生态的因果工具，但上线前仍需统一 adapter 和 evidence gate。",
    },
    {
        "framework": "causallib",
        "owner": "IBM Research",
        "source_type": "official_open_source",
        "url": "https://github.com/IBM/causallib",
        "supported_models": "standardization, IPW, doubly robust estimation, matching and evaluation utilities",
        "treatment_support": "binary / multi-treatment causal effect estimation workflows",
        "scenario_fit": "propensity/IPW、standardization、DR 估计和 overlap 诊断的理论/实现补充",
        "local_status": "optional_diagnostic",
        "integration_path": "planned: deepuplift/core/frontier_plugins/causallib_adapter.py",
        "ui_evidence": "Evaluation -> overlap/IPW formula cards; Models -> framework audit",
        "limitations": "更偏 effect estimation 和诊断，不直接输出工业 Top-K uplift policy",
        "next_step": "把 IPW/standardization diagnostic 做成 optional cross-check",
        "interview_line": "causallib 放在 propensity weighting 和 DR 诊断层，用于验证平台评估公式不是凭空写的。",
    },
    {
        "framework": "CausalTune",
        "owner": "Wise / community",
        "source_type": "official_open_source",
        "url": "https://github.com/py-why/causaltune",
        "supported_models": "AutoML-style causal estimator selection across EconML/DoWhy style estimators",
        "treatment_support": "binary / continuous depending on wrapped estimator",
        "scenario_fit": "模型选择自动化、nuisance 模型调参、候选 estimator 搜索",
        "local_status": "optional_automl",
        "integration_path": "planned: deepuplift/core/frontier_plugins/causaltune_adapter.py",
        "ui_evidence": "Models -> advanced framework audit; Agent -> model selection Q&A",
        "limitations": "自动化调参可能较重；需要和当前 deterministic demo preset 分离",
        "next_step": "先做离线 candidate recommendation，不默认进入 Streamlit 训练按钮",
        "interview_line": "CausalTune 适合讲模型选择自动化，但工业 demo 要控制可复现和运行时。",
    },
    {
        "framework": "GeoLift",
        "owner": "Meta Open Source",
        "source_type": "official_open_source",
        "url": "https://github.com/facebookincubator/GeoLift",
        "supported_models": "geo experiment design, power analysis, synthetic controls and incrementality measurement",
        "treatment_support": "geo/time-level binary or staggered treatment experiments",
        "scenario_fit": "广告、补贴、城市预算的 geo holdout / switchback / 增量实验验证",
        "local_status": "optional_experiment_design",
        "integration_path": "planned: deepuplift/core/frontier_plugins/geolift_artifact_import.py",
        "ui_evidence": "Evaluation -> online holdout cards; Evidence -> geo holdout scenario",
        "limitations": "R 生态和实验设计工具，不应伪装成个体 uplift trainer",
        "next_step": "先在 synthetic_geo_holdout_incrementality 数据集里展示 geo-level policy/evaluation，再考虑 R bridge",
        "interview_line": "GeoLift 用来讲离线模型如何接线上 geo incrementality，而不是只看 AUUC。",
    },
    {
        "framework": "UpliftML",
        "owner": "Booking.com",
        "source_type": "official_open_source",
        "url": "https://github.com/bookingcom/upliftml",
        "supported_models": "Spark-based uplift modeling pipelines, transformed outcome and large-scale marketing workflows",
        "treatment_support": "binary treatment in distributed Spark setting",
        "scenario_fit": "大规模营销/旅行场景、Spark 离线批处理、企业级 pipeline 参考",
        "local_status": "optional_spark_backlog",
        "integration_path": "planned: evidence-only source card or Spark artifact import, not local Streamlit training",
        "ui_evidence": "Models -> framework audit; Knowledge -> source backlog",
        "limitations": "Spark runtime 不适合默认本地 demo；价值在工业 pipeline 设计和证据参考",
        "next_step": "抽象其 pipeline 思路到 Evidence bundle 和 batch scoring playbook",
        "interview_line": "UpliftML 说明工业大规模 uplift 经常是批处理 pipeline，而本项目用轻量本地 demo 复现核心闭环。",
    },
    {
        "framework": "CausalLift",
        "owner": "community",
        "source_type": "official_open_source",
        "url": "https://github.com/Minyus/causallift",
        "supported_models": "uplift modeling workflow around propensity, transformed outcome and model evaluation",
        "treatment_support": "binary treatment",
        "scenario_fit": "早期 uplift workflow 参考、propensity-aware uplift 教学和评估对照",
        "local_status": "backlog",
        "integration_path": "planned: source card only unless package/runtime is validated",
        "ui_evidence": "Models -> framework audit; Knowledge -> backlog",
        "limitations": "生态活跃度和依赖状态需要单独审计；暂不作为默认训练后端",
        "next_step": "只吸收 workflow/evaluation 思路，避免引入不稳定 runtime",
        "interview_line": "CausalLift 放在 backlog，体现我会区分可讲参考和可运行依赖。",
    },
    {
        "framework": "causal-learn",
        "owner": "Causal AI Lab / community",
        "source_type": "official_open_source",
        "url": "https://github.com/py-why/causal-learn",
        "supported_models": "causal discovery algorithms such as PC, GES, FCI and independence tests",
        "treatment_support": "graph discovery / structure learning, not uplift training",
        "scenario_fit": "特征泄漏、潜在混杂、因果图假设探索和 Evidence 中的诊断补充",
        "local_status": "optional_discovery",
        "integration_path": "planned: graph discovery diagnostic card, separated from training registry",
        "ui_evidence": "Evaluation -> leakage/confounding diagnostics; Knowledge -> framework audit",
        "limitations": "发现因果图不等于识别 treatment effect；只能作为假设探索和风险提示",
        "next_step": "补一个 synthetic leakage/confounding diagnostic smoke，而不是放进模型菜单",
        "interview_line": "causal-learn 适合讲因果图和混杂诊断，但我不会把 discovery 当作 uplift estimator。",
    },
    {
        "framework": "LightGBM",
        "owner": "Microsoft / community",
        "source_type": "official_open_source",
        "url": "https://lightgbm.readthedocs.io/",
        "supported_models": "GBDT base learner for S/T/X/R/DR learners, calibrated variants, policy ranking baseline",
        "treatment_support": "adapter-defined binary / multi / dose-response baselines",
        "scenario_fit": "工业大表强 baseline，尤其发券、广告、CRM 的第一轮 compare",
        "local_status": "ready",
        "integration_path": "deepuplift/core/sklearn_models.py; deepuplift/core/registry.py",
        "ui_evidence": "Models -> Ready Backends; Compare -> P0 baseline ranking",
        "limitations": "本身不是 causal estimator；必须通过 learner/adapter 和因果评估包装",
        "next_step": "继续补 CatBoost/XGBoost backend parity 和 scenario-specific presets",
        "interview_line": "LightGBM baseline 强，是工业落地里不盲目追复杂深度模型的关键证据。",
    },
    {
        "framework": "XGBoost / CatBoost",
        "owner": "XGBoost / Yandex community",
        "source_type": "official_open_source",
        "url": "https://catboost.ai/",
        "supported_models": "boosting base learners for S/T/X/R/DR adapters and categorical-heavy tables",
        "treatment_support": "adapter-defined binary treatment; future multi-treatment through wrappers",
        "scenario_fit": "广告、电商、CRM 的高维类别特征和稳定表格 baseline",
        "local_status": "guarded_optional",
        "integration_path": "deepuplift/core/sklearn_models.py guarded model registry",
        "ui_evidence": "Models -> Guarded Backend Runtime Guide",
        "limitations": "本地 XGBoost 曾有运行时不稳定；CatBoost 是重依赖，不默认安装",
        "next_step": "单独环境 smoke 后再进入默认 Compare 候选",
        "interview_line": "工业 boosting 后端要治理依赖和稳定性，不能只因为常用就默认打开。",
    },
    {
        "framework": "RouteLLM",
        "owner": "LMSYS / UC Berkeley ecosystem",
        "source_type": "official_open_source",
        "url": "https://github.com/lm-sys/RouteLLM",
        "supported_models": "LLM router serving/evaluation, weak-vs-strong model routing, cost-quality benchmarking",
        "treatment_support": "multi-action LLM routing actions through policy adapter",
        "scenario_fit": "LLM routing、RAG/tool/人工审核升级、质量-成本策略、budget pacing",
        "local_status": "optional_llm_routing",
        "integration_path": "deepuplift/core/policy.py; examples/datasets/synthetic_llm_multi_action_routing_9k.csv; planned routellm adapter",
        "ui_evidence": "Policy -> LLM 多动作 Routing OPE Mini-Lab; 场景工作台 -> LLM 多动作 Routing / OPE 场景证据",
        "limitations": "真实线上接入需要 router logs、judge calibration、action propensity 和服务成本；不应默认依赖其 server runtime",
        "next_step": "做 route-log import schema，把 RouteLLM benchmark 输出转成 DeepUplift OPE/action coverage evidence",
        "interview_line": "RouteLLM 给了 LLM routing 的服务化参考；我把它抽象成 uplift 多动作 treatment 和 OPE/policy value，而不是只接一个 router API。",
    },
    {
        "framework": "Open Bandit Pipeline",
        "owner": "ZOZO / st-tech",
        "source_type": "official_open_source / paper",
        "url": "https://github.com/st-tech/zr-obp",
        "supported_models": "logged bandit data, offline policy learning, DM/IPW/SNIPW/DR/MRDR/DRos, slate and continuous-action OPE",
        "treatment_support": "discrete / slate / continuous actions in logged bandit settings",
        "scenario_fit": "LLM routing OPE、广告/推荐 policy evaluation、budgeted targeting、contextual bandit 升级前评估",
        "local_status": "guarded_ope_plugin",
        "integration_path": "deepuplift/core/policy.py OPE mini-lab; planned obp import adapter",
        "ui_evidence": "Policy -> LLM 多动作 Routing OPE Mini-Lab; Evidence -> LLM Routing OPE Smoke",
        "limitations": "依赖和接口会把 uplift 数据转成 logged bandit schema；默认平台先保留本地 OPE calculator",
        "next_step": "新增 OBP optional probe，验证 SyntheticBanditDataset + DR OPE 后再考虑 adapter",
        "interview_line": "OBP 是 OPE/bandit 的高质量参考，启发我把 IPS/SNIPS/DR、coverage 和 effective sample size 作为上线前证据。",
    },
    {
        "framework": "Vowpal Wabbit",
        "owner": "VowpalWabbit community",
        "source_type": "official_open_source",
        "url": "https://github.com/VowpalWabbit/vowpal_wabbit",
        "supported_models": "online learning, contextual bandit algorithms, exploration, reductions and large-scale serving-oriented learners",
        "treatment_support": "contextual bandit actions; not a direct uplift y0/y1 trainer",
        "scenario_fit": "线上 bandit、预算节奏、广告/推荐/LLM routing 的自适应策略层",
        "local_status": "optional_online_learning",
        "integration_path": "planned: vw log/export adapter after fixed uplift policy and OPE gates are stable",
        "ui_evidence": "Policy -> 从离线 uplift 到在线 bandit 的上线门禁",
        "limitations": "服务化/在线学习系统较重；没有 randomized logging 和 guardrail 时不应直接上线",
        "next_step": "先导出 DeepUplift policy logs 到 VW-friendly action/reward/propensity schema",
        "interview_line": "VW 适合讲 uplift 到 contextual bandit 的生产演进，但必须排在随机探索、OPE 和 guardrail 之后。",
    },
    {
        "framework": "LangGraph Open Deep Research",
        "owner": "LangChain",
        "source_type": "official_open_source",
        "url": "https://github.com/langchain-ai/open_deep_research",
        "supported_models": "plan-search-summarize-report research agent, search provider/MCP configuration, evaluation with Deep Research Bench",
        "treatment_support": "research/evidence workflow; not uplift training",
        "scenario_fit": "source-backed paper/industry research ingestion, claim-to-artifact gate, Agent evidence card",
        "local_status": "optional_research_workflow",
        "integration_path": "docs/DEEPUplift_RESEARCH_EVIDENCE_GATE.md; deepuplift/core/resume_scenario_workbench.py:research_evidence_gate_rows_cn",
        "ui_evidence": "Evidence -> Research Evidence Gate; 场景工作台 -> Research Evidence Gate",
        "limitations": "研究代理输出不能直接进入模型目录；必须经过 source quality、UI proof 和 smoke proof",
        "next_step": "用其 plan/research/report 思路做本地 paper/source adoption backlog，而不是替代 uplift evaluator",
        "interview_line": "Open Deep Research 给了 source-backed agent 工作流启发；我把它约束成 evidence gate，防止 Agent 胡说。",
    },
    {
        "framework": "GPT Researcher",
        "owner": "GPT Researcher community",
        "source_type": "official_open_source",
        "url": "https://github.com/assafelovic/gpt-researcher",
        "supported_models": "web/local research, citation-backed reports, multi-source aggregation, MCP/server-style research workflows",
        "treatment_support": "research/evidence workflow; not uplift training",
        "scenario_fit": "工业案例检索、论文摘要、来源引用、面试材料和 evidence pack 自动化",
        "local_status": "optional_research_workflow",
        "integration_path": "docs/DEEPUplift_RESEARCH_EVIDENCE_GATE.md; planned source-card generator",
        "ui_evidence": "Evidence -> Research Evidence Gate; Knowledge -> Open Source Framework Audit",
        "limitations": "LLM 总结只能作为辅助，关键结论必须落到代码、UI、文档、smoke 或 evidence",
        "next_step": "做 GitHub/paper source-card generator，把输出转成 framework/model/scenario cards",
        "interview_line": "GPT Researcher 提供 citation-backed research 参考；我把结果落成本地证据链，而不是让 Agent 自由发挥。",
    },
    {
        "framework": "Local Deep Researcher",
        "owner": "LangChain",
        "source_type": "official_open_source",
        "url": "https://github.com/langchain-ai/local-deep-researcher",
        "supported_models": "local web research/report assistant with local LLM backends",
        "treatment_support": "local research workflow; not uplift training",
        "scenario_fit": "本地 paper 文件夹、公司案例材料、离线知识库索引和隐私友好的 research workflow",
        "local_status": "backlog_research_tooling",
        "integration_path": "planned: local paper index / source backlog importer",
        "ui_evidence": "Evidence -> Research Evidence Gate",
        "limitations": "当前 DeepUplift 的优先级是模型/评估/Policy 闭环；本地 researcher 暂不进入主 runtime",
        "next_step": "等 source-card schema 稳定后，再考虑本地 PDF/paper ingestion",
        "interview_line": "Local Deep Researcher 是后续资料 ingestion 方向；现阶段只吸收本地可控、证据优先的设计原则。",
    },
    {
        "framework": "Deep uplift paper code family",
        "owner": "paper / local DeepUplift",
        "source_type": "paper_code_or_local_impl",
        "url": "docs/UPLIFT_DEEP_LOSS_AND_ARCHITECTURE.md",
        "supported_models": "TARNet, CFRNet, DragonNet, DESCN, EFIN, ContrastiveUpliftNet, ECUP/CFR-DF guarded routes",
        "treatment_support": "binary ready; multi-treatment / full-funnel / delayed as guarded or evaluator-first",
        "scenario_fit": "需要表示学习、treatment interaction、full-funnel、delayed feedback、LLM routing 的高级演示",
        "local_status": "mixed_ready_guarded",
        "integration_path": "deepuplift/models/*.py; deepuplift/models/deep_losses.py; deepuplift/core/frontier_models.py",
        "ui_evidence": "Models -> Deep Uplift Loss Catalog; Evaluation -> Loss-to-Metric Map",
        "limitations": "深度模型对数据量、调参、偏差和稳定性更敏感，必须和 P0 baseline 对比",
        "next_step": "把 full-funnel/delayed neural heads 从 evaluator-first 推进到 guarded adapter smoke",
        "interview_line": "深度 uplift 是 P1/P2，不是替代 P0；亮点在 loss、结构和业务指标对齐。",
    },
]


def _load_github_metadata() -> dict[str, dict[str, str]]:
    path = REPORTS / "github_framework_audit_latest.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    rows = payload.get("rows") if isinstance(payload, dict) else []
    if not isinstance(rows, list):
        return {}
    out = {}
    for row in rows:
        if isinstance(row, dict) and row.get("framework"):
            out[str(row["framework"])] = row
    return out


def _github_activity(row: dict[str, str]) -> str:
    status = str(row.get("api_status") or "")
    if status and status != "ok":
        return status
    pushed = str(row.get("pushed_at") or row.get("updated_at") or "")
    if pushed >= "2025":
        return "active_recent"
    if pushed >= "2023":
        return "warm"
    if pushed:
        return "stale_check_before_install"
    return "metadata_pending"


def _with_github_metadata(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    metadata = _load_github_metadata()
    enriched: list[dict[str, str]] = []
    for raw in rows:
        row = dict(raw)
        framework = row["framework"]
        meta = metadata.get(framework, {})
        row["github_repo"] = row.get("github_repo") or GITHUB_REPOS.get(framework, "")
        row["github_stars"] = str(meta.get("stargazers_count", row.get("github_stars", "pending")))
        row["github_updated_at"] = str(meta.get("updated_at", row.get("github_updated_at", "pending")))
        row["github_pushed_at"] = str(meta.get("pushed_at", row.get("github_pushed_at", "pending")))
        row["github_archived"] = str(meta.get("archived", row.get("github_archived", "unknown")))
        row["github_api_status"] = str(meta.get("api_status", row.get("github_api_status", "pending_refresh")))
        row["github_activity"] = _github_activity(row)
        enriched.append(row)
    return enriched


def open_source_framework_rows() -> list[dict[str, str]]:
    return _with_github_metadata(deepcopy(OPEN_SOURCE_FRAMEWORKS))


def open_source_framework_counts() -> dict[str, int]:
    counts = {"total": len(OPEN_SOURCE_FRAMEWORKS), "ready": 0, "guarded": 0, "optional": 0, "backlog": 0}
    for row in OPEN_SOURCE_FRAMEWORKS:
        status = row["local_status"].lower()
        if "ready" in status:
            counts["ready"] += 1
        if "guarded" in status:
            counts["guarded"] += 1
        if "optional" in status:
            counts["optional"] += 1
        if "backlog" in status:
            counts["backlog"] += 1
    return counts


def framework_status_summary() -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    status_order = ["ready", "guarded", "optional", "backlog", "mixed"]
    for status in status_order:
        matched = [
            row
            for row in OPEN_SOURCE_FRAMEWORKS
            if status in row["local_status"].lower()
            or (status == "mixed" and "mixed" in row["local_status"].lower())
        ]
        if not matched:
            continue
        rows.append(
            {
                "status": status,
                "frameworks": len(matched),
                "examples": ", ".join(row["framework"] for row in matched[:5]),
                "how_to_read": {
                    "ready": "当前环境可直接进入训练/评估闭环。",
                    "guarded": "有 adapter/model card，但默认不打开，先跑 dependency/import/training smoke。",
                    "optional": "适合插件化接入，不影响主流程。",
                    "backlog": "保留来源和路线，不宣称可训练。",
                    "mixed": "同一家族里有 ready 和 guarded 能力。",
                }[status],
            }
        )
    return rows


def open_source_framework_agent_reply(prompt: str) -> str | None:
    text = prompt.lower()
    if not any(
        key in text
        for key in [
            "开源框架",
            "框架",
            "econml",
            "causalml",
            "scikit-uplift",
            "utboost",
            "pylift",
            "doubleml",
            "dowhy",
            "grf",
            "ylearn",
            "causallib",
            "causaltune",
            "geolift",
            "upliftml",
            "causallift",
            "causal-learn",
            "routellm",
            "route llm",
            "open bandit",
            "obp",
            "vowpal",
            "deep research",
            "gpt researcher",
            "github",
            "哪些库",
            "代码来源",
            "可安装",
        ]
    ):
        return None

    if any(key in text for key in ["ready", "guarded", "optional", "backlog", "限制", "用不了", "不可用", "为什么"]):
        status_rows = framework_status_summary()
        status_text = "\n".join(
            f"- `{row['status']}`：{row['frameworks']} 个，例子：{row['examples']}。{row['how_to_read']}"
            for row in status_rows
        )
        return (
            "### 开源框架接入状态\n\n"
            "这个平台不是把所有开源库都直接塞进训练菜单，而是按 `ready / guarded / optional / backlog` 做工程治理。\n\n"
            f"{status_text}\n\n"
            "当前最稳的生产路径是：`LightGBM + S/T/X/R/DR`、`EconML DR/DML/CausalForest`、`scikit-uplift baseline/metrics`。"
            "CausalML、UTBoost、CatBoost、grf、DoubleML、DoWhy、OBP、RouteLLM、Vowpal Wabbit 会作为 guarded/optional 插件进入 Evidence gate。\n\n"
            "UI 证据：`Models -> Open Source Framework Audit`、`Knowledge -> Open Source Framework Audit`、"
            "`Evidence -> optional backend probe`。代码证据：`deepuplift/core/open_source_frameworks.py`。"
        )

    highlights = [
        ("EconML", "P0 稳健因果层：DML/DR/CausalForest 和 inference。"),
        ("CausalML", "uplift tree/forest 和 meta-learner 后端，当前走 Python 3.11 helper env。"),
        ("scikit-uplift", "轻量模型与 QINI/AUUC 指标校验。"),
        ("UTBoost", "工业 uplift GBDT guarded 插件，适合大表但要先 smoke。"),
        ("DoWhy/DoubleML/grf", "用于因果假设、refutation、正交化和 causal forest 理论补强。"),
        ("GeoLift/UpliftML/YLearn", "用于 geo incrementality、Spark pipeline 和中文因果生态的可选扩展。"),
        ("RouteLLM", "LLM routing 服务化参考，被抽象成多动作 uplift + cost-quality OPE。"),
        ("Open Bandit Pipeline / Vowpal Wabbit", "OPE、contextual bandit、action propensity 和 online learning 的 guarded 参考。"),
        ("Open Deep Research / GPT Researcher", "研究代理参考，转化成 source quality、claim-to-artifact 和 evidence gate。"),
    ]
    return (
        "### 平台参考和接入的开源框架\n\n"
        + "\n".join(f"- **{name}**：{line}" for name, line in highlights)
        + "\n\n面试讲法：我不是只调用一个库，而是把开源框架拆成训练后端、诊断/refutation、评估指标、"
        "optional 插件和 evidence gate 五层；这样既能覆盖工业先进方案，也不会牺牲稳定性。"
    )
