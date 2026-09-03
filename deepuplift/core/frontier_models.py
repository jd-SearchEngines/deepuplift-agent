from __future__ import annotations

from copy import deepcopy


FRONTIER_MODEL_CAPABILITIES: list[dict[str, str]] = [
    {
        "priority": "P0",
        "capability": "Meta-Learners + LightGBM/CatBoost",
        "layer": "baseline",
        "integration_status": "ready + optional",
        "training_status": "LightGBM ready; CatBoost optional dependency",
        "treatment_type": "binary treatment",
        "outcome_type": "classification / regression",
        "business_scenarios": "coupon, ads audience, CRM, push, recommendation intervention, LLM routing",
        "adapter_path": "deepuplift/core/sklearn_models.py",
        "model_examples": "S/T/X/R/DR Learner, IPW, transformed outcome, Domain Adaptation, calibrated variants",
        "metrics": "QINI, AUUC, uplift@K, policy value, calibration, bootstrap CI",
        "assumptions": "randomized or overlap-supported treatment assignment; stable outcome definition",
        "risks": "S-Learner can underfit treatment heterogeneity; IPW can have high variance; CatBoost must be installed",
        "source_title": "scikit-uplift and CausalML meta-learner APIs",
        "source_url": "https://www.uplift-modeling.com/en/latest/api/models/TwoModels.html",
        "recommended_action": "default first pass for industrial tabular uplift; compare simple learners before complex neural nets",
    },
    {
        "priority": "P0",
        "capability": "DR/DML + Causal Forest",
        "layer": "robust causal",
        "integration_status": "ready + optional",
        "training_status": "EconML ready in current env; CausalML guarded helper env",
        "treatment_type": "binary treatment, guarded multi-action forest",
        "outcome_type": "classification / regression",
        "business_scenarios": "observational targeting, biased treatment logs, heterogeneous treatment effect explanation",
        "adapter_path": "deepuplift/core/external_models.py",
        "model_examples": "CausalForestDML, LinearDML, DRLearner, ForestDRLearner, OrthoForest",
        "metrics": "QINI, AUUC, uplift@K, overlap trimming, bootstrap CI, segment stability",
        "assumptions": "positivity/common support; nuisance models are adequate enough for orthogonalization",
        "risks": "requires overlap diagnostics; causal forest intervals do not rescue unsupported treatment regions",
        "source_title": "EconML CausalForestDML and DRLearner documentation",
        "source_url": "https://econml.azurewebsites.net/_autosummary/econml.dml.CausalForestDML.html",
        "recommended_action": "use when treatment assignment is biased or interview asks about robust causal baselines",
    },
    {
        "priority": "P0",
        "capability": "Uplift Tree / Uplift Random Forest",
        "layer": "tree",
        "integration_status": "optional",
        "training_status": "registered via CausalML helper path",
        "treatment_type": "binary or multi-treatment classification",
        "outcome_type": "classification",
        "business_scenarios": "marketing rules, explainable targeting, CRM segmentation",
        "adapter_path": "deepuplift/core/external_models.py",
        "model_examples": "CausalMLUpliftTree, CausalMLUpliftRandomForest, CausalMLCausalRandomForest",
        "metrics": "QINI, AUUC, uplift@K, segment support, rule stability",
        "assumptions": "bucket/rule-level treatment and control support are sufficient",
        "risks": "can overfit small leaves; guarded because CausalML runtime is kept outside the default env",
        "source_title": "CausalML methodology documentation",
        "source_url": "https://causalml.readthedocs.io/en/latest/methodology.html",
        "recommended_action": "use when business owners need rules and interpretable segments, not only score lists",
    },
    {
        "priority": "P1",
        "capability": "UTBoost",
        "layer": "tree",
        "integration_status": "guarded plugin",
        "training_status": "registered guarded adapter + model card + smoke gate; dependency not installed by default",
        "treatment_type": "binary or multi-treatment uplift",
        "outcome_type": "classification / regression",
        "business_scenarios": "large-scale tabular coupon, CRM and ads targeting",
        "adapter_path": "deepuplift/core/external_models.py:_fit_utboost; deepuplift/core/registry.py:UTBoostGBM",
        "model_examples": "UTBoost uplift GBDT",
        "metrics": "QINI, AUUC, uplift@K, policy value, feature importance",
        "assumptions": "GBDT split criteria match treatment-effect objective; sufficient rows per treatment arm",
        "risks": "native dependency and split-criterion differences require guarded smoke, no-UI compare, and evidence gates before default enablement",
        "source_title": "UTBoost: Rigorous Uplift Modeling with Causal Ensemble",
        "source_url": "https://github.com/jd-opensource/UTBoost",
        "recommended_action": "next plugin candidate after stable P0 baselines because it is industrially convenient for big tables",
    },
    {
        "priority": "P1",
        "capability": "EFIN",
        "layer": "deep",
        "integration_status": "ready",
        "training_status": "local DeepUplift implementation behind unified trainer",
        "treatment_type": "binary or coupon-feature interaction proxy",
        "outcome_type": "classification / regression",
        "business_scenarios": "coupon, ads creative, recommendation intervention with user-treatment interaction",
        "adapter_path": "deepuplift/models/EFIN.py; deepuplift/core/registry.py",
        "model_examples": "EFIN",
        "metrics": "QINI, AUUC, uplift@K, calibration, policy value",
        "assumptions": "interaction representation is supported by enough treatment/control examples",
        "risks": "deep interaction models can win offline while being less stable under drift; compare against LightGBM baselines",
        "source_title": "Entire Space Counterfactual Learning for Online Marketing",
        "source_url": "https://arxiv.org/abs/2306.00315",
        "recommended_action": "use as advanced neural candidate after P0 baselines pass overlap and sensitivity checks",
    },
    {
        "priority": "P1",
        "capability": "DESCN / ESX",
        "layer": "deep",
        "integration_status": "ready",
        "training_status": "local DeepUplift implementation; y0/y1 contract normalized by trainer/evaluator",
        "treatment_type": "binary treatment with biased exposure or selection",
        "outcome_type": "classification",
        "business_scenarios": "recommendation marketing, biased campaign logs, entire-space ITE estimation",
        "adapter_path": "deepuplift/models/DESCN.py; deepuplift/core/registry.py",
        "model_examples": "DESCN, ESX",
        "metrics": "QINI, AUUC, uplift@K, overlap trimming, calibration",
        "assumptions": "observed exposure/treatment bias can be represented by model heads and available features",
        "risks": "output order and treatment semantics must be normalized before metrics; compare against DR/R learners",
        "source_title": "KDD 2022 DESCN / ESX-style entire-space uplift modeling",
        "source_url": "https://arxiv.org/search/?query=DESCN+uplift+modeling&searchtype=all",
        "recommended_action": "use when explaining biased treatment logs and why adapters must normalize model outputs",
    },
    {
        "priority": "P1",
        "capability": "TARNet / CFRNet / DragonNet",
        "layer": "deep",
        "integration_status": "ready",
        "training_status": "local DeepUplift neural baselines",
        "treatment_type": "binary treatment",
        "outcome_type": "classification / regression",
        "business_scenarios": "representation learning baseline, overlap imbalance, propensity-aware modeling",
        "adapter_path": "deepuplift/models/TarNet.py; deepuplift/models/CFRNet.py; deepuplift/models/DragonNet.py",
        "model_examples": "TarNet, CFRNet, DragonNet",
        "metrics": "QINI, AUUC, uplift@K, calibration, bootstrap CI",
        "assumptions": "neural representation learns confounding-relevant features and does not overfit small data",
        "risks": "slower than GBM baselines; can be sensitive to epochs and learning rate",
        "source_title": "CFRNet / DragonNet CATE neural baselines",
        "source_url": "https://arxiv.org/abs/1606.03976",
        "recommended_action": "keep as NN benchmark family, not the first and only production choice",
    },
    {
        "priority": "P2",
        "capability": "UMLC",
        "layer": "business extension",
        "integration_status": "research backlog",
        "training_status": "model card + source gate; no default adapter yet",
        "treatment_type": "contextual marketing treatment / item-treatment interaction",
        "outcome_type": "classification / conversion",
        "business_scenarios": "real-time marketing, context-item-user interaction, campaign ranking",
        "adapter_path": "planned: deepuplift/core/frontier_plugins/umlc.py",
        "model_examples": "UMLC-style real-time marketing learner",
        "metrics": "QINI, AUUC, uplift@K, policy value, latency-aware score stability",
        "assumptions": "rich context/item/treatment features are available online and offline with matched schemas",
        "risks": "source and implementation need stronger audit before it becomes a trainable plugin",
        "source_title": "UMLC KDD 2025 uplift marketing direction",
        "source_url": "source-backlog",
        "recommended_action": "keep visible as frontier backlog, not a claimed runnable model",
    },
    {
        "priority": "P2",
        "capability": "ECUP / full-funnel uplift",
        "layer": "business extension",
        "integration_status": "research backlog",
        "training_status": "evaluator, UI, evidence and synthetic training smoke support; no ECUP neural adapter yet",
        "treatment_type": "impression-click-conversion funnel treatment",
        "outcome_type": "multi-stage classification",
        "business_scenarios": "ads, recommendation exposure, ecommerce funnel, coupon impression-click-conversion",
        "adapter_path": "deepuplift/core/evaluator.py:full_funnel_uplift_metrics",
        "model_examples": "ECUP, MT-LIFT-style full-funnel uplift",
        "metrics": "full-funnel ECUP metric, QINI by stage, conversion uplift, policy value",
        "assumptions": "exposure, click and conversion are all logged with consistent identity and delay windows",
        "risks": "optimizing conversion uplift alone can hide click/exposure selection and delayed attribution bias",
        "source_title": "Enhancing Conversion Uplift Modeling via Multi-Task Learning",
        "source_url": "https://arxiv.org/abs/2402.03379",
        "recommended_action": "use as an interview-level extension for ads/recommendation full-funnel treatment design",
    },
    {
        "priority": "P2",
        "capability": "CFR-DF / delayed feedback uplift",
        "layer": "business extension",
        "integration_status": "research backlog",
        "training_status": "evaluator, UI, evidence and synthetic training smoke support; no CFR-DF adapter yet",
        "treatment_type": "binary treatment with delayed conversion",
        "outcome_type": "delayed classification / survival-like conversion",
        "business_scenarios": "7/14/30-day conversion, ads attribution, delayed purchase, lifecycle CRM",
        "adapter_path": "deepuplift/core/evaluator.py:delayed_feedback_uplift_metrics",
        "model_examples": "CFR-DF",
        "metrics": "delayed feedback uplift, windowed QINI/AUUC, censoring-aware policy value",
        "assumptions": "conversion delay and censoring windows are explicitly modeled or at least audited",
        "risks": "short windows bias against slow responders; leakage appears if future conversions enter training features",
        "source_title": "Counterfactual Uplift Estimation with Delayed Feedback",
        "source_url": "https://ojs.aaai.org/index.php/AAAI/article/view/38686",
        "recommended_action": "prioritize if business outcome is delayed, e.g. ad conversion or 30-day retention",
    },
    {
        "priority": "P2",
        "capability": "Continuous Treatment / Dose-Response",
        "layer": "business extension",
        "integration_status": "ready baseline + research backlog",
        "training_status": "local dose-response baseline ready; DRNet/VCNet-style plugin backlog",
        "treatment_type": "continuous treatment",
        "outcome_type": "classification / regression",
        "business_scenarios": "discount depth, subsidy amount, bid multiplier, contact frequency, price intensity",
        "adapter_path": "deepuplift/core/doseresponse.py",
        "model_examples": "DoseResponseGBM, DoseResponseRF, planned DRNet/VCNet",
        "metrics": "dose-response gain, recommended dose distribution, policy value, monotonicity sanity checks",
        "assumptions": "dose is observed over a useful range with enough support for interpolation",
        "risks": "extrapolating to unseen doses is unsafe; optimize with business constraints and online holdout",
        "source_title": "Continuous treatment uplift and policy learning references",
        "source_url": "https://arxiv.org/abs/2412.09232",
        "recommended_action": "ready for demo when treatment is numeric; keep neural dose-response as plugin backlog",
    },
    {
        "priority": "P2",
        "capability": "Cost-aware / Incremental Profit",
        "layer": "policy",
        "integration_status": "ready",
        "training_status": "Policy page and evaluator artifacts ready",
        "treatment_type": "binary, multi-treatment, continuous thresholding",
        "outcome_type": "classification / value / cost-quality",
        "business_scenarios": "coupon gross profit, ads iROAS, push fatigue, LLM routing cost-quality",
        "adapter_path": "deepuplift/core/policy.py; deepuplift/core/evaluator.py",
        "model_examples": "policy value curve, constrained audience export, LLM routing ROI simulator",
        "metrics": "incremental profit, net policy value, constrained Top-K value, iROAS proxy",
        "assumptions": "business value and cost assumptions are explicit and versioned with the run",
        "risks": "wrong margin/cost assumptions can reverse the launch decision even when AUUC is positive",
        "source_title": "Value-driven uplift decision layer",
        "source_url": "local-evidence",
        "recommended_action": "always run after model compare because deployment is a thresholding problem",
    },
    {
        "priority": "P2",
        "capability": "LLM Routing Uplift",
        "layer": "policy",
        "integration_status": "ready synthetic + policy simulator",
        "training_status": "LLM routing dataset smoke and policy simulator ready",
        "treatment_type": "binary or multi-action model escalation",
        "outcome_type": "quality / resolution / conversion minus cost",
        "business_scenarios": "small-vs-large model routing, tool/retrieval escalation, creative generation treatment",
        "adapter_path": "deepuplift/core/industry_playbook.py; scripts/smoke_llm_routing_policy.py",
        "model_examples": "route/no-route uplift, strong-model escalation, cost-quality policy value",
        "metrics": "quality uplift, net value, budget-constrained policy value, calibration",
        "assumptions": "LLM treatment assignment and outcome labels are logged with enough counterfactual exploration",
        "risks": "offline labels can be judge-biased; use holdout or randomized exploration before production routing",
        "source_title": "DeepUplift local LLM routing playbook",
        "source_url": "docs/LLM_ROUTING_POLICY_PLAYBOOK.md",
        "recommended_action": "use in interviews to show uplift is a general causal decision layer, not only marketing coupons",
    },
]


SCENARIO_MODEL_LADDERS: list[dict[str, str]] = [
    {
        "scenario": "Coupon / subsidy allocation",
        "treatment_design": "send/no-send, coupon type, discount depth, subsidy amount",
        "p0_first": "T/X/DR/R Learner + LightGBM; CausalForestDML for heterogeneity",
        "p1_candidate": "EFIN for user-treatment interaction; UTBoost guarded for large tables; DESCN if treatment is biased",
        "p2_extension": "multi-treatment coupon type, continuous discount dose-response, ECUP funnel when impression/click/conversion are logged",
        "evaluation_focus": "incremental profit, uplift@K, policy value, calibration, bootstrap CI",
        "policy_focus": "gross margin minus coupon cost; avoid sure-thing waste, sleeping dogs and subsidy abuse",
        "go_no_go_gate": "launch only if Top-K incremental profit is positive and overlap/CI gates are stable",
        "ui_evidence": "Policy: Scenario ROI Lab; Models: Industrial Model Layer Architecture; Evaluation: cost-aware gain",
    },
    {
        "scenario": "Ads / audience targeting / bidding",
        "treatment_design": "ad exposure, bid multiplier, creative, budget on/off, geo holdout",
        "p0_first": "DR/R Learner, CausalForestDML, ForestDRLearner, geo/holdout-aware evaluation",
        "p1_candidate": "UTBoost guarded for uplift GBDT; EFIN for creative-treatment interaction",
        "p2_extension": "ECUP full-funnel uplift, delayed feedback uplift, causal bandit for budget pacing",
        "evaluation_focus": "QINI/AUUC, iROAS proxy, delayed conversion windows, full-funnel stage uplift",
        "policy_focus": "incremental conversion value minus media cost; avoid attribution-only pCTR/pCVR decisions",
        "go_no_go_gate": "offline uplift must agree with holdout/geo lift design before scaling bidding policy",
        "ui_evidence": "Evaluation: full-funnel ECUP metric; Policy: Ads ROI template; Evidence: source gate",
    },
    {
        "scenario": "Growth / recall / push / SMS",
        "treatment_design": "push/SMS/call/email, timing, frequency, creative",
        "p0_first": "DR/R Learner + calibrated ranking; uplift tree/forest for explainable rules",
        "p1_candidate": "CFRNet/DragonNet for representation and propensity-aware baselines",
        "p2_extension": "CFR-DF delayed feedback uplift, fatigue-aware policy value, lifecycle multi-touch treatment",
        "evaluation_focus": "uplift@K, negative-action uplift, delayed feedback uplift, calibration and bootstrap CI",
        "policy_focus": "activation or retention uplift minus contact cost, fatigue, opt-out and long-term harm",
        "go_no_go_gate": "do not launch if short-window uplift is positive but opt-out/fatigue or delayed CI is weak",
        "ui_evidence": "Evaluation: delayed feedback uplift; Policy: Growth Push ROI template; Agent: fatigue Q&A",
    },
    {
        "scenario": "Recommendation intervention",
        "treatment_design": "ranking boost, exposure policy, creator/content intervention, exploration arm",
        "p0_first": "DR/R Learner, CausalForestDML, domain-adaptation learners",
        "p1_candidate": "DESCN/ESX for biased entire-space logs; EFIN for item-treatment interaction",
        "p2_extension": "ECUP full-funnel recommendation uplift, switchback/cluster experiments, dynamic uplift",
        "evaluation_focus": "QINI/AUUC, segment stability, full-funnel uplift, long-term retention uplift",
        "policy_focus": "incremental engagement or value after position bias, network effects and feedback-loop checks",
        "go_no_go_gate": "requires exposure/position bias diagnostics and online holdout/switchback validation",
        "ui_evidence": "Models: scenario matrix; Knowledge: recommendation intervention sources; Evidence: source audit",
    },
    {
        "scenario": "Marketplace subsidy / interference",
        "treatment_design": "buyer subsidy, driver/seller incentive, price/fee intensity, regional policy",
        "p0_first": "MultiDRLearner, DoseResponseGBM, CausalForestDML",
        "p1_candidate": "UTBoost guarded for large subsidy tables; EFIN when offer/treatment features are rich",
        "p2_extension": "continuous treatment, spillover-aware experiment design, budget-constrained causal bandit",
        "evaluation_focus": "supply-demand policy value, spillover-adjusted lift, dose-response gain, bootstrap CI",
        "policy_focus": "GMV/profit lift under budget and interference constraints",
        "go_no_go_gate": "do not treat SUTVA-violating marketplace logs as independent rows without experiment design",
        "ui_evidence": "Policy: Marketplace ROI template; Models: continuous/multi-treatment rows; Agent: spillover answer",
    },
    {
        "scenario": "LLM routing / cost-quality escalation",
        "treatment_design": "small model vs strong model, retrieval/tool escalation, human review, prompt variant",
        "p0_first": "DR/R Learner + calibrated ranking; CausalForestDML for heterogeneity",
        "p1_candidate": "EFIN-style query-treatment interaction when model/prompt features are rich",
        "p2_extension": "budget-constrained bandit, non-stationary routing, universal model portfolio routing",
        "evaluation_focus": "incremental quality, cost-aware gain, calibration, budget-constrained policy value",
        "policy_focus": "quality uplift times business value minus extra model/tool/latency cost",
        "go_no_go_gate": "requires randomized exploration or overlap; judge labels and drift must be monitored",
        "ui_evidence": "Policy: LLM Routing ROI Simulator; Knowledge: LLM + Causal Frontier Map; Agent: LLM routing answer",
    },
]


def frontier_model_capabilities() -> list[dict[str, str]]:
    return deepcopy(FRONTIER_MODEL_CAPABILITIES)


def scenario_model_ladders() -> list[dict[str, str]]:
    return deepcopy(SCENARIO_MODEL_LADDERS)


def frontier_source_rows() -> list[dict[str, str]]:
    rows = []
    seen: set[tuple[str, str]] = set()
    for item in FRONTIER_MODEL_CAPABILITIES:
        key = (item["source_title"], item["source_url"])
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "priority": item["priority"],
                "capability": item["capability"],
                "source_title": item["source_title"],
                "source_url": item["source_url"],
                "integration_status": item["integration_status"],
                "adapter_path": item["adapter_path"],
                "source_gate": "core" if item["source_url"].startswith(("http://", "https://", "docs/", "local-")) else "backlog",
            }
        )
    return rows


def capability_counts() -> dict[str, int]:
    counts = {"total": len(FRONTIER_MODEL_CAPABILITIES), "ready": 0, "guarded": 0, "optional": 0, "backlog": 0}
    for item in FRONTIER_MODEL_CAPABILITIES:
        status = item["integration_status"].lower()
        if "ready" in status:
            counts["ready"] += 1
        if "guarded" in status:
            counts["guarded"] += 1
        if "optional" in status:
            counts["optional"] += 1
        if "backlog" in status:
            counts["backlog"] += 1
    return counts


def scenario_ladder_agent_reply(prompt: str) -> str | None:
    text = prompt.lower()
    if not any(key in text for key in ["场景", "业务", "发券", "广告", "push", "crm", "推荐", "marketplace", "llm routing", "模型路径", "模型路线"]):
        return None
    scenario_keywords = {
        "Coupon / subsidy allocation": ["coupon", "subsidy", "发券", "补贴", "优惠券"],
        "Ads / audience targeting / bidding": ["ads", "ad ", "广告", "bidding", "iroas", "pctr", "pcvr"],
        "Growth / recall / push / SMS": ["growth", "push", "sms", "召回", "增长", "短信"],
        "Recommendation intervention": ["recommendation", "推荐", "rerank", "重排"],
        "Marketplace subsidy / interference": ["marketplace", "spillover", "平台", "供需", "干扰"],
        "LLM routing / cost-quality escalation": ["llm", "大模型", "routing", "强模型", "小模型"],
    }
    matched = []
    for row in SCENARIO_MODEL_LADDERS:
        tokens = scenario_keywords.get(row["scenario"], [])
        if any(token and token.lower() in text for token in tokens):
            matched.append(row)
    generic_request = any(key in text for key in ["场景", "业务", "模型路径", "模型路线"])
    if not matched and generic_request:
        matched = SCENARIO_MODEL_LADDERS[:3]
    elif not matched:
        return None
    matched = matched[:4]
    blocks = []
    for row in matched:
        blocks.append(
            f"#### {row['scenario']}\n"
            f"- Treatment design: {row['treatment_design']}\n"
            f"- P0 first: {row['p0_first']}\n"
            f"- P1 candidate: {row['p1_candidate']}\n"
            f"- P2 extension: {row['p2_extension']}\n"
            f"- Evaluation: {row['evaluation_focus']}\n"
            f"- Policy: {row['policy_focus']}\n"
            f"- Gate: {row['go_no_go_gate']}"
        )
    return (
        "### 业务场景模型路线图\n\n"
        "我会先看 treatment 设计和数据偏差，再决定 P0/P1/P2，而不是直接追最新模型。\n\n"
        + "\n\n".join(blocks)
        + "\n\nUI 证据：`Models -> Scenario-to-Model Decision Ladder`，`Evaluation -> metric map`，`Policy -> Scenario ROI Lab`。"
    )


def frontier_demo_trace_rows() -> list[dict[str, str]]:
    return [
        {
            "demo_preset": "Coupon ROI",
            "dataset_id": "synthetic_coupon_profit_uplift_8k",
            "manifest_source": "examples/datasets/manifest.json",
            "data_columns": "gross_margin, coupon_face_value, expected_coupon_cost, subsidy_abuse_risk, uplift_type, oracle_policy_value",
            "evaluator_contract": "deepuplift/core/evaluator.py:oracle_policy_top_k; deepuplift/core/policy.py",
            "metrics_json": "runs/{run_id}/metrics.json:oracle_policy_top_k; policy_best",
            "compare_history_columns": "oracle_top10_recall, policy_best_net, business_score_alignment",
            "validation": "scripts/generate_industrial_scenario_datasets.py; scripts/smoke_industrial_scenario_datasets.py; scripts/smoke_industrial_scenario_training.py --dataset-id synthetic_coupon_profit_uplift_8k",
        },
        {
            "demo_preset": "ECUP full-funnel",
            "dataset_id": "synthetic_ads_full_funnel_ecup_7k",
            "manifest_source": "examples/datasets/manifest.json",
            "data_columns": "impression, click, conversion, true_*_uplift, media_cost, oracle_policy_value",
            "evaluator_contract": "deepuplift/core/evaluator.py:full_funnel_uplift_metrics",
            "metrics_json": "runs/{run_id}/metrics.json:full_funnel",
            "compare_history_columns": "full_funnel_top10_conversion_uplift, full_funnel_top10_click_uplift",
            "validation": "scripts/smoke_frontier_evaluator_metrics.py; scripts/smoke_compare_manifest.py --dataset-id synthetic_ads_full_funnel_ecup_7k",
        },
        {
            "demo_preset": "Delayed feedback",
            "dataset_id": "synthetic_growth_delayed_feedback_7k",
            "manifest_source": "examples/datasets/manifest.json",
            "data_columns": "converted_d1/d7/d14/d30, conversion_delay_days, censored, true_uplift_d*, contact_cost",
            "evaluator_contract": "deepuplift/core/evaluator.py:delayed_feedback_uplift_metrics",
            "metrics_json": "runs/{run_id}/metrics.json:delayed_feedback",
            "compare_history_columns": "delayed_d30_top10_uplift, delayed_censored_rate, delayed_observed_delay_p90",
            "validation": "scripts/smoke_frontier_evaluator_metrics.py; scripts/smoke_compare_manifest.py --dataset-id synthetic_growth_delayed_feedback_7k",
        },
        {
            "demo_preset": "Recommendation intervention",
            "dataset_id": "synthetic_recommendation_intervention_7k",
            "manifest_source": "examples/datasets/manifest.json",
            "data_columns": "creative_type, item_category, impression/click/conversion, negative_uplift_risk, segment_stability_score",
            "evaluator_contract": "deepuplift/core/evaluator.py:full_funnel_uplift_metrics; oracle_policy_top_k",
            "metrics_json": "runs/{run_id}/metrics.json:full_funnel; oracle_policy_top_k",
            "compare_history_columns": "full_funnel_top10_conversion_uplift, oracle_top10_recall, policy_best_net",
            "validation": "scripts/generate_industrial_scenario_datasets.py; scripts/smoke_industrial_scenario_datasets.py; scripts/smoke_industrial_scenario_training.py --dataset-id synthetic_recommendation_intervention_7k",
        },
        {
            "demo_preset": "Marketplace subsidy",
            "dataset_id": "synthetic_marketplace_subsidy_uplift_7k",
            "manifest_source": "examples/datasets/manifest.json",
            "data_columns": "planned_subsidy_amount, spillover_risk, interference_risk, net_marketplace_value, oracle_policy_value",
            "evaluator_contract": "deepuplift/core/evaluator.py:oracle_policy_top_k; overlap_trim_diagnostics",
            "metrics_json": "runs/{run_id}/metrics.json:oracle_policy_top_k; overlap_trim; policy_best",
            "compare_history_columns": "policy_best_net, oracle_top10_recall, weak_overlap_rate",
            "validation": "scripts/generate_industrial_scenario_datasets.py; scripts/smoke_industrial_scenario_datasets.py; scripts/smoke_industrial_scenario_training.py --dataset-id synthetic_marketplace_subsidy_uplift_7k",
        },
        {
            "demo_preset": "LLM routing",
            "dataset_id": "synthetic_llm_routing_uplift_8k",
            "manifest_source": "examples/datasets/manifest.json",
            "data_columns": "prompt/task/context features, treatment=strong model route, oracle_policy_value",
            "evaluator_contract": "deepuplift/core/policy.py; app.py Policy LLM routing simulator",
            "metrics_json": "reports/llm_routing_policy_smoke_latest.json; runs/{run_id}/metrics.json:policy_best",
            "compare_history_columns": "policy_best_net, business_score_alignment, oracle_top10_recall when available",
            "validation": "scripts/smoke_llm_routing_dataset.py; scripts/smoke_llm_routing_policy.py",
        },
        {
            "demo_preset": "LLM multi-action routing",
            "dataset_id": "synthetic_llm_multi_action_routing_9k",
            "manifest_source": "examples/datasets/manifest.json",
            "data_columns": "route_action, best_action, cheap/strong/rag/tool/human quality, hallucination_risk, evidence_failure_risk, oracle_policy_value",
            "evaluator_contract": "deepuplift/core/evaluator.py:industrial_policy_metrics(llm_multi_action_routing); app.py Policy multi-action router",
            "metrics_json": "reports/industrial_scenario_compare_smoke_latest.json; runs/{run_id}/metrics.json:industrial_policy",
            "compare_history_columns": "industrial_top10_policy_value_sum, industrial_top10_roi_proxy, oracle_policy_top10_capture",
            "validation": "scripts/smoke_llm_routing_dataset.py --dataset-id synthetic_llm_multi_action_routing_9k; scripts/smoke_industrial_scenario_compare.py --dataset-id synthetic_llm_multi_action_routing_9k",
        },
    ]


def frontier_agent_reply(prompt: str) -> str | None:
    text = prompt.lower()
    ladder_reply = scenario_ladder_agent_reply(prompt)
    if ladder_reply:
        return ladder_reply
    if any(key in text for key in ["demo preset", "source trace", "演示入口", "演示证据", "指标来源", "字段来源"]):
        rows = frontier_demo_trace_rows()
        return (
            "### Demo Preset Source Trace\n\n"
            "Demo preset 不是松散样例文件，而是绑定到 manifest、evaluator contract、metrics.json、Compare/History 字段和 smoke 验证的证据链。\n\n"
            f"核心链路：`{rows}`\n\n"
            "UI 证据：`Workflow -> Demo Preset Source Trace`、`Evaluation -> Frontier Metric Source Trace` 和 `Evidence -> Demo Preset Source Trace`。"
            "代码证据：`deepuplift/core/frontier_models.py:frontier_demo_trace_rows`、"
            "`scripts/smoke_demo_preset_source_trace.py`、`deepuplift/core/evaluator.py`。"
        )
    if any(
        key in text
        for key in [
            "guarded",
            "optional",
            "readiness",
            "模型治理",
            "可选模型",
            "不是所有模型",
            "为什么不全部 ready",
            "为什么不是所有模型都 ready",
        ]
    ):
        return (
            "### Guarded / Optional 模型怎么治理\n\n"
            "我不会把所有 uplift 后端都伪装成 ready。平台把模型分成 `registered`、`ready`、`guarded/optional`："
            "`registered` 代表有模型卡和来源，`ready` 代表当前环境可以训练，`guarded/optional` 代表依赖、Python 版本或运行时稳定性还需要验证。"
            "这对面试很关键，因为工业平台更重视可复现和上线风险，而不是模型名单越长越好。\n\n"
            "落地证据：`Models -> Model Card Governance Matrix` 和 `Evidence -> Model Card Governance` 展示每个可选后端的 "
            "`integration_gate`、`source_evidence`、`promotion_rule`；"
            "`scripts/probe_optional_uplift_backends.py` 检查 UTBoost/CatBoost/XGBoost/CausalML importability；"
            "`scripts/smoke_utboost_guarded_adapter.py` 检查 UTBoost 的 guarded adapter 和 model-card governance。"
            "只有 dependency/import、catalog audit、tiny training/no-UI compare、截图/evidence gate 都通过，才从 guarded 晋级到 demo-ready。"
        )
    if any(key in text for key in ["不要只追最新", "最新模型", "只追最新", "为什么不要", "s-learner", "lightgbm", "强 baseline"]):
        return (
            "### 为什么不要只追最新 uplift 模型\n\n"
            "工业 uplift 的第一原则是先把强 baseline 和评估闭环做稳。S/T/X/R/DR learner + LightGBM/CatBoost "
            "通常已经覆盖大部分发券、广告、CRM 和推荐干预问题，而且训练快、解释清楚、方便做 bootstrap/overlap/policy value。"
            "复杂深度模型应该作为 P1/P2 插件进入 Compare，而不是绕过 P0 基线。\n\n"
            "**平台落点**：`Models -> Industrial Model Layer Architecture` 显示 P0/P1/P2；"
            "`Evaluation` 用 QINI、AUUC、uplift@K、policy value、calibration、bootstrap CI 统一比较；"
            "`Evidence` 记录 adapter path 和 source gate。"
        )
    if any(key in text for key in ["qini", "auuc", "增量利润", "incremental profit", "roi", "成本收益"]):
        return (
            "### 为什么 QINI/AUUC 不等于增量利润\n\n"
            "QINI/AUUC 衡量模型排序是否比随机投放更早找到增量用户，但上线决策还要乘以业务价值并扣除成本。"
            "发券要扣券成本、毛利损失和疲劳/套利风险；广告要看 iROAS 和真实 holdout lift；"
            "LLM routing 要看质量增量是否超过强模型成本与时延。"
            "因此最终要看 policy value / incremental profit：`uplift * value - treatment cost - risk penalty`。\n\n"
            "平台落点：`Evaluation` 解释 QINI/AUUC/uplift@K/policy value，`Policy -> Metric Disagreement Lab` 展示同一离线排序在不同成本假设下可能得出相反上线结论。"
        )
    if any(key in text for key in ["utboost", "uplift gbdt"]):
        return (
            "### UTBoost 怎么接\n\n"
            "UTBoost 是 GBDT uplift 插件候选，适合大规模表格营销数据。当前平台把它放在 P1 `guarded plugin`："
            "先在 Models/Evidence 暴露 model card、source、adapter path、guarded smoke 和评估指标，等依赖与 smoke 回归稳定后再进入默认训练菜单。"
            "上线前必须和 S/T/X/DR LightGBM、CausalForest 做 QINI/AUUC/Top-K/policy value 对比。\n\n"
            "工程取舍是：注册 `UTBoostGBM`，但依赖 `utboost` 不在默认环境时保持 `needs dependency`，不会污染主流程。"
            "面试时可以强调这不是简单罗列模型，而是把前沿插件放进统一 registry、adapter、model card、metric 和 source gate。"
            "本地证据在 `Evidence -> UTBoost Guarded Adapter Smoke`，代码路径是 `deepuplift/core/external_models.py:_fit_utboost` 和 `scripts/smoke_utboost_guarded_adapter.py`。"
            "如果某天安装依赖，下一步要先跑 catalog audit、no-UI compare、bootstrap/overlap/sensitivity，再决定是否从 guarded 升级为 ready。"
        )
    if any(key in text for key in ["efin", "descn", "esx"]):
        return (
            "### EFIN / DESCN 适合什么\n\n"
            "EFIN 更强调 user-treatment interaction，适合 coupon、广告 creative、推荐干预等 treatment 特征复杂的场景。"
            "DESCN/ESX 更适合 biased treatment / entire-space ITE 这类有选择偏差的转化建模。"
            "平台把它们放在 P1 deep layer：能训练，但必须和 P0 DR/R/DML/LightGBM baseline 对比，并检查 output contract、calibration 和 overlap。"
        )
    if any(key in text for key in ["umlc", "ecup", "cfr-df", "delayed", "延迟", "full-funnel", "全链路"]):
        return (
            "### P2 前沿插件怎么落地\n\n"
            "UMLC、ECUP、CFR-DF 这类模型现在不应该被包装成默认 ready。正确做法是先把业务问题和指标接进平台："
            "ECUP 对应 impression-click-conversion 全链路指标，CFR-DF 对应 7/14/30 天 delayed feedback uplift，"
            "UMLC 对应实时营销里的 context/item/treatment 交互。等数据 schema、指标、source gate 和 smoke 都稳定后，再接 guarded adapter。\n\n"
            "当前平台已经补了两类可跑证据：`Synthetic Ads Full-Funnel ECUP 7k` 和 `Synthetic Growth Delayed Feedback 7k`。"
            "它们让面试官看到 full-funnel / delayed-feedback 不是口号，而是会改变数据 schema、泄漏检查、评估窗口和 policy value。"
            "真正上线时还要补曝光/点击/转化口径一致性、label maturity、censoring、长期 holdout 和在线实验。"
        )
    if any(key in text for key in ["continuous", "dose", "剂量", "连续 treatment", "剂量响应"]):
        return (
            "### Continuous treatment / dose-response\n\n"
            "当 treatment 是折扣深度、补贴金额、触达频次、bid multiplier 时，二元 uplift 不够。"
            "平台当前已用 `DoseResponseGBM/RF` 做 ready baseline：输出 dose grid、推荐剂量和 gain-vs-baseline。"
            "后续 DRNet/VCNet/VC-Transformer 作为 P2 插件进入，但必须避免对未观测 dose 的外推。"
        )
    if any(key in text for key in ["llm routing", "大模型路由", "强模型", "小模型", "llm"]):
        return (
            "### LLM routing 为什么是 uplift 问题\n\n"
            "路由决策不是预测哪个请求本来就会成功，而是判断“调用更贵模型/检索/工具/人工升级”是否带来增量质量，且增量价值是否超过成本。"
            "所以它天然是 treatment effect：treatment=升级强模型，outcome=质量/转化/解决率，policy value=quality uplift * value - extra cost。"
            "平台在 `Policy` 页有 LLM Routing ROI Simulator，在 `Models` 页把它纳入 cost-aware uplift layer。"
        )
    return None
