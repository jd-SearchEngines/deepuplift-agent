from __future__ import annotations

from collections.abc import Mapping
from typing import Any


INDUSTRY_REFERENCES = [
    {
        "id": "TENCENT_ADS_UPLIFT",
        "company": "Tencent Ads",
        "source_type": "official_product_doc",
        "scenario": "ads_incrementality",
        "title": "Uplift广告增效衡量FAQ",
        "url": "https://tencentads.com/Faqlist/Detail/676",
        "lesson": "Ad lift measurement should use an exposure/control design and full conversion feedback, not only click-attributed conversions.",
        "interview_hook": "广告投放要回答广告是否真正带来增量，而不是只解释归因口径下的转化。",
    },
    {
        "id": "TENCENT_GHOST_AD",
        "company": "Tencent Ads",
        "source_type": "official_product_article",
        "scenario": "ads_incrementality",
        "title": "腾讯广告上线Uplift广告增效衡量",
        "url": "https://tencentads.com/Special/Detail/1bed8340-bd3",
        "lesson": "Ghost-ad/holdout style designs can measure the causal relationship between exposure and conversion.",
        "interview_hook": "工业广告增效衡量的核心是构造用户看见广告与没看见广告的可比反事实。",
    },
    {
        "id": "DOORDASH_GHOST_ADS_INCREMENTALITY",
        "company": "DoorDash Ads",
        "source_type": "official_product_article",
        "scenario": "ads_incrementality_ghost_ads",
        "title": "Measuring incrementality with ghost ads at DoorDash",
        "url": "https://advertising.doordash.com/en-au/resources/measuring-incrementality-with-ghost-ads-at-doordash",
        "lesson": "Ghost ads create a control opportunity in auctions by simulating an ad exposure for holdout users, helping estimate incremental lift and diagnose ad-attribution bias.",
        "interview_hook": "DoorDash ghost ads 可以用来讲广告曝光的反事实构造、sales lift、iROAS 和 attribution bias audit。",
    },
    {
        "id": "DOORDASH_SALES_LIFT_MEASUREMENT",
        "company": "DoorDash Ads",
        "source_type": "official_product_article",
        "scenario": "ads_sales_lift_measurement",
        "title": "DoorDash Ads launches sales lift measurement",
        "url": "https://about.doordash.com/en-us/news/doordash-ads-launches-sales-lift-measurement",
        "lesson": "Business-facing incrementality reporting should translate lift into sales and advertiser value, not stop at click attribution.",
        "interview_hook": "广告增量汇报要从模型指标落到 sales lift 和 iROAS，方便业务判断预算是否值得继续投。",
    },
    {
        "id": "MEITUAN_ECUP",
        "company": "Meituan",
        "source_type": "WWW_2024_paper",
        "scenario": "coupon_full_funnel",
        "title": "Entire Chain Uplift Modeling with Context-Enhanced Learning for Intelligent Marketing",
        "url": "https://arxiv.org/abs/2402.03379",
        "lesson": "Coupon/discount treatment affects the whole chain from impression to click to conversion; optimizing only final conversion can introduce chain bias.",
        "interview_hook": "发券/营销不是单点 CVR 问题，而是全链路增量和 ROI 问题。",
    },
    {
        "id": "ALI_ENTERTAINMENT_UPLIFT",
        "company": "Alibaba Entertainment",
        "source_type": "industry_talk_mirror",
        "scenario": "coupon_subsidy",
        "title": "阿里文娱智能营销增益模型 Uplift Model 技术实践",
        "url": "https://cloud.tencent.com/developer/article/1620903",
        "lesson": "Response models can mislead subsidy allocation; uplift modeling separates persuadable users from sure things, lost causes, and sleeping dogs.",
        "interview_hook": "淘票票票补这类场景要找营销敏感人群，而不是找自然转化概率最高的人。",
    },
    {
        "id": "ALIBABA_ABTEST_HETEROGENEITY_XLEARNER",
        "company": "Alibaba Cloud Developer Community",
        "source_type": "official_developer_article",
        "scenario": "logistics_subsidy_heterogeneity",
        "title": "使用Uplift模型进行AB实验的异质性分析与策略优化",
        "url": "https://developer.aliyun.com/article/1606416",
        "lesson": "AB-test heterogeneity analysis can use uplift/ITE/CATE, X-Learner, SHAP and subgroup drilldowns to decide where a subsidy strategy is effective or should be optimized.",
        "interview_hook": "中国业务面试可以讲随机分流、X-Learner、CATE 分层、SHAP 解释和补贴降本/替换决策。",
    },
    {
        "id": "ALIBABA_EDGEREC_REQUEST_UPLIFT",
        "company": "Alibaba Cloud Developer Community",
        "source_type": "official_developer_article",
        "scenario": "recommendation_request_uplift",
        "title": "EdgeRec电商端上推荐的重排请求与个性化模型算法",
        "url": "https://developer.aliyun.com/article/807710",
        "lesson": "Request uplift estimates purchase-rate gain caused by an extra recommendation request rather than purchase probability itself; reported online gains include transaction lift and QPS reduction.",
        "interview_hook": "推荐干预可以把 treatment 定义成是否追加请求/重排，目标是请求带来的真实购买增益和系统成本收益。",
    },
    {
        "id": "DIDI_SUBSIDY_DCN",
        "company": "DiDi",
        "source_type": "ECML_PKDD_2024_industry_paper",
        "scenario": "multi_level_subsidy",
        "title": "A Multi-Class Ride-Hailing Service Subsidy System Utilizing Deep Causal Networks",
        "url": "https://ecmlpkdd-storage.s3.eu-central-1.amazonaws.com/2024/industry_track_papers/1580_AMultiClassRideHailingServiceSubsidySystemUtilizingDeepCausalNetworks.pdf",
        "lesson": "Ride-hailing subsidy levels are multi-class treatments; the decision is not just contact/no-contact but which subsidy level to assign.",
        "interview_hook": "出行补贴天然是多 treatment、预算约束和 marketplace 干扰问题。",
    },
    {
        "id": "BYTEDANCE_DELAYED_UPLIFT",
        "company": "ByteDance",
        "source_type": "AAAI_2026_paper",
        "scenario": "push_growth_delayed_feedback",
        "title": "Uplift Modeling with Delayed Feedback: Identifiability and Algorithms",
        "url": "https://ojs.aaai.org/index.php/AAAI/article/view/38686",
        "lesson": "Push frequency and activation effects can be delayed; ignoring time-to-effect biases uplift labels and online decisions.",
        "interview_hook": "增长召回不能只看当天转化，要考虑延迟反馈、频控和负向体验指标。",
    },
    {
        "id": "UBER_CAUSAL_INFERENCE",
        "company": "Uber",
        "source_type": "official_tech_blog",
        "scenario": "platform_experimentation",
        "title": "Using Causal Inference to Improve the Uber User Experience",
        "url": "https://www.uber.com/us/en/blog/causal-inference-at-uber/",
        "lesson": "A practical HTE workflow starts with A/B test data and then trains uplift/heterogeneous-treatment-effect models.",
        "interview_hook": "工业界常把随机实验作为 HTE/uplift 模型训练和验证的基础。",
    },
    {
        "id": "UBER_CAUSALML",
        "company": "Uber",
        "source_type": "official_open_source",
        "scenario": "model_backend",
        "title": "CausalML",
        "url": "https://github.com/uber/causalml",
        "lesson": "CausalML provides uplift and CATE algorithms behind a common interface for experimental and observational data.",
        "interview_hook": "模型扩展要用 adapter/guard 管理第三方 uplift 后端。",
    },
    {
        "id": "SPOTIFY_IN_APP_MESSAGING_UPLIFT",
        "company": "Spotify",
        "source_type": "official_tech_blog",
        "scenario": "growth_in_app_messaging",
        "title": "Experimenting with Machine Learning to Target In-App Messaging",
        "url": "https://engineering.atspotify.com/2023/06/experimenting-with-machine-learning-to-target-in-app-messaging",
        "lesson": "In-app messaging can have mixed effects across users; Spotify used holdout data, CATE/uplift modeling, offline policy evaluation, and A/B testing before rollout.",
        "interview_hook": "增长触达要讲 holdout、HTE/CATE、多目标 uplift score、offline policy evaluation 和线上 retention 验证。",
    },
    {
        "id": "SPOTIFY_PERSONALIZATION_EXPERIMENT_SEPARATION",
        "company": "Spotify",
        "source_type": "official_tech_blog",
        "scenario": "personalization_experimentation",
        "title": "Why We Use Separate Tech Stacks for Personalization and Experimentation",
        "url": "https://engineering.atspotify.com/2026/1/why-we-use-separate-tech-stacks-for-personalization-and-experimentation",
        "lesson": "Personalization/bandit systems should be served by an ML stack and evaluated by a separate experimentation stack with guardrails and scalable experiment governance.",
        "interview_hook": "面试里可以把 Agent 的 scoring、policy、experiment gate 分层，解释为什么不能把推荐模型和实验平台揉成一个黑盒。",
    },
    {
        "id": "GOOGLE_GEO_EXPERIMENTS",
        "company": "Google",
        "source_type": "official_research_paper",
        "scenario": "ads_geo_incrementality",
        "title": "Measuring Ad Effectiveness Using Geo Experiments",
        "url": "https://research.google/pubs/pub38355",
        "lesson": "Geo experiments randomize non-overlapping regions when user-level randomization is infeasible for ad measurement.",
        "interview_hook": "广告 incrementality 经常需要 geo holdout，而不是只依赖平台归因。",
    },
    {
        "id": "GOOGLE_ADS_GEO_CONVERSION_LIFT",
        "company": "Google Ads",
        "source_type": "official_product_help",
        "scenario": "ads_geo_incrementality",
        "title": "Understand Conversion Lift based on geography measurement data",
        "url": "https://support.google.com/google-ads/answer/14102986?hl=en-EN",
        "lesson": "A business-facing lift report should translate experiments into incremental conversions and conversion value.",
        "interview_hook": "上线汇报要把 uplift 指标翻译成 incremental conversion value。",
    },
    {
        "id": "JD_AD_MARKETPLACE_EXPERIMENTS",
        "company": "JD.com",
        "source_type": "Stanford_GSB_working_paper",
        "scenario": "competitive_ads_marketplace",
        "title": "Parallel Experimentation in a Competitive Advertising Marketplace",
        "url": "https://www.gsb.stanford.edu/faculty-research/working-papers/parallel-experimentation-competitive-advertising-marketplace",
        "lesson": "Advertising marketplaces have interference across advertisers; experiments may need designs that account for competitive interactions.",
        "interview_hook": "广告平台不是独立用户实验那么简单，还要考虑竞价和广告主之间的干扰。",
    },
    {
        "id": "AMAZON_MARKETPLACE_EXPERIMENTS",
        "company": "Amazon",
        "source_type": "official_research_paper",
        "scenario": "marketplace_experimentation",
        "title": "Experimental design in marketplaces",
        "url": "https://www.amazon.science/publications/experimental-design-in-marketplaces",
        "lesson": "Marketplace RCTs can violate SUTVA because units interact; multiple randomization layers are often needed.",
        "interview_hook": "平台型业务的 uplift 要讲 interference/spillover，而不只是个体独立假设。",
    },
    {
        "id": "AIRBNB_INCREMENTAL_LTV",
        "company": "Airbnb",
        "source_type": "official_tech_blog",
        "scenario": "marketplace_incremental_ltv",
        "title": "How Airbnb Measures Listing Lifetime Value",
        "url": "https://airbnb.tech/ai-ml/how-airbnb-measures-listing-lifetime-value/",
        "lesson": "Marketplace value should separate baseline LTV, incremental LTV, and marketing-induced incremental LTV while accounting for cannibalization and delayed value realization.",
        "interview_hook": "发券/补贴/供给增长要讲 incremental LTV、cannibalization、长期窗口和业务收益校正，而不是只看短期转化。",
    },
    {
        "id": "MERCARI_COUPON_METRIC",
        "company": "Mercari",
        "source_type": "JSAI_2020_paper",
        "scenario": "coupon_roi_metric",
        "title": "A new evaluation metric and issue of uplift modeling for coupon marketing",
        "url": "https://cir.nii.ac.jp/crid/1390566775142606208",
        "lesson": "Qini does not always align with coupon cost-effectiveness; coupon marketing needs cost-aware metrics.",
        "interview_hook": "发券场景不能只说 QINI，要把券成本、毛利和 sure things 浪费放进评估。",
    },
    {
        "id": "VALUE_DRIVEN_UPLIFT_METRICS",
        "company": "Decision Support Systems",
        "source_type": "journal_paper",
        "scenario": "profit_aware_evaluation",
        "title": "Uplift modeling with value-driven evaluation metrics",
        "url": "https://doi.org/10.1016/j.dss.2021.113648",
        "lesson": "ITE-based targeting is not always consistent with profit maximization; evaluation should integrate treatment effect and expected business value.",
        "interview_hook": "工业面试里要说清楚：最高 uplift 不一定最高利润，要结合用户价值和处理成本。",
    },
    {
        "id": "GRFLIFT_GMV",
        "company": "Applied Intelligence",
        "source_type": "journal_paper",
        "scenario": "multi_treatment_gmv",
        "title": "GRFlift: uplift modeling for multi-treatment within GMV constraints",
        "url": "https://doi.org/10.1007/s10489-022-03769-w",
        "lesson": "Commercial uplift can use GMV/profit-aware splitting and multi-treatment assignment rather than binary conversion-only optimization.",
        "interview_hook": "多券面额/权益包场景可以把 GMV 和 treatment cost 直接放进目标。",
    },
    {
        "id": "MULTI_TREATMENT_REVENUE_UPLIFT",
        "company": "Academic / revenue uplift",
        "source_type": "arXiv_paper",
        "scenario": "multi_treatment_revenue",
        "title": "Interpretable Multiple Treatment Revenue Uplift Modeling",
        "url": "https://arxiv.org/abs/2101.03336",
        "lesson": "Revenue uplift with multiple treatments should optimize value, not just binary response uplift.",
        "interview_hook": "工业策略常常是多个券面额/权益包/触达渠道的多 treatment revenue uplift。",
    },
    {
        "id": "KUAISHOU_CDUM",
        "company": "Kuaishou",
        "source_type": "arXiv_industrial_recommendation",
        "scenario": "video_recommendation_uplift",
        "title": "Coarse-to-fine Dynamic Uplift Modeling for Real-time Video Recommendation",
        "url": "https://arxiv.org/abs/2410.16755",
        "lesson": "Uplift can be used beyond marketing, e.g. choosing recommendation treatments such as video-duration distribution in real-time feeds.",
        "interview_hook": "推荐干预的 treatment 可以是内容分布、时长或曝光策略，不只是发券。",
    },
    {
        "id": "KUAISHOU_HMUM",
        "company": "Kuaishou",
        "source_type": "arXiv_industrial_recommendation",
        "scenario": "multi_treatment_recommendation",
        "title": "Heterogeneous Multi-treatment Uplift Modeling for Trade-off Optimization in Short-Video Recommendation",
        "url": "https://arxiv.org/abs/2511.18997",
        "lesson": "Recommendation uplift can be a heterogeneous multi-treatment problem where strategies trade off watch time, views, and other business responses.",
        "interview_hook": "推荐场景可以从二元 uplift 扩展到多策略、多响应权衡和在线动态决策。",
    },
    {
        "id": "EFIN_ONLINE_MARKETING",
        "company": "Online marketing research",
        "source_type": "arXiv_paper",
        "scenario": "feature_interaction_uplift",
        "title": "Explicit Feature Interaction-aware Uplift Network for Online Marketing",
        "url": "https://arxiv.org/abs/2306.00315",
        "lesson": "Feature interactions can be first-class modeling targets in online-marketing uplift networks.",
        "interview_hook": "当面试官问 DeepUplift 里 EFIN/特征交互模型为什么有意义，可以用在线营销交互效应来解释。",
    },
    {
        "id": "BUDGET_CONSTRAINED_CAUSAL_BANDITS",
        "company": "Causal bandits research",
        "source_type": "arXiv_paper",
        "scenario": "budget_constrained_targeting",
        "title": "Budget-Constrained Causal Bandits: Bridging Uplift Modeling and Sequential Decision-Making",
        "url": "https://arxiv.org/abs/2604.26169",
        "lesson": "Budget-constrained treatment allocation connects uplift modeling with sequential decision-making and causal bandits.",
        "interview_hook": "预算受限投放不仅是离线排序，还可以演进到带预算约束的 sequential causal decision。",
    },
    {
        "id": "UNIMVT_COUPON_CTR_UPLIFT",
        "company": "Coupon marketing research",
        "source_type": "arXiv_paper",
        "scenario": "multi_valued_coupon_ctr_uplift",
        "title": "Jointly Optimizing Debiased CTR and Uplift for Coupons Marketing: A Unified Causal Framework",
        "url": "https://arxiv.org/abs/2602.12972",
        "lesson": "Coupon interventions can bias CTR prediction; multi-valued coupon treatments need debiased base CTR and intensity-response uplift modeling together.",
        "interview_hook": "发券会影响 CTR/CVR 估计本身，面试里可以讲 debiased CTR + uplift 的统一建模。",
    },
    {
        "id": "HIERARCHICAL_CAUSAL_LIFT_JOURNEYS",
        "company": "Digital travel platform research",
        "source_type": "arXiv_paper",
        "scenario": "overlapping_customer_journeys",
        "title": "Hierarchical Causal Uplift Modeling in Overlapping Customer Journeys",
        "url": "https://arxiv.org/abs/2604.24533",
        "lesson": "When users are exposed to multiple marketing journeys, standard single-journey lift can underestimate pure effects and miss synergy or cannibalization between journeys.",
        "interview_hook": "CRM/增长面试可以讲多旅程重叠、纯增量 vs 全局增量、协同/蚕食和 Monte Carlo 不确定性。",
    },
    {
        "id": "ORTHOGONAL_COMBINATORIAL_UPLIFT",
        "company": "Combinatorial treatment research",
        "source_type": "arXiv_paper",
        "scenario": "combinatorial_treatment_uplift",
        "title": "Orthogonal Uplift Learning with Permutation-Invariant Representations for Combinatorial Treatments",
        "url": "https://arxiv.org/abs/2602.19851",
        "lesson": "Real interventions can be combinatorial policies; permutation-invariant treatment representations plus orthogonalization can improve robustness in long-tail policy regimes.",
        "interview_hook": "多个触达动作组合不是普通多分类 treatment，可以讲 combinatorial treatment representation。",
    },
    {
        "id": "CONTINUOUS_TREATMENT_PTO",
        "company": "European Journal of Operational Research",
        "source_type": "EJOR_2026_paper",
        "scenario": "continuous_treatment_optimization",
        "title": "Uplift modeling with continuous treatments: A predict-then-optimize approach",
        "url": "https://arxiv.org/abs/2412.09232",
        "lesson": "Continuous-dose uplift should estimate conditional average dose responses and then solve a constrained dose-allocation problem.",
        "interview_hook": "券面额、折扣率、出价和触达频次不是二元 treatment，可以讲 CADR + optimization。",
    },
    {
        "id": "PMLR_UPLIFT_REVIEW",
        "company": "PMLR",
        "source_type": "review_paper",
        "scenario": "model_foundation",
        "title": "Causal Inference and Uplift Modelling: A Review of the Literature",
        "url": "https://proceedings.mlr.press/v67/gutierrez17a.html",
        "lesson": "Uplift modeling bridges causal inference and machine learning; common families include two-model, class transformation and direct modeling.",
        "interview_hook": "面试基础题要能把 uplift 讲回 potential outcome 和 treatment effect。",
    },
    {
        "id": "CAUSAL_COPILOT_LLM_AGENT",
        "company": "Causal-Copilot research",
        "source_type": "arXiv_paper",
        "scenario": "llm_causal_copilot",
        "title": "Causal-Copilot: An Autonomous Causal Analysis Agent",
        "url": "https://arxiv.org/abs/2504.13263",
        "lesson": "LLM agents can orchestrate causal discovery, causal inference, algorithm selection, hyperparameter optimization, interpretation, and actionable insight generation.",
        "interview_hook": "LLM 不替代因果估计器，而是把诊断、模型选择、解释和证据生成编排成可交互工作流。",
    },
    {
        "id": "CATE_B_LLM_COPILOT",
        "company": "CATE-B research",
        "source_type": "arXiv_paper",
        "scenario": "llm_treatment_effect_copilot",
        "title": "Facilitating the Adoption of Causal Inference Methods Through LLM-Empowered Co-Pilot",
        "url": "https://arxiv.org/abs/2508.10581",
        "lesson": "A treatment-effect co-pilot can help construct causal structure, orient edges, identify adjustment sets, and recommend estimation methods.",
        "interview_hook": "Agent 的关键价值是把 domain question 转成 treatment、outcome、confounders、adjustment set 和 estimator，而不是只聊天。",
    },
    {
        "id": "NAACL_LLM_CAUSAL_SURVEY",
        "company": "NAACL Findings",
        "source_type": "review_paper",
        "scenario": "llm_causal_survey",
        "title": "Causal Inference with Large Language Model: A Survey",
        "url": "https://aclanthology.org/2025.findings-naacl.327/",
        "lesson": "LLMs can support causal reasoning workflows, but hallucination and formal identification risks mean answers need source grounding and statistical validation.",
        "interview_hook": "LLM causal agent 必须有 evidence card、source quality audit 和 regression gate，不能靠生成式文本自证。",
    },
    {
        "id": "STRUCTURED_TEXT_TREATMENTS",
        "company": "NeurIPS structured treatment research",
        "source_type": "arXiv_paper",
        "scenario": "text_creative_treatment",
        "title": "Causal Effect Inference for Structured Treatments",
        "url": "https://arxiv.org/abs/2106.01939",
        "lesson": "Texts, images and graphs can be treated as structured treatments by representing interventions in embedding space and estimating CATE over treatment pairs.",
        "interview_hook": "广告文案、push 内容、优惠券描述和推荐解释可以是 text treatment，不只是二元发不发。",
    },
    {
        "id": "LATENT_TEXTUAL_TREATMENTS",
        "company": "Text treatment research",
        "source_type": "arXiv_paper",
        "scenario": "llm_generated_text_treatment",
        "title": "Causal Effect Estimation with Latent Textual Treatments",
        "url": "https://arxiv.org/abs/2602.15730",
        "lesson": "LLM-generated text interventions need controlled variation and causal estimation over latent textual features rather than naive prompt winner selection.",
        "interview_hook": "LLM 生成文案要用实验和文本 treatment 表示评估增量，不能只按离线偏好打分挑 prompt。",
    },
    {
        "id": "LLM_BANDIT_ROUTING",
        "company": "LLM routing research",
        "source_type": "arXiv_paper",
        "scenario": "llm_routing_cost_quality",
        "title": "LLM Bandit: Cost-Efficient LLM Generation via Preference-Conditioned Dynamic Routing",
        "url": "https://arxiv.org/abs/2502.02743",
        "lesson": "Per-query LLM selection can be framed as a multi-armed bandit to trade response quality against inference cost.",
        "interview_hook": "LLM routing 是一个 uplift/policy 问题：强模型相对弱模型的增量质量是否值得增量成本。",
    },
    {
        "id": "ADAPTIVE_LLM_ROUTING_BUDGET",
        "company": "LLM routing research",
        "source_type": "arXiv_paper",
        "scenario": "budget_constrained_llm_routing",
        "title": "Adaptive LLM Routing under Budget Constraints",
        "url": "https://arxiv.org/abs/2508.21141",
        "lesson": "LLM routing under budgets can be studied as a contextual bandit using embeddings and online cost policies.",
        "interview_hook": "从 uplift Top-K 可以自然升级到 budget-constrained contextual bandit，用预算约束决定哪些 query 升级强模型。",
    },
    {
        "id": "PARETOBANDIT_LLM_SERVING",
        "company": "ParetoBandit research",
        "source_type": "arXiv_paper",
        "scenario": "nonstationary_llm_serving",
        "title": "ParetoBandit: Budget-Paced Adaptive Routing for Non-Stationary LLM Serving",
        "url": "https://arxiv.org/abs/2604.00136",
        "lesson": "Production LLM serving needs budget pacing and adaptation to non-stationary quality/cost changes across model portfolios.",
        "interview_hook": "上线后模型价格、质量和流量分布会变，LLM routing 需要 drift monitoring、budget pacing 和回归守卫。",
    },
    {
        "id": "MICROSOFT_COST_AWARE_LLM_SELECTION",
        "company": "Microsoft Research",
        "source_type": "official_research_paper",
        "scenario": "cost_aware_llm_selection",
        "title": "One Head, Many Models: Cross-Attention Routing for Cost-Aware LLM Selection",
        "url": "https://www.microsoft.com/en-us/research/publication/one-head-many-models-cross-attention-routing-for-cost-aware-llm-selection-2/",
        "lesson": "Fine-grained query-model interaction modeling can predict both quality and cost for cost-aware LLM routing.",
        "interview_hook": "LLM routing 的 feature 可以是 query embedding、模型特征和 query-model interaction，而 uplift 框架负责判断增量收益。",
    },
    {
        "id": "FRUGALGPT_LLM_CASCADE",
        "company": "FrugalGPT research",
        "source_type": "arXiv_paper",
        "scenario": "llm_cascade_cost_quality",
        "title": "FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance",
        "url": "https://arxiv.org/abs/2305.05176",
        "lesson": "LLM cost-quality optimization can use cascaded model calls and per-query routing instead of always calling the largest model.",
        "interview_hook": "LLM serving 可以从固定强模型升级为 cascade/router，再用 uplift 判断哪些 query 真正需要升级。",
    },
    {
        "id": "ROUTELLM_OPEN_SOURCE_ROUTER",
        "company": "LMSYS RouteLLM",
        "source_type": "official_open_source",
        "scenario": "llm_router_open_source",
        "title": "RouteLLM: Learning to Route LLMs with Preference Data",
        "url": "https://github.com/lm-sys/RouteLLM",
        "lesson": "Open-source LLM routers learn when to send prompts to stronger models using preference-style supervision and cost-quality tradeoffs.",
        "interview_hook": "工程上可以把现成 LLM router 当候选 backend，但业务上线仍要用 holdout 和 policy value 验证增量质量。",
    },
    {
        "id": "BEST_ROUTE_TEST_TIME_COMPUTE",
        "company": "BEST-Route research",
        "source_type": "arXiv_paper",
        "scenario": "test_time_compute_llm_routing",
        "title": "BEST-Route: Adaptive LLM Routing with Test-Time Optimal Compute",
        "url": "https://arxiv.org/abs/2506.22716",
        "lesson": "Routing can be optimized under test-time compute constraints rather than only static model cost.",
        "interview_hook": "当强模型、RAG、工具调用、长思考都消耗 compute 时，routing policy 要优化增量质量、成本和时延的联合目标。",
    },
    {
        "id": "UNIVERSAL_MODEL_ROUTING",
        "company": "Universal Model Routing research",
        "source_type": "arXiv_paper",
        "scenario": "universal_llm_model_routing",
        "title": "Universal Model Routing for Efficient LLM Inference",
        "url": "https://arxiv.org/abs/2502.08773",
        "lesson": "A routing policy can generalize model selection across diverse tasks and model portfolios, but still needs online validation and drift monitoring.",
        "interview_hook": "模型池会持续变化，LLM routing 需要从离线 uplift 扩展到可迁移、可监控的 model portfolio policy。",
    },
]


BUSINESS_SCENARIOS = [
    {
        "id": "coupon",
        "name": "Coupon / subsidy allocation",
        "business_question": "给谁发券、发多少钱、如何避免补贴本来就会买的人？",
        "treatments": "send/no-send, coupon amount, discount rate, subsidy package",
        "primary_metrics": ["incremental gross profit", "net value", "redemption uplift", "GMV uplift", "ROI"],
        "risks": ["sure-thing waste", "negative uplift", "coupon arbitrage", "margin erosion", "user fatigue"],
        "model_families": ["T/X learner", "DR/R learner", "causal forest", "multi-treatment learner", "dose-response learner"],
        "ready_models": ["TLearnerLightGBM", "XLearnerLightGBM", "DRLearnerLightGBM", "RLearnerLightGBM", "MultiDRLearnerGBM", "DoseResponseGBM"],
        "evidence": [
            "ALI_ENTERTAINMENT_UPLIFT",
            "ALIBABA_ABTEST_HETEROGENEITY_XLEARNER",
            "MEITUAN_ECUP",
            "MERCARI_COUPON_METRIC",
            "VALUE_DRIVEN_UPLIFT_METRICS",
            "GRFLIFT_GMV",
            "MULTI_TREATMENT_REVENUE_UPLIFT",
            "EFIN_ONLINE_MARKETING",
            "UNIMVT_COUPON_CTR_UPLIFT",
        ],
        "interview_answer": "发券不能只看转化率，因为高转化用户可能不需要券。应该估计券带来的增量毛利，扣除券成本，并识别 persuadable / sure thing / sleeping dog。",
    },
    {
        "id": "ads",
        "name": "Ads / audience targeting / bidding",
        "business_question": "广告曝光、出价或预算是否带来真实增量，而不是归因口径下的转化？",
        "treatments": "ad exposure, bid multiplier, budget on/off, creative/channel",
        "primary_metrics": ["incremental conversions", "incremental conversion value", "iROAS", "lift", "holdout value"],
        "risks": ["attribution bias", "view-through inflation", "auction interference", "cross-channel spillover", "privacy-limited user logs"],
        "model_families": ["geo experiment", "ghost-ad holdout", "DR/R learner", "causal forest", "incrementality bidding"],
        "ready_models": ["DRLearnerLightGBM", "RLearnerLightGBM", "EconMLCausalForestDML", "EconMLDRLearner", "CausalForest"],
        "evidence": [
            "TENCENT_ADS_UPLIFT",
            "TENCENT_GHOST_AD",
            "DOORDASH_GHOST_ADS_INCREMENTALITY",
            "DOORDASH_SALES_LIFT_MEASUREMENT",
            "GOOGLE_GEO_EXPERIMENTS",
            "JD_AD_MARKETPLACE_EXPERIMENTS",
            "BUDGET_CONSTRAINED_CAUSAL_BANDITS",
        ],
        "interview_answer": "pCTR/pCVR 预测相关性，uplift/incrementality 估计展示广告相对不展示广告的因果增量；广告平台还要考虑竞价和跨广告主干扰。",
    },
    {
        "id": "growth",
        "name": "Growth / recall / push / SMS",
        "business_question": "触达谁能被召回，同时不造成退订、关闭通知或长期疲劳？",
        "treatments": "push frequency, SMS, call, message timing, notification creative",
        "primary_metrics": ["activation uplift", "retention uplift", "negative action uplift", "long-term value", "fatigue-adjusted ROI"],
        "risks": ["delayed feedback", "notification opt-out", "sleeping dogs", "frequency fatigue", "short-term metric gaming"],
        "model_families": ["DR/R learner", "dynamic uplift", "survival/delayed-feedback uplift", "multi-objective policy"],
        "ready_models": ["DRLearnerLightGBM", "RLearnerLightGBM", "PAVCalibratedDRLearnerLightGBM", "EconMLDRLearner"],
        "evidence": [
            "BYTEDANCE_DELAYED_UPLIFT",
            "SPOTIFY_IN_APP_MESSAGING_UPLIFT",
            "SPOTIFY_PERSONALIZATION_EXPERIMENT_SEPARATION",
            "UBER_CAUSAL_INFERENCE",
            "HIERARCHICAL_CAUSAL_LIFT_JOURNEYS",
            "PMLR_UPLIFT_REVIEW",
        ],
        "interview_answer": "增长触达要同时优化激活增量和负向体验增量，尤其要处理延迟反馈和频控，而不是把当天点击/打开当成最终收益。",
    },
    {
        "id": "crm",
        "name": "CRM lifecycle intervention",
        "business_question": "在用户生命周期里选择哪个触达动作，什么时候触达，是否保留长期 holdout？",
        "treatments": "email, push, coupon, phone call, lifecycle journey",
        "primary_metrics": ["retention uplift", "repeat purchase uplift", "LTV uplift", "contact cost", "fatigue"],
        "risks": ["channel cannibalization", "contact fatigue", "delayed LTV", "non-random assignment", "leakage"],
        "model_families": ["multi-treatment learner", "DR/R learner", "uplift tree/forest", "policy value"],
        "ready_models": ["MultiTLearnerGBM", "MultiDRLearnerGBM", "DRLearnerLightGBM", "SkLiftTwoModelsLightGBM"],
        "evidence": [
            "UBER_CAUSAL_INFERENCE",
            "SPOTIFY_IN_APP_MESSAGING_UPLIFT",
            "PMLR_UPLIFT_REVIEW",
            "MULTI_TREATMENT_REVENUE_UPLIFT",
            "HIERARCHICAL_CAUSAL_LIFT_JOURNEYS",
            "ORTHOGONAL_COMBINATORIAL_UPLIFT",
        ],
        "interview_answer": "CRM 不只是二元发不发，还要在多个渠道和多次触达之间做策略选择，所以需要 multi-treatment 和长期 holdout 思路。",
    },
    {
        "id": "recommendation",
        "name": "Recommendation intervention",
        "business_question": "推荐、重排或曝光策略是否真正提升用户价值，而不是强化已有偏好？",
        "treatments": "ranking exposure, content boost, creator boost, exploration policy",
        "primary_metrics": ["incremental engagement", "creator/customer value", "long-term retention", "negative feedback"],
        "risks": ["position bias", "network effects", "interference", "delayed retention", "feedback loops"],
        "model_families": ["causal forest", "DR/R learner", "dynamic uplift", "switchback/cluster experiments"],
        "ready_models": ["EconMLCausalForestDML", "DRLearnerLightGBM", "RLearnerLightGBM", "CausalForest"],
        "evidence": [
            "KUAISHOU_CDUM",
            "KUAISHOU_HMUM",
            "ALIBABA_EDGEREC_REQUEST_UPLIFT",
            "AMAZON_MARKETPLACE_EXPERIMENTS",
            "UBER_CAUSAL_INFERENCE",
            "BYTEDANCE_DELAYED_UPLIFT",
            "SPOTIFY_PERSONALIZATION_EXPERIMENT_SEPARATION",
        ],
        "interview_answer": "推荐干预要区分自然兴趣和被策略改变的增量行为，且常有位置偏差、干扰和反馈回路，不能只看离线排序 AUC。",
    },
    {
        "id": "marketplace",
        "name": "Marketplace pricing / incentive",
        "business_question": "补贴、价格或激励如何影响供需两侧，并避免 spillover？",
        "treatments": "driver incentive, buyer subsidy, surge/discount, marketplace fee",
        "primary_metrics": ["supply uplift", "demand uplift", "market balance", "gross profit", "spillover-adjusted value"],
        "risks": ["interference", "spillover", "budget constraints", "supply-demand feedback", "continuous treatment"],
        "model_families": ["multi-treatment learner", "dose-response", "causal forest", "marketplace experiment design"],
        "ready_models": ["MultiDRLearnerGBM", "DoseResponseGBM", "EconMLCausalForestDML", "DRLearnerLightGBM"],
        "evidence": [
            "DIDI_SUBSIDY_DCN",
            "ALIBABA_ABTEST_HETEROGENEITY_XLEARNER",
            "GRFLIFT_GMV",
            "CONTINUOUS_TREATMENT_PTO",
            "AIRBNB_INCREMENTAL_LTV",
            "AMAZON_MARKETPLACE_EXPERIMENTS",
            "JD_AD_MARKETPLACE_EXPERIMENTS",
        ],
        "interview_answer": "平台补贴不是独立样本问题，价格和激励会改变供需两侧，还可能影响邻近区域或竞争者，需要更强的实验设计和策略约束。",
    },
    {
        "id": "llm_routing",
        "name": "LLM routing / cost-quality escalation",
        "business_question": "哪些请求值得从便宜模型升级到更强模型，如何在质量、成本、延迟和预算之间做增量决策？",
        "treatments": "small model, strong model, tool call, retrieval, human review, prompt/template variant",
        "primary_metrics": ["incremental quality", "cost-adjusted quality", "latency", "budget burn", "fallback rate"],
        "risks": ["judge bias", "non-stationary model quality", "hidden latency cost", "budget overshoot", "hallucinated explanations"],
        "model_families": ["T/X/DR/R learner", "contextual bandit", "cost-aware router", "policy learning", "calibrated uplift ranking"],
        "ready_models": ["DRLearnerLightGBM", "RLearnerLightGBM", "PAVCalibratedDRLearnerLightGBM", "EconMLDRLearner", "CausalForest"],
        "evidence": [
            "LLM_BANDIT_ROUTING",
            "ADAPTIVE_LLM_ROUTING_BUDGET",
            "PARETOBANDIT_LLM_SERVING",
            "MICROSOFT_COST_AWARE_LLM_SELECTION",
            "FRUGALGPT_LLM_CASCADE",
            "ROUTELLM_OPEN_SOURCE_ROUTER",
            "BEST_ROUTE_TEST_TIME_COMPUTE",
            "UNIVERSAL_MODEL_ROUTING",
            "CAUSAL_COPILOT_LLM_AGENT",
            "NAACL_LLM_CAUSAL_SURVEY",
        ],
        "interview_answer": "LLM routing 可以讲成 uplift：treatment 是调用强模型，control 是便宜模型，目标是估计强模型带来的增量质量是否大于增量成本和延迟风险。",
    },
]


INDUSTRY_INTERVIEW_PROMPTS = [
    "发券为什么不能只看转化率？",
    "广告投放里 uplift 和 pCTR/pCVR 有什么区别？",
    "怎么识别 persuadable / sure thing / lost cause / sleeping dog？",
    "怎么把 uplift score 转成投放阈值和 ROI？",
    "怎么处理 treatment selection bias？",
    "怎么做 uplift online A/B test？",
    "工业界为什么常用 T/X/DR/R learner、causal forest、uplift tree、GBDT？",
    "多 treatment、连续 treatment、长期 treatment 怎么做？",
    "发券场景如何考虑成本、毛利、补贴套利、用户疲劳？",
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
    "Uplift 和 LLM 怎么结合？",
    "LLM routing 为什么是 uplift 问题？",
    "LLM 生成文案怎么做 uplift 实验？",
    "LLM causal agent 如何防止 hallucination？",
    "LLM routing 离线训练数据怎么采集？",
    "什么时候用 uplift，什么时候升级到 bandit？",
]


LLM_INTERVIEW_QA = [
    {
        "question": "Uplift 和 LLM 结合的核心架构是什么？",
        "answer": "LLM 做 causal copilot 和 workflow orchestrator，负责把业务问题转成 treatment/outcome/confounders/guardrails；因果估计仍由 DR/R/DML、causal forest、policy value 和 holdout 验证完成。",
        "engineering_design": "Agent answer -> source/evidence card -> deterministic diagnostics/model registry/evaluator -> policy simulator -> regression gate.",
        "evidence": "CAUSAL_COPILOT_LLM_AGENT, CATE_B_LLM_COPILOT, NAACL_LLM_CAUSAL_SURVEY",
        "interview_line": "我没有让 LLM 自证因果，而是让它编排可验证的 causal workflow。",
    },
    {
        "question": "LLM routing 为什么是 uplift，而不是普通分类？",
        "answer": "普通分类会预测哪个模型绝对分数高；uplift 关心强模型相对便宜模型对某个 query 的增量质量是否超过增量成本、延迟和预算风险。",
        "engineering_design": "control=small model, treatment=strong model/tool/RAG, outcome=judge score or task success, policy value=quality uplift * value - incremental cost - latency penalty.",
        "evidence": "LLM_BANDIT_ROUTING, MICROSOFT_COST_AWARE_LLM_SELECTION, ROUTELLM_OPEN_SOURCE_ROUTER",
        "interview_line": "不是所有 query 都值得上强模型，关键是识别 persuadable queries。",
    },
    {
        "question": "LLM routing 离线训练数据怎么采集？",
        "answer": "需要保留一小部分随机探索流量，让同类 query 在 cheap/strong/tool/RAG 等候选 action 间随机或分层随机，记录质量、成本、时延和失败原因。",
        "engineering_design": "randomized exploration -> query/task embeddings -> action features -> judge/human label -> DR/R learner -> calibrated Top-K router -> online holdout.",
        "evidence": "ADAPTIVE_LLM_ROUTING_BUDGET, FRUGALGPT_LLM_CASCADE, BEST_ROUTE_TEST_TIME_COMPUTE",
        "interview_line": "没有随机探索或 overlap，router 只是在复制历史路由策略，不能可靠估计升级强模型的增量收益。",
    },
    {
        "question": "什么时候用 uplift，什么时候升级到 bandit？",
        "answer": "流量、模型池、价格和任务分布稳定时先用离线 uplift 排序和固定阈值；当预算、质量、价格或模型候选持续变化时，再升级到 contextual bandit 和 budget pacing。",
        "engineering_design": "uplift baseline -> calibrated threshold -> holdout -> bandit with guardrails only after offline calibration and monitoring are stable.",
        "evidence": "BUDGET_CONSTRAINED_CAUSAL_BANDITS, PARETOBANDIT_LLM_SERVING, UNIVERSAL_MODEL_ROUTING",
        "interview_line": "我会先用可解释的 uplift 做离线 policy，再在生产稳定后用 bandit 做在线自适应。",
    },
    {
        "question": "LLM 生成文案怎么做 uplift 实验？",
        "answer": "把文案、prompt、offer 描述视作 structured/text treatment，用随机曝光和 no-message/control holdout 估计不同用户对不同文本的增量响应、退订和长期价值。",
        "engineering_design": "LLM generates candidates -> freeze prompt families -> randomize exposure -> embed text/action -> multi-treatment or text-treatment CATE -> policy value with fatigue/opt-out penalty.",
        "evidence": "STRUCTURED_TEXT_TREATMENTS, LATENT_TEXTUAL_TREATMENTS, SPOTIFY_IN_APP_MESSAGING_UPLIFT",
        "interview_line": "LLM 可以生成候选干预，但哪个文案真正有增量必须靠实验和 CATE 验证。",
    },
    {
        "question": "LLM causal agent 如何防止 hallucination？",
        "answer": "每个回答绑定本地知识库、外部来源、UI 页面和验证脚本；弱来源进 backlog；关键功能进入 smoke/full regression，避免只靠生成文本自证。",
        "engineering_design": "Agent Evidence Card, source quality audit, source adoption backlog, screenshot smoke, interview-material audit, full_regression_check.",
        "evidence": "NAACL_LLM_CAUSAL_SURVEY, INDUSTRY_SOURCE_QUALITY_AUDIT, DEEPUplift_AGENT_EVIDENCE_INDEX",
        "interview_line": "Agent 可以解释，但真正让它可信的是 evidence card 和 regression gate。",
    },
]


POLICY_SCENARIO_TEMPLATES = [
    {
        "scenario": "Coupon",
        "conversion_value_hint": "gross_margin_per_order",
        "contact_cost_hint": "coupon_face_value * redemption_probability + channel_cost",
        "extra_guardrails": "cap per-user subsidy, exclude sure things, track redemption abuse, keep long-term holdout",
    },
    {
        "scenario": "Ads",
        "conversion_value_hint": "incremental conversion value or profit",
        "contact_cost_hint": "incremental ad spend or CPM/CPC-implied exposure cost",
        "extra_guardrails": "geo/user holdout, attribution sanity check, iROAS lower bound, cross-channel spillover",
    },
    {
        "scenario": "Growth Push",
        "conversion_value_hint": "activation or LTV proxy",
        "contact_cost_hint": "message cost + fatigue penalty",
        "extra_guardrails": "opt-out uplift, frequency cap, delayed feedback window, negative uplift suppression",
    },
    {
        "scenario": "Marketplace Subsidy",
        "conversion_value_hint": "supply-demand balance value or gross profit",
        "contact_cost_hint": "subsidy payout + platform incentive cost",
        "extra_guardrails": "budget cap, city/region spillover, marketplace balance, continuous treatment response",
    },
    {
        "scenario": "LLM Routing",
        "conversion_value_hint": "quality point value, task success value, or saved human-review value",
        "contact_cost_hint": "strong_model_cost - small_model_cost + latency penalty",
        "extra_guardrails": "budget pacing, latency SLO, judge calibration, hallucination/evidence guard, drift monitoring",
    },
]


def industry_references() -> list[dict[str, Any]]:
    return [dict(row) for row in INDUSTRY_REFERENCES]


def business_scenarios() -> list[dict[str, Any]]:
    return [dict(row) for row in BUSINESS_SCENARIOS]


def policy_templates() -> list[dict[str, str]]:
    return [dict(row) for row in POLICY_SCENARIO_TEMPLATES]


def llm_interview_qa() -> list[dict[str, str]]:
    return [dict(row) for row in LLM_INTERVIEW_QA]


def scenario_by_id(scenario_id: str) -> dict[str, Any] | None:
    normalized = scenario_id.strip().lower()
    for scenario in BUSINESS_SCENARIOS:
        if scenario["id"] == normalized:
            return dict(scenario)
    return None


def scenario_model_rows() -> list[dict[str, str]]:
    rows = []
    reference_lookup = {ref["id"]: ref for ref in INDUSTRY_REFERENCES}
    for scenario in BUSINESS_SCENARIOS:
        rows.append(
            {
                "scenario": scenario["name"],
                "business_question": scenario["business_question"],
                "recommended_models": ", ".join(scenario["ready_models"]),
                "model_families": ", ".join(scenario["model_families"]),
                "primary_metrics": ", ".join(scenario["primary_metrics"]),
                "major_risks": ", ".join(scenario["risks"]),
                "evidence": ", ".join(reference_lookup[ref_id]["company"] for ref_id in scenario["evidence"] if ref_id in reference_lookup),
            }
        )
    return rows


def _has_any(text: str, keywords: list[str]) -> bool:
    return any(keyword.lower() in text for keyword in keywords)


def _ctx(context: Mapping[str, Any], key: str, default: Any = "NA") -> Any:
    value = context.get(key, default)
    return default if value is None else value


def _reference_lines(ref_ids: list[str]) -> str:
    refs = [ref for ref in INDUSTRY_REFERENCES if ref["id"] in ref_ids]
    return "\n".join(f"- `{ref['company']}` {ref['title']}: {ref['url']}" for ref in refs)


def _scenario_answer(scenario_id: str, context: Mapping[str, Any]) -> str:
    scenario = scenario_by_id(scenario_id)
    if not scenario:
        return ""
    refs = [ref for ref in INDUSTRY_REFERENCES if ref["id"] in scenario["evidence"]]
    refs_text = "\n".join(f"- `{ref['company']}` {ref['title']}: {ref['url']}" for ref in refs)
    return (
        f"### 工业场景：{scenario['name']}\n\n"
        f"**业务问题**：{scenario['business_question']}\n\n"
        f"**为什么不能用普通预测**：普通 pCTR/pCVR/转化率模型更容易找到自然会转化的人，"
        f"但这个场景真正要优化的是 treatment 相对 no-treatment 的增量和净收益。\n\n"
        f"**推荐模型**：{', '.join(scenario['ready_models'])}。\n\n"
        f"**模型家族**：{', '.join(scenario['model_families'])}。\n\n"
        f"**评估指标**：{', '.join(scenario['primary_metrics'])}。\n\n"
        "**ROI落地**：按 uplift score 排序做 Top-K policy curve，用 `uplift * business_value - treatment_cost` 找预算/触达约束下的最优阈值。\n\n"
        f"**主要风险**：{', '.join(scenario['risks'])}。\n\n"
        f"**面试讲法**：{scenario['interview_answer']}\n\n"
        f"**本地证据**：`docs/UPLIFT_BUSINESS_SCENARIOS.md`、`docs/UPLIFT_COUPON_ADS_GROWTH_PLAYBOOK.md`、Story 页 `Industrial Case Studies / Business Scenarios / Interview Hard Questions`。\n\n"
        f"**外部参考**：\n{refs_text}\n\n"
        f"当前页面上下文：任务 `{_ctx(context, 'task')}`，模型 `{_ctx(context, 'model_name')}`，"
        f"treatment `{_ctx(context, 'treatment_col')}`，outcome `{_ctx(context, 'outcome_col')}`。"
    )


def industry_agent_reply(prompt: str, context: Mapping[str, Any] | None = None) -> str | None:
    text = prompt.strip().lower()
    context = context or {}
    if not text:
        return None

    if _has_any(text, ["persuadable", "sure thing", "lost cause", "sleeping dog", "四象限", "四类人群"]):
        return (
            "### 工业面试题：怎么识别 persuadable / sure thing / lost cause / sleeping dog？\n\n"
            "**核心回答**：用 `y1_pred` 和 `y0_pred` 同时解释人群，而不是只看 `uplift_score`。"
            "`persuadable` 是 y1 高、y0 低；`sure thing` 是 y1/y0 都高；`lost cause` 是 y1/y0 都低；`sleeping dog` 是 y1 低于 y0。\n\n"
            "**业务落地**：发券优先投 persuadable，限制 sure thing 补贴浪费，过滤 sleeping dog，lost cause 通常不投或进入低成本实验。\n\n"
            "**本地证据**：`docs/UPLIFT_COUPON_ADS_GROWTH_PLAYBOOK.md`、`docs/UPLIFT_BUSINESS_SCENARIOS.md`、Policy 页 ROI 模板。\n\n"
            f"当前页面上下文：任务 `{_ctx(context, 'task')}`，模型 `{_ctx(context, 'model_name')}`。"
        )

    if _has_any(text, ["online a/b", "a/b test", "ab test", "线上实验", "在线实验", "holdout", "灰度", "uplift online"]):
        return (
            "### 工业面试题：怎么做 uplift online A/B test？\n\n"
            "**核心回答**：离线 uplift 只负责生成候选策略和阈值，线上要固定策略、保留随机 holdout，并按增量转化/增量毛利/ROI 验证。\n\n"
            "**实验设计**：用户级随机优先；广告或平台干扰明显时使用 geo、cluster、switchback 或长期 holdout。实验前固定 Top-K 阈值、成本收益口径、预算上限和 guardrail 指标。\n\n"
            "**验证指标**：incremental conversions、incremental gross profit、iROAS/ROI、opt-out/疲劳、长期 retention。不要上线后再按结果挑阈值。\n\n"
            "**本地证据**：`docs/UPLIFT_ONLINE_EXPERIMENT_AND_INCREMENTALITY.md`、Policy 页、Story 页 `Industrial Case Studies`。\n\n"
            f"当前页面上下文：任务 `{_ctx(context, 'task')}`，模型 `{_ctx(context, 'model_name')}`。"
        )

    if _has_any(text, ["treatment selection bias", "selection bias", "选择偏差", "分配偏差", "observational assignment"]):
        return (
            "### 工业面试题：怎么处理 treatment selection bias？\n\n"
            "**核心回答**：先承认观测日志里的 treatment 往往不是随机分配，再用 propensity、overlap、feature balance 和 refutation checks 判断可比性。\n\n"
            "**模型策略**：随机实验可先用 T/X learner；观测数据优先 DR/R/DML、Domain Adaptation、CFR/representation balancing 和 causal forest，并用 bootstrap/sensitivity 标注不确定性。\n\n"
            "**业务策略**：如果 overlap 很差，模型不应该直接上线，而应缩到 common support、补做随机 holdout，或重新设计人群包和触达实验。\n\n"
            "**本地证据**：Diagnostics 页 `Selection Bias / SSB Proxy`、`docs/UPLIFT_MODEL_SELECTION_PLAYBOOK.md`、`docs/UPLIFT_ONLINE_EXPERIMENT_AND_INCREMENTALITY.md`。\n\n"
            f"当前页面上下文：任务 `{_ctx(context, 'task')}`，模型 `{_ctx(context, 'model_name')}`。"
        )

    if _has_any(text, ["为什么常用", "t/x", "t learner", "x learner", "dr/r", "dr learner", "r learner", "causal forest", "uplift tree", "gbdt", "工业界为什么"]):
        return (
            "### 工业面试题：为什么常用 T/X/DR/R learner、causal forest、uplift tree、GBDT？\n\n"
            "**核心回答**：工业表格数据常见高维稀疏、非线性、样本不均衡和 treatment selection bias，所以需要从简单基线到稳健估计逐步升级。\n\n"
            "- `T/S learner`：快速、可解释，适合随机实验 baseline。\n"
            "- `X learner`：适合 treatment/control 样本不均衡。\n"
            "- `DR/R learner`：结合 outcome/propensity 或残差化，适合观测数据偏差。\n"
            "- `causal forest / uplift tree`：适合解释异质性和分群。\n"
            "- `GBDT/LightGBM`：工业表格数据强基线，训练快、可调、适合大规模 Compare。\n\n"
            "**本地证据**：Models 页 `Business Scenario Model Recommendation`，`docs/UPLIFT_MODEL_SELECTION_PLAYBOOK.md`。\n\n"
            f"当前页面上下文：任务 `{_ctx(context, 'task')}`，模型 `{_ctx(context, 'model_name')}`。"
        )

    if _has_any(text, ["羊毛党", "套利", "abuse", "fraud", "补贴套利", "券成本", "毛利"]):
        refs_text = _reference_lines(["MEITUAN_ECUP", "MERCARI_COUPON_METRIC", "VALUE_DRIVEN_UPLIFT_METRICS", "GRFLIFT_GMV"])
        return (
            "### 工业面试题：发券如何处理补贴套利和羊毛党？\n\n"
            "**核心回答**：发券策略不能只按 uplift 排序，还要把毛利、券成本、核销概率、套利风险和用户疲劳放进 policy value。"
            "上线时我会用 `uplift * gross_margin - expected_coupon_cost - channel_cost - abuse/fatigue_penalty` 计算净价值。\n\n"
            "**工程做法**：\n"
            "- 对 sure thing 降权，避免补贴本来就会买的人。\n"
            "- 对 sleeping dog/negative uplift 过滤，避免刺激反效果。\n"
            "- 加用户级、设备级、支付/地址/账号关系的风控特征，但这些特征必须是 pre-treatment。\n"
            "- 用 budget cap、单用户补贴上限、冷却期和长期 holdout 监控 subsidy dependency。\n\n"
            "**本地证据**：Policy 页 `Scenario ROI Helper`，`docs/UPLIFT_COUPON_ADS_GROWTH_PLAYBOOK.md`，`docs/UPLIFT_SCENARIO_MODEL_MATRIX.md`。\n\n"
            f"**外部参考**：\n{refs_text}\n\n"
            f"当前页面上下文：任务 `{_ctx(context, 'task')}`，模型 `{_ctx(context, 'model_name')}`。"
        )

    if _has_any(text, ["预算", "budget", "budget cap", "有限预算", "约束", "投放比例", "top-k"]):
        refs_text = _reference_lines(["BUDGET_CONSTRAINED_CAUSAL_BANDITS", "GOOGLE_ADS_GEO_CONVERSION_LIFT", "VALUE_DRIVEN_UPLIFT_METRICS"])
        return (
            "### 工业面试题：预算有限时 uplift 阈值怎么定？\n\n"
            "**核心回答**：先按 uplift score 排序，再按 `uplift * value - cost` 画 Top-K policy curve，"
            "在预算、最大触达数、频控和风险约束下选择净价值最高的 K。\n\n"
            "**落地步骤**：\n"
            "1. 设定单次转化价值、触达成本、券成本或广告曝光成本。\n"
            "2. 计算每个 Top-K 桶的 predicted/observed net value。\n"
            "3. 用 budget cap 和 max contacts 过滤不可行阈值。\n"
            "4. 对推荐阈值做 bootstrap CI、calibration 和 holdout/A/B 验证。\n"
            "5. 如果预算在线动态变化，可以把离线 uplift 排序升级为 budget-constrained causal bandit。\n\n"
            "**本地证据**：Policy 页 `Scenario ROI Helper` 与 constrained recommendation，`docs/UPLIFT_ONLINE_EXPERIMENT_AND_INCREMENTALITY.md`。\n\n"
            f"**外部参考**：\n{refs_text}\n\n"
            f"当前页面上下文：任务 `{_ctx(context, 'task')}`，模型 `{_ctx(context, 'model_name')}`。"
        )

    if _has_any(text, ["interference", "spillover", "sutva", "干扰", "溢出", "供需两侧", "marketplace"]):
        refs_text = _reference_lines(["DIDI_SUBSIDY_DCN", "AMAZON_MARKETPLACE_EXPERIMENTS", "JD_AD_MARKETPLACE_EXPERIMENTS", "CONTINUOUS_TREATMENT_PTO"])
        return (
            "### 工业面试题：Marketplace 干扰和 spillover 怎么处理？\n\n"
            "**核心回答**：平台补贴/价格/广告竞价可能改变供需两侧和竞争者行为，SUTVA 很容易被破坏。"
            "所以不能只把每个用户当独立样本做普通 Top-K。\n\n"
            "**实验设计**：用户级随机化适合低干扰场景；如果存在城市、门店、司机、广告主或供需网络干扰，"
            "更适合 geo/cluster/switchback、分层随机、长期 holdout 或 marketplace-specific experiment design。\n\n"
            "**建模策略**：多券面额/多补贴包用 multi-treatment learner；价格、折扣、出价、频次用 dose-response；"
            "评估时看 spillover-adjusted value、供需平衡和预算约束，而不只是个人转化 uplift。\n\n"
            "**本地证据**：Models 页 `Scenario Tag = Marketplace pricing / incentive`，`docs/UPLIFT_SCENARIO_MODEL_MATRIX.md`。\n\n"
            f"**外部参考**：\n{refs_text}\n\n"
            f"当前页面上下文：任务 `{_ctx(context, 'task')}`，模型 `{_ctx(context, 'model_name')}`。"
        )

    if _has_any(text, ["频控", "疲劳", "fatigue", "opt-out", "退订", "关闭通知", "push"]):
        refs_text = _reference_lines(["BYTEDANCE_DELAYED_UPLIFT", "SPOTIFY_IN_APP_MESSAGING_UPLIFT", "UBER_CAUSAL_INFERENCE", "PMLR_UPLIFT_REVIEW"])
        return (
            "### 工业面试题：Push 频控和用户疲劳怎么建模？\n\n"
            "**核心回答**：增长触达不是只最大化当日打开/转化，而是要估计正向激活增量和负向疲劳增量。"
            "Policy value 里应加入 message cost、fatigue penalty、opt-out risk 和长期留存窗口。\n\n"
            "**建模做法**：\n"
            "- 用 delayed-feedback uplift 或更长标签窗口处理转化延迟。\n"
            "- 把频次、时间、渠道作为 multi-treatment 或 dose-response 问题。\n"
            "- 对 sleeping dogs 和高 opt-out uplift 人群做 suppress。\n"
            "- 线上用长期 holdout 监控通知关闭、退订和 retention。\n\n"
            "**本地证据**：Policy 页 `Growth Push` ROI helper，`docs/UPLIFT_COUPON_ADS_GROWTH_PLAYBOOK.md`。\n\n"
            f"**外部参考**：\n{refs_text}\n\n"
            f"当前页面上下文：任务 `{_ctx(context, 'task')}`，模型 `{_ctx(context, 'model_name')}`。"
        )

    if _has_any(text, ["spotify", "in-app messaging", "in app messaging", "站内信", "应用内消息", "消息触达"]):
        refs_text = _reference_lines(["SPOTIFY_IN_APP_MESSAGING_UPLIFT", "SPOTIFY_PERSONALIZATION_EXPERIMENT_SEPARATION"])
        return (
            "### 工业案例：Spotify in-app messaging uplift\n\n"
            "**怎么讲**：Spotify 的站内消息不是普通点击率排序问题，因为同一条消息可能对不同用户产生正向、无效或负向影响。"
            "工程上应从 holdout/RCT 数据估计 CATE/uplift，再用 offline policy evaluation 筛策略，最后用线上实验验证 retention、engagement 和 guardrail。\n\n"
            "**Agent 落地**：DeepUplift Agent 对应把 `Diagnostics -> Model Compare -> Policy -> Evidence` 串起来：先诊断 overlap/selection bias，"
            "再比较 DR/R learner、causal forest、delayed-feedback uplift，最后把 uplift score 转成频控、Top-K 阈值和长期 holdout 方案。\n\n"
            "**面试亮点**：这能解释为什么增长触达要抑制 sleeping dogs，为什么个性化 scoring 和 experimentation/governance 要分层，以及为什么离线 QINI 要接 online incrementality。\n\n"
            f"**外部参考**：\n{refs_text}\n\n"
            "**本地证据**：Story 页 `Fresh Industrial Additions`、Knowledge 页 `Industrial Knowledge Map`、"
            "`docs/UPLIFT_BUSINESS_SCENARIOS.md`。"
        )

    if _has_any(text, ["airbnb", "ltv", "lifetime value", "长期价值", "cannibalization", "蚕食", "增量ltv"]):
        refs_text = _reference_lines(["AIRBNB_INCREMENTAL_LTV", "AMAZON_MARKETPLACE_EXPERIMENTS", "VALUE_DRIVEN_UPLIFT_METRICS"])
        return (
            "### 工业案例：Airbnb incremental LTV / marketplace value\n\n"
            "**怎么讲**：平台补贴或供给增长不能只看短期成交 uplift，还要分清 baseline LTV、incremental LTV 和 marketing-induced incremental LTV。"
            "如果忽略 cannibalization、延迟价值和供需干扰，离线 uplift 排名可能把预算投给短期看起来有效但长期净价值不高的人群。\n\n"
            "**Agent 落地**：Policy 页的 ROI helper 可以把 `conversion_value` 扩展为 LTV/gross profit，把 `contact_cost` 扩展为补贴、渠道成本、疲劳和风控惩罚；"
            "Marketplace 场景推荐 multi-treatment、dose-response、causal forest，并要求长期 holdout 或 cluster/geo 实验兜底。\n\n"
            "**面试亮点**：这类题可以从 offline CATE 过渡到 online incrementality、SUTVA/interference 和业务利润口径，说明项目不是只会跑模型。\n\n"
            f"**外部参考**：\n{refs_text}\n\n"
            "**本地证据**：Policy 页 `Scenario ROI Helper`、Story 页 `Industrial Case Studies`、"
            "`docs/UPLIFT_COUPON_ADS_GROWTH_PLAYBOOK.md`。"
        )

    if _has_any(text, ["阿里", "alibaba", "x-learner", "xlearner", "异质性", "ab实验", "ab 实验", "shap"]):
        refs_text = _reference_lines(["ALIBABA_ABTEST_HETEROGENEITY_XLEARNER", "MEITUAN_ECUP", "DIDI_SUBSIDY_DCN"])
        return (
            "### 工业案例：阿里 AB 实验异质性 / X-Learner\n\n"
            "**怎么讲**：随机分流后，不同人群对补贴/策略的增量效果可能完全不同，所以不能只看整体 ATE。"
            "可以用 X-Learner 从实验数据估计 ITE/CATE，再用 SHAP 或分群钻取解释哪些特征段收益高、哪些段应该降本或替换策略。\n\n"
            "**Agent 落地**：Diagnostics 先确认 treatment/control 可比，Models 页可以按 Coupon/Marketplace 场景选择 `XLearnerLightGBM`、`DRLearnerLightGBM`、"
            "`MultiDRLearnerGBM`，Policy 页用 Scenario ROI Lab 把分群 uplift 转成补贴 ROI。\n\n"
            "**面试亮点**：这能把 AB 实验、异质性分析、模型解释和业务策略优化串起来，适合讲发券、物流补贴、供需激励和降本增效。\n\n"
            f"**外部参考**：\n{refs_text}\n\n"
            "**本地证据**：Knowledge 页 `Industrial Knowledge Map / Source Adoption Drilldown`，Story 页 `Fresh Industrial Additions`，"
            "`docs/UPLIFT_BUSINESS_SCENARIOS.md`。"
        )

    if _has_any(text, ["edgerec", "请求价值", "request uplift", "端上推荐", "重排请求", "追加请求"]):
        refs_text = _reference_lines(["ALIBABA_EDGEREC_REQUEST_UPLIFT", "KUAISHOU_CDUM", "KUAISHOU_HMUM"])
        return (
            "### 工业案例：EdgeRec 请求价值增益模型\n\n"
            "**怎么讲**：推荐系统里的 treatment 不一定是发券，也可以是是否追加一次重排/端上推荐请求。"
            "模型不只预测购买概率，而是估计 extra request 相对 no-extra-request 的购买率增益，同时还要考虑 QPS、延迟和系统成本。\n\n"
            "**Agent 落地**：Recommendation 场景会推荐 causal forest、DR/R learner 和动态 uplift；Policy 页可以把 value 设置为增量成交价值，"
            "cost 设置为请求成本、延迟惩罚或系统资源成本，得到是否触发请求的阈值。\n\n"
            "**面试亮点**：这能把 uplift 从营销扩展到推荐干预和系统资源决策，说明 treatment design 可以是产品/系统策略，而不只是用户触达。\n\n"
            f"**外部参考**：\n{refs_text}\n\n"
            "**本地证据**：Models 页 `Scenario Tag = Recommendation intervention`，Knowledge 页 `Industrial Knowledge Map`。"
        )

    if _has_any(text, ["journey", "customer journey", "营销旅程", "旅程重叠", "多旅程", "overlap journey", "协同", "蚕食"]):
        refs_text = _reference_lines(["HIERARCHICAL_CAUSAL_LIFT_JOURNEYS", "SPOTIFY_IN_APP_MESSAGING_UPLIFT", "ORTHOGONAL_COMBINATORIAL_UPLIFT"])
        return (
            "### 工业难点：多个 CRM / growth journey 重叠时 uplift 怎么评估？\n\n"
            "**怎么讲**：用户可能同时处在召回、促活、优惠、推荐等多个 journey 里，单独评估某一条 journey 会把其他 journey 的影响混进来，"
            "导致 pure lift、global lift、协同和蚕食难以区分。\n\n"
            "**Agent 落地**：先把 treatment design 从二元触达升级成 multi-treatment / combinatorial treatment，"
            "再用长期 holdout、分层实验或 journey-level policy value 验证；Policy 页的成本收益要加入 fatigue、channel cannibalization 和长期 LTV 窗口。\n\n"
            "**面试亮点**：这能回答 CRM 落地里最常见的追问：为什么一个活动 offline uplift 看起来有效，但整体用户增长没有变多，"
            "因为它可能只是抢走了另一个 journey 的转化。\n\n"
            f"**外部参考**：\n{refs_text}\n\n"
            "**本地证据**：Knowledge 页 `Industrial Knowledge Map`、Models 页 `Scenario Tag = CRM lifecycle intervention`、"
            "`docs/UPLIFT_ONLINE_EXPERIMENT_AND_INCREMENTALITY.md`。"
        )

    if _has_any(text, ["llm routing 离线", "离线训练数据", "随机探索", "exploration", "routing logs", "路由日志", "采集"]):
        refs_text = _reference_lines(["ADAPTIVE_LLM_ROUTING_BUDGET", "FRUGALGPT_LLM_CASCADE", "BEST_ROUTE_TEST_TIME_COMPUTE", "ROUTELLM_OPEN_SOURCE_ROUTER"])
        return (
            "### 前沿问题：LLM routing 离线训练数据怎么采集？\n\n"
            "**核心回答**：要估计强模型升级的因果增量，必须有一小部分随机探索或分层随机流量，让相似 query 在 cheap model、strong model、RAG、tool call、human review 等 action 间有 overlap。"
            "如果只学习历史路由日志，模型会复制旧策略，无法知道没被升级的 query 调用强模型后是否真的更好。\n\n"
            "**数据字段**：query embedding、任务类型、用户/场景价值、SLA、候选模型特征、实际 action、judge score、人工标注、任务成功、成本、延迟、fallback 和安全失败原因。\n\n"
            "**建模路径**：先用随机探索数据训练 DR/R learner 或 causal forest，做 calibrated Top-K escalation；"
            "再用固定阈值 online holdout 验证 cost-adjusted quality；稳定后才升级到 contextual bandit 和 budget pacing。\n\n"
            "**本地证据**：Story/Knowledge `LLM Interview Q&A`、Policy `LLM Routing ROI Simulator`、`docs/UPLIFT_LLM_INTERVIEW_QA.md`。\n\n"
            f"**外部参考**：\n{refs_text}"
        )

    if _has_any(text, ["什么时候用 uplift", "什么时候升级到 bandit", "升级到 bandit", "uplift 到 bandit", "contextual bandit", "预算节奏", "budget pacing"]):
        refs_text = _reference_lines(["BUDGET_CONSTRAINED_CAUSAL_BANDITS", "PARETOBANDIT_LLM_SERVING", "UNIVERSAL_MODEL_ROUTING", "ADAPTIVE_LLM_ROUTING_BUDGET"])
        return (
            "### 前沿问题：什么时候用 uplift，什么时候升级到 bandit？\n\n"
            "**核心回答**：先用 uplift，因为它更容易解释、审计和复现；当模型池、价格、流量分布、预算或质量反馈持续变化，固定阈值开始失效时，再升级到 contextual bandit。\n\n"
            "**落地顺序**：\n"
            "1. 用随机实验或探索日志训练 uplift/DR/R baseline。\n"
            "2. 做 calibration、bootstrap CI、policy value 和 holdout。\n"
            "3. 用 fixed threshold / 固定阈值小流量上线，验证 cost-adjusted quality、latency SLO 和安全指标。\n"
            "4. 只有在反馈闭环可靠后，才加入 bandit 做 online adaptation、budget pacing 和 drift response。\n\n"
            "**面试讲法**：uplift 是离线可解释 policy，bandit 是在线自适应控制器；二者不是互斥，而是从稳定决策到动态决策的演进。\n\n"
            "**本地证据**：Models `LLM routing / cost-quality escalation` tag、Policy `LLM Routing ROI Simulator`、`docs/UPLIFT_LLM_INTERVIEW_QA.md`。\n\n"
            f"**外部参考**：\n{refs_text}"
        )

    if _has_any(text, ["llm routing", "模型路由", "强模型", "便宜模型", "升级模型", "cost-quality", "cost quality", "小模型", "大模型"]):
        refs_text = _reference_lines(["LLM_BANDIT_ROUTING", "ADAPTIVE_LLM_ROUTING_BUDGET", "PARETOBANDIT_LLM_SERVING", "MICROSOFT_COST_AWARE_LLM_SELECTION"])
        return (
            "### 前沿问题：LLM routing 为什么是 uplift 问题？\n\n"
            "**核心回答**：把 `调用强模型 / strong model` 看作 treatment，`调用便宜模型` 看作 control。"
            "我们真正关心的不是强模型绝对分数高不高，而是对某个 query 来说，strong model 相对便宜模型带来的 **增量质量 / incremental quality** 是否大于增量成本、延迟和预算风险。\n\n"
            "**建模方式**：query embedding、任务类型、历史难度、用户价值、SLA、模型候选特征作为 X；"
            "treatment 是 model escalation；outcome 是 judge score、任务成功、人工复核节省或业务价值。"
            "先用 uplift/DR/R learner 做离线 Top-K，再升级到 contextual bandit 做预算约束下的在线自适应。\n\n"
            "**Agent 落地**：Policy 页 `LLM Routing ROI Simulator` 会把 `quality uplift * quality value - incremental model cost - latency penalty` 转成阈值和预算讨论；"
            "Knowledge 页 `LLM + Causal Frontier Map` 记录来源，Agent Evidence Card 绑定验证脚本，避免只靠 LLM 生成解释。\n\n"
            "**面试亮点**：这是把增长投放里的 uplift 思路迁移到 LLM serving：不是所有请求都值得上最贵模型，关键是识别 persuadable queries。\n\n"
            f"**外部参考**：\n{refs_text}"
        )

    if _has_any(text, ["llm", "大模型", "causal copilot", "causal agent", "因果 copilot", "因果agent", "hallucination", "幻觉"]):
        refs_text = _reference_lines(["CAUSAL_COPILOT_LLM_AGENT", "CATE_B_LLM_COPILOT", "NAACL_LLM_CAUSAL_SURVEY"])
        return (
            "### 前沿问题：Uplift 和 LLM 怎么结合？\n\n"
            "**我的判断**：LLM 不应该替代 uplift/CATE 估计器；更合理的架构是 `LLM as causal copilot + deterministic causal engine`。"
            "LLM 负责把业务问题拆成 treatment/outcome/confounders/guardrails，解释模型和生成 evidence card；统计估计仍由 DR/R/DML、causal forest、uplift tree、policy value 完成。\n\n"
            "**三种结合方式**：\n"
            "1. **LLM as workflow orchestrator**：自动诊断、推荐模型、解释 overlap/selection bias、生成证据包。\n"
            "2. **Text/creative as treatment**：广告文案、push 内容、优惠券描述、推荐解释用 embedding 表示，配合随机曝光和 holdout，再估计 text treatment uplift。\n"
            "3. **Uplift for LLM routing**：估计强模型相对弱模型的增量质量，做成本质量阈值和 budget-constrained bandit。\n\n"
            "**防幻觉设计**：Agent 回答必须绑定来源、UI 页面、文档和验证脚本；弱来源进 backlog；关键结论走 smoke/full regression。\n\n"
            f"**外部参考**：\n{refs_text}\n\n"
            "**本地证据**：Story 页 `LLM + Uplift Frontier`、Knowledge 页 `LLM + Causal Frontier Map`、Policy 页 `LLM Routing ROI Simulator`。"
        )

    if _has_any(text, ["文案", "creative", "prompt", "offer", "text treatment", "文本 treatment", "文本干预", "生成文案"]):
        refs_text = _reference_lines(["STRUCTURED_TEXT_TREATMENTS", "LATENT_TEXTUAL_TREATMENTS", "ALIBABA_EDGEREC_REQUEST_UPLIFT"])
        return (
            "### 前沿问题：LLM 生成文案怎么做 uplift 实验？\n\n"
            "**核心回答**：LLM 文案不能只用离线偏好分数选 winner。文案本身是 structured/text treatment，需要随机曝光、holdout 和 CATE/uplift 评估。"
            "可以把文案 embedding、offer 类型、情绪、价格/权益描述、渠道上下文作为 treatment representation。\n\n"
            "**实验设计**：先固定候选文案或 prompt family，随机分配用户，保留 no-message/control；"
            "估计每类用户对不同文案的增量转化、退订/疲劳和长期 LTV。多文案/多渠道时进入 multi-treatment 或 combinatorial treatment。\n\n"
            "**Agent 落地**：Models 页按 `Recommendation / CRM / LLM routing` 场景推荐 DR/R、causal forest、multi-treatment learner；"
            "Policy 页把 uplift 转成增量毛利、信息成本和 opt-out penalty。\n\n"
            f"**外部参考**：\n{refs_text}"
        )

    if _has_any(text, ["发券", "补贴", "优惠券", "coupon", "票补", "毛利", "补贴套利", "sure thing", "sleeping dog", "roi", "投放阈值", "阈值", "policy value"]):
        return _scenario_answer("coupon", context)
    if _has_any(text, ["广告", "pctr", "pcvr", "roas", "iroas", "出价", "bidding", "曝光", "geo", "incrementality"]):
        return _scenario_answer("ads", context)
    if _has_any(text, ["用户增长", "召回", "push", "短信", "频控", "疲劳", "activation", "opt-out", "延迟反馈"]):
        return _scenario_answer("growth", context)
    if _has_any(text, ["crm", "生命周期", "电话", "email", "渠道", "触达动作"]):
        return _scenario_answer("crm", context)
    if _has_any(text, ["推荐", "重排", "曝光策略", "position bias", "干预策略"]):
        return _scenario_answer("recommendation", context)
    if _has_any(text, ["marketplace", "供需", "司机", "出行", "价格", "连续 treatment", "多 treatment", "多处理", "长期 treatment"]):
        return _scenario_answer("marketplace", context)
    if _has_any(text, ["llm", "大模型", "模型路由", "routing", "强模型", "小模型"]):
        return _scenario_answer("llm_routing", context)
    if _has_any(text, ["工业", "大厂", "落地", "业务场景", "case", "案例", "阿里", "腾讯", "美团", "滴滴", "uber", "google"]):
        rows = "\n".join(
            f"- **{scenario['name']}**：{scenario['business_question']} 推荐 `{', '.join(scenario['ready_models'][:3])}`"
            for scenario in BUSINESS_SCENARIOS
        )
        refs = "\n".join(
            f"- `{ref['company']}` {ref['title']} ({ref['source_type']}): {ref['url']}"
            for ref in INDUSTRY_REFERENCES[:10]
        )
        return (
            "### 工业级 uplift 落地主线\n\n"
            "我会按业务问题来讲模型，而不是按算法名堆列表：先判断 treatment 形态和成本收益，再选 T/X/DR/R、causal forest、multi-treatment 或 dose-response。\n\n"
            f"**业务场景**：\n{rows}\n\n"
            f"**核心参考**：\n{refs}\n\n"
            "本地证据：`docs/INDUSTRIAL_UPLIFT_CASE_STUDIES.md`、`docs/UPLIFT_BUSINESS_SCENARIOS.md`、`docs/UPLIFT_MODEL_SELECTION_PLAYBOOK.md`。"
        )
    return None
