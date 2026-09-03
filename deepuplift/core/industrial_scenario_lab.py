from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


SCENARIO_DATASET_IDS = [
    "synthetic_coupon_profit_uplift_8k",
    "synthetic_growth_delayed_feedback_7k",
    "synthetic_ads_full_funnel_ecup_7k",
    "synthetic_recommendation_intervention_7k",
    "synthetic_marketplace_subsidy_uplift_7k",
    "synthetic_llm_routing_uplift_8k",
    "synthetic_llm_multi_action_routing_9k",
    "synthetic_baidu_waimai_red_packet_8k",
    "synthetic_didi_passenger_subsidy_8k",
    "synthetic_didi_driver_supply_subsidy_8k",
    "synthetic_didi_budget_allocation_5k",
    "synthetic_shopee_ads_full_funnel_8k",
    "synthetic_spotify_inapp_message_uplift_6k",
    "synthetic_doordash_ghost_ads_lift_6k",
    "synthetic_merchant_subsidy_margin_uplift_6k",
    "synthetic_ecommerce_multi_coupon_uplift_6k",
    "synthetic_crm_overlap_journey_uplift_6k",
    "synthetic_continuous_bid_discount_uplift_5k",
    "synthetic_geo_holdout_incrementality_uplift_6k",
    "synthetic_game_ops_gift_uplift_6k",
    "synthetic_fintech_credit_line_uplift_6k",
    "synthetic_customer_service_escalation_uplift_6k",
    "synthetic_healthcare_followup_uplift_6k",
    "synthetic_saas_retention_offer_uplift_6k",
]


SCENARIO_CARDS: list[dict[str, Any]] = [
    {
        "scenario": "Coupon / Subsidy Allocation",
        "dataset_id": "synthetic_coupon_profit_uplift_8k",
        "business_problem": "Who should receive a coupon when coupon payout, margin, fatigue and abuse risk differ by user?",
        "treatment_design": "binary coupon exposure with pre-treatment coupon face value and redemption propensity features",
        "outcome": "conversion / order",
        "policy_objective": "true_uplift * gross_margin - expected_coupon_cost - channel_cost - fatigue_cost - abuse_risk_buffer",
        "primary_metrics": "QINI, AUUC, uplift@K, oracle policy capture, incremental profit, ROI, calibration, bootstrap CI",
        "recommended_models": "TLearnerLightGBM, XLearnerLightGBM, DRLearnerLightGBM, RLearnerLightGBM, CausalForest, CFRNet, DragonNet, EFIN, ContrastiveUpliftNet",
        "deep_extension": "cost-aware policy loss + treatment-aware interaction network for coupon/user interaction",
        "interview_hook": "发券不能找自然转化率最高的人，而要避开 sure things、补贴套利和 sleeping dogs，最大化增量毛利。",
        "ui_pages": "Data, Models, Evaluation, Policy, Evidence, Agent",
    },
    {
        "scenario": "Growth / CRM / Push",
        "dataset_id": "synthetic_growth_delayed_feedback_7k",
        "business_problem": "Which dormant or at-risk users should receive push/SMS when conversion can arrive after D1/D7/D14/D30?",
        "treatment_design": "binary lifecycle contact with delayed conversion windows and fatigue signals",
        "outcome": "D30 conversion / retention proxy",
        "policy_objective": "long-term uplift * value_score - contact_cost - fatigue_penalty",
        "primary_metrics": "delayed window uplift, censored rate, D30 policy capture, uplift@K, calibration, sensitivity",
        "recommended_models": "DRLearnerLightGBM, RLearnerLightGBM, PAVCalibratedDRLearnerLightGBM, CFRNet, DragonNet, CausalForest, CFR-DF guarded",
        "deep_extension": "delayed feedback temporal loss and multi-window heads",
        "interview_hook": "增长召回不能只看当天转化，否则会错杀慢响应用户，也会低估疲劳和退订风险。",
        "ui_pages": "Evaluation, Policy, Evidence, Agent",
    },
    {
        "scenario": "Ads / Audience Targeting",
        "dataset_id": "synthetic_ads_full_funnel_ecup_7k",
        "business_problem": "Which audience should be exposed when impressions, clicks, conversions and bid cost all affect incrementality?",
        "treatment_design": "binary ad exposure with impression-click-conversion stages",
        "outcome": "conversion",
        "policy_objective": "conversion uplift * order_value - media_cost",
        "primary_metrics": "full-funnel ECUP, click uplift, conversion uplift, iROAS, incremental conversion value, policy value",
        "recommended_models": "DRLearnerLightGBM, RLearnerLightGBM, TLearnerLightGBM, CausalForest, ECUP guarded, DESCN, EFIN",
        "deep_extension": "full-funnel multi-task loss with exposure/click/conversion heads",
        "interview_hook": "pCTR/pCVR 解释的是相关性/归因概率，广告 uplift 要回答广告是否带来额外转化。",
        "ui_pages": "Evaluation, Policy, Evidence, Knowledge, Agent",
    },
    {
        "scenario": "Recommendation Intervention",
        "dataset_id": "synthetic_recommendation_intervention_7k",
        "business_problem": "Should the feed run an extra request, rerank or creative intervention for a user-item context?",
        "treatment_design": "binary intervention plus creative/item treatment features and user-item affinity",
        "outcome": "conversion after request / rerank",
        "policy_objective": "request uplift * request_value - intervention_cost - negative_uplift_risk",
        "primary_metrics": "request uplift, click/conversion uplift, negative uplift rate, segment stability, Top-K policy capture",
        "recommended_models": "EFIN, DESCN, DRLearnerLightGBM, CausalForest, ContrastiveUpliftNet, SkLiftTwoModelsLightGBM",
        "deep_extension": "treatment-aware interaction network + contrastive uplift encoder",
        "interview_hook": "推荐干预的 treatment 可以是是否追加请求、重排或换素材，目标是干预带来的真实增量而不是自然偏好。",
        "ui_pages": "Models, Evaluation, Policy, Evidence, Agent",
    },
    {
        "scenario": "Marketplace Subsidy / Pricing",
        "dataset_id": "synthetic_marketplace_subsidy_uplift_7k",
        "business_problem": "Which side/city/cohort should receive subsidy when supply-demand balance and spillover can violate SUTVA?",
        "treatment_design": "binary subsidy decision with planned subsidy amount, supply pressure and demand pressure",
        "outcome": "order / match success",
        "policy_objective": "direct uplift * marketplace_value - subsidy_cost - spillover_risk_buffer",
        "primary_metrics": "direct uplift, spillover risk, net marketplace value, budget guardrail, overlap diagnostics",
        "recommended_models": "DRLearnerLightGBM, RLearnerLightGBM, CausalForest, MultiTLearnerGBM, DoseResponseGBM, GRF/causal forest guarded",
        "deep_extension": "multi-treatment or continuous dose-response model with interference guardrails",
        "interview_hook": "平台补贴不是独立个体 uplift，供需两侧会相互影响，必须把 spillover/interference 作为上线风险。",
        "ui_pages": "Workflow, Policy, Evidence, Knowledge, Agent",
    },
    {
        "scenario": "LLM Routing / Cost-Quality Uplift",
        "dataset_id": "synthetic_llm_routing_uplift_8k",
        "business_problem": "Which requests should escalate from cheap model to strong model/RAG/tool/human review?",
        "treatment_design": "binary strong-model escalation with prompt difficulty, value, cost and latency features",
        "outcome": "task success / quality pass",
        "policy_objective": "quality uplift * quality_value - extra_model_cost - latency_penalty",
        "primary_metrics": "quality uplift, cost saving, latency penalty, budget constrained routing value, calibration",
        "recommended_models": "DRLearnerLightGBM, RLearnerLightGBM, PAVCalibratedDRLearnerLightGBM, CausalForest, ContrastiveUpliftNet",
        "deep_extension": "LLM embedding uplift head + cost-quality routing loss",
        "interview_hook": "LLM routing 本质是 uplift 问题：强模型不是越多越好，而是只在边际质量收益大于成本时升级。",
        "ui_pages": "Policy, Models, Knowledge, Agent, Evidence",
    },
    {
        "scenario": "LLM Multi-Action Routing / RAG-Tool-Human Escalation",
        "dataset_id": "synthetic_llm_multi_action_routing_9k",
        "business_problem": "Which requests should stay on cheap model, or escalate to strong model, RAG, tool use, or human review under cost, latency and evidence-risk constraints?",
        "treatment_design": "binary execute planned escalation vs cheap model, with route_action identifying strong_model/RAG/tool/human_review as the candidate action",
        "outcome": "task success / quality pass",
        "policy_objective": "quality uplift * user_value - incremental_cost - latency_penalty - hallucination/evidence/judge/budget risk penalties",
        "primary_metrics": "QINI, AUUC, uplift@K, Top-K routing value, hallucination/evidence risk, budget pressure, oracle best-action capture",
        "recommended_models": "DRLearnerLightGBM, RLearnerLightGBM, TLearnerLightGBM, PAVCalibratedDRLearnerLightGBM, CausalForest, ContrastiveUpliftNet",
        "deep_extension": "query encoder + action-aware heads + LLM routing cost-quality loss; upgrade to contextual bandit after randomized exploration is stable",
        "interview_hook": "LLM routing 不是强模型二选一，而是 RAG/tool/human review 多动作 causal policy；离线先估增量质量和成本，线上再用 budget pacing/bandit。",
        "ui_pages": "Data, Models, Evaluation, Policy, Evidence, Knowledge, Agent",
    },
    {
        "scenario": "Baidu Waimai / Red Packet Subsidy",
        "dataset_id": "synthetic_baidu_waimai_red_packet_8k",
        "business_problem": "Which food-delivery users should receive a red packet when hunger intent, margin, ETA, abuse risk and reorder value differ?",
        "treatment_design": "binary red-packet exposure with face value, threshold, expiry and touch channel as pre-treatment policy features",
        "outcome": "placed order / GMV / red-packet redemption / reorder value",
        "policy_objective": "true_uplift * gross_margin + reorder_value_7d - expected_red_packet_cost - touch_cost - abuse_risk_buffer",
        "primary_metrics": "QINI, AUUC, uplift@K, incremental GMV, incremental profit, coupon ROI, subsidy abuse risk, policy capture",
        "recommended_models": "DRLearnerLightGBM, XLearnerLightGBM, TLearnerLightGBM, CausalForest, uplift forest, EFIN, ContrastiveUpliftNet",
        "deep_extension": "treatment-aware coupon interaction network and contrastive persona separation for sure thing / persuadable / sleeping dog users",
        "interview_hook": "天降红包不能只投下单概率最高的人；核心是识别被红包真正改变的人，并扣除红包成本、触达成本和套利风险。",
        "ui_pages": "Data, Models, Evaluation, Policy, Evidence, Agent, Story",
    },
    {
        "scenario": "DiDi Passenger Subsidy",
        "dataset_id": "synthetic_didi_passenger_subsidy_8k",
        "business_problem": "Which passengers should receive subsidy when ETA, fare, price sensitivity, supply gap and long-term retention affect net profit?",
        "treatment_design": "binary passenger subsidy exposure with subsidy amount, discount rate and coupon type as policy features",
        "outcome": "request / completed order / GMV / D7 retention value",
        "policy_objective": "completed-order uplift * fare * take_rate + retention_value_d7 - expected_subsidy_cost - abuse/drift penalty",
        "primary_metrics": "conversion uplift, completed-order uplift, incremental platform profit, subsidy ROI, delayed retention uplift, budget policy value",
        "recommended_models": "DRLearnerLightGBM, RLearnerLightGBM, CausalForest, multi-treatment learner, dose-response learner, delayed feedback uplift",
        "deep_extension": "delayed retention head and dose-response subsidy curve for coupon amount / discount intensity",
        "interview_hook": "乘客补贴既要看当次完单增量，也要看补贴是否培养价格敏感和未来补贴依赖。",
        "ui_pages": "Data, Models, Evaluation, Policy, Evidence, Agent, Story",
    },
    {
        "scenario": "DiDi Driver Supply Subsidy",
        "dataset_id": "synthetic_didi_driver_supply_subsidy_8k",
        "business_problem": "Which drivers/zones should receive incentive when supply coverage, match rate, wait-time reduction and spillover risk interact?",
        "treatment_design": "binary driver-side incentive with hot-zone bonus, quest, guarantee or step bonus as incentive features",
        "outcome": "driver online-and-accept / accepted orders / completed orders",
        "policy_objective": "match-rate uplift value + wait-time reduction value - driver_subsidy_cost - spillover/interference guardrail",
        "primary_metrics": "supply uplift, match-rate uplift, wait-time reduction value, net marketplace value, spillover risk, SUTVA guardrail",
        "recommended_models": "DRLearnerLightGBM, RLearnerLightGBM, CausalForest, multi-treatment learner, dose-response, marketplace policy simulator",
        "deep_extension": "marketplace-aware uplift with interference guardrails and multi-treatment incentive-type heads",
        "interview_hook": "B 端供给补贴不是单个司机的独立 uplift，城市热区的供给迁移和 spillover 会影响乘客侧体验。",
        "ui_pages": "Workflow, Models, Evaluation, Policy, Evidence, Agent, Story",
    },
    {
        "scenario": "DiDi City-Time Budget Allocation",
        "dataset_id": "synthetic_didi_budget_allocation_5k",
        "business_problem": "How should a city/time budget pool be allocated across passenger and driver subsidies under ROI and supply-demand constraints?",
        "treatment_design": "city-time high-budget treatment plus continuous budget share, C-side/B-side split and historical marginal ROI",
        "outcome": "positive city/time incremental profit / order growth",
        "policy_objective": "incremental_orders * marginal ROI - allocated_budget - cross_market_spillover_risk_buffer",
        "primary_metrics": "budget constrained policy value, marginal ROI, city-level incremental profit, budget share, spillover risk",
        "recommended_models": "DRLearnerLightGBM, RLearnerLightGBM, CausalForest, dose-response learner, budget constrained policy learner",
        "deep_extension": "budget-constrained policy layer that turns individual uplift into city/time allocation under total budget guardrails",
        "interview_hook": "预算分配不能简单按历史 ROI 排序，因为边际 ROI 会随预算强度下降，还会跨城市/时段产生供需迁移。",
        "ui_pages": "Workflow, Models, Evaluation, Policy, Evidence, Agent, Story",
    },
    {
        "scenario": "Shopee Ads / Full-Funnel iROAS",
        "dataset_id": "synthetic_shopee_ads_full_funnel_8k",
        "business_problem": "Which audience/creative/bid contexts should receive ads when impression, click, conversion, GMV and media cost all affect incrementality?",
        "treatment_design": "binary ad exposure with campaign objective, creative, bid CPC, market, device and geo/audience holdout cell",
        "outcome": "impression / click / conversion / incremental GMV",
        "policy_objective": "conversion uplift * margin-adjusted GMV - ad_cost, reported as incremental GMV and iROAS",
        "primary_metrics": "impression uplift, click uplift, conversion uplift, full-funnel ECUP, incremental GMV, iROAS, budget-constrained targeting",
        "recommended_models": "DRLearnerLightGBM, RLearnerLightGBM, TLearnerLightGBM, CausalForest, ECUP, EFIN, DESCN",
        "deep_extension": "full-funnel multi-task uplift heads for impression-click-conversion and treatment-aware creative/user interaction",
        "interview_hook": "广告 pCTR/pCVR 只说明谁会点/买，uplift 要回答广告是否制造了额外点击和额外成交，并扣除竞价成本。",
        "ui_pages": "Data, Models, Evaluation, Policy, Evidence, Agent, Story",
    },
    {
        "scenario": "Spotify-Style In-App Message Diet",
        "dataset_id": "synthetic_spotify_inapp_message_uplift_6k",
        "business_problem": "Which users should receive an in-app/push/email message when engagement uplift, fatigue, opt-out and retention value differ?",
        "treatment_design": "binary message exposure with topic, channel, prior message count and fatigue features",
        "outcome": "meaningful engagement / retention proxy",
        "policy_objective": "engagement uplift * value + retention_value - message_cost - fatigue_penalty - optout_risk_buffer",
        "primary_metrics": "uplift@K, policy value, opt-out risk, fatigue-adjusted ROI, calibration, delayed retention",
        "recommended_models": "DRLearnerLightGBM, RLearnerLightGBM, PAVCalibratedDRLearnerLightGBM, CausalForest, CFRNet, DragonNet",
        "deep_extension": "delayed-feedback head + fatigue-aware policy loss for message diet and lifecycle journeys",
        "interview_hook": "站内消息不是越多越好，uplift 要找被消息真正改变的人，并把疲劳、退订和长期留存放进 policy value。",
        "ui_pages": "Workflow, Data, Models, Evaluation, Policy, Evidence, Agent",
    },
    {
        "scenario": "DoorDash-Style Ghost Ads Sales Lift",
        "dataset_id": "synthetic_doordash_ghost_ads_lift_6k",
        "business_problem": "Which ad opportunities create incremental sales rather than attributed sales when ghost-ad or geo holdout evidence is available?",
        "treatment_design": "binary sponsored ad exposure with ghost/geo holdout cell, bid CPC, slot quality and auction pressure",
        "outcome": "conversion / sales",
        "policy_objective": "sales_lift * margin - ghost_ad_cost - holdout/auction risk buffer",
        "primary_metrics": "sales lift, iROAS, full-funnel uplift, ghost holdout policy value, budget constrained value",
        "recommended_models": "DRLearnerLightGBM, RLearnerLightGBM, TLearnerLightGBM, CausalForest, ECUP, calibrated DR",
        "deep_extension": "full-funnel ECUP heads plus ghost-ad counterfactual evidence and budget pacing",
        "interview_hook": "广告归因不等于增量，ghost ads/geo holdout 的价值是把本来会买的人和广告制造的增量销售分开。",
        "ui_pages": "Workflow, Models, Evaluation, Policy, Evidence, Knowledge, Agent",
    },
    {
        "scenario": "Merchant Subsidy / Cofund Margin Uplift",
        "dataset_id": "synthetic_merchant_subsidy_margin_uplift_6k",
        "business_problem": "Which merchants should receive platform cofunding or ranking support when margin, churn, cannibalization and spillover differ?",
        "treatment_design": "binary merchant-side subsidy/cofund exposure with promo type, cofund rate and budget share features",
        "outcome": "merchant order growth / active merchant value",
        "policy_objective": "platform margin value + merchant incremental profit share - platform subsidy - cannibalization/spillover guardrails",
        "primary_metrics": "merchant margin uplift, incremental orders, cannibalization risk, marketplace spillover, budget value",
        "recommended_models": "DRLearnerLightGBM, RLearnerLightGBM, CausalForest, MultiTLearnerGBM, DoseResponseGBM, UTBoost guarded",
        "deep_extension": "multi-treatment promo-type learner and continuous cofund-rate dose-response policy",
        "interview_hook": "商家补贴要同时看平台毛利、商家增长、自然需求蚕食和 marketplace spillover，不能只看 GMV。",
        "ui_pages": "Workflow, Models, Policy, Evidence, Agent",
    },
    {
        "scenario": "E-commerce Multi-Coupon / Multi-Threshold",
        "dataset_id": "synthetic_ecommerce_multi_coupon_uplift_6k",
        "business_problem": "Which coupon face value and minimum-spend threshold should be offered when margin, threshold fit and arbitrage risk differ?",
        "treatment_design": "binary send/no-send plus multi-level coupon_face_value, min_spend and treatment_level features",
        "outcome": "purchase conversion",
        "policy_objective": "true_uplift * gross_margin - expected_coupon_cost - arbitrage_risk_buffer",
        "primary_metrics": "QINI, AUUC, uplift@K, multi-coupon incremental profit, ROI, oracle policy capture, calibration",
        "recommended_models": "DRLearnerLightGBM, XLearnerLightGBM, RLearnerLightGBM, MultiTLearnerGBM, DoseResponseGBM, UTBoost guarded",
        "deep_extension": "multi-treatment coupon learner and continuous dose-response for coupon amount / threshold intensity",
        "interview_hook": "电商优惠券不是发不发这么简单，还要选择券面额和门槛；高 uplift 但高成本/套利的人群不一定上线。",
        "ui_pages": "Data, Models, Evaluation, Policy, Evidence, Agent",
    },
    {
        "scenario": "CRM Overlapping Customer Journeys",
        "dataset_id": "synthetic_crm_overlap_journey_uplift_6k",
        "business_problem": "Which users should receive a contact when multiple CRM journeys overlap and fatigue/opt-out/cannibalization risks exist?",
        "treatment_design": "binary contact with channel/topic plus journey_count, overlap_index and fatigue features",
        "outcome": "activation / purchase / retention proxy",
        "policy_objective": "true_uplift * long_term_value - contact_cost - optout_penalty - cannibalization_risk_buffer",
        "primary_metrics": "fatigue-adjusted policy value, opt-out uplift, uplift@K, delayed retention proxy, calibration, bootstrap CI",
        "recommended_models": "DRLearnerLightGBM, RLearnerLightGBM, PAVCalibratedDRLearnerLightGBM, CausalForest, CFRNet",
        "deep_extension": "journey-aware sequence encoder and fatigue-aware loss as guarded backlog",
        "interview_hook": "多个 CRM journey 重叠时，单 journey uplift 会把协同/蚕食算错，需要区分纯增量、全局增量和疲劳成本。",
        "ui_pages": "Data, Models, Evaluation, Policy, Knowledge, Agent",
    },
    {
        "scenario": "Continuous Bid / Discount Dose-Response",
        "dataset_id": "synthetic_continuous_bid_discount_uplift_5k",
        "business_problem": "How strong should the bid multiplier or discount intensity be when response saturates and cost rises nonlinearly?",
        "treatment_design": "binary high-intensity treatment plus continuous bid_multiplier, discount_rate and treatment_strength",
        "outcome": "conversion",
        "policy_objective": "true_uplift * order_value * margin_rate - media_cost - discount_cost",
        "primary_metrics": "dose-response gain, incremental profit, iROAS proxy, saturation risk, budget constrained policy value",
        "recommended_models": "DoseResponseGBM, DRLearnerLightGBM, RLearnerLightGBM, TLearnerLightGBM, CausalForest",
        "deep_extension": "DRNet/VCNet-style continuous treatment adapter backlog",
        "interview_hook": "出价、折扣、补贴金额是连续 treatment；最优不是越大越好，而是边际 uplift 等于边际成本的位置。",
        "ui_pages": "Dose-Response, Models, Evaluation, Policy, Evidence, Agent",
    },
    {
        "scenario": "Geo Holdout / Incrementality Experiment",
        "dataset_id": "synthetic_geo_holdout_incrementality_uplift_6k",
        "business_problem": "How can ads or subsidy uplift be validated when online decision makers need geo/time holdout evidence, not only model ranking?",
        "treatment_design": "geo-day binary treatment with holdout cells, switchback blocks, planned spend and synthetic-control gap features",
        "outcome": "conversion / incremental sales proxy",
        "policy_objective": "true_uplift * avg_order_value * margin_rate - media_cost - spillover_risk - weak_overlap_risk",
        "primary_metrics": "geo holdout lift, iROAS, policy value, overlap support, switchback stability, spillover guardrail, bootstrap CI",
        "recommended_models": "DRLearnerLightGBM, RLearnerLightGBM, TLearnerLightGBM, CausalForest, GeoLift optional, synthetic-control evidence",
        "deep_extension": "geo/time sequence encoder and online holdout evidence import backlog",
        "interview_hook": "广告或补贴上线不能只说 QINI 高；geo holdout / switchback 能验证模型推荐是否真的带来增量。",
        "ui_pages": "Workflow, Evaluation, Policy, Evidence, Knowledge, Agent",
    },
    {
        "scenario": "Game Ops Gift / First-Purchase Uplift",
        "dataset_id": "synthetic_game_ops_gift_uplift_6k",
        "business_problem": "Which players should receive starter packs, energy or battle-pass coupons when retention, LTV and backlash risk differ?",
        "treatment_design": "binary gift exposure with gift_type and gift_cost as policy features",
        "outcome": "D7 retention / return-to-play proxy",
        "policy_objective": "retention uplift * incremental_ltv - gift_cost - backlash_risk_penalty",
        "primary_metrics": "QINI, AUUC, uplift@K, incremental LTV, gift ROI, churn risk, backlash risk",
        "recommended_models": "DRLearnerLightGBM, RLearnerLightGBM, TLearnerLightGBM, CausalForest, uplift tree, ContrastiveUpliftNet",
        "deep_extension": "persona contrastive uplift for frustrated persuadables vs sure-thing whales",
        "interview_hook": "游戏礼包不能只给付费概率最高的玩家，很多鲸鱼是 sure thing；真正价值在于挽回高挫败但可被礼包改变的玩家。",
        "ui_pages": "Data, Models, Evaluation, Policy, Evidence, Agent",
    },
    {
        "scenario": "FinTech Credit-Line / APR Offer",
        "dataset_id": "synthetic_fintech_credit_line_uplift_6k",
        "business_problem": "Which users should receive credit-line increase, APR discount or fee waiver when take-up, revenue and default risk trade off?",
        "treatment_design": "binary financial offer with offer_type, capital cost and default-risk-lift fields",
        "outcome": "offer take-up / good utilization proxy",
        "policy_objective": "take-up uplift * incremental_interest_revenue - credit_cost - capital_cost * default_risk_lift",
        "primary_metrics": "uplift@K, incremental interest revenue, default-risk lift, risk-adjusted policy value, overlap",
        "recommended_models": "DRLearnerLightGBM, RLearnerLightGBM, CausalForest, DML, TLearnerLightGBM",
        "deep_extension": "risk-aware policy loss and monotonic guardrails for regulated decisions",
        "interview_hook": "金融优惠不能只看响应率，必须把增量收入和违约风险的增量一起放进 policy value。",
        "ui_pages": "Data, Models, Evaluation, Policy, Evidence, Agent",
    },
    {
        "scenario": "Customer Service Human Escalation",
        "dataset_id": "synthetic_customer_service_escalation_uplift_6k",
        "business_problem": "Which bot/chat cases should escalate to human agents when CSAT, retention value, queue cost and handle time conflict?",
        "treatment_design": "binary human escalation with issue complexity, bot confidence, sentiment and channel features",
        "outcome": "resolved / satisfied case proxy",
        "policy_objective": "CSAT uplift * retention_value - escalation_cost - handle_time_penalty",
        "primary_metrics": "CSAT uplift, retention value, escalation cost, queue penalty, Top-K policy value",
        "recommended_models": "DRLearnerLightGBM, RLearnerLightGBM, TLearnerLightGBM, CausalForest, LLM routing uplift",
        "deep_extension": "LLM/customer-support routing uplift with cost-quality and latency constraints",
        "interview_hook": "客服升级人工和 LLM routing 很像：不是所有问题都该升级，关键是升级带来的增量满意度是否覆盖人力和排队成本。",
        "ui_pages": "Data, Models, Evaluation, Policy, Evidence, Agent",
    },
    {
        "scenario": "Healthcare Follow-Up / Adherence",
        "dataset_id": "synthetic_healthcare_followup_uplift_6k",
        "business_problem": "Which patients should receive SMS, nurse calls, app reminders or community-worker outreach when adherence and no-show risks differ?",
        "treatment_design": "binary outreach with channel, outreach_cost, clinical_risk and no_show_risk features",
        "outcome": "adherence / attended follow-up proxy",
        "policy_objective": "adherence uplift * adherence_value - outreach_cost - clinical-risk guardrail penalty",
        "primary_metrics": "adherence uplift, no-show reduction, clinical risk guardrail, outreach ROI, calibration",
        "recommended_models": "DRLearnerLightGBM, RLearnerLightGBM, CausalForest, TLearnerLightGBM, interpretable policy tree",
        "deep_extension": "interpretable CATE + safety guardrail; avoid black-box-only decisioning for high-risk patients",
        "interview_hook": "医疗随访里 uplift 要找会被干预改变的人，同时不能把高临床风险误当成纯 ROI 问题。",
        "ui_pages": "Data, Models, Evaluation, Policy, Evidence, Agent",
    },
    {
        "scenario": "SaaS Retention / Renewal Offer",
        "dataset_id": "synthetic_saas_retention_offer_uplift_6k",
        "business_problem": "Which accounts should receive discounts, CSM calls, training or feature credits when churn risk, ARR and discount dependency differ?",
        "treatment_design": "binary retention intervention with offer_type, ARR, churn risk, expansion potential and offer cost",
        "outcome": "renewal / retained account proxy",
        "policy_objective": "renewal uplift * expansion_value - offer_cost - discount_dependency_risk * ARR",
        "primary_metrics": "renewal uplift, ARR-weighted policy value, discount dependency risk, uplift@K, bootstrap CI",
        "recommended_models": "DRLearnerLightGBM, RLearnerLightGBM, CausalForest, TLearnerLightGBM, uplift tree",
        "deep_extension": "account-level policy optimization with ARR weighting and treatment-cost calibration",
        "interview_hook": "SaaS 挽留不能只按 churn risk 排序，高风险客户未必能被优惠改变，高 ARR 客户也可能是 sure thing。",
        "ui_pages": "Data, Models, Evaluation, Policy, Evidence, Agent",
    },
]


def scenario_cards() -> list[dict[str, Any]]:
    return list(SCENARIO_CARDS)


def _safe_float(series: pd.Series, default: float = 0.0) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return default
    return float(values.mean())


def _lift_type_mix(df: pd.DataFrame) -> str:
    if "uplift_type" not in df.columns:
        return ""
    counts = df["uplift_type"].astype(str).value_counts(normalize=True).head(5)
    return "; ".join(f"{idx}={value:.1%}" for idx, value in counts.items())


def scenario_dataset_summary(manifest_path: str | Path = "examples/datasets/manifest.json") -> list[dict[str, Any]]:
    import json

    manifest = Path(manifest_path)
    if not manifest.exists():
        return []
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    by_id = {row.get("id"): row for row in payload.get("datasets", [])}
    rows: list[dict[str, Any]] = []
    cards = {row["dataset_id"]: row for row in SCENARIO_CARDS}
    root = Path(".")
    for dataset_id in SCENARIO_DATASET_IDS:
        dataset = by_id.get(dataset_id)
        card = cards.get(dataset_id, {})
        if not dataset:
            rows.append({"dataset_id": dataset_id, "status": "missing_manifest", **card})
            continue
        path = root / str(dataset.get("path", ""))
        if not path.exists():
            rows.append({"dataset_id": dataset_id, "status": "missing_file", **card})
            continue
        try:
            df = pd.read_csv(path)
        except Exception as exc:  # pragma: no cover - UI fallback
            rows.append({"dataset_id": dataset_id, "status": f"read_error: {exc}", **card})
            continue
        treatment_col = dataset.get("treatment_col", "treatment")
        outcome_col = dataset.get("outcome_col", "outcome")
        policy_col = "oracle_policy_value" if "oracle_policy_value" in df.columns else None
        true_col = "true_uplift" if "true_uplift" in df.columns else None
        rows.append(
            {
                "scenario": card.get("scenario", dataset.get("name")),
                "dataset_id": dataset_id,
                "name": dataset.get("name"),
                "kind": dataset.get("kind"),
                "status": "ready",
                "rows": len(df),
                "features": len(dataset.get("feature_cols") or []),
                "treatment_rate": _safe_float(df[treatment_col]) if treatment_col in df.columns else None,
                "outcome_rate": _safe_float(df[outcome_col]) if outcome_col in df.columns else None,
                "true_uplift_mean": _safe_float(df[true_col]) if true_col else None,
                "policy_value_mean": _safe_float(df[policy_col]) if policy_col else None,
                "policy_value_p90": float(pd.to_numeric(df[policy_col], errors="coerce").quantile(0.9)) if policy_col else None,
                "lift_type_mix": _lift_type_mix(df),
                "recommended_models": ", ".join(dataset.get("recommended_models") or []),
                "policy_objective": card.get("policy_objective", ""),
                "primary_metrics": card.get("primary_metrics", ""),
                "ui_pages": card.get("ui_pages", ""),
            }
        )
    return rows


def scenario_agent_reply(prompt: str) -> str:
    text = prompt.lower()
    if not any(
        key in text
        for key in [
            "场景数据",
            "模拟数据",
            "synthetic",
            "发券数据",
            "广告数据",
            "增长数据",
            "推荐数据",
            "marketplace",
            "训练闭环",
            "业务场景",
            "百度",
            "外卖",
            "红包",
            "天降红包",
            "滴滴",
            "乘客",
            "司机",
            "预算分配",
            "虾皮",
            "shopee",
            "iroas",
            "spotify",
            "站内消息",
            "message",
            "doordash",
            "ghost",
            "商家",
            "merchant",
            "共投",
            "电商",
            "多券",
            "券面额",
            "门槛",
            "crm",
            "journey",
            "旅程",
            "重叠",
            "连续",
            "出价",
            "折扣",
            "剂量",
            "geo",
            "holdout",
            "地理",
            "增量实验",
            "switchback",
            "游戏",
            "礼包",
            "首充",
            "金融",
            "授信",
            "额度",
            "利率",
            "客服",
            "人工",
            "升级",
            "医疗",
            "随访",
            "健康",
            "saas",
            "续费",
            "订阅",
            "llm",
            "rag",
            "tool",
            "human review",
            "人工审核",
            "多动作",
            "路由",
        ]
    ):
        return ""

    sections: list[str] = []
    if any(key in text for key in ["百度", "外卖", "红包", "天降红包"]):
        sections.extend(
            [
                "### 百度外卖天降红包",
                "",
                "- 业务问题：红包不是给下单概率最高的人，而是给 `persuadable` 人群；`sure thing` 会浪费补贴，`sleeping dog` 可能引入负 uplift 和套利。",
                "- 数据 schema：`synthetic_baidu_waimai_red_packet_8k` 包含饥饿/下单意图、历史订单、距离、天气、时段、红包金额、门槛、有效期、渠道、毛利、核销概率、套利风险。",
                "- 策略指标：QINI/AUUC 看排序，`incremental_gmv` 看 GMV 增量，`incremental_profit` 和 `oracle_policy_value` 扣除红包成本、触达成本、复购价值和套利风险。",
                "- 推荐路径：先跑 T/X/DR learner + LightGBM/CausalForest，再把 EFIN/ContrastiveUpliftNet 作为红包金额和用户偏好交互的高级插件。",
            ]
        )
    if "滴滴" in text and any(key in text for key in ["乘客", "c端", "c 端", "补贴"]):
        sections.extend(
            [
                "### 滴滴 C 端乘客补贴",
                "",
                "- 业务问题：乘客补贴要同时优化发单、完单、平台抽佣和 D7 留存，不能只看当次叫车转化。",
                "- 数据 schema：`synthetic_didi_passenger_subsidy_8k` 包含 ETA、价格敏感、供需缺口、补贴金额、折扣率、券类型、竞争强度、完单 uplift、留存价值和价格敏感漂移。",
                "- 评估指标：completed-order uplift、incremental platform profit、subsidy ROI、budget constrained policy value、delayed retention uplift。",
                "- 推荐路径：R/DR learner 和 CausalForest 做稳健主线；连续补贴金额时进入 dose-response；长期留存时接 delayed feedback evaluator。",
            ]
        )
    if "滴滴" in text and any(key in text for key in ["司机", "b端", "b 端", "供给", "司机补贴"]):
        sections.extend(
            [
                "### 滴滴 B 端司机/供给侧补贴",
                "",
                "- 业务问题：司机补贴优化的是供给覆盖、接单/完单、乘客等待时间下降和平台净价值，同时要防止热区迁移带来的 spillover。",
                "- 数据 schema：`synthetic_didi_driver_supply_subsidy_8k` 包含在线时长、接单率、供需压力、收入缺口、冲单奖/热区奖/保底/阶梯奖、疲劳、spillover/interference 风险。",
                "- 评估指标：supply uplift、match-rate uplift、wait-time reduction value、net marketplace value、SUTVA/interference guardrail。",
                "- 推荐路径：DR/R learner + CausalForest 做异质性，multi-treatment/dose-response 处理多种激励和补贴强度，Policy 页面看净平台价值而不是单侧 ROI。",
            ]
        )
    if "滴滴" in text and any(key in text for key in ["预算", "分配", "城市", "时段"]):
        sections.extend(
            [
                "### 滴滴城市/时段预算分配",
                "",
                "- 业务问题：预算分配不是按历史 ROI 排序，因为边际 ROI 会随预算强度下降，C 端和 B 端补贴还会互相影响供需平衡。",
                "- 数据 schema：`synthetic_didi_budget_allocation_5k` 是 city-time 粒度，包含预算池、预算比例、C/B 侧 split、历史 ROI、供需缺口、竞争强度、边际 ROI 和跨市场 spillover。",
                "- 评估指标：budget constrained policy value、marginal ROI、city incremental profit、budget share、cross-market spillover risk。",
                "- 推荐路径：先用 DR/R/CausalForest 估计高预算 treatment 的异质性，再用 budget-constrained policy layer 做 city/time allocation。",
            ]
        )
    if any(key in text for key in ["虾皮", "shopee", "iroas", "广告"]):
        sections.extend(
            [
                "### Shopee 广告投放 / iROAS",
                "",
                "- 业务问题：pCTR/pCVR 只回答谁会点/买，uplift 回答广告是否制造额外点击、额外转化和额外 GMV。",
                "- 数据 schema：`synthetic_shopee_ads_full_funnel_8k` 包含用户购买意图、类目偏好、seller quality、广告位、bid CPC、campaign objective、creative、market/device、geo/audience holdout。",
                "- 评估指标：impression uplift、click uplift、conversion uplift、full-funnel ECUP、incremental GMV、iROAS、budget constrained audience targeting。",
                "- 推荐路径：DR/R learner + CausalForest 做稳健增量，人群/素材/出价强交互时用 EFIN/DESCN/ECUP guarded plugin。",
            ]
        )
    if any(key in text for key in ["spotify", "站内消息", "message", "消息疲劳", "退订"]):
        sections.extend(
            [
                "### Spotify 风格站内消息 / message diet",
                "",
                "- 业务问题：站内消息、push、email 不是触达越多越好；要估计消息带来的真实 engagement/retention uplift，同时扣除疲劳、退订和打扰成本。",
                "- 数据 schema：`synthetic_spotify_inapp_message_uplift_6k` 包含 topic、channel、14 天消息频次、message fatigue、message cost、opt-out risk、retention value 和 true engagement uplift。",
                "- 评估指标：uplift@K、fatigue-adjusted policy value、opt-out risk、calibration、delayed retention。",
                "- 推荐路径：DR/R learner + calibrated DR 做稳定主线；多窗口留存时接 delayed feedback head；消息/用户偏好强交互时用 CFRNet/DragonNet 或 treatment-aware interaction。",
            ]
        )
    if any(key in text for key in ["doordash", "ghost", "sales lift", "增量销售"]):
        sections.extend(
            [
                "### DoorDash 风格 ghost ads / sales lift",
                "",
                "- 业务问题：广告归因销售不等于广告增量销售；ghost ads 或 geo holdout 的价值是构造反事实，把本来会买的人和广告制造的增量分开。",
                "- 数据 schema：`synthetic_doordash_ghost_ads_lift_6k` 包含 sponsored exposure、ghost/holdout cell、bid CPC、slot quality、impression/click/conversion、sales lift 和 iROAS。",
                "- 评估指标：full-funnel uplift、conversion uplift、sales lift、iROAS、holdout policy value、budget constrained value。",
                "- 推荐路径：先用 T/DR/R learner + CausalForest 做 audience targeting；当 impression-click-conversion 链路很强时接 ECUP full-funnel evaluator/guarded plugin。",
            ]
        )
    if any(key in text for key in ["llm", "rag", "tool", "human review", "人工审核", "多动作", "路由"]):
        sections.extend(
            [
                "### LLM multi-action routing / RAG-Tool-Human escalation",
                "",
                "- 业务问题：便宜模型、强模型、RAG、tool 和人工审核不是模型分数排序，而是多动作 treatment policy；关键是每个请求升级后的增量质量是否覆盖成本、延迟、幻觉风险和证据失败风险。",
                "- 数据 schema：`synthetic_llm_multi_action_routing_9k` 包含 prompt tokens、ambiguity、retrieval/tool need、safety risk、user value、latency SLO、judge noise、budget pressure、`route_action`、`true_uplift`、`oracle_policy_value` 和 `oracle_best_policy_value`。",
                "- 评估指标：QINI/AUUC 看增量排序，Top-K routing value 看上线净收益，hallucination/evidence risk 看安全护栏，oracle best-action capture 看路由动作是否接近最优。",
                "- 推荐路径：先用随机探索数据跑 DR/R learner + calibrated uplift；稳定后加入 action-aware neural head、cost-quality loss，再升级到 contextual bandit 和 budget pacing。",
            ]
        )
    if any(key in text for key in ["商家", "merchant", "共投", "cofund", "毛利"]):
        sections.extend(
            [
                "### 商家补贴 / 共投毛利 uplift",
                "",
                "- 业务问题：商家侧补贴要同时看平台毛利、商家活跃、商家增量利润、自然需求蚕食和 marketplace spillover，不能只看 GMV。",
                "- 数据 schema：`synthetic_merchant_subsidy_margin_uplift_6k` 包含 merchant quality、margin rate、cofund rate、promo type、subsidy cost、cannibalization risk、spillover risk 和 platform margin value。",
                "- 评估指标：merchant margin uplift、incremental orders、platform/merchant profit、cannibalization guardrail、marketplace spillover、budget value。",
                "- 推荐路径：DR/R learner + CausalForest 做二元补贴主线；promo type 多样时走 multi-treatment；cofund rate 或补贴强度连续时走 dose-response/UTBoost guarded 路径。",
            ]
        )
    if any(key in text for key in ["电商", "多券", "券面额", "门槛", "multi-coupon"]):
        sections.extend(
            [
                "### 电商多券面额 / 多门槛",
                "",
                "- 业务问题：优惠券不是发不发，还要决定券面额、门槛和渠道；高增量但高成本或高套利的人群不一定上线。",
                "- 数据 schema：`synthetic_ecommerce_multi_coupon_uplift_6k` 包含 coupon_face_value、min_spend、treatment_level、gross_margin、expected_coupon_cost、arbitrage_risk。",
                "- 评估指标：QINI/AUUC、uplift@K、多券增量利润、ROI、oracle policy capture、calibration。",
                "- 推荐路径：DR/X/R learner 做稳健 baseline；多券面额走 multi-treatment；券金额/门槛强度走 dose-response；大表可以把 UTBoost 作为 guarded 插件。",
            ]
        )
    if any(key in text for key in ["crm", "journey", "旅程", "重叠", "overlap", "疲劳"]):
        sections.extend(
            [
                "### CRM 多 journey 重叠触达",
                "",
                "- 业务问题：多条营销旅程同时命中同一用户时，单 journey uplift 会混入协同、蚕食和疲劳成本。",
                "- 数据 schema：`synthetic_crm_overlap_journey_uplift_6k` 包含 journey_count、overlap_index、recent_push_count、sms_count、fatigue、optout_uplift、cannibalization_risk。",
                "- 评估指标：fatigue-adjusted policy value、opt-out uplift、uplift@K、calibration、bootstrap CI。",
                "- 推荐路径：DR/R learner + calibrated DR 做主线；高 overlap 人群单独看 trimming/sensitivity；后续可接 sequence encoder 或 journey-aware loss。",
            ]
        )
    if any(key in text for key in ["连续", "出价", "折扣", "剂量", "dose", "bid", "discount"]):
        sections.extend(
            [
                "### 连续出价 / 折扣剂量响应",
                "",
                "- 业务问题：bid multiplier、discount rate、补贴金额不是越大越好，存在饱和和边际成本上升。",
                "- 数据 schema：`synthetic_continuous_bid_discount_uplift_5k` 包含 bid_multiplier、discount_rate、treatment_strength、saturation、media_cost、discount_cost。",
                "- 评估指标：dose-response gain、incremental profit、iROAS proxy、saturation risk、budget constrained policy value。",
                "- 推荐路径：先用二元 high-intensity treatment 跑 DR/R baseline，再用 `DoseResponseGBM` 学习强度曲线，DRNet/VCNet 作为深度插件路线。",
            ]
        )
    if any(key in text for key in ["geo", "holdout", "地理", "增量实验", "switchback", "geolift"]):
        sections.extend(
            [
                "### Geo holdout / 地理增量实验",
                "",
                "- 业务问题：广告、补贴、城市预算不能只靠离线 QINI，需要用 geo holdout、switchback 或 synthetic control 验证真实增量。",
                "- 数据 schema：`synthetic_geo_holdout_incrementality_uplift_6k` 包含 geo/day、holdout_cell、switchback_block、planned_spend、weak_overlap_risk、geo_spillover_risk 和 synthetic_control_gap。",
                "- 评估指标：geo holdout lift、iROAS、policy value、overlap support、switchback stability、spillover guardrail、bootstrap CI。",
                "- 推荐路径：DR/R learner + CausalForest 先做 geo/time CATE；上线前用 GeoLift/geo experiment evidence 验证增量，而不是把模型排序直接当收益。",
            ]
        )
    if any(key in text for key in ["游戏", "礼包", "首充"]):
        sections.extend(
            [
                "### 游戏运营礼包 / 首充 uplift",
                "",
                "- 业务问题：礼包不是给付费概率最高的玩家，而是给会被礼包真正挽回的玩家；高付费鲸鱼可能是 sure thing。",
                "- 数据 schema：`synthetic_game_ops_gift_uplift_6k` 包含玩家等级、安装天数、7 日会话、30 日付费、挫败/难度、礼包类型、礼包成本、churn risk、pay-to-win backlash risk。",
                "- 评估指标：retention uplift、incremental LTV、gift ROI、backlash risk、Top-K policy value。",
                "- 推荐路径：DR/R learner + CausalForest 做主线，uplift tree 用于运营规则解释，ContrastiveUpliftNet 分离 frustrated persuadable 与 sure-thing whales。",
            ]
        )
    if any(key in text for key in ["金融", "授信", "额度", "利率", "免息"]):
        sections.extend(
            [
                "### 金融授信额度 / 利率优惠 uplift",
                "",
                "- 业务问题：金融 offer 不能只看响应率，还要看增量收入和违约风险增量。",
                "- 数据 schema：`synthetic_fintech_credit_line_uplift_6k` 包含 credit score、utilization、delinquency、liquidity need、offer type、capital cost、default_risk_lift、incremental interest revenue。",
                "- 评估指标：risk-adjusted policy value、default-risk lift、uplift@K、overlap、calibration 和受监管场景的可解释性。",
                "- 推荐路径：DR/R learner、DML、CausalForest；上线必须把高 default-risk 人群做 guardrail，不把 uplift 当纯收益指标。",
            ]
        )
    if any(key in text for key in ["客服", "人工", "升级", "escalation"]):
        sections.extend(
            [
                "### 客服升级人工 / 智能客服 routing uplift",
                "",
                "- 业务问题：不是所有 case 都该升级人工，关键是升级带来的增量满意度/留存是否覆盖人力和排队成本。",
                "- 数据 schema：`synthetic_customer_service_escalation_uplift_6k` 包含 issue complexity、bot confidence、情绪、渠道、问题类型、escalation cost、retention value、handle-time penalty。",
                "- 评估指标：CSAT uplift、retention policy value、queue/handle-time penalty、Top-K escalation ROI。",
                "- 推荐路径：DR/R learner + CausalForest，和 LLM routing 共用 cost-quality uplift 思路。",
            ]
        )
    if any(key in text for key in ["医疗", "随访", "健康", "patient", "healthcare"]):
        sections.extend(
            [
                "### 医疗随访 / 健康管理 uplift",
                "",
                "- 业务问题：随访要找会被短信/护士电话/app 提醒改变的人，同时高临床风险不能被简单 ROI 化。",
                "- 数据 schema：`synthetic_healthcare_followup_uplift_6k` 包含年龄、慢病风险、历史 no-show、依从性、渠道、outreach cost、clinical risk、adherence value。",
                "- 评估指标：adherence uplift、no-show reduction、clinical risk guardrail、outreach ROI、calibration。",
                "- 推荐路径：DR/R learner、CausalForest、可解释 policy tree；高风险场景需要 safety guardrail。",
            ]
        )
    if any(key in text for key in ["saas", "续费", "订阅"]):
        sections.extend(
            [
                "### SaaS 续费 / 流失挽回 uplift",
                "",
                "- 业务问题：高 churn risk 不等于应该给折扣，高 ARR 也可能是 sure thing；要看续费干预的增量和折扣依赖风险。",
                "- 数据 schema：`synthetic_saas_retention_offer_uplift_6k` 包含 ARR、席位数、产品使用、工单、NPS、续费天数、offer type、offer cost、discount dependency risk。",
                "- 评估指标：renewal uplift、ARR-weighted policy value、discount dependency risk、bootstrap CI、uplift@K。",
                "- 推荐路径：DR/R learner + CausalForest，uplift tree 做 CSM 可解释规则，Policy 页面看 ARR 加权净收益。",
            ]
        )

    if sections:
        sections.extend(
            [
                "",
                "UI 证据：`Workflow -> Scenario Data Lab`、`Data -> Demo preset`、`Models -> Scenario Data-To-Model Matrix`、`Policy -> Industrial Scenario Dataset ROI Benchmarks`、`Evidence -> Industrial Scenario Smokes`。",
                "代码证据：`scripts/generate_industrial_scenario_datasets.py`、`deepuplift/core/industrial_scenario_lab.py`、`deepuplift/core/evaluator.py`、`scripts/smoke_industrial_scenario_training.py`。",
            ]
        )
        return "\n".join(sections)

    rows = scenario_dataset_summary()
    ready = [row for row in rows if row.get("status") == "ready"]
    lines = [
        "### Industrial Scenario Lab",
        "",
        f"当前有 `{len(ready)}` 个可演示工业场景数据集，覆盖发券、增长、广告、推荐干预、marketplace 补贴、LLM routing，以及百度外卖、滴滴和 Shopee 履历场景。",
        "",
    ]
    for row in ready:
        lines.append(
            "- `{scenario}`：dataset `{dataset_id}`，rows `{rows}`，policy objective `{policy}`，推荐模型 `{models}`。".format(
                scenario=row.get("scenario"),
                dataset_id=row.get("dataset_id"),
                rows=row.get("rows"),
                policy=row.get("policy_objective"),
                models=row.get("recommended_models"),
            )
        )
    lines.extend(
        [
            "",
            "UI 证据：`Workflow -> Scenario Data Lab`、`Policy -> Industrial Scenario Dataset ROI Benchmarks`、"
            "`Evidence -> Industrial Scenario Dataset/Training Smoke`。",
            "代码证据：`scripts/generate_industrial_scenario_datasets.py`、`scripts/smoke_industrial_scenario_datasets.py`、"
            "`scripts/smoke_industrial_scenario_training.py`。",
        ]
    )
    return "\n".join(lines)
