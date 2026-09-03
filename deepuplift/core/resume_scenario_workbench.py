from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .frontier_models import frontier_model_capabilities
from .industrial_scenario_lab import scenario_cards, scenario_dataset_summary
from .policy import llm_routing_action_coverage_rows, llm_routing_bandit_drift_rows, llm_routing_ope_summary
from .resume_scenario_policy import RESUME_SCENARIO_IDS, resume_scenario_policy_benchmarks


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"


RESUME_SCENARIOS_CN: list[dict[str, Any]] = [
    {
        "场景": "百度外卖天降红包",
        "dataset_id": "synthetic_baidu_waimai_red_packet_8k",
        "业务问题": "红包应该发给会被红包真正改变的人，而不是自然下单概率最高的人。",
        "Treatment": "是否发放红包；红包金额、门槛、有效期、触达渠道是策略特征。",
        "Control": "不发红包或进入 holdout。",
        "Outcome": "是否下单、GMV、核销、复购价值。",
        "Cost / Value": "红包成本、触达成本、毛利、套利风险、复购价值。",
        "推荐模型": "T/X/DR/R learner、LightGBM、CausalForest、uplift forest、EFIN、ContrastiveUpliftNet",
        "上线目标": "Top-K incremental profit > 0，且 subsidy abuse risk 可控。",
        "面试讲法": "从转化率模型升级到 uplift，是为了避开 sure thing 和 sleeping dog，把补贴花在 persuadable 用户上。",
        "证据": "examples/datasets/synthetic_baidu_waimai_red_packet_8k.csv; scripts/smoke_industrial_scenario_compare.py",
    },
    {
        "场景": "滴滴 C 端乘客补贴",
        "dataset_id": "synthetic_didi_passenger_subsidy_8k",
        "业务问题": "乘客补贴要提升发单/完单和留存，同时控制补贴依赖和套利。",
        "Treatment": "是否给乘客补贴；补贴金额、折扣率、券类型是策略特征。",
        "Control": "不补贴或进入随机 holdout。",
        "Outcome": "发单、完单、GMV、平台抽佣、D7 留存价值。",
        "Cost / Value": "乘客补贴成本、完单抽佣、留存价值、价格敏感漂移。",
        "推荐模型": "DR/R learner、CausalForest、dose-response、delayed feedback uplift",
        "上线目标": "completed-order uplift 和 retention-adjusted policy value 同时为正。",
        "面试讲法": "C 端补贴不能只看当次叫车转化，要同时看完单、供需缺口、留存和补贴依赖。",
        "证据": "examples/datasets/synthetic_didi_passenger_subsidy_8k.csv; reports/industrial_scenario_training_smoke_latest.json",
    },
    {
        "场景": "滴滴 B 端司机补贴",
        "dataset_id": "synthetic_didi_driver_supply_subsidy_8k",
        "业务问题": "司机激励要增加供给覆盖和降低等待时长，同时防止热区迁移与 spillover。",
        "Treatment": "是否给司机侧激励；冲单奖、热区奖、保底、阶梯奖励是 treatment 类型。",
        "Control": "不给激励或仅保留自然供给。",
        "Outcome": "上线接单、在线时长、接单数、完单数。",
        "Cost / Value": "司机补贴成本、等待时长下降价值、match-rate value、spillover/interference 风险。",
        "推荐模型": "DR/R learner、CausalForest、multi-treatment、dose-response、marketplace policy simulator",
        "上线目标": "net marketplace value 为正，且 SUTVA/interference guardrail 不触发。",
        "面试讲法": "B 端补贴不是单人群 uplift，供给迁移会影响乘客体验，所以要把 marketplace 风险放进 policy value。",
        "证据": "examples/datasets/synthetic_didi_driver_supply_subsidy_8k.csv; deepuplift/core/resume_scenario_policy.py",
    },
    {
        "场景": "滴滴城市/时段预算分配",
        "dataset_id": "synthetic_didi_budget_allocation_5k",
        "业务问题": "城市/时段预算要在 C 端、B 端和供需缺口之间做约束优化。",
        "Treatment": "高预算/低预算分配；预算占比、C/B 侧 split 是连续策略变量。",
        "Control": "低预算或历史预算策略。",
        "Outcome": "订单增量、城市/时段增量利润、供需平衡。",
        "Cost / Value": "预算池、已分配预算、边际 ROI、跨市场 spillover 风险。",
        "推荐模型": "DR/R learner、CausalForest、dose-response、budget-constrained policy optimizer",
        "上线目标": "在总预算约束下最大化 city-time policy value，而不是简单按历史 ROI 排序。",
        "面试讲法": "历史 ROI 是平均收益，预算分配看的是边际 ROI 和供需联动，必须做 constrained allocation。",
        "证据": "examples/datasets/synthetic_didi_budget_allocation_5k.csv; Policy -> DiDi City-Time Budget Allocation Simulator",
    },
    {
        "场景": "Shopee 广告投放 / iROAS",
        "dataset_id": "synthetic_shopee_ads_full_funnel_8k",
        "业务问题": "广告投放要找增量点击/增量成交人群，而不是只找 pCTR/pCVR 高的人。",
        "Treatment": "广告曝光/出价策略/素材策略。",
        "Control": "不曝光广告或 geo/audience holdout。",
        "Outcome": "impression、click、conversion、GMV。",
        "Cost / Value": "bid CPC、ad cost、incremental GMV、margin-adjusted iROAS。",
        "推荐模型": "DR/R/T learner、CausalForest、ECUP、EFIN、DESCN",
        "上线目标": "Top-K conversion uplift、incremental GMV 和 iROAS 同时稳定。",
        "面试讲法": "pCTR/pCVR 是相关性和归因概率，uplift 回答广告是否制造额外 GMV，并扣除竞价成本。",
        "证据": "examples/datasets/synthetic_shopee_ads_full_funnel_8k.csv; Evaluation -> full-funnel ECUP",
    },
]


EXTENSION_SCENARIO_CN: dict[str, str] = {
    "synthetic_coupon_profit_uplift_8k": "通用发券/补贴利润",
    "synthetic_growth_delayed_feedback_7k": "用户增长/Push 延迟反馈",
    "synthetic_ads_full_funnel_ecup_7k": "广告全链路 ECUP",
    "synthetic_recommendation_intervention_7k": "推荐干预/素材 uplift",
    "synthetic_marketplace_subsidy_uplift_7k": "Marketplace 补贴/供需平衡",
    "synthetic_llm_routing_uplift_8k": "LLM routing 成本质量",
    "synthetic_llm_multi_action_routing_9k": "LLM 多动作 routing/RAG-tool-人工",
    "synthetic_spotify_inapp_message_uplift_6k": "Spotify 风格站内消息",
    "synthetic_doordash_ghost_ads_lift_6k": "DoorDash 风格 ghost ads",
    "synthetic_merchant_subsidy_margin_uplift_6k": "商家补贴/共投毛利",
    "synthetic_ecommerce_multi_coupon_uplift_6k": "电商多券面额/多门槛",
    "synthetic_crm_overlap_journey_uplift_6k": "CRM 多 journey 重叠触达",
    "synthetic_continuous_bid_discount_uplift_5k": "连续出价/折扣剂量响应",
    "synthetic_geo_holdout_incrementality_uplift_6k": "Geo holdout/地理增量实验",
    "synthetic_game_ops_gift_uplift_6k": "游戏运营礼包/首充",
    "synthetic_fintech_credit_line_uplift_6k": "金融授信/利率优惠",
    "synthetic_customer_service_escalation_uplift_6k": "客服升级人工",
    "synthetic_healthcare_followup_uplift_6k": "医疗随访/健康管理",
    "synthetic_saas_retention_offer_uplift_6k": "SaaS 续费/流失挽回",
}


EXTENSION_SOURCE_EVIDENCE: dict[str, dict[str, str]] = {
    "synthetic_spotify_inapp_message_uplift_6k": {
        "来源质量": "官方工程博客",
        "来源标题": "Spotify Engineering: ML to target in-app messaging",
        "来源链接": "https://engineering.atspotify.com/2023/6/experimenting-with-machine-learning-to-target-in-app-messaging",
        "落地启发": "随机 holdout + CATE/uplift + 多目标消息 eligibility + offline policy evaluation。",
    },
    "synthetic_doordash_ghost_ads_lift_6k": {
        "来源质量": "官方广告技术文章",
        "来源标题": "DoorDash Ads: Measuring incrementality with ghost ads",
        "来源链接": "https://advertising.doordash.com/en-au/resources/measuring-incrementality-with-ghost-ads-at-doordash",
        "落地启发": "ghost ads/holdout 构造真实曝光反事实，用 iROAS 和 bias audit 判断广告增量。",
    },
    "synthetic_merchant_subsidy_margin_uplift_6k": {
        "来源质量": "工业抽象/平台补贴经验",
        "来源标题": "Marketplace subsidy / merchant cofund scenario abstraction",
        "来源链接": "docs/UPLIFT_INDUSTRIAL_SCENARIO_LAB.md",
        "落地启发": "把商家共投拆成平台毛利、商家增量利润、自然需求蚕食和 spillover guardrail。",
    },
    "synthetic_llm_routing_uplift_8k": {
        "来源质量": "论文/官方开源",
        "来源标题": "FrugalGPT / RouteLLM cost-quality routing",
        "来源链接": "https://arxiv.org/abs/2305.05176; https://github.com/lm-sys/RouteLLM",
        "落地启发": "把强模型升级看作 treatment，用质量 uplift - 成本 - 延迟做 policy objective。",
    },
    "synthetic_llm_multi_action_routing_9k": {
        "来源质量": "论文/官方开源/系统抽象",
        "来源标题": "FrugalGPT cascade + RouteLLM router + RAG/tool/human review escalation",
        "来源链接": "https://arxiv.org/abs/2305.05176; https://github.com/lm-sys/RouteLLM; docs/LLM_ROUTING_POLICY_PLAYBOOK.md",
        "落地启发": "把 LLM routing 从 cheap/strong 二元升级到 strong/RAG/tool/human 多动作 policy，并用风险、延迟和预算约束做上线护栏。",
    },
    "synthetic_ecommerce_multi_coupon_uplift_6k": {
        "来源质量": "工业抽象/论文启发",
        "来源标题": "Multi-valued coupon and revenue uplift scenario abstraction",
        "来源链接": "docs/UPLIFT_INDUSTRIAL_CASE_EXPANSION.md",
        "落地启发": "把优惠券从二元 send/no-send 扩展成券面额、门槛、渠道和成本约束下的多 treatment / dose-response 问题。",
    },
    "synthetic_crm_overlap_journey_uplift_6k": {
        "来源质量": "工业抽象/论文启发",
        "来源标题": "Overlapping customer journeys uplift scenario abstraction",
        "来源链接": "docs/UPLIFT_INDUSTRIAL_CASE_EXPANSION.md",
        "落地启发": "把多旅程触达拆成 pure/global uplift、疲劳、退订、蚕食和长期价值。",
    },
    "synthetic_continuous_bid_discount_uplift_5k": {
        "来源质量": "工业抽象/continuous-treatment 论文启发",
        "来源标题": "Continuous bid / discount dose-response scenario abstraction",
        "来源链接": "docs/UPLIFT_MODEL_OPTIMIZATION_PATH.md",
        "落地启发": "把出价、折扣、补贴金额看作连续 treatment，用边际 uplift 与边际成本决定强度。",
    },
    "synthetic_geo_holdout_incrementality_uplift_6k": {
        "来源质量": "官方论文/开源启发",
        "来源标题": "Google geo experiments / Meta GeoLift incrementality scenario abstraction",
        "来源链接": "https://research.google/pubs/pub38355; https://github.com/facebookincubator/GeoLift",
        "落地启发": "把离线 uplift 模型接到 geo holdout、switchback 和 synthetic control 风格的线上增量验证。",
    },
    "synthetic_game_ops_gift_uplift_6k": {
        "来源质量": "工业抽象/游戏个性化实验启发",
        "来源标题": "Game personalization / retention intervention scenario abstraction",
        "来源链接": "docs/UPLIFT_INDUSTRIAL_CASE_EXPANSION.md",
        "落地启发": "把礼包、首充、energy gift 看成有成本和 backlash 风险的 retention/LTV treatment。",
    },
    "synthetic_fintech_credit_line_uplift_6k": {
        "来源质量": "工业抽象/金融风控与 uplift 结合",
        "来源标题": "Credit-line / APR offer risk-adjusted uplift scenario abstraction",
        "来源链接": "docs/UPLIFT_INDUSTRIAL_CASE_EXPANSION.md",
        "落地启发": "把授信、利率优惠和免息券转成增量收入减 capital/default risk 的 policy value。",
    },
    "synthetic_customer_service_escalation_uplift_6k": {
        "来源质量": "工业抽象/客服 routing 与 LLM routing 启发",
        "来源标题": "Customer support human escalation cost-quality uplift scenario abstraction",
        "来源链接": "docs/UPLIFT_INDUSTRIAL_CASE_EXPANSION.md",
        "落地启发": "把升级人工看成 treatment，目标是 CSAT/留存增量覆盖人力与排队成本。",
    },
    "synthetic_healthcare_followup_uplift_6k": {
        "来源质量": "医疗 HTE/随访干预论文启发",
        "来源标题": "Care-management and patient-outreach HTE scenario abstraction",
        "来源链接": "docs/UPLIFT_INDUSTRIAL_CASE_EXPANSION.md",
        "落地启发": "把短信、护士电话、社区随访做成带 clinical risk guardrail 的 uplift 决策。",
    },
    "synthetic_saas_retention_offer_uplift_6k": {
        "来源质量": "B2B churn/uplift 论文与 SaaS 续费业务抽象",
        "来源标题": "B2B retention / renewal offer uplift scenario abstraction",
        "来源链接": "docs/UPLIFT_INDUSTRIAL_CASE_EXPANSION.md",
        "落地启发": "把折扣、CSM call、training 和 feature credit 转成 ARR 加权续费增量与折扣依赖风险。",
    },
}


def _scenario_cn_name(dataset_id: str, fallback: str | None = None) -> str:
    resume_names = {row["dataset_id"]: row["场景"] for row in RESUME_SCENARIOS_CN}
    if dataset_id in resume_names:
        return resume_names[dataset_id]
    return EXTENSION_SCENARIO_CN.get(dataset_id, fallback or dataset_id)


OPTIMIZATION_STAGES = [
    ("V0", "业务规则 / 转化率模型", "先用规则或 P(Y=1|X) 识别高响应人群。", "容易投给 sure thing，浪费补贴或广告预算。"),
    ("V1", "T/S learner baseline", "用 S/T learner 建第一个 uplift baseline，统一 y0/y1/uplift 输出。", "快速稳定，但可能低估异质性或受 treatment bias 影响。"),
    ("V2", "X/DR/R learner", "引入 pseudo-outcome、orthogonalization 或 doubly robust，降低偏差。", "适合观测数据和非随机投放日志。"),
    ("V3", "CausalForest / uplift forest", "增强异质性解释、分群稳定性和业务可解释规则。", "需要检查 leaf/support，避免小样本叶子过拟合。"),
    ("V4", "深度模型 / treatment interaction", "尝试 EFIN、DESCN、CFRNet、DragonNet、ContrastiveUpliftNet。", "必须和 LightGBM baseline 对照，不能只因为模型复杂就上线。"),
    ("V5", "policy value / ROI / budget optimizer", "把 uplift score 转成 Top-K、ROI、iROAS、预算分配和风险 guardrail。", "这是从模型分数到业务决策的关键层。"),
    ("V6", "online A/B / holdout / regression gate", "通过随机 holdout、bootstrap、截图、evidence pack 和回归门守住上线风险。", "离线 QINI/AUUC 不能代替线上增量实验。"),
]


def resume_scenario_cards_cn() -> list[dict[str, Any]]:
    return [dict(row) for row in RESUME_SCENARIOS_CN]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _readiness_score(level: Any) -> float:
    text = str(level or "").lower()
    if "ready" in text:
        return 100.0
    if "pilot" in text or "shadow" in text:
        return 75.0
    if "research" in text:
        return 45.0
    return 30.0


def _normalized(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == 0:
        return pd.Series([0.0] * len(series), index=series.index)
    min_value = float(numeric.min(skipna=True))
    max_value = float(numeric.max(skipna=True))
    filled = numeric.fillna(min_value)
    if abs(max_value - min_value) < 1e-12:
        return pd.Series([50.0] * len(series), index=series.index)
    return ((filled - min_value) / (max_value - min_value) * 100.0).clip(0.0, 100.0)


def _score_compare_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    df = pd.DataFrame(rows)
    scored_parts = []
    for _, part in df.groupby("dataset_id", dropna=False):
        view = part.copy()
        view["policy_norm"] = _normalized(view.get("Top10 policy value", pd.Series(index=view.index)))
        view["roi_norm"] = _normalized(view.get("ROI / iROAS proxy", pd.Series(index=view.index)))
        view["qini_norm"] = _normalized(view.get("QINI", pd.Series(index=view.index)))
        view["top10_norm"] = _normalized(view.get("Top10 observed uplift", pd.Series(index=view.index)))
        view["readiness_norm"] = view["Readiness"].map(_readiness_score)
        view["综合分"] = (
            0.35 * view["policy_norm"]
            + 0.20 * view["roi_norm"]
            + 0.20 * view["qini_norm"]
            + 0.15 * view["top10_norm"]
            + 0.10 * view["readiness_norm"]
        ).round(2)
        judgments = []
        reasons = []
        for _, row in view.iterrows():
            policy_value = pd.to_numeric(row.get("Top10 policy value"), errors="coerce")
            roi_value = pd.to_numeric(row.get("ROI / iROAS proxy"), errors="coerce")
            calibration_value = pd.to_numeric(row.get("Calibration MAE"), errors="coerce")
            weak_overlap = pd.to_numeric(row.get("Weak overlap rate"), errors="coerce")
            oracle_capture = pd.to_numeric(row.get("Oracle policy capture"), errors="coerce")
            readiness = str(row.get("Readiness") or "")
            if pd.notna(policy_value) and float(policy_value) > 0 and "Research" not in readiness:
                judgments.append("可进入 controlled rollout / shadow")
            elif pd.notna(policy_value) and float(policy_value) > 0:
                judgments.append("有收益信号，先补 overlap/calibration 证据")
            else:
                judgments.append("暂不建议上线，先查成本/overlap/calibration")
            reasons.append(
                "综合分按 Top-K policy value、ROI/iROAS、QINI、Top10 uplift 和 readiness 加权；"
                f"policy={policy_value if pd.notna(policy_value) else 'NA'}，"
                f"ROI={roi_value if pd.notna(roi_value) else 'NA'}，readiness={readiness or 'NA'}。"
                f"上线前还要看 calibration MAE={calibration_value if pd.notna(calibration_value) else 'NA'}，"
                f"weak overlap={weak_overlap if pd.notna(weak_overlap) else 'NA'}，"
                f"oracle policy capture={oracle_capture if pd.notna(oracle_capture) else 'NA'}。"
            )
        view["上线判断"] = judgments
        view["排序理由"] = reasons
        view = view.drop(columns=["policy_norm", "roi_norm", "qini_norm", "top10_norm", "readiness_norm"], errors="ignore")
        scored_parts.append(view)
    return pd.concat(scored_parts, ignore_index=True).to_dict("records")


def resume_scenario_compare_rows() -> list[dict[str, Any]]:
    payload = _read_json(REPORTS / "industrial_scenario_compare_smoke_latest.json")
    scenario_names = {row["dataset_id"]: row["场景"] for row in RESUME_SCENARIOS_CN}
    rows = []
    for row in payload.get("runs", []):
        dataset_id = row.get("dataset_id")
        if dataset_id not in scenario_names:
            continue
        rows.append(
            {
                "场景": scenario_names[dataset_id],
                "dataset_id": dataset_id,
                "模型": row.get("model"),
                "状态": row.get("status"),
                "QINI": row.get("qini"),
                "AUUC": row.get("auuc"),
                "Top10 observed uplift": row.get("top10_observed_uplift"),
                "Calibration MAE": row.get("calibration_mae"),
                "Weak overlap rate": row.get("weak_overlap_rate"),
                "Trim@0.05 kept": row.get("trim05_kept_fraction"),
                "Oracle top10 recall": row.get("oracle_top10_recall"),
                "Oracle policy capture": row.get("oracle_policy_top10_capture"),
                "Top10 policy value": row.get("industrial_top10_policy_value_sum"),
                "ROI / iROAS proxy": row.get("industrial_top10_roi_proxy"),
                "Readiness": row.get("readiness_level"),
                "业务判断": row.get("business_decision"),
                "指标类型": row.get("frontier_metric_schema"),
                "run_id": row.get("run_id"),
                "metrics_path": row.get("metrics_path"),
            }
        )
    return _score_compare_rows(rows)


def industrial_extension_summary_rows_cn() -> list[dict[str, Any]]:
    resume_ids = {row["dataset_id"] for row in RESUME_SCENARIOS_CN}
    out = []
    for row in scenario_dataset_summary():
        dataset_id = row.get("dataset_id")
        if dataset_id in resume_ids:
            continue
        source = EXTENSION_SOURCE_EVIDENCE.get(str(dataset_id), {})
        out.append(
            {
                "场景": _scenario_cn_name(str(dataset_id), row.get("scenario")),
                "dataset_id": dataset_id,
                "状态": row.get("status"),
                "来源质量": source.get("来源质量", "本地工业场景抽象"),
                "来源标题": source.get("来源标题", "DeepUplift scenario lab"),
                "来源链接": source.get("来源链接", "docs/UPLIFT_INDUSTRIAL_SCENARIO_LAB.md"),
                "落地启发": source.get("落地启发", "把业务问题转成 treatment/outcome/cost/value/risk schema。"),
                "样本数": row.get("rows"),
                "Treatment 占比": row.get("treatment_rate"),
                "Outcome 均值": row.get("outcome_rate"),
                "P90 policy value": row.get("policy_value_p90"),
                "推荐模型": row.get("recommended_models"),
                "Policy 目标": row.get("policy_objective"),
                "核心指标": row.get("primary_metrics"),
            }
        )
    return out


def industrial_extension_compare_rows_cn() -> list[dict[str, Any]]:
    payload = _read_json(REPORTS / "industrial_scenario_compare_smoke_latest.json")
    resume_ids = {row["dataset_id"] for row in RESUME_SCENARIOS_CN}
    cards = {row["dataset_id"]: row for row in scenario_cards()}
    rows = []
    for row in payload.get("runs", []):
        dataset_id = row.get("dataset_id")
        if not dataset_id or dataset_id in resume_ids:
            continue
        card = cards.get(dataset_id, {})
        rows.append(
            {
                "场景": _scenario_cn_name(str(dataset_id), card.get("scenario")),
                "dataset_id": dataset_id,
                "模型": row.get("model"),
                "状态": row.get("status"),
                "QINI": row.get("qini"),
                "AUUC": row.get("auuc"),
                "Top10 observed uplift": row.get("top10_observed_uplift"),
                "Calibration MAE": row.get("calibration_mae"),
                "Weak overlap rate": row.get("weak_overlap_rate"),
                "Trim@0.05 kept": row.get("trim05_kept_fraction"),
                "Oracle top10 recall": row.get("oracle_top10_recall"),
                "Oracle policy capture": row.get("oracle_policy_top10_capture"),
                "Top10 policy value": row.get("industrial_top10_policy_value_sum"),
                "ROI / iROAS proxy": row.get("industrial_top10_roi_proxy"),
                "Readiness": row.get("readiness_level"),
                "业务判断": row.get("business_decision"),
                "指标类型": row.get("frontier_metric_schema"),
                "run_id": row.get("run_id"),
                "metrics_path": row.get("metrics_path"),
            }
        )
    return _score_compare_rows(rows)


def llm_routing_ope_scenario_rows_cn() -> list[dict[str, Any]]:
    """Chinese OPE rows for the scenario workbench.

    The Policy page owns the calculator; this function makes the same evidence
    first-class in the scenario workbench so LLM routing is reviewed like an
    industrial uplift scenario rather than a standalone toy.
    """

    summary = llm_routing_ope_summary()
    rows = []
    for row in summary.get("rows") or []:
        rows.append(
            {
                "策略": row.get("policy"),
                "覆盖率": row.get("coverage_rate"),
                "有效样本量": row.get("effective_sample_size"),
                "DM value": row.get("direct_method_value"),
                "IPS value": row.get("ips_value"),
                "SNIPS value": row.get("snips_value"),
                "DR value": row.get("doubly_robust_value"),
                "Oracle net value": row.get("oracle_net_value"),
                "上线就绪": row.get("launch_readiness"),
                "面试解释": row.get("interview_note"),
                "证据": summary.get("path"),
            }
        )
    return rows


def llm_routing_action_coverage_rows_cn() -> list[dict[str, Any]]:
    rows = []
    for row in llm_routing_action_coverage_rows():
        status = row.get("support_status")
        rows.append(
            {
                "历史动作": row.get("logged_action"),
                "样本数": row.get("rows"),
                "占比": row.get("share"),
                "平均 logged propensity": row.get("mean_logged_propensity"),
                "support 状态": status,
                "诊断结论": row.get("recommendation"),
                "上线含义": (
                    "可做 OPE 和 controlled rollout"
                    if status == "strong"
                    else "先保守上线或补 exploration"
                    if status == "thin"
                    else "不要信任该动作的离线 OPE，先补随机探索"
                ),
            }
        )
    return rows


def llm_routing_bandit_drift_rows_cn() -> list[dict[str, Any]]:
    rows = []
    for row in llm_routing_bandit_drift_rows():
        delta = row.get("adaptive_minus_fixed")
        rows.append(
            {
                "周期": row.get("period"),
                "漂移事件": row.get("event"),
                "单次 routing 净增益": row.get("gain_per_route"),
                "固定阈值比例": row.get("fixed_fraction"),
                "自适应比例": row.get("adaptive_fraction"),
                "固定阈值净收益": row.get("fixed_threshold_net"),
                "Bandit 净收益": row.get("adaptive_bandit_net"),
                "Bandit - 固定阈值": delta,
                "预算": row.get("budget"),
                "诊断": "自适应策略优于固定阈值" if delta is not None and float(delta) > 0 else "固定阈值更稳，先不要扩大探索",
                "面试讲法": row.get("interview_line"),
            }
        )
    return rows


def research_evidence_gate_rows_cn() -> list[dict[str, Any]]:
    """Source-backed research-agent design patterns useful for DeepUplift Agent."""

    return [
        {
            "参考框架": "LangGraph Open Deep Research",
            "可信来源": "官方 GitHub / LangChain",
            "可借鉴机制": "计划、检索、压缩、最终报告、评测基准和多模型/多搜索工具配置",
            "落到本平台": "把外部 uplift paper/工业案例进入 Knowledge 前先过 source quality、claim-to-artifact、UI proof 和 smoke proof。",
            "UI 证据": "Evidence -> Research Evidence Gate；Knowledge -> source quality；Agent -> Evidence Card",
            "链接": "https://github.com/langchain-ai/open_deep_research",
        },
        {
            "参考框架": "GPT Researcher",
            "可信来源": "官方 GitHub / Apache-2.0",
            "可借鉴机制": "多源聚合、引用报告、web+local research、并行化 agent work、可定制 domain-specific research agent",
            "落到本平台": "把研究输出从自然语言报告升级成模型卡、场景卡、公式卡、Evidence artifact 和回归脚本。",
            "UI 证据": "Evidence -> Research Evidence Gate；Story -> Interview Evidence；Agent -> source-backed answer",
            "链接": "https://github.com/assafelovic/gpt-researcher",
        },
        {
            "参考框架": "Local / lightweight deep researcher variants",
            "可信来源": "官方开源仓库",
            "可借鉴机制": "本地网页研究、递归搜索、来源 drill-down 和轻量 UI",
            "落到本平台": "适合后续做本地 paper 文件夹索引，但当前优先保留手工审核门禁，避免低质量来源污染模型目录。",
            "UI 证据": "Evidence -> Research Evidence Gate；docs/DEEPUplift_RESEARCH_EVIDENCE_GATE.md",
            "链接": "https://github.com/langchain-ai/local-deep-researcher",
        },
    ]


def resume_scenario_best_rows() -> list[dict[str, Any]]:
    rows = resume_scenario_compare_rows()
    if not rows:
        return []
    df = pd.DataFrame(rows)
    out = []
    for _, part in df.groupby("dataset_id"):
        view = part.copy()
        for col in ["综合分", "Top10 policy value", "QINI", "Top10 observed uplift"]:
            view[col] = pd.to_numeric(view[col], errors="coerce")
        view = view.sort_values(["综合分", "Top10 policy value", "QINI", "Top10 observed uplift"], ascending=False, na_position="last")
        out.append(view.iloc[0].to_dict())
    return out


def rollout_risk_rows_cn() -> list[dict[str, Any]]:
    extension_rows = industrial_extension_compare_rows_cn()
    extension_best = []
    if extension_rows:
        extension_df = pd.DataFrame(extension_rows)
        if "综合分" in extension_df.columns:
            extension_df["综合分"] = pd.to_numeric(extension_df["综合分"], errors="coerce")
        for _, part in extension_df.groupby("dataset_id"):
            view = part.sort_values("综合分", ascending=False, na_position="last") if "综合分" in part.columns else part
            extension_best.append(view.iloc[0].to_dict())
    candidates = resume_scenario_best_rows() + extension_best
    rows = []
    for row in candidates:
        policy_value = pd.to_numeric(row.get("Top10 policy value"), errors="coerce")
        calibration = pd.to_numeric(row.get("Calibration MAE"), errors="coerce")
        weak_overlap = pd.to_numeric(row.get("Weak overlap rate"), errors="coerce")
        readiness = str(row.get("Readiness") or "")
        reasons = []
        if pd.notna(policy_value) and float(policy_value) <= 0:
            reasons.append("Top10 policy value <= 0，离线净收益信号不足")
        if "Research" in readiness:
            reasons.append("Readiness 仍是 Research only，需要 shadow/holdout 证据")
        if pd.notna(calibration) and float(calibration) > 0.20:
            reasons.append("Calibration MAE 偏高，Top-K 分数排序需要校准")
        if pd.notna(weak_overlap) and float(weak_overlap) > 0.05:
            reasons.append("Weak overlap rate 偏高，需缩小 common support 或补随机探索")
        if not reasons:
            reasons.append("可进入 controlled rollout，但仍需 bootstrap CI 和线上 holdout")
        rows.append(
            {
                "场景": row.get("场景"),
                "模型": row.get("模型"),
                "Top10 policy value": row.get("Top10 policy value"),
                "ROI / iROAS proxy": row.get("ROI / iROAS proxy"),
                "Readiness": readiness,
                "Calibration MAE": row.get("Calibration MAE"),
                "Weak overlap rate": row.get("Weak overlap rate"),
                "上线解释": "；".join(reasons),
            }
        )
    return rows


def resume_policy_rows_cn() -> list[dict[str, Any]]:
    scenario_names = {row["dataset_id"]: row["场景"] for row in RESUME_SCENARIOS_CN}
    rows = []
    for row in resume_scenario_policy_benchmarks():
        if row.get("dataset_id") not in scenario_names:
            continue
        rows.append(
            {
                "场景": scenario_names[row["dataset_id"]],
                "Top-K": row.get("top_fraction"),
                "样本数": row.get("rows"),
                "policy value": row.get("policy_value_sum"),
                "成本": row.get("cost_sum"),
                "业务价值": row.get("value_sum"),
                "ROI / iROAS proxy": row.get("roi_proxy"),
                "阈值": row.get("threshold_oracle_policy_value"),
                "平均 true uplift": row.get("mean_true_uplift"),
                "风险字段": "; ".join([key for key in row if key.startswith("mean_")]),
            }
        )
    return rows


def resume_optimization_path_rows(dataset_id: str | None = None) -> list[dict[str, Any]]:
    selected = [row for row in RESUME_SCENARIOS_CN if dataset_id in (None, row["dataset_id"])]
    rows = []
    for scenario in selected:
        for stage_id, stage_name, action, why in OPTIMIZATION_STAGES:
            rows.append(
                {
                    "场景": scenario["场景"],
                    "阶段": stage_id,
                    "优化层": stage_name,
                    "做什么": action,
                    "为什么": why,
                    "对应证据": scenario["证据"],
                }
            )
    return rows


def frontier_model_audit_rows_cn() -> list[dict[str, Any]]:
    rows = []
    for item in frontier_model_capabilities():
        status = item.get("integration_status", "")
        if "ready" in status:
            status_cn = "ready：当前环境可训练/评估或已有 evaluator/policy 闭环"
        elif "guarded" in status:
            status_cn = "guarded：已注册或有适配路径，但依赖/运行时需要证据门"
        elif "optional" in status:
            status_cn = "optional：需要额外依赖或辅助环境"
        else:
            status_cn = "research backlog：先保留模型卡、来源和接入路线"
        rows.append(
            {
                "优先级": item.get("priority"),
                "模型/能力": item.get("capability"),
                "层级": item.get("layer"),
                "当前状态": status_cn,
                "是否业内先进": "是，工业核心/前沿方向" if item.get("priority") in {"P0", "P1", "P2"} else "待验证",
                "为什么可用或受限": item.get("training_status"),
                "依赖/风险": item.get("risks"),
                "代码路径": item.get("adapter_path"),
                "来源": item.get("source_title"),
                "source_url": item.get("source_url"),
                "适合场景": item.get("business_scenarios"),
                "下一步": item.get("recommended_action"),
            }
        )
    return rows


def model_limitations_rows_cn() -> list[dict[str, Any]]:
    rows = []
    for row in frontier_model_audit_rows_cn():
        status = row["当前状态"]
        if "ready" not in status:
            rows.append(row)
    return rows


def resume_workbench_agent_reply(prompt: str) -> str | None:
    text = prompt.lower()
    if not any(
        key in text
        for key in [
            "履历",
            "从0到1",
            "从 0-1",
            "相对最优",
            "模型是不是业内先进",
            "业内先进",
            "依赖限制",
            "框架限制",
            "为什么有些模型",
            "每个场景",
            "优化路径",
            "模型对比",
        ]
    ):
        return None
    best = resume_scenario_best_rows()
    limitations = model_limitations_rows_cn()
    lines = [
        "### 中文履历场景工作台回答",
        "",
        "当前平台不是只堆模型名，而是把百度外卖、滴滴和 Shopee 的业务场景做成了可跑数据、模型对比、policy value、预算决策和 evidence。",
        "",
        "#### 每个履历场景的当前最佳可跑模型",
    ]
    if best:
        for row in best:
            lines.append(
                f"- {row['场景']}：当前 compare 最优 `{row['模型']}`，Top10 policy value `{row.get('Top10 policy value')}`，Readiness `{row.get('Readiness')}`。"
            )
    else:
        lines.append("- 还没有最新 compare evidence，请运行 `scripts/smoke_industrial_scenario_compare.py --rows 700`。")
    lines.extend(
        [
            "",
            "#### 是否已经业内先进",
            "",
            "P0 层已经覆盖工业核心：S/T/X/R/DR learner、LightGBM、EconML DML/CausalForest、scikit-uplift baseline。P1 层已有 TARNet/CFRNet/DragonNet/DESCN/EFIN/ContrastiveUpliftNet 等深度 uplift。P2 层把 ECUP、delayed feedback、continuous treatment、budget-constrained policy、LLM routing 以 evaluator/policy/plugin backlog 形式接入。",
            "",
            "#### 当前限制",
        ]
    )
    for row in limitations[:8]:
        lines.append(f"- `{row['模型/能力']}`：{row['当前状态']}；限制：{row['依赖/风险']}")
    lines.extend(
        [
            "",
            "UI 证据：`场景工作台`、`Models`、`Policy`、`Evidence`。",
            "代码证据：`deepuplift/core/resume_scenario_workbench.py`、`deepuplift/core/resume_scenario_policy.py`、`scripts/smoke_industrial_scenario_compare.py`。",
        ]
    )
    return "\n".join(lines)
