from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd


MODEL_FAILURE_MODES: list[dict[str, str]] = [
    {
        "family": "S-Learner",
        "risk": "Treatment effect 被主效应淹没",
        "symptom": "整体 AUC/LogLoss 很好，但 uplift@K、QINI 或 oracle top-k recall 很弱。",
        "root_cause": "把 treatment 当普通特征时，强自然转化率特征会压过弱 uplift 信号。",
        "diagnostic": "比较 S-Learner 与 T/X/DR learner；检查 treatment feature importance 和 Top-K true_uplift。",
        "misleading_metric": "普通 CVR AUC、overall accuracy。",
        "fix": "先用 T/X/DR/R learner 或 interaction features；对弱 uplift 信号用 policy/top-k 指标而不是 CVR 指标。",
        "ui_evidence": "Models -> Model Risk Matrix; Evaluation -> Failure Modes",
        "interview_line": "S-Learner 是强 baseline，但当 uplift 信号很弱时，必须证明它没有只学自然转化率。",
    },
    {
        "family": "T-Learner",
        "risk": "Treatment/control 不均衡导致方差大",
        "symptom": "两个 outcome model 在少数 treatment arm 上不稳定，Top-K 波动大，bootstrap CI 很宽。",
        "root_cause": "T-Learner 分开训练两套模型，少数 arm 支持不足会放大估计方差。",
        "diagnostic": "看 treatment rate、每个 arm 的样本量、bootstrap CI、overlap trimming 前后变化。",
        "misleading_metric": "单次 QINI 均值。",
        "fix": "使用 X/DR/R learner、shared representation、calibration 或增加随机探索样本。",
        "ui_evidence": "Diagnostics -> Treatment Balance; Evaluation -> Bootstrap CI",
        "interview_line": "T-Learner 简洁，但样本不均衡时我会用 DR/R 或 shared model 降低方差。",
    },
    {
        "family": "X-Learner",
        "risk": "Pseudo outcome 依赖 imputation 和 propensity",
        "symptom": "在 treatment imbalance 下可能比 T-Learner 好，但 propensity 错或 imputation 差时会反向。",
        "root_cause": "X-Learner 先补反事实，再用 propensity 加权融合；两个环节都可能带偏。",
        "diagnostic": "比较 pseudo-outcome 分布、propensity 分布、X vs DR/R 的 Top-K policy value。",
        "misleading_metric": "AUUC 单点提升。",
        "fix": "用 DR/R 做对照，propensity clipping，按 overlap 支持域报告结果。",
        "ui_evidence": "Evaluation -> overlap / trimming; Models -> Scenario Model Matrix",
        "interview_line": "X-Learner 适合不均衡，但不是万能，必须和 propensity 质量一起讲。",
    },
    {
        "family": "DR / R Learner",
        "risk": "Nuisance model 错误或 propensity 极端",
        "symptom": "pseudo outcome 出现极端值，少数样本主导 policy value，overlap trimming 后结论大变。",
        "root_cause": "DR/R 依赖 outcome nuisance 和 propensity nuisance；当 e(x) 接近 0/1，残差或权重会爆。",
        "diagnostic": "propensity histogram、weak overlap rate、trim sensitivity、pseudo outcome quantile。",
        "misleading_metric": "未裁剪的 DR policy value。",
        "fix": "propensity clipping/trimming、交叉拟合、更稳健 base learner、限制部署到 common support。",
        "ui_evidence": "Diagnostics -> Overlap; Evaluation -> DR/R formula cards",
        "interview_line": "DR/R 是工业核心，但我会把 overlap 当作上线闸门，而不是只相信 doubly robust 名字。",
    },
    {
        "family": "CausalForest / ForestDR",
        "risk": "叶子支持不足和置信区间过宽",
        "symptom": "局部 uplift 很极端，但 leaf support 小、CI 宽，segment 解释不稳定。",
        "root_cause": "异质性森林在 weak overlap 或小叶子上容易把噪声解释成 heterogeneity。",
        "diagnostic": "看 leaf/support proxy、bootstrap CI、segment stability、overlap trimming。",
        "misleading_metric": "局部最高 uplift segment。",
        "fix": "提高 min leaf、honest split、只解释稳定 segment，用 DR/R 或 T/X baseline 对照。",
        "ui_evidence": "Models -> CausalForest cards; Evaluation -> Bootstrap CI",
        "interview_line": "CausalForest 强在异质性解释，但 CI 宽时我会降级成探索发现，不直接上线。",
    },
    {
        "family": "Uplift Tree / Forest",
        "risk": "小叶子过拟合和噪声分裂",
        "symptom": "训练集 uplift 很高，换 bootstrap 或新样本后 Top-K 大幅回落。",
        "root_cause": "uplift split criterion 容易偏向高噪声、小样本叶子。",
        "diagnostic": "bootstrap stability、leaf min support、segment lift variance。",
        "misleading_metric": "训练集或单次验证集分箱 uplift。",
        "fix": "剪枝、最小叶子约束、校准、和 LightGBM DR/R baseline 做 shadow compare。",
        "ui_evidence": "Models -> guarded tree/forest; Evidence -> scenario compare smoke",
        "interview_line": "可解释树适合营销规则，但必须用稳定性和叶子支持来治理。",
    },
    {
        "family": "Class Transformation",
        "risk": "随机实验假设强，观测数据下偏差大",
        "symptom": "随机样本上简单有效，观测投放日志中会把 selection bias 当 uplift。",
        "root_cause": "转换标签隐含 treatment assignment 近似随机，否则标签目标会偏。",
        "diagnostic": "看 treatment/control balance、propensity overlap、placebo treatment。",
        "misleading_metric": "转换标签 AUC 或 QINI。",
        "fix": "仅在随机实验或 overlap 好时使用；观测数据转 DR/R/DML 或 propensity-aware 评估。",
        "ui_evidence": "Diagnostics -> SSB; Models -> transformation model cards",
        "interview_line": "Class transformation 是教学和实验 baseline，不能直接套到有偏投放日志。",
    },
    {
        "family": "TARNet / CFRNet",
        "risk": "表示学习不平衡和 IPM/MMD 权重难调",
        "symptom": "factual loss 下降但 treatment/control 表示仍分离，反事实预测不稳。",
        "root_cause": "深度表示可能拟合事实样本，却没有学到平衡的 common representation。",
        "diagnostic": "balance loss 曲线、treatment discriminator、CFRNet differentiable balance smoke。",
        "misleading_metric": "只看 train loss 或 factual BCE/MSE。",
        "fix": "调 IPM/MMD 权重、early stopping、和 P0 LightGBM DR/R baseline 对照。",
        "ui_evidence": "Models -> Deep Uplift Loss Catalog; Evidence -> CFRNet balance smoke",
        "interview_line": "CFRNet 的关键不是网络名字，而是 representation balance 是否真正可微并改善反事实泛化。",
    },
    {
        "family": "DragonNet",
        "risk": "Targeted regularization 依赖 propensity/outcome head 稳定",
        "symptom": "propensity head 极端或 targeted reg 权重过高时，uplift 分数校准变差。",
        "root_cause": "DragonNet 同时学习 outcome、propensity 和 targeted regularization，多个 loss 尺度需要平衡。",
        "diagnostic": "propensity distribution、calibration MAE、ablation without targeted reg。",
        "misleading_metric": "单一 AUUC 或训练 loss。",
        "fix": "loss weight ablation、propensity clipping、calibration gate、和 TARNet/CFRNet 对照。",
        "ui_evidence": "Models -> Network Architecture Catalog; Evaluation -> calibration",
        "interview_line": "DragonNet 适合讲 targeted regularization，但我会用 ablation 和 calibration 证明它没有过正则。",
    },
    {
        "family": "DESCN / EFIN",
        "risk": "Treatment feature、样本量和调参敏感",
        "symptom": "在复杂 coupon/creative/treatment feature 场景表现好，但小数据或 feature 噪声下不稳。",
        "root_cause": "工业深度 interaction 网络依赖足够 treatment 变化和高质量 user-treatment 特征。",
        "diagnostic": "treatment feature coverage、ablation、Top-K policy value、segment stability。",
        "misleading_metric": "论文模型名或单场景最高 QINI。",
        "fix": "先跑 P0 baseline，再用深度模型做增益验证；缺少 treatment feature 时 guarded。",
        "ui_evidence": "Models -> P1 deep plugins; Scenario -> model compare",
        "interview_line": "EFIN/DESCN 是高级插件，不是默认答案；要证明 user-treatment interaction 值得复杂度。",
    },
    {
        "family": "UTBoost",
        "risk": "依赖和 split criterion 差异需要对照",
        "symptom": "安装/运行时可能受环境影响，不同 uplift split criterion 与 LightGBM DR/R 排序差异大。",
        "root_cause": "工业 uplift GBDT 是独立实现，依赖和分裂准则都需要 evidence gate。",
        "diagnostic": "guarded adapter smoke、tiny train、和 LightGBM DR/R no-UI compare。",
        "misleading_metric": "未经过本地 smoke 的框架支持声明。",
        "fix": "保持 guarded，安装后跑 tiny train + scenario compare，再晋级 demo-ready。",
        "ui_evidence": "Models -> Open Source Framework Audit; Evidence -> UTBoost smoke",
        "interview_line": "我会明确 UTBoost 是工业先进 GBDT 插件，但不会把未验证依赖伪装成 ready。",
    },
    {
        "family": "Multi / Continuous Treatment",
        "risk": "Positivity 难满足、剂量响应不单调、成本非线性",
        "symptom": "高强度 treatment 看起来 uplift 高，但扣成本后 policy value 为负。",
        "root_cause": "券面额、出价、折扣和补贴金额不是二元变量，强度越大成本也越高，还可能饱和。",
        "diagnostic": "dose-response curve、treatment_strength overlap、marginal ROI、policy value by dose。",
        "misleading_metric": "二元化后的 QINI/AUUC。",
        "fix": "dose-response learner、budget optimizer、边际 uplift = 边际成本的阈值解释。",
        "ui_evidence": "Dose-Response; Policy -> ROI; Evaluation -> metric misleading cases",
        "interview_line": "连续 treatment 的最优点不是最大补贴，而是边际增量收益等于边际成本。",
    },
    {
        "family": "Delayed Feedback / ECUP",
        "risk": "短窗口和全链路选择偏差",
        "symptom": "D1 无效但 D30 有效；click uplift 高但 conversion/profit uplift 低。",
        "root_cause": "标签成熟需要时间，impression-click-conversion 每层都有选择偏差。",
        "diagnostic": "D1/D7/D14/D30 uplift、censored rate、full-funnel stage uplift。",
        "misleading_metric": "当天转化、click uplift、单一 terminal conversion。",
        "fix": "延迟反馈 evaluator、成熟窗口、ECUP full-funnel、stage-specific policy value。",
        "ui_evidence": "Evaluation -> delayed/full-funnel metrics",
        "interview_line": "广告和增长要看成熟窗口与全链路，不能只用当天转化或点击。",
    },
    {
        "family": "LLM Routing Uplift",
        "risk": "探索不足、质量标签噪声、成本/延迟冲突",
        "symptom": "强模型平均更好，但大部分请求的增量质量不足以覆盖成本和延迟。",
        "root_cause": "强模型调用通常非随机，缺少 cheap/strong 反事实；LLM judge 标签也有噪声。",
        "diagnostic": "random exploration bucket、quality uplift、cost-quality policy value、latency penalty。",
        "misleading_metric": "强模型绝对质量分或 win-rate。",
        "fix": "随机探索、DR/R learner、budget-constrained routing、calibration 和 guardrail。",
        "ui_evidence": "Policy -> LLM Routing ROI; Models -> LLM routing uplift",
        "interview_line": "LLM routing 要问的是强模型相对便宜模型的增量质量是否值得成本。",
    },
]


SCENARIO_PITFALLS: list[dict[str, str]] = [
    {
        "scenario": "发券 / 红包",
        "pitfall": "sure thing / sleeping dog / 补贴套利",
        "symptom": "自然转化率高的人拿券后并没有增量，甚至被打扰或套利导致负 uplift。",
        "diagnostic": "uplift type mix、Top-K incremental profit、abuse risk、gross margin vs expected coupon cost。",
        "fix": "用 cost-aware policy value，设置套利风险和毛利 guardrail。",
        "interview_line": "发券不是找最会买的人，而是找会被券改变且净利润为正的人。",
    },
    {
        "scenario": "滴滴 C 端乘客补贴",
        "pitfall": "发单、完单、留存和价格敏感漂移冲突",
        "symptom": "短期发单 uplift 正，但完单/抽佣/长期留存或价格敏感度变差。",
        "diagnostic": "request uplift、completed-order uplift、retention uplift、subsidy ROI、price_sensitivity_drift。",
        "fix": "把 outcome 从发单升级到完单和留存，把补贴依赖作为风险惩罚。",
        "interview_line": "乘客补贴要优化平台净收益，不是只把用户推到点击叫车。",
    },
    {
        "scenario": "滴滴 B 端司机补贴",
        "pitfall": "SUTVA / spillover / 热区迁移",
        "symptom": "某热区司机上线增加，但别的区域供给减少，整体等待时间或匹配率没有改善。",
        "diagnostic": "spillover risk、interference risk、city/zone supply balance、wait-time reduction value。",
        "fix": "用 marketplace policy simulator 和 geo/switchback 评估，不只看单司机 uplift。",
        "interview_line": "B 端补贴是 marketplace 问题，必须讲干扰和供需平衡。",
    },
    {
        "scenario": "滴滴预算分配",
        "pitfall": "历史 ROI 幸存者偏差和边际 ROI 递减",
        "symptom": "按历史 ROI 排序会把预算继续给已饱和城市，新增预算收益下降。",
        "diagnostic": "marginal ROI curve、budget share、cross-market spillover risk。",
        "fix": "用 budget-constrained policy optimizer，按边际 ROI 和供需缺口分配。",
        "interview_line": "预算分配不是 Top-K 用户问题，而是 city/time/c-side/b-side 的约束优化。",
    },
    {
        "scenario": "Shopee 广告 / iROAS",
        "pitfall": "pCTR/pCVR 与 incrementality 混淆",
        "symptom": "广告点击或归因 GMV 高，但 holdout 增量收入和 iROAS 不达标。",
        "diagnostic": "ghost/geo holdout、conversion uplift、incremental GMV、media cost、iROAS。",
        "fix": "用 uplift 和 holdout 验证增量，把 attribution revenue 换成 incremental revenue。",
        "interview_line": "广告模型要回答广告制造了多少额外成交，而不是谁本来就会买。",
    },
    {
        "scenario": "CRM / Push",
        "pitfall": "疲劳、退订、journey 重叠和延迟反馈",
        "symptom": "D1 转化不明显或多 journey 重复触达导致 opt-out 上升。",
        "diagnostic": "D1/D7/D30 uplift、journey overlap、fatigue score、optout uplift、global vs pure lift。",
        "fix": "频控、journey arbitration、delayed feedback evaluator 和 fatigue-adjusted policy value。",
        "interview_line": "增长触达要把长期价值和打扰成本一起建模。",
    },
    {
        "scenario": "推荐干预",
        "pitfall": "用户-item-treatment 交互和负 uplift",
        "symptom": "追加请求或重排对部分用户提升，对另一些用户造成体验下降或成本浪费。",
        "diagnostic": "request uplift、negative uplift rate、segment stability、intervention cost。",
        "fix": "EFIN/DESCN/interaction 模型作为插件，并用 negative uplift guardrail。",
        "interview_line": "推荐干预的 treatment 是动作本身，不能把自然偏好当作增量效果。",
    },
    {
        "scenario": "Marketplace 补贴",
        "pitfall": "两侧市场干扰和自然需求蚕食",
        "symptom": "补贴带来局部订单增长，但蚕食自然需求或挤压另一侧供给。",
        "diagnostic": "direct uplift、spillover risk、cannibalization risk、net marketplace value。",
        "fix": "加入 interference guardrail、geo holdout、multi-treatment 或 dose-response。",
        "interview_line": "平台补贴一定要从系统净价值看，而不是单侧 GMV。",
    },
    {
        "scenario": "LLM routing",
        "pitfall": "强模型过用和反事实缺失",
        "symptom": "强模型平均质量高，但大多数请求升级后的增量质量小于成本和延迟。",
        "diagnostic": "random exploration bucket、quality uplift、model cost、latency penalty、budget routing value。",
        "fix": "用探索数据训练 uplift routing，按 cost-quality policy value 做阈值。",
        "interview_line": "LLM routing 是 uplift，因为问题是强模型相对弱模型多带来多少增量质量。",
    },
    {
        "scenario": "游戏运营礼包",
        "pitfall": "付费鲸鱼 sure thing 与 pay-to-win 反感",
        "symptom": "礼包发给高付费玩家看似 LTV 高，但真实增量小，甚至引发公平性反感。",
        "diagnostic": "incremental LTV、gift cost、churn risk、pay_to_win_backlash_risk、payer segment Top-K mix。",
        "fix": "用 cost-aware uplift 找高挫败但可被礼包挽回的人群，并设置 backlash guardrail。",
        "interview_line": "游戏礼包不是给最会付费的人，而是给会被礼包改变且不会破坏体验的人。",
    },
    {
        "scenario": "金融授信 / 利率优惠",
        "pitfall": "响应率和风险收益混淆",
        "symptom": "高响应用户可能带来更高 default risk，增量收入被 capital/default cost 吃掉。",
        "diagnostic": "incremental interest revenue、default_risk_lift、capital_cost、risk-adjusted policy value。",
        "fix": "把违约风险增量纳入 policy value，并对高风险人群做解释性和合规 guardrail。",
        "interview_line": "金融 uplift 必须讲风险调整后的增量收益，而不是单纯 offer take-up。",
    },
    {
        "scenario": "客服升级人工",
        "pitfall": "升级成本、队列延迟和 CSAT uplift 冲突",
        "symptom": "复杂问题升级后满意度上升，但人力成本和等待时间让净价值为负。",
        "diagnostic": "CSAT uplift、retention_value、escalation_cost、handle_time_penalty、queue capacity。",
        "fix": "把升级人工建成 cost-quality uplift routing，只升级边际满意度覆盖成本的 case。",
        "interview_line": "客服升级人工和 LLM routing 一样，本质是成本约束下的增量质量决策。",
    },
    {
        "scenario": "医疗随访 / 健康管理",
        "pitfall": "高临床风险不等于高可改变性",
        "symptom": "只按 risk score 触达会把资源投给高风险但难以被短信/电话改变的人。",
        "diagnostic": "adherence uplift、no_show_risk、clinical_risk、outreach_cost、safety guardrail。",
        "fix": "使用 CATE/uplift 识别可改变人群，同时对高风险患者保留安全策略。",
        "interview_line": "医疗随访要区分 risk 和 treatment effect，且不能把安全约束完全交给 ROI。",
    },
    {
        "scenario": "SaaS 续费 / 流失挽回",
        "pitfall": "churn risk 排序导致折扣浪费和价格依赖",
        "symptom": "高流失风险账号不一定可挽回，高 ARR 账号可能本来就会续费。",
        "diagnostic": "renewal uplift、ARR-weighted policy value、offer_cost、discount_dependency_risk、bootstrap CI。",
        "fix": "用 uplift 替代 churn score 排序，并把折扣依赖作为长期风险惩罚。",
        "interview_line": "SaaS 挽留不能只按 churn risk，要看干预是否真正改变续费且净 ARR 为正。",
    },
]


DIAGNOSTIC_CASES: list[dict[str, str]] = [
    {
        "case_id": "weak_uplift_signal",
        "case_name": "弱 uplift 信号被主效应淹没",
        "problem_type": "model_failure",
        "mock_signal": "base conversion dominates true_uplift; treatment effect amplitude is tiny",
        "affected_models": "S-Learner, plain CVR model",
        "misleading_metric": "CVR AUC high but uplift@K weak",
        "diagnostic_method": "compare S vs T/X/DR; inspect top10 true_uplift and feature importances",
        "fix_path": "add treatment interactions or move to T/X/DR/R learner",
        "interview_line": "这个 case 说明普通预测做得好，不代表增量排序做得好。",
    },
    {
        "case_id": "imbalanced_treatment",
        "case_name": "Treatment/control 极不均衡",
        "problem_type": "data_design",
        "mock_signal": "treatment_rate around 0.90 and control support is thin",
        "affected_models": "T-Learner, uplift tree, CausalForest",
        "misleading_metric": "single-run QINI without CI",
        "diagnostic_method": "treatment balance, bootstrap CI, arm-level sample size",
        "fix_path": "use X/DR/R learner, reweighting, collect exploration/control samples",
        "interview_line": "这个 case 说明 T-Learner 的简单性会在样本不均衡时变成方差风险。",
    },
    {
        "case_id": "extreme_propensity",
        "case_name": "Propensity 极端导致 DR/IPW 权重爆炸",
        "problem_type": "overlap",
        "mock_signal": "propensity near 0/1 for large population slices",
        "affected_models": "DR-Learner, R-Learner, IPW, CausalForest",
        "misleading_metric": "untrimmed policy value",
        "diagnostic_method": "weak overlap rate, propensity quantiles, trim sensitivity",
        "fix_path": "clip propensity, trim unsupported rows, restrict deployment to common support",
        "interview_line": "这个 case 说明 doubly robust 不等于无条件可靠。",
    },
    {
        "case_id": "negative_uplift_sleeping_dog",
        "case_name": "Sleeping dog / 负 uplift",
        "problem_type": "scenario_pitfall",
        "mock_signal": "high intent users have negative true_uplift due to annoyance or subsidy gaming",
        "affected_models": "CVR model, cost-blind uplift ranking",
        "misleading_metric": "raw conversion rate and attributed GMV",
        "diagnostic_method": "uplift type mix, negative uplift segment, abuse risk",
        "fix_path": "exclude sleeping dogs and add risk penalty to policy value",
        "interview_line": "发券里最怕把 sure thing 和 sleeping dog 当成好人群。",
    },
    {
        "case_id": "high_qini_low_roi",
        "case_name": "QINI 高但 ROI 低",
        "problem_type": "metric_misleading",
        "mock_signal": "true uplift is positive in top rank, but cost exceeds incremental value",
        "affected_models": "all cost-blind ranking models",
        "misleading_metric": "QINI, AUUC, uplift@K",
        "diagnostic_method": "top10 policy value and incremental profit",
        "fix_path": "rank by net value or add cost-aware threshold",
        "interview_line": "这个 case 是面试重点：排序指标不等于上线收益。",
    },
    {
        "case_id": "delayed_feedback_trap",
        "case_name": "D1 无效但 D30 有效",
        "problem_type": "label_window",
        "mock_signal": "converted_d1 uplift near zero, converted_d30 uplift positive",
        "affected_models": "short-window growth/ads models",
        "misleading_metric": "D1 conversion uplift",
        "diagnostic_method": "D1/D7/D14/D30 uplift and censoring rate",
        "fix_path": "use mature label windows and delayed feedback evaluator",
        "interview_line": "短窗口会错杀慢响应用户。",
    },
    {
        "case_id": "full_funnel_click_trap",
        "case_name": "Click uplift 高但 conversion uplift 低",
        "problem_type": "full_funnel",
        "mock_signal": "click uplift positive while conversion uplift is zero or negative",
        "affected_models": "CTR/pCTR ranking, click-optimized uplift",
        "misleading_metric": "click uplift",
        "diagnostic_method": "impression/click/conversion stage uplift",
        "fix_path": "optimize conversion/profit stage or ECUP full-funnel objective",
        "interview_line": "广告点击提升不是最终增量价值。",
    },
    {
        "case_id": "geo_holdout_conflict",
        "case_name": "离线 uplift 好但 geo holdout 不稳定",
        "problem_type": "online_validation",
        "mock_signal": "offline top-k uplift positive; geo holdout lift has high variance or negative lower bound",
        "affected_models": "all offline-only policies",
        "misleading_metric": "offline QINI / policy value",
        "diagnostic_method": "geo holdout lift, switchback stability, bootstrap lower bound",
        "fix_path": "shadow rollout, geo experiment, reduce budget until holdout confirms",
        "interview_line": "离线模型只是上线前证据，geo/holdout 才能验证真实增量。",
    },
    {
        "case_id": "llm_routing_cost_trap",
        "case_name": "强模型质量提升小于成本/延迟",
        "problem_type": "llm_routing",
        "mock_signal": "strong model win-rate positive but net value negative after cost and latency",
        "affected_models": "quality-only router, absolute score router",
        "misleading_metric": "strong model quality or win rate",
        "diagnostic_method": "quality uplift, model_cost, latency_penalty, budget constrained value",
        "fix_path": "uplift routing with random exploration and cost-quality objective",
        "interview_line": "LLM routing 不看绝对质量，而看增量质量是否值得成本。",
    },
    {
        "case_id": "continuous_dose_saturation",
        "case_name": "连续 treatment 饱和和边际成本上升",
        "problem_type": "dose_response",
        "mock_signal": "higher dose raises response initially but saturates while cost keeps rising",
        "affected_models": "binary uplift, high-dose policy",
        "misleading_metric": "high-intensity treatment uplift",
        "diagnostic_method": "dose-response curve, marginal ROI by dose",
        "fix_path": "dose-response learner and marginal ROI optimizer",
        "interview_line": "补贴金额、折扣率和出价要找边际最优，不是越大越好。",
    },
]


def model_failure_mode_rows() -> list[dict[str, str]]:
    return deepcopy(MODEL_FAILURE_MODES)


def scenario_pitfall_rows() -> list[dict[str, str]]:
    return deepcopy(SCENARIO_PITFALLS)


def diagnostic_case_rows() -> list[dict[str, str]]:
    return deepcopy(DIAGNOSTIC_CASES)


def failure_mode_counts() -> dict[str, int]:
    return {
        "model_failure_modes": len(MODEL_FAILURE_MODES),
        "scenario_pitfalls": len(SCENARIO_PITFALLS),
        "diagnostic_cases": len(DIAGNOSTIC_CASES),
    }


def diagnostic_case_summary_rows(path: str | Path = "reports/uplift_diagnostic_cases_latest.csv") -> list[dict[str, Any]]:
    case_path = Path(path)
    if not case_path.exists():
        return []
    try:
        frame = pd.read_csv(case_path)
    except Exception:
        return []
    return frame.to_dict(orient="records")


def failure_mode_agent_reply(prompt: str) -> str | None:
    text = prompt.lower()
    keywords = [
        "failure",
        "风险",
        "问题",
        "坑",
        "误导",
        "qini 高",
        "roi 低",
        "s-learner",
        "t-learner",
        "x-learner",
        "learner",
        "dr learner",
        "r learner",
        "causalforest",
        "overlap",
        "propensity",
        "sleeping dog",
        "sure thing",
        "sutva",
        "delayed feedback",
        "llm routing",
        "pctr",
        "pcvr",
        "置信区间",
        "geo holdout",
    ]
    if not any(key in text for key in keywords):
        return None

    if any(key in text for key in ["s-learner", "t-learner", "x-learner", "learner", "dr learner", "r learner", "causalforest", "模型"]):
        rows = MODEL_FAILURE_MODES[:]
        selected = []
        if "s/t/x" in text or "分别" in text:
            selected = [row for row in rows if row["family"] in {"S-Learner", "T-Learner", "X-Learner", "DR / R Learner"}]
        else:
            for row in rows:
                family = row["family"].lower()
                if family.replace(" ", "") in text.replace(" ", "") or any(token in text for token in family.lower().split("/")):
                    selected.append(row)
        if not selected:
            selected = rows[:6]
        lines = ["### Uplift 模型风险诊断", ""]
        for row in selected[:8]:
            lines.append(
                f"- **{row['family']}**：风险是 {row['risk']}；诊断看 `{row['diagnostic']}`；修复路径是 {row['fix']}"
            )
        lines.extend(
            [
                "",
                "UI 证据：`Models -> Model Risk Matrix`、`Evaluation -> Failure Modes / 指标误导案例`、`Evidence -> Diagnostic Cases`。",
                "代码证据：`deepuplift/core/failure_modes.py`、`scripts/generate_uplift_diagnostic_cases.py`、`scripts/smoke_uplift_failure_modes.py`。",
            ]
        )
        return "\n".join(lines)

    if any(key in text for key in ["qini", "roi", "指标", "误导", "policy value"]):
        return (
            "### 为什么 QINI 高但 ROI 可能低\n\n"
            "QINI/AUUC 只说明模型把增量响应人群排在前面，但它不扣券成本、广告成本、触达成本、LLM 强模型成本，也不处理套利、疲劳、延迟和预算约束。"
            "所以一个模型可以 `QINI > 0`，但如果 Top-K 人群的 treatment cost 高于 `uplift * value`，上线就是亏的。\n\n"
            "平台里用 `high_qini_low_roi` 和 `continuous_dose_saturation` 两个 mock case 演示这个问题："
            "前者说明排序好但成本高，后者说明高剂量 treatment 有饱和，边际收益低于边际成本。\n\n"
            "修复路径：把 `uplift_score` 转成 `net_value = uplift * value - cost - risk_penalty`，再用 Top-K policy value、ROI/iROAS、budget constrained value 和 bootstrap CI 做上线判断。"
        )

    if any(key in text for key in ["发券", "红包", "sleeping", "sure thing", "套利"]):
        return (
            "### 发券/红包场景最常见的坑\n\n"
            "- `sure thing`：本来就会买，发券浪费补贴。\n"
            "- `sleeping dog`：被触达后反而反感或延迟购买，出现负 uplift。\n"
            "- `subsidy abuse`：用户为了薅券改变行为，短期转化高但净利润低。\n"
            "- `margin mismatch`：券成本高于增量毛利，QINI 可能好看但 ROI 为负。\n\n"
            "平台里的诊断路径是：先看 uplift type mix 和 abuse risk，再看 Top-K incremental profit / policy value，最后用 cost-aware threshold 决定投放比例。"
        )

    if any(key in text for key in ["overlap", "propensity", "极端", "common support"]):
        return (
            "### Overlap / Propensity 不好怎么办\n\n"
            "先判断是不是设计问题：如果 e(X) 接近 0 或 1，某些人群几乎只出现在 treatment 或 control，模型在这些区域是在外推。"
            "DR/R learner 也会因为 propensity denominator 过小而产生不稳定 pseudo outcome。\n\n"
            "处理路径：1. propensity clipping；2. overlap trimming；3. 只在 common support 上报告指标；4. 增加随机探索或 holdout；"
            "5. 上线时加 support guardrail。对应 mock case 是 `extreme_propensity` 和 `imbalanced_treatment`。"
        )

    if any(key in text for key in ["pctr", "pcvr", "广告", "click", "full-funnel"]):
        return (
            "### 广告里 pCTR/pCVR 为什么不等于 uplift\n\n"
            "pCTR/pCVR 预测谁会点击或转化，但 uplift 要回答广告是否制造了额外点击/转化。"
            "高 pCVR 用户可能本来就会买，广告只拿到了归因；click uplift 高也可能没有 conversion/profit uplift。\n\n"
            "平台用 `full_funnel_click_trap` 展示 click uplift 与 conversion uplift 分离，用 `geo_holdout_conflict` 展示离线 uplift 需要 geo holdout 验证。"
        )

    if any(key in text for key in ["llm", "routing", "强模型", "随机探索"]):
        return (
            "### LLM routing 为什么需要 uplift 和随机探索\n\n"
            "强模型平均质量更高不代表每个请求都该升级。routing 的 treatment 是 `cheap -> strong/RAG/tool`，目标是"
            "`quality_uplift * value - model_cost - latency_penalty`。如果历史上只给困难请求调用强模型，就缺少反事实和 overlap。\n\n"
            "平台的 `llm_routing_cost_trap` mock case 展示：win-rate 为正但扣成本后 net value 为负。修复路径是随机探索桶、DR/R learner、calibration 和预算约束。"
        )

    return (
        "### Uplift 风险诊断总览\n\n"
        f"当前平台沉淀了 `{len(MODEL_FAILURE_MODES)}` 个模型 failure modes、`{len(SCENARIO_PITFALLS)}` 个业务场景 pitfall、"
        f"`{len(DIAGNOSTIC_CASES)}` 个 mock diagnostic cases。"
        "它们覆盖模型失效、数据设计、指标误导、成本/ROI、延迟反馈、全链路广告、geo holdout 和 LLM routing。"
    )
