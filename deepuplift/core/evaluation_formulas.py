from __future__ import annotations

from copy import deepcopy


EVALUATION_FORMULAS: list[dict[str, str]] = [
    {
        "metric": "ITE / CATE",
        "formula": "ITE_i = Y_i(1) - Y_i(0); CATE(x)=E[Y(1)-Y(0)|X=x]",
        "derivation_intuition": "uplift 的核心是同一个用户在 treatment 和 control 两个潜在结果之间的差，但真实世界只能观测一个结果。",
        "business_meaning": "回答发券、广告、push 或强模型调用到底带来多少增量，而不是用户本来就会转化多少。",
        "when_to_use": "所有 uplift 建模的定义起点；用于解释为什么普通 CVR 模型不够。",
        "misleading_when": "如果 treatment/control 不可比或 overlap 很差，CATE 会变成高风险外推。",
        "code_source": "deepuplift/core/trainer.py; deepuplift/core/evaluator.py",
        "artifact_field": "predictions.csv:y0_pred,y1_pred,uplift_score",
        "interview_line": "先讲 potential outcome，再讲我们用 y1-y0 的可观测估计做 ranking 和 policy。",
    },
    {
        "metric": "uplift score",
        "formula": "u_hat(x)=mu1_hat(x)-mu0_hat(x)",
        "derivation_intuition": "把 treatment outcome model 和 control outcome model 的差作为个体增量排序分数。",
        "business_meaning": "用来排序 Top-K 人群：谁更可能因为红包、补贴、广告或强模型路由而产生额外价值。",
        "when_to_use": "所有二元 treatment 打分、导出名单、Policy Top-K 阈值选择。",
        "misleading_when": "模型只学到自然转化率或 treatment selection bias 时，分数可能不是因果增量。",
        "code_source": "deepuplift/core/trainer.py; deepuplift/core/predictor.py",
        "artifact_field": "predictions.csv:uplift_score",
        "interview_line": "普通模型预测 P(Y=1|X)，uplift 模型预测 treatment 相对 control 的差。",
    },
    {
        "metric": "QINI",
        "formula": "Q(k)=Y_t(k)-Y_c(k)*N_t(k)/N_c(k); QINI = area(Q(k)-Q_random(k))",
        "derivation_intuition": "按 uplift score 从高到低累积，比较目标人群中的 treatment 响应和按比例缩放的 control 响应。",
        "business_meaning": "看模型是否比随机投放更早找到 persuadable users。",
        "when_to_use": "二元 treatment 的主排序指标；Model Compare 排名和 regression gate。",
        "misleading_when": "成本、毛利、预算、校准和 overlap 没纳入时，QINI 高不等于上线收益高。",
        "code_source": "deepuplift/core/evaluator.py:qini/uplift curves",
        "artifact_field": "metrics.json:qini_score; curves.json:qini",
        "interview_line": "QINI 是 ranking 指标，不是 ROI 指标；上线前必须接 policy value。",
    },
    {
        "metric": "AUUC",
        "formula": "AUUC = integral_0^1 U(p) dp, where U(p)=mean(Y|T=1,top p)-mean(Y|T=0,top p)",
        "derivation_intuition": "把不同投放比例下的累计 observed uplift 积起来，衡量全局排序质量。",
        "business_meaning": "看模型在多个 audience size 下是否稳定产生增量，而不是只在一个桶里好看。",
        "when_to_use": "模型广义排序质量对比；和 QINI 一起看。",
        "misleading_when": "早期 Top-K 很差但中后段补回来时，AUUC 仍可能不错；业务投放通常更关心 Top-K。",
        "code_source": "deepuplift/core/evaluator.py",
        "artifact_field": "metrics.json:auuc_score; curves.json:auuc",
        "interview_line": "AUUC 是全曲线面积，QINI 更强调相对随机基线的累计增益。",
    },
    {
        "metric": "uplift@K / Top-K observed uplift",
        "formula": "uplift@K = E[Y|T=1, score in top K] - E[Y|T=0, score in top K]",
        "derivation_intuition": "只看真正会被投放的 Top-K 人群里，treatment 和 control 的观测差异。",
        "business_meaning": "业务常问 Top 5%/10%/20% 是否值得投放，这比全局 AUUC 更贴近投放名单。",
        "when_to_use": "选择灰度投放桶、导出名单、解释业务收益。",
        "misleading_when": "Top-K 样本量小、treatment/control 覆盖不足或没有 bootstrap CI 时会高方差。",
        "code_source": "deepuplift/core/evaluator.py:top_k",
        "artifact_field": "metrics.json:top_k",
        "interview_line": "我会把 Top-K 当成 launch bucket 指标，而不是只汇报一个全局分数。",
    },
    {
        "metric": "policy value",
        "formula": "V(pi)=E[Y(pi(X))]; binary targeting approx = E[pi(X)*(Y(1)-Y(0))]",
        "derivation_intuition": "模型排序最终要变成一个 policy：哪些用户 treatment，哪些用户 holdout。",
        "business_meaning": "衡量策略本身能带来多少增量，而不是模型分数是否好看。",
        "when_to_use": "最终阈值、预算、上线候选选择。",
        "misleading_when": "没有显式成本/价值、没有 holdout、或者 policy 超出 overlap support 时会过乐观。",
        "code_source": "deepuplift/core/policy.py; app.py:Policy",
        "artifact_field": "metrics.json:policy_best; curves.json:policy_value",
        "interview_line": "上线不是选模型分数最高，而是选在预算和约束下 policy value 最高且可信的策略。",
    },
    {
        "metric": "incremental profit / cost-aware gain",
        "formula": "profit_i = uplift_i * value_i - cost_i - risk_penalty_i",
        "derivation_intuition": "把增量概率乘以业务价值，再扣掉券成本、广告成本、触达成本、强模型成本或疲劳惩罚。",
        "business_meaning": "发券看增量毛利，广告看增量收入/成本，LLM routing 看质量提升是否值得更贵模型。",
        "when_to_use": "所有有边际成本的工业场景。",
        "misleading_when": "毛利、成本、疲劳、套利、长期价值假设不准确时，阈值会错。",
        "code_source": "deepuplift/core/policy.py; deepuplift/core/industrial_scenario_lab.py",
        "artifact_field": "metrics.json:policy_best; scenario policy benchmark reports",
        "interview_line": "QINI 解释排序，profit 解释为什么值得投。",
    },
    {
        "metric": "iROAS",
        "formula": "iROAS = incremental_revenue / ad_cost = (uplift * conversion_value) / media_cost",
        "derivation_intuition": "只把广告带来的增量收入放到分子，而不是归因收入或自然转化收入。",
        "business_meaning": "广告预算是否真正创造增量价值。",
        "when_to_use": "广告投放、audience targeting、ghost ads、geo holdout 汇报。",
        "misleading_when": "如果 incremental revenue 来自 attribution 而非 holdout/causal estimate，就会高估。",
        "code_source": "deepuplift/core/industrial_scenario_lab.py; deepuplift/core/policy.py",
        "artifact_field": "industrial scenario compare: scenario_policy_value / iROAS proxies",
        "interview_line": "pCTR/pCVR 是相关性，iROAS 要用 incrementality 做分子。",
    },
    {
        "metric": "DR pseudo-outcome",
        "formula": "phi = mu1_hat-mu0_hat + T*(Y-mu1_hat)/e_hat - (1-T)*(Y-mu0_hat)/(1-e_hat)",
        "derivation_intuition": "把 outcome model 和 propensity weighting 组合起来；只要一边建得足够好，就更稳健。",
        "business_meaning": "观测投放/非随机触达中降低 selection bias 对 uplift 估计的影响。",
        "when_to_use": "treatment/control 不均衡、观测日志、强选择偏差。",
        "misleading_when": "propensity 极端或 outcome/propensity 都很差时，DR 仍会失真。",
        "code_source": "deepuplift/core/sklearn_models.py:DR/R learners; deepuplift/core/external_models.py:EconMLDRLearner",
        "artifact_field": "model card: DRLearner*; metrics.json overlap_trim",
        "interview_line": "DR 是工业观测数据里非常值得讲的稳健 baseline。",
    },
    {
        "metric": "R-learner objective",
        "formula": "min_tau sum_i ((Y_i-m_hat(X_i)) - (T_i-e_hat(X_i))*tau(X_i))^2",
        "derivation_intuition": "先残差化 outcome 和 treatment，再学习 effect function，减少 nuisance bias。",
        "business_meaning": "适合解释为什么 uplift 不是直接拟合 Y，而是学习 treatment residual 对 outcome residual 的影响。",
        "when_to_use": "观测数据、偏差较强、需要正交化的 robust baseline。",
        "misleading_when": "nuisance model 质量差或 overlap 差时，残差化也无法恢复 unsupported 区域。",
        "code_source": "deepuplift/core/sklearn_models.py:RLearner*; deepuplift/core/external_models.py:EconMLDML",
        "artifact_field": "model catalog: RLearner* / EconML*",
        "interview_line": "R-learner 的关键是 residual-on-residual，把主效应从 CATE 学习里剥离。",
    },
    {
        "metric": "IPW / propensity weighted uplift",
        "formula": "ATE/IPW approx = E[T*Y/e(X) - (1-T)*Y/(1-e(X))]",
        "derivation_intuition": "用 treatment assignment probability 的倒数把样本重加权，模拟更均衡的实验分布。",
        "business_meaning": "当 treatment/control 比例不均衡时，避免多数 treatment arm 主导估计。",
        "when_to_use": "随机化概率已知或 propensity 可估计且 overlap 尚可。",
        "misleading_when": "e(X) 接近 0/1 会产生极高方差，需要 trimming / clipping。",
        "code_source": "deepuplift/core/sklearn_models.py; deepuplift/core/diagnostics.py",
        "artifact_field": "metrics.json:overlap_trim; diagnostics overlap",
        "interview_line": "IPW 是因果估计基本工具，但工业里必须和 overlap trimming 一起讲。",
    },
    {
        "metric": "calibration MAE",
        "formula": "CalibMAE = mean_b | mean(u_hat in bin b) - observed_uplift_b |",
        "derivation_intuition": "把分数分桶，比较预测 uplift 和实际 treatment-control uplift 是否对齐。",
        "business_meaning": "判断模型分数是否能用于阈值、预算 pacing 和分群解释。",
        "when_to_use": "上线前、策略阈值选择、compare 胜者可信度判断。",
        "misleading_when": "小桶样本少或 control/treatment 覆盖差时，observed uplift 噪声很大。",
        "code_source": "deepuplift/core/evaluator.py; deepuplift/core/readiness.py",
        "artifact_field": "metrics.json:calibration_mae; curves.json:calibration",
        "interview_line": "排序好不代表分数可解释，calibration 解决阈值可信度。",
    },
    {
        "metric": "bootstrap CI",
        "formula": "CI_alpha(metric)=percentile({metric(sample_b)}_b, [alpha/2,1-alpha/2])",
        "derivation_intuition": "重复抽样评估指标，估计模型排名和 Top-K 结果的波动范围。",
        "business_meaning": "避免把随机噪声当成模型胜出，尤其在 Top-K 和 policy value 上。",
        "when_to_use": "模型 compare、上线评审、面试证明结果稳定。",
        "misleading_when": "抽样不能修复设计偏差；如果原始数据有 selection bias，CI 只是偏差周围的不确定性。",
        "code_source": "deepuplift/core/evaluator.py; deepuplift/core/readiness.py",
        "artifact_field": "metrics.json:bootstrap",
        "interview_line": "我按下界排序或降级不稳定模型，不只看均值。",
    },
    {
        "metric": "overlap / trimming sensitivity",
        "formula": "keep rows with e(X) in [a,1-a]; compare metric_full vs metric_trimmed",
        "derivation_intuition": "只在 treatment/control 都有支持的 common support 区域评价和部署。",
        "business_meaning": "判断模型是否在不可比较人群上外推。",
        "when_to_use": "观测数据、有偏投放、treatment rate 极不均衡。",
        "misleading_when": "trimming 后人群变小，业务覆盖降低；需要解释可部署范围。",
        "code_source": "deepuplift/core/diagnostics.py; deepuplift/core/evaluator.py",
        "artifact_field": "metrics.json:overlap_trim",
        "interview_line": "overlap 是上线可信度闸门，不是可选图表。",
    },
    {
        "metric": "oracle top-k recall",
        "formula": "recall@K = |TopK(u_hat) intersection TopK(true_uplift)| / K",
        "derivation_intuition": "在仿真数据有 true uplift 时，直接检查模型是否抓到真实高增量人群。",
        "business_meaning": "验证 synthetic scenario 和 regression gate 是否真的能区分模型能力。",
        "when_to_use": "模拟数据、benchmark、面试演示和模型回归测试。",
        "misleading_when": "真实业务没有 true_uplift，不能当成线上指标。",
        "code_source": "deepuplift/core/evaluator.py; scripts/smoke_industrial_scenario_compare.py",
        "artifact_field": "metrics.json:oracle_top_k; reports/industrial_scenario_compare_latest.csv",
        "interview_line": "synthetic 数据用于验证框架，线上仍要 holdout/A-B。",
    },
    {
        "metric": "delayed feedback uplift",
        "formula": "uplift_Dw@K = E[Y_Dw|T=1,TopK] - E[Y_Dw|T=0,TopK]",
        "derivation_intuition": "把 outcome 明确绑定到 D1/D7/D14/D30 等成熟窗口，避免短窗口漏掉慢响应。",
        "business_meaning": "增长召回、广告转化、留存常常不是当天发生。",
        "when_to_use": "CRM、push、广告归因、复购、留存。",
        "misleading_when": "标签窗口泄漏、右删失和训练/评估窗口不一致会导致虚假提升。",
        "code_source": "deepuplift/core/evaluator.py:delayed_feedback_uplift_metrics",
        "artifact_field": "metrics.json:delayed_feedback",
        "interview_line": "我会同时看短期和成熟窗口，避免因为 D1 弱就误杀 D30 有价值人群。",
    },
    {
        "metric": "full-funnel ECUP metric",
        "formula": "uplift_stage@K = E[Y_stage|T=1,TopK] - E[Y_stage|T=0,TopK], stage in impression/click/conversion",
        "derivation_intuition": "把曝光、点击、转化拆开看，定位 uplift 来自链路哪一层。",
        "business_meaning": "广告和推荐不能只看最终转化，可能点击增益高但转化或利润为负。",
        "when_to_use": "广告投放、Shopee iROAS、推荐干预、ECUP 全链路场景。",
        "misleading_when": "stage selection bias、延迟转化、归因口径混乱会扭曲结果。",
        "code_source": "deepuplift/core/evaluator.py:full_funnel_uplift_metrics",
        "artifact_field": "metrics.json:full_funnel",
        "interview_line": "ECUP 让广告/推荐从单点 CVR 变成全链路增量诊断。",
    },
    {
        "metric": "LLM routing cost-quality value",
        "formula": "value_i = quality_uplift_i * business_value - model_cost_i - latency_penalty_i",
        "derivation_intuition": "把 strong model / RAG / tool escalation 看成 treatment，只在增量质量超过增量成本时升级。",
        "business_meaning": "控制 LLM 成本，同时保留高价值 query 的质量提升。",
        "when_to_use": "LLM routing、RAG 调用、人审升级、复杂 prompt 强模型路由。",
        "misleading_when": "judge label 有偏、没有随机探索、成本/延迟随流量漂移。",
        "code_source": "deepuplift/core/policy.py; scripts/smoke_llm_routing_policy.py",
        "artifact_field": "reports/llm_routing_policy_smoke_latest.json",
        "interview_line": "LLM routing 本质是 uplift：强模型相对弱模型的增量质量是否值得成本。",
    },
    {
        "metric": "off-policy evaluation / DR OPE",
        "formula": "V_DR(pi)=mean[mu_hat(pi(x),x)+1{a=pi(x)}/p(a|x)*(r-mu_hat(a,x))]",
        "derivation_intuition": "先用 direct model 估计目标 policy 的 reward，再用 logged action 与 propensity 对命中的样本做残差修正。",
        "business_meaning": "上线前评估新投放、补贴、广告或 LLM routing policy 的预期价值，避免直接全量试错。",
        "when_to_use": "有 logged action、logged propensity 和 reward 的随机探索/历史策略数据时。",
        "misleading_when": "target policy 与 logged policy overlap 很低、propensity 缺失/错误、reward model 也差时会失真。",
        "code_source": "deepuplift/core/policy.py:llm_routing_ope_summary",
        "artifact_field": "reports/llm_routing_ope_smoke_latest.json",
        "interview_line": "OPE 是 offline uplift 到 online bandit 的桥：没有 propensity 和 coverage，就先补探索而不是相信新 policy。",
    },
]


POLICY_DERIVATIONS: list[dict[str, str]] = [
    {
        "topic": "从 uplift 到投放阈值",
        "steps": "1. 估计 u_hat(x)=mu1-mu0；2. 计算 net_i=u_hat_i*value_i-cost_i；3. 按 net_i 或 uplift score 排序；4. 遍历 Top-K 找最大 cumulative net；5. 加预算/频控/风险约束。",
        "formula": "K* = argmax_K sum_{i in TopK} (u_hat_i * value_i - cost_i - risk_i), subject to budget(K)<=B",
        "business_example": "发券：value_i 是毛利或 GMV margin，cost_i 是券成本；广告：value_i 是转化价值，cost_i 是 media cost；LLM routing：cost_i 是强模型/工具/时延成本。",
        "ui_location": "Policy -> Scenario ROI Lab; Evaluation -> Policy Value Formula Cards",
        "code_source": "deepuplift/core/policy.py",
    },
    {
        "topic": "DR learner 为什么稳健",
        "steps": "1. outcome model 估计 mu1/mu0；2. propensity model 估计 e(x)；3. 用 residual correction 修正 treatment/control 观测误差；4. 得到 DR pseudo outcome；5. 再拟合 CATE。",
        "formula": "phi = mu1-mu0 + T(Y-mu1)/e - (1-T)(Y-mu0)/(1-e)",
        "business_example": "滴滴补贴或广告投放日志不是完全随机时，DR 能减少 selection bias，但仍依赖 overlap。",
        "ui_location": "Models -> P0 robust causal layer; Evaluation -> DR/R Formula Cards",
        "code_source": "deepuplift/core/sklearn_models.py; deepuplift/core/external_models.py",
    },
    {
        "topic": "为什么 QINI 高不一定上线",
        "steps": "1. QINI 只看增量排序；2. 不看单人成本/毛利/预算；3. 不保证分数校准；4. 不自动处理 overlap 和 CI；5. 因此必须接 policy value 和 trust gates。",
        "formula": "deployment_score = policy_value - guardrail_penalty, not just QINI",
        "business_example": "红包模型 QINI 高，但 Top 10% 多是高券成本或套利用户，净收益可能为负。",
        "ui_location": "Evaluation -> Metric Risk Legend; Policy -> Metric Disagreement Lab",
        "code_source": "deepuplift/core/evaluator.py; deepuplift/core/readiness.py",
    },
    {
        "topic": "从 logged policy 到新策略 OPE",
        "steps": "1. 记录 action、reward、propensity；2. 定义目标 policy pi(x)；3. 计算 action match 和 IPS 权重；4. 用 reward model 得到 direct value；5. 用 DR 修正 residual；6. 检查 coverage 和 effective sample size。",
        "formula": "V_DR(pi)=mean[mu_hat(pi(x),x)+1{a=pi(x)}/p(a|x)*(r-mu_hat(a,x))]",
        "business_example": "LLM routing：历史请求在 cheap/strong/RAG/tool/human review 间有探索，才能离线评估新的 cost-quality router。",
        "ui_location": "Policy -> LLM 多动作 Routing OPE Mini-Lab; Evaluation -> Policy Value / ROI 推导卡",
        "code_source": "deepuplift/core/policy.py:llm_routing_ope_summary",
    },
]


OPTIMIZATION_PATH_ROWS: list[dict[str, str]] = [
    {
        "stage": "V0 业务规则 / 转化率模型",
        "why": "先复现业务原始策略，建立 baseline 和反例。",
        "what_changes": "按自然 CVR、GMV、活跃度或人工规则投放。",
        "metric_effect": "CVR 可能高，但 uplift/profit 经常浪费在 sure things。",
        "risk": "把相关性误当因果增量。",
        "evidence": "Scenario Workbench -> 优化路径; docs/UPLIFT_MODEL_OPTIMIZATION_PATH.md",
    },
    {
        "stage": "V1 S/T learner baseline",
        "why": "最快建立 uplift 可训练闭环。",
        "what_changes": "从预测 Y 升级为估计 y1-y0。",
        "metric_effect": "通常显著改善 QINI/AUUC 和 Top-K observed uplift。",
        "risk": "S-learner 可能低估 treatment heterogeneity，T-learner 在小样本 arm 上不稳。",
        "evidence": "Models -> P0 baseline; Compare -> model ranking",
    },
    {
        "stage": "V2 X/DR/R learner",
        "why": "处理样本不均衡和观测选择偏差。",
        "what_changes": "加入反事实 imputation、DR correction 或 residual objective。",
        "metric_effect": "改善 overlap 敏感场景下的稳定性和 CI 下界。",
        "risk": "propensity 极端或 nuisance model 差时仍会失败。",
        "evidence": "Evaluation -> DR/R Formula Cards; Diagnostics -> overlap",
    },
    {
        "stage": "V3 CausalForest / uplift tree",
        "why": "增强异质性解释和业务规则可读性。",
        "what_changes": "用 forest/tree 发现稳定子群或规则。",
        "metric_effect": "提升 segment stability 和异质性解释，不一定总提升 AUUC。",
        "risk": "小叶子样本不足会过拟合。",
        "evidence": "Models -> Causal Forest / Tree layer; Evaluation -> Segment Analysis",
    },
    {
        "stage": "V4 强表格 boosting baseline",
        "why": "工业大表常被 LightGBM/CatBoost/XGBoost 打穿。",
        "what_changes": "把 GBM 接入 S/T/X/R/DR learner 作为强基线。",
        "metric_effect": "经常提升 Top-K 和 policy value，是上线候选主力。",
        "risk": "boosting 仍不是因果识别器；需要实验/overlap 保障。",
        "evidence": "Models -> Open Source Framework Audit; Model Catalog -> LightGBM",
    },
    {
        "stage": "V5 深度 uplift / interaction / contrastive",
        "why": "处理复杂 user-treatment interaction、文本/素材/券特征、表示不平衡。",
        "what_changes": "接入 TARNet/CFRNet/DragonNet/EFIN/DESCN/ContrastiveUpliftNet。",
        "metric_effect": "在复杂 treatment 特征或非线性交互场景可能提升，但必须对比 P0。",
        "risk": "调参敏感、样本要求高、可解释性和校准可能变差。",
        "evidence": "Models -> Deep Uplift Loss Catalog; docs/UPLIFT_DEEP_LOSS_AND_ARCHITECTURE.md",
    },
    {
        "stage": "V6 Policy / ROI / budget optimizer",
        "why": "把模型指标转成业务决策。",
        "what_changes": "加入 cost-aware gain、iROAS、预算约束、Top-K 阈值。",
        "metric_effect": "即使 QINI 不最高，也可能选择净收益更稳的模型。",
        "risk": "业务成本/毛利假设错误会改变结论。",
        "evidence": "Policy -> Scenario ROI Lab; Evaluation -> policy value",
    },
    {
        "stage": "V7 Trust gates",
        "why": "防止偶然胜出和 unsupported extrapolation。",
        "what_changes": "加入 calibration、bootstrap CI、overlap trimming、sensitivity、regression gate。",
        "metric_effect": "模型排序从均值胜出升级为下界/稳定性胜出。",
        "risk": "过严会牺牲覆盖，需要和业务灰度策略平衡。",
        "evidence": "Evaluation -> Trust Gates; Evidence -> regression artifacts",
    },
    {
        "stage": "V8 Online holdout / geo / shadow rollout",
        "why": "离线 uplift 仍需线上实验确认。",
        "what_changes": "接入 holdout、ghost ads、geo experiment、灰度和监控。",
        "metric_effect": "把离线 QINI/Policy Value 对齐线上 incremental revenue / iROAS。",
        "risk": "实验干扰、spillover、预算 pacing 和长期效应需要独立设计。",
        "evidence": "Knowledge -> Industrial Knowledge Map; docs/UPLIFT_ONLINE_EXPERIMENT_AND_INCREMENTALITY.md",
    },
]


def evaluation_formula_rows() -> list[dict[str, str]]:
    return deepcopy(EVALUATION_FORMULAS)


def policy_derivation_rows() -> list[dict[str, str]]:
    return deepcopy(POLICY_DERIVATIONS)


def optimization_path_rows() -> list[dict[str, str]]:
    return deepcopy(OPTIMIZATION_PATH_ROWS)


def evaluation_formula_counts() -> dict[str, int]:
    return {
        "metrics": len(EVALUATION_FORMULAS),
        "policy_derivations": len(POLICY_DERIVATIONS),
        "optimization_stages": len(OPTIMIZATION_PATH_ROWS),
    }


def evaluation_formula_agent_reply(prompt: str) -> str | None:
    text = prompt.lower()
    if not any(
        key in text
        for key in [
            "怎么评估",
            "评估uplift",
            "评估 uplift",
            "公式",
            "推导",
            "qini",
            "auuc",
            "uplift@",
            "policy value",
            "iroas",
            "dr learner",
            "r learner",
            "校准",
            "bootstrap",
            "overlap",
            "从0-1",
            "0-1",
            "优化路径",
        ]
    ):
        return None

    if any(key in text for key in ["dr learner", "r learner", "pseudo", "正交", "残差"]):
        dr = next(row for row in EVALUATION_FORMULAS if row["metric"] == "DR pseudo-outcome")
        r = next(row for row in EVALUATION_FORMULAS if row["metric"] == "R-learner objective")
        return (
            "### DR / R learner 目标函数怎么讲\n\n"
            f"**DR pseudo-outcome**：`{dr['formula']}`。\n"
            f"直觉：{dr['derivation_intuition']} 业务含义：{dr['business_meaning']}\n\n"
            f"**R-learner objective**：`{r['formula']}`。\n"
            f"直觉：{r['derivation_intuition']} 业务含义：{r['business_meaning']}\n\n"
            "面试落点：观测投放/补贴日志有 selection bias，所以我不会只用 T-learner；"
            "会用 DR/R/DML 作为 P0 robust baseline，并在 Diagnostics/Evaluation 里检查 overlap、trimming 和 bootstrap CI。\n\n"
            "UI/代码证据：`Evaluation -> Formula Cards`，`Models -> Robust Causal layer`，"
            "`deepuplift/core/sklearn_models.py`，`deepuplift/core/external_models.py`。"
        )

    if any(key in text for key in ["policy value", "roi", "iroas", "发券", "广告", "投放阈值", "阈值"]):
        policy = POLICY_DERIVATIONS[0]
        iros = next(row for row in EVALUATION_FORMULAS if row["metric"] == "iROAS")
        profit = next(row for row in EVALUATION_FORMULAS if row["metric"] == "incremental profit / cost-aware gain")
        return (
            "### 从 uplift 到 ROI / 投放阈值\n\n"
            f"核心公式：`{policy['formula']}`。\n"
            f"步骤：{policy['steps']}\n\n"
            f"发券净收益：`{profit['formula']}`。{profit['business_meaning']}\n\n"
            f"广告 iROAS：`{iros['formula']}`。{iros['business_meaning']}\n\n"
            "面试落点：QINI/AUUC 只回答排序，Policy Value 回答是否值得投。"
            "我会把 margin、coupon cost、media cost、疲劳/套利风险和预算约束显式版本化，"
            "所以即使某模型 QINI 第一，也可能因为净收益下界差而不推荐上线。\n\n"
            "UI/代码证据：`Policy -> Scenario ROI Lab`，`Evaluation -> Policy Value Formula Cards`，"
            "`deepuplift/core/policy.py`。"
        )

    if any(key in text for key in ["qini", "auuc", "uplift@k", "uplift@", "区别"]):
        chosen = [row for row in EVALUATION_FORMULAS if row["metric"] in {"QINI", "AUUC", "uplift@K / Top-K observed uplift"}]
        bullets = "\n".join(
            f"- **{row['metric']}**：`{row['formula']}`。{row['business_meaning']} 风险：{row['misleading_when']}"
            for row in chosen
        )
        return (
            "### QINI / AUUC / uplift@K 的区别\n\n"
            f"{bullets}\n\n"
            "我的使用顺序：先看 QINI/AUUC 判断排序是否超过随机；再看 Top-K 是否对应真实投放桶；"
            "最后用 policy value、calibration、bootstrap CI 和 overlap 决定是否上线。\n\n"
            "UI 证据：`Evaluation -> Metric Formula Cards` 和 `Evaluation -> Metric Risk Legend`。"
        )

    if any(key in text for key in ["从0-1", "0-1", "优化路径", "持续优化", "怎么优化"]):
        path = "\n".join(
            f"- **{row['stage']}**：{row['why']} 作用：{row['metric_effect']} 风险：{row['risk']}"
            for row in OPTIMIZATION_PATH_ROWS
        )
        return (
            "### Uplift 项目从 0-1 到相对最优的优化路径\n\n"
            f"{path}\n\n"
            "面试讲法：我不是一次性堆模型，而是从业务规则反例开始，逐步加入 CATE、稳健因果、异质性解释、"
            "深度交互、ROI optimizer、可信度 gate 和线上 holdout，把每一步的收益和风险都用指标验证。"
        )

    return (
        "### uplift 模型评估体系\n\n"
        "我把评估分成四层：\n"
        "1. **排序指标**：QINI、AUUC、uplift@K，判断是否找到增量人群。\n"
        "2. **业务指标**：policy value、incremental profit、ROI/iROAS，判断是否值得投。\n"
        "3. **可信度指标**：calibration、bootstrap CI、overlap/trimming、sensitivity，判断是否稳。\n"
        "4. **特殊场景指标**：delayed feedback、full-funnel ECUP、LLM routing cost-quality、spillover guardrail。\n\n"
        "公式和推导已经沉淀在 `Evaluation -> Metric Formula Cards`、"
        "`docs/UPLIFT_EVALUATION_METRICS_FORMULAS.md` 和 `docs/UPLIFT_POLICY_VALUE_AND_ROI_DERIVATION.md`。"
    )
