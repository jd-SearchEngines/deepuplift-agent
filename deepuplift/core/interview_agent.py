from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .agent_v2 import agent_v2_reply


INTERVIEW_AGENT_QUICK_PROMPTS = [
    "为什么不用普通转化率模型？",
    "观测数据有偏怎么办？",
    "模型 registry 难点怎么讲？",
    "offline 指标怎么对齐上线收益？",
    "深度 uplift loss 面试怎么讲？",
    "LLM routing 为什么是 uplift 问题？",
    "什么时候从 uplift 升级到 bandit？",
    "OPE / IPS / DR 怎么评估新策略？",
    "DRNet / VCNet 连续 treatment 怎么讲？",
    "为什么 Evidence 里有很多 warning，哪些不能上线？",
]


INTERVIEW_AGENT_TOPICS = [
    {
        "id": "I01",
        "topic": "uplift vs response prediction",
        "question": "为什么不用普通转化率模型？",
        "evidence": "Story -> Interview Difficulties, Compare, Policy",
    },
    {
        "id": "I02",
        "topic": "treatment design and causal assumptions",
        "question": "随机实验、观测数据和 treatment/control 设计怎么讲？",
        "evidence": "Diagnostics -> Design, Overlap, Selection Bias / SSB Proxy",
    },
    {
        "id": "I03",
        "topic": "selection bias, SSB, confounding",
        "question": "观测数据有偏怎么办？",
        "evidence": "Diagnostics -> Overlap, Sensitivity, Bootstrap CI",
    },
    {
        "id": "I04",
        "topic": "registry, adapter, model card",
        "question": "101 个模型怎么统一接入？",
        "evidence": "Models -> Catalog, Model Card, Presets, Guarded Backend Runtime Guide",
    },
    {
        "id": "I05",
        "topic": "prediction contract",
        "question": "不同模型输出格式不一致怎么统一评估？",
        "evidence": "Predict -> y0_pred / y1_pred / uplift_score, Evaluation -> metrics",
    },
    {
        "id": "I06",
        "topic": "evaluation system",
        "question": "QINI、AUUC、Top-K、calibration、bootstrap CI 各自看什么？",
        "evidence": "Evaluation -> Decision Readiness, Compare -> ranking, Policy -> value curve",
    },
    {
        "id": "I07",
        "topic": "offline to online value",
        "question": "offline 指标怎么对齐线上业务收益？",
        "evidence": "Policy -> cost-value threshold, Offline-To-Online Validation Playbook",
    },
    {
        "id": "I08",
        "topic": "agent workflow orchestration",
        "question": "Agent 怎么从聊天升级成 causal workflow orchestrator？",
        "evidence": "Agent -> Run Causal Workflow, reports/agent_compare_bundle_*.json",
    },
    {
        "id": "I09",
        "topic": "evidence and regression gate",
        "question": "怎么保证简历数字和 demo 结果可复现？",
        "evidence": "Story -> Evidence Index, full_regression_check, freshness screenshot",
    },
    {
        "id": "I10",
        "topic": "guarded optional backends",
        "question": "为什么 XGBoost、CatBoost、CausalML 有些 guarded/optional？",
        "evidence": "Models -> Guarded Backend Runtime Guide, backend strategy doc",
    },
    {
        "id": "I11",
        "topic": "streamlit to production",
        "question": "Streamlit MVP 怎么演进到生产系统？",
        "evidence": "Story -> Productionization Roadmap, Architecture Diagram & Data Flow",
    },
    {
        "id": "I12",
        "topic": "resume pitch",
        "question": "这个项目面试开场怎么讲？",
        "evidence": "Story -> Live Resume Snapshot, Chinese Project One-Pager",
    },
    {
        "id": "I13",
        "topic": "deep uplift loss and architecture",
        "question": "TARNet/CFRNet/DragonNet/EFIN/ECUP 的 loss 和网络结构怎么讲？",
        "evidence": "Models -> 深度 Uplift Loss Catalog / 深度网络结构 Catalog / 深度算法面试追问 Drill",
    },
    {
        "id": "I14",
        "topic": "contrastive uplift and representation learning",
        "question": "对比学习怎么和 uplift 结合，正负样本怎么定义？",
        "evidence": "Models -> Contrastive uplift encoder, docs/UPLIFT_CONTRASTIVE_AND_LLM_UPLIFT.md",
    },
    {
        "id": "I15",
        "topic": "LLM routing as uplift",
        "question": "LLM routing / RAG / tool / 人工审核为什么是 uplift 问题？",
        "evidence": "Policy -> LLM Routing ROI Simulator, Synthetic LLM Multi-Action Routing 9k, llm_multi_action_routing_loss",
    },
    {
        "id": "I16",
        "topic": "metric derivation and misleading metrics",
        "question": "QINI/AUUC/policy value/ROI/calibration 的公式、误导点和上线判断怎么讲？",
        "evidence": "Evaluation -> formula cards, Failure Modes, Metric Risk Legend",
    },
    {
        "id": "I17",
        "topic": "uplift to contextual bandit",
        "question": "什么时候用离线 uplift 固定阈值，什么时候升级到 contextual bandit？",
        "evidence": "Policy -> 从离线 uplift 到在线 bandit 的上线门禁",
    },
    {
        "id": "I18",
        "topic": "off-policy evaluation",
        "question": "OPE / IPS / SNIPS / DR 怎么评估一个新策略？",
        "evidence": "Policy -> LLM 多动作 Routing OPE Mini-Lab; Evaluation -> OPE formula cards",
    },
    {
        "id": "I19",
        "topic": "continuous treatment and dose-response",
        "question": "DRNet / VCNet 连续 treatment 怎么讲？",
        "evidence": "Dose-Response -> Cost-Aware Dose Policy Optimizer; Evidence -> Continuous Treatment Policy Smoke",
    },
    {
        "id": "I20",
        "topic": "regression warning audit and launch guardrails",
        "question": "为什么 Evidence 里有很多 warning，哪些不能上线，下一步怎么补实验？",
        "evidence": "Evidence -> Regression Warning / Launch Guardrail Audit; reports/regression_warning_audit_latest.json",
    },
]


def interview_agent_topics() -> list[dict[str, str]]:
    return [dict(topic) for topic in INTERVIEW_AGENT_TOPICS]


def _has_any(text: str, keywords: list[str]) -> bool:
    return any(keyword.lower() in text for keyword in keywords)


def _ctx(context: Mapping[str, Any], key: str, default: Any = "NA") -> Any:
    value = context.get(key, default)
    if value is None:
        return default
    return value


def _context_sentence(context: Mapping[str, Any]) -> str:
    return (
        f"当前页面上下文：任务 `{_ctx(context, 'task')}`，模型 `{_ctx(context, 'model_name')}`，"
        f"treatment `{_ctx(context, 'treatment_col')}`，outcome `{_ctx(context, 'outcome_col')}`，"
        f"模型目录 `{_ctx(context, 'registered_models')}` 个，ready `{_ctx(context, 'ready_models')}` 个。"
    )


def _format_answer(
    *,
    title: str,
    short: str,
    why_hard: str,
    solution: str,
    tradeoff: str,
    proof: str,
    interview_line: str,
    context: Mapping[str, Any],
) -> str:
    return (
        f"### {title}\n\n"
        f"**一句话回答**：{short}\n\n"
        f"**为什么难**：{why_hard}\n\n"
        f"**我的解决方案**：{solution}\n\n"
        f"**技术取舍**：{tradeoff}\n\n"
        f"**验证证据**：{proof}\n\n"
        f"**面试讲法**：{interview_line}\n\n"
        f"{_context_sentence(context)}"
    )


def _overview(context: Mapping[str, Any]) -> str:
    rows = "\n".join(
        f"- `{item['id']}` **{item['topic']}**：{item['question']} 证据：`{item['evidence']}`"
        for item in INTERVIEW_AGENT_TOPICS
    )
    return (
        "### 面试可讲技术难点总览\n\n"
        "我建议把项目讲成一条 causal decision workflow，而不是一个模型列表。\n\n"
        f"{rows}\n\n"
        "现场演示顺序：`Story -> Diagnostics -> Models -> Agent -> Compare -> Policy -> Evidence`。\n\n"
        f"{_context_sentence(context)}"
    )


def interview_agent_reply(prompt: str, context: Mapping[str, Any] | None = None) -> str | None:
    text = prompt.strip().lower()
    context = context or {}
    if not text:
        return None

    if _has_any(text, ["面试难点", "技术难点", "难点总结", "都有什么难点", "可以讲的难点", "interview difficulty"]):
        return _overview(context)

    upgraded = agent_v2_reply(prompt, context)
    if upgraded:
        return upgraded

    if _has_any(text, ["深度 uplift", "深度模型", "loss 面试", "算法面试", "network 面试", "tarnet", "cfrnet", "dragonnet", "efin", "ecup"]):
        return _format_answer(
            title="难点 I13：深度 uplift loss / network 面试追问",
            short="深度 uplift 不能只讲接了哪些模型，要讲每个结构解决哪个因果问题：TARNet 是双 potential-outcome heads，CFRNet 加 representation balance，DragonNet 加 propensity + targeted regularization，EFIN 建 user-treatment interaction，ECUP 做 full-funnel uplift。",
            why_hard="反事实标签缺失，BCE/MSE 只能拟合 factual outcome；深度 loss 如果不和 overlap、pseudo-outcome、policy value、calibration 对齐，很容易学到 selection bias 或 campaign ID。",
            solution="平台把 deep loss 拆成 factual outcome、MMD/IPM balance、targeted regularization、DR/R pseudo-outcome、ranking/policy/calibration、full-funnel/delayed、contrastive 和 LLM routing cost-quality，并在 Models 页给出公式、代码路径和追问 Drill。",
            tradeoff="深度模型是 P1/P2 能力，必须先和 P0 LightGBM/DR/R/CausalForest baseline 比；interaction/contrastive/LLM routing adapter 需要 guarded evidence 和随机探索日志。",
            proof="`Models -> 深度 Uplift Loss Catalog / 深度网络结构 Catalog / 深度算法面试追问 Drill`，`deepuplift/models/deep_losses.py`，`scripts/smoke_deep_loss_functions.py`，`docs/UPLIFT_DEEP_LOSS_AND_ARCHITECTURE.md`。",
            interview_line="我会按问题链讲：BCE 是 factual base，CFR 解决 representation imbalance，DragonNet 利用 propensity，EFIN 解决 treatment feature interaction，ECUP 处理 full-funnel，contrastive 改善表示，LLM routing 把强模型/RAG/tool/人工审核当成 treatment arms。",
            context=context,
        )

    if _has_any(text, ["contrastive", "对比学习", "对比 uplift", "正负样本", "representation learning", "表示学习"]):
        return _format_answer(
            title="难点 I14：contrastive uplift 和表示学习",
            short="contrastive uplift 不是按 outcome 相同就拉近，而是按 treatment effect 语义拉近：persuadable、sure thing、lost cause、sleeping dog 或 DR/R pseudo-effect bucket 才是更合理的监督信号。",
            why_hard="uplift 的反事实标签缺失，同一个用户只能看到 treatment 或 control 下的一个 factual outcome。若直接用点击/转化做正样本，encoder 会学成 response representation，而不是 treatment-effect representation。",
            solution="DeepUplift 的设计是先用随机实验真值或 DR/R learner 生成 pseudo-effect bucket，再定义同上下文、同 uplift 符号/分位桶为 positives，跨 persuadable/sure thing/lost cause/sleeping dog 为 hard negatives，并把 factual loss、supervised contrastive loss、balance/overlap 诊断一起纳入 evidence。",
            tradeoff="contrastive encoder 是高级候选，不应该替代 P0 GBM/DR/R baseline。它更适合样本量足够、特征表示复杂、segment stability 重要的场景；若 overlap 很差或 pseudo-effect 很噪，contrastive loss 会放大错误分群。",
            proof="`docs/UPLIFT_CONTRASTIVE_AND_LLM_UPLIFT.md`，`deepuplift/models/ContrastiveUpliftNet.py`，`scripts/smoke_contrastive_uplift_training.py`，Models 页 `Deep Uplift Loss Catalog`。",
            interview_line="我会说：正样本不是同 label 用户，而是同 treatment-effect 语义用户；负样本也不是随机负样本，而是跨增量人群的 hard negatives。最后仍用 Top-K uplift、calibration、segment stability 和 policy value 验证是否真的提升决策质量。",
            context=context,
        )

    if _has_any(text, ["llm routing", "大模型路由", "rag", "tool", "人工审核", "强模型", "小模型"]):
        return _format_answer(
            title="难点 I15：LLM routing 为什么是 uplift 问题",
            short="LLM routing 的 control 是 cheap model，treatment 是 strong model/RAG/tool/human review，目标是增量质量是否覆盖增量成本、延迟和风险。",
            why_hard="如果只做请求难度分类，会过度升级高风险请求；如果只做成本控制，会错过高价值请求的质量增益。历史 routing 日志还可能复制旧策略偏差。",
            solution="平台新增 LLM binary routing 和 multi-action routing synthetic datasets、policy simulator、cost-quality loss 和 evaluator schema，统一看 Top-K routing value、ROI proxy、risk guardrail 和 oracle action capture。",
            tradeoff="当前多动作 routing 是 synthetic + guarded adapter，生产要补 randomized exploration、judge calibration、budget pacing、drift monitoring 和安全审计。",
            proof="`Synthetic LLM Multi-Action Routing 9k`、`Policy -> 多动作路由策略样例`、`deepuplift/models/deep_losses.py:llm_multi_action_routing_loss`、`scripts/smoke_llm_routing_dataset.py`、`reports/industrial_scenario_compare_latest.csv`。",
            interview_line="我会说：routing 不是选最强模型，而是估计每个动作相对 cheap baseline 的增量收益。RAG、tool 和人工审核都是不同 treatment arms，成本和风险函数不同。",
            context=context,
        )

    if _has_any(text, ["ope", "off-policy", "ips", "snips", "dr ope", "离线评估新策略"]):
        return _format_answer(
            title="难点 I18：OPE / IPS / DR 怎么评估新策略",
            short="OPE 用历史 logged action、propensity 和 reward 评估一个还没全量上线的新 policy；DR OPE 把 direct reward model 和 IPS residual correction 结合起来。",
            why_hard="如果 target policy 选择的 action 在历史日志里几乎没出现，IPS 权重会爆炸，DR 也只能做外推；这时不是调公式，而是补随机探索。",
            solution="平台在 Policy 页新增 LLM 多动作 OPE mini-lab，计算 coverage、effective sample size、DM、IPS、SNIPS、DR value 和 readiness，并在 Evaluation 公式卡里补 OPE 推导。",
            tradeoff="OPE 是上线前证据，不是最终因果证明；需要 holdout、budget guardrail、延迟反馈监控和线上回收。",
            proof="`Policy -> LLM 多动作 Routing OPE Mini-Lab`，`Evaluation -> off-policy evaluation / DR OPE`，`deepuplift/core/policy.py:llm_routing_ope_summary`，`reports/llm_routing_ope_smoke_latest.json`。",
            interview_line="我会说：OPE 是 uplift 到 bandit 的桥。没有 logged propensity 和 action overlap，新 router 的离线价值就是不可信的。",
            context=context,
        )

    if _has_any(text, ["bandit", "contextual bandit", "随机探索", "预算节奏", "budget pacing", "什么时候升级"]):
        return _format_answer(
            title="难点 I17：什么时候从离线 uplift 升级到 contextual bandit",
            short="先用 uplift 做可解释、可审计的固定 policy；当预算、流量、模型池、价格或反馈持续变化时，再在随机探索和 OPE 稳定后升级到 bandit。",
            why_hard="没有随机探索时，历史日志只覆盖旧策略选择过的 action；直接上 bandit 可能优化 biased reward、耗尽预算或过载人工/强模型 action arm。",
            solution="平台在 Policy 页新增 Exploration / OPE / Bandit Guardrails：随机探索桶、IPS/DR OPE、校准固定阈值、budget pacing、contextual bandit 升级条件。",
            tradeoff="固定 uplift policy 更稳定、更好解释；bandit 更适合非平稳场景，但需要 causal logging、propensity、holdout、guardrail 和容量约束。",
            proof="`Policy -> 从离线 uplift 到在线 bandit 的上线门禁`，`deepuplift/core/policy.py:exploration_ope_guardrail_rows`，`docs/LLM_ROUTING_POLICY_PLAYBOOK.md`。",
            interview_line="我会说：uplift 是上线前的可解释 policy，bandit 是上线后的自适应控制器。没有 overlap、OPE 和 guardrails，我不会直接把 bandit 推到生产。",
            context=context,
        )

    if _has_any(text, ["drnet", "vcnet", "dose-response", "dose response", "连续 treatment", "连续treatment", "剂量响应", "补贴金额", "折扣率"]):
        return _format_answer(
            title="难点 I19：DRNet / VCNet 连续 treatment 怎么讲",
            short="当 treatment 是补贴金额、折扣率、出价倍率或触达频次时，问题不是投/不投，而是给多大强度；这需要 dose-response 和成本感知 optimizer。",
            why_hard="连续 treatment 有 positivity/support 风险：很多用户从未观察到高 dose 或低 dose，神经网络很容易在未观测区域外推；同时最大 dose 往往成本最高，未必 ROI 最优。",
            solution="平台先用 `DoseResponseGBM/RF` 做 ready baseline，输出 dose grid、recommended_dose、gain-vs-baseline，再用 `continuous_treatment_policy_optimizer` 计算 `gain*value - incremental_dose_cost`、Top-K net value 和 unsafe extrapolation。DRNet/VCNet 作为 guarded 插件，必须通过同一套训练、optimizer、support smoke 后才能晋级。",
            tradeoff="GBM baseline 稳、可解释、适合先验证数据闭环；DRNet/VCNet 能表达更平滑/复杂的 dose-response，但需要更多样本、更严格 support 诊断和调参。",
            proof="`Dose-Response -> Cost-Aware Dose Policy Optimizer`，`Evidence -> Continuous Treatment Policy Smoke`，`deepuplift/core/doseresponse.py:continuous_treatment_policy_optimizer`，`scripts/smoke_continuous_treatment_policy.py`。",
            interview_line="我会说：连续 treatment 的关键不是追求最大补贴，而是找到边际增量收益等于边际成本的位置，并明确哪些推荐剂量是在 observed support 之外。",
            context=context,
        )

    if _has_any(text, ["普通转化率", "转化率模型", "response model", "conversion model", "为什么不用", "incremental", "增量"]):
        return _format_answer(
            title="难点 I01：uplift modeling 和普通转化率预测的区别",
            short="普通模型学 `P(Y=1|X)`，容易找到本来就会转化的人；uplift 学 `E[Y|T=1,X]-E[Y|T=0,X]`，目标是找到真正被触达改变的人。",
            why_hard="业务上常把高转化人群误当作高增量人群，但这会浪费预算，甚至打扰本来会自然转化的用户。",
            solution="平台把训练输出统一成 `y0_pred / y1_pred / uplift_score`，并用 QINI、AUUC、Top-K observed uplift 和 policy value 评估排序是否真的带来增量。",
            tradeoff="转化率模型更稳定、更容易解释，但目标错了；uplift 模型方差更高，所以必须配合 overlap、bootstrap CI、calibration 和 policy value 一起看。",
            proof="`Compare` 页展示 QINI/AUUC/Top-K，`Policy` 页把 uplift score 转成成本收益曲线，`Predict` 页导出 Top-K 触达名单。",
            interview_line="我会强调：这个项目不是预测谁会买，而是预测触达谁才会多买，这也是营销投放和推荐干预里最核心的目标函数差异。",
            context=context,
        )

    if _has_any(text, ["selection bias", "ssb", "confounding", "混杂", "有偏", "偏差", "overlap 不足", "overlap不足", "观测数据"]):
        return _format_answer(
            title="难点 I03：overlap / selection bias / SSB / confounding 诊断",
            short="观测数据不能直接相信 uplift 排名，必须先证明 treatment/control 在可比区域内有足够重叠。",
            why_hard="营销和推荐日志里，谁被触达通常由历史活跃、价值、人群包策略决定，模型很容易学到投放规则本身。",
            solution="我做了 propensity overlap、feature balance、weak-overlap rate、selection bias / SSB proxy、overlap trimming 和 sensitivity/refutation checks。",
            tradeoff="诊断不能替代随机实验；它的作用是把不可比样本、弱 common support 和高风险结论暴露出来，避免把离线结果包装成确定因果结论。",
            proof="`Diagnostics` 页展示 overlap 和 SSB proxy；`Evaluation` 页展示 overlap trimming；`Sensitivity samples` 会做 random score 和 permuted treatment refutation。",
            interview_line="我会把这个讲成治理能力：系统会主动告诉你哪些结果只能探索，哪些结果可以进入灰度或 A/B 验证。",
            context=context,
        )

    if _has_any(text, ["随机实验", "treatment design", "control", "因果假设", "positivity", "unconfounded"]):
        return _format_answer(
            title="难点 I02：treatment/control 数据设计与因果假设",
            short="训练前先判断它是随机实验、观测数据、多 treatment 还是连续 treatment，再决定模型和验证口径。",
            why_hard="如果 treatment 分配不是随机的，模型看到的差异可能来自人群选择，而不是触达造成的因果效果。",
            solution="`Diagnostics` 先输出 treatment 类型、outcome 类型、treatment/control 占比、overlap、特征平衡和 SSB proxy；Agent 再根据诊断推荐 T/X/DR/R/CFR/forest 等候选模型。",
            tradeoff="随机实验可以从简单 learner 起步；观测数据必须提高稳健性权重，优先 DR/R/DML/CFR/causal forest，并把结果标成需要线上实验确认。",
            proof="Story 页的 Architecture Flow 固定把 Diagnostics 放在 Recommend 前面，Agent workflow 也是先诊断再训练候选模型。",
            interview_line="我会说：这个系统的第一步不是调模型，而是识别实验设计和因果假设，否则后面所有指标都可能是在解释 selection bias。",
            context=context,
        )

    if _has_any(text, ["registry", "adapter", "model card", "preset", "模型统一", "模型接入", "101", "模型目录"]):
        return _format_answer(
            title="难点 I04：多模型 registry、adapter、model card、preset 的工程设计",
            short="模型目录负责发现和治理模型，adapter 负责把不同后端统一成同一个训练、预测、评估契约。",
            why_hard="DeepUplift、LightGBM、EconML、scikit-uplift、XGBoost、CatBoost、CausalML 的 fit/predict 参数、依赖、输出形态都不同。",
            solution="我把模型拆成 registry、availability guard、model card、recommended preset 和 prediction contract；UI 只和统一接口交互。",
            tradeoff="不把所有包强行装进主环境。ready 模型保证可跑，optional/guarded 模型保留目录、文档和启用路径，避免 demo 被不稳定依赖拖垮。",
            proof="`Models` 页展示 catalog、family、status、缺失依赖和 guarded runtime guide；full regression 会审计最小注册量、ready 数和 model card 覆盖。",
            interview_line="我会说：模型多不是难点，难点是让 100+ 模型在一个稳定的工程契约下可选择、可解释、可验证、可降级。",
            context=context,
        )

    if _has_any(text, ["输出格式", "prediction contract", "y0", "y1", "uplift_score", "统一评估", "反向"]):
        return _format_answer(
            title="难点 I05：不同 uplift 模型输出格式不一致时如何统一评估",
            short="所有模型进入评估前都必须归一成 `y0_pred`、`y1_pred` 和 `uplift_score = y1_pred - y0_pred`。",
            why_hard="有的模型直接输出 uplift，有的输出 treatment/control outcome，有的顺序是 `[y1, y0]`，如果不封装会悄悄把 uplift 方向算反。",
            solution="我通过 adapter 和 trainer/evaluator contract 统一预测列，并把 predictions.csv、run note、metrics.json 都落盘，方便复查。",
            tradeoff="统一 contract 会牺牲一部分后端原生灵活性，但换来 Compare、Policy、Predict、History 可以复用同一套评估和导出逻辑。",
            proof="`Predict` 页固定导出 `y0_pred/y1_pred/uplift_score`，`Evaluation` 和 `Compare` 都从同一 evaluator 读取指标。",
            interview_line="我会强调这个细节很关键：uplift 方向错了不是报错，而是把该触达的人和不该触达的人完全反过来。",
            context=context,
        )

    if _has_any(text, ["qini", "auuc", "top-k", "top k", "calibration", "bootstrap", "ci", "评估体系"]):
        return _format_answer(
            title="难点 I06：QINI / AUUC / Top-K / policy value / calibration / bootstrap CI 的评估体系",
            short="QINI/AUUC 看整体排序，Top-K 看实际投放桶，policy value 看成本收益，calibration 和 bootstrap CI 看可用性与稳定性。",
            why_hard="单个离线指标容易误导：QINI 高不代表 Top 10% 可投，Top-K 高也可能是样本噪声或 calibration 很差。",
            solution="平台把 evaluator 做成组合指标：QINI、AUUC、分箱 uplift、Top-K、policy curve、calibration MAE、bootstrap CI、oracle/business-score alignment。",
            tradeoff="指标多会增加解释成本，所以我又做了 Decision Readiness，把关键 blocker 和上线风险聚合成一个面向决策的摘要。",
            proof="`Evaluation` 页给指标解释、readiness、Top-K、calibration、sensitivity；`Compare` 页按 QINI 下界或 readiness 排序；`Policy` 页输出最优阈值。",
            interview_line="我会说：我不是让面试官相信一个 QINI，而是展示一套从排序质量到业务收益再到稳定性的证据链。",
            context=context,
        )

    if _has_any(text, ["offline", "online", "线上", "上线", "ab", "a/b", "policy value", "业务收益", "投放收益"]):
        return _format_answer(
            title="难点 I07：offline metric 如何对齐 online business value",
            short="offline 指标只负责筛选候选策略，最终要用固定阈值、holdout 或 A/B 验证增量收益。",
            why_hard="离线 uplift 排序可能和真实触达成本、疲劳、库存、预算、人群排除规则不一致。",
            solution="我把 `Policy` 页做成成本收益模拟：输入单次触达成本和转化收益，自动计算 Top-K net value、推荐阈值和约束后名单。",
            tradeoff="policy value 是上线前估计，不是最终因果证明；系统会保留 evidence bundle，并建议线上实验验证 incremental conversion/revenue/net value。",
            proof="`Policy` 页有 cost-value curve、budget/max-contact constraints 和下载名单；Story 页有 Offline-To-Online Validation Playbook。",
            interview_line="我会讲：offline 的目标不是代替 A/B，而是把进入 A/B 的策略做得更少、更清楚、更有收益假设。",
            context=context,
        )

    if _has_any(text, ["workflow orchestrator", "orchestrator", "工作流编排", "聊天升级", "agent workflow", "自动工作流"]):
        return _format_answer(
            title="难点 I08：Agent workflow 如何从聊天升级成 causal workflow orchestrator",
            short="Agent 不只是回答问题，而是按 Diagnose -> Recommend -> Compare -> Explain -> Evidence 编排因果建模流程。",
            why_hard="如果 Agent 只会聊天，它无法保证每次训练前都检查假设，也无法把模型选择和证据留痕。",
            solution="`Run Causal Workflow` 会自动诊断数据、选择候选模型、套用 model-card preset、训练 compare、排序并生成 workflow report/bundle manifest。",
            tradeoff="当前是 Streamlit 内同步执行，适合 MVP 和面试演示；生产化会迁到 FastAPI + worker queue，避免长任务阻塞 UI。",
            proof="`Agent` 页有 workflow candidate preview、训练进度、compare ranking 和 `Download workflow bundle manifest`；full regression 可选跑 Agent UI smoke。",
            interview_line="我会说：这个 Agent 的价值是把因果工作流固化成可复现流程，而不是临时问答。",
            context=context,
        )

    if _has_any(text, ["证据", "复现", "evidence", "regression", "回归", "简历数字", "可信", "截图"]):
        return _format_answer(
            title="难点 I09：Evidence bundle、run history、regression gate 如何保证可复现",
            short="每次训练、对比和面试材料刷新都要落证据，并由 regression gate 检查模型目录、文档、截图和 artifact 新鲜度。",
            why_hard="简历项目最怕只能现场口述，无法证明模型数量、ready 数、截图、指标和下载包来自同一版代码。",
            solution="我做了 run artifacts、compare bundle、interview pack ZIP、Evidence Index、Story freshness screenshot 和 full_regression_check。",
            tradeoff="严格检查会让迭代慢一点，但它让项目从 demo 变成可审计的工程资产。",
            proof="`docs/DEEPUplift_AGENT_EVIDENCE_INDEX.md` 汇总证据；`scripts/full_regression_check.sh` 会刷新并验证截图、pack、模型审计和训练 smoke。",
            interview_line="我会把它讲成工程可信度：每个简历数字都能追到 JSON、截图或命令行 gate，而不是手填数字。",
            context=context,
        )

    if _has_any(text, ["guarded", "optional", "xgboost", "catboost", "causalml", "依赖", "为什么有些模型"]):
        return _format_answer(
            title="难点 I10：为什么有些模型 guarded / optional，而不是默认全部启用",
            short="模型目录覆盖面和默认可运行能力要分开治理：ready 保证稳定，guarded/optional 给出明确启用路径。",
            why_hard="工业 uplift 后端依赖复杂，XGBoost、CatBoost、CausalML 可能受本机 runtime、Python 版本、编译包影响。",
            solution="我把这类后端纳入 catalog 和 model card，但用 dependency guard、环境变量或 helper env 控制启用，UI 显示缺失依赖和原因。",
            tradeoff="这比宣称全部默认可跑更诚实，也能保护主 demo 流程；需要某后端时再按 guide 安装和单独 smoke。",
            proof="`Models` 页有 Guarded Backend Runtime Guide，`docs/DEEPUplift_AGENT_MODEL_BACKEND_STRATEGY.md` 解释 XGBoost/CatBoost/CausalML 路径。",
            interview_line="我会说：我的原则是 reliable demo over fake coverage，模型治理要能解释为什么可用、为什么暂缓、怎么启用。",
            context=context,
        )

    if _has_any(text, ["fastapi", "react", "worker", "生产化", "架构演进", "streamlit mvp", "生产系统"]):
        return _format_answer(
            title="难点 I11：Streamlit MVP 到 FastAPI + worker + React 的架构演进",
            short="当前 Streamlit 负责快速闭环和现场演示，生产化会把数据、训练、评估、artifact、权限拆成 API 和异步任务。",
            why_hard="uplift compare 可能训练多个模型，耗时长且需要 artifact lineage；单进程 UI 不适合多人协作、权限和任务恢复。",
            solution="文档里设计了 FastAPI endpoints、worker queue、object storage、metadata DB、RBAC、model promotion gate 和 evidence lineage。",
            tradeoff="先做 Streamlit 是为了快速验证 causal workflow；等 contract 稳定后再迁移前后端，避免过早工程化。",
            proof="Story 页有 Productionization Roadmap 和 Architecture Diagram；`docs/DEEPUplift_AGENT_ARCHITECTURE.md` 说明数据流和服务边界。",
            interview_line="我会讲：MVP 的重点是证明产品和因果流程成立，生产化的重点是异步训练、审计、权限和可复现。",
            context=context,
        )

    if _has_any(text, ["简历", "项目怎么讲", "面试开场", "自我介绍", "resume", "pitch", "怎么表达"]):
        return _format_answer(
            title="难点 I12：简历和面试中的项目表达",
            short="把它定位成面向增长/营销/推荐的 Uplift Modeling & Causal Decision Agent，而不是一个模型训练页面。",
            why_hard="如果只说加了很多模型，听起来像堆库；真正有含金量的是因果假设诊断、模型治理、决策评估和证据闭环。",
            solution="简历表达围绕 4 件事：因果诊断、101 模型 registry、自动 compare/policy/predict 工作流、evidence bundle + regression gate。",
            tradeoff="少讲 UI 炫酷，多讲决策链路和可复现；模型数量是亮点，但不是唯一卖点。",
            proof="Story 页的 Live Resume Snapshot、Chinese Project One-Pager、Interview Evidence Pack 都能下载；模型审计显示 catalog/ready 数。",
            interview_line="开场可以说：我把 DeepUplift 从模型脚本库改造成 causal decision workbench，支持上传数据、诊断偏差、多模型对比、投放阈值模拟、名单导出和证据包回归验证。",
            context=context,
        )

    return None
