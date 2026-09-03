from __future__ import annotations

from collections import Counter


PUBLIC_PRODUCT_AUDIT: list[dict[str, str]] = [
    {
        "product": "Volcengine DataLeap",
        "area": "Data governance / data R&D",
        "official_url": "https://developer.volcengine.com/articles/7208366445791543356",
        "source_type": "official public article",
        "stars_or_scale": "n/a - public product",
        "latest_update": "official article/docs active in 2026 crawl",
        "activity": "active public docs/articles",
        "page_structure": "Data source -> integration -> processing -> service -> governance -> asset map -> security",
        "core_capability": "data integration, data development, operations, governance, asset map, fine-grained permissions",
        "ui_pattern": "governance cockpit with plan, diagnosis, acceptance and review; asset map with lineage and permission audit",
        "visual_pattern": "enterprise control-plane layout, restrained status colors, dense resource tables, governance progress cards",
        "deepuplift_absorption": "Evidence and Workflow should show dataset manifest, source gate, artifact lineage, smoke gates and ownership as first-class evidence.",
        "streamlit_direct": "ready-pattern",
        "future_react_fit": "ready",
        "dependency_risk": "low as design reference; no runtime dependency",
        "migration_cost": "low",
        "status": "ready",
        "interview_line": "I borrowed DataLeap's governance idea: every data/model/policy claim needs lineage, quality and audit status.",
    },
    {
        "product": "Volcengine ByteHouse",
        "area": "Realtime warehouse / OLAP",
        "official_url": "https://www.volcengine.com/docs/6517/76321",
        "source_type": "official docs",
        "stars_or_scale": "n/a - public product",
        "latest_update": "official docs active in 2026 crawl",
        "activity": "active public docs",
        "page_structure": "Compute groups -> SQL worksheets -> data import/export -> tasks -> monitoring -> audit logs",
        "core_capability": "realtime analytics, elastic compute, resource isolation, SQL worksheets, monitoring and audit",
        "ui_pattern": "separate analytics result from compute/resource health; expose latency, freshness and isolation as operating signals",
        "visual_pattern": "low-latency analytics console, compact query/task tables, resource status and audit entries",
        "deepuplift_absorption": "Policy should expose freshness, supported audience coverage and OPE coverage before recommending a launch threshold.",
        "streamlit_direct": "inspired",
        "future_react_fit": "optional",
        "dependency_risk": "low as pattern; high only if adding warehouse integration",
        "migration_cost": "medium",
        "status": "inspired",
        "interview_line": "ByteHouse inspires the realtime decision layer: ROI curves are only useful if freshness and compute health are visible.",
    },
    {
        "product": "Volcengine DataFinder",
        "area": "Growth analytics",
        "official_url": "https://www.volcengine.com/docs/84129/1261564",
        "source_type": "official docs",
        "stars_or_scale": "n/a - public product",
        "latest_update": "official docs active in 2026 crawl",
        "activity": "active public docs",
        "page_structure": "Metrics -> time range -> segment/filter -> detail hover -> trend/drilldown",
        "core_capability": "active users, new users, stickiness, growth metric analysis and segmentation",
        "ui_pattern": "metric dictionary, time filter, segment filter and hover detail are visible together",
        "visual_pattern": "growth dashboard with core KPI cards, trend panels and slice filters",
        "deepuplift_absorption": "Evaluation should pair QINI/AUUC with business metric meaning, audience segment and launch bucket.",
        "streamlit_direct": "ready-pattern",
        "future_react_fit": "ready",
        "dependency_risk": "low",
        "migration_cost": "low",
        "status": "ready",
        "interview_line": "DataFinder inspired the metric dictionary: uplift metrics must explain business meaning and segment scope.",
    },
    {
        "product": "Volcengine DataTester",
        "area": "A/B testing / experiment lifecycle",
        "official_url": "https://www.volcengine.com/docs/56651/783746",
        "source_type": "official docs",
        "stars_or_scale": "n/a - public product",
        "latest_update": "official docs active in 2026 crawl",
        "activity": "active public docs",
        "page_structure": "Access app -> choose experiment type -> design experiment -> create experiment -> traffic split -> analysis",
        "core_capability": "A/B experiment setup, SDK/data access, experiment design and visual experiment workflows",
        "ui_pattern": "launch checklist before experiment creation; experiment type and traffic split are explicit",
        "visual_pattern": "stepper-like experiment preparation flow, clear owner/action gates",
        "deepuplift_absorption": "Policy should require holdout, overlap/OPE support and launch-readiness checks before online treatment.",
        "streamlit_direct": "ready-pattern",
        "future_react_fit": "ready",
        "dependency_risk": "low as pattern",
        "migration_cost": "low",
        "status": "ready",
        "interview_line": "DataTester is the online-validation reference: offline uplift becomes credible only when launch gates and holdout design are explicit.",
    },
    {
        "product": "Volcengine Machine Learning Platform",
        "area": "ML platform / training lifecycle",
        "official_url": "https://www.volcengine.com/docs/6459",
        "source_type": "official docs",
        "stars_or_scale": "n/a - public product",
        "latest_update": "official docs active in 2026 crawl",
        "activity": "active public docs",
        "page_structure": "Data hosting -> code development -> training -> model management -> deployment/inference",
        "core_capability": "development machines, custom training, multi-framework inference and model lifecycle workflow",
        "ui_pattern": "lifecycle navigation from data/code to train/deploy with explicit resources and images",
        "visual_pattern": "ML workbench with resource queues, jobs, models and service states",
        "deepuplift_absorption": "Models/Train/History should separate model registry, run evidence, deployment readiness and resource assumptions.",
        "streamlit_direct": "inspired",
        "future_react_fit": "future-react",
        "dependency_risk": "medium if integrated; low as UI reference",
        "migration_cost": "medium",
        "status": "future-react",
        "interview_line": "The ML platform reference explains the future FastAPI + worker split: data, code, jobs, model registry and serving are separate contracts.",
    },
    {
        "product": "Volcengine DataWind",
        "area": "BI / data insight",
        "official_url": "https://www.volcengine.com/docs/86403/1829774",
        "source_type": "official docs",
        "stars_or_scale": "n/a - public product",
        "latest_update": "official docs active in 2026 crawl",
        "activity": "active public docs",
        "page_structure": "Dataset -> visual query -> dashboard -> field descriptions -> access statistics",
        "core_capability": "visual query, dashboard, metric explanation, resource access statistics",
        "ui_pattern": "field descriptions and metric explanations appear near charts; dashboard usage is observable",
        "visual_pattern": "business BI surface with readable chart titles, field help and resource PV/UV statistics",
        "deepuplift_absorption": "Evaluation and Evidence should explain each metric beside the chart and show whether screenshots/pages are used.",
        "streamlit_direct": "inspired",
        "future_react_fit": "optional",
        "dependency_risk": "low",
        "migration_cost": "low",
        "status": "inspired",
        "interview_line": "DataWind inspired metric explainability: a chart without field meaning and usage evidence is not enough for decision makers.",
    },
    {
        "product": "Volcengine Data Agent",
        "area": "Data + AI agent",
        "official_url": "https://developer.volcengine.com/articles/7517866292922843147",
        "source_type": "official public article",
        "stars_or_scale": "n/a - public product",
        "latest_update": "official article/docs active in 2026 crawl",
        "activity": "active public articles",
        "page_structure": "Question -> analysis -> evidence -> action recommendation -> marketing/insight workflow",
        "core_capability": "enterprise data agent for intelligent analysis and intelligent marketing",
        "ui_pattern": "agent answer must connect data analysis to action and cite traceable evidence",
        "visual_pattern": "assistant as workbench copilot, not a detached chat box",
        "deepuplift_absorption": "Agent should answer UI/product references with evidence cards and point to concrete pages and docs.",
        "streamlit_direct": "ready-pattern",
        "future_react_fit": "ready",
        "dependency_risk": "low as pattern",
        "migration_cost": "low",
        "status": "ready",
        "interview_line": "The public Data Agent pattern supports my design: the assistant should convert analysis into governed actions, not just talk.",
    },
    {
        "product": "Volcengine Intelligent Recommendation Platform",
        "area": "Recommendation / online service",
        "official_url": "https://www.volcengine.com/docs/6435/69166",
        "source_type": "official docs",
        "stars_or_scale": "n/a - public product",
        "latest_update": "official docs active in 2026 crawl",
        "activity": "active public docs",
        "page_structure": "Data management -> feature engineering -> model development -> recall/ranking/rules -> A/B testing -> effects",
        "core_capability": "end-to-end recommendation service, feature engineering, recall/ranking/rules and A/B evaluation",
        "ui_pattern": "online service flow makes recall, ranking, rules and experiment evaluation explicit",
        "visual_pattern": "service pipeline plus effect dashboard, suitable for online recommendation decisions",
        "deepuplift_absorption": "Scenario and Policy should show where uplift sits in recommendation: treatment is intervention, metric is incremental value, gate is online A/B.",
        "streamlit_direct": "inspired",
        "future_react_fit": "future-react",
        "dependency_risk": "low as pattern; high as full platform",
        "migration_cost": "high",
        "status": "future-react",
        "interview_line": "Recommendation-platform references explain why uplift should connect to recall/ranking/rules and online effect validation.",
    },
    {
        "product": "Volcengine VeCDP / GMP",
        "area": "Customer data / growth marketing",
        "official_url": "https://www.volcengine.com/docs/6356",
        "source_type": "official docs",
        "stars_or_scale": "n/a - public product",
        "latest_update": "official docs active in 2026 crawl",
        "activity": "active public docs",
        "page_structure": "Data prep -> ID graph -> tags/audience packs -> insight -> channel activation",
        "core_capability": "customer data platform and growth marketing activation over audience assets",
        "ui_pattern": "audience pack, tag governance, channel activation and approval workflow are one decision loop",
        "visual_pattern": "growth operation console with audience/label/channel/action separation",
        "deepuplift_absorption": "Policy should treat Top-K audience export as an activation asset with frequency, channel and approval guardrails.",
        "streamlit_direct": "inspired",
        "future_react_fit": "optional",
        "dependency_risk": "low as pattern",
        "migration_cost": "medium",
        "status": "inspired",
        "interview_line": "CDP/GMP references help explain uplift as audience activation governance, not only model scoring.",
    },
]


PUBLIC_PRODUCT_INSPIRATION_MATRIX: list[dict[str, str]] = [
    {
        "pattern": "Data asset governance cockpit",
        "borrowed_from": "DataLeap / DataWind",
        "absorbed_now": "Workflow and Evidence expose data manifest, source gate, screenshots, model gate and artifact ownership.",
        "deepuplift_value": "Keeps causal claims tied to data quality, lineage and reproducibility.",
        "future_react": "Asset detail route with lineage graph and permission/audit drawer.",
        "status": "ready",
    },
    {
        "pattern": "Experiment launch checklist",
        "borrowed_from": "DataTester / MLflow / Evidently",
        "absorbed_now": "Policy launch guardrail board checks design, overlap, OPE coverage, threshold, budget and evidence pack.",
        "deepuplift_value": "Prevents offline uplift metrics from being mistaken for online launch proof.",
        "future_react": "Experiment creation wizard with holdout/traffic split and approval workflow.",
        "status": "ready",
    },
    {
        "pattern": "Growth metric dictionary",
        "borrowed_from": "DataFinder / DataWind",
        "absorbed_now": "Evaluation and Policy explain QINI/AUUC/Top-K/OPE/policy value in business terms.",
        "deepuplift_value": "Turns model metrics into growth decision language.",
        "future_react": "Metric catalog with formulas, owner, freshness and linked dashboards.",
        "status": "ready",
    },
    {
        "pattern": "Realtime decision operating signals",
        "borrowed_from": "ByteHouse / Ray Dashboard / Grafana",
        "absorbed_now": "Workflow shows freshness, run/evidence state and next action; Policy shows coverage and risk before launch.",
        "deepuplift_value": "Adds operating health to the causal decision, not just offline score quality.",
        "future_react": "Live status cards for data freshness, job state, warehouse latency and policy drift.",
        "status": "inspired",
    },
    {
        "pattern": "Agent-to-action workbench",
        "borrowed_from": "Data Agent / Open WebUI / Dify",
        "absorbed_now": "Agent answers cite pages, docs and evidence cards for product/UI references.",
        "deepuplift_value": "Makes the assistant an evidence-aware causal copilot.",
        "future_react": "Right-side evidence drawer with action buttons and run timeline.",
        "status": "ready",
    },
    {
        "pattern": "Online recommendation service flow",
        "borrowed_from": "Intelligent Recommendation Platform / Kubeflow / LangGraph Studio",
        "absorbed_now": "Scenario and Policy describe treatment, ranking, rules, threshold and online A/B gate.",
        "deepuplift_value": "Frames uplift as a deployable decision layer inside recommendation/growth systems.",
        "future_react": "React Flow DAG for feature -> train -> rank -> rule -> experiment -> monitor.",
        "status": "future-react",
    },
]


GROWTH_CONTROL_PLANE_ROWS: list[dict[str, str]] = [
    {
        "stage": "Data Asset",
        "signal": "manifest / schema / lineage",
        "borrowed_from": "DataLeap asset map",
        "deepuplift_gate": "treatment, outcome, features and artifact path are visible before training.",
        "status": "ready",
    },
    {
        "stage": "Experiment Design",
        "signal": "holdout / treatment type / traffic support",
        "borrowed_from": "DataTester experiment preparation",
        "deepuplift_gate": "offline policy must be paired with holdout or OPE plan before launch.",
        "status": "guarded",
    },
    {
        "stage": "Audience & Segment",
        "signal": "Top-K / cohort / channel",
        "borrowed_from": "DataFinder + VeCDP audience packs",
        "deepuplift_gate": "Top-K audience export has channel, frequency and segment-risk notes.",
        "status": "ready",
    },
    {
        "stage": "Decision Metric",
        "signal": "QINI / AUUC / ROI / OPE",
        "borrowed_from": "DataWind metric explanation",
        "deepuplift_gate": "every metric has business meaning and failure warning.",
        "status": "ready",
    },
    {
        "stage": "Action & Guardrail",
        "signal": "threshold / budget / risk",
        "borrowed_from": "GMP approval + Evidently gates",
        "deepuplift_gate": "policy launch board checks threshold, budget, OPE coverage and evidence pack.",
        "status": "decision",
    },
    {
        "stage": "Agent Copilot",
        "signal": "question -> evidence -> action",
        "borrowed_from": "Data Agent",
        "deepuplift_gate": "Agent answers link to source matrices, docs, screenshots and local code paths.",
        "status": "ready",
    },
]


LAUNCH_GUARDRAIL_ROWS: list[dict[str, str]] = [
    {
        "gate": "数据设计",
        "check": "treatment/control、outcome、feature timing 已明确",
        "why": "不先定义干预和结果，uplift score 无法解释。",
        "page": "Workflow / Diagnostics",
        "status": "required",
    },
    {
        "gate": "样本支持",
        "check": "overlap、propensity、segment support 无明显 blocker",
        "why": "Top-K 人群若缺少 control 支持，offline uplift 可能是外推。",
        "page": "Diagnostics / Evaluation",
        "status": "required",
    },
    {
        "gate": "模型对比",
        "check": "至少 P0 baseline + robust learner 对比，有 winner reasoning",
        "why": "单模型结果不能说明策略稳健。",
        "page": "Compare / History",
        "status": "required",
    },
    {
        "gate": "业务价值",
        "check": "Top-K ROI / policy value 覆盖触达成本、补贴成本和风险成本",
        "why": "QINI/AUUC 高不等于上线赚钱。",
        "page": "Policy",
        "status": "required",
    },
    {
        "gate": "OPE / Holdout",
        "check": "新策略有 logged propensity、coverage/ESS 或在线 holdout 计划",
        "why": "没有支持度的新 policy 不能靠离线估计直接上线。",
        "page": "Policy / Evidence",
        "status": "guarded",
    },
    {
        "gate": "证据包",
        "check": "metrics、predictions、run note、screenshot、source gate 可下载",
        "why": "面试/评审需要可复现，而不是口头解释。",
        "page": "Evidence",
        "status": "required",
    },
]


PUBLIC_PRODUCT_PROMPT_GROUPS: list[dict[str, list[str] | str]] = [
    {
        "group": "公开产品参考",
        "intent": "解释本轮参考的火山/字节公开数据产品，以及如何吸收到 DeepUplift。",
        "prompts": [
            "参考了哪些字节/火山引擎公开数据产品？",
            "哪些公开产品设计被吸收到当前页面？",
            "为什么只参考公开资料，不复制私有 UI？",
            "DeepUplift 怎么像增长分析和 A/B 实验平台？",
        ],
    }
]


def public_product_audit_rows() -> list[dict[str, str]]:
    return [dict(row) for row in PUBLIC_PRODUCT_AUDIT]


def public_product_inspiration_rows() -> list[dict[str, str]]:
    return [dict(row) for row in PUBLIC_PRODUCT_INSPIRATION_MATRIX]


def growth_control_plane_rows() -> list[dict[str, str]]:
    return [dict(row) for row in GROWTH_CONTROL_PLANE_ROWS]


def launch_guardrail_rows() -> list[dict[str, str]]:
    return [dict(row) for row in LAUNCH_GUARDRAIL_ROWS]


def public_product_prompt_groups() -> list[dict[str, list[str] | str]]:
    return [dict(row) for row in PUBLIC_PRODUCT_PROMPT_GROUPS]


def public_product_counts() -> dict[str, int]:
    rows = public_product_audit_rows()
    status_counts = Counter(row["status"] for row in rows)
    return {
        "total": len(rows),
        "ready": status_counts.get("ready", 0),
        "inspired": status_counts.get("inspired", 0),
        "future_react": status_counts.get("future-react", 0),
        "backlog": status_counts.get("backlog", 0),
    }


def public_product_agent_reply(prompt: str) -> str | None:
    text = prompt.strip().lower()
    if not text:
        return None
    trigger_terms = [
        "字节",
        "火山",
        "volcengine",
        "dataleap",
        "bytehouse",
        "datafinder",
        "datatester",
        "datawind",
        "vedi",
        "公开产品",
        "增长分析",
        "a/b",
        "ab测试",
        "data agent",
        "私有",
        "复制",
    ]
    if not any(term in text for term in trigger_terms):
        return None
    counts = public_product_counts()
    ready = [row["product"] for row in PUBLIC_PRODUCT_AUDIT if row["status"] == "ready"]
    future = [row["product"] for row in PUBLIC_PRODUCT_AUDIT if row["status"] == "future-react"]

    if any(key in text for key in ["吸收", "设计", "页面", "像增长", "a/b"]):
        patterns = "\n".join(
            f"- **{row['pattern']}**：{row['absorbed_now']} 价值：{row['deepuplift_value']}"
            for row in PUBLIC_PRODUCT_INSPIRATION_MATRIX
        )
        return (
            "### 已吸收的公开数据产品设计\n\n"
            f"{patterns}\n\n"
            "落地位置：`Workflow -> Growth Decision Control Plane`、`Models -> ByteDance-style Public Product Inspiration Matrix`、"
            "`Policy -> Launch Guardrail Board`、`Agent -> 公开产品参考`。"
        )

    if any(key in text for key in ["为什么", "私有", "复制"]):
        return (
            "### 为什么只参考公开资料\n\n"
            "我只采用火山引擎官网、官方文档、开发者社区公开文章和开源 GitHub。"
            "吸收的是信息架构：数据治理、指标解释、实验 lifecycle、上线护栏和 Agent-to-action，"
            "不复制任何公司私有 UI、品牌元素或内部交互。这样面试时可以讲产品判断和工程抽象，而不是视觉临摹。"
        )

    return (
        "### 字节/火山公开产品参考\n\n"
        f"本轮审计了 `{counts['total']}` 个公开数据产品方向。ready pattern：`{', '.join(ready)}`。\n\n"
        f"未来 React/FastAPI 更适合承接的方向：`{', '.join(future)}`。\n\n"
        "面试讲法：我把 DataLeap 的治理、DataFinder 的增长指标、DataTester 的实验生命周期、"
        "ByteHouse 的实时分析信号和 Data Agent 的 evidence-to-action 思路，转成 DeepUplift 的 causal decision workbench。"
    )
