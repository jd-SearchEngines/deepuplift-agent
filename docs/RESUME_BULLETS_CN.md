# DeepUplift Agent 中文简历素材

## 项目标题

DeepUplift Agent：面向增长/营销/推荐场景的 Uplift Modeling & Causal Decision Workbench

## 简历 Bullet Points

- 搭建端到端 uplift modeling / causal decision workbench，支持数据上传、因果诊断、模型自动推荐、多模型训练对比、投放策略评估、预测打分导出和 evidence bundle 生成。
- 设计统一 model registry、model card 和 parameter preset 体系，覆盖 **103 个 uplift/causal 模型**，其中 **73 个模型可在本地环境直接运行**，包括 DeepUplift、LightGBM、EconML、scikit-uplift 等后端。
- 实现 Agent workflow，将 Diagnostics -> Candidate Recommendation -> Model Compare -> Readiness Scoring -> Report/Bundle Export 串成可复现的因果决策流程。
- 将评估从单一 QINI/AUUC 扩展到 decision-grade evaluation，包含 Top-K uplift、policy value、calibration、overlap diagnostics、bootstrap CI、sensitivity check、segment analysis、full-funnel ECUP、delayed feedback windows 和 decision readiness。
- 构建 Industrial Scenario Lab，模拟发券/补贴、用户增长/Push、广告增量、推荐干预、marketplace 补贴、LLM routing，以及百度外卖天降红包、滴滴 C 端乘客补贴、滴滴 B 端司机补贴、滴滴城市/时段预算分配、Shopee 广告 iROAS、Spotify 风格站内消息、DoorDash 风格 ghost ads、商家补贴/共投毛利、电商多券面额、CRM 多 journey 重叠触达、连续出价/折扣剂量响应、Geo holdout 地理增量实验、游戏运营礼包、金融授信/利率优惠、客服升级人工、医疗随访、SaaS 续费、LLM 多动作 RAG/tool/人工审核 routing 等 **24 类工业/履历业务数据**，并打通 manifest、trainer、evaluator、policy value、smoke、Evidence 和 Agent 问答。
- 新增履历业务 Policy Lab，将百度外卖/滴滴/Shopee 场景转成 Top-K policy value、ROI、风险均值和阈值表，并实现滴滴 city-time 预算分配 simulator，支持总预算约束下的城市/时段/C-B 侧预算建议。
- 重构中文优先的履历场景工作台，把百度外卖天降红包、滴滴 C 端补贴、滴滴 B 端司机补贴、滴滴城市/时段预算分配、Shopee 广告 iROAS 拆成独立业务页；每个场景展示 schema、模型推荐、Model Compare、Top-K ROI、V0-V6 优化路径、模型先进性审计和依赖限制。
- 为场景内 Model Compare 增加业务综合评分和上线判断，将 Top-K policy value、ROI/iROAS、QINI、Top10 uplift、readiness 加权成可解释排序，并输出“为什么 QINI 高但不一定适合上线”的排序理由。
- 新增从离线 uplift 到在线 contextual bandit 的上线门禁，覆盖随机探索桶、IPS/DR OPE、校准固定阈值、budget pacing 和 bandit 升级条件，用于解释 LLM routing、广告投放、补贴预算和人工审核队列的生产演进路径。
- 实现 LLM 多动作 routing OPE mini-lab，基于 logged action/propensity 计算 DM、IPS、SNIPS、DR policy value、coverage 和 effective sample size，用可跑数据解释“新策略上线前如何离线评估，以及何时必须补随机探索”。
- 增强连续 treatment / dose-response 决策能力，将补贴金额、折扣率、出价倍率转成 `recommended_dose` 和 `gain-vs-baseline`，新增成本感知 dose optimizer、unsafe extrapolation 诊断和 DRNet/VCNet guarded promotion gate，解释“最大补贴不等于最优 ROI”。
- 系统化审计 24 个 uplift/CATE/causal ML/OPE/LLM routing/research agent 开源框架（EconML、CausalML、scikit-uplift、UTBoost、pylift、grf、DoWhy、DoubleML、YLearn、causallib、CausalTune、GeoLift、UpliftML、CausalLift、causal-learn、LightGBM、XGBoost/CatBoost、RouteLLM、Open Bandit Pipeline、Vowpal Wabbit、Open Deep Research、GPT Researcher、Local Deep Researcher、深度 uplift 论文代码），按 ready/guarded/optional/backlog 标注依赖限制、adapter 路线、业务场景和 evidence gate。
- 将 GitHub 框架研究产品化为可刷新 evidence：新增 RouteLLM、Open Bandit Pipeline、Vowpal Wabbit、LangGraph Open Deep Research、GPT Researcher 等参考审计，区分训练后端、OPE/bandit 层、LLM routing 层和 research evidence 层，并用 GitHub metadata、adapter route、UI proof、smoke gate 管理是否晋级 ready。
- 深化评估体系，将 ITE/CATE、QINI、AUUC、uplift@K、policy value、incremental profit、iROAS、DR pseudo-outcome、R-learner objective、calibration、bootstrap CI、overlap trimming、delayed feedback、ECUP full-funnel、LLM routing cost-quality 等 18 类指标做成公式卡、推导卡和 Agent 面试问答。
- 构建模型风险诊断与场景问题库，沉淀 **14 类模型 failure modes、9 类业务 pitfall、10 个 mock diagnostic cases**，覆盖 weak uplift、poor overlap、propensity 极端、QINI 高但 ROI 低、延迟反馈、全链路广告、geo holdout 和 LLM routing 成本陷阱，并接入 Models/Evaluation/Evidence/Agent 页面。
- 扩展 Uplift + LLM 前沿场景，将 LLM routing 建模为 cheap model vs strong/RAG/tool/人工审核多动作 treatment-effect / cost-quality policy 问题，并在 Policy 页面实现本地化 ROI simulator 和多动作路由策略样例，在 Story/Knowledge 中展示 LLM Interview Q&A 矩阵。
- 增强深度 uplift 算法能力：修复 CFRNet 表示平衡损失为 Torch 可微 IPM/MMD，新增 `ContrastiveUpliftNet`，实现 pairwise uplift ranking、cost-aware policy、uplift calibration、contrastive uplift、ECUP full-funnel、LLM routing cost-quality 和 `llm_multi_action_routing_loss` 等可复用 loss，并配套梯度 smoke 与训练 smoke。
- 构建 Streamlit 可视化工作台，覆盖 Data、Diagnostics、Train、Compare、Predict、Policy、History、Models、Knowledge、Story、Agent 等页面，支持现场演示和业务解释。
- 建立回归验证和实验治理机制，包括 model catalog audit、preset audit、no-UI compare smoke、browser screenshot smoke、artifact validation、run tags/notes、evidence summary 和 evidence ZIP。
- 新增场景工作台回归门：`scripts/smoke_resume_scenario_workbench.py` 检查 5 个履历场景、19 个工业扩展场景、72 条全量 scenario compare 结果、15 条履历 Top-K policy benchmark、35 条 V0-V6 优化路径、前沿模型审计、模型限制和 Agent 回答。
- 增加简历级项目材料和可验证证据链，包括 Story 页面、一键 interview demo、架构说明、风险登记、live resume snapshot、evidence index 和 8 分钟 walkthrough。

## 30 秒面试讲法

这个项目的核心不是“多加几个 uplift 模型”，而是把 uplift modeling 产品化成一个 causal decision workflow。用户上传 treatment/outcome 数据后，系统先做数据设计和 overlap 诊断，再基于模型卡和知识规则推荐候选模型，自动训练和比较多个模型，最后用 QINI、AUUC、Top-K uplift、policy value、calibration、bootstrap CI 和 readiness score 给出可解释的投放建议，并导出完整 evidence bundle。

## 最强简历版本

```text
DeepUplift Agent | Uplift Modeling & Causal Decision Workbench
- 搭建面向增长/CRM/推荐场景的因果决策工作台，支持数据诊断、uplift 模型训练、多模型对比、policy value 投放模拟、预测打分导出和 evidence bundle 生成。
- 设计统一 registry/model-card/preset 体系，覆盖 103 个 uplift/causal 模型，73 个模型可本地运行，集成 DeepUplift、LightGBM、EconML、scikit-uplift 等后端。
- 实现 Agent workflow，自动完成 Diagnostics -> Candidate Recommendation -> Model Compare -> Readiness Scoring -> Report/Bundle Export。
- 引入 decision-grade evaluation：QINI、AUUC、Top-K uplift、policy value、calibration、overlap diagnostics、bootstrap CI、sensitivity check、segment analysis、full-funnel ECUP、delayed feedback uplift 和 readiness scoring。
- 建设工业场景实验室：24 类 synthetic datasets 覆盖发券利润、增长延迟反馈、广告全链路、推荐干预、平台补贴、LLM routing、LLM 多动作 RAG/tool/人工审核 routing、Spotify 风格 message diet、DoorDash ghost ads、商家补贴共投、电商多券面额、CRM 多 journey 重叠触达、连续出价/折扣剂量响应、Geo holdout 地理增量实验、游戏礼包、金融授信、客服升级人工、医疗随访、SaaS 续费，以及百度外卖/滴滴/Shopee 履历场景，并用统一训练/评估/证据链验证。
- 建设中文场景工作台：每个履历场景都有独立模型对比、Top-K ROI、风险指标、V0-V6 从规则模型到 policy optimizer 的优化路径，以及 ready/guarded/optional/backlog 模型审计。
- 场景模型对比不是单看 QINI，而是以 policy value/ROI 为主、readiness 为护栏，形成“baseline -> robust learner -> tree/forest -> deep model -> policy optimizer”的可讲优化路径。
- 开源框架接入不是简单调用库，而是把框架拆成训练后端、诊断/refutation、评估指标、optional 插件和 Evidence gate 五层；EconML/LightGBM/scikit-uplift 作为 P0 ready，UTBoost/CausalML/grf/DoWhy/DoubleML 等按依赖风险做 guarded/optional 接入。
- 评估体系有明确公式和业务解释：QINI/AUUC 负责排序，uplift@K 负责投放桶，policy value/ROI/iROAS 负责上线收益，calibration/bootstrap/overlap/sensitivity 负责可信度，delayed/full-funnel/LLM routing 指标负责特殊工业场景。
- 模型风险诊断体系覆盖 14 类 estimator failure、9 类业务 pitfall 和 10 个 mock diagnostic cases，把 weak uplift、extreme propensity、high-QINI-low-ROI、delayed feedback、full-funnel click trap、geo holdout conflict 和 LLM routing cost trap 做成可复现数据、smoke、UI evidence 和 Agent 面试问答。
- 增加 Uplift + LLM 能力：LLM causal copilot、text/creative treatment、cheap/strong/RAG/tool/人工审核多动作 routing、cost-quality simulator、随机探索/预算节奏面试问答和 source-grounded evidence card。
- 补强深度模型细节：CFRNet differentiable representation balance、DragonNet targeted regularization、EFIN/DESCN loss stack、ContrastiveUpliftNet、LLM routing cost-quality / multi-action routing loss，并用 smoke 脚本验证梯度和训练证据。
- 构建 regression/gov 机制：model catalog audit、no-UI compare smoke、browser screenshot smoke、artifact validation、run notes、tags、evidence summary 和 downloadable evidence ZIP。
```

## 深度算法面试补充

```text
这个项目不仅是接模型框架。我把 uplift 的 loss 设计拆成 factual outcome、representation balance、DR/R pseudo-outcome、ranking/policy value、calibration、full-funnel/delayed feedback、contrastive representation 和 LLM routing cost-quality。比如 CFRNet 的 IPM/MMD balance 必须在 Torch 内可微，否则只是日志项不会训练 encoder；这部分我补了梯度 smoke 和端到端训练 smoke。
```

## 面试可强调的技术关键词

- Causal inference / uplift modeling / heterogeneous treatment effect
- S/T/X/DR/R learner, Causal Forest, DragonNet, CFRNet, TarNet
- Model registry, model card, parameter preset, dependency readiness
- QINI, AUUC, Top-K uplift, policy value, calibration, bootstrap CI
- Agent workflow orchestration
- LLM routing, text treatment, causal copilot, cost-quality policy
- Experiment governance, evidence bundle, regression gate
- Frontend framework audit, Streamlit data workbench, visual regression evidence
- shadcn/ui, Tremor, Grafana, Metabase, Superset, Dify, Langflow, ECharts, Plotly

## 可借鉴的工业表达

- 增长触达场景：push、短信、优惠券、电话外呼、RTA、资源位、弹窗、CRM 召回。
- 业务目标表达：增量 DAU、增量转化、ROI、Top-K 人群收益、policy value、触达成本约束下的净收益。
- 因果建模表达：从传统 uplift / CATE baseline 到 DragonNet、DRNet、VCNet、VC-Transformer、GGBM 等深度/工业 uplift 模型演进。
- 消偏表达：SSB 问题、探索组消偏、propensity overlap、treatment/control balance、calibration、bootstrap AB 评估。
- 评估指标表达：QINI、AUUC、Top-K recall、DM-score / direct-method policy value、DM Top-K recall、offline-online metric alignment。
- 工程落地表达：从 0-1 搭建因果推断框架，从 1-N 扩展为平台化能力，沉淀模型注册、模型卡、证据包、回归门和可视化工作台。

## 前端框架研究与可视化体验升级补充

- 系统审计 26 个官方 GitHub / 官方文档前端与数据产品参考，包括 Streamlit、Dash、Panel、Gradio、Plotly、Apache ECharts、Altair、Metabase、Superset、Grafana、shadcn/ui、Tremor、Ant Design Pro、Material UI、Carbon、Radix、Dify、Langflow、Flowise、Open WebUI、LangGraph Studio、MLflow、Evidently、W&B 和 React Flow。
- 将前端调研产品化为 `Frontend Framework Audit / UI Inspiration Matrix`，在 Models 页面展示 repo 链接、stars、活跃度、适配页面、可借鉴 pattern、Streamlit 直接吸收价值、未来 React/FastAPI 适配、依赖风险、迁移成本和面试讲法。
- 基于 shadcn/ui、Tremor、Carbon、Grafana、MLflow/Evidently 的设计模式，升级当前 Streamlit 为中文优先的工业 uplift workbench：状态标签、审计表、证据卡、视觉回归截图、Agent prompt 分组和 Story before/after 讲法。
- 坚持不为了“炫技”直接重写 React：当前阶段优先打通 causal workflow、模型 registry、评估指标、policy simulator、Evidence gate 和面试演示；同时沉淀未来 FastAPI + React 的页面边界、组件 tokens、artifact contract 和 ECharts/React Flow 迁移路线。
- 将 Evidence 页面从文件列表升级为审计台，直接展示 UI screenshot gallery、visual regression gate、source evidence、smoke artifact 和 evidence pack 下载，让简历 claim 可被截图和 artifact 证明。

## 前端升级 30 秒讲法

这次我不是单纯美化 Streamlit，而是先审计成熟的数据产品和 Agent workbench，然后把最适合 DeepUplift 的设计模式吸收到现有平台。当前版本保留 Streamlit，是因为核心价值在因果建模、指标评估、policy decision 和证据闭环；React/FastAPI 会作为下一阶段产品化路线。面试中我会讲：UI 的作用不是装饰，而是把 uplift workflow 变成可读、可追责、可复现实验的工业工作台。

## 开源训练平台 UI/UX 重构补充

- 系统审计 MLflow、W&B、ClearML、Kubeflow Pipelines、Polyaxon、Metaflow、ZenML、DVC、Determined AI、Ray Train/Dashboard、Flyte、Dagster、BentoML/Yatai、Evidently、Aim、Neptune 等开源训练/实验追踪/MLOps 平台，并记录 stars、活跃度、页面结构、训练能力、UI pattern、配色风格、迁移成本和面试讲法。
- 将训练平台信息架构吸收到当前 Streamlit：Workflow 新增 `ML Training Workbench Overview`，Models 新增 `ML Training Platform Inspiration Matrix`，Policy 新增 `Uplift Application Expansion Matrix`，Evidence 新增训练平台 source gate 和 screenshot/visual regression evidence。
- 将 DeepUplift 从“模型 demo”升级为类 MLflow/W&B/ClearML 的 causal experiment workbench：项目状态、数据 manifest、experiment run、model registry、evaluation、policy、evidence 和 Agent 形成统一页面路径。
- 扩展 uplift 应用矩阵，覆盖优惠券/补贴 ROI、广告 iROAS、CRM message diet、marketplace 双边补贴、LLM routing/人工升级、剂量响应、客服队列、医疗/金融等场景，并明确每类场景对应 Top-K、policy value、OPE、coverage、risk warning 等指标。

## 训练平台 UI 30 秒讲法

我参考的不是某个单一模板，而是成熟训练平台的产品结构：MLflow 的 runs/metrics/artifacts，W&B 的 reports/artifact lineage，ClearML 的 task/source/env，Dagster/DVC 的 asset lineage 和 checks，Evidently/Ray 的 monitoring gates。当前仍保留 Streamlit，因为项目核心是先证明 uplift 因果决策链可信；未来再把 dataset、run、model、artifact、policy 和 evidence contracts 迁到 FastAPI + React。

## 公开数据产品 UI/UX 升级补充

- 参考火山引擎 DataLeap、ByteHouse、DataFinder、DataTester、机器学习平台、DataWind、Data Agent、智能推荐平台、VeCDP/GMP 等公开文档和官方文章，只吸收信息架构与工作台模式，不复制私有 UI 或品牌。
- 将 DataLeap 的治理/资产/血缘、DataFinder 的增长指标字典、DataTester 的实验 lifecycle、ByteHouse 的实时分析与审计信号、Data Agent 的 evidence-to-action 模式转成 DeepUplift 的 `Growth Decision Control Plane`。
- 在 Policy 页面新增 `Launch Guardrail Board`，把数据设计、overlap 支持、模型对比、Top-K ROI、OPE/holdout 和 evidence pack 做成上线前检查，避免把离线 QINI/AUUC 当作直接上线证明。
- 在 Models/Evidence/Agent 中新增公开产品灵感矩阵、source gate 和 Agent evidence card，使 UI/UX 参考可以被官方公开来源和本地截图 smoke 共同验证。

## 公开产品 UI 30 秒讲法

我参考字节/火山公开数据产品，不是为了临摹视觉，而是学习工业数据产品如何组织决策：DataLeap 管治理和资产，DataFinder 管增长指标，DataTester 管实验上线，ByteHouse 管实时分析信号，Data Agent 管分析到行动。我把这些结构落到 DeepUplift：Workflow 是增长决策控制台，Policy 是上线护栏，Evidence 是审计台，Agent 是带证据的 causal copilot。
