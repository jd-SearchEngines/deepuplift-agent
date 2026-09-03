from __future__ import annotations

from collections import Counter


FRONTEND_FRAMEWORK_AUDIT: list[dict[str, str | int]] = [
    {
        "framework": "Streamlit",
        "category": "Python data app",
        "repo": "streamlit/streamlit",
        "github_url": "https://github.com/streamlit/streamlit",
        "official_url": "https://docs.streamlit.io/develop/concepts/configuration/theming",
        "stars": 44641,
        "latest_update": "2026-05-19",
        "activity": "very active",
        "deepuplift_page": "全站 MVP / Evidence / Agent",
        "useful_capability": "快速把 Python trainer、evaluator、artifact 和中文解释接成可演示工作台。",
        "ui_pattern": "深色主题、metric cards、tabbed workflow、dataframe column config、download artifacts。",
        "streamlit_direct": "ready",
        "future_react_fit": "API contract 可以保留，UI 未来迁到 React。",
        "dependency_risk": "低；当前项目已稳定运行。",
        "migration_cost": "低到中；需要把长任务拆到 FastAPI/worker。",
        "status": "ready",
        "interview_line": "当前阶段选 Streamlit 是为了把算法、证据和面试演示闭环做快做稳，而不是把时间耗在前后端重写上。",
    },
    {
        "framework": "Plotly Dash",
        "category": "Python dashboard",
        "repo": "plotly/dash",
        "github_url": "https://github.com/plotly/dash",
        "official_url": "https://dash.plotly.com/layout",
        "stars": 24200,
        "latest_update": "2026-05-19",
        "activity": "very active",
        "deepuplift_page": "Evaluation / Policy",
        "useful_capability": "layout tree + callbacks 的思路适合未来把阈值、过滤器和图表联动拆清楚。",
        "ui_pattern": "Graph + controls + callback state；页面以指标问题组织，而不是堆图。",
        "streamlit_direct": "inspired",
        "future_react_fit": "future-react",
        "dependency_risk": "中；重写成本高于当前收益。",
        "migration_cost": "中",
        "status": "inspired",
        "interview_line": "借鉴 Dash 的交互模型，但当前不迁移，因为 DeepUplift 关键风险是因果证据闭环，不是 callback 框架选择。",
    },
    {
        "framework": "Panel / HoloViz",
        "category": "Python data app",
        "repo": "holoviz/panel",
        "github_url": "https://github.com/holoviz/panel",
        "official_url": "https://panel.holoviz.org/",
        "stars": 5676,
        "latest_update": "2026-05-19",
        "activity": "very active",
        "deepuplift_page": "Scenario / Evaluation",
        "useful_capability": "模板化数据探索、参数控件和可视化对象组合。",
        "ui_pattern": "Template + sidebar controls + linked analytic panes。",
        "streamlit_direct": "inspired",
        "future_react_fit": "backlog",
        "dependency_risk": "中；与当前 Streamlit 栈重叠。",
        "migration_cost": "中",
        "status": "inspired",
        "interview_line": "Panel 的 template 思路被吸收到工作台 page hierarchy，但不引入第二套 Python UI runtime。",
    },
    {
        "framework": "Gradio",
        "category": "ML demo UI",
        "repo": "gradio-app/gradio",
        "github_url": "https://github.com/gradio-app/gradio",
        "official_url": "https://www.gradio.app/docs/gradio/themes",
        "stars": 42622,
        "latest_update": "2026-05-18",
        "activity": "very active",
        "deepuplift_page": "Agent",
        "useful_capability": "机器学习 demo 的 Blocks、Chatbot、themes 和快速分享体验。",
        "ui_pattern": "prompt gallery、chat transcript、clear input/output boundaries。",
        "streamlit_direct": "inspired",
        "future_react_fit": "backlog",
        "dependency_risk": "低到中；与 Streamlit 功能重叠。",
        "migration_cost": "中",
        "status": "inspired",
        "interview_line": "Agent 页吸收 Gradio demo 的 prompt 分组和聊天可读性，但保留 Streamlit 统一证据台。",
    },
    {
        "framework": "Plotly.py",
        "category": "Visualization",
        "repo": "plotly/plotly.py",
        "github_url": "https://github.com/plotly/plotly.py",
        "official_url": "https://plotly.com/python/",
        "stars": 18540,
        "latest_update": "2026-05-18",
        "activity": "very active",
        "deepuplift_page": "Evaluation / Policy",
        "useful_capability": "交互曲线、hover、multi-axis 和导出，适合 Top-K / ROI / OPE 曲线升级。",
        "ui_pattern": "decision chart first, table second；曲线必须带业务解释和阈值建议。",
        "streamlit_direct": "future-ready",
        "future_react_fit": "ready",
        "dependency_risk": "低；当前未新增依赖，未来可选加入。",
        "migration_cost": "低",
        "status": "future-react",
        "interview_line": "现在先用 Streamlit 原生图降低依赖；未来 React/FastAPI 可用 Plotly/ECharts 承接复杂交互。",
    },
    {
        "framework": "Apache ECharts",
        "category": "Visualization",
        "repo": "apache/echarts",
        "github_url": "https://github.com/apache/echarts",
        "official_url": "https://echarts.apache.org/en/",
        "stars": 66383,
        "latest_update": "2026-05-15",
        "activity": "very active",
        "deepuplift_page": "Policy / Scenario / Future React",
        "useful_capability": "工业 dashboard 常用的主题、图例、数据缩放、Sankey/DAG/heatmap。",
        "ui_pattern": "业务阈值曲线、Top-K 漏斗、模型 readiness 热力图、workflow DAG。",
        "streamlit_direct": "lightweight-html-possible",
        "future_react_fit": "ready",
        "dependency_risk": "中；Streamlit 内嵌 JS 需要隔离和测试。",
        "migration_cost": "中",
        "status": "future-react",
        "interview_line": "ECharts 是未来 React 版的主可视化候选；当前只吸收其 dashboard pattern，不把 JS 运行时塞进 MVP。",
    },
    {
        "framework": "Altair / Vega-Lite",
        "category": "Visualization grammar",
        "repo": "altair-viz/altair",
        "github_url": "https://github.com/vega/altair",
        "official_url": "https://altair-viz.github.io/",
        "stars": 10379,
        "latest_update": "2026-05-18",
        "activity": "very active",
        "deepuplift_page": "Evaluation",
        "useful_capability": "声明式 encoding 适合指标公式卡、分布对比、calibration 和 segment analysis。",
        "ui_pattern": "data -> encoding -> mark 的解释路径，帮助面试讲清图表为何这样画。",
        "streamlit_direct": "ready-if-needed",
        "future_react_fit": "inspired",
        "dependency_risk": "低",
        "migration_cost": "低",
        "status": "inspired",
        "interview_line": "借鉴 Vega-Lite 的声明式思路：每张图都要说明数据字段、编码和决策含义。",
    },
    {
        "framework": "Metabase",
        "category": "Open-source BI",
        "repo": "metabase/metabase",
        "github_url": "https://github.com/metabase/metabase",
        "official_url": "https://www.metabase.com/dashboards/",
        "stars": 47367,
        "latest_update": "2026-05-19",
        "activity": "very active",
        "deepuplift_page": "Evidence / Story",
        "useful_capability": "业务问题优先、dashboard usage、自然语言到指标探索。",
        "ui_pattern": "question-first cards, drilldown, saved questions, data lineage hints。",
        "streamlit_direct": "inspired",
        "future_react_fit": "backlog",
        "dependency_risk": "高；不是嵌入组件库。",
        "migration_cost": "高",
        "status": "inspired",
        "interview_line": "Metabase 启发 Evidence/Story 的 question-first 表达：不是文件列表，而是回答简历 claim 是否可追溯。",
    },
    {
        "framework": "Apache Superset",
        "category": "Open-source BI",
        "repo": "apache/superset",
        "github_url": "https://github.com/apache/superset",
        "official_url": "https://superset.apache.org/",
        "stars": 72906,
        "latest_update": "2026-05-19",
        "activity": "very active",
        "deepuplift_page": "Models / Evaluation / Evidence",
        "useful_capability": "图表库、dashboard filters、数据探索和多源分析平台形态。",
        "ui_pattern": "filter bar、chart catalog、dashboard publishing、semantic layer hint。",
        "streamlit_direct": "inspired",
        "future_react_fit": "backlog",
        "dependency_risk": "高；作为产品参考，不直接依赖。",
        "migration_cost": "高",
        "status": "inspired",
        "interview_line": "Superset 提供工业 BI 结构参考，但 DeepUplift 的核心是因果训练和 evidence gate，不能变成通用 BI。",
    },
    {
        "framework": "Grafana",
        "category": "Observability dashboard",
        "repo": "grafana/grafana",
        "github_url": "https://github.com/grafana/grafana",
        "official_url": "https://grafana.com/docs/grafana/latest/visualizations/dashboards/",
        "stars": 73863,
        "latest_update": "2026-05-18",
        "activity": "very active",
        "deepuplift_page": "Evidence",
        "useful_capability": "panel、row、变量、annotations、JSON/provisioning 的观测性治理。",
        "ui_pattern": "evidence panels, health status, regression rows, screenshot smoke as observability。",
        "streamlit_direct": "ready-pattern",
        "future_react_fit": "ready",
        "dependency_risk": "低；只吸收模式。",
        "migration_cost": "低",
        "status": "ready",
        "interview_line": "Evidence 页按 Grafana 的 panel/health 思路做审计台：看系统状态，而不是翻文件夹。",
    },
    {
        "framework": "shadcn/ui",
        "category": "React design system",
        "repo": "shadcn-ui/ui",
        "github_url": "https://github.com/shadcn-ui/ui",
        "official_url": "https://ui.shadcn.com/docs/components",
        "stars": 114658,
        "latest_update": "2026-05-18",
        "activity": "very active",
        "deepuplift_page": "全站视觉语言 / Future React",
        "useful_capability": "open-code 组件、tokens、Card/Badge/Table/Tabs/Command/Data Table。",
        "ui_pattern": "compact cards, badges, command-like prompt groups, neutral borders, accessible states。",
        "streamlit_direct": "ready-pattern",
        "future_react_fit": "ready",
        "dependency_risk": "低；当前只吸收 CSS/token/pattern。",
        "migration_cost": "低",
        "status": "ready",
        "interview_line": "shadcn/ui 的价值不是照搬组件，而是把 tokens、badge、table 和 card pattern 转成 Streamlit 可用的工作台语言。",
    },
    {
        "framework": "shadcn-admin",
        "category": "React admin template",
        "repo": "satnaing/shadcn-admin",
        "github_url": "https://github.com/satnaing/shadcn-admin",
        "official_url": "https://github.com/satnaing/shadcn-admin",
        "stars": 12074,
        "latest_update": "2026-04-21",
        "activity": "active",
        "deepuplift_page": "Models / Evidence / Agent",
        "useful_capability": "sidebar、global search、10+ dashboard pages、dark mode、responsive admin shell。",
        "ui_pattern": "页面层级先行，卡片和表格承载密集信息，导航用业务任务命名。",
        "streamlit_direct": "ready-pattern",
        "future_react_fit": "ready",
        "dependency_risk": "低；只吸收布局。",
        "migration_cost": "低",
        "status": "ready",
        "interview_line": "它启发 DeepUplift 把模型目录做成 admin workbench，而不是 notebook 表格。",
    },
    {
        "framework": "Tremor",
        "category": "React dashboard components",
        "repo": "tremorlabs/tremor",
        "github_url": "https://github.com/tremorlabs/tremor",
        "official_url": "https://tremor.so/",
        "stars": 3424,
        "latest_update": "2025-10-10",
        "activity": "stable / slower",
        "deepuplift_page": "Policy / Evaluation",
        "useful_capability": "KPI card、badge、chart、table 的数据产品默认语言。",
        "ui_pattern": "Metric cards with explanation, compact status badges, charts beside tables。",
        "streamlit_direct": "ready-pattern",
        "future_react_fit": "optional",
        "dependency_risk": "中；生态方向需要继续观察。",
        "migration_cost": "低",
        "status": "inspired",
        "interview_line": "Tremor 启发指标卡和状态标签；当前用自有 CSS 复刻关键模式，避免引入 React 依赖。",
    },
    {
        "framework": "Ant Design Pro",
        "category": "Enterprise admin",
        "repo": "ant-design/ant-design-pro",
        "github_url": "https://github.com/ant-design/ant-design-pro",
        "official_url": "https://pro.ant.design/",
        "stars": 38251,
        "latest_update": "2026-05-19",
        "activity": "very active",
        "deepuplift_page": "Models / History / Evidence",
        "useful_capability": "企业后台模板、表单、列表、profile、result、i18n、AI Assistant page。",
        "ui_pattern": "advanced table, result state, profile-like model cards, enterprise workflow copy。",
        "streamlit_direct": "inspired",
        "future_react_fit": "ready",
        "dependency_risk": "中；React 版选型时会影响整体设计系统。",
        "migration_cost": "中",
        "status": "future-react",
        "interview_line": "Ant Design Pro 适合未来企业版，但 Streamlit 当前更适合算法闭环快速迭代。",
    },
    {
        "framework": "Material UI",
        "category": "React design system",
        "repo": "mui/material-ui",
        "github_url": "https://github.com/mui/material-ui",
        "official_url": "https://mui.com/material-ui/",
        "stars": 98362,
        "latest_update": "2026-05-19",
        "activity": "very active",
        "deepuplift_page": "Future React",
        "useful_capability": "成熟 React 组件、DataGrid、theme、accessibility。",
        "ui_pattern": "dense data grid, controlled forms, responsive layout。",
        "streamlit_direct": "backlog",
        "future_react_fit": "optional",
        "dependency_risk": "中；商业组件和样式体系取舍。",
        "migration_cost": "中",
        "status": "future-react",
        "interview_line": "MUI 是未来 React 候选之一，但当前不需要把 Streamlit MVP 变成前端组件库项目。",
    },
    {
        "framework": "Carbon Design System",
        "category": "Enterprise design system",
        "repo": "carbon-design-system/carbon",
        "github_url": "https://github.com/carbon-design-system/carbon",
        "official_url": "https://carbondesignsystem.com/",
        "stars": 9127,
        "latest_update": "2026-05-18",
        "activity": "very active",
        "deepuplift_page": "Evidence / Models",
        "useful_capability": "企业级数据表、tag、tile、structured content 和审计体验。",
        "ui_pattern": "status tag + dense table + evidence tile + serious enterprise tone。",
        "streamlit_direct": "ready-pattern",
        "future_react_fit": "optional",
        "dependency_risk": "低；只吸收治理模式。",
        "migration_cost": "低",
        "status": "inspired",
        "interview_line": "Carbon 启发 DeepUplift 的审计台气质：冷静、密集、可追责。",
    },
    {
        "framework": "Radix UI",
        "category": "Accessible primitives",
        "repo": "radix-ui/primitives",
        "github_url": "https://github.com/radix-ui/primitives",
        "official_url": "https://www.radix-ui.com/primitives/docs/overview/getting-started",
        "stars": 18912,
        "latest_update": "2026-02-13",
        "activity": "active",
        "deepuplift_page": "Future React",
        "useful_capability": "Popover、Dialog、Tooltip、Tabs 等无样式可访问 primitives。",
        "ui_pattern": "accessibility first；复杂控件先保证行为，再做视觉。",
        "streamlit_direct": "backlog",
        "future_react_fit": "ready",
        "dependency_risk": "低",
        "migration_cost": "低",
        "status": "future-react",
        "interview_line": "未来 React 版会优先选 Radix/shadcn 这类可控 primitives，避免黑盒 UI 影响因果工作流。",
    },
    {
        "framework": "Dify",
        "category": "Agent workbench",
        "repo": "langgenius/dify",
        "github_url": "https://github.com/langgenius/dify",
        "official_url": "https://docs.dify.ai/",
        "stars": 141875,
        "latest_update": "2026-05-19",
        "activity": "very active",
        "deepuplift_page": "Agent / Evidence",
        "useful_capability": "agentic workflow、RAG pipeline、model management、workflow canvas。",
        "ui_pattern": "workflow as product surface, node evidence, execution logs, app/model separation。",
        "streamlit_direct": "inspired",
        "future_react_fit": "ready",
        "dependency_risk": "高；不是要嵌入 Dify。",
        "migration_cost": "高",
        "status": "inspired",
        "interview_line": "Dify 启发 Agent workflow 可视化，但 DeepUplift 的 agent 是 causal workflow orchestrator，不是通用 LLM app builder。",
    },
    {
        "framework": "Langflow",
        "category": "Agent workflow UI",
        "repo": "langflow-ai/langflow",
        "github_url": "https://github.com/langflow-ai/langflow",
        "official_url": "https://docs.langflow.org/",
        "stars": 148498,
        "latest_update": "2026-05-19",
        "activity": "very active",
        "deepuplift_page": "Workflow / Agent",
        "useful_capability": "visual builder、playground、deploy as API/MCP、observability integrations。",
        "ui_pattern": "node graph for workflow, side panel for config, playground for immediate testing。",
        "streamlit_direct": "inspired",
        "future_react_fit": "ready",
        "dependency_risk": "高；功能域不同。",
        "migration_cost": "高",
        "status": "future-react",
        "interview_line": "Langflow 的视觉 workflow 适合未来解释 Agent 状态机；当前先用流程卡和 evidence path 表达。",
    },
    {
        "framework": "Flowise",
        "category": "Agent workflow UI",
        "repo": "FlowiseAI/Flowise",
        "github_url": "https://github.com/FlowiseAI/Flowise",
        "official_url": "https://docs.flowiseai.com/",
        "stars": 52930,
        "latest_update": "2026-05-18",
        "activity": "very active",
        "deepuplift_page": "Agent / Future workflow canvas",
        "useful_capability": "visual agent builder and chatflow orchestration。",
        "ui_pattern": "node canvas, tool config, chatflow test, deployable endpoint。",
        "streamlit_direct": "inspired",
        "future_react_fit": "future-react",
        "dependency_risk": "中到高；安全和插件执行边界必须审计。",
        "migration_cost": "高",
        "status": "backlog",
        "interview_line": "Flowise 说明 workflow canvas 的价值，也提醒 agent 平台必须做执行权限和证据边界。",
    },
    {
        "framework": "Open WebUI",
        "category": "AI chat UI",
        "repo": "open-webui/open-webui",
        "github_url": "https://github.com/open-webui/open-webui",
        "official_url": "https://docs.openwebui.com/",
        "stars": 137711,
        "latest_update": "2026-05-10",
        "activity": "very active",
        "deepuplift_page": "Agent",
        "useful_capability": "self-hosted chat interface、model/provider management、tools/RAG。",
        "ui_pattern": "readable chat turns, markdown, tool/evidence panels, prompt actions。",
        "streamlit_direct": "ready-pattern",
        "future_react_fit": "optional",
        "dependency_risk": "中；许可证和安全边界需要审计。",
        "migration_cost": "中",
        "status": "inspired",
        "interview_line": "Agent 页吸收 chat readability 和 evidence card，而不是复制通用聊天产品。",
    },
    {
        "framework": "LangGraph Studio / LangSmith Studio",
        "category": "Agent debugging UI",
        "repo": "LangChain official docs",
        "github_url": "https://docs.langchain.com/langgraph-platform/use-studio",
        "official_url": "https://docs.langchain.com/oss/python/langchain/studio",
        "stars": 0,
        "latest_update": "2026 docs active",
        "activity": "active docs",
        "deepuplift_page": "Workflow / Agent / Evidence",
        "useful_capability": "graph mode、chat mode、threads、fork/re-run、state inspection。",
        "ui_pattern": "run timeline, node state, thread history, graph/debug split。",
        "streamlit_direct": "inspired",
        "future_react_fit": "future-react",
        "dependency_risk": "中；官方桌面 repo deprecated，采用 docs pattern。",
        "migration_cost": "中",
        "status": "future-react",
        "interview_line": "LangGraph Studio 启发 Agent 的 Diagnose -> Recommend -> Compare -> Evidence 流程追踪，但当前用可复现 artifacts 代替复杂图编辑器。",
    },
    {
        "framework": "MLflow",
        "category": "Experiment tracking",
        "repo": "mlflow/mlflow",
        "github_url": "https://github.com/mlflow/mlflow",
        "official_url": "https://mlflow.org/docs/latest/index.html",
        "stars": 26001,
        "latest_update": "2026-05-19",
        "activity": "very active",
        "deepuplift_page": "History / Evidence",
        "useful_capability": "runs、params、metrics、artifacts、traces 和 model registry。",
        "ui_pattern": "run comparison table, params/metrics/artifacts split, experiment lineage。",
        "streamlit_direct": "ready-pattern",
        "future_react_fit": "ready",
        "dependency_risk": "中；当前已有本地 artifacts，不急于引入服务。",
        "migration_cost": "中",
        "status": "inspired",
        "interview_line": "History/Evidence 吸收 MLflow 的 run/metric/artifact 结构，先保持本地轻量实现。",
    },
    {
        "framework": "Evidently",
        "category": "ML/LLM observability",
        "repo": "evidentlyai/evidently",
        "github_url": "https://github.com/evidentlyai/evidently",
        "official_url": "https://www.evidentlyai.com/evidently-oss",
        "stars": 7504,
        "latest_update": "2026-05-02",
        "activity": "active",
        "deepuplift_page": "Evidence / Diagnostics",
        "useful_capability": "Reports, Test Suites, monitoring dashboards, 100+ metrics。",
        "ui_pattern": "quality gate, drift/test status, report-to-dashboard path。",
        "streamlit_direct": "ready-pattern",
        "future_react_fit": "optional",
        "dependency_risk": "中；metrics 体系不同，需要 adapter。",
        "migration_cost": "中",
        "status": "inspired",
        "interview_line": "Evidently 启发把 drift/test suite 放进 Evidence，而 DeepUplift 重点是 uplift-specific metrics 和 policy gates。",
    },
    {
        "framework": "W&B",
        "category": "Experiment tracking",
        "repo": "wandb/wandb",
        "github_url": "https://github.com/wandb/wandb",
        "official_url": "https://docs.wandb.ai/",
        "stars": 11074,
        "latest_update": "2026-05-18",
        "activity": "very active",
        "deepuplift_page": "History / Evidence / Future production",
        "useful_capability": "experiment comparison, artifacts, reports, model/version governance。",
        "ui_pattern": "sweep/run comparison, artifact lineage, report for stakeholders。",
        "streamlit_direct": "inspired",
        "future_react_fit": "optional",
        "dependency_risk": "中；外部服务和账号依赖。",
        "migration_cost": "中",
        "status": "backlog",
        "interview_line": "W&B 是未来团队协作参考；当前 demo 保持本地可复现，不依赖第三方账号。",
    },
    {
        "framework": "React Flow / XYFlow",
        "category": "Graph / DAG UI",
        "repo": "xyflow/xyflow",
        "github_url": "https://github.com/xyflow/xyflow",
        "official_url": "https://reactflow.dev/",
        "stars": 36654,
        "latest_update": "2026-05-12",
        "activity": "active",
        "deepuplift_page": "Future Agent workflow canvas",
        "useful_capability": "node-based workflow editor for agent/model/policy DAG。",
        "ui_pattern": "nodes, edges, side-panel config, execution state badges。",
        "streamlit_direct": "backlog",
        "future_react_fit": "ready",
        "dependency_risk": "低 for React；不适合当前 Streamlit 直接引入。",
        "migration_cost": "中",
        "status": "future-react",
        "interview_line": "未来 React 版可用 React Flow 展示 causal workflow DAG；当前先用流程卡保证演示稳定。",
    },
]


UI_INSPIRATION_MATRIX: list[dict[str, str]] = [
    {
        "pattern": "工业 workflow 总览",
        "borrowed_from": "Grafana panels + shadcn-admin page shell + LangGraph Studio graph/debug split",
        "absorbed_now": "Workflow/Story 增加 design -> diagnostics -> model -> evaluation -> policy -> evidence 的可视流程和 gate 表。",
        "deepuplift_value": "面试时先讲业务和因果假设，再讲模型；降低“模型列表”感。",
        "future_react": "React Flow DAG + right-side evidence inspector。",
        "status": "ready",
    },
    {
        "pattern": "Metric card + status badge",
        "borrowed_from": "Tremor, Carbon, shadcn/ui",
        "absorbed_now": "统一 du-stat-card、du-chip、du-status-row、readiness/evidence badges。",
        "deepuplift_value": "把 ready/guarded/optional/backlog、QINI/AUUC/OPE/ROI 变成可扫读状态。",
        "future_react": "shadcn Badge + Card + DataTable。",
        "status": "ready",
    },
    {
        "pattern": "Evidence 审计台",
        "borrowed_from": "Grafana observability + MLflow artifacts + Evidently test suites",
        "absorbed_now": "Evidence 展示 regression gate、source gate、visual screenshot evidence 和下载证据包。",
        "deepuplift_value": "每个简历 claim 能追到截图、artifact、smoke 和文档。",
        "future_react": "Run timeline + artifact drawer + visual regression diff。",
        "status": "ready",
    },
    {
        "pattern": "Question-first BI",
        "borrowed_from": "Metabase and Superset dashboard filters",
        "absorbed_now": "Story/Evidence 用“这个 claim 怎么证明”组织，而不是文件列表。",
        "deepuplift_value": "让非算法面试官也能理解为什么这个平台可信。",
        "future_react": "semantic metric catalog + saved dashboard questions。",
        "status": "inspired",
    },
    {
        "pattern": "Agent prompt gallery",
        "borrowed_from": "Open WebUI / Gradio / Dify",
        "absorbed_now": "Agent quick prompts 按模型选择、评估策略、工业场景、LLM/OPE、前端设计分组。",
        "deepuplift_value": "把聊天框变成面试工作台入口，避免几十个按钮无结构堆叠。",
        "future_react": "Command palette + prompt collections + transcript evidence drawer。",
        "status": "ready",
    },
    {
        "pattern": "Framework audit matrix",
        "borrowed_from": "Ant Design Pro advanced table + Carbon governance table",
        "absorbed_now": "Models 新增 Frontend Framework Audit / UI Inspiration Matrix，字段覆盖 stars/activity/risk/migration/status。",
        "deepuplift_value": "能回答“参考了哪些前端框架，为什么不重写 React”。",
        "future_react": "typed audit table with filters and detail sheets。",
        "status": "ready",
    },
    {
        "pattern": "Decision-first visualization",
        "borrowed_from": "ECharts, Plotly, Altair/Vega-Lite",
        "absorbed_now": "Policy/Evaluation 的 Top-K、ROI、OPE、readiness chart 先解释阈值和业务含义，再给表。",
        "deepuplift_value": "图表服务模型评估和业务决策，而不是装饰性可视化。",
        "future_react": "ECharts/Plotly interactive Top-K and OPE panels。",
        "status": "inspired",
    },
    {
        "pattern": "Run / artifact tracking",
        "borrowed_from": "MLflow and W&B",
        "absorbed_now": "History/Evidence 保留 run_id、metrics、readiness、predictions、run note、evidence ZIP。",
        "deepuplift_value": "训练不是一次性按钮，而是可追溯实验。",
        "future_react": "experiment detail route + artifact preview。",
        "status": "ready",
    },
]


VISUAL_UPGRADE_ROWS: list[dict[str, str]] = [
    {
        "area": "信息架构",
        "before": "页面很多，但用户容易从模型名开始看。",
        "after": "以工业 uplift workflow 组织：业务设计、诊断、模型、指标、Policy、Evidence、Agent。",
        "proof": "Workflow / Story / Models / Evidence",
    },
    {
        "area": "Models",
        "before": "模型目录、开源框架和依赖状态散在不同表格。",
        "after": "模型目录 + 前端框架审计 + ready/guarded/optional/backlog + future React 路线放在同一工作台。",
        "proof": "Models -> Frontend Framework Audit / UI Inspiration Matrix",
    },
    {
        "area": "Evidence",
        "before": "artifact 列表可用，但视觉证据和回归含义不够像审计台。",
        "after": "加入 screenshot evidence、visual regression gate、source gate、artifact cards。",
        "proof": "Evidence -> UI Screenshot Gallery / Visual Regression Gate",
    },
    {
        "area": "Agent",
        "before": "quick prompts 数量多，像未分组按钮墙。",
        "after": "按模型、评估、工业场景、LLM/OPE、前端设计分组，回答能落到 evidence card。",
        "proof": "Agent -> Quick Prompt Groups",
    },
    {
        "area": "面试讲法",
        "before": "算法和 UI/UX 可以分开讲，但关系不够显性。",
        "after": "明确讲 UI/UX 如何服务 causal decision：减少堆叠、突出阈值、证据和风险。",
        "proof": "docs/FRONTEND_FRAMEWORK_RESEARCH.md; Agent frontend answers",
    },
]


PROMPT_GROUPS: list[dict[str, list[str] | str]] = [
    {
        "group": "模型选择",
        "intent": "从数据设计和依赖状态解释推荐模型。",
        "prompts": ["推荐哪个模型？", "解释当前模型卡？", "对比结果怎么看？", "当前平台用了哪些开源框架，哪些 ready/guarded？"],
    },
    {
        "group": "评估与 Policy",
        "intent": "把 QINI/AUUC/Top-K/OPE/ROI 讲成上线决策。",
        "prompts": ["uplift 模型到底怎么评估？公式怎么讲？", "为什么 QINI/AUUC 不等于增量利润？", "OPE / IPS / DR 怎么评估新策略？", "投放阈值怎么解释？"],
    },
    {
        "group": "工业场景",
        "intent": "发券、广告、marketplace、CRM、履历场景面试问答。",
        "prompts": ["发券为什么不能只看转化率？", "广告投放里 uplift 和 pCTR/pCVR 有什么区别？", "Marketplace 干扰和 spillover 怎么处理？", "从0-1优化 uplift 项目怎么讲？"],
    },
    {
        "group": "LLM / Agent",
        "intent": "LLM routing、agent workflow、workflow orchestration。",
        "prompts": ["LLM routing 为什么是 uplift 问题？", "什么时候从 uplift 升级到 bandit？", "Agent 怎么从聊天升级成 causal workflow orchestrator？", "5分钟怎么演示？"],
    },
    {
        "group": "前端设计",
        "intent": "回答本轮 GitHub 前端研究和 Streamlit vs React 取舍。",
        "prompts": ["参考了哪些 GitHub 前端框架？", "为什么当前阶段还是 Streamlit，而不是直接 React？", "哪些设计模式被吸收到当前页面？", "哪些适合未来 FastAPI + React 版本？"],
    },
]


def frontend_framework_audit_rows() -> list[dict[str, str | int]]:
    return [dict(row) for row in FRONTEND_FRAMEWORK_AUDIT]


def ui_inspiration_matrix_rows() -> list[dict[str, str]]:
    return [dict(row) for row in UI_INSPIRATION_MATRIX]


def visual_upgrade_rows() -> list[dict[str, str]]:
    return [dict(row) for row in VISUAL_UPGRADE_ROWS]


def agent_prompt_groups() -> list[dict[str, list[str] | str]]:
    return [dict(row) for row in PROMPT_GROUPS]


def frontend_research_counts() -> dict[str, int]:
    rows = frontend_framework_audit_rows()
    status_counts = Counter(str(row["status"]) for row in rows)
    return {
        "total": len(rows),
        "ready": status_counts.get("ready", 0),
        "inspired": status_counts.get("inspired", 0),
        "future_react": status_counts.get("future-react", 0),
        "backlog": status_counts.get("backlog", 0),
    }


def visual_regression_gate_rows() -> list[dict[str, str]]:
    return [
        {
            "gate": "py_compile",
            "why": "先验证 Streamlit 入口和新增 research module 没有语法错误。",
            "command": "python -m py_compile app.py deepuplift/core/frontend_research.py scripts/smoke_ui_screenshots.py",
            "status": "required",
        },
        {
            "gate": "UI screenshot smoke",
            "why": "验证 Workflow/Story/Models/Evidence/Agent/Policy 页面仍可打开且截图非空。",
            "command": "python scripts/smoke_ui_screenshots.py --url http://localhost:8501 --tabs Workflow Scenario Models Evidence Agent Story Policy --min-bytes 20000",
            "status": "required",
        },
        {
            "gate": "visual evidence visible",
            "why": "Evidence 页必须直接展示最新 screenshot pack，而不是只给文件名。",
            "command": "Evidence -> UI Screenshot Gallery",
            "status": "ready",
        },
        {
            "gate": "Agent answer smoke",
            "why": "Agent 必须能解释参考框架、Streamlit 取舍、吸收模式和未来 React 路线。",
            "command": "Ask Agent: 参考了哪些 GitHub 前端框架？",
            "status": "ready",
        },
    ]


def frontend_agent_reply(prompt: str) -> str | None:
    text = prompt.strip().lower()
    if not text:
        return None
    frontend_terms = [
        "frontend",
        "前端",
        "ui/ux",
        "streamlit",
        "react",
        "fastapi",
        "shadcn",
        "tremor",
        "echarts",
        "dify",
        "langflow",
        "open webui",
        "前端框架",
        "ui 框架",
        "dashboard 框架",
        "视觉",
        "可视化体验",
        "可视化",
        "设计模式",
        "信息架构",
        "workbench",
    ]
    if not any(term in text for term in frontend_terms):
        return None

    counts = frontend_research_counts()
    if any(key in text for key in ["哪些", "参考", "github", "框架", "调研", "审计"]):
        ready = [row["framework"] for row in FRONTEND_FRAMEWORK_AUDIT if row["status"] == "ready"]
        inspired = [row["framework"] for row in FRONTEND_FRAMEWORK_AUDIT if row["status"] == "inspired"]
        future = [row["framework"] for row in FRONTEND_FRAMEWORK_AUDIT if row["status"] == "future-react"]
        return (
            "### 前端框架参考清单\n\n"
            f"我审计了 `{counts['total']}` 个官方 GitHub repo / 官方文档来源。当前直接吸收的是："
            f"`{', '.join(ready)}`。\n\n"
            f"作为 UI/UX inspiration 的是：`{', '.join(inspired[:10])}`。\n\n"
            f"未来 FastAPI + React 候选是：`{', '.join(future[:10])}`。\n\n"
            "落地证据在 `Models -> Frontend Framework Audit / UI Inspiration Matrix`、"
            "`Evidence -> UI Screenshot Gallery / Visual Regression Gate`、"
            "`docs/FRONTEND_FRAMEWORK_RESEARCH.md`。"
        )

    if any(key in text for key in ["为什么", "不是直接", "不直接", "streamlit", "react"]):
        return (
            "### 为什么当前阶段仍是 Streamlit\n\n"
            "当前 DeepUplift Agent 的核心风险是算法闭环和证据可信度：数据诊断、103 个模型 registry、训练/对比、"
            "QINI/AUUC/Top-K/policy value/OPE、artifact、screenshot smoke 和 Agent 面试回答要先打通。"
            "Streamlit 让 Python trainer/evaluator/artifacts 可以在同一进程快速迭代。\n\n"
            "如果现在直接重写 React，会把时间转移到路由、状态管理、API、任务队列和图表组件上，"
            "但不会立刻提升 causal decision 的可信度。所以当前策略是：Streamlit 先做专业工作台，"
            "同时把页面信息架构、组件 tokens、Evidence contract 和 API 边界设计成未来 FastAPI + React 可迁移。"
        )

    if any(key in text for key in ["吸收", "设计模式", "pattern", "页面", "工作台"]):
        patterns = "\n".join(
            f"- **{row['pattern']}**：{row['absorbed_now']} 价值：{row['deepuplift_value']}"
            for row in UI_INSPIRATION_MATRIX
        )
        return (
            "### 已吸收到当前 Streamlit 的 UI/UX pattern\n\n"
            f"{patterns}\n\n"
            "一句话讲法：我不是换皮，而是把成熟 dashboard / agent workbench 的信息架构压缩成 Streamlit 可运行组件，"
            "让 Story、Models、Evaluation、Policy、Evidence、Agent 都围绕工业 uplift 决策闭环组织。"
        )

    if any(key in text for key in ["fastapi", "future", "未来", "迁移"]):
        return (
            "### 未来 FastAPI + React 路线\n\n"
            "后端保留现有 `deepuplift/core` contracts：datasets、diagnostics、runs、metrics、predictions、artifacts。"
            "FastAPI 暴露异步训练和 artifact 查询；worker 执行 compare/OPE/smoke；React 负责可视化工作台。\n\n"
            "前端选型候选：shadcn/ui + Radix 负责基础组件，ECharts/Plotly 负责 Top-K/ROI/OPE 曲线，"
            "React Flow 负责 Agent/workflow DAG，MLflow/W&B/Evidently 的 run/artifact/quality gate 模式用于生产实验治理。"
        )

    if any(key in text for key in ["信息架构", "ia", "工业级", "workbench"]):
        return (
            "工业级 uplift workbench 的信息架构应该从业务决策倒推："
            "`业务问题 -> treatment/control 设计 -> 数据诊断/overlap -> 模型候选 -> 统一评估 -> ROI/Policy -> Evidence -> Agent explanation`。"
            "模型目录只是一层，不是入口。这样面试时能解释为什么 UI 服务因果推断：它强制用户先看假设和证据，再看排行榜。"
        )

    if any(key in text for key in ["可视化", "图表", "业务决策", "模型评估"]):
        return (
            "可视化服务决策的方式是：QINI/AUUC 看排序质量，Top-K curve 看投放桶，ROI/policy value curve 看阈值，"
            "OPE coverage/ESS 看新策略是否有历史支持，readiness matrix 看模型是否能上线。"
            "所以图表不是装饰，而是把模型输出转成预算、风险和 evidence gate。"
        )

    return (
        "这次 UI/UX 升级的核心讲法：我系统审计了 Streamlit、Dash、Panel、Gradio、shadcn/ui、Tremor、Ant Design Pro、"
        "Grafana、Metabase、Superset、Dify、Langflow、Open WebUI、MLflow、Evidently、ECharts/Plotly 等官方来源；"
        "当前先把最有价值的 dashboard / agent workbench pattern 吸收到 Streamlit，未来再按 FastAPI + React 分层迁移。"
    )
