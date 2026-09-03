from __future__ import annotations

from collections import Counter


ML_TRAINING_PLATFORM_AUDIT: list[dict[str, str]] = [
    {
        "platform": "MLflow",
        "category": "Experiment tracking / model registry",
        "repo": "mlflow/mlflow",
        "github_url": "https://github.com/mlflow/mlflow",
        "official_url": "https://mlflow.org/docs/latest/ml/tracking",
        "stars": "26k",
        "latest_update": "2026-05-19",
        "activity": "very active",
        "page_structure": "Experiments -> Runs -> Params/Metrics/Artifacts -> Model Registry",
        "core_capability": "experiment tracking, run comparison, artifacts, model registry, lineage tags",
        "ui_pattern": "run table with filters; run detail split into params/metrics/artifacts/notes; registry stages",
        "visual_style": "quiet neutral shell, dense tables, small status chips, chart-first run comparison",
        "deepuplift_absorption": "History/Compare should read like MLflow: run_id, params, QINI/AUUC/Top-K/OPE, artifacts, tags and notes.",
        "streamlit_direct": "ready-pattern",
        "future_react_fit": "ready",
        "dependency_risk": "medium if embedded as service; low as UI pattern",
        "migration_cost": "medium",
        "status": "ready",
        "interview_line": "MLflow is the strongest reference for explaining DeepUplift runs as reproducible experiments rather than one-off notebooks.",
    },
    {
        "platform": "Weights & Biases",
        "category": "Experiment tracking / artifact registry",
        "repo": "wandb/wandb",
        "github_url": "https://github.com/wandb/wandb",
        "official_url": "https://docs.wandb.ai/models/registry/model_registry/create-model-cards",
        "stars": "11.1k",
        "latest_update": "2026-05-18",
        "activity": "very active",
        "page_structure": "Projects -> Runs -> Sweeps/Reports -> Artifacts/Registry",
        "core_capability": "interactive experiment comparison, artifacts, reports, registry governance",
        "ui_pattern": "custom dashboard panels, run overlays, artifact lineage, registry aliases and model cards",
        "visual_style": "dark neutral analytics panels, colorful charts only where data needs contrast",
        "deepuplift_absorption": "Story/Evidence should turn metrics and screenshots into a shareable report-style evidence surface.",
        "streamlit_direct": "inspired",
        "future_react_fit": "optional",
        "dependency_risk": "medium: account/service dependency if integrated directly",
        "migration_cost": "medium",
        "status": "inspired",
        "interview_line": "W&B inspired the report and artifact lineage language, but the current demo stays local and account-free.",
    },
    {
        "platform": "ClearML",
        "category": "MLOps / experiment manager",
        "repo": "clearml/clearml",
        "github_url": "https://github.com/clearml/clearml",
        "official_url": "https://clear.ml/docs/latest/docs/webapp/webapp_overview",
        "stars": "6.7k",
        "latest_update": "2026-05-15",
        "activity": "active",
        "page_structure": "Projects -> Tasks -> Logs/Scalars/Artifacts -> Queues/Workers",
        "core_capability": "experiment tracking, source capture, queue execution, resource scheduling",
        "ui_pattern": "task detail page with source, environment, logs, metrics and artifacts in one place",
        "visual_style": "enterprise control-plane tone; compact sections with strong provenance labels",
        "deepuplift_absorption": "Evidence should expose source/code, environment, smoke and screenshot proof beside model results.",
        "streamlit_direct": "ready-pattern",
        "future_react_fit": "ready",
        "dependency_risk": "medium if adopting queue/server; low as UI pattern",
        "migration_cost": "medium",
        "status": "ready",
        "interview_line": "ClearML is useful to explain why DeepUplift records source, dependencies and artifacts, not only final metrics.",
    },
    {
        "platform": "Kubeflow Pipelines",
        "category": "Pipeline orchestration",
        "repo": "kubeflow/pipelines",
        "github_url": "https://github.com/kubeflow/pipelines",
        "official_url": "https://www.kubeflow.org/docs/components/pipelines/",
        "stars": "4.1k",
        "latest_update": "2026-05-17",
        "activity": "active",
        "page_structure": "Pipelines -> Experiments -> Runs -> Graph/Artifacts",
        "core_capability": "compose, deploy and manage end-to-end ML workflows on Kubernetes",
        "ui_pattern": "workflow DAG, step status, run detail, artifact handoff between pipeline nodes",
        "visual_style": "Kubernetes workbench: functional, sparse, status-driven",
        "deepuplift_absorption": "Workflow should show business design -> diagnostics -> train -> compare -> policy -> evidence as a pipeline.",
        "streamlit_direct": "inspired",
        "future_react_fit": "future-react",
        "dependency_risk": "high for current local demo; Kubernetes stack is overkill",
        "migration_cost": "high",
        "status": "future-react",
        "interview_line": "Kubeflow is the production workflow reference; Streamlit keeps the local MVP light.",
    },
    {
        "platform": "Polyaxon",
        "category": "MLOps / orchestration",
        "repo": "polyaxon/polyaxon",
        "github_url": "https://github.com/polyaxon/polyaxon",
        "official_url": "https://polyaxon.com/docs/",
        "stars": "3.7k",
        "latest_update": "2026-04-26",
        "activity": "active",
        "page_structure": "Projects -> Operations -> Runs -> Artifacts/Lineage",
        "core_capability": "experiment orchestration, jobs, services, model lifecycle and artifacts",
        "ui_pattern": "operation-centric tables with lifecycle states and reproducible specs",
        "visual_style": "cloud control-plane dashboard; restrained cards plus operational status",
        "deepuplift_absorption": "Use operation states for DeepUplift run readiness: draft, trained, compared, gated, exported.",
        "streamlit_direct": "inspired",
        "future_react_fit": "optional",
        "dependency_risk": "high as runtime; low as pattern",
        "migration_cost": "high",
        "status": "inspired",
        "interview_line": "Polyaxon helps frame DeepUplift runs as lifecycle-managed operations.",
    },
    {
        "platform": "Metaflow",
        "category": "ML workflow framework",
        "repo": "Netflix/metaflow",
        "github_url": "https://github.com/Netflix/metaflow",
        "official_url": "https://docs.metaflow.org/",
        "stars": "10.1k",
        "latest_update": "2026-05-14",
        "activity": "active",
        "page_structure": "Flow -> Steps -> Runs -> Cards/Artifacts",
        "core_capability": "Pythonic ML workflows, run metadata, data artifacts and cards",
        "ui_pattern": "flow-first narrative; each step has inputs, outputs and evidence cards",
        "visual_style": "developer-friendly cards, not a BI dashboard",
        "deepuplift_absorption": "Story should explain the causal workflow as a flow with evidence at each step.",
        "streamlit_direct": "inspired",
        "future_react_fit": "optional",
        "dependency_risk": "medium",
        "migration_cost": "medium",
        "status": "inspired",
        "interview_line": "Metaflow inspired the step-by-step narrative: every model decision has upstream data and downstream policy evidence.",
    },
    {
        "platform": "ZenML",
        "category": "MLOps pipeline platform",
        "repo": "zenml-io/zenml",
        "github_url": "https://github.com/zenml-io/zenml",
        "official_url": "https://docs.zenml.io/",
        "stars": "5.4k",
        "latest_update": "2026-05-12",
        "activity": "active",
        "page_structure": "Pipelines -> Runs -> Artifacts -> Models/Stacks",
        "core_capability": "pipeline metadata, stack components, artifact lineage, model control plane",
        "ui_pattern": "stack-aware deployment: compute/storage/orchestrator are explicit, not hidden",
        "visual_style": "clean MLOps console, clear state tags and component boundaries",
        "deepuplift_absorption": "Future production path should show trainer/evaluator/artifact store as separate system contracts.",
        "streamlit_direct": "inspired",
        "future_react_fit": "future-react",
        "dependency_risk": "medium",
        "migration_cost": "medium",
        "status": "future-react",
        "interview_line": "ZenML is useful for explaining the future FastAPI + worker + artifact store split.",
    },
    {
        "platform": "DVC",
        "category": "Data/model versioning",
        "repo": "treeverse/dvc",
        "github_url": "https://github.com/treeverse/dvc",
        "official_url": "https://dvc.org/doc",
        "stars": "15.6k",
        "latest_update": "2026-04-28",
        "activity": "active",
        "page_structure": "Data/Model versions -> Pipelines -> Experiments -> Compare",
        "core_capability": "version data/models, lightweight pipelines, local experiment tracking and comparison",
        "ui_pattern": "Git-like reproducibility: data, params, metrics and plots remain tied to code",
        "visual_style": "developer workflow tone, less glossy, high trust",
        "deepuplift_absorption": "Evidence should make dataset manifest, params, metrics and screenshots first-class artifacts.",
        "streamlit_direct": "ready-pattern",
        "future_react_fit": "optional",
        "dependency_risk": "low as pattern; medium if adding DVC remotes",
        "migration_cost": "medium",
        "status": "ready",
        "interview_line": "DVC inspired the evidence contract: every uplift result should point back to data, params and code.",
    },
    {
        "platform": "Determined AI",
        "category": "Distributed training platform",
        "repo": "determined-ai/determined",
        "github_url": "https://github.com/determined-ai/determined",
        "official_url": "https://docs.determined.ai/",
        "stars": "3.2k",
        "latest_update": "2025-03-20",
        "activity": "slower / stable",
        "page_structure": "Experiments -> Trials -> Checkpoints -> Resources",
        "core_capability": "distributed training, hyperparameter tuning, checkpoints and resource management",
        "ui_pattern": "trial table, checkpoint lineage, GPU/resource status and experiment progress",
        "visual_style": "training console: compact status and progress, not marketing visuals",
        "deepuplift_absorption": "Train page can borrow trial/checkpoint language for multiple candidate learners.",
        "streamlit_direct": "inspired",
        "future_react_fit": "backlog",
        "dependency_risk": "high for current project",
        "migration_cost": "high",
        "status": "backlog",
        "interview_line": "Determined is a reminder that training UI should expose progress and resources, but DeepUplift is not yet a distributed trainer.",
    },
    {
        "platform": "Ray Train / Ray Dashboard",
        "category": "Distributed training / observability",
        "repo": "ray-project/ray",
        "github_url": "https://github.com/ray-project/ray",
        "official_url": "https://docs.ray.io/en/latest/train/train.html",
        "stars": "42.6k",
        "latest_update": "2026-05-19",
        "activity": "very active",
        "page_structure": "Jobs -> Actors/Tasks -> Train metrics -> Cluster dashboard",
        "core_capability": "distributed training and application observability across frameworks",
        "ui_pattern": "cluster/application health cards, logs, events, metrics and resource utilization",
        "visual_style": "observability dashboard with dark-friendly charts and low-noise status colors",
        "deepuplift_absorption": "Evidence should treat smoke, screenshots and policy gates as observability signals.",
        "streamlit_direct": "inspired",
        "future_react_fit": "future-react",
        "dependency_risk": "medium to high if adding runtime",
        "migration_cost": "high",
        "status": "future-react",
        "interview_line": "Ray is a future scaling option; current Streamlit keeps local training deterministic.",
    },
    {
        "platform": "Flyte",
        "category": "Workflow orchestration",
        "repo": "flyteorg/flyte",
        "github_url": "https://github.com/flyteorg/flyte",
        "official_url": "https://docs.flyte.org/",
        "stars": "7k",
        "latest_update": "2026-05-15",
        "activity": "active",
        "page_structure": "Projects/Domains -> Workflows -> Executions -> Tasks/Artifacts",
        "core_capability": "typed workflow orchestration, reproducible executions, task lineage",
        "ui_pattern": "execution graph with inputs/outputs and retry state",
        "visual_style": "serious workflow console, status-first DAG",
        "deepuplift_absorption": "Future React workflow can show each causal step as typed input/output contracts.",
        "streamlit_direct": "inspired",
        "future_react_fit": "future-react",
        "dependency_risk": "high",
        "migration_cost": "high",
        "status": "future-react",
        "interview_line": "Flyte is a production orchestration reference for long-running training and evidence generation.",
    },
    {
        "platform": "Dagster",
        "category": "Data asset orchestration",
        "repo": "dagster-io/dagster",
        "github_url": "https://github.com/dagster-io/dagster",
        "official_url": "https://docs.dagster.io/",
        "stars": "15.5k",
        "latest_update": "2026-05-19",
        "activity": "very active",
        "page_structure": "Assets -> Asset details -> Runs -> Events/Checks/Lineage",
        "core_capability": "asset lineage, run observation, checks and declarative automation",
        "ui_pattern": "asset catalog, asset checks, lineage graph, run events",
        "visual_style": "polished dark/light neutral shell with colored states used sparingly",
        "deepuplift_absorption": "Evidence should label outputs as assets: dataset, metrics, predictions, screenshots and interview pack.",
        "streamlit_direct": "ready-pattern",
        "future_react_fit": "ready",
        "dependency_risk": "medium if runtime added; low as pattern",
        "migration_cost": "medium",
        "status": "ready",
        "interview_line": "Dagster inspired the asset/check framing for evidence: every artifact has status and lineage.",
    },
    {
        "platform": "BentoML / Yatai",
        "category": "Model serving / deployment",
        "repo": "bentoml/BentoML; bentoml/Yatai",
        "github_url": "https://github.com/bentoml/BentoML",
        "official_url": "https://docs.bentoml.com/",
        "stars": "8.7k / 843",
        "latest_update": "2026-05-07 / 2024-05-08",
        "activity": "active / slower",
        "page_structure": "Models -> Services -> Deployments -> Monitoring",
        "core_capability": "package and serve models, deployment workflows, scalable inference",
        "ui_pattern": "model-to-service transition; deployment status should be separate from training readiness",
        "visual_style": "deployment console, clear environment and version labels",
        "deepuplift_absorption": "Policy/Predict should separate offline readiness from deployment readiness.",
        "streamlit_direct": "inspired",
        "future_react_fit": "future-react",
        "dependency_risk": "medium",
        "migration_cost": "medium",
        "status": "future-react",
        "interview_line": "BentoML is the serving reference; DeepUplift currently focuses on offline causal decision evidence before serving.",
    },
    {
        "platform": "Evidently",
        "category": "ML observability / test suites",
        "repo": "evidentlyai/evidently",
        "github_url": "https://github.com/evidentlyai/evidently",
        "official_url": "https://www.evidentlyai.com/evidently-oss",
        "stars": "7.5k",
        "latest_update": "2026-05-02",
        "activity": "active",
        "page_structure": "Reports -> Test Suites -> Monitoring Dashboards",
        "core_capability": "data/ML monitoring, drift reports, tests and quality gates",
        "ui_pattern": "test suite status, metric cards, report-to-dashboard promotion",
        "visual_style": "quality dashboard: green/amber/red used as semantic states",
        "deepuplift_absorption": "Evidence should use pass/warn/blocker gates for overlap, OPE coverage, smoke and screenshots.",
        "streamlit_direct": "ready-pattern",
        "future_react_fit": "optional",
        "dependency_risk": "medium if integrating metrics",
        "migration_cost": "medium",
        "status": "ready",
        "interview_line": "Evidently inspired the audit-gate language for model and UI evidence.",
    },
    {
        "platform": "Aim",
        "category": "Open-source experiment tracker",
        "repo": "aimhubio/aim",
        "github_url": "https://github.com/aimhubio/aim",
        "official_url": "https://aimstack.readthedocs.io/en/latest/using/manage_runs.html",
        "stars": "6.1k",
        "latest_update": "2025-12-31",
        "activity": "active / stable",
        "page_structure": "Runs -> Explorers -> Metrics/Params/Contexts",
        "core_capability": "local-first experiment tracking and rich metric explorers",
        "ui_pattern": "fast run filtering, metric explorers, configurable comparison views",
        "visual_style": "lean, developer-friendly experiment UI with high data density",
        "deepuplift_absorption": "Evaluation/Compare should support metric-first exploration without hiding formulas.",
        "streamlit_direct": "inspired",
        "future_react_fit": "optional",
        "dependency_risk": "low to medium",
        "migration_cost": "medium",
        "status": "inspired",
        "interview_line": "Aim is a good reference for dense metric exploration while keeping a local workflow.",
    },
    {
        "platform": "Neptune.ai",
        "category": "Experiment tracking reference",
        "repo": "official docs / SaaS",
        "github_url": "https://github.com/neptune-ai",
        "official_url": "https://docs.neptune.ai/",
        "stars": "docs only",
        "latest_update": "2026 docs active",
        "activity": "active docs",
        "page_structure": "Projects -> Runs -> Metadata -> Model Registry",
        "core_capability": "experiment metadata, model registry, comparisons and collaborative review",
        "ui_pattern": "metadata-first table, filterable runs, reviewer-friendly model notes",
        "visual_style": "modern SaaS dashboard with restrained color and high readability",
        "deepuplift_absorption": "History run notes and tags should be interview/reviewer friendly.",
        "streamlit_direct": "inspired",
        "future_react_fit": "optional",
        "dependency_risk": "medium due SaaS dependency",
        "migration_cost": "medium",
        "status": "inspired",
        "interview_line": "Neptune is used as a public design reference, not a dependency.",
    },
]


ML_TRAINING_INSPIRATION_MATRIX: list[dict[str, str]] = [
    {
        "pattern": "Project cockpit",
        "borrowed_from": "MLflow projects, W&B projects, ClearML projects",
        "absorbed_now": "Workflow adds a project status strip for dataset, experiment, registry, evaluation, policy and evidence.",
        "deepuplift_value": "The first screen explains where the current uplift project is in the training lifecycle.",
        "future_react": "Dedicated project route with run timeline and artifact drawer.",
        "status": "ready",
    },
    {
        "pattern": "Run comparison table",
        "borrowed_from": "MLflow, W&B, Aim",
        "absorbed_now": "Compare/History are framed as experiment tracking surfaces: params, metrics, artifacts, tags and notes.",
        "deepuplift_value": "QINI/AUUC/Top-K/OPE/ROI can be compared like experiment metrics, not read as isolated charts.",
        "future_react": "Resizable data grid with run detail side panel.",
        "status": "inspired",
    },
    {
        "pattern": "Model registry with readiness states",
        "borrowed_from": "MLflow registry, W&B registry, BentoML model store",
        "absorbed_now": "Models page keeps registered/ready/guarded/optional/backlog explicit.",
        "deepuplift_value": "Avoids overclaiming: a model can be known, guarded, locally ready or deployment-ready.",
        "future_react": "Registry route with lifecycle stages and promotion actions.",
        "status": "ready",
    },
    {
        "pattern": "Asset lineage / evidence assets",
        "borrowed_from": "Dagster assets, DVC data/model versioning, Flyte executions",
        "absorbed_now": "Evidence treats metrics, predictions, screenshots, source gates and interview packs as auditable assets.",
        "deepuplift_value": "Every resume claim can trace back to data, code, metrics and screenshots.",
        "future_react": "Lineage DAG with artifact previews.",
        "status": "ready",
    },
    {
        "pattern": "Observability gates",
        "borrowed_from": "Ray Dashboard, Evidently, Grafana",
        "absorbed_now": "Evidence and Policy show pass/warn/blocker semantics for smoke, OPE coverage, visual regression and source gates.",
        "deepuplift_value": "Model quality, UI health and evidence health share one operational language.",
        "future_react": "Health dashboard with trend cards and alerts.",
        "status": "ready",
    },
    {
        "pattern": "Workflow DAG",
        "borrowed_from": "Kubeflow Pipelines, Flyte, LangGraph Studio",
        "absorbed_now": "Workflow uses a stage rail rather than a model-name-first layout.",
        "deepuplift_value": "Interviewers see the causal decision lifecycle before algorithms.",
        "future_react": "React Flow causal workflow canvas.",
        "status": "future-react",
    },
]


UPLIFT_APPLICATION_EXPANSION_MATRIX: list[dict[str, str]] = [
    {
        "application": "优惠券 / 补贴 ROI",
        "treatment": "coupon, red packet, subsidy amount",
        "decision_unit": "user x campaign",
        "outcome": "incremental conversion, GMV, margin",
        "policy_metric": "Top-K incremental profit / ROI",
        "recommended_models": "DR/R learner + LightGBM; X-Learner for imbalance; dose response for amount",
        "ui_surface": "Policy -> Scenario ROI Lab; Workflow -> Demo preset; Agent -> coupon interview answer",
        "evidence_gate": "Top-K profit positive, abuse/fatigue risk, bootstrap lower bound",
        "status": "ready",
    },
    {
        "application": "广告增量 / iROAS",
        "treatment": "ad exposure, bid, audience inclusion",
        "decision_unit": "user/ad request/geo cell",
        "outcome": "incremental conversion, incremental GMV, full-funnel conversion",
        "policy_metric": "iROAS, ECUP full-funnel uplift, geo holdout consistency",
        "recommended_models": "DR learner, causal forest, ECUP full-funnel metrics",
        "ui_surface": "Evaluation -> full-funnel metrics; Policy -> iROAS; Evidence -> scenario compare",
        "evidence_gate": "click uplift cannot conflict with conversion/value uplift",
        "status": "ready",
    },
    {
        "application": "CRM / Push / Message diet",
        "treatment": "push, SMS, lifecycle message, suppress/holdout",
        "decision_unit": "user x journey x day",
        "outcome": "retention, activation, opt-out, fatigue-adjusted value",
        "policy_metric": "incremental retention minus fatigue and opt-out cost",
        "recommended_models": "uplift@K, delayed feedback, segment calibration",
        "ui_surface": "Scenario Workbench; Evaluation delayed feedback; Agent CRM prompts",
        "evidence_gate": "frequency cap, sleeping-dog risk, journey overlap",
        "status": "ready",
    },
    {
        "application": "Marketplace 双边补贴",
        "treatment": "buyer coupon, seller co-fund, driver incentive",
        "decision_unit": "city x time x side",
        "outcome": "completion, wait time, margin, supply-demand balance",
        "policy_metric": "constrained budget value with spillover risk",
        "recommended_models": "scenario-specific policy benchmark, causal forest, geo holdout",
        "ui_surface": "Policy budget allocation; Story industrial path",
        "evidence_gate": "interference/spillover and budget pacing warnings visible",
        "status": "ready",
    },
    {
        "application": "LLM routing / Agent escalation",
        "treatment": "cheap model, strong model, RAG, tool, human review",
        "decision_unit": "request x intent x risk",
        "outcome": "quality, task success, latency, evidence risk",
        "policy_metric": "cost-quality policy value, DR OPE, coverage/ESS",
        "recommended_models": "multi-action DR learner, OPE, bandit guardrail",
        "ui_surface": "Policy LLM OPE; Agent LLM routing answer; Evidence smoke",
        "evidence_gate": "logged propensity, coverage, risk threshold",
        "status": "ready",
    },
    {
        "application": "剂量响应 / 价格和折扣",
        "treatment": "discount rate, bid multiplier, credit line amount",
        "decision_unit": "user/item x dose",
        "outcome": "incremental value as dose changes",
        "policy_metric": "gain-vs-baseline and unsafe extrapolation flag",
        "recommended_models": "DoseResponseGBM/RF; future DRNet/VCNet",
        "ui_surface": "Dose-Response; Policy dose optimizer",
        "evidence_gate": "recommended dose is not simply max dose",
        "status": "inspired",
    },
    {
        "application": "客服升级人工 / 审核队列",
        "treatment": "self-serve, AI assist, human escalation",
        "decision_unit": "ticket/request",
        "outcome": "resolution, CSAT, cost, SLA risk",
        "policy_metric": "incremental resolution value minus queue cost",
        "recommended_models": "multi-treatment DR learner, OPE, budgeted queue policy",
        "ui_surface": "Scenario Workbench; Policy decision simulator",
        "evidence_gate": "queue capacity and high-risk routing guardrails",
        "status": "inspired",
    },
    {
        "application": "医疗随访 / 金融授信",
        "treatment": "outreach, reminder, credit offer, rate discount",
        "decision_unit": "patient/account",
        "outcome": "adherence, repayment, risk-adjusted revenue",
        "policy_metric": "incremental value with safety/risk constraints",
        "recommended_models": "interpretable DR/R learner, causal forest, sensitivity checks",
        "ui_surface": "Scenario Workbench; Evidence source gate",
        "evidence_gate": "safety, fairness and compliance notes required",
        "status": "backlog",
    },
]


TRAINING_PROMPT_GROUPS: list[dict[str, list[str] | str]] = [
    {
        "group": "训练平台设计",
        "intent": "解释本轮参考的 ML training / MLOps platform。",
        "prompts": [
            "参考了哪些开源模型训练平台？",
            "哪些训练平台设计被吸收到当前页面？",
            "如何设计工业级 uplift training workbench？",
            "DeepUplift 未来 FastAPI + React 怎么演进？",
        ],
    }
]


def ml_training_platform_audit_rows() -> list[dict[str, str]]:
    return [dict(row) for row in ML_TRAINING_PLATFORM_AUDIT]


def ml_training_inspiration_matrix_rows() -> list[dict[str, str]]:
    return [dict(row) for row in ML_TRAINING_INSPIRATION_MATRIX]


def uplift_application_expansion_rows() -> list[dict[str, str]]:
    return [dict(row) for row in UPLIFT_APPLICATION_EXPANSION_MATRIX]


def ml_training_prompt_groups() -> list[dict[str, list[str] | str]]:
    return [dict(row) for row in TRAINING_PROMPT_GROUPS]


def ml_training_platform_counts() -> dict[str, int]:
    rows = ml_training_platform_audit_rows()
    status_counts = Counter(row["status"] for row in rows)
    return {
        "total": len(rows),
        "ready": status_counts.get("ready", 0),
        "inspired": status_counts.get("inspired", 0),
        "future_react": status_counts.get("future-react", 0),
        "backlog": status_counts.get("backlog", 0),
    }


def training_workbench_stage_rows(context: dict[str, str]) -> list[dict[str, str]]:
    return [
        {
            "stage": "Project",
            "status": "active",
            "signal": "DeepUplift Agent",
            "borrowed_from": "W&B Projects / ClearML Projects",
            "deepuplift_role": "统一展示业务目标、数据、训练、评估、策略和证据。",
        },
        {
            "stage": "Dataset",
            "status": "ready",
            "signal": context.get("dataset", "current dataset"),
            "borrowed_from": "DVC / W&B Dataset Registry",
            "deepuplift_role": "保留 treatment/outcome/schema 与 manifest 证据。",
        },
        {
            "stage": "Experiment",
            "status": "needs-compare" if context.get("compare_ready") != "yes" else "compared",
            "signal": context.get("model", "selected model"),
            "borrowed_from": "MLflow Runs / Aim Runs",
            "deepuplift_role": "把训练配置、params、metrics、run note 和 tags 串起来。",
        },
        {
            "stage": "Model Registry",
            "status": "governed",
            "signal": context.get("model_counts", "registered / ready"),
            "borrowed_from": "MLflow Registry / W&B Registry",
            "deepuplift_role": "区分 registered、ready、guarded、optional、backlog。",
        },
        {
            "stage": "Evaluation",
            "status": context.get("eval_status", "pending"),
            "signal": "QINI / AUUC / Top-K / OPE",
            "borrowed_from": "W&B panels / Aim metric explorers",
            "deepuplift_role": "指标必须服务阈值和业务决策，而不是装饰。",
        },
        {
            "stage": "Policy",
            "status": "decision",
            "signal": "ROI / budget / coverage",
            "borrowed_from": "Evidently gates / Ray observability",
            "deepuplift_role": "把模型分数转成可上线的阈值、预算和风险提示。",
        },
        {
            "stage": "Evidence",
            "status": "audited",
            "signal": "screenshots / smoke / artifacts",
            "borrowed_from": "Dagster checks / DVC lineage",
            "deepuplift_role": "每个 claim 都能追到本地 artifact 和截图。",
        },
    ]


def ml_training_platform_agent_reply(prompt: str) -> str | None:
    text = prompt.strip().lower()
    if not text:
        return None
    trigger_terms = [
        "训练平台",
        "开源模型训练",
        "mlflow",
        "wandb",
        "w&b",
        "clearml",
        "kubeflow",
        "aim",
        "experiment tracking",
        "mlops",
        "training workbench",
        "training platform",
        "模型训练平台",
        "实验追踪",
    ]
    if not any(term in text for term in trigger_terms):
        return None
    counts = ml_training_platform_counts()
    ready = [row["platform"] for row in ML_TRAINING_PLATFORM_AUDIT if row["status"] == "ready"]
    inspired = [row["platform"] for row in ML_TRAINING_PLATFORM_AUDIT if row["status"] == "inspired"]
    future = [row["platform"] for row in ML_TRAINING_PLATFORM_AUDIT if row["status"] == "future-react"]

    if any(key in text for key in ["吸收", "设计", "页面", "workbench"]):
        patterns = "\n".join(
            f"- **{row['pattern']}**：{row['absorbed_now']} 价值：{row['deepuplift_value']}"
            for row in ML_TRAINING_INSPIRATION_MATRIX
        )
        return (
            "### 已吸收的训练平台设计\n\n"
            f"{patterns}\n\n"
            "一句话：DeepUplift 不复制某个训练平台，而是把 experiment tracking、model registry、asset lineage、"
            "observability gate 这些成熟模式压缩进当前 Streamlit。"
        )

    if any(key in text for key in ["参考", "哪些", "github", "平台", "审计"]):
        return (
            "### 开源模型训练平台参考\n\n"
            f"本轮审计了 `{counts['total']}` 个训练 / 实验追踪 / MLOps 平台。直接吸收的 ready pattern："
            f"`{', '.join(ready)}`。\n\n"
            f"作为 inspiration 的平台：`{', '.join(inspired)}`。\n\n"
            f"未来 React/FastAPI 生产化候选：`{', '.join(future)}`。\n\n"
            "落地位置：`Workflow -> ML Training Workbench Overview`、`Models -> ML Training Platform Inspiration Matrix`、"
            "`Policy -> Uplift Application Expansion Matrix`、`Evidence -> UI Screenshot Gallery / Visual Regression Evidence`。"
        )

    if any(key in text for key in ["fastapi", "react", "未来", "演进"]):
        return (
            "### 未来 FastAPI + React 演进\n\n"
            "后端保留当前 contracts：dataset manifest、diagnostics、trainer、evaluator、policy、artifacts、screenshots。"
            "FastAPI 负责 experiments/runs/models/artifacts API，worker 负责训练、compare、OPE 和 smoke，"
            "React 负责 run comparison grid、artifact drawer、workflow DAG 和 evidence observability。"
            "可借鉴 MLflow/W&B 的 run registry，Dagster/Flyte 的 lineage，ECharts/Plotly 的交互曲线。"
        )

    return (
        "工业级 uplift training workbench 的信息架构是："
        "`Project -> Dataset -> Diagnostics -> Experiment Runs -> Model Registry -> Evaluation -> Policy -> Evidence -> Agent`。"
        "这套结构来自 MLflow/W&B/ClearML/Kubeflow/Aim/Dagster 等平台，但 DeepUplift 的差异是把所有指标都对齐到 uplift 决策："
        "QINI/AUUC 看排序，Top-K/ROI 看投放，OPE/coverage 看离线可上线性，Evidence 看可复现。"
    )
