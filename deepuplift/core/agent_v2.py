from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from .model_deconstruction import model_deconstruction_reply

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class EvidenceItem:
    id: str
    title: str
    path: str
    kind: str
    keywords: tuple[str, ...]
    claim: str


@dataclass
class AgentIntent:
    scenario: str
    question_type: str
    treatment_type: str
    treatment: str
    outcome: str
    unit: str
    time_window: str
    confounders: list[str]
    cost_value: str
    guardrails: list[str]
    risks: list[str]
    model_families: list[str]
    evaluation_plan: list[str]
    rollout_plan: list[str]
    answer_modes: list[str]
    evidence_ids: list[str]


EVIDENCE_CATALOG: tuple[EvidenceItem, ...] = (
    EvidenceItem(
        "agent_v2",
        "Agent 2.0 architecture",
        "docs/DEEPUplift_AGENT_2_0_ARCHITECTURE.md",
        "doc",
        ("agent", "copilot", "architecture", "intent", "rubric"),
        "Agent 2.0 parses causal questions, routes to risk/model/evidence blocks, and is regression scored.",
    ),
    EvidenceItem(
        "agent_regression",
        "Agent answer regression",
        "reports/agent_v2_regression_latest.json",
        "report",
        ("agent", "golden", "rubric", "regression", "question"),
        "Golden-question regression checks answer quality, evidence grounding, and interview readiness.",
    ),
    EvidenceItem(
        "evidence_index",
        "Evidence index",
        "docs/DEEPUplift_AGENT_EVIDENCE_INDEX.md",
        "doc",
        ("evidence", "artifact", "report", "run", "可信", "复现"),
        "The evidence index maps claims to reports, docs, screenshots, and run artifacts.",
    ),
    EvidenceItem(
        "benchmark",
        "Benchmark suite v2",
        "docs/DEEPUplift_BENCHMARK_SUITE.md",
        "doc",
        ("benchmark", "leaderboard", "seed", "diff", "paper", "对标"),
        "Benchmark v2 supports multi-seed runs, leaderboard, failure attribution, and run diff.",
    ),
    EvidenceItem(
        "benchmark_report",
        "Latest benchmark report",
        "reports/benchmark_suite_latest.json",
        "report",
        ("benchmark", "leaderboard", "qini", "auuc"),
        "Latest benchmark artifact contains runs, leaderboard, failure attribution, and evidence manifests.",
    ),
    EvidenceItem(
        "warning_audit",
        "Regression Warning / Launch Guardrail Audit",
        "docs/DEEPUplift_REGRESSION_WARNING_AUDIT.md",
        "doc",
        ("warning", "guardrail", "readiness", "launch", "不能上线", "风险", "门禁"),
        "Regression warning audit converts negative ROI, weak overlap, sensitivity failures, metric conflicts, and deep-model baseline losses into launch review actions.",
    ),
    EvidenceItem(
        "promotion_cards",
        "Promotion Launch Cards",
        "docs/DEEPUplift_PROMOTION_LAUNCH_CARDS.md",
        "doc",
        ("promotion", "launch card", "review card", "上线评审", "放行", "阻断", "owner"),
        "Promotion launch cards aggregate warning rows into decision, owner, safe/unsafe claim, next experiment, and rollout-gate requirements.",
    ),
    EvidenceItem(
        "warning_triage",
        "Regression Warning Triage",
        "docs/DEEPUplift_REGRESSION_WARNING_TRIAGE.md",
        "doc",
        (
            "warning triage",
            "risk family",
            "claim boundary",
            "action lane",
            "smoke-only",
            "launch blocker",
            "blocked_for_pilot",
            "风险分流",
        ),
        "Regression warning triage separates smoke-only caveats from launch-blocking causal, ROI, overlap, readiness, and benchmark risks.",
    ),
    EvidenceItem(
        "pilot_readiness",
        "Pilot Readiness Scorecard",
        "docs/DEEPUplift_PILOT_READINESS_SCORECARD.md",
        "doc",
        (
            "pilot readiness",
            "pilot gate",
            "readiness score",
            "shadow candidate",
            "paper review candidate",
            "rollback conditions",
            "上线试点",
            "试点门禁",
        ),
        "Pilot readiness joins promotion cards and warning triage into a gate, score, next gate, rollback plan, and required artifacts.",
    ),
    EvidenceItem(
        "pilot_experiment_plan",
        "Pilot Experiment Plan",
        "docs/DEEPUplift_PILOT_EXPERIMENT_PLAN.md",
        "doc",
        (
            "pilot experiment",
            "next experiment",
            "pass gate",
            "fail action",
            "claim upgrade",
            "recommended command",
            "试点实验",
            "下一步怎么跑",
        ),
        "Pilot experiment plan turns readiness gates into executable commands, metrics to watch, expected artifacts, pass gates, fail actions, and claim-upgrade boundaries.",
    ),
    EvidenceItem(
        "pilot_experiment_command_smoke",
        "Pilot Experiment Command Smoke",
        "docs/DEEPUplift_PILOT_EXPERIMENT_COMMAND_SMOKE.md",
        "doc",
        (
            "pilot command smoke",
            "command smoke",
            "safe smoke command",
            "no-update-latest",
            "sampled smoke",
            "命令 smoke",
            "命令真的跑",
        ),
        "Pilot experiment command smoke executes sampled next-experiment commands with tiny-row safe overrides and isolated outputs, proving command executability without overwriting main benchmark evidence.",
    ),
    EvidenceItem(
        "algorithm_claim_ledger",
        "Algorithm Claim Ledger",
        "docs/DEEPUplift_ALGORITHM_CLAIM_LEDGER.md",
        "doc",
        ("claim", "safe claim", "unsafe claim", "algorithm claim", "过度包装", "证据层级", "源码证明"),
        "Algorithm claim ledger binds every model claim to formula, source path, benchmark verdict, license boundary, safe/unsafe claim, and UI route.",
    ),
    EvidenceItem(
        "paper_reproduction_gap_ledger",
        "Paper Reproduction Gap Ledger",
        "docs/DEEPUplift_PAPER_REPRODUCTION_GAP_LEDGER.md",
        "doc",
        ("paper reproduction", "reproduction gap", "论文复现", "复现程度", "license risk", "复现差距"),
        "Paper reproduction gap ledger separates paper citation, open-source reference, local reproduction scope, proof level, license risk, and remaining gaps.",
    ),
    EvidenceItem(
        "model_evidence_promotion_matrix",
        "Model Evidence Promotion Matrix",
        "docs/DEEPUplift_MODEL_EVIDENCE_PROMOTION_MATRIX.md",
        "doc",
        ("promotion matrix", "model evidence", "模型推广", "推广矩阵", "promotion tier", "safe claim", "blocked claim"),
        "Model evidence promotion matrix turns source, reproduction, benchmark, objective, license and launch evidence into one promotion-tier table.",
    ),
    EvidenceItem(
        "model_upgrade_recipe_cards",
        "Model Upgrade Recipe Cards",
        "docs/DEEPUplift_MODEL_UPGRADE_RECIPE_CARDS.md",
        "doc",
        ("upgrade recipe", "recipe card", "升级实验", "下一步实验", "pass gate", "fail action"),
        "Model upgrade recipe cards convert promotion gaps into executable hypotheses, commands, metrics, pass gates, fail actions and expected artifacts.",
    ),
    EvidenceItem(
        "deep_model_telemetry_gap_matrix",
        "Deep Model Telemetry Gap Matrix",
        "docs/DEEPUplift_DEEP_MODEL_TELEMETRY_GAP_MATRIX.md",
        "doc",
        ("telemetry gap", "loss telemetry", "可观测性", "representation telemetry", "propensity calibration", "attention weights", "head consistency"),
        "Deep model telemetry gap matrix maps each deep loss term to current signals, missing telemetry, implementation hooks, pass gates and failure actions.",
    ),
    EvidenceItem(
        "deep_model_telemetry_smoke",
        "Deep Model Telemetry Smoke",
        "docs/DEEPUplift_DEEP_MODEL_TELEMETRY_SMOKE.md",
        "doc",
        ("telemetry smoke", "basic telemetry", "loss history", "prediction spread", "top-k known-cate", "advanced hook"),
        "Deep model telemetry smoke reads latest run artifacts and verifies basic loss, prediction, known-CATE, metric and manifest observability.",
    ),
    EvidenceItem(
        "deep_model_forward_hook_contracts",
        "Deep Model Forward Hook Contracts",
        "docs/DEEPUplift_DEEP_MODEL_FORWARD_HOOK_CONTRACTS.md",
        "doc",
        ("forward hook", "hook contract", "tensor keys", "artifact files", "phi_x", "e_hat", "tarreg", "attention", "head export"),
        "Deep model forward hook contracts define guarded tensor schemas, artifact schemas, pass gates, fallbacks and backward-compatibility rules before model-source refactors.",
    ),
    EvidenceItem(
        "deep_model_forward_hook_smoke",
        "Deep Model Forward Hook Smoke",
        "docs/DEEPUplift_DEEP_MODEL_FORWARD_HOOK_SMOKE.md",
        "doc",
        ("forward hook smoke", "sidecar artifacts", "tensor export", "phi_x", "e_hat", "tarreg", "attention", "heads", "constraints"),
        "Deep model forward hook smoke executes guarded model-forward instrumentation and writes sidecar artifacts without changing default model APIs.",
    ),
    EvidenceItem(
        "deep_model_hook_training_bridge",
        "Deep Model Hook Training Bridge",
        "docs/DEEPUplift_DEEP_MODEL_HOOK_TRAINING_BRIDGE.md",
        "doc",
        ("hook training bridge", "trainer telemetry", "per-epoch", "epoch columns", "optional collector", "训练桥接", "训练期可观测"),
        "Deep model hook training bridge maps forward-hook sidecars to training runs, per-epoch telemetry columns, pass gates and fail actions.",
    ),
    EvidenceItem(
        "deep_model_trainer_collector_smoke",
        "Deep Model Trainer Collector Smoke",
        "docs/DEEPUplift_DEEP_MODEL_TRAINER_COLLECTOR_SMOKE.md",
        "doc",
        ("trainer collector", "collector smoke", "per-epoch telemetry", "advanced telemetry", "训练期采集", "epoch telemetry"),
        "Deep model trainer collector smoke runs tiny training loops and writes per-epoch advanced telemetry sidecars for deep uplift models.",
    ),
    EvidenceItem(
        "deep_model_telemetry_benchmark_linkage",
        "Deep Model Telemetry-Benchmark Linkage",
        "docs/DEEPUplift_DEEP_MODEL_TELEMETRY_BENCHMARK_LINKAGE.md",
        "doc",
        ("telemetry benchmark", "benchmark linkage", "遥测和benchmark", "胜负归因", "next ablation", "failure hypothesis"),
        "Deep model telemetry-benchmark linkage joins trainer collector metrics with paper/tuned benchmark verdicts, failure hypotheses and next ablations.",
    ),
    EvidenceItem(
        "deep_model_telemetry_ablation_gate",
        "Deep Model Telemetry Ablation Gate",
        "docs/DEEPUplift_DEEP_MODEL_TELEMETRY_ABLATION_GATE.md",
        "doc",
        (
            "ablation gate",
            "telemetry ablation",
            "variant coverage",
            "missing variants",
            "EFIN_no_attention",
            "EFIN_path_regularized",
            "DESCN_no_constraint",
            "实验门禁",
            "消融门禁",
        ),
        "Deep model telemetry ablation gate turns each linkage row into required variants, metrics, pass gate, fail action, safe claim and blocked claim.",
    ),
    EvidenceItem(
        "deep_model_ablation_interpretation",
        "Deep Model Ablation Interpretation",
        "docs/DEEPUplift_DEEP_MODEL_ABLATION_INTERPRETATION.md",
        "doc",
        (
            "ablation interpretation",
            "消融解读",
            "default vs variant",
            "metric movement",
            "variant scorecard",
            "EFIN_no_attention",
            "EFIN_path_regularized",
            "DESCN_no_constraint",
        ),
        "Deep model ablation interpretation explains PEHE/QINI/policy/loss movement for each runnable variant versus the default model.",
    ),
    EvidenceItem(
        "deep_model_ablation_promotion_matrix",
        "Deep Model Ablation Promotion Matrix",
        "docs/DEEPUplift_DEEP_MODEL_ABLATION_PROMOTION_MATRIX.md",
        "doc",
        (
            "ablation promotion",
            "ablation promotion matrix",
            "claim tier",
            "promotion decision",
            "default or variant",
            "variant promotion",
            "消融推广",
            "claim 升级",
            "变体能不能讲",
        ),
        "Deep model ablation promotion matrix converts runnable ablation evidence into claim tiers, default-or-variant recommendations and blocked claims.",
    ),
    EvidenceItem(
        "deep_model_ablation_next_experiment_plan",
        "Deep Model Ablation Next Experiment Plan",
        "docs/DEEPUplift_DEEP_MODEL_ABLATION_NEXT_EXPERIMENT_PLAN.md",
        "doc",
        (
            "ablation next experiment",
            "next experiment plan",
            "recommended command",
            "metrics to watch",
            "pass gate",
            "fail action",
            "下一步实验",
            "面试追问",
        ),
        "Deep model ablation next-experiment plan turns claim tiers into concrete commands, metrics, pass gates, fail actions and owner roles.",
    ),
    EvidenceItem(
        "deep_model_ablation_command_contract",
        "Deep Model Ablation Command Contract",
        "docs/DEEPUplift_DEEP_MODEL_ABLATION_COMMAND_CONTRACT.md",
        "doc",
        (
            "ablation command contract",
            "command contract",
            "runnable fallback",
            "unsupported flags",
            "runner cli",
            "可执行命令",
            "命令契约",
            "fallback command",
        ),
        "Deep model ablation command contract audits planned experiment commands against the current tuned benchmark runner CLI and requires a runnable fallback.",
    ),
    EvidenceItem(
        "deep_model_ablation_command_contract_smoke",
        "Deep Model Ablation Command Contract Smoke",
        "docs/DEEPUplift_DEEP_MODEL_ABLATION_COMMAND_CONTRACT_SMOKE.md",
        "doc",
        (
            "command contract smoke",
            "fallback smoke",
            "fallback command smoke",
            "no-update-latest",
            "isolated output",
            "命令 smoke",
            "真的执行",
        ),
        "Deep model ablation command contract smoke executes sampled fallback commands with isolated output so command executability has run evidence.",
    ),
    EvidenceItem(
        "deep_model_ablation_recommended_command_smoke",
        "Deep Model Ablation Recommended Command Smoke",
        "docs/DEEPUplift_DEEP_MODEL_ABLATION_RECOMMENDED_COMMAND_SMOKE.md",
        "doc",
        (
            "recommended command smoke",
            "runner-native command",
            "runner native aliases",
            "--models",
            "--include-variants",
            "--emit-battle-cards",
            "recommended command 真的能跑",
            "推荐命令 smoke",
        ),
        "Deep model ablation recommended command smoke executes sampled runner-native recommended commands with isolated output.",
    ),
    EvidenceItem(
        "deep_model_ablation_command_smoke_coverage",
        "Deep Model Ablation Command Smoke Coverage",
        "docs/DEEPUplift_DEEP_MODEL_ABLATION_COMMAND_SMOKE_COVERAGE.md",
        "doc",
        (
            "command smoke coverage",
            "smoke coverage",
            "coverage gate",
            "contract ready not smoked",
            "哪些已经 smoke",
            "哪些还没 smoke",
            "覆盖率",
            "不要过度包装",
        ),
        "Deep model ablation command smoke coverage joins command contracts with fallback and runner-native smoke rows so smoke coverage gaps are explicit.",
    ),
    EvidenceItem(
        "deep_model_ablation_command_smoke_coverage_diff",
        "Deep Model Ablation Command Smoke Coverage Diff",
        "docs/DEEPUplift_DEEP_MODEL_ABLATION_COMMAND_SMOKE_COVERAGE_DIFF.md",
        "doc",
        (
            "coverage diff",
            "smoke coverage diff",
            "gate delta",
            "review_required -> ok",
            "1/9",
            "9/9",
            "覆盖率变化",
            "本轮优化提升",
        ),
        "Deep model ablation command smoke coverage diff explains the before/after upgrade from fallback-only coverage to full runner-native smoke coverage.",
    ),
    EvidenceItem(
        "dataset_registry",
        "Dataset registry",
        "docs/DEEPUplift_DATASET_REGISTRY.md",
        "doc",
        ("dataset", "data", "registry", "card", "license", "数据"),
        "Dataset cards document data profile, license/reuse risk, causal assumptions, and benchmark tier.",
    ),
    EvidenceItem(
        "ope",
        "General OPE engine",
        "deepuplift/core/ope.py",
        "code",
        ("ope", "ips", "snips", "dr", "off-policy", "ess"),
        "General OPE engine estimates IPS, SNIPS, DR value, ESS, coverage, and clipping sensitivity.",
    ),
    EvidenceItem(
        "nuisance",
        "Cross-fitted nuisance diagnostics",
        "docs/DEEPUplift_NUISANCE_DIAGNOSTICS.md",
        "doc",
        ("dr", "r learner", "orthogonal", "nuisance", "propensity", "overlap", "ess"),
        "DR/R/IPW learners persist nuisance diagnostics, propensity summaries, pseudo-outcome stability, and ESS.",
    ),
    EvidenceItem(
        "policy",
        "Policy value and ROI derivation",
        "docs/UPLIFT_POLICY_VALUE_AND_ROI_DERIVATION.md",
        "doc",
        ("policy", "roi", "iroas", "budget", "profit", "成本", "收益"),
        "Policy docs derive incremental profit, ROI/iROAS, OPE, and budget-constrained decisions.",
    ),
    EvidenceItem(
        "policy_code",
        "Budget policy optimizer",
        "deepuplift/core/policy.py",
        "code",
        ("budget", "optimizer", "threshold", "policy", "roi"),
        "Budget policy optimizer maps uplift scores to constrained treatment/routing decisions.",
    ),
    EvidenceItem(
        "model_matrix",
        "Scenario model matrix",
        "docs/UPLIFT_SCENARIO_MODEL_MATRIX.md",
        "doc",
        ("model", "scenario", "recommend", "matrix", "模型"),
        "Scenario model matrix maps business treatment design to candidate model families.",
    ),
    EvidenceItem(
        "deep_loss",
        "Deep uplift loss catalog",
        "docs/UPLIFT_DEEP_LOSS_AND_ARCHITECTURE.md",
        "doc",
        ("tarnet", "cfrnet", "dragonnet", "descn", "efin", "loss", "deep"),
        "Deep loss docs explain representation balance, targeted regularization, full-funnel, and LLM routing losses.",
    ),
    EvidenceItem(
        "llm_routing",
        "LLM routing playbook",
        "docs/LLM_ROUTING_POLICY_PLAYBOOK.md",
        "doc",
        ("llm", "routing", "rag", "tool", "human", "strong model"),
        "LLM routing is treated as uplift over cheap-model control with cost, latency, and quality guardrails.",
    ),
    EvidenceItem(
        "online",
        "Offline-to-online validation",
        "docs/DEEPUplift_AGENT_OFFLINE_ONLINE_PLAYBOOK.md",
        "doc",
        ("online", "ab", "a/b", "holdout", "geo", "switchback", "上线"),
        "Offline uplift candidates must move through holdout, shadow, A/B, geo/switchback, and ramp-up gates.",
    ),
    EvidenceItem(
        "failure_modes",
        "Failure modes",
        "docs/UPLIFT_MODEL_FAILURE_MODES.md",
        "doc",
        ("leakage", "sutva", "interference", "delayed", "roi trap", "metric", "warning", "guardrail"),
        "Failure mode cards cover leakage, weak overlap, interference, delayed feedback, and metric traps.",
    ),
    EvidenceItem(
        "evidence_store",
        "Experiment evidence store",
        "docs/DEEPUplift_EXPERIMENT_EVIDENCE_STORE.md",
        "doc",
        ("evidence store", "run diff", "manifest", "artifact"),
        "Experiment evidence store indexes manifests and run diffs for auditability.",
    ),
)


AGENT_V2_GOLDEN_QUESTIONS: tuple[dict[str, Any], ...] = (
    {"prompt": "为什么不用普通转化率模型？", "expect": ("P(Y=1|X)", "CATE", "policy value", "证据")},
    {"prompt": "观测数据有偏怎么办？", "expect": ("selection bias", "overlap", "DR/R", "nuisance")},
    {"prompt": "给我解析一下 treatment、outcome、unit、time window 和 confounders", "expect": ("treatment", "outcome", "unit", "time window")},
    {"prompt": "怎么识别 leakage、SUTVA 干扰和 delayed feedback？", "expect": ("leakage", "SUTVA", "delayed feedback", "guardrail")},
    {"prompt": "发券场景怎么从 uplift 转成 ROI？", "expect": ("coupon", "incremental profit", "budget", "ROI")},
    {"prompt": "广告投放里 uplift 和 pCTR/pCVR 有什么区别？", "expect": ("pCTR", "pCVR", "incrementality", "iROAS")},
    {"prompt": "LLM routing 为什么是 uplift 问题？", "expect": ("cheap model", "strong model", "treatment arms", "budget")},
    {"prompt": "OPE / IPS / SNIPS / DR 怎么评估新策略？", "expect": ("IPS", "SNIPS", "DR", "effective sample size")},
    {"prompt": "什么时候从 uplift 升级到 contextual bandit？", "expect": ("fixed policy", "contextual bandit", "OPE", "budget pacing")},
    {"prompt": "multi-treatment 优惠券怎么建模？", "expect": ("multi-treatment", "action", "MultiDR", "propensity")},
    {"prompt": "DRNet / VCNet 连续 treatment 怎么讲？", "expect": ("dose-response", "support", "DRNet", "VCNet")},
    {"prompt": "TARNet/CFRNet/DragonNet/EFIN/DESCN 怎么选？", "expect": ("TARNet", "CFRNet", "DragonNet", "EFIN")},
    {"prompt": "怎么跑 benchmark 并和论文/开源框架对标？", "expect": ("benchmark", "leaderboard", "multi-seed", "failure attribution")},
    {"prompt": "怎么证明一次训练是可信的？", "expect": ("evidence manifest", "run artifacts", "nuisance", "environment")},
    {"prompt": "模型 registry 和 guarded optional dependency 怎么讲？", "expect": ("registry", "guarded", "optional", "model card")},
    {"prompt": "上线试点路径怎么走？", "expect": ("offline", "shadow", "A/B", "ramp-up")},
    {"prompt": "预算有限时 uplift 阈值怎么定？", "expect": ("budget", "net value", "optimizer", "ROI")},
    {"prompt": "Marketplace 干扰和 spillover 怎么处理？", "expect": ("interference", "spillover", "cluster", "geo")},
    {"prompt": "Push 频控和用户疲劳怎么建模？", "expect": ("fatigue", "frequency", "delayed feedback", "holdout")},
    {"prompt": "5分钟、15分钟、30分钟分别怎么讲？", "expect": ("5分钟", "15分钟", "30分钟", "evidence")},
    {"prompt": "面试官追问 QINI/AUUC 不等于利润时怎么答？", "expect": ("QINI", "AUUC", "policy value", "profit")},
    {"prompt": "Agent 2.0 相比旧问答助手强在哪里？", "expect": ("causal decision copilot", "intent", "evidence", "rubric")},
    {"prompt": "为什么 Evidence 里有很多 warning，哪些不能上线，下一步怎么补实验？", "expect": ("regression warning", "launch guardrail", "readiness", "holdout")},
    {"prompt": "给我一张上线评审 promotion card，说明哪些 claim 能讲、哪些不能讲？", "expect": ("promotion", "safe claim", "unsafe claim", "owner")},
    {"prompt": "warning triage 怎么区分 smoke-only caveat 和真正阻塞 pilot 的 launch blocker？", "expect": ("warning triage", "smoke-only", "launch_blocking", "claim boundary")},
    {"prompt": "pilot readiness scorecard 怎么判断 blocked、paper review、shadow candidate 和 monitor？", "expect": ("pilot readiness", "pilot_gate", "shadow", "rollback")},
    {"prompt": "pilot experiment plan 怎么把 readiness gate 转成可执行实验命令、pass gate 和 fail action？", "expect": ("pilot experiment", "recommended command", "pass gate", "fail action")},
    {"prompt": "pilot command smoke 怎么证明这些试点实验命令真的能跑，又不会覆盖主 benchmark latest？", "expect": ("pilot command smoke", "safe smoke command", "no-update-latest", "runner")},
    {"prompt": "怎么用 algorithm claim ledger 证明模型 claim 没有过度包装？", "expect": ("algorithm claim ledger", "safe claim", "unsafe claim", "evidence")},
    {"prompt": "怎么回答论文复现到什么程度，哪些只是参考没有复制代码？", "expect": ("paper reproduction", "license", "benchmark", "evidence")},
    {"prompt": "怎么用 model evidence promotion matrix 说明每个模型能讲到什么程度？", "expect": ("promotion matrix", "safe claim", "blocked claim", "evidence")},
    {"prompt": "如果面试官问下一步怎么把模型证据升级，你怎么用 recipe card 回答？", "expect": ("recipe card", "pass gate", "fail action", "expected artifacts")},
    {"prompt": "CFRNet/DragonNet/EFIN/DESCN 的 deep model telemetry gap matrix 怎么帮助失败归因？", "expect": ("telemetry gap", "loss", "pass gate", "failure")},
    {"prompt": "deep model telemetry smoke 证明了什么，还缺哪些高级 hook？", "expect": ("telemetry smoke", "loss history", "advanced hook", "evidence")},
    {"prompt": "forward hook contract 怎么指导 CFRNet/DragonNet/EFIN/DESCN 的源码改造？", "expect": ("forward hook", "tensor keys", "artifact", "pass gate")},
    {"prompt": "forward hook smoke 现在实际导出了哪些 tensor 和 sidecar artifacts？", "expect": ("forward hook smoke", "sidecar artifacts", "phi_x", "attention")},
    {"prompt": "hook training bridge 怎么把一次 forward tensor 导出升级成训练期证据？", "expect": ("hook training bridge", "per-epoch", "pass gate", "evidence")},
    {"prompt": "trainer collector smoke 现在证明了哪些 per-epoch advanced telemetry？", "expect": ("trainer collector", "per-epoch", "CFRNet", "DragonNet")},
    {"prompt": "telemetry benchmark linkage 怎么解释深度模型为什么赢或输？", "expect": ("telemetry", "benchmark", "next ablation", "safe claim")},
    {"prompt": "telemetry ablation gate 怎么说明下一步实验能不能升级 claim？", "expect": ("ablation gate", "variant", "pass gate", "blocked claim")},
    {"prompt": "EFIN_no_attention、EFIN_path_regularized、DESCN_no_constraint 这些消融变体现在有什么证据？", "expect": ("EFIN_no_attention", "EFIN_path_regularized", "DESCN_no_constraint", "ablation gate")},
    {"prompt": "deep model ablation interpretation 怎么解释消融结果，而不是只说 variant 覆盖？", "expect": ("ablation interpretation", "default", "PEHE", "safe claim")},
    {"prompt": "deep model ablation promotion matrix 怎么判断一个消融 variant 能不能升级成面试 claim？", "expect": ("ablation promotion", "claim tier", "promotion decision", "blocked claim")},
    {"prompt": "消融 promotion matrix 之后，下一步实验计划怎么回答面试官追问？", "expect": ("next experiment", "recommended command", "pass gate", "fail action")},
    {"prompt": "next experiment 里的命令哪些真的能跑，哪些只是 runner adapter 规划？", "expect": ("command contract", "runnable fallback", "unsupported", "runner")},
    {"prompt": "fallback command smoke 真的执行了吗，会不会覆盖主 benchmark latest？", "expect": ("fallback smoke", "no-update-latest", "manifest", "benchmark")},
    {"prompt": "recommended command 现在真的能跑吗，--models 和 --include-variants 是不是原生支持？", "expect": ("recommended command smoke", "--models", "--include-variants", "runner-native")},
    {"prompt": "9 个 ablation 推荐实验里哪些已经 smoke，哪些只是 contract-ready backlog？", "expect": ("smoke coverage", "contract-ready", "recommended_smoked", "coverage gate")},
    {"prompt": "这轮 ablation command smoke coverage 相比之前提升了什么，gate 为什么从 review_required 变成 ok？", "expect": ("coverage diff", "1/9", "9/9", "review_required -> ok")},
)


def _has_any(text: str, keywords: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def _dedupe(items: Iterable[str]) -> list[str]:
    rows: list[str] = []
    for item in items:
        if item and item not in rows:
            rows.append(item)
    return rows


def _context_value(context: Mapping[str, Any], key: str, default: str) -> str:
    value = context.get(key)
    return default if value in {None, ""} else str(value)


def _scenario(text: str) -> str:
    if _has_any(text, ["llm", "routing", "rag", "tool", "强模型", "人工审核"]):
        return "LLM routing / cost-quality escalation"
    if _has_any(text, ["multi", "多 treatment", "多动作", "多券", "多 treatment"]):
        return "multi-treatment action optimization"
    if _has_any(text, ["发券", "红包", "coupon", "补贴", "优惠券"]):
        return "coupon / subsidy allocation"
    if _has_any(text, ["广告", "ads", "pctr", "pcvr", "iroas", "ghost ads"]):
        return "ads / incrementality"
    if _has_any(text, ["marketplace", "spillover", "供需", "干扰", "sutva"]):
        return "marketplace / interference"
    if _has_any(text, ["push", "crm", "频控", "疲劳", "journey", "消息"]):
        return "growth / CRM lifecycle"
    if _has_any(text, ["dose", "连续", "drnet", "vcnet", "剂量", "折扣率"]):
        return "continuous treatment / dose-response"
    if _has_any(text, ["benchmark", "论文", "开源", "leaderboard"]):
        return "benchmark / research validation"
    if _has_any(text, ["warning", "guardrail", "readiness", "pilot", "不能上线", "门禁", "风险审计", "promotion", "上线评审", "放行", "阻断", "试点"]):
        return "launch guardrail review"
    if _has_any(text, ["上线", "ab", "a/b", "holdout", "shadow", "ramp"]):
        return "offline-to-online rollout"
    return "general uplift modeling"


def parse_agent_intent(prompt: str, context: Mapping[str, Any] | None = None) -> AgentIntent:
    context = context or {}
    text = prompt.lower()
    scenario = _scenario(text)

    treatment_type = "binary"
    if "multi" in scenario:
        treatment_type = "multi-treatment"
    elif "continuous" in scenario or _has_any(text, ["drnet", "vcnet", "dose", "连续", "剂量"]):
        treatment_type = "continuous"
    elif "llm routing" in scenario:
        treatment_type = "binary or multi-action routing"

    treatment = _context_value(context, "treatment_col", "business intervention / treatment")
    outcome = _context_value(context, "outcome_col", "incremental outcome / reward")
    unit = _context_value(context, "unit_col", "user/request/city-time cell")
    time_window = _context_value(context, "time_window", "label window matched to decision cycle")

    confounders = ["pre-treatment behavior", "user value", "channel/device/region", "historical exposure"]
    if "marketplace" in scenario:
        confounders += ["supply-demand gap", "city/time cluster", "competition intensity"]
    if "LLM" in scenario:
        confounders += ["prompt difficulty", "latency SLO", "safety risk", "model load"]

    cost_value = "net value = uplift * business value - treatment cost - risk/fatigue/latency penalty"
    guardrails = ["overlap / positivity", "leakage check", "calibration", "bootstrap/sensitivity", "holdout validation"]
    if _has_any(text, ["warning", "guardrail", "readiness", "pilot", "不能上线", "门禁", "风险审计", "warning triage", "风险分流", "claim boundary", "action lane", "smoke-only", "promotion", "上线评审", "放行", "阻断", "试点"]):
        guardrails += ["Regression Warning Audit", "Launch Guardrail Audit", "readiness gate"]
    risks: list[str] = []
    if _has_any(text, ["有偏", "selection", "confounding", "观测", "overlap"]):
        risks += ["selection bias", "weak overlap", "confounding"]
    if _has_any(text, ["leakage", "泄漏"]):
        risks += ["feature/label leakage"]
    if _has_any(text, ["sutva", "interference", "spillover", "干扰"]):
        risks += ["SUTVA violation / interference / spillover"]
    if _has_any(text, ["delayed", "延迟", "疲劳", "频控"]):
        risks += ["delayed feedback", "fatigue", "long-term cannibalization"]
    if _has_any(text, ["roi", "profit", "利润", "预算", "成本", "qini", "auuc"]):
        risks += ["metric-to-ROI mismatch", "budget exhaustion"]
    if _has_any(text, ["warning", "guardrail", "readiness", "pilot", "不能上线", "门禁", "风险审计", "warning triage", "风险分流", "claim boundary", "action lane", "smoke-only", "promotion", "上线评审", "放行", "阻断", "试点"]):
        risks += [
            "launch guardrail failure",
            "negative ROI or non-positive policy value",
            "weak overlap or low ESS",
            "sensitivity / metric conflict not resolved",
        ]
    if not risks:
        risks = ["metric mismatch", "overlap risk", "online generalization risk"]

    models = ["TLearnerGBM baseline", "XLearnerGBM for imbalance", "DRLearnerGBM / RLearnerGBM for observational bias"]
    if _has_any(text, ["dml", "orthogonal", "r learner", "有偏", "观测"]):
        models += ["OrthogonalDMLGBM", "CausalForest / EconMLDML if dependency is ready"]
    if _has_any(text, ["tarnet", "cfrnet", "dragonnet", "efin", "descn", "deep", "深度"]):
        models += ["TARNet", "CFRNet", "DragonNet", "DESCN/EFIN"]
    if treatment_type == "multi-treatment":
        models += ["MultiTLearnerGBM", "MultiDRLearnerGBM"]
    if treatment_type == "continuous":
        models += ["DoseResponseGBM", "guarded DRNet/VCNet"]
    if "LLM" in scenario:
        models += ["LLM routing uplift baseline", "multi-action routing policy", "contextual bandit after OPE"]

    evaluation_plan = [
        "data diagnostics: treatment balance, feature balance, overlap, leakage scan",
        "offline metrics: QINI, AUUC, uplift@K, calibration, bootstrap CI",
        "business metrics: policy value, incremental profit, ROI/iROAS, budget sensitivity",
        "trust checks: sensitivity/refutation, nuisance diagnostics, evidence manifest, environment snapshot",
    ]
    if _has_any(text, ["ope", "ips", "snips", "dr", "off-policy", "bandit"]):
        evaluation_plan += ["OPE: IPS, SNIPS, DR, coverage, effective sample size / ESS, clipping sensitivity"]
    if "benchmark" in scenario:
        evaluation_plan += ["benchmark: multi-seed leaderboard, run diff, failure attribution, paper/open-source baselines"]
    if "launch guardrail" in scenario:
        evaluation_plan += [
            "warning audit: rank high-severity rows by negative ROI, weak overlap, sensitivity fail, metric conflict, and deep-vs-baseline loss",
            "promotion card: separate safe claim, unsafe claim, owner, before-shadow, before-A/B, and before-ramp requirements",
            "pilot readiness: convert promotion and triage evidence into blocked/paper-review/shadow/monitor gates with rollback conditions",
            "pilot experiment plan: turn each readiness gate into a command, metrics, expected artifacts, pass gate, fail action, and claim-upgrade boundary",
            "pilot command smoke: execute sampled next-experiment commands with no-update-latest safe overrides before claiming runner evidence",
            "补实验: rerun larger multi-seed benchmark, inspect nuisance/overlap, add holdout/shadow validation, and only promote after readiness gate clears",
        ]
    if _has_any(text, ["claim", "safe claim", "unsafe claim", "过度包装", "证据层级", "源码证明", "凭什么"]):
        evaluation_plan += [
            "algorithm claim ledger: verify formula, code refs, evidence refs, benchmark verdict, license gate, safe claim and unsafe claim before stronger wording",
        ]
    if _has_any(text, ["paper reproduction", "reproduction gap", "论文复现", "复现程度", "没有复制代码", "复现差距"]):
        evaluation_plan += [
            "paper reproduction gap ledger: separate cited paper, external repo, local implementation scope, license risk, benchmark proof level and remaining gaps",
        ]

    rollout_plan = ["offline benchmark", "fixed policy threshold", "shadow scoring", "small traffic A/B or holdout", "guardrail monitoring", "ramp-up with rollback"]
    if "marketplace" in scenario:
        rollout_plan += ["cluster randomized / geo holdout / switchback"]
    if "LLM" in scenario:
        rollout_plan += ["randomized exploration bucket", "budget pacing", "judge calibration", "drift monitoring"]
    if _has_any(text, ["bandit", "contextual bandit", "预算节奏", "budget pacing"]):
        rollout_plan += ["contextual bandit only after OPE and budget pacing guardrails are stable"]
    if "launch guardrail" in scenario:
        rollout_plan += [
            "do not promote high-severity warning rows to online treatment until root cause and owner are documented",
            "use holdout / geo holdout / switchback for risks that cannot be resolved offline",
        ]

    modes = ["算法解释", "工程架构", "业务 ROI", "上线试点", "面试话术"]
    if _has_any(text, ["5分钟", "15分钟", "30分钟", "演示"]):
        modes += ["5分钟", "15分钟", "30分钟"]

    evidence_ids = ["evidence_index", "model_matrix", "policy"]
    if _has_any(text, ["agent", "copilot", "rubric"]):
        evidence_ids += ["agent_v2", "agent_regression"]
    if "benchmark" in scenario:
        evidence_ids += ["benchmark", "benchmark_report"]
    if "launch guardrail" in scenario or _has_any(text, ["warning", "guardrail", "readiness", "pilot", "pilot experiment", "pilot command", "command smoke", "不能上线", "门禁", "风险审计", "warning triage", "风险分流", "claim boundary", "action lane", "smoke-only", "promotion", "上线评审", "放行", "阻断", "试点", "试点实验", "命令 smoke", "命令真的跑"]):
        evidence_ids += ["warning_audit", "promotion_cards", "warning_triage", "pilot_readiness", "pilot_experiment_plan", "pilot_experiment_command_smoke", "benchmark", "benchmark_report", "online"]
    if _has_any(text, ["dataset", "data", "数据"]):
        evidence_ids += ["dataset_registry"]
    if _has_any(text, ["ope", "ips", "snips", "dr"]):
        evidence_ids += ["ope"]
    if _has_any(text, ["nuisance", "propensity", "orthogonal", "有偏", "r learner", "dr learner"]):
        evidence_ids += ["nuisance"]
    if _has_any(text, ["budget", "roi", "阈值", "profit", "预算"]):
        evidence_ids += ["policy_code"]
    if _has_any(text, ["deep", "tarnet", "cfrnet", "dragonnet", "efin", "descn"]):
        evidence_ids += ["deep_loss", "algorithm_claim_ledger"]
    if _has_any(text, ["claim", "safe claim", "unsafe claim", "过度包装", "证据层级", "源码证明", "凭什么"]):
        evidence_ids += ["algorithm_claim_ledger", "benchmark", "benchmark_report", "promotion_cards"]
    if _has_any(text, ["paper reproduction", "reproduction gap", "论文复现", "复现程度", "没有复制代码", "复现差距"]):
        evidence_ids += ["paper_reproduction_gap_ledger", "algorithm_claim_ledger", "benchmark", "benchmark_report"]
    if _has_any(text, ["promotion matrix", "model evidence", "模型推广", "推广矩阵", "promotion tier", "blocked claim", "能讲到什么程度"]):
        evidence_ids += ["model_evidence_promotion_matrix", "algorithm_claim_ledger", "paper_reproduction_gap_ledger", "promotion_cards"]
    if _has_any(text, ["upgrade recipe", "recipe card", "升级实验", "下一步实验", "next experiment", "pass gate", "fail action", "claim upgrade", "safe smoke command", "no-update-latest", "怎么升级"]):
        evidence_ids += ["model_upgrade_recipe_cards", "pilot_experiment_plan", "pilot_experiment_command_smoke", "model_evidence_promotion_matrix", "algorithm_claim_ledger", "benchmark"]
    if _has_any(text, ["telemetry gap", "loss telemetry", "可观测性", "representation telemetry", "propensity calibration", "attention weights", "head consistency", "失败归因"]):
        evidence_ids += ["deep_model_telemetry_gap_matrix", "model_upgrade_recipe_cards", "deep_loss", "algorithm_claim_ledger"]
    if _has_any(text, ["telemetry smoke", "basic telemetry", "loss history", "prediction spread", "top-k known-cate", "advanced hook"]):
        evidence_ids += ["deep_model_telemetry_smoke", "deep_model_telemetry_gap_matrix", "model_upgrade_recipe_cards"]
    if _has_any(text, ["forward hook", "hook contract", "tensor keys", "artifact files", "phi_x", "e_hat", "tarreg", "attention", "head export", "源码改造"]):
        evidence_ids += ["deep_model_forward_hook_contracts", "deep_model_telemetry_smoke", "deep_model_telemetry_gap_matrix"]
    if _has_any(text, ["forward hook smoke", "sidecar artifacts", "tensor export", "heads", "constraints", "实际导出"]):
        evidence_ids += ["deep_model_forward_hook_smoke", "deep_model_forward_hook_contracts", "deep_model_telemetry_smoke"]
    if _has_any(text, ["hook training bridge", "trainer telemetry", "per-epoch", "epoch columns", "optional collector", "训练桥接", "训练期可观测"]):
        evidence_ids += ["deep_model_hook_training_bridge", "deep_model_forward_hook_smoke", "deep_loss"]
    if _has_any(text, ["trainer collector", "collector smoke", "per-epoch telemetry", "advanced telemetry", "训练期采集", "epoch telemetry"]):
        evidence_ids += ["deep_model_trainer_collector_smoke", "deep_model_hook_training_bridge", "deep_model_forward_hook_smoke"]
    if _has_any(text, ["telemetry benchmark", "benchmark linkage", "遥测和benchmark", "胜负归因", "next ablation", "failure hypothesis", "为什么赢", "为什么输"]):
        evidence_ids += ["deep_model_telemetry_benchmark_linkage", "deep_model_trainer_collector_smoke", "benchmark", "benchmark_report"]
    if _has_any(
        text,
        [
            "ablation gate",
            "telemetry ablation",
            "variant coverage",
            "missing variants",
            "实验门禁",
            "消融门禁",
            "消融变体",
            "升级 claim",
            "efin_no_attention",
            "efin_path_regularized",
            "descn_no_constraint",
            "ablation interpretation",
            "消融解读",
            "default vs variant",
            "metric movement",
            "variant scorecard",
            "ablation promotion",
            "ablation promotion matrix",
            "claim tier",
            "promotion decision",
            "default or variant",
            "variant promotion",
            "claim 升级",
            "变体能不能讲",
            "ablation next experiment",
            "next experiment plan",
            "recommended command",
            "metrics to watch",
            "command contract",
            "runnable fallback",
            "unsupported flags",
            "runner cli",
            "runner adapter",
            "adapter 规划",
            "真的能跑",
            "命令哪些",
            "哪些真的能跑",
            "下一步实验",
            "面试追问",
            "可执行命令",
            "命令契约",
            "fallback smoke",
            "command smoke",
            "recommended command smoke",
            "command smoke coverage",
            "smoke coverage",
            "coverage gate",
            "coverage diff",
            "smoke coverage diff",
            "gate delta",
            "review_required -> ok",
            "覆盖率变化",
            "contract ready not smoked",
            "contract-ready",
            "已经 smoke",
            "只是 contract",
            "runner-native command",
            "--models",
            "--include-variants",
            "--emit-battle-cards",
            "no-update-latest",
            "真的执行",
        ],
    ):
        evidence_ids += [
            "deep_model_telemetry_ablation_gate",
            "deep_model_ablation_interpretation",
            "deep_model_ablation_promotion_matrix",
            "deep_model_ablation_next_experiment_plan",
            "deep_model_ablation_command_contract",
            "deep_model_ablation_command_contract_smoke",
            "deep_model_ablation_recommended_command_smoke",
            "deep_model_ablation_command_smoke_coverage",
            "deep_model_ablation_command_smoke_coverage_diff",
            "deep_model_telemetry_benchmark_linkage",
            "model_upgrade_recipe_cards",
            "benchmark",
        ]
    if "LLM" in scenario:
        evidence_ids += ["llm_routing", "ope"]
    if _has_any(text, ["上线", "online", "ab", "holdout", "shadow", "ramp"]):
        evidence_ids += ["online"]
    if risks:
        evidence_ids += ["failure_modes"]
    evidence_ids += ["evidence_store"]

    return AgentIntent(
        scenario=scenario,
        question_type="causal decision review",
        treatment_type=treatment_type,
        treatment=treatment,
        outcome=outcome,
        unit=unit,
        time_window=time_window,
        confounders=_dedupe(confounders)[:8],
        cost_value=cost_value,
        guardrails=_dedupe(guardrails),
        risks=_dedupe(risks),
        model_families=_dedupe(models),
        evaluation_plan=_dedupe(evaluation_plan),
        rollout_plan=_dedupe(rollout_plan),
        answer_modes=_dedupe(modes),
        evidence_ids=_dedupe(evidence_ids),
    )


def selected_evidence(intent: AgentIntent, limit: int = 8) -> list[EvidenceItem]:
    by_id = {item.id: item for item in EVIDENCE_CATALOG}
    rows = [by_id[item] for item in intent.evidence_ids if item in by_id]
    if len(rows) < limit:
        for item in EVIDENCE_CATALOG:
            if item not in rows:
                rows.append(item)
            if len(rows) >= limit:
                break
    return rows[:limit]


def _evidence_lines(items: list[EvidenceItem]) -> str:
    lines = []
    for item in items:
        path = Path(item.path)
        exists = "exists" if (path if path.is_absolute() else REPO_ROOT / path).exists() else "planned/missing"
        lines.append(f"- `{item.path}` ({item.kind}, {exists}): {item.claim}")
    return "\n".join(lines)


def _read_repo_json(path: str) -> dict[str, Any]:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    if not candidate.is_file():
        return {}
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _warning_audit_snapshot(intent: AgentIntent) -> str:
    if "warning_audit" not in intent.evidence_ids:
        return ""
    payload = _read_repo_json("reports/regression_warning_audit_latest.json")
    summary = payload.get("summary") or {}
    warnings = payload.get("warnings") or []
    if not summary or not isinstance(warnings, list):
        return "- `reports/regression_warning_audit_latest.json` is missing or not refreshed; rerun `scripts/generate_regression_warning_audit.py`."

    severity = summary.get("severity_counts") or {}
    domains = summary.get("domain_counts") or {}
    severity_text = ", ".join(f"{key}={value}" for key, value in severity.items()) or "NA"
    domain_text = ", ".join(f"{key}={value}" for key, value in domains.items()) or "NA"
    high_rows = [row for row in warnings if row.get("severity") == "high"][:3]

    lines = [
        f"- Warning audit live snapshot: launch_gate=`{summary.get('launch_gate', 'NA')}`, total_warnings=`{summary.get('total_warnings', len(warnings))}`, severity=`{severity_text}`.",
        f"- Domain concentration: {domain_text}.",
    ]
    if high_rows:
        lines.append("- Top high-severity blockers:")
        for row in high_rows:
            entity = row.get("entity", "unknown")
            signal = row.get("signal", "signal")
            value = row.get("value", "NA")
            decision = row.get("promotion_decision", "block_promotion")
            owner = row.get("owner_role", "owner")
            action = row.get("next_experiment") or row.get("recommended_action") or "Document owner, root cause, and follow-up validation."
            lines.append(f"  - `{entity}`: `{signal}={value}`, decision=`{decision}`, owner=`{owner}` -> {action}")
    lines.append("- Launch rule: high-severity rows are review blockers and 不能上线; they require owner, root cause, follow-up experiment evidence, and readiness-gate clearance.")
    return "\n".join(lines)


def _promotion_cards_snapshot(intent: AgentIntent) -> str:
    if "promotion_cards" not in intent.evidence_ids:
        return ""
    payload = _read_repo_json("reports/promotion_launch_cards_latest.json")
    summary = payload.get("summary") or {}
    cards = payload.get("cards") or []
    if not summary or not isinstance(cards, list):
        return "- `reports/promotion_launch_cards_latest.json` is missing; rerun `scripts/generate_promotion_launch_cards.py` after warning audit."

    decisions = summary.get("decision_counts") or {}
    decision_text = ", ".join(f"{key}={value}" for key, value in decisions.items()) or "NA"
    lines = [
        f"- Promotion card snapshot: launch_gate=`{summary.get('launch_gate', 'NA')}`, cards=`{summary.get('total_cards', len(cards))}`, decisions=`{decision_text}`.",
    ]
    top_cards = cards[:3]
    if top_cards:
        lines.append("- Top launch review cards:")
        for card in top_cards:
            entity = card.get("entity", "unknown")
            decision = card.get("overall_decision", "monitor")
            owner = card.get("owner_role", "owner")
            safe_claim = card.get("safe_claim", "")
            unsafe_claim = card.get("unsafe_claim", "")
            next_experiment = (card.get("next_experiments") or ["Document root cause and validation."])[0]
            lines.append(
                f"  - `{entity}`: decision=`{decision}`, owner=`{owner}`; safe claim={safe_claim}; unsafe claim={unsafe_claim}; next={next_experiment}"
            )
    return "\n".join(lines)


def _warning_triage_snapshot(intent: AgentIntent) -> str:
    if "warning_triage" not in intent.evidence_ids:
        return ""
    payload = _read_repo_json("reports/regression_warning_triage_latest.json")
    summary = payload.get("summary") or {}
    rows = payload.get("rows") or []
    if not summary or not isinstance(rows, list):
        return "- `reports/regression_warning_triage_latest.json` is missing; rerun `scripts/generate_regression_warning_triage.py` after warning audit and promotion cards."

    family_counts = summary.get("risk_family_counts") or {}
    family_text = ", ".join(f"{key}={value}" for key, value in family_counts.items()) or "NA"
    lines = [
        f"- Warning triage snapshot: gate=`{summary.get('triage_gate', 'NA')}`, rows=`{summary.get('warning_rows', len(rows))}`, smoke_only=`{summary.get('smoke_only_rows', 'NA')}`, launch_blocking=`{summary.get('launch_blocking_rows', 'NA')}`.",
        f"- Risk-family split: {family_text}.",
    ]
    for row in rows[:4]:
        lines.append(
            f"  - `{row.get('entity')}` / `{row.get('signal')}`: family=`{row.get('risk_family')}`, boundary=`{row.get('claim_boundary')}`, action_lane=`{row.get('action_lane')}`, launch_blocking={row.get('launch_blocking')}."
        )
    lines.append("- Claim boundary: triage does not make warnings disappear; it decides what can be claimed, what blocks pilot, and which experiment must run next.")
    return "\n".join(lines)


def _pilot_readiness_snapshot(intent: AgentIntent) -> str:
    if "pilot_readiness" not in intent.evidence_ids:
        return ""
    payload = _read_repo_json("reports/pilot_readiness_scorecard_latest.json")
    summary = payload.get("summary") or {}
    rows = payload.get("scorecards") or []
    if not summary or not isinstance(rows, list):
        return "- `reports/pilot_readiness_scorecard_latest.json` is missing; rerun `scripts/generate_pilot_readiness_scorecard.py` after warning triage."

    gate_counts = summary.get("gate_counts") or {}
    gate_text = ", ".join(f"{key}={value}" for key, value in gate_counts.items()) or "NA"
    lines = [
        f"- Pilot readiness snapshot: pilot_gate=`{summary.get('pilot_gate', 'NA')}`, scorecards=`{summary.get('scorecards', len(rows))}`, gates=`{gate_text}`.",
        f"- Candidate split: blocked=`{summary.get('blocked', 'NA')}`, paper_review=`{summary.get('paper_review_candidates', 'NA')}`, shadow=`{summary.get('shadow_candidates', 'NA')}`, monitor=`{summary.get('monitor', 'NA')}`.",
    ]
    for row in rows[:4]:
        lines.append(
            f"  - `{row.get('entity')}`: gate=`{row.get('pilot_gate')}`, score=`{row.get('readiness_score')}`, grade=`{row.get('evidence_grade')}`, next_gate=`{row.get('next_gate')}`, rollback=`{'; '.join(str(item) for item in (row.get('rollback_conditions') or [])[:2])}`."
        )
    lines.append("- Pilot boundary: this scorecard is not launch approval; it says whether the next safe step is offline repair, paper review, shadow scoring, or monitored holdout.")
    return "\n".join(lines)


def _pilot_experiment_plan_snapshot(intent: AgentIntent) -> str:
    if "pilot_experiment_plan" not in intent.evidence_ids:
        return ""
    payload = _read_repo_json("reports/pilot_experiment_plan_latest.json")
    summary = payload.get("summary") or {}
    rows = payload.get("rows") or []
    if not summary or not isinstance(rows, list):
        return "- `reports/pilot_experiment_plan_latest.json` is missing; rerun `scripts/generate_pilot_experiment_plan.py` after pilot readiness scorecard."

    type_counts = summary.get("experiment_type_counts") or {}
    type_text = ", ".join(f"{key}={value}" for key, value in type_counts.items()) or "NA"
    lines = [
        f"- Pilot experiment plan snapshot: experiments=`{summary.get('experiments', len(rows))}`, P0=`{summary.get('p0_rows', 'NA')}`, P1=`{summary.get('p1_rows', 'NA')}`, P2=`{summary.get('p2_rows', 'NA')}`.",
        f"- Experiment types: {type_text}.",
    ]
    for row in rows[:4]:
        metrics = ", ".join(str(item) for item in (row.get("metrics_to_watch") or [])[:4])
        artifacts = ", ".join(str(item) for item in (row.get("expected_artifacts") or [])[:3])
        lines.append(
            f"  - `{row.get('entity')}`: priority=`{row.get('priority')}`, type=`{row.get('experiment_type')}`, recommended command=`{row.get('recommended_command')}`, metrics={metrics}; artifacts={artifacts}; pass={row.get('pass_gate')}; fail={row.get('fail_action')}."
        )
    lines.append("- Experiment boundary: this is not proof that the model is ready; it is the runnable evidence contract that decides whether a claim can be upgraded or must stay offline-only.")
    return "\n".join(lines)


def _pilot_experiment_command_smoke_snapshot(intent: AgentIntent) -> str:
    if "pilot_experiment_command_smoke" not in intent.evidence_ids:
        return ""
    payload = _read_repo_json("reports/pilot_experiment_command_smoke_latest.json")
    summary = payload.get("summary") or {}
    rows = payload.get("rows") or []
    if not summary or not isinstance(rows, list):
        return "- `reports/pilot_experiment_command_smoke_latest.json` is missing; rerun `scripts/smoke_pilot_experiment_commands.py --limit 3` after pilot experiment plan."

    lines = [
        f"- Pilot command smoke snapshot: smoke_rows=`{summary.get('smoke_rows', len(rows))}`, ok_rows=`{summary.get('ok_rows', 'NA')}`, failed_rows=`{summary.get('failed_rows', 'NA')}`.",
        f"- Smoke type split: `{json.dumps(summary.get('experiment_type_counts') or {}, ensure_ascii=False, sort_keys=True)}`.",
    ]
    for row in rows[:4]:
        lines.append(
            f"  - `{row.get('entity')}`: type=`{row.get('experiment_type')}`, status=`{row.get('status')}`, safe smoke command=`{row.get('safe_smoke_command')}`, runner artifact=`{row.get('runner_manifest') or row.get('runner_json') or row.get('runner_csv')}`."
        )
    lines.append("- Smoke boundary: command smoke proves runner executability with tiny rows and isolated outputs; full pilot claim still needs larger benchmark/OPE/holdout evidence.")
    return "\n".join(lines)


def _algorithm_claim_ledger_snapshot(intent: AgentIntent) -> str:
    if "algorithm_claim_ledger" not in intent.evidence_ids:
        return ""
    payload = _read_repo_json("reports/algorithm_claim_ledger_latest.json")
    summary = payload.get("summary") or {}
    rows = payload.get("rows") or []
    if not summary or not isinstance(rows, list):
        return "- `reports/algorithm_claim_ledger_latest.json` is missing; rerun `scripts/generate_algorithm_claim_ledger.py`."

    claim_types = summary.get("claim_type_counts") or {}
    evidence_levels = summary.get("evidence_level_counts") or {}
    claim_text = ", ".join(f"{key}={value}" for key, value in claim_types.items()) or "NA"
    level_text = ", ".join(f"{key}={value}" for key, value in evidence_levels.items()) or "NA"
    lines = [
        f"- Algorithm claim ledger snapshot: models=`{summary.get('models', 'NA')}`, claims=`{summary.get('claims', len(rows))}`, claim_types=`{claim_text}`.",
        f"- Evidence levels: {level_text}.",
    ]
    for row in rows[:3]:
        lines.append(
            f"  - `{row.get('claim_id')}`: evidence_level=`{row.get('evidence_level')}`, verdict=`{row.get('benchmark_verdict')}`, safe={row.get('safe_claim')}"
        )
    lines.append("- Claim rule: if the ledger says source/smoke/objective only, the answer must not be upgraded to production lift or SOTA.")
    return "\n".join(lines)


def _paper_reproduction_gap_snapshot(intent: AgentIntent) -> str:
    if "paper_reproduction_gap_ledger" not in intent.evidence_ids:
        return ""
    payload = _read_repo_json("reports/paper_reproduction_gap_ledger_latest.json")
    summary = payload.get("summary") or {}
    rows = payload.get("rows") or []
    if not summary or not isinstance(rows, list):
        return "- `reports/paper_reproduction_gap_ledger_latest.json` is missing; rerun `scripts/generate_paper_reproduction_gap_ledger.py`."

    proof_levels = summary.get("proof_level_counts") or {}
    decisions = summary.get("adoption_decision_counts") or {}
    level_text = ", ".join(f"{key}={value}" for key, value in proof_levels.items()) or "NA"
    decision_text = ", ".join(f"{key}={value}" for key, value in decisions.items()) or "NA"
    lines = [
        f"- Paper reproduction gap snapshot: models=`{summary.get('models', 'NA')}`, source_rows=`{summary.get('source_rows', len(rows))}`, proof_levels=`{level_text}`.",
        f"- Adoption decisions: {decision_text}.",
    ]
    for row in rows[:3]:
        lines.append(
            f"  - `{row.get('model')}` / `{row.get('source_name')}`: license=`{row.get('source_license')}`, proof_level=`{row.get('proof_level')}`, adoption=`{row.get('adoption_decision')}`."
        )
    lines.append("- Reproduction rule: cited paper/reference code is not a full reproduction unless local code, benchmark proof and license gate all support that claim.")
    return "\n".join(lines)


def _model_evidence_promotion_matrix_snapshot(intent: AgentIntent) -> str:
    if "model_evidence_promotion_matrix" not in intent.evidence_ids:
        return ""
    payload = _read_repo_json("reports/model_evidence_promotion_matrix_latest.json")
    summary = payload.get("summary") or {}
    rows = payload.get("rows") or []
    if not summary or not isinstance(rows, list):
        return "- `reports/model_evidence_promotion_matrix_latest.json` is missing; rerun `scripts/generate_model_evidence_promotion_matrix.py`."

    stage_counts = summary.get("stage_counts") or {}
    tier_counts = summary.get("promotion_tier_counts") or {}
    stage_text = ", ".join(f"{key}={value}" for key, value in stage_counts.items()) or "NA"
    tier_text = ", ".join(f"{key}={value}" for key, value in tier_counts.items()) or "NA"
    ranked = sorted(rows, key=lambda row: (-int(row.get("evidence_score") or 0), str(row.get("model"))))[:3]
    lines = [
        f"- Model evidence promotion matrix snapshot: models=`{summary.get('models', len(rows))}`, stages=`{stage_text}`.",
        f"- Promotion tiers: {tier_text}.",
    ]
    for row in ranked:
        lines.append(
            f"  - `{row.get('model')}`: stage=`{row.get('evidence_stage')}`, tier=`{row.get('promotion_tier')}`, score=`{row.get('evidence_score')}`, blocked claim={row.get('blocked_claim')}"
        )
    lines.append("- Promotion rule: T0/T1 is interview/benchmark readiness, not production launch; stronger claims need OPE, holdout/A/B, guardrails and rollback evidence.")
    return "\n".join(lines)


def _model_upgrade_recipe_cards_snapshot(intent: AgentIntent) -> str:
    if "model_upgrade_recipe_cards" not in intent.evidence_ids:
        return ""
    payload = _read_repo_json("reports/model_upgrade_recipe_cards_latest.json")
    summary = payload.get("summary") or {}
    rows = payload.get("rows") or []
    if not summary or not isinstance(rows, list):
        return "- `reports/model_upgrade_recipe_cards_latest.json` is missing; rerun `scripts/generate_model_upgrade_recipe_cards.py`."

    goals = summary.get("upgrade_goal_counts") or {}
    goal_text = ", ".join(f"{key}={value}" for key, value in goals.items()) or "NA"
    lines = [
        f"- Model upgrade recipe cards snapshot: recipes=`{summary.get('recipes', len(rows))}`, models=`{summary.get('models', 'NA')}`, goals=`{goal_text}`.",
    ]
    for row in rows[:3]:
        lines.append(
            f"  - `{row.get('recipe_id')}`: pass gate={row.get('pass_gate')}; fail action={row.get('fail_action')}; expected artifacts={', '.join(row.get('expected_artifacts') or [])}."
        )
    lines.append("- Recipe rule: a recipe card is a next experiment contract; it upgrades evidence only after the expected artifacts are generated and validators pass.")
    return "\n".join(lines)


def _deep_model_telemetry_gap_matrix_snapshot(intent: AgentIntent) -> str:
    if "deep_model_telemetry_gap_matrix" not in intent.evidence_ids:
        return ""
    payload = _read_repo_json("reports/deep_model_telemetry_gap_matrix_latest.json")
    summary = payload.get("summary") or {}
    rows = payload.get("rows") or []
    if not summary or not isinstance(rows, list):
        return "- `reports/deep_model_telemetry_gap_matrix_latest.json` is missing; rerun `scripts/generate_deep_model_telemetry_gap_matrix.py`."

    priorities = summary.get("priority_counts") or {}
    families = summary.get("telemetry_family_counts") or {}
    priority_text = ", ".join(f"{key}={value}" for key, value in priorities.items()) or "NA"
    family_text = ", ".join(f"{key}={value}" for key, value in families.items()) or "NA"
    p0_rows = [row for row in rows if row.get("priority") == "P0"][:3]
    lines = [
        f"- Deep model telemetry gap matrix snapshot: models=`{summary.get('models', 'NA')}`, telemetry_rows=`{summary.get('telemetry_rows', len(rows))}`, priorities=`{priority_text}`.",
        f"- Telemetry families: {family_text}.",
    ]
    for row in p0_rows:
        lines.append(
            f"  - `{row.get('model')}` / `{row.get('loss_term')}`: missing={row.get('missing_telemetry')}; pass gate={row.get('pass_gate')}."
        )
    lines.append("- Telemetry rule: a loss term is not a benchmark claim until the needed representation/propensity/attention/head telemetry is exported and linked to PEHE/QINI/policy movement.")
    return "\n".join(lines)


def _deep_model_telemetry_smoke_snapshot(intent: AgentIntent) -> str:
    if "deep_model_telemetry_smoke" not in intent.evidence_ids:
        return ""
    payload = _read_repo_json("reports/deep_model_telemetry_smoke_latest.json")
    summary = payload.get("summary") or {}
    rows = payload.get("rows") or []
    if not summary or not isinstance(rows, list):
        return "- `reports/deep_model_telemetry_smoke_latest.json` is missing; rerun `scripts/smoke_deep_model_telemetry.py`."

    lines = [
        f"- Deep model telemetry smoke snapshot: models=`{summary.get('models', 'NA')}`, rows=`{summary.get('rows', len(rows))}`, basic_pass=`{summary.get('basic_telemetry_pass', 'NA')}`, advanced_hook_gaps=`{summary.get('advanced_hook_gaps', 'NA')}`.",
        "- It explicitly verifies loss history, prediction spread, metrics and evidence manifest while marking each advanced hook as still guarded.",
    ]
    for row in rows[:3]:
        lines.append(
            f"  - `{row.get('model')}`: train_loss_delta=`{row.get('train_loss_delta')}`, top10_true_uplift_gain=`{row.get('top10_true_uplift_gain')}`, advanced_hook=`{row.get('advanced_hook_status')}`."
        )
    lines.append("- Smoke rule: basic telemetry proves the run artifact pipe is readable; each advanced hook such as phi/e_hat/attention/head export is still required for stronger deep-model claims.")
    return "\n".join(lines)


def _deep_model_forward_hook_contracts_snapshot(intent: AgentIntent) -> str:
    if "deep_model_forward_hook_contracts" not in intent.evidence_ids:
        return ""
    payload = _read_repo_json("reports/deep_model_forward_hook_contracts_latest.json")
    summary = payload.get("summary") or {}
    rows = payload.get("rows") or []
    if not summary or not isinstance(rows, list):
        return "- `reports/deep_model_forward_hook_contracts_latest.json` is missing; rerun `scripts/generate_deep_model_forward_hook_contracts.py`."

    priorities = summary.get("priority_counts") or {}
    types = summary.get("contract_type_counts") or {}
    priority_text = ", ".join(f"{key}={value}" for key, value in priorities.items()) or "NA"
    type_text = ", ".join(f"{key}={value}" for key, value in types.items()) or "NA"
    lines = [
        f"- Deep model forward hook contracts snapshot: models=`{summary.get('models', 'NA')}`, contracts=`{summary.get('contracts', len(rows))}`, P0=`{summary.get('p0_contracts', 'NA')}`, guarded=`{summary.get('guarded_contracts', 'NA')}`.",
        f"- Contract priorities: {priority_text}; types: {type_text}.",
        "- Contract rule: a source refactor is only promotable after tensor keys, artifact files, smoke tests, pass gates, fallback behavior and backward compatibility are all explicit.",
    ]
    for row in rows[:3]:
        keys = ", ".join(str(item) for item in row.get("expected_tensor_keys") or [])
        artifacts = ", ".join(str(item) for item in (row.get("artifact_files") or [])[:2])
        lines.append(
            f"  - `{row.get('contract_id')}`: tensor keys={keys}; artifact files={artifacts}; pass gate={row.get('pass_gate')}."
        )
    lines.append("- Interview boundary: the contract proves the refactor standard, not that advanced hooks are already producing paper-level evidence.")
    return "\n".join(lines)


def _deep_model_forward_hook_smoke_snapshot(intent: AgentIntent) -> str:
    if "deep_model_forward_hook_smoke" not in intent.evidence_ids:
        return ""
    payload = _read_repo_json("reports/deep_model_forward_hook_smoke_latest.json")
    summary = payload.get("summary") or {}
    rows = payload.get("rows") or []
    if not summary or not isinstance(rows, list):
        return "- `reports/deep_model_forward_hook_smoke_latest.json` is missing; rerun `scripts/smoke_deep_model_forward_hooks.py`."

    lines = [
        f"- Deep model forward hook smoke snapshot: models=`{summary.get('models', 'NA')}`, contracts=`{summary.get('contracts', len(rows))}`, ok_contracts=`{summary.get('ok_contracts', 'NA')}`, sidecar_artifacts=`{summary.get('artifact_files', 'NA')}`.",
        f"- Artifact dir: `{payload.get('artifact_dir', 'NA')}`.",
        "- It actually exports tensor outputs such as phi_x, e_hat, tarreg components, attention, heads and constraints as sidecar artifacts while preserving default model APIs.",
    ]
    for row in rows[:8]:
        keys = ", ".join(str(item) for item in row.get("tensor_keys_exported") or [])
        artifacts = ", ".join(str(item) for item in (row.get("artifact_files") or [])[:2])
        lines.append(
            f"  - `{row.get('contract_id')}`: status={row.get('status')}; tensor export={keys}; sidecar artifacts={artifacts}."
        )
    lines.append("- Smoke boundary: this is executable hook evidence; per-epoch trainer telemetry and benchmark-linked attribution are still the next promotion step.")
    return "\n".join(lines)


def _deep_model_hook_training_bridge_snapshot(intent: AgentIntent) -> str:
    if "deep_model_hook_training_bridge" not in intent.evidence_ids:
        return ""
    payload = _read_repo_json("reports/deep_model_hook_training_bridge_latest.json")
    summary = payload.get("summary") or {}
    rows = payload.get("rows") or []
    if not summary or not isinstance(rows, list):
        return "- `reports/deep_model_hook_training_bridge_latest.json` is missing; rerun `scripts/generate_deep_model_hook_training_bridge.py`."

    lines = [
        f"- Deep model hook training bridge snapshot: models=`{summary.get('models', 'NA')}`, bridges=`{summary.get('bridges', len(rows))}`, bridge_ready=`{summary.get('bridge_ready', 'NA')}`, proposed_epoch_columns=`{summary.get('proposed_epoch_columns', 'NA')}`.",
        "- It joins forward-hook sidecar artifacts with existing training run history/manifests, then defines optional trainer collectors and per-epoch artifact schemas.",
    ]
    for row in rows[:8]:
        columns = ", ".join(str(item) for item in row.get("proposed_epoch_columns") or [])
        lines.append(
            f"  - `{row.get('contract_id')}`: status={row.get('bridge_status')}; epoch columns={columns}; pass gate={row.get('pass_gate')}."
        )
    lines.append("- Migration boundary: bridge-ready means the evidence path is designed and joinable, while actual per-epoch collectors and ablations are the next implementation step.")
    return "\n".join(lines)


def _deep_model_trainer_collector_smoke_snapshot(intent: AgentIntent) -> str:
    if "deep_model_trainer_collector_smoke" not in intent.evidence_ids:
        return ""
    payload = _read_repo_json("reports/deep_model_trainer_collector_smoke_latest.json")
    summary = payload.get("summary") or {}
    rows = payload.get("model_rows") or []
    if not summary or not isinstance(rows, list):
        return "- `reports/deep_model_trainer_collector_smoke_latest.json` is missing; rerun `scripts/smoke_deep_model_trainer_collectors.py`."

    lines = [
        f"- Deep model trainer collector smoke snapshot: models=`{summary.get('models', 'NA')}`, ok_models=`{summary.get('ok_models', 'NA')}`, epoch_rows=`{summary.get('epoch_rows', 'NA')}`, contracts_covered=`{summary.get('contracts_covered', 'NA')}`.",
        f"- Artifact dir: `{payload.get('artifact_dir', 'NA')}`.",
        "- It runs tiny training loops and emits per-epoch advanced telemetry sidecars while keeping default model APIs unchanged.",
    ]
    for row in rows[:8]:
        columns = ", ".join(str(item) for item in (row.get("collected_epoch_columns") or [])[:6])
        lines.append(
            f"  - `{row.get('model')}`: status={row.get('status')}; contracts={', '.join(row.get('contracts_covered') or [])}; columns={columns}."
        )
    lines.append("- Claim boundary: this proves executable trainer collection, not production monitoring or paper-level ablation wins.")
    return "\n".join(lines)


def _deep_model_telemetry_benchmark_linkage_snapshot(intent: AgentIntent) -> str:
    if "deep_model_telemetry_benchmark_linkage" not in intent.evidence_ids:
        return ""
    payload = _read_repo_json("reports/deep_model_telemetry_benchmark_linkage_latest.json")
    summary = payload.get("summary") or {}
    rows = payload.get("rows") or []
    if not summary or not isinstance(rows, list):
        return "- `reports/deep_model_telemetry_benchmark_linkage_latest.json` is missing; rerun `scripts/generate_deep_model_telemetry_benchmark_linkage.py`."

    lines = [
        f"- Deep model telemetry-benchmark linkage snapshot: models=`{summary.get('models', 'NA')}`, linkage_rows=`{summary.get('linkage_rows', len(rows))}`, link_ready=`{summary.get('link_ready', 'NA')}`.",
        "- It joins trainer collector metrics with paper scorecards, tuned battle verdicts, baseline context, failure hypotheses and next ablations.",
    ]
    for row in rows[:4]:
        lines.append(
            f"  - `{row.get('contract_id')}`: family={row.get('telemetry_family')}; paper={row.get('paper_verdict')}; tuned={row.get('tuned_verdict')}; safe claim={row.get('safe_claim')}; next ablation={row.get('next_ablation')}."
        )
    lines.append("- Claim boundary: linkage proves reviewability and separates safe claim from blocked claim; causal attribution still requires the named next ablation to connect telemetry movement to PEHE/QINI/policy movement.")
    return "\n".join(lines)


def _deep_model_telemetry_ablation_gate_snapshot(intent: AgentIntent) -> str:
    if "deep_model_telemetry_ablation_gate" not in intent.evidence_ids:
        return ""
    payload = _read_repo_json("reports/deep_model_telemetry_ablation_gate_latest.json")
    summary = payload.get("summary") or {}
    rows = payload.get("rows") or []
    if not summary or not isinstance(rows, list):
        return "- `reports/deep_model_telemetry_ablation_gate_latest.json` is missing; rerun `scripts/generate_deep_model_telemetry_ablation_gate.py`."

    lines = [
        f"- Deep model telemetry ablation gate snapshot: models=`{summary.get('models', 'NA')}`, gates=`{summary.get('gates', len(rows))}`, missing_variant_rows=`{summary.get('missing_variant_rows', 'NA')}`.",
        f"- Decision counts: `{json.dumps(summary.get('decision_counts') or {}, ensure_ascii=False, sort_keys=True)}`; coverage counts: `{json.dumps(summary.get('coverage_counts') or {}, ensure_ascii=False, sort_keys=True)}`.",
        "- The ablation gate is the reviewer contract after linkage: variant coverage, metrics to watch, pass gate, fail action, safe claim and blocked claim.",
    ]
    for row in rows[:8]:
        missing = ", ".join(str(item) for item in row.get("missing_variants") or []) or "none"
        required = ", ".join(str(item) for item in row.get("required_variants") or []) or "none"
        lines.append(
            f"  - `{row.get('contract_id')}`: ablation gate={row.get('ablation_family')}; required variants={required}; variant coverage={row.get('variant_coverage_status')}; missing variants={missing}; pass gate={row.get('pass_gate')}; blocked claim={row.get('blocked_claim')}."
        )
    lines.append("- Claim boundary: the gate makes the next experiment executable, but it does not promote a blocked claim until the required variants and metric movement pass.")
    return "\n".join(lines)


def _deep_model_ablation_interpretation_snapshot(intent: AgentIntent) -> str:
    if "deep_model_ablation_interpretation" not in intent.evidence_ids:
        return ""
    payload = _read_repo_json("reports/deep_model_ablation_interpretation_latest.json")
    summary = payload.get("summary") or {}
    rows = payload.get("rows") or []
    scorecards = payload.get("variant_scorecards") or []
    if not summary or not isinstance(rows, list):
        return "- `reports/deep_model_ablation_interpretation_latest.json` is missing; rerun `scripts/generate_deep_model_ablation_interpretation.py`."

    lines = [
        f"- Deep model ablation interpretation snapshot: rows=`{summary.get('rows', len(rows))}`, variants=`{summary.get('variants', 'NA')}`, support_rows=`{summary.get('support_rows', 'NA')}`, tradeoff_rows=`{summary.get('tradeoff_rows', 'NA')}`, blocked_rows=`{summary.get('blocked_rows', 'NA')}`.",
        f"- Verdict counts: `{json.dumps(summary.get('verdict_counts') or {}, ensure_ascii=False, sort_keys=True)}`.",
        "- This separates variant coverage from metric evidence: the gate says what must run; the interpretation says what the latest default-vs-variant deltas support.",
    ]
    for row in scorecards[:6]:
        lines.append(
            f"  - `{row.get('variant_id')}`: mechanism={row.get('mechanism')}; PEHE improved={row.get('pehe_improved_datasets')}/{row.get('datasets')}; QINI improved={row.get('qini_improved_datasets')}/{row.get('datasets')}; policy improved={row.get('policy_improved_datasets')}/{row.get('datasets')}; line={row.get('interview_line')}."
        )
    lines.append("- Claim boundary: a variant can be runnable and still blocked if PEHE/QINI/policy movement does not match the mechanism or if the tabular baseline remains stronger.")
    return "\n".join(lines)


def _deep_model_ablation_promotion_matrix_snapshot(intent: AgentIntent) -> str:
    if "deep_model_ablation_promotion_matrix" not in intent.evidence_ids:
        return ""
    payload = _read_repo_json("reports/deep_model_ablation_promotion_matrix_latest.json")
    summary = payload.get("summary") or {}
    rows = payload.get("rows") or []
    if not summary or not isinstance(rows, list):
        return "- `reports/deep_model_ablation_promotion_matrix_latest.json` is missing; rerun `scripts/generate_deep_model_ablation_promotion_matrix.py`."

    lines = [
        f"- Deep model ablation promotion matrix snapshot: variants=`{summary.get('variants', 'NA')}`, promotion_ready_rows=`{summary.get('promotion_ready_rows', 'NA')}`, tradeoff_rows=`{summary.get('tradeoff_rows', 'NA')}`, blocked_or_missing_rows=`{summary.get('blocked_or_missing_rows', 'NA')}`.",
        f"- Claim tiers: `{json.dumps(summary.get('claim_tier_counts') or {}, ensure_ascii=False, sort_keys=True)}`; decisions: `{json.dumps(summary.get('decision_counts') or {}, ensure_ascii=False, sort_keys=True)}`.",
        "- It turns ablation interpretation into interview-safe claim tiers and promotion decision rows: local mechanism evidence, tradeoff case study, coverage-only, or blocked claim.",
    ]
    for row in rows[:6]:
        lines.append(
            f"  - `{row.get('variant_id')}`: tier={row.get('claim_tier')}; decision={row.get('promotion_decision')}; score={row.get('promotion_score')}; recommendation={row.get('recommended_claim_boundary')}; blocked={row.get('blocked_claim')}."
        )
    lines.append("- Claim boundary: even a B-tier variant is not a production or SOTA claim; it only tells reviewers what can be said safely from the latest local evidence.")
    return "\n".join(lines)


def _deep_model_ablation_next_experiment_plan_snapshot(intent: AgentIntent) -> str:
    if "deep_model_ablation_next_experiment_plan" not in intent.evidence_ids:
        return ""
    payload = _read_repo_json("reports/deep_model_ablation_next_experiment_plan_latest.json")
    summary = payload.get("summary") or {}
    rows = payload.get("rows") or []
    if not summary or not isinstance(rows, list):
        return "- `reports/deep_model_ablation_next_experiment_plan_latest.json` is missing; rerun `scripts/generate_deep_model_ablation_next_experiment_plan.py`."

    lines = [
        f"- Deep model ablation next experiment snapshot: variants=`{summary.get('variants', 'NA')}`, P0=`{summary.get('p0_rows', 'NA')}`, P1=`{summary.get('p1_rows', 'NA')}`.",
        f"- Experiment types: `{json.dumps(summary.get('experiment_type_counts') or {}, ensure_ascii=False, sort_keys=True)}`.",
        "- This answers the interview follow-up after promotion matrix: each variant has a next experiment, recommended command, metrics to watch, pass gate and fail action.",
    ]
    for row in rows[:6]:
        lines.append(
            f"  - `{row.get('variant_id')}`: priority={row.get('priority')}; experiment={row.get('experiment_type')}; recommended command=`{row.get('recommended_command')}`; pass gate={row.get('pass_gate')}; fail action={row.get('fail_action')}."
        )
    lines.append("- Claim boundary: planned experiments do not prove the claim yet; they define what evidence would promote or demote it.")
    return "\n".join(lines)


def _deep_model_ablation_command_contract_snapshot(intent: AgentIntent) -> str:
    if "deep_model_ablation_command_contract" not in intent.evidence_ids:
        return ""
    payload = _read_repo_json("reports/deep_model_ablation_command_contract_latest.json")
    summary = payload.get("summary") or {}
    rows = payload.get("rows") or []
    if not summary or not isinstance(rows, list):
        return "- `reports/deep_model_ablation_command_contract_latest.json` is missing; rerun `scripts/generate_deep_model_ablation_command_contract.py`."

    lines = [
        f"- Deep model ablation command contract snapshot: variants=`{summary.get('variants', 'NA')}`, fallback_ready_rows=`{summary.get('fallback_ready_rows', 'NA')}`, planner_adapter_rows=`{summary.get('planner_adapter_rows', 'NA')}`.",
        f"- Contract statuses: `{json.dumps(summary.get('status_counts') or {}, ensure_ascii=False, sort_keys=True)}`; unsupported flags: `{json.dumps(summary.get('unsupported_flag_counts') or {}, ensure_ascii=False, sort_keys=True)}`.",
        "- This prevents over-claiming: planning-only flags are roadmap items, while runnable fallback commands are audited against the current tuned benchmark runner CLI.",
    ]
    for row in rows[:6]:
        lines.append(
            f"  - `{row.get('variant_id')}`: status={row.get('contract_status')}; fallback_ready={row.get('fallback_ready')}; unsupported_planner_flags={row.get('recommended_unsupported_flags')}; fallback=`{row.get('runnable_fallback_command')}`."
        )
    lines.append("- Claim boundary: a fallback command proves executable scope only; it does not replace the wider paper-level experiment or online validation.")
    return "\n".join(lines)


def _deep_model_ablation_command_contract_smoke_snapshot(intent: AgentIntent) -> str:
    if "deep_model_ablation_command_contract_smoke" not in intent.evidence_ids:
        return ""
    payload = _read_repo_json("reports/deep_model_ablation_command_contract_smoke_latest.json")
    summary = payload.get("summary") or {}
    rows = payload.get("rows") or []
    if not summary or not isinstance(rows, list):
        return "- `reports/deep_model_ablation_command_contract_smoke_latest.json` is missing; rerun `scripts/smoke_deep_model_ablation_command_contract.py`."

    lines = [
        f"- Deep model ablation command contract smoke / fallback smoke snapshot: rows=`{summary.get('rows', 'NA')}`, ok_rows=`{summary.get('ok_rows', 'NA')}`, failed_rows=`{summary.get('failed_rows', 'NA')}`.",
        "- It executes sampled runnable fallback commands with `--output-prefix deep_model_ablation_command_contract_smoke` and `--no-update-latest`, so smoke proof does not overwrite main benchmark latest artifacts.",
    ]
    for row in rows[:4]:
        lines.append(
            f"  - `{row.get('variant_id')}`: status={row.get('status')}; runs={row.get('runs')}; ok_runs={row.get('ok_runs')}; manifest=`{row.get('runner_manifest')}`."
        )
    lines.append("- Claim boundary: command smoke proves the fallback path can run; it is still a small-sample executable proof, not a robust multi-seed benchmark claim.")
    return "\n".join(lines)


def _deep_model_ablation_recommended_command_smoke_snapshot(intent: AgentIntent) -> str:
    if "deep_model_ablation_recommended_command_smoke" not in intent.evidence_ids:
        return ""
    payload = _read_repo_json("reports/deep_model_ablation_recommended_command_smoke_latest.json")
    summary = payload.get("summary") or {}
    rows = payload.get("rows") or []
    if not summary or not isinstance(rows, list):
        return "- `reports/deep_model_ablation_recommended_command_smoke_latest.json` is missing; rerun `scripts/smoke_deep_model_ablation_recommended_commands.py`."

    lines = [
        f"- Deep model ablation recommended command smoke snapshot: rows=`{summary.get('rows', 'NA')}`, ok_rows=`{summary.get('ok_rows', 'NA')}`, models_alias_rows=`{summary.get('models_alias_rows', 'NA')}`, include_variants_alias_rows=`{summary.get('include_variants_alias_rows', 'NA')}`, emit_battle_cards_alias_rows=`{summary.get('emit_battle_cards_alias_rows', 'NA')}`.",
        "- It executes runner-native recommended commands using `--models` / `--include-variants` and isolated `--output-prefix deep_model_ablation_recommended_command_smoke` with `--no-update-latest`, so the main benchmark latest artifacts are not overwritten.",
    ]
    for row in rows[:4]:
        lines.append(
            f"  - `{row.get('variant_id')}`: status={row.get('status')}; runner-native aliases: --models={row.get('uses_models_alias')}, --include-variants={row.get('uses_include_variants_alias')}, --emit-battle-cards={row.get('uses_emit_battle_cards_alias')}; manifest=`{row.get('runner_manifest')}`."
        )
    lines.append("- Claim boundary: recommended command smoke proves CLI executability and alias support; it is still a small-sample smoke, not a full tuned or paper-level benchmark.")
    return "\n".join(lines)


def _deep_model_ablation_command_smoke_coverage_snapshot(intent: AgentIntent) -> str:
    if "deep_model_ablation_command_smoke_coverage" not in intent.evidence_ids:
        return ""
    payload = _read_repo_json("reports/deep_model_ablation_command_smoke_coverage_latest.json")
    summary = payload.get("summary") or {}
    rows = payload.get("rows") or []
    if not summary or not isinstance(rows, list):
        return "- `reports/deep_model_ablation_command_smoke_coverage_latest.json` is missing; rerun `scripts/generate_deep_model_ablation_command_smoke_coverage.py`."

    lines = [
        f"- Deep model ablation command smoke coverage snapshot: contract_rows=`{summary.get('contract_rows', 'NA')}`, any_smoked_rows=`{summary.get('any_smoked_rows', 'NA')}`, recommended_smoked_rows=`{summary.get('recommended_smoked_rows', 'NA')}`, fallback_smoked_rows=`{summary.get('fallback_smoked_rows', 'NA')}`, coverage gate=`{summary.get('coverage_gate', 'NA')}`.",
        "- It separates runner-supported / contract-ready rows from smoke-executed rows, so the project can discuss command coverage without overclaiming benchmark evidence.",
    ]
    for row in rows[:6]:
        lines.append(
            f"  - `{row.get('variant_id')}`: coverage_status={row.get('coverage_status')}; fallback_smoked={row.get('fallback_smoked')}; recommended_smoked={row.get('recommended_smoked')}; next_action={row.get('next_action')}."
        )
    lines.append("- Claim boundary: smoke coverage is evidence governance; it proves command execution coverage, not model-effect superiority.")
    return "\n".join(lines)


def _deep_model_ablation_command_smoke_coverage_diff_snapshot(intent: AgentIntent) -> str:
    if "deep_model_ablation_command_smoke_coverage_diff" not in intent.evidence_ids:
        return ""
    payload = _read_repo_json("reports/deep_model_ablation_command_smoke_coverage_diff_latest.json")
    summary = payload.get("summary") or {}
    rows = payload.get("rows") or []
    if not summary or not isinstance(rows, list):
        return "- `reports/deep_model_ablation_command_smoke_coverage_diff_latest.json` is missing; rerun `scripts/generate_deep_model_ablation_command_smoke_coverage_diff.py`."

    lines = [
        f"- Deep model ablation command smoke coverage diff snapshot: baseline=`{summary.get('baseline_fallback_smoked_rows', 'NA')}/{summary.get('contract_rows', 'NA')}`, current=`{summary.get('current_any_smoked_rows', 'NA')}/{summary.get('contract_rows', 'NA')}`, runner_native=`{summary.get('runner_native_recommended_smoked_rows', 'NA')}/{summary.get('contract_rows', 'NA')}`, gate delta=`{summary.get('coverage_gate_delta', 'NA')}`.",
        "- This is the interview-friendly before/after story: fallback smoke proved a narrow compatibility path, then runner-native recommended smoke expanded executable evidence to all current ablation variants.",
    ]
    for row in rows[:6]:
        lines.append(
            f"  - `{row.get('variant_id')}`: transition={row.get('transition')}; smoke_gain={row.get('smoke_gain')}; claim={row.get('interview_claim')}."
        )
    lines.append("- Claim boundary: coverage diff proves an engineering/evidence upgrade; model superiority still needs tuned or paper-level benchmark evidence.")
    return "\n".join(lines)


def _scenario_notes(intent: AgentIntent) -> list[str]:
    scenario = intent.scenario.lower()
    notes = [
        "普通 response model 学 `P(Y=1|X)`；uplift/CATE 学 `E[Y(1)-Y(0)|X]`，服务的是增量决策；所有结论都必须落到证据链。",
    ]
    if "coupon" in scenario:
        notes.append("coupon/subsidy 场景要从 uplift 转成 incremental profit 和 net value：`uplift * gross_margin - coupon_cost - abuse/fatigue_penalty`。")
    if "ads" in scenario:
        notes.append("ads 场景里 pCTR/pCVR 是响应或转化概率，incrementality/iROAS 才回答广告是否带来增量收入。")
    if "llm" in scenario:
        notes.append("LLM routing 的 cheap model 是 control，strong model/RAG/tool/human review 是 treatment arms，目标是增量质量覆盖成本和延迟。")
    if "multi-treatment" in scenario:
        notes.append("multi-treatment 不是只判断触不触达，而是在多个 action/coupon/channel 之间估计 action-vs-control uplift 和 action propensity；默认对比 MultiTLearnerGBM 与 MultiDRLearnerGBM。")
    if "continuous" in scenario:
        notes.append("continuous treatment 需要 dose-response support 诊断；DRNet/VCNet 只有在 observed support 足够时才适合升级。")
    if "marketplace" in scenario:
        notes.append("marketplace 需要显式处理 interference/spillover，线上实验优先 cluster randomized、geo holdout 或 switchback。")
    if "growth" in scenario:
        notes.append("CRM/Push 场景要显式处理 frequency cap、fatigue、opt-out、delayed feedback 和长期 holdout。")
    if "launch guardrail" in scenario or "warning_audit" in intent.evidence_ids:
        notes.append("Regression Warning / Launch Guardrail Audit 的目的不是把 demo 判死刑，而是把不能上线的证据前置：高严重度 warning 需要 root cause、补实验和 readiness gate 通过后才能 promotion。")
        notes.append("Promotion card 把每个风险聚合成 safe claim / unsafe claim / owner / before-shadow / before-A/B / before-ramp，面试时能直接说明哪些能讲、哪些不能过度宣传。")
        notes.append("Warning triage 再把 warning 分成 smoke-only caveat、production blocker、benchmark/research blocker 和 review-before-pilot，避免把所有 warning 都混成一类。")
        notes.append("面试时可以直接说：能跑 benchmark 只是 smoke，能解释 warning、补 holdout/shadow/OPE/多 seed 实验，才像工业级 launch review。")
    if "agent" in intent.scenario.lower() or "agent_v2" in intent.evidence_ids:
        notes.append("Agent 2.0 不是关键词问答，而是 intent parse -> risk diagnosis -> evidence retrieval -> rubric regression 的 causal decision copilot。")
    if "model_matrix" in intent.evidence_ids:
        notes.append("模型治理要讲 registry、model card、guarded optional dependency 和 readiness gate，避免把不可用后端伪装成默认可跑。")
    return notes


def _topic_badge(intent: AgentIntent) -> str:
    text = " ".join(intent.model_families).lower()
    if "dragonnet" in text or "cfrnet" in text:
        return "难点 I13：深度 uplift loss / representation learning"
    if "llm" in intent.scenario.lower():
        return "难点 I15：LLM routing as uplift"
    if "bandit" in " ".join(intent.rollout_plan).lower():
        return "难点 I17：uplift to contextual bandit"
    if any("OPE" in item or "IPS" in item for item in intent.evaluation_plan):
        return "难点 I18：OPE / IPS / SNIPS / DR"
    if intent.treatment_type == "continuous":
        return "难点 I19：continuous treatment / dose-response"
    return "Agent 2.0：causal decision copilot"


def agent_v2_reply(prompt: str, context: Mapping[str, Any] | None = None) -> str | None:
    if not prompt or not prompt.strip():
        return None
    deconstruction_terms = (
        "源码",
        "拆解",
        "loss",
        "forward",
        "fit",
        "predict",
        "0-1",
        "从0",
        "从 0",
        "model deconstruction",
        "license",
        "开源代码",
        "直接复制",
        "训练评估",
    )
    if any(term in prompt.lower() or term in prompt for term in deconstruction_terms) and not _has_any(
        prompt, ["forward hook", "hook contract", "hook training bridge"]
    ):
        deconstruction = model_deconstruction_reply(prompt, context)
        if deconstruction:
            return deconstruction
    intent = parse_agent_intent(prompt, context)
    evidence = selected_evidence(intent)
    registered = _context_value(context or {}, "registered_models", "103")
    ready = _context_value(context or {}, "ready_models", "47")

    parse_rows = [
        f"- Scenario: `{intent.scenario}`",
        f"- Treatment type: `{intent.treatment_type}`; treatment: `{intent.treatment}`",
        f"- Outcome/reward: `{intent.outcome}`; unit: `{intent.unit}`; time window: `{intent.time_window}`",
        f"- Confounders to review: {', '.join(intent.confounders)}",
        f"- Cost/value objective: `{intent.cost_value}`",
        f"- Guardrails: {', '.join(intent.guardrails)}",
    ]
    model_rows = "\n".join(f"- {item}" for item in intent.model_families)
    risk_rows = "\n".join(f"- {item}" for item in intent.risks)
    note_rows = "\n".join(f"- {item}" for item in _scenario_notes(intent))
    eval_rows = "\n".join(f"- {item}" for item in intent.evaluation_plan)
    rollout_rows = "\n".join(f"- {item}" for item in intent.rollout_plan)
    live_snapshot = _warning_audit_snapshot(intent)
    promotion_snapshot = _promotion_cards_snapshot(intent)
    warning_triage_snapshot = _warning_triage_snapshot(intent)
    pilot_readiness_snapshot = _pilot_readiness_snapshot(intent)
    pilot_experiment_snapshot = _pilot_experiment_plan_snapshot(intent)
    pilot_command_smoke_snapshot = _pilot_experiment_command_smoke_snapshot(intent)
    claim_snapshot = _algorithm_claim_ledger_snapshot(intent)
    reproduction_snapshot = _paper_reproduction_gap_snapshot(intent)
    promotion_matrix_snapshot = _model_evidence_promotion_matrix_snapshot(intent)
    recipe_snapshot = _model_upgrade_recipe_cards_snapshot(intent)
    telemetry_snapshot = _deep_model_telemetry_gap_matrix_snapshot(intent)
    telemetry_smoke_snapshot = _deep_model_telemetry_smoke_snapshot(intent)
    forward_hook_snapshot = _deep_model_forward_hook_contracts_snapshot(intent)
    forward_hook_smoke_snapshot = _deep_model_forward_hook_smoke_snapshot(intent)
    hook_training_bridge_snapshot = _deep_model_hook_training_bridge_snapshot(intent)
    trainer_collector_snapshot = _deep_model_trainer_collector_smoke_snapshot(intent)
    telemetry_benchmark_linkage_snapshot = _deep_model_telemetry_benchmark_linkage_snapshot(intent)
    telemetry_ablation_gate_snapshot = _deep_model_telemetry_ablation_gate_snapshot(intent)
    ablation_interpretation_snapshot = _deep_model_ablation_interpretation_snapshot(intent)
    ablation_promotion_snapshot = _deep_model_ablation_promotion_matrix_snapshot(intent)
    ablation_next_experiment_snapshot = _deep_model_ablation_next_experiment_plan_snapshot(intent)
    ablation_command_contract_snapshot = _deep_model_ablation_command_contract_snapshot(intent)
    ablation_command_contract_smoke_snapshot = _deep_model_ablation_command_contract_smoke_snapshot(intent)
    ablation_recommended_command_smoke_snapshot = _deep_model_ablation_recommended_command_smoke_snapshot(intent)
    ablation_command_smoke_coverage_snapshot = _deep_model_ablation_command_smoke_coverage_snapshot(intent)
    ablation_command_smoke_coverage_diff_snapshot = _deep_model_ablation_command_smoke_coverage_diff_snapshot(intent)
    combined_snapshots = "\n".join(
        item
        for item in [
            live_snapshot,
            promotion_snapshot,
            warning_triage_snapshot,
            pilot_readiness_snapshot,
            pilot_experiment_snapshot,
            pilot_command_smoke_snapshot,
            claim_snapshot,
            reproduction_snapshot,
            promotion_matrix_snapshot,
            recipe_snapshot,
            telemetry_snapshot,
            telemetry_smoke_snapshot,
            forward_hook_snapshot,
            forward_hook_smoke_snapshot,
            hook_training_bridge_snapshot,
            trainer_collector_snapshot,
            telemetry_benchmark_linkage_snapshot,
            telemetry_ablation_gate_snapshot,
            ablation_interpretation_snapshot,
            ablation_promotion_snapshot,
            ablation_next_experiment_snapshot,
            ablation_command_contract_snapshot,
            ablation_command_contract_smoke_snapshot,
            ablation_recommended_command_smoke_snapshot,
            ablation_command_smoke_coverage_snapshot,
            ablation_command_smoke_coverage_diff_snapshot,
        ]
        if item
    )
    live_snapshot_section = f"\n\n**Live artifact snapshot / 最新证据摘要**\n{combined_snapshots}" if combined_snapshots else ""

    demo = (
        "- 5分钟：Story -> Agent 2.0 intent parse -> Evidence Index -> one benchmark/policy artifact。\n"
        "- 15分钟：加 Diagnostics、Model Matrix、Benchmark leaderboard、OPE/Budget optimizer。\n"
        "- 30分钟：展开 DR/R/DML 公式、nuisance artifacts、offline->online rollout、failure modes 和 regression gate。"
    )

    return (
        f"### {_topic_badge(intent)}\n\n"
        "**一句话回答**：我会把这个问题先拆成 causal decision，而不是直接报模型名：先识别 treatment/outcome/unit/time window，再诊断偏差和上线风险，最后给模型、评估、ROI 和 evidence 链路。\n\n"
        "**因果问题理解**\n"
        f"{chr(10).join(parse_rows)}\n\n"
        "**场景诊断**\n"
        f"{risk_rows}\n\n"
        "**关键公式 / 业务翻译**\n"
        f"{note_rows}\n\n"
        "**模型与算法建议**\n"
        f"{model_rows}\n\n"
        "**评估与上线试点**\n"
        f"{eval_rows}\n"
        f"{rollout_rows}\n\n"
        "**Evidence grounding / 证据链**\n"
        f"{_evidence_lines(evidence)}"
        f"{live_snapshot_section}\n\n"
        "**面试讲法**\n"
        f"{demo}\n\n"
        f"**当前仓库上下文**：model catalog `{registered}`，runtime-ready `{ready}`；Agent 2.0 输出覆盖 `算法解释 / 工程架构 / 业务 ROI / 上线试点 / 面试话术` 五个视角。"
    )


def rubric_score_answer(answer: str, expected_terms: Iterable[str] = ()) -> dict[str, Any]:
    lowered = answer.lower()
    checks = {
        "causal_parse": all(term in lowered for term in ["treatment", "outcome", "unit"]),
        "risk_diagnosis": any(term in lowered for term in ["selection bias", "overlap", "leakage", "sutva", "delayed", "metric"]),
        "model_recommendation": any(term in lowered for term in ["learner", "dml", "cfrnet", "dragonnet", "multidr", "doseresponse"]),
        "evaluation_online": any(term in lowered for term in ["qini", "auuc", "policy value", "ope", "a/b", "holdout", "ramp-up"]),
        "evidence_grounding": "evidence grounding" in lowered and ("docs/" in lowered or "reports/" in lowered or "deepuplift/" in lowered),
        "interview_packaging": "5分钟" in answer and "15分钟" in answer and "30分钟" in answer,
        "answer_modes": all(term in answer for term in ["算法解释", "工程架构", "业务 ROI", "上线试点", "面试话术"]),
    }
    expected = {term: term.lower() in lowered for term in expected_terms}
    score = sum(10 for ok in checks.values() if ok)
    score += sum(3 for ok in expected.values() if ok)
    max_score = len(checks) * 10 + len(expected) * 3
    return {
        "score": score,
        "max_score": max_score,
        "ratio": round(score / max_score, 4) if max_score else 0.0,
        "checks": checks,
        "expected": expected,
        "missing_expected": [term for term, ok in expected.items() if not ok],
    }


def agent_v2_regression_rows(context: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    rows = []
    for case in AGENT_V2_GOLDEN_QUESTIONS:
        answer = agent_v2_reply(case["prompt"], context or {}) or ""
        score = rubric_score_answer(answer, case.get("expect") or ())
        rows.append(
            {
                "prompt": case["prompt"],
                "answer_chars": len(answer),
                "score": score["score"],
                "max_score": score["max_score"],
                "ratio": score["ratio"],
                "checks": score["checks"],
                "expected": score["expected"],
                "missing_expected": score["missing_expected"],
            }
        )
    return rows


def agent_v2_capability_summary() -> Dict[str, Any]:
    return {
        "architecture": "thin agent-layer rewrite over existing trainer/benchmark/evidence/UI contracts",
        "capabilities": [
            "causal question parsing",
            "risk and failure-mode diagnosis",
            "model family recommendation",
            "evaluation, OPE, benchmark and rollout planning",
            "evidence-grounded answer generation",
            "interview packaging",
            "golden-question rubric regression",
        ],
        "evidence_items": [asdict(item) for item in EVIDENCE_CATALOG],
        "golden_questions": list(AGENT_V2_GOLDEN_QUESTIONS),
    }
