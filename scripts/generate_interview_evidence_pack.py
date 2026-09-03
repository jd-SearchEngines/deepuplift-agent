from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def latest(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def latest_complete_ui_screenshots(root: Path, min_files: int = 4) -> Path | None:
    matches = sorted(root.glob("ui_screenshots_*"), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in matches:
        if path.is_dir() and len(list(path.glob("*.png"))) >= min_files:
            return path
    return matches[0] if matches else None


def read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def file_status(path: Path | None) -> str:
    if path is None:
        return "missing"
    if path.is_dir():
        files = [item for item in path.glob("*") if item.is_file()]
        return f"directory, {len(files)} files"
    if path.is_file():
        return f"{path.stat().st_size} bytes"
    return "missing"


DIFFICULTY_ROWS = [
    {
        "id": "D01",
        "topic": "Uplift modeling vs conversion prediction",
        "tabs": "Story, Compare, Policy",
        "proof": "Compare ranks treatment-effect models by QINI/AUUC/Top-K; Policy converts uplift score into targeting value.",
        "artifact": "agent regression, no-UI compare manifest, run notes",
        "talk": "I framed the task as incremental effect estimation instead of response scoring, so the system optimizes who changes because of treatment.",
    },
    {
        "id": "D02",
        "topic": "Treatment/control design and causal assumptions",
        "tabs": "Data, Diagnostics, Agent",
        "proof": "Treatment ratio, outcome rate, randomization-like checks, and model recommendations are shown before training.",
        "artifact": "diagnostic warnings, knowledge rules",
        "talk": "The workflow starts with design diagnosis because a strong model cannot repair a broken treatment assignment mechanism.",
    },
    {
        "id": "D03",
        "topic": "Overlap, selection bias, SSB and confounding",
        "tabs": "Diagnostics, Agent",
        "proof": "Selection Bias / SSB Proxy, propensity AUC, weak-overlap rate, feature balance, and robust-model recommendations.",
        "artifact": "diagnostic report, interview audit",
        "talk": "I added guardrails that flag when observed treatment/control data are not comparable and route users toward DR/R/CFR-style learners.",
    },
    {
        "id": "D04",
        "topic": "Model registry, adapters, cards and presets",
        "tabs": "Models, Story",
        "proof": "live registered model count, model cards, preset coverage, dependency/readiness state, and strict catalog audit.",
        "artifact": "model catalog audit, source/family readiness CSV",
        "talk": "The UI never calls raw model classes directly; it goes through a registry contract so model expansion is controlled and auditable.",
    },
    {
        "id": "D05",
        "topic": "Unifying inconsistent uplift outputs",
        "tabs": "Train, Compare, Evaluation",
        "proof": "All runnable models produce a normalized prediction table with y0/y1/uplift score and saved artifacts.",
        "artifact": "predictions CSV, metrics JSON, run evidence ZIP",
        "talk": "I hid each backend's output quirks behind adapters, then evaluated only the normalized contract.",
    },
    {
        "id": "D06",
        "topic": "Decision-grade evaluation",
        "tabs": "Compare, Evaluation, Policy",
        "proof": "QINI, AUUC, Top-K uplift, calibration, bootstrap CI, sensitivity checks and policy value are available in one run surface.",
        "artifact": "agent regression, compare manifest, run notes",
        "talk": "I avoided one-number model selection by combining ranking quality, uncertainty, calibration and business value.",
    },
    {
        "id": "D07",
        "topic": "Offline metrics to online business value",
        "tabs": "Policy, Predict, History",
        "proof": "Cost/revenue inputs, optimal top-fraction threshold, scored export and run evidence make offline selection actionable.",
        "artifact": "policy report, scored predictions, history evidence",
        "talk": "The offline metric only becomes useful when it tells the business who to contact and what value/risk to expect.",
    },
    {
        "id": "D08",
        "topic": "Agent workflow orchestration",
        "tabs": "Agent, Story",
        "proof": "The Agent can diagnose, recommend models, summarize compare results, and explain parameters from current page state.",
        "artifact": "knowledge base JSON, Story workflow, UI screenshots",
        "talk": "I moved the agent from chat answers toward a causal workflow orchestrator with deterministic tools and evidence outputs.",
    },
    {
        "id": "D09",
        "topic": "Evidence bundle, history and regression gate",
        "tabs": "History, Knowledge, Story",
        "proof": "Run notes, metrics/config/predictions, evidence ZIP, screenshot smoke and full regression are refreshed together.",
        "artifact": "evidence index, resume snapshot, screenshots",
        "talk": "Every demo claim has a file behind it, and the regression gate verifies that the story still matches the code.",
    },
    {
        "id": "D10",
        "topic": "Model expansion across common uplift libraries",
        "tabs": "Models, Knowledge",
        "proof": "DeepUplift, LightGBM, EconML and scikit-uplift are runnable; XGBoost/CatBoost/CausalML are cataloged as guarded/optional.",
        "artifact": "model audit source readiness",
        "talk": "I separated model catalog coverage from local readiness, so dependency-heavy industrial models are visible without making the demo fragile.",
    },
    {
        "id": "D11",
        "topic": "Guarded optional dependencies",
        "tabs": "Models, Risk Register",
        "proof": "The catalog exposes why a model is ready, guarded, missing dependency, or Python-version-specific.",
        "artifact": "model catalog audit, risk register",
        "talk": "Some backends are intentionally guarded because reliable product behavior is more important than pretending every package is always safe locally.",
    },
    {
        "id": "D12",
        "topic": "Streamlit MVP to production architecture",
        "tabs": "Story, Architecture",
        "proof": "Architecture docs describe FastAPI + worker + React evolution while the Streamlit MVP remains testable.",
        "artifact": "architecture doc, release note",
        "talk": "The current workbench is a fast demo surface, while the architecture is already shaped around async runs, artifacts and service boundaries.",
    },
    {
        "id": "D13",
        "topic": "Uplift + LLM cost-quality routing",
        "tabs": "Story, Knowledge, Policy, Agent, Models",
        "proof": "Story shows LLM + Uplift Frontier, Knowledge shows LLM + Causal Frontier Map, Policy simulates LLM routing ROI, and Models tags routing-ready estimators.",
        "artifact": "LLM frontier docs, scenario matrix audit, industry Agent smoke, UI screenshots",
        "talk": "I framed LLM routing as uplift: treatment is calling the strong model, control is the cheap model, and policy value is incremental quality minus incremental cost and latency risk.",
    },
    {
        "id": "D14",
        "topic": "Deep model source deconstruction with training evidence",
        "tabs": "Model Deconstruction, Evidence, Agent",
        "proof": "CFRNet, DragonNet, EFIN and DESCN have source-level docs plus shared smoke training rows, loss curves and evidence manifests.",
        "artifact": "deep model training evidence JSON/CSV, model deconstruction docs, UI screenshots",
        "talk": "I can explain the formula and source code, then open the training curve and run manifest that prove the model path is executable.",
    },
    {
        "id": "D15",
        "topic": "Paper-level CATE benchmark evidence",
        "tabs": "Model Deconstruction, Evidence, Evaluation",
        "proof": "IHDP, ACIC-style synthetic and known-CATE synthetic benchmarks compare T-Learner, DRLearner, CFRNet, DragonNet, EFIN and DESCN with PEHE, ATE error, QINI, AUUC, policy value, bootstrap CI and run diff.",
        "artifact": "paper benchmark JSON/leaderboard/run diff/manifest, paper benchmark doc, per-run evidence manifests",
        "talk": "I separated smoke proof from paper-level proof: smoke proves the route is executable, while known-CATE benchmark rows prove effect accuracy, ranking quality and decision value under reproducible artifacts.",
    },
    {
        "id": "D17",
        "topic": "UI glossary and metric explainability",
        "tabs": "Evidence, Evaluation, Knowledge, Model Deconstruction",
        "proof": "The UI now explains PEHE, ATE error, QINI, AUUC, OPE, overlap, policy value and deep uplift model terms beside the benchmark and evidence tables.",
        "artifact": "glossary JSON/doc, UI screenshot smoke, Benchmark Dashboard glossary panel",
        "talk": "I made the workbench reviewable for people who do not already know every causal-ML term: the same term dictionary appears in docs, reports and UI panels.",
    },
    {
        "id": "D18",
        "topic": "Benchmark win/loss interpretation",
        "tabs": "Evidence, Model Deconstruction",
        "proof": "Benchmark interpretation explains dataset winners, PEHE/QINI/policy conflicts, stability, failure attribution and model-specific interview lines.",
        "artifact": "paper benchmark interpretation JSON/doc, paper benchmark leaderboard, run diff",
        "talk": "I do not only show a leaderboard; I explain why a model won or lost, whether the evidence is multi-seed stable, and what claim is safe to make in an interview or review.",
    },
    {
        "id": "D19",
        "topic": "Deep model tuned benchmark and failure attribution",
        "tabs": "Evidence, Model Deconstruction",
        "proof": "CFRNet, DragonNet, EFIN and DESCN now have tuned/ablation benchmark rows against T-Learner and DRLearner baselines, including PEHE, QINI, policy value, loss deltas, battle cards and run manifests.",
        "artifact": "deep model tuned benchmark JSON/leaderboard/battle cards/manifest/doc",
        "talk": "I can explain why a deep model lost to T/DR on small tabular data, whether tuning improved PEHE or ranking, and which claim is safe: CATE accuracy, ranking utility, or guarded follow-up.",
    },
    {
        "id": "D20",
        "topic": "Regression warning audit and launch guardrails",
        "tabs": "Evidence, Agent, Policy",
        "proof": "Regression Warning / Launch Guardrail Audit turns negative ROI, weak overlap, sensitivity failures, metric conflicts and deep-model baseline losses into review actions.",
        "artifact": "regression warning audit JSON/CSV/doc, agent v2 regression",
        "talk": "I treat warnings as launch-readiness evidence, not noise: high-severity rows block promotion until root cause, owner,补实验 and readiness gate are documented.",
    },
    {
        "id": "D21",
        "topic": "Promotion launch cards and safe claims",
        "tabs": "Evidence, Agent, Interview Pack",
        "proof": "Promotion Launch Cards aggregate warning rows into per-entity decisions, owner roles, safe/unsafe claims, before-shadow/A/B/ramp requirements and next experiments.",
        "artifact": "promotion launch cards JSON/CSV/doc, regression warning audit",
        "talk": "I can say exactly what a result proves and what it does not prove: the card separates safe interview claims from unsafe production claims and turns each blocker into a next experiment.",
    },
    {
        "id": "D22",
        "topic": "Deep objective diagnostics",
        "tabs": "Model Deconstruction, Evidence",
        "proof": "Deep Model Objective Diagnostics maps CFRNet, DragonNet, EFIN and DESCN loss terms to formulas, observable diagnostics, failure signals, benchmark evidence and safe/unsafe claims.",
        "artifact": "deep model objective diagnostics JSON/CSV/doc, tuned benchmark, training evidence",
        "talk": "I can walk from formula to code to metric: factual loss, IPM/propensity/interaction/cross-head terms, then PEHE/QINI/policy value and the claim boundary.",
    },
    {
        "id": "D23",
        "topic": "Algorithm claim ledger",
        "tabs": "Evidence, Model Deconstruction, Interview Pack",
        "proof": "Algorithm Claim Ledger binds every model claim to formula, source path, benchmark verdict, license boundary, safe/unsafe claim and UI route.",
        "artifact": "algorithm claim ledger JSON/CSV/doc, model deconstruction catalog, benchmark interpretation, promotion cards",
        "talk": "When I say a model is advanced, I can immediately show what evidence level supports that sentence and which stronger production claim is still blocked.",
    },
    {
        "id": "D24",
        "topic": "Paper reproduction gap ledger",
        "tabs": "Evidence, Model Deconstruction, Interview Pack",
        "proof": "Paper Reproduction Gap Ledger separates paper citation, open-source reference, local source reproduction, benchmark evidence and license risk.",
        "artifact": "paper reproduction gap ledger JSON/CSV/doc, model deconstruction catalog, paper/deep benchmark artifacts",
        "talk": "I do not say every paper is fully reproduced; I show the exact proof level and the remaining gap before making a stronger claim.",
    },
    {
        "id": "D25",
        "topic": "Model evidence promotion matrix",
        "tabs": "Evidence, Model Deconstruction, Interview Pack",
        "proof": "Model Evidence Promotion Matrix combines source deconstruction, claim ledger, paper reproduction gaps, objective diagnostics, benchmark verdicts, license gate and launch decision into one review table.",
        "artifact": "model evidence promotion matrix JSON/CSV/doc, algorithm claim ledger, paper reproduction gap ledger, objective diagnostics",
        "talk": "I can open one row per model and explain exactly what I can safely claim, what is blocked, and which evidence step upgrades it toward benchmark or pilot readiness.",
    },
    {
        "id": "D26",
        "topic": "Model upgrade recipe cards",
        "tabs": "Evidence, Model Deconstruction, Interview Pack",
        "proof": "Model Upgrade Recipe Cards turn each model's next evidence step into a hypothesis, command, metrics, pass gate, fail action and expected artifacts.",
        "artifact": "model upgrade recipe cards JSON/CSV/doc, model evidence promotion matrix",
        "talk": "When asked what I would do next, I do not hand-wave: I show the exact experiment recipe and the gate that would upgrade or block the model.",
    },
    {
        "id": "D27",
        "topic": "Deep model telemetry gap matrix",
        "tabs": "Evidence, Model Deconstruction, Interview Pack",
        "proof": "Deep Model Telemetry Gap Matrix maps CFRNet, DragonNet, EFIN and DESCN loss terms to missing telemetry, implementation hooks, pass gates and failure actions.",
        "artifact": "deep model telemetry gap matrix JSON/CSV/doc, objective diagnostics, tuned benchmark, training evidence",
        "talk": "When a deep model wins or loses, I can explain which loss-term telemetry is already observable and which hook must be added before stronger paper-level claims.",
    },
    {
        "id": "D28",
        "topic": "Deep model telemetry smoke",
        "tabs": "Evidence, Model Deconstruction, Interview Pack",
        "proof": "Deep Model Telemetry Smoke reads latest run artifacts and verifies loss history, prediction spread, top-K known-CATE signal, metrics and evidence manifests for CFRNet, DragonNet, EFIN and DESCN.",
        "artifact": "deep model telemetry smoke JSON/CSV/doc, deep model training evidence, telemetry gap matrix",
        "talk": "I separate basic telemetry that already runs from advanced hooks still needed; that makes the next refactor evidence-driven instead of speculative.",
    },
    {
        "id": "D29",
        "topic": "Deep model forward hook contracts",
        "tabs": "Evidence, Model Deconstruction, Interview Pack",
        "proof": "Deep Model Forward Hook Contracts freeze tensor keys, artifact files, smoke tests, pass gates, fallback behavior and backward-compatibility rules before changing CFRNet, DragonNet, EFIN or DESCN internals.",
        "artifact": "deep model forward hook contracts JSON/CSV/doc, telemetry smoke, telemetry gap matrix, model source refs",
        "talk": "I do not jump straight into fragile model rewrites; I first define hook contracts for phi_x, e_hat, tarreg, attention and DESCN heads, then implement guarded exports against those gates.",
    },
    {
        "id": "D30",
        "topic": "Deep model forward hook smoke",
        "tabs": "Evidence, Model Deconstruction, Interview Pack",
        "proof": "Deep Model Forward Hook Smoke executes guarded forward instrumentation and writes sidecar artifacts for CFRNet phi_x, DragonNet e_hat/tarreg, EFIN attention/path split and DESCN exposure/heads/constraints.",
        "artifact": "deep model forward hook smoke JSON/CSV/doc, sidecar artifact directory, forward hook contracts",
        "talk": "The next step is no longer just a plan: I can show the actual exported tensor keys and artifact files, while still keeping default model APIs backward-compatible.",
    },
    {
        "id": "D31",
        "topic": "Deep model hook training bridge",
        "tabs": "Evidence, Model Deconstruction, Interview Pack",
        "proof": "Deep Model Hook Training Bridge joins forward-hook sidecar artifacts with training runs/manifests and defines per-epoch trainer telemetry columns, sidecar schemas, pass gates and fail actions.",
        "artifact": "deep model hook training bridge JSON/CSV/doc, forward hook smoke, deep model training evidence",
        "talk": "I can explain the safe migration path from one-batch tensor export to trainer-level paper evidence: optional collectors, epoch artifacts, metric linkage, then ablation or benchmark promotion.",
    },
    {
        "id": "D32",
        "topic": "Deep model trainer collector smoke",
        "tabs": "Evidence, Model Deconstruction, Interview Pack",
        "proof": "Deep Model Trainer Collector Smoke runs tiny training loops and emits per-epoch advanced telemetry for CFRNet representation balance, DragonNet propensity/tarreg, EFIN attention/path and DESCN heads/constraints.",
        "artifact": "deep model trainer collector JSON/model CSV/epoch CSV/doc, sidecar artifact directory, hook training bridge",
        "talk": "The bridge is no longer only a contract: I can show executable per-epoch telemetry sidecars while still saying the safe boundary is smoke, not production/paper ablation proof.",
    },
    {
        "id": "D33",
        "topic": "Deep model telemetry to benchmark linkage",
        "tabs": "Evidence, Model Deconstruction, Interview Pack",
        "proof": "Deep Model Telemetry-Benchmark Linkage joins trainer collector metrics with paper scorecards, tuned benchmark verdicts, baseline context, failure hypotheses and next ablations.",
        "artifact": "deep model telemetry-benchmark linkage JSON/CSV/doc, trainer collector, tuned benchmark, paper benchmark interpretation",
        "talk": "I can now explain a deep model loss term as a reviewable evidence chain: internal telemetry, benchmark outcome, safe claim, blocked claim and the exact ablation needed before stronger claims.",
    },
    {
        "id": "D34",
        "topic": "Deep model telemetry ablation gate",
        "tabs": "Evidence, Model Deconstruction, Interview Pack",
        "proof": "Deep Model Telemetry Ablation Gate turns each telemetry-benchmark linkage row into required variants, command, metrics to watch, pass gate, fail action, safe claim and blocked claim.",
        "artifact": "deep model telemetry ablation gate JSON/CSV/doc, telemetry-benchmark linkage, tuned benchmark",
        "talk": "I can say which deep-model claim is still blocked, which ablation variant is missing, and what metric movement would be enough to promote the claim.",
    },
    {
        "id": "D35",
        "topic": "Deep model ablation interpretation",
        "tabs": "Evidence, Model Deconstruction, Interview Pack",
        "proof": "Deep Model Ablation Interpretation reads the tuned benchmark and explains each runnable variant's PEHE/QINI/policy/loss movement versus the default model.",
        "artifact": "deep model ablation interpretation JSON/CSV/doc, tuned benchmark, telemetry ablation gate",
        "talk": "I can now distinguish variant coverage from actual ablation evidence: the gate says what to run, while the interpretation says what the latest run proves and what still remains blocked.",
    },
    {
        "id": "D36",
        "topic": "Deep model ablation promotion matrix",
        "tabs": "Evidence, Model Deconstruction, Interview Pack",
        "proof": "Deep Model Ablation Promotion Matrix upgrades ablation interpretation into claim tiers, promotion decisions, default-or-variant recommendations and blocked claims.",
        "artifact": "deep model ablation promotion matrix JSON/CSV/doc, ablation interpretation, telemetry ablation gate, model evidence promotion matrix",
        "talk": "I can say which variant is safe as local mechanism evidence, which one is only a tradeoff story, and why a runnable ablation still cannot be sold as SOTA or production lift.",
    },
    {
        "id": "D37",
        "topic": "Deep model ablation next-experiment plan",
        "tabs": "Evidence, Model Deconstruction, Interview Pack",
        "proof": "Deep Model Ablation Next Experiment Plan turns each ablation claim tier into priority, planning command, runnable fallback command, metrics to watch, pass gate, fail action and owner role.",
        "artifact": "deep model ablation next-experiment JSON/CSV/doc, ablation promotion matrix, telemetry ablation gate",
        "talk": "When asked what I would optimize next, I can point to concrete P0/P1 experiments instead of vague tuning: seed/baseline stress for B-tier variants and metric-conflict triage for C-tier variants.",
    },
    {
        "id": "D38",
        "topic": "Deep model ablation command contract",
        "tabs": "Evidence, Model Deconstruction, Interview Pack",
        "proof": "Deep Model Ablation Command Contract audits each next-experiment command against the current tuned benchmark runner CLI and separates planning-only flags from runner-supported fallback commands.",
        "artifact": "deep model ablation command contract JSON/CSV/doc, next-experiment plan, tuned benchmark runner",
        "talk": "I can show that the roadmap is not inflated: future runner-adapter ideas are labeled planning-only, while every variant has a fallback command that is executable with the current benchmark CLI.",
    },
    {
        "id": "D39",
        "topic": "Deep model ablation command contract smoke",
        "tabs": "Evidence, Model Deconstruction, Interview Pack",
        "proof": "Deep Model Ablation Command Contract Smoke executes a small sample of fallback commands with isolated output prefix and `--no-update-latest`, proving the fallback path can run without overwriting main benchmark evidence.",
        "artifact": "deep model ablation command contract smoke JSON/CSV/doc, command contract, isolated tuned benchmark runner artifacts",
        "talk": "If challenged on whether the fallback commands are real, I can point to smoke rows that actually ran and produced isolated runner manifests without touching the main leaderboard.",
    },
    {
        "id": "D40",
        "topic": "Deep model ablation recommended command smoke",
        "tabs": "Evidence, Model Deconstruction, Interview Pack",
        "proof": "Deep Model Ablation Recommended Command Smoke executes runner-native recommended commands that use `--models`, `--include-variants` and at least one `--emit-battle-cards` row, with safe overrides and `--no-update-latest`.",
        "artifact": "deep model ablation recommended command smoke JSON/CSV/doc, command contract, isolated tuned benchmark runner artifacts",
        "talk": "If challenged on whether the new runner-native aliases are real, I can show sampled recommended commands that actually ran without relying on the fallback `--variants` path.",
    },
    {
        "id": "D41",
        "topic": "Deep model ablation command smoke coverage",
        "tabs": "Evidence, Model Deconstruction, Interview Pack",
        "proof": "Deep Model Ablation Command Smoke Coverage joins the command contract, fallback-command smoke and runner-native recommended-command smoke, separating smoke-executed variants from contract-ready backlog.",
        "artifact": "deep model ablation command smoke coverage JSON/CSV/doc, command contract smoke, recommended command smoke",
        "talk": "I avoid overclaiming: runner-supported means the command is accepted by the CLI, while smoke-covered means a safe isolated execution produced manifests; the coverage report shows both.",
    },
    {
        "id": "D42",
        "topic": "Deep model ablation command smoke coverage diff",
        "tabs": "Evidence, Model Deconstruction, Interview Pack",
        "proof": "Deep Model Ablation Command Smoke Coverage Diff turns fallback-only smoke coverage into a before/after upgrade story and shows the gate moving from review_required to ok when runner-native smoke reaches full coverage.",
        "artifact": "deep model ablation command smoke coverage diff JSON/CSV/doc, command smoke coverage, recommended command smoke",
        "talk": "I can explain exactly what changed in this iteration: coverage went from 1/9 fallback-smoked rows to 9/9 current smoke-covered rows, while keeping the boundary that smoke proves executability, not model superiority.",
    },
    {
        "id": "D43",
        "topic": "Regression warning triage",
        "tabs": "Evidence, Agent, Interview Pack",
        "proof": "Regression Warning Triage classifies every warning into risk family, claim boundary, action lane, smoke-only caveat, launch blocker, owner and next experiment.",
        "artifact": "regression warning triage JSON/CSV/doc, regression warning audit, promotion launch cards",
        "talk": "I do not hide warnings; I classify them. Some are smoke-only caveats, while high-severity business value, overlap and readiness rows block pilot until targeted experiments pass.",
    },
    {
        "id": "D44",
        "topic": "Pilot readiness scorecard",
        "tabs": "Evidence, Agent, Interview Pack",
        "proof": "Pilot Readiness Scorecard joins promotion launch cards and warning triage into a pilot gate, readiness score, evidence grade, next gate, rollback conditions and required artifacts.",
        "artifact": "pilot readiness scorecard JSON/CSV/doc, promotion launch cards, regression warning triage",
        "talk": "I can answer the上线 question directly: which rows are blocked, which are paper-review candidates, which can enter shadow, and what artifact must be refreshed before A/B.",
    },
    {
        "id": "D45",
        "topic": "Pilot experiment plan",
        "tabs": "Evidence, Agent, Interview Pack",
        "proof": "Pilot Experiment Plan turns readiness gates into executable next experiments with a known CLI command, metrics to watch, expected artifacts, pass gate, fail action and claim-upgrade boundary.",
        "artifact": "pilot experiment plan JSON/CSV/doc, pilot readiness scorecard, warning triage",
        "talk": "I do not stop at scoring readiness. For each row I can say exactly what to run next, what would count as pass/fail, and how the resume or launch claim changes after evidence refresh.",
    },
    {
        "id": "D46",
        "topic": "Pilot experiment command smoke",
        "tabs": "Evidence, Agent, Interview Pack",
        "proof": "Pilot Experiment Command Smoke executes a sampled set of pilot next-experiment commands with tiny-row safe overrides and isolated output prefixes, proving command executability without overwriting main benchmark evidence.",
        "artifact": "pilot experiment command smoke JSON/CSV/doc, pilot experiment plan, runner artifacts",
        "talk": "I distinguish planned commands from executed evidence: sampled smoke proves the runner path works; full pilot readiness still requires larger benchmark/OPE/holdout artifacts.",
    },
]


def build_pack(reports_dir: Path, runs_dir: Path) -> tuple[str, dict[str, Any]]:
    generated_at = time.strftime("%Y-%m-%d %H:%M:%S %z")
    artifacts = {
        "model_catalog_audit": latest(reports_dir, "model_catalog_audit_*.json"),
        "source_readiness_csv": latest(reports_dir, "model_source_readiness_*.csv"),
        "family_readiness_csv": latest(reports_dir, "model_family_readiness_*.csv"),
        "agent_regression": latest(reports_dir, "agent_regression_*.json"),
        "agent_v2_regression": latest(reports_dir, "agent_v2_regression_latest.json"),
        "model_deconstruction_catalog": latest(reports_dir, "model_deconstruction_catalog_latest.json"),
        "model_deconstruction_agent": latest(reports_dir, "model_deconstruction_agent_latest.json"),
        "deep_model_training_evidence": latest(reports_dir, "deep_model_training_evidence_latest.json"),
        "deep_model_training_leaderboard": latest(reports_dir, "deep_model_training_leaderboard_latest.csv"),
        "deep_model_tuned_benchmark": latest(reports_dir, "deep_model_tuned_benchmark_latest.json"),
        "deep_model_tuned_leaderboard": latest(reports_dir, "deep_model_tuned_benchmark_leaderboard_latest.csv"),
        "deep_model_tuned_battle_cards": latest(reports_dir, "deep_model_tuned_benchmark_battle_cards_latest.csv"),
        "deep_model_tuned_run_diff": latest(reports_dir, "deep_model_tuned_benchmark_run_diff_latest.json"),
        "deep_model_tuned_manifest": latest(reports_dir, "deep_model_tuned_benchmark_manifest_latest.json"),
        "deep_model_objective_diagnostics": latest(reports_dir, "deep_model_objective_diagnostics_latest.json"),
        "deep_model_objective_diagnostics_csv": latest(reports_dir, "deep_model_objective_diagnostics_latest.csv"),
        "algorithm_claim_ledger": latest(reports_dir, "algorithm_claim_ledger_latest.json"),
        "algorithm_claim_ledger_csv": latest(reports_dir, "algorithm_claim_ledger_latest.csv"),
        "paper_reproduction_gap_ledger": latest(reports_dir, "paper_reproduction_gap_ledger_latest.json"),
        "paper_reproduction_gap_ledger_csv": latest(reports_dir, "paper_reproduction_gap_ledger_latest.csv"),
        "model_evidence_promotion_matrix": latest(reports_dir, "model_evidence_promotion_matrix_latest.json"),
        "model_evidence_promotion_matrix_csv": latest(reports_dir, "model_evidence_promotion_matrix_latest.csv"),
        "model_upgrade_recipe_cards": latest(reports_dir, "model_upgrade_recipe_cards_latest.json"),
        "model_upgrade_recipe_cards_csv": latest(reports_dir, "model_upgrade_recipe_cards_latest.csv"),
        "deep_model_telemetry_gap_matrix": latest(reports_dir, "deep_model_telemetry_gap_matrix_latest.json"),
        "deep_model_telemetry_gap_matrix_csv": latest(reports_dir, "deep_model_telemetry_gap_matrix_latest.csv"),
        "deep_model_telemetry_smoke": latest(reports_dir, "deep_model_telemetry_smoke_latest.json"),
        "deep_model_telemetry_smoke_csv": latest(reports_dir, "deep_model_telemetry_smoke_latest.csv"),
        "deep_model_forward_hook_contracts": latest(reports_dir, "deep_model_forward_hook_contracts_latest.json"),
        "deep_model_forward_hook_contracts_csv": latest(reports_dir, "deep_model_forward_hook_contracts_latest.csv"),
        "deep_model_forward_hook_smoke": latest(reports_dir, "deep_model_forward_hook_smoke_latest.json"),
        "deep_model_forward_hook_smoke_csv": latest(reports_dir, "deep_model_forward_hook_smoke_latest.csv"),
        "deep_model_hook_training_bridge": latest(reports_dir, "deep_model_hook_training_bridge_latest.json"),
        "deep_model_hook_training_bridge_csv": latest(reports_dir, "deep_model_hook_training_bridge_latest.csv"),
        "deep_model_trainer_collector_smoke": latest(reports_dir, "deep_model_trainer_collector_smoke_latest.json"),
        "deep_model_trainer_collector_models_csv": latest(reports_dir, "deep_model_trainer_collector_models_latest.csv"),
        "deep_model_trainer_collector_epochs_csv": latest(reports_dir, "deep_model_trainer_collector_epochs_latest.csv"),
        "deep_model_telemetry_benchmark_linkage": latest(reports_dir, "deep_model_telemetry_benchmark_linkage_latest.json"),
        "deep_model_telemetry_benchmark_linkage_csv": latest(reports_dir, "deep_model_telemetry_benchmark_linkage_latest.csv"),
        "deep_model_telemetry_ablation_gate": latest(reports_dir, "deep_model_telemetry_ablation_gate_latest.json"),
        "deep_model_telemetry_ablation_gate_csv": latest(reports_dir, "deep_model_telemetry_ablation_gate_latest.csv"),
        "deep_model_ablation_interpretation": latest(reports_dir, "deep_model_ablation_interpretation_latest.json"),
        "deep_model_ablation_interpretation_csv": latest(reports_dir, "deep_model_ablation_interpretation_latest.csv"),
        "deep_model_ablation_promotion_matrix": latest(reports_dir, "deep_model_ablation_promotion_matrix_latest.json"),
        "deep_model_ablation_promotion_matrix_csv": latest(reports_dir, "deep_model_ablation_promotion_matrix_latest.csv"),
        "deep_model_ablation_next_experiment_plan": latest(reports_dir, "deep_model_ablation_next_experiment_plan_latest.json"),
        "deep_model_ablation_next_experiment_plan_csv": latest(reports_dir, "deep_model_ablation_next_experiment_plan_latest.csv"),
        "deep_model_ablation_command_contract": latest(reports_dir, "deep_model_ablation_command_contract_latest.json"),
        "deep_model_ablation_command_contract_csv": latest(reports_dir, "deep_model_ablation_command_contract_latest.csv"),
        "deep_model_ablation_command_contract_smoke": latest(reports_dir, "deep_model_ablation_command_contract_smoke_latest.json"),
        "deep_model_ablation_command_contract_smoke_csv": latest(reports_dir, "deep_model_ablation_command_contract_smoke_latest.csv"),
        "deep_model_ablation_recommended_command_smoke": latest(reports_dir, "deep_model_ablation_recommended_command_smoke_latest.json"),
        "deep_model_ablation_recommended_command_smoke_csv": latest(reports_dir, "deep_model_ablation_recommended_command_smoke_latest.csv"),
        "deep_model_ablation_command_smoke_coverage": latest(reports_dir, "deep_model_ablation_command_smoke_coverage_latest.json"),
        "deep_model_ablation_command_smoke_coverage_csv": latest(reports_dir, "deep_model_ablation_command_smoke_coverage_latest.csv"),
        "deep_model_ablation_command_smoke_coverage_diff": latest(reports_dir, "deep_model_ablation_command_smoke_coverage_diff_latest.json"),
        "deep_model_ablation_command_smoke_coverage_diff_csv": latest(reports_dir, "deep_model_ablation_command_smoke_coverage_diff_latest.csv"),
        "paper_benchmark": latest(reports_dir, "paper_benchmark_latest.json"),
        "paper_benchmark_leaderboard": latest(reports_dir, "paper_benchmark_leaderboard_latest.csv"),
        "paper_benchmark_run_diff": latest(reports_dir, "paper_benchmark_run_diff_latest.json"),
        "paper_benchmark_manifest": latest(reports_dir, "paper_benchmark_manifest_latest.json"),
        "paper_benchmark_interpretation": latest(reports_dir, "paper_benchmark_interpretation_latest.json"),
        "regression_warning_audit": latest(reports_dir, "regression_warning_audit_latest.json"),
        "regression_warning_audit_csv": latest(reports_dir, "regression_warning_audit_latest.csv"),
        "promotion_launch_cards": latest(reports_dir, "promotion_launch_cards_latest.json"),
        "promotion_launch_cards_csv": latest(reports_dir, "promotion_launch_cards_latest.csv"),
        "regression_warning_triage": latest(reports_dir, "regression_warning_triage_latest.json"),
        "regression_warning_triage_csv": latest(reports_dir, "regression_warning_triage_latest.csv"),
        "pilot_readiness_scorecard": latest(reports_dir, "pilot_readiness_scorecard_latest.json"),
        "pilot_readiness_scorecard_csv": latest(reports_dir, "pilot_readiness_scorecard_latest.csv"),
        "pilot_experiment_plan": latest(reports_dir, "pilot_experiment_plan_latest.json"),
        "pilot_experiment_plan_csv": latest(reports_dir, "pilot_experiment_plan_latest.csv"),
        "pilot_experiment_command_smoke": latest(reports_dir, "pilot_experiment_command_smoke_latest.json"),
        "pilot_experiment_command_smoke_csv": latest(reports_dir, "pilot_experiment_command_smoke_latest.csv"),
        "pilot_experiment_command_coverage": latest(reports_dir, "pilot_experiment_command_coverage_latest.json"),
        "pilot_experiment_command_coverage_csv": latest(reports_dir, "pilot_experiment_command_coverage_latest.csv"),
        "glossary": latest(reports_dir, "glossary_latest.json"),
        "no_ui_compare_manifest": latest(reports_dir, "no_ui_compare_manifest_*.json"),
        "interview_materials_audit": latest(reports_dir, "interview_materials_audit*.json"),
        "industry_playbook_audit": latest(reports_dir, "industry_playbook_audit*.json"),
        "industry_playbook_json": latest(reports_dir, "industry_playbook*.json"),
        "agent_interview_transcript_sample_json": latest(reports_dir, "agent_interview_transcript_sample*.json"),
        "ui_screenshots": latest_complete_ui_screenshots(reports_dir),
        "evidence_index": Path("docs/DEEPUplift_AGENT_EVIDENCE_INDEX.md"),
        "interview_difficulties": Path("docs/INTERVIEW_DIFFICULTIES_AND_SOLUTIONS.md"),
        "interview_deep_dive": Path("docs/INTERVIEW_DEEPUplift_AGENT_DEEP_DIVE.md"),
        "agent_interview_transcript_sample": Path("docs/AGENT_INTERVIEW_TRANSCRIPT_SAMPLE.md"),
        "model_deconstruction_index": Path("docs/DEEPUplift_MODEL_DECONSTRUCTION_INDEX.md"),
        "deep_model_training_evidence_doc": Path("docs/UPLIFT_DEEP_MODEL_TRAINING_EVIDENCE.md"),
        "deep_model_tuned_benchmark_doc": Path("docs/DEEPUplift_DEEP_MODEL_TUNING_BENCHMARK.md"),
        "deep_model_objective_diagnostics_doc": Path("docs/DEEPUplift_DEEP_MODEL_OBJECTIVE_DIAGNOSTICS.md"),
        "algorithm_claim_ledger_doc": Path("docs/DEEPUplift_ALGORITHM_CLAIM_LEDGER.md"),
        "paper_reproduction_gap_ledger_doc": Path("docs/DEEPUplift_PAPER_REPRODUCTION_GAP_LEDGER.md"),
        "model_evidence_promotion_matrix_doc": Path("docs/DEEPUplift_MODEL_EVIDENCE_PROMOTION_MATRIX.md"),
        "model_upgrade_recipe_cards_doc": Path("docs/DEEPUplift_MODEL_UPGRADE_RECIPE_CARDS.md"),
        "deep_model_telemetry_gap_matrix_doc": Path("docs/DEEPUplift_DEEP_MODEL_TELEMETRY_GAP_MATRIX.md"),
        "deep_model_telemetry_smoke_doc": Path("docs/DEEPUplift_DEEP_MODEL_TELEMETRY_SMOKE.md"),
        "deep_model_forward_hook_contracts_doc": Path("docs/DEEPUplift_DEEP_MODEL_FORWARD_HOOK_CONTRACTS.md"),
        "deep_model_forward_hook_smoke_doc": Path("docs/DEEPUplift_DEEP_MODEL_FORWARD_HOOK_SMOKE.md"),
        "deep_model_hook_training_bridge_doc": Path("docs/DEEPUplift_DEEP_MODEL_HOOK_TRAINING_BRIDGE.md"),
        "deep_model_trainer_collector_smoke_doc": Path("docs/DEEPUplift_DEEP_MODEL_TRAINER_COLLECTOR_SMOKE.md"),
        "deep_model_telemetry_benchmark_linkage_doc": Path("docs/DEEPUplift_DEEP_MODEL_TELEMETRY_BENCHMARK_LINKAGE.md"),
        "deep_model_telemetry_ablation_gate_doc": Path("docs/DEEPUplift_DEEP_MODEL_TELEMETRY_ABLATION_GATE.md"),
        "deep_model_ablation_interpretation_doc": Path("docs/DEEPUplift_DEEP_MODEL_ABLATION_INTERPRETATION.md"),
        "deep_model_ablation_promotion_matrix_doc": Path("docs/DEEPUplift_DEEP_MODEL_ABLATION_PROMOTION_MATRIX.md"),
        "deep_model_ablation_next_experiment_plan_doc": Path("docs/DEEPUplift_DEEP_MODEL_ABLATION_NEXT_EXPERIMENT_PLAN.md"),
        "deep_model_ablation_command_contract_doc": Path("docs/DEEPUplift_DEEP_MODEL_ABLATION_COMMAND_CONTRACT.md"),
        "deep_model_ablation_command_contract_smoke_doc": Path("docs/DEEPUplift_DEEP_MODEL_ABLATION_COMMAND_CONTRACT_SMOKE.md"),
        "deep_model_ablation_recommended_command_smoke_doc": Path("docs/DEEPUplift_DEEP_MODEL_ABLATION_RECOMMENDED_COMMAND_SMOKE.md"),
        "deep_model_ablation_command_smoke_coverage_doc": Path("docs/DEEPUplift_DEEP_MODEL_ABLATION_COMMAND_SMOKE_COVERAGE.md"),
        "deep_model_ablation_command_smoke_coverage_diff_doc": Path("docs/DEEPUplift_DEEP_MODEL_ABLATION_COMMAND_SMOKE_COVERAGE_DIFF.md"),
        "paper_benchmark_doc": Path("docs/DEEPUplift_PAPER_LEVEL_BENCHMARK.md"),
        "paper_benchmark_interpretation_doc": Path("docs/DEEPUplift_BENCHMARK_V3_INTERPRETATION.md"),
        "regression_warning_audit_doc": Path("docs/DEEPUplift_REGRESSION_WARNING_AUDIT.md"),
        "promotion_launch_cards_doc": Path("docs/DEEPUplift_PROMOTION_LAUNCH_CARDS.md"),
        "regression_warning_triage_doc": Path("docs/DEEPUplift_REGRESSION_WARNING_TRIAGE.md"),
        "pilot_readiness_scorecard_doc": Path("docs/DEEPUplift_PILOT_READINESS_SCORECARD.md"),
        "pilot_experiment_plan_doc": Path("docs/DEEPUplift_PILOT_EXPERIMENT_PLAN.md"),
        "pilot_experiment_command_smoke_doc": Path("docs/DEEPUplift_PILOT_EXPERIMENT_COMMAND_SMOKE.md"),
        "pilot_experiment_command_coverage_doc": Path("docs/DEEPUplift_PILOT_EXPERIMENT_COMMAND_COVERAGE.md"),
        "glossary_doc": Path("docs/DEEPUplift_GLOSSARY.md"),
        "chinese_interview_qa": Path("docs/INTERVIEW_QA_REHEARSAL_CN.md"),
        "chinese_demo_checklist": Path("docs/INTERVIEW_DEMO_CHECKLIST_CN.md"),
        "chinese_project_onepager": Path("docs/DEEPUplift_AGENT_ONEPAGER_CN.md"),
        "resume_snapshot": Path("docs/RESUME_DEEPUplift_AGENT_SNAPSHOT.md"),
        "architecture": Path("docs/DEEPUplift_AGENT_ARCHITECTURE.md"),
        "agent_v2_architecture": Path("docs/DEEPUplift_AGENT_2_0_ARCHITECTURE.md"),
        "production_roadmap": Path("docs/DEEPUplift_AGENT_PRODUCTION_ROADMAP.md"),
        "model_backend_strategy": Path("docs/DEEPUplift_AGENT_MODEL_BACKEND_STRATEGY.md"),
        "model_readiness_trend": Path("docs/DEEPUplift_AGENT_MODEL_READINESS_TREND.md"),
        "offline_online_playbook": Path("docs/DEEPUplift_AGENT_OFFLINE_ONLINE_PLAYBOOK.md"),
        "industrial_case_studies": Path("docs/INDUSTRIAL_UPLIFT_CASE_STUDIES.md"),
        "uplift_business_scenarios": Path("docs/UPLIFT_BUSINESS_SCENARIOS.md"),
        "uplift_industry_deep_dive": Path("docs/UPLIFT_INTERVIEW_INDUSTRY_DEEP_DIVE.md"),
        "uplift_model_selection": Path("docs/UPLIFT_MODEL_SELECTION_PLAYBOOK.md"),
        "coupon_ads_growth_playbook": Path("docs/UPLIFT_COUPON_ADS_GROWTH_PLAYBOOK.md"),
        "online_incrementality": Path("docs/UPLIFT_ONLINE_EXPERIMENT_AND_INCREMENTALITY.md"),
        "industry_references": Path("docs/UPLIFT_INDUSTRY_REFERENCES.md"),
        "risk_register": Path("docs/DEEPUplift_AGENT_RISK_REGISTER.md"),
        "llm_uplift_frontier": Path("docs/UPLIFT_LLM_CAUSAL_FRONTIER.md"),
        "llm_routing_policy": Path("docs/LLM_ROUTING_POLICY_PLAYBOOK.md"),
        "llm_interview_qa": Path("docs/UPLIFT_LLM_INTERVIEW_QA.md"),
    }
    latest_run_dirs = [
        path
        for path in runs_dir.glob("*")
        if path.is_dir() and (path / "metrics.json").exists() and (path / "config.json").exists()
    ]
    artifacts["latest_run_evidence"] = max(latest_run_dirs, key=lambda path: path.stat().st_mtime) if latest_run_dirs else None

    audit = read_json(artifacts["model_catalog_audit"])
    regression = read_json(artifacts["agent_regression"])
    compare = read_json(artifacts["no_ui_compare_manifest"])
    interview_audit = read_json(artifacts["interview_materials_audit"])

    artifact_rows = "\n".join(
        f"| {name} | `{path}` | {file_status(path)} |"
        for name, path in artifacts.items()
    )
    difficulty_rows = "\n".join(
        "| {id} | {topic} | {tabs} | {proof} | {artifact} |".format(**row)
        for row in DIFFICULTY_ROWS
    )
    talk_rows = "\n".join(
        f"### {row['id']} {row['topic']}\n\n{row['talk']}\n"
        for row in DIFFICULTY_ROWS
    )
    ready_sources = audit.get("ready_source_counts") or {}
    ready_sources_text = ", ".join(f"{key}={value}" for key, value in sorted(ready_sources.items())) or "NA"
    best_regression = {}
    regression_results = regression.get("results") or []
    if regression_results:
        best_regression = max(regression_results, key=lambda row: row.get("qini") or float("-inf"))
    compare_candidates = compare.get("ranking") or []

    markdown = f"""# DeepUplift Agent Interview Evidence Pack

Generated at: `{generated_at}`

This pack maps the interview story to live product surfaces, local commands, and generated artifacts. Use it when a reviewer asks, "Can you prove this system actually supports the hard parts you described?"

## Current Proof Snapshot

| Metric | Value |
| --- | --- |
| Registered models | {audit.get("registered_models", "NA")} |
| Runnable models | {audit.get("ready_models", "NA")} |
| Ready backends | {ready_sources_text} |
| Regression runs | {len(regression_results)} |
| Compare candidates | {len(compare_candidates)} |
| Interview audit status | {interview_audit.get("status", "NA")} |
| Best smoke model | {best_regression.get("model", "NA")} |
| Best smoke QINI | {best_regression.get("qini", "NA")} |
| Best smoke AUUC | {best_regression.get("auuc", "NA")} |

## Difficulty-To-Evidence Matrix

| ID | Interview Difficulty | UI Tabs To Show | What Proves It | Artifact |
| --- | --- | --- | --- | --- |
{difficulty_rows}

## Demo Commands

```bash
scripts/deepuplift_agent.sh start
scripts/deepuplift_agent.sh story
scripts/deepuplift_agent.sh demo
PYTHON_BIN=python3 ROWS=180 SENSITIVITY_SAMPLES=1 APP_URL=http://localhost:8501 scripts/full_regression_check.sh
```

## Evidence Files

| Artifact | Path | Status |
| --- | --- | --- |
{artifact_rows}

## How To Answer Follow-Ups

{talk_rows}
## Interview Close

The strongest project framing is: I built a decision workbench, not a model zoo. The system diagnoses causal design, chooses suitable uplift learners, evaluates ranking and policy value, exports scoring artifacts, and keeps the demo honest through evidence bundles and regression gates.
"""
    payload = {
        "status": "ok",
        "generated_at": generated_at,
        "registered_models": audit.get("registered_models"),
        "ready_models": audit.get("ready_models"),
        "ready_sources": ready_sources,
        "regression_runs": len(regression_results),
        "compare_candidates": len(compare_candidates),
        "interview_audit_status": interview_audit.get("status"),
        "artifacts": {name: str(path) if path else "" for name, path in artifacts.items()},
        "difficulties": DIFFICULTY_ROWS,
    }
    return markdown, payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an interview evidence pack from latest DeepUplift artifacts.")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--output", default="docs/DEEPUplift_AGENT_INTERVIEW_EVIDENCE_PACK.md")
    parser.add_argument("--json-output", default="reports/interview_evidence_pack_latest.json")
    args = parser.parse_args()

    markdown, payload = build_pack(Path(args.reports_dir), Path(args.runs_dir))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")
    json_output = Path(args.json_output)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "output": str(output), "json_output": str(json_output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
