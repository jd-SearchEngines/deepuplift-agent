# DeepUplift causal research execution plan

**Owner:** Hermes Supervisor  
**Coding executor:** Official OpenAI Codex SDK (Python preferred)  
**Evidence system of record:** GitHub  
**Status:** PLAN — orchestration is not yet verified on the local machine  
**Tracking issue:** https://github.com/jd-SearchEngines/deepuplift-agent/issues/6

## 1. Objective

Finish the original DeepUplift P1–P9 roadmap and the focused Push MRT P0–P6 benchmark programme. Do not confuse implementation completion with causal validation. Every phase must end with an explicit evidence-based gate: PASS, PARTIAL, BLOCKED, NOT_AVAILABLE, or NOT_STARTED.

The target operating loop is:

Research question → phase/task DAG → bounded Codex implementation or experiment → deterministic validation → independent review → evidence report → GitHub update → next eligible task.

The system should keep working while the local machine and required services are available. It must stop safely at data-license, credential, paid-compute, production, online-experiment, destructive-operation, or merge-to-main approval gates.

## 2. Division of responsibilities

### Hermes Supervisor

Hermes owns:
- the long-running project objective and task decomposition;
- phase dependencies, task prioritization and scheduling;
- dispatching Codex workers and tracking their lifecycle;
- retries, timeouts, heartbeats, checkpoints and restart recovery;
- deciding whether a task is ACCEPTED, REJECTED, BLOCKED or needs repair;
- escalating only decisions that require the user.

Hermes should use its supported task/Kanban and delegation interfaces. Do not write directly into undocumented Hermes database tables or patch Hermes internals.

### Codex SDK Worker

Use the official Codex SDK as the coding and experiment worker. Prefer the Python SDK because DeepUplift is a Python research repository and the current official SDK supports Python 3.10+, synchronous and asynchronous clients, thread start/resume, streaming turn handles, interruption, and read-only/workspace-write sandbox modes.

Recommended package: openai-codex. Use the installed SDK's actual public API and inspect its version before coding the adapter. Persist the Codex thread ID and the current turn/task mapping. Reuse a thread for follow-up work on the same task; do not create a new thread for every retry if the previous thread can safely be resumed.

Use workspace-write only for implementation tasks. Use read-only for independent review. Do not use unrestricted filesystem access as the default.

### Independent Validator

Validation is not delegated to the coding worker alone. Use:
1. deterministic pytest, type/syntax, schema and benchmark checks;
2. statistical acceptance scripts that verify estimands, support and uncertainty;
3. a separate read-only Codex review as a secondary review, not as a substitute for deterministic checks.

A successful process exit or passing unit tests does not by itself establish causal validity.

### GitHub

GitHub is the cross-session source of truth for:
- phase status and acceptance criteria;
- task issues and pull requests;
- commit SHAs and code review;
- exact commands, configs, seeds and evidence references;
- blockers, decisions and next tasks.

Local SQLite is runtime state only. It must never become the only copy of project progress.

## 3. Runtime and recovery contract

Use Hermes's native task system for the user-facing task board. If the native task system cannot represent the required project run metadata, add a small project adapter with a local SQLite ledger outside the repository (for example under the user's home directory). Do not duplicate the whole Hermes scheduler.

Each execution record must persist:
- project_task_id and phase_id;
- task status and dependency IDs;
- attempt number and bounded retry count;
- Codex thread ID and turn ID when available;
- Git branch and worktree path;
- start time, heartbeat time and last checkpoint;
- exact command/configuration, seed and dataset fingerprint;
- validation status and machine-readable result path;
- commit SHA, PR URL and final evidence links.

State machine:
- READY → RUNNING → VERIFYING → ACCEPTED;
- VERIFYING → REJECTED → REPAIR_READY → RUNNING;
- any state → BLOCKED when a human decision, missing data, unsupported estimand or external dependency is required;
- interrupted RUNNING tasks resume from the last durable checkpoint after restart.

Idempotency requirements:
- completed phases must not be rerun merely because Hermes restarted;
- every expensive benchmark is keyed by dataset fingerprint + config hash + seed + code commit;
- before rerunning, inspect whether a complete artifact already exists;
- never overwrite a previous run; create a new run ID and retain the old evidence;
- do not mark a task complete until its report and GitHub state update are durable.

Retry policy:
- retry transient infrastructure failures with bounded backoff;
- allow at most two automatic code-repair cycles after a deterministic validation failure;
- after the second failed repair, mark BLOCKED and publish the failure evidence;
- do not repeatedly tune models without a predeclared experiment budget and acceptance criterion.

Concurrency:
- one write-enabled Codex worker per repository worktree at a time;
- read-only review may run separately;
- run memory-intensive benchmarks sequentially unless measured memory headroom supports parallel execution;
- use isolated Git worktrees for independent code tasks.

## 4. Original nine-stage roadmap and gates

| Stage | Focus | Required acceptance evidence | Current starting point |
|---|---|---|---|
| P1 Contracts | Dataset, prediction and policy contracts | schema/serialization/validation/backward-compatibility tests; stable public API | Core contracts exist; verify remaining gaps |
| P2 Data | RCT and observational ingestion | provenance, assignment/propensity, missingness, balance, overlap, leakage and readiness reports | Several loaders exist; Push provenance needs reconciliation |
| P3 Binary Model | S/T/X/R/DR and supported baselines | known-truth tests, held-out evaluation, randomized and observational evidence with correct estimands | Broad model comparison already run |
| P4 Multi + Continuous | Multi-arm and continuous treatment | three-arm treatment contracts, generalized propensity where justified, dose-response support guards and known-truth curves | Synthetic suites exist; integrated Push chain remains incomplete |
| P5 Evaluation | Ranking, calibration and uncertainty | hand-checked Qini/AUUC, uplift@K, calibration, participant-cluster uncertainty and coverage | Metric PR adds several pieces; follow-up gaps remain |
| P6 Decision | Cost/budget/capacity/frequency constrained decisions | common candidate set, feasibility, objective, constraints, solver stopping and small-instance optimality gap | Optimizers run; large-scale convergence/dual bound remain open |
| P7 OPE | Offline policy evaluation | IPS/SNIPS/DR only when assumptions/support pass; weight tails, ESS, cluster CI, sensitivity and policy confidence gate | Sequential IPS/SNIPS exist; long-horizon support and sequential DR remain open |
| P8 Benchmark | Reproducible cross-dataset comparisons | fixed configs/seeds, dataset evidence classes, common metrics, benchmark matrix and limitations | Criteo/Hillstrom/synthetic lanes exist; unified matrix not complete |
| P9 Product | CLI, reports, examples and release | clean install, runnable examples, documented interfaces, CI and release checklist | Beta exists; update after the research gates close |

A phase may be marked PARTIAL or BLOCKED. Never convert unavailable data or unsupported estimands into a PASS.

## 5. Focused Push MRT P0–P6 workstream

The Push programme is a specialized validation lane inside the nine-stage roadmap.

| Phase | Work | Acceptance gate |
|---|---|---|
| P0 Data audit | Reconcile raw archive, license, dictionary, participant/decision IDs, timestamps, eligibility, assignment, delivery, assignment probabilities, outcome windows and label maturity | Fingerprinted local archive and reproducible audit; otherwise DATA_NOT_AVAILABLE / LICENSE_UNVERIFIED |
| P1 Randomized baseline | Outcome-specific ITT with known randomization probabilities and participant-cluster uncertainty | Correct assignment-based estimand, CI, balance and participant-held-out checks; no individual PEHE claim |
| P2 Identification stress | Measured confounding, MAR selection, MNAR sensitivity, overlap failure and hidden confounding | Five-seed known-truth matrix with bias, RMSE, coverage, ESS, balance, weight tails and sensitivity |
| P3 Multi-treatment/continuous | None/Activity/Diet and monotone, saturating and fatigue dose-response | Three-arm model path and continuous support diagnostics; no extrapolation outside dose support |
| P4 Calibration | No correction, PAV, monotonic penalty and graph/Laplacian smoothing | Report monotonic violation, curve error, calibration and policy impact on both monotone and fatigue truth |
| P5 Optimizer | Budget, slot capacity, frequency, spacing, quota and eligibility constraints | Independent feasibility; exact small reference; best-feasible heuristic comparison; convergence and dual-bound status |
| P6 X5 observational | Optional RetailHero SMS lane | Only run after data provenance and assignment mechanism audit; treat as observational unless randomization is proven |

P6/X5 is optional and must not block completion of the primary Push chain.

## 6. Execution sequence

### Gate 0 — Prove the autonomous execution system

Before launching a long research sweep:
1. Hermes creates a three-stage task with explicit dependencies.
2. Hermes dispatches a Codex SDK worker and persists thread/task/attempt identifiers.
3. Stage 1 completes and is checkpointed.
4. Stage 2 starts; forcibly stop the supervisor or worker process.
5. Restart Hermes and prove Stage 1 is not duplicated and Stage 2 resumes safely.
6. Complete Stage 2; verify Stage 3 is unlocked only after validation.
7. Force one validation failure and prove the task returns to a bounded repair cycle.
8. Force a second failure and prove it becomes BLOCKED rather than silently accepted.
9. Publish a timestamped orchestration report and GitHub links.

Until this test passes, orchestration status remains NOT_VERIFIED and long-running unattended research must not be enabled.

### Gate 1 — Close evaluation correctness gaps

Complete the outstanding metric review:
- hand-calculated tiny Qini/AUUC fixture, including edge cases with missing treatment arms;
- participant-cluster bootstrap for repeated-decision MRT data;
- multi-arm policy value using the actual None/Activity/Diet assignment probabilities;
- explicit randomized, observational, semi-synthetic known-truth and synthetic scale-stress evidence labels.

### Gate 2 — Reconcile Push data provenance

Inspect local data paths without uploading raw records. Record:
- exact upstream source and license/terms;
- archive filename, byte size and SHA-256 fingerprint;
- data dictionary version and field mapping;
- run ID, code commit, config and seed;
- whether the archive used in P0/P1 reports is the same archive used in formal Experiment A.

If the archive is absent or license is unclear, mark real-data P0/P1 BLOCKED/NOT_AVAILABLE and continue with synthetic work. Do not invent real results.

### Gate 3 — Build push_e2e_v2

Create one reproducible experiment entry point with three isolated evidence lanes:

**Lane A — Semi-synthetic known truth**
- five fixed seeds;
- binary and three-arm actions;
- monotone, saturating, fatigue and carryover scenarios;
- participant-group train/validation/test separation;
- proximal effect model → candidate builder → constrained optimizer → OPE;
- evaluate against known policy value/regret and report uncertainty.

**Lane B — Randomized real MRT**
- execute only when the licensed archive and provenance fingerprint are verified;
- preserve participant clustering, time, eligibility, assignment probability, delivery and outcome windows;
- use ITT for the randomized primary analysis;
- estimate only policy quantities identified by the logged design and observed support;
- do not fabricate unit-level counterfactual labels.

**Lane C — Observational stress**
- measured/unmeasured confounding, MAR/MNAR selection and positivity failures;
- compare adjustment methods and sensitivity analyses;
- explicitly mark unsupported identification regions.

All three lanes must share the same report schema but keep evidence classes separate.

### Gate 4 — Make OPE decision-grade

- add multi-arm OPE for the actual logged action probabilities;
- run multi-seed positivity and weight-tail stress;
- report per-decision and trajectory ESS separately;
- add participant-cluster bootstrap and coverage analysis;
- add a policy confidence gate that blocks policies when overlap, ESS, weight tails or uncertainty fail predeclared thresholds;
- implement a continuation-value nuisance model before claiming full sequential DR;
- compare IPS/SNIPS/DR under supported horizons and document where long-horizon estimation is not identified or numerically unstable.

Do not cap or clip weights and then report the result as if it were the original unbiased estimator. Report clipping/trimming as a distinct sensitivity variant.

### Gate 5 — Finish optimizer quality evidence

- compare all solvers on identical candidate arrays, objective, constraints and seeds;
- preserve exact MILP results for small instances;
- report feasibility, objective, best-feasible heuristic gap, convergence, stopping reason, dual residual and runtime/memory separately;
- implement a mathematically valid dual upper bound before reporting a primal-dual optimality gap;
- do not claim convergence just because the iteration budget ended;
- retain Top-K as the correct baseline for the single global budget/equal-cost special case.

### Gate 6 — Unified benchmark and product closeout

Generate one benchmark matrix mapping every P1–P9 and Push P0–P6 gate to:
- status;
- dataset/evidence class;
- code commit and config;
- seed set;
- metrics and uncertainty;
- limitations;
- report and artifact links;
- next task or reason blocked.

Then run clean-install and end-to-end CLI examples, update documentation and release notes, and run CI. Product/release completion does not imply online business lift.

## 7. Required report contract

Every benchmark report must include:
- run ID, date/time, branch and commit SHA;
- dataset class, source, version, fingerprint and license status;
- exact command, config, seed and environment;
- model/estimator version and split strategy;
- sample, participant and decision-point counts;
- estimand and identification assumptions;
- model metrics and uncertainty;
- propensity, overlap, balance, ESS, missingness and weight-tail diagnostics;
- policy value, CI, regret where known truth exists, and OPE support status;
- optimizer objective, feasibility, constraint residuals, convergence and stopping reason;
- runtime and peak memory;
- what the evidence supports and explicitly does not support;
- phase gate and next action.

Store machine-readable JSON alongside a human-readable Markdown report. Never commit raw participant rows, credentials, local caches or model weights.

## 8. Human approval gates

Stop and request approval before:
- downloading or redistributing data with unclear licensing;
- paid compute or unusually large model/data downloads;
- accessing credentials or changing authentication;
- online experiments, production changes or external customer contact;
- deleting or overwriting user data;
- merging a PR to main or publishing a release.

Normal local tests, bounded synthetic experiments, branch commits, draft PR creation and status updates may proceed without asking each time.

## 9. Current known blockers

1. Local Desktop Commander device was OFFLINE at the 2026-11-16 audit. Reconnect it before local Hermes/Codex execution.
2. Hermes→Codex dispatch, DAG dependency unlock, durable checkpoints, process restart and failed-review recovery are not verified for DeepUplift.
3. myBPmyLife availability/provenance differs between Push P0/P1 reports and the 2026-11-14 Experiment A summary; reconcile the exact archive and run.
4. Multi-arm policy value, participant-cluster bootstrap in the new metric layer and hand-calculated Qini/AUUC remain follow-up review items.
5. Integrated three-arm Push model-to-policy chain is not run.
6. Horizon-16 trajectory support fails in the current five-seed stress; full sequential DR is unavailable.
7. P5 large-scale optimizer convergence and a valid dual upper bound remain open.

## 10. Definition of done

The research programme is complete only when:
- each required P1–P9 and Push P0–P5 gate has an evidence-backed final status;
- all required reproducible benchmark lanes have machine-readable artifacts;
- any unavailable real-data or online-validation item is explicitly BLOCKED/NOT_AVAILABLE with a concrete reason, not silently called PASS;
- the integrated semi-synthetic three-arm end-to-end chain passes its predeclared acceptance criteria;
- OPE confidence gates prevent unsupported policies from being promoted;
- the orchestration kill/restart and failed-review recovery tests pass;
- GitHub PROJECT_STATUS.md points to all final reports and the remaining limitations.

## 11. 2026-11-16 runner and Batch 3 execution update

The earlier blockers about local orchestration are superseded for the bounded
run `batch3-front-end-20261116`: Hermes public Kanban, asynchronous official
Codex SDK streaming, heartbeat, same-thread recovery, bounded repair and an
independent read-only review were executed with evidence in
`docs/benchmarks/AUTONOMOUS_RESEARCH_RUNNER_REPORT.md/json`.

Batch 3 `push_e2e_v2` is **PASS for the known-truth semi-synthetic lane** across
five fixed seeds. The overall research gate remains **PARTIAL** because the
licensed/local real MRT lane is still blocked and full Sequential DR is not
available without a continuation-value model. Launchd is prepared but not
installed. The next eligible tasks are continuation-value OPE, multi-arm
candidate/OPE integration, and optimizer dual-bound/convergence evidence.

