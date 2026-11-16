# DeepUplift Project Status

**Last updated:** 2026-11-16  
**Repository:** `jd-SearchEngines/deepuplift-agent`  
**Current evidence branch:** `codex/issue12-long-term-runner`  
**Current PR:** [#13 — make autonomous research runner continuous](https://github.com/jd-SearchEngines/deepuplift-agent/pull/13) (OPEN, DRAFT; not merged)

> This file is the shared entry point for future GPT/Codex sessions. Read it before proposing or executing work, then inspect the linked phase reports and current GitHub state. Update this file after every meaningful experiment, phase gate, blocker, or scheduling-system test. Do not rely on chat history as the sole source of project state.

## 1. How to read the roadmap: two plans, not one

There are two related but different plans:

1. **Original nine-stage DeepUplift product/research roadmap (P1–P9):** the overall causal-decision platform, from contracts and data through models, evaluation, policy, OPE, benchmarks, and productization.
2. **Push MRT focused benchmark plan (P0–P6):** a later experiment programme used to validate advanced causal identification, continuous-dose modelling, monotonic calibration, and constrained optimization.

The Push MRT P0–P6 plan is a focused workstream inside the wider platform. It does not replace or renumber the original nine stages.

## 2. Original nine-stage roadmap

| Stage | Name | What it tests | Acceptance evidence |
|---|---|---|---|
| P1 | Contracts | Whether `CausalDataset`, `EffectPrediction`, and `PolicyResult` provide stable, consistent contracts across data, models, and decisions | Contract tests, serialization/validation tests, backwards compatibility |
| P2 | Data | Whether randomized-trial (RCT) and observational data can enter a common pipeline without losing assignment, propensity, provenance, or diagnostics | Loaders, schema/readiness checks, missingness, balance, overlap, leakage and provenance reports |
| P3 | Binary Model | Whether S/T/X learners, DR learners, causal forests and supported baselines estimate binary-treatment effects correctly | Finite predictions, known-truth synthetic tests, randomized/observational benchmark evidence |
| P4 | Multi + Continuous | Whether multi-arm treatments and continuous treatment/dose-response can be modelled correctly | Multi-arm contract and evaluation; dose-response curves; support guards; known-truth curve metrics |
| P5 | Evaluation | Whether ranking, effect magnitude/calibration, and uncertainty are evaluated separately and correctly | Qini/AUUC, uplift@K, calibration/error metrics, bootstrap intervals and explicit estimands |
| P6 | Decision | Whether effect estimates can be converted into business actions under cost, budget, capacity, eligibility and frequency constraints | Top-K/budget allocation, feasible policy output, objective and constraint diagnostics |
| P7 | OPE | Whether a candidate policy can be evaluated offline without deploying it, using IPS, SNIPS, DR and sensitivity checks | Estimator definitions, overlap/weight diagnostics, uncertainty, sensitivity and policy-value reports |
| P8 | Benchmark | Whether methods are reproducible and comparable across synthetic known-truth, IHDP and marketing-uplift datasets | Fixed seeds/configs, dataset provenance, common metrics, benchmark tables and limitations |
| P9 | Product | Whether the research workflow is usable as a product through CLI, Streamlit, examples and documentation | End-to-end runnable examples, documented interfaces, smoke tests and release checks |

### Roadmap logic

- P1–P2 establish reliable inputs and contracts.
- P3–P4 estimate treatment effects for binary, multiple, and continuous treatments.
- P5 checks whether those estimates are useful and trustworthy.
- P6 converts estimates into constrained decisions.
- P7 evaluates policies offline.
- P8 tests reproducibility across datasets.
- P9 packages the workflow for users.

## 3. Current focused workstream: Push MRT P0–P6

Detailed specification: [PUSH_CAUSAL_BENCHMARK_PLAN.md](benchmarks/PUSH_CAUSAL_BENCHMARK_PLAN.md)

| Phase | Current status | What has been demonstrated | Main boundary |
|---|---|---|---|
| P0 — Data audit | PASS for supplied local snapshot | MRT loader/schema, assignment mapping, arm probabilities, repeated-participant metadata, outcome windows, audit and participant split | Upstream license remains unverified; raw data stays local |
| P1 — Real randomized baseline | PASS for supplied local snapshot | Separate outcome-specific ITT analyses, known assignment probabilities, participant-cluster bootstrap, participant-held-out context analysis | Both reported 95% CIs include zero; no business-lift claim; no individual PEHE on real MRT |
| P2 — Bias/identification | PASS for reproducible 5,000-row semi-synthetic suite | Measured confounding, MAR selection, overlap failure, hidden confounding, MNAR sensitivity with known truth | Semi-synthetic evidence only; arbitrary hidden confounding is not identified |
| P3 — Multi-treatment/continuous dose | PASS for recorded runs | Three-arm MRT handling; monotone, saturating and fatigue dose-response comparisons | Continuous suite is semi-synthetic; no VC-Transformer reproduction claim; respect dose support |
| P4 — Monotonic calibration | PASS for recorded comparison | No correction, PAV, monotonic penalty and Laplacian smoothing compared on monotone and fatigue curves | Monotonicity can damage non-monotone/fatigue truth |
| P5 — Constrained optimizer | PASS for feasibility/benchmark execution; convergence remains solver-specific | Exact small-instance reference, independent feasibility checks, 1K/10K/100K five-seed runs, 500K/1M isolated stress | Large-scale exact optimality gap unavailable; Primal-Dual/Line Search/Coordinate convergence not universal |
| P6 — X5 external observational data | OPTIONAL / NOT STARTED | No formal experiment yet | Data access and assignment mechanism must be verified; do not call it randomized |

### Evidence reports

- [P0 data audit](benchmarks/PUSH_CAUSAL_P0_AUDIT.md)
- [P1 randomized baseline](benchmarks/PUSH_CAUSAL_P1_BASELINE.md)
- [P2 semi-synthetic bias benchmark](benchmarks/PUSH_CAUSAL_P2_SEMISYNTHETIC.md)
- [P3 multi-treatment and continuous dose](benchmarks/PUSH_CAUSAL_P3_CONTINUOUS.md)
- [P4 monotonic calibration](benchmarks/PUSH_CAUSAL_P4_CALIBRATION.md)
- [P5 constrained optimizer](benchmarks/PUSH_CAUSAL_P5_OPTIMIZER.md)

### Key findings to preserve

1. **P1:** Activity assignment vs none had ITT +6.287 steps (95% CI [-3.402, 17.119]); dietary assignment vs none had +0.0113 low-sodium choices (95% CI [-0.0307, 0.0466]). Both intervals include zero.
2. **P2:** DR bias was small in the measured-confounding semi-synthetic case (-0.00173) and overlap-failure case (-0.00671), but remained large under hidden confounding (+0.34239). Correct selection weighting materially helped the known MAR selection scenario; it does not solve arbitrary MNAR.
3. **P3:** VCNet/GIKS variants performed well on some monotone/saturating synthetic curves, but fatigue remains harder. Do not extrapolate outside observed dose support.
4. **P4:** PAV reduced monotonic violation to zero on monotone truth, but increased fatigue MISE from 0.00876 to 0.01938. Monotonic constraints must be justified by the domain.
5. **P5:** At 1M candidates, Greedy took 1.943s, Heap 4.621s, Primal-Dual 75.658s, Line Search 20.779s, Coordinate 185.404s. Primal-Dual had the best feasible heuristic objective in that run, but hit its iteration budget; this is not proof of convergence or global optimality.

## 4. Current metrics PR

[PR #5 — feat: strengthen causal benchmark metrics](https://github.com/jd-SearchEngines/deepuplift-agent/pull/5)

**Status:** OPEN, DRAFT, not merged. The PR adds/strengthens:
- explicit Qini/AUUC metric definitions and identifiable-support metadata;
- ranking bootstrap stability;
- fixed-budget Top-K IPW policy value for binary treatment;
- separate continuous response MISE and effect MISE;
- numerical validation and tests.

PR description reports **109 passed, 2 skipped** in the project Python 3.11 environment, and the latest inspected CI workflows passed.

### Required follow-up review before treating the metric layer as complete

1. Add a hand-calculated tiny fixture to verify Qini/AUUC curve points and area conventions.
2. Add participant-cluster bootstrap support for repeated-decision MRT data; row bootstrap is not sufficient for clustered longitudinal observations.
3. Add a multi-arm policy-value estimator using the actual arm assignment probabilities for None/Activity/Diet; the current fixed-budget policy-value function is binary-treatment only.
4. Keep randomized, observational, semi-synthetic known-truth, and synthetic scale-stress evidence labels separate.

## 5. Lagrangian multipliers — plain-language reference

A Lagrange multiplier is a **dynamic shadow price for a scarce resource or binding constraint**.

For example, a candidate action has incremental business value and consumes budget, a time slot, user contact frequency, or risk capacity. The optimizer calculates an adjusted value:

`adjusted_value = incremental_value - direct_cost - budget_price × budget_used - slot_price × slot_capacity_used - frequency_price × frequency_used - risk_price × risk_used`

Interpretation:

- If total budget is being exceeded, increase the budget multiplier. Budget-consuming actions become less attractive.
- If a time slot is over capacity, increase that slot's multiplier. Actions in that slot are penalized.
- If a user's contact frequency is too high, increase the frequency multiplier. Further contacts to that user become less attractive.
- If a constraint has slack and its resource is underused, its multiplier can decrease.

Each multiplier is the algorithm's current price for consuming one more unit of that constrained resource. Recompute adjusted values and allocations, inspect constraint violations, update prices, and repeat until a stopping condition is met.

Eligibility is normally a hard filter: ineligible actions are removed rather than softly penalized by a multiplier.

**Important special case:** with only one global budget and equal treatment cost per candidate, sorting by value and selecting Top-K is sufficient. A Lagrangian solver is not needed merely for that case. It becomes useful when several coupled constraints (budget, slot capacity, frequency, quotas, risk) interact.

## 6. Autonomous orchestration status

**Status: NOT VERIFIED from repository evidence.**

A PR, commit, or passing CI proves code was submitted/tested; it does not prove Hermes is autonomously scheduling Codex or recovering long-running tasks.

Do not mark orchestration PASS until there is reproducible local evidence for:
1. Hermes creates a multi-stage task and persists its DAG/state.
2. Hermes dispatches a Codex worker and records task/attempt/commit identifiers.
3. Worker completion automatically unlocks the next dependent stage.
4. A forced process interruption followed by restart resumes from the last checkpoint without duplicating completed work.
5. A failed validation is recorded as REJECTED/BLOCKED and routed back for repair, not silently marked complete.
6. Final evidence and PR links are written back to the project state.

Required evidence artifact: a timestamped orchestration run report containing task ID, stage IDs, state transitions, worker invocations, checkpoint/recovery events, validation results, and Git commit/PR links. Until this exists, status remains NOT VERIFIED.

## 7. Next execution order

1. **Complete PR #5 metric review:** hand-calculated Qini/AUUC, participant-cluster bootstrap, and multi-arm policy value.
2. **Continue P5 optimizer:** improve convergence diagnostics and stopping criteria; distinguish best-feasible heuristic quality from a certified optimality gap.
3. **Verify Hermes orchestration:** run the interruption/restart and failed-review recovery tests; publish the run report here.
4. **Only then expand optional P6/X5** after data access, provenance and assignment mechanism are verified.

## 8. Mandatory progress-update protocol

After every meaningful run or phase transition, update this file and the relevant phase report with:

- date/time, branch and commit SHA;
- dataset class, source/provenance and data fingerprint (never raw personal rows);
- exact command/config, model version, seed and environment;
- metrics, uncertainty and constraint diagnostics;
- test/CI results;
- what the result supports and what it does not support;
- phase gate: PASS / PARTIAL / BLOCKED / NOT STARTED;
- remaining blocker and the next concrete task.

Never fabricate data or metrics. Never treat CI success as proof of causal validity, production readiness, or autonomous orchestration. Keep raw datasets, participant-level records, credentials, model weights and local caches out of Git.

## 10. Issue #12 long-term autonomous research runner (2026-11-16)

Work is being continued on branch `codex/issue12-long-term-runner` from the validated Batch 3 runner. This section records only the current evidence for Issue #12; the historical Batch 3 reports above remain unchanged.

### Engineering changes in this phase

- `ResearchPlanner` now selects the next eligible research task from persisted stage/report evidence instead of recreating a fixed Batch 3 DAG.
- The supervisor re-plans before and after every cycle, including idle cycles, and persists the public Hermes task ID alongside the external SQLite ledger.
- State and logs use `~/Library/Application Support/DeepUplift/runner/` by default; `plan.sh`, `logs.sh`, `status.sh`, `resume.sh`, `stop.sh` and `install_launchd.sh` are provided.
- The existing official `openai-codex` SDK worker and Hermes public CLI adapter remain the execution path. No Hermes private database is opened or modified.

### New research evidence

- [Sequential DR report](benchmarks/SEQUENTIAL_DR_REPORT.md) / [JSON](benchmarks/SEQUENTIAL_DR_REPORT.json): 5 fixed seeds on known-truth trajectories; true continuation-value Sequential DR is separated from ordinary per-decision DR. Synthetic gate PASS; real MRT BLOCKED; online lift NOT_RUN.
- [Multi-arm candidate chain report](benchmarks/MULTIARM_PUSH_CHAIN_REPORT.md) / [JSON](benchmarks/MULTIARM_PUSH_CHAIN_REPORT.json): 5 fixed seeds; three-arm logged arm probabilities, cluster bootstrap, candidate construction and constrained optimizer are connected on a known-truth Push-shaped DGP. Semi-synthetic gate PASS; real MRT BLOCKED; online lift NOT_RUN.

### Honest gates at the start of this phase

| Gate | Status | Evidence boundary |
|---|---|---|
| Hermes local installation | PASS | Hermes Agent v0.23.1 is installed at the user-local path; credentials are not reported |
| Official Codex SDK import | PASS | `openai_codex` imports in the validated Python 3.11 SDK environment |
| Dynamic phase-aware planner | PASS (code/tests) | Planner chooses the next evidence-backed task; long-lived multi-phase execution is separately verified below |
| Stable local state path | PASS (code) | Default is the macOS Application Support runner directory |
| launchd service | NOT_INSTALLED at initial audit | Must be verified with `launchctl print` after installation |
| Sequential DR known-truth experiment | PASS | 5 seeds and continuation-value diagnostics; no real MRT claim |
| Multi-arm candidate → optimizer → OPE | PASS | Oracle-effect semi-synthetic diagnostic; not a fitted-model claim |
| Real myBPmyLife MRT | BLOCKED | No locally authorized raw archive was available in this run |

The observed launchd, interruption/recovery and multi-stage evidence is recorded in [Issue #12 long-term runner report](benchmarks/ISSUE12_LONG_TERM_RUNNER_REPORT.md) and its [machine-readable JSON](benchmarks/ISSUE12_LONG_TERM_RUNNER_REPORT.json). The optimizer Worker remains in progress; autonomous continuous execution therefore remains **PARTIAL**, not PASS.


## 9. 2026-11-16 end-to-end chain audit

This section supersedes any interpretation that P0–P5 completion means the complete causal-decision workflow has been validated.

### 9.1 What currently runs end to end

The repository records a **semi-synthetic engineering smoke chain**:

`MRT-context DGP → participant-group OOF proximal effect model → model-produced candidates → constrained optimizer → sequential OPE diagnostics`

Evidence: [PUSH_END_TO_END_RESULTS.md](push/PUSH_END_TO_END_RESULTS.md), [PUSH_OPTIMIZATION.md](push/PUSH_OPTIMIZATION.md), [PUSH_OPE.md](push/PUSH_OPE.md).

| Link | Current status | What the evidence actually means |
|---|---|---|
| Canonical decision-point schema | PASS | Schema/engineering tests pass; does not establish raw real-MRT availability |
| Longitudinal fatigue/carryover DGP | PASS | Semi-synthetic known-truth generator and smoke run work |
| Participant-group proximal effect model | PASS on synthetic; real fixture NOT_AVAILABLE | A WCLS-style OOF model runs on generated data; not validated on the real MRT archive |
| Model prediction → candidate builder | PASS | Model-produced predictions are converted into explicit user/slot/action candidates |
| Candidate builder → constrained optimizer | PASS | All five tested solver paths returned feasible solutions on the 800-candidate smoke |
| Sequential IPS/SNIPS | PASS as API/smoke; support limited | Sequential estimators run, but trajectory support can collapse at longer horizons |
| Sequential DR | NOT_AVAILABLE | No continuation-value nuisance model is implemented/supplied |
| Real MRT long-term policy value | NOT_AVAILABLE | Real availability, exposure, delivery, time and outcome semantics are not established end to end |
| Real three-arm Activity/Diet policy through optimizer and OPE | NOT_RUN | Three-arm synthetic model tests exist, but the integrated Push policy chain is still binary/synthetic |

### 9.2 Important evidence reconciliation: myBPmyLife

There are two different run records and they must not be silently combined:

- The Push P0/P1 reports say PASS **for a supplied local upstream snapshot**.
- The formal Experiment A summary dated 2026-11-14 records myBPmyLife as `NOT_AVAILABLE`, with zero formal rows; its failure analysis says no raw locally licensed archive was available.

Until a run ID, source archive fingerprint, data dictionary version, and local availability record reconcile these entries, the overall **real Push data gate is PARTIAL / NOT VERIFIED**, not an unconditional PASS. No raw participant data should be committed to resolve this.

### 9.3 OPE and longitudinal limitations

- Sequential IPS/SNIPS API and cluster bootstrap are implemented.
- Five-seed trajectory-support stress: at horizon 8, finite trajectory SNIPS in 4/5 seeds with trajectory ESS 0–2; at horizon 16, finite estimates in 0/5 seeds and ESS 0. This is a support failure, not a successful long-horizon policy evaluation.
- One-step/per-decision DR is not full sequential DR. Full sequential DR remains `NOT_AVAILABLE` until a continuation-value model and its diagnostics exist.
- The learned proximal model has not yet been integrated with the fatigue/carryover longitudinal lab and a valid continuation-value DR pipeline.

### 9.4 Does the current test programme cover the entire causal chain?

**No. It covers a meaningful partial end-to-end smoke, not the full real-data causal decision chain.**

Covered in separate or semi-synthetic lanes:
- binary S/T/X/R/DR model comparison on synthetic and randomized Hillstrom/Criteo data;
- multi-treatment synthetic A2;
- continuous-dose synthetic A3, including monotone, saturating and fatigue shapes;
- measured-confounding, selection-bias, overlap and hidden-confounding stress tests;
- monotonic calibration;
- constrained allocation feasibility and scale tests;
- sequential IPS/SNIPS API and support diagnostics;
- one binary synthetic model-to-candidate-to-optimizer-to-OPE smoke.

Not yet covered as one integrated, decision-grade experiment:
1. a verified real Push MRT archive with assignment, eligibility, delivery/exposure, timestamp, availability and proximal outcome fields;
2. three-arm Activity/Diet effect predictions passed through the same candidate builder, constrained optimizer and policy evaluator;
3. learned proximal effects connected to fatigue/carryover and delayed outcomes;
4. sequential OPE with acceptable positivity/ESS and multi-seed bias/coverage evidence;
5. full sequential DR with a continuation-value model;
6. a policy confidence gate that blocks unsupported policies;
7. real randomized online policy validation. Offline OPE and semi-synthetic regret are not online lift.

### 9.5 Next integrated acceptance run

Create a single reproducible `push_e2e_v2` benchmark with separate evidence lanes:

- **Lane A — known-truth semi-synthetic:** five seeds; binary and three-arm actions; monotone/saturating/fatigue/carryover scenarios; participant-group splits; model → candidate → optimizer → OPE.
- **Lane B — randomized real data:** only run when the licensed MRT archive and its fingerprint are verified; estimate assignment-based proximal ITT and supported policy estimands; never invent individual counterfactual labels.
- **Lane C — observational stress:** explicit measured/unmeasured confounding, MAR/MNAR selection and overlap scenarios; report sensitivity and unsupported regions.

Minimum report fields:
- model: ATE bias, PEHE only where unit-level truth exists, CATE correlation, Qini/AUUC, uplift@K, calibration and interval coverage;
- data/identification: propensity distribution, overlap, ESS, balance, missingness and selection diagnostics;
- policy/OPE: IPS/SNIPS/DR only when supported, policy value and CI, importance-weight tails, cluster count, horizon support and regret where truth exists;
- optimizer: objective, budget/capacity/frequency violations, feasibility, solver stopping reason, dual diagnostics, runtime and memory;
- provenance: dataset class, source/version/fingerprint, seed, configuration, model version, commit and exact command.

Compare at least random, treat-none, treat-all where feasible, naive outcome ranking, uplift Top-K, cost-aware greedy, and constrained solver policies. A stage is PASS only if the experiment runs, its estimand is supported, uncertainty/diagnostics are reported, and limitations are explicit. A code smoke alone is not a causal-validity PASS.

### 9.6 CI snapshot before this documentation addendum

On commit `ff75e455893ec2fa2043aecc897fcebead068ece`, GitHub Actions runs `quality` (Python 3.10/3.11/3.12) and `continuous-frontier` completed successfully. The PR description records 109 passed and 2 skipped for the local Python 3.11 suite. These checks validate the tested code paths, not the missing real-data or full-chain evidence above. PR #5 remains OPEN / DRAFT / not merged.


## 10. Causal research execution and autonomous orchestration plan

Tracking issue: [#6 — Complete causal research phases with Hermes + Codex SDK orchestration](https://github.com/jd-SearchEngines/deepuplift-agent/issues/6)

Execution plan: [CAUSAL_RESEARCH_EXECUTION_PLAN.md](CAUSAL_RESEARCH_EXECUTION_PLAN.md)

Local executor prompt: [CAUSAL_RESEARCH_CODEX_EXECUTOR_PROMPT.md](CAUSAL_RESEARCH_CODEX_EXECUTOR_PROMPT.md)

**Recommended operating model:** Hermes is the persistent Supervisor (objective, DAG, task scheduling, retry/recovery, approval gates); the official Codex SDK is the coding/experiment Worker; deterministic tests and statistical gates plus a read-only reviewer validate outputs; GitHub is the shared evidence ledger.

The preferred Codex integration for this Python repository is the official Python SDK package `openai-codex`. It supports thread start/resume, async usage, turn streaming/interrupt and sandbox modes. The adapter must inspect the installed SDK version and use only its supported public API. Do not use the removed `codex mcp-server` integration.

### Execution order

1. Prove Hermes↔Codex orchestration reliability with a three-stage DAG, process interruption/restart, idempotent checkpoint recovery, dependency unlock and bounded failed-review recovery.
2. Close metric gaps: hand-calculated Qini/AUUC, participant-cluster bootstrap, multi-arm policy value.
3. Reconcile myBPmyLife archive provenance, license, dictionary version and run IDs.
4. Run integrated `push_e2e_v2` known-truth semi-synthetic lanes for binary and three-arm actions across five seeds.
5. Improve OPE support/ESS/coverage, implement a policy confidence gate and full sequential DR only after adding a continuation-value model.
6. Finish optimizer convergence and valid dual-bound evidence.
7. Complete unified P1–P9 benchmark matrix and product closeout; keep X5 optional.

### Orchestration gate

**Status: NOT VERIFIED.** At the 2026-11-16 audit the registered local Desktop Commander device was offline. No local Hermes/Codex execution is claimed from this session. Do not enable unattended research until the interruption/restart and failed-review recovery test has produced a timestamped local run report.

### Operating constraints

- One write-enabled Codex worker per worktree; read-only review may run separately.
- Use bounded tasks, persistent Codex thread IDs, heartbeat/checkpoint metadata and a local project execution ledger if Hermes native task metadata is insufficient. Do not mutate undocumented Hermes database tables.
- Maximum two automatic code-repair cycles after a deterministic validation failure; then BLOCKED.
- Stop a model/config sweep after three pre-registered attempts without practically meaningful improvement and record the negative result.
- No raw participant data, credentials, model weights or local caches in Git.
- No paid data/compute, online experiments, production changes, destructive operations or merge to main without explicit approval.
- Update this file and the phase report after every meaningful run.

## 11. 2026-11-16 Batch 0 Hermes + Codex SDK acceptance

**Gate status: PASS for the bounded local orchestration lane.** This does not authorize long-running causal experiments until the dependent data, metric, OPE and optimizer gates are independently satisfied.

Evidence: [HERMES_BATCH0_ORCHESTRATION_REPORT.md](benchmarks/HERMES_BATCH0_ORCHESTRATION_REPORT.md), [machine-readable run record](benchmarks/HERMES_BATCH0_ORCHESTRATION_REPORT.json), and [test results](benchmarks/evidence/HERMES_BATCH0_TEST_RESULTS.md).

What was actually verified on the local machine:

- Hermes `0.23.1` Kanban created a durable A → B → C DAG using the public `hermes kanban` interface; no Hermes private database tables were modified.
- Official `openai-codex` was installed in a separate Python 3.11 runtime and its public `Codex.thread_start`, `Thread.run`, and `Codex.thread_resume` APIs were exercised. Existing Codex ChatGPT authentication was reused; no credential was copied into the repository.
- The supervisor persisted project-specific thread/turn/checkpoint/heartbeat metadata in an external SQLite ledger; Hermes remained the task lifecycle owner.
- B was interrupted after `b_checkpoint_written`; restart idempotently skipped accepted A, resumed B on the same Codex thread, and unlocked C only after B validation passed.
- A validation rejection entered `REPAIR_READY` and was repaired on the same Codex thread. A separate repeated-failure case entered `BLOCKED` after three failed attempts / two automatic repair cycles.
- Adapter tests passed (`2 passed`); full project suite passed (`111 passed, 2 skipped`).
- Draft PR #8 is open; remote SHA was verified, and all Python 3.10/3.11/3.12 syntax plus continuous-frontier checks passed.

This gate proves a local bounded engineering workflow, not causal validity, real myBPmyLife availability, online lift, or unattended operation when the local machine/device is offline. The next eligible task is Gate 1 metric correctness review; do not launch the long causal sweep yet.

## 12. 2026-11-16 Gate 1 metric correctness

**Gate status: PASS for deterministic metric correctness; independent SDK review: NOT_COMPLETED due to timeout.** Report: [GATE1_METRIC_CORRECTNESS_REPORT.md](benchmarks/GATE1_METRIC_CORRECTNESS_REPORT.md), machine-readable result: [GATE1_METRIC_CORRECTNESS_REPORT.json](benchmarks/GATE1_METRIC_CORRECTNESS_REPORT.json).

The metric layer now has a hand-calculated Qini/AUUC fixture (`qini=0.625`, `auuc=3.625`), participant-cluster bootstrap support, and a multi-arm policy-value estimator using the logged per-row arm probability matrix. Focused tests passed (`10 passed`); the full Python 3.11 suite passed (`114 passed, 2 skipped`). The SDK read-only reviewer timed out and was not counted as a pass. No raw MRT data or business claims are attached to this gate.

The next eligible phase is Gate 2 Push data provenance reconciliation. Real-data and long-running causal experiments remain gated until archive source, license, fingerprint, dictionary version and run IDs agree.

## 13. 2026-11-16 Gate 2 Push data provenance

**Gate status: BLOCKED / NOT_AVAILABLE for real local myBPmyLife execution.** Report: [PUSH_DATA_PROVENANCE_RECONCILIATION.md](benchmarks/PUSH_DATA_PROVENANCE_RECONCILIATION.md), machine-readable result: [PUSH_DATA_PROVENANCE_RECONCILIATION.json](benchmarks/PUSH_DATA_PROVENANCE_RECONCILIATION.json).

The bounded local filename audit found no raw `Data_MRT.zip` or participant-level myBPmyLife archive. The upstream repository is public, but GitHub reports no license metadata or root license file; therefore the archive was not downloaded. Historical P0/P1 supplied-snapshot PASS evidence and the Experiment A zero-row `NOT_AVAILABLE` result remain separate and reconcile to `PARTIAL / NOT_VERIFIED`, not a real-data PASS.

The real Push lane is paused at the license/archive-fingerprint gate. Semi-synthetic known-truth work may continue; no raw data download, redistribution or long-running real MRT experiment will be started without authorized provenance evidence.

## 14. 2026-11-16 Autonomous Research Runner and Batch 3

This section supersedes the earlier orchestration `NOT VERIFIED` note for the
specific local run below; it does not claim that the whole causal research
programme is complete.

**Runner gate: PASS for the local engineering acceptance run.** Evidence:
[Autonomous Research Runner report](benchmarks/AUTONOMOUS_RESEARCH_RUNNER_REPORT.md),
[machine-readable runner record](benchmarks/AUTONOMOUS_RESEARCH_RUNNER_REPORT.json),
and [Batch 3 validation log](benchmarks/evidence/BATCH3_PUSH_E2E_VALIDATION.md).

- Hermes public Kanban created and persisted a three-task DAG: engineering →
  `push_e2e_v2` → independent report. Task IDs were `t_e7126026`,
  `t_2b94bd22`, and `t_57462294`.
- The official asynchronous Codex SDK produced streaming events while the
  Supervisor emitted Hermes heartbeats. The engineering task encountered a
  real shell failure (`python` unavailable), entered `REPAIR_READY`, then
  passed after the bounded repair using the same Codex thread. No failure was
  silently converted to PASS.
- A separate forced-kill test returned `-9`; the resumed worker reused the
  original Codex thread and completed from the persisted checkpoint. Completed
  stages were not duplicated.
- Batch 3 completed the known-truth semi-synthetic lane across five seeds:
  20 model rows, 25 optimizer rows, 25 OPE rows, and a separate existing
  three-arm comparison. The experiment gate is PASS for this synthetic lane;
  the research gate remains PARTIAL because the real MRT lane is blocked and
  full Sequential DR lacks a continuation-value model.
- Independent read-only review accepted the evidence only as PARTIAL synthetic
  evidence, then the report provenance/test-pass issues were repaired and the
  focused validation returned `7 passed`.

Draft PR [#11](https://github.com/jd-SearchEngines/deepuplift-agent/pull/11)
contains the runner and Batch 3 evidence; it is intentionally not merged.

Launchd is **NOT_INSTALLED**. The checked-in template and
`start.sh`/`stop.sh`/`status.sh`/`resume.sh` passed shell validation; service
installation remains an explicit local deployment step after review.

Next eligible tasks: add a continuation-value model for full Sequential DR,
add multi-arm candidate/OPE integration, complete optimizer convergence and
dual-bound evidence, and reopen real MRT only after the license/archive gate.
