# DeepUplift Project Status

**Last updated:** 2026-11-15  
**Repository:** `jd-SearchEngines/deepuplift-agent`  
**Current evidence branch:** `codex/hermes-metrics-pilot`  
**Current PR:** [#5 — strengthen causal benchmark metrics](https://github.com/jd-SearchEngines/deepuplift-agent/pull/5) (OPEN, DRAFT; not merged)

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
