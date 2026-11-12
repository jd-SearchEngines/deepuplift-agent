# Copilot / Codex Executor Prompt — Push MRT Causal Benchmark

You are working in the existing repository jd-SearchEngines/deepuplift-agent. Implement the specification in docs/benchmarks/PUSH_CAUSAL_BENCHMARK_PLAN.md.

## Mandatory first actions

1. Read README.md, docs/recovery/02_CODEX_RESTORE.md, docs/architecture/THREE_LAYER_PLATFORM.md, docs/benchmarks/BENCHMARK_PROTOCOL.md, docs/benchmarks/DATASETS.md, docs/models/CONTINUOUS_BENCHMARK_EVIDENCE_V06.md, docs/data/OBSERVATIONAL_PIPELINE.md, and docs/data/PROPENSITY_AND_OVERLAP.md.
2. Inspect the actual code and tests before editing. Reuse existing contracts, nuisance estimation, diagnostics, model adapters, continuous benchmark runner, evidence artifacts, and CLI conventions.
3. Work on branch codex/push-mrt-causal-benchmark (or create a child branch from it). Do not merge to main.
4. Do not rewrite existing modules wholesale. Preserve public APIs and backwards compatibility.
5. Do not commit raw datasets, participant-level rows, health data, model weights, caches, or generated run outputs.

## Primary objective

Implement the benchmark as gated phases P0 → P1 → P2 → P3 → P4 → P5. P6 (X5 SMS) is optional and must not block the Push MRT work.

### Dataset handling

Primary dataset: myBPmyLife MRT:
https://github.com/WIRED-L-um/MRT_myBPmyLife

The upstream README says each decision point has a 25% probability of any notification, split equally between activity and dietary notifications. The implied probabilities are 0.75 no notification, 0.125 activity, 0.125 dietary. Verify these values and field mappings against the actual data dictionary and source files before using them. Never infer randomization probabilities solely from observed treatment counts.

The study has repeated decision points per participant and two distinct outcomes:
- activity notification → steps in the next 60 minutes;
- dietary notification → lower-sodium food choice in the next 24 hours.

Preserve separate outcome windows. Treat participant as the cluster. Do not treat every decision row as an independent user. Assignment and delivery are separate variables; primary randomized analysis is ITT.

The raw source repository contains Data_MRT.zip and a data dictionary. Check upstream terms; store data only in a local ignored data directory. If the raw data is unavailable or terms cannot be confirmed, implement the loader contract, schema tests, download instructions, and synthetic fixture, and report DATA_NOT_AVAILABLE / LICENSE_UNVERIFIED. Do not fabricate real-data results.

Optional secondary dataset: X5 RetailHero SMS communication. Its public description documents treatment_flg and target, plus clients and pre-campaign purchases, but does not prove randomization. Treat as OBSERVATIONAL / ASSIGNMENT_UNVERIFIED unless primary campaign documentation proves otherwise. It is SMS communication, not a verified coupon offer. Do not use it as ground truth.

Existing Hillstrom, Criteo, generic retail, synthetic coupon, and continuous dose benchmarks must be reused, not duplicated.

### P0 — Loader and data audit

Implement a robust MRT-specific data contract/adapter. Inspect the raw data dictionary and notebooks; do not guess field names. Preserve participant ID, decision ID/time, eligibility/availability, assigned treatment, delivery, per-arm assignment probability, pre-treatment context/history, outcome name/value/window, label maturity, and provenance.

Audit duplicates, missingness, time ordering, outcome maturity, treatment counts, arm probabilities, balance, participant counts, decision counts, leakage, and participant-level split integrity.

If the existing CausalDataset cannot represent repeated decision metadata without losing information, add a separate longitudinal/MRT contract and a safe adapter to existing estimators. Do not flatten away participant/time/availability metadata.

### P1 — Real randomized Push baseline

Run outcome-specific ITT analyses using known randomization probabilities. Use participant-clustered bootstrap or a valid cluster-robust method. Include participant-held-out evaluation and a separate temporal split only if clearly labeled.

Do not calculate PEHE or individual ITE error on real MRT data. There is no per-decision observed counterfactual. Report design-based ITT and valid uncertainty. Any CATE/Qini/AUUC report must state its estimand and limitations.

### P2 — Bias stress tests with known truth

Build a semi-synthetic generator using real pre-treatment feature/context distributions but known potential outcomes generated before treatment assignment. Keep true effects out of model inputs.

Implement separate scenarios:
- measured confounding: treatment assignment depends on pre-treatment X/history;
- sample-selection bias: generate a full population then a selected sample with known selection probability; include MAR and an MNAR non-identifiability/sensitivity demonstration;
- overlap failure: arm-specific propensities near 0/1, with clipping, trimming, overlap weighting and ESS;
- hidden confounding: latent U affects treatment and outcome; demonstrate the limits of observed-X adjustment.

Compare unadjusted, outcome regression/T-Learner, IPW/stabilized IPW, overlap weighting, DR/R learners with cross-fitting, and existing deep candidates only when actually runnable. Include explicit selection weighting/modeling for SSB. MMOE or representation architecture alone is not a causal correction.

Evaluate ATE bias/RMSE, semi-synthetic-only PEHE, policy value/regret, CI coverage, ESS, weight tails, and balance. Record DGP truth and seed.

### P3 — Multi-treatment and continuous dose

For real Push, preserve three arms and known randomization probabilities. Do not route multi-arm data through binary propensity APIs. Implement or adapt generalized propensity only for observational multi-arm use.

For continuous frequency, create a semi-synthetic dose-response benchmark with known monotone, saturating, and non-monotone fatigue curves. Compare existing DoseResponseGBM, DRNet, VCNet and GIKS variants. Keep dose-support guards active.

Do not claim VC-Transformer is implemented unless you locate and document the exact paper/official implementation. Do not rename TransTEE or a generic Transformer interaction model as an exact VC-Transformer reproduction.

### P4 — Monotonic calibration

Compare no correction, PAV/isotonic, explicit monotonic constraints/penalty, and graph/Laplacian smoothness. Laplacian smoothing alone is not monotonicity. Include both monotonic and non-monotonic fatigue DGPs; reject a monotonic constraint when it damages known non-monotonic truth. Report violation rate, curve error, calibration, policy value and regret.

### P5 — Constrained optimizer

Current budget.py is only Top-K simulation. Add a separate optimizer API for binary candidate decisions x[user,time,action] with multiple coupled constraints: global budget, time-slot capacity, per-user rolling frequency cap, minimum interval, action quota, eligibility, and one action per decision point.

Objective: maximize predicted incremental business value, net of resource cost. Dual reduced utility must be inspectable. Compare Greedy Top-K, exact small-instance MILP/CP-SAT, basic primal-dual/subgradient, line-search primal-dual, heap-based updates, and coordinate-iterator dual updates.

All solvers must use identical candidates, value/cost matrices, constraints and seeds. Report objective, net value, feasibility, max constraint violation, dual residual, stopping reason, iterations, small-instance optimality gap, runtime, peak memory, and scale. Scale through 1K/10K/100K/1M; only attempt 10M after sparse-memory profiling. Include the single global budget / equal cost Top-K special case. Do not claim a Lagrangian solver is needed for that special case.

Treat these solver variants as documented research implementations inspired by the algorithm families, not exact reproductions of unpublished proprietary code.

### Evidence and quality

All raw data and generated run outputs stay outside Git. Use existing evidence patterns and write reports under an ignored local runs directory. Evidence must record dataset class, source, license, file hashes, git SHA, seeds, config, environment, model maturity, diagnostics, estimand, metrics, solver traces, limitations and warnings.

Separate evidence labels:
- REAL_RANDOMIZED_MRT
- REAL_OBSERVATIONAL_ASSIGNMENT_UNVERIFIED
- SEMI_SYNTHETIC_KNOWN_TRUTH
- SYNTHETIC_SCALE_STRESS

Add unit, integration, and CLI smoke tests. Run focused tests and the full relevant regression suite. Do not report PASS for unavailable raw data; report DATA_NOT_AVAILABLE while keeping synthetic tests runnable.

## Execution discipline

- Execute P0 first. Do not start advanced models or optimizer work until the data contract and audit pass.
- After each phase, write a concise evidence report and update the phase status in the plan.
- If a phase is blocked by missing source files, license uncertainty, unsupported upstream schema, or absent dependencies, record the exact blocker and continue only with independent synthetic work.
- Do not fabricate data, metrics, citations, or online lift.
- Do not remove existing functionality or alter the stable binary quickstart.
- Keep the changes small, testable, and reversible.
- At completion, report changed files, tests run, commands, results, data availability, license status, phase gates, known limitations, and the next concrete task.

Start by inspecting the branch and repository state, then implement P0 only. Once P0 is tested and its evidence report is saved, proceed to P1. Continue sequentially through the remaining phases only when their gates pass.
