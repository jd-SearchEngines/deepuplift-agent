# Push MRT Causal Benchmark & Optimization Validation Plan

**Status:** P0–P5 PASS for feasibility and benchmark execution; P5 solution quality/convergence are reported as separate evidence dimensions; P6 optional / not started; no production or causal-performance claim is made by this document.
**Target base:** `codex/v06-frontier-causal-models`  
**Work branch:** `codex/push-mrt-causal-benchmark`  
**Repository:** https://github.com/jd-SearchEngines/deepuplift-agent

## Execution status

| Phase | Status | Evidence |
|---|---|---|
| P0 | PASS for supplied local upstream snapshot | [`PUSH_CAUSAL_P0_AUDIT.md`](PUSH_CAUSAL_P0_AUDIT.md) |
| P1 | PASS for supplied local upstream snapshot | [`PUSH_CAUSAL_P1_BASELINE.md`](PUSH_CAUSAL_P1_BASELINE.md) |
| P2 | PASS for reproducible 5,000-row Push-derived semi-synthetic suite | [`PUSH_CAUSAL_P2_SEMISYNTHETIC.md`](PUSH_CAUSAL_P2_SEMISYNTHETIC.md) |
| P3 | PASS for real three-arm handling and Push-context continuous suite | [`PUSH_CAUSAL_P3_CONTINUOUS.md`](PUSH_CAUSAL_P3_CONTINUOUS.md) |
| P4 | PASS for monotone/fatigue calibration comparison | [`PUSH_CAUSAL_P4_CALIBRATION.md`](PUSH_CAUSAL_P4_CALIBRATION.md) |
| P5 | PASS for V1.1 O(N²) remediation, MILP/enumeration exact reference, independent feasibility, 1K/10K/100K five-seed stability, and 500K/1M isolated stress; convergence remains solver-specific | [`PUSH_CAUSAL_P5_OPTIMIZER.md`](PUSH_CAUSAL_P5_OPTIMIZER.md) |
| P6 | OPTIONAL / NOT_STARTED | — |

## 1. Objective

Extend the existing DeepUplift workbench with a reproducible, evidence-led benchmark for the advanced causal-decision workflow discussed in the project:

1. identify and diagnose confounding, sample-selection bias (SSB), and insufficient overlap;
2. compare weighting, outcome-model, doubly robust, representation-learning, and multi-task approaches;
3. evaluate multi-treatment and continuous-dose response models;
4. test monotonic calibration only where the domain supports a monotonicity assumption;
5. convert treatment-effect estimates into a constrained allocation problem and benchmark primal-dual solver variants;
6. keep randomized evidence, observational evidence, semi-synthetic ground truth, and scale stress tests strictly separated.

Reuse the existing Data → Models → Decision architecture and existing public benchmark APIs. Do not create a second causal platform or replace working model adapters.

## 2. Dataset decision

### 2.1 Primary real-world dataset: myBPmyLife Push MRT

Source repository: https://github.com/WIRED-L-um/MRT_myBPmyLife

The upstream repository contains:
- Data_MRT.zip (13,389,920 bytes at the time of inspection);
- myBPmyLife Data Dictionaries.xlsx;
- README and analysis notebooks.

The upstream README describes a micro-randomized push-notification trial. At each decision point, the probability of receiving any notification is 25%, split equally between activity and dietary notifications. The implied assignment probabilities are therefore 0.75 for no notification, 0.125 for activity notification, and 0.125 for dietary notification, subject to verification against the data dictionary and actual row-level assignment fields.

The published analysis reports 187,517 randomized decision points over six months, 298 participants in the MRT intervention cohort, and 287 participants with analyzable data. The study has two distinct proximal outcomes:
- activity notification → step count in the following 60 minutes;
- dietary notification → lower-sodium food choice in the following 24 hours.

Do not merge these outcome definitions into one generic label. Preserve the outcome-specific eligibility and observation windows.

Why this dataset is primary:
- real randomized push assignment with known design probabilities;
- repeated decision points per participant;
- multiple notification types;
- temporal context and proximal outcomes;
- suitable for validating repeated-treatment data contracts and randomized causal estimation.

Important limits:
- 187,517 decision rows are not 187,517 independent users; cluster by participant;
- this is not a randomized continuous-frequency experiment;
- real data does not expose each decision's individual counterfactual;
- raw data must remain local and must not be committed to Git;
- upstream license/redistribution terms must be checked before any redistribution. Treat the license as unverified until confirmed.

### 2.2 Secondary observational dataset: X5 RetailHero SMS communication

Competition/data page: https://ods.ai/competitions/x5-retailhero-uplift-modeling/data  
Reference loader documentation: https://www.uplift-modeling.com/en/latest/api/datasets/fetch_x5.html

Published dataset documentation describes:
- uplift_train.csv: 200,039 customer records with treatment_flg and target;
- clients.csv: 400,162 customers;
- purchases.csv: 45,786,568 pre-communication purchase records;
- treatment_flg: whether communication was performed;
- target: whether a purchase followed communication.

This is an SMS/marketing communication dataset, not a verified coupon dataset. The public description inspected does not establish randomized assignment. Therefore:
- classify it as observational / assignment mechanism unverified unless original campaign documentation proves randomization;
- do not use it as randomized ground truth;
- do not claim that IPW/DR recovers the true causal effect from this dataset;
- use it for covariate construction, observational diagnostics, overlap, sensitivity, and external robustness checks;
- raw files are manually obtained from the upstream competition page and must not be committed.

### 2.3 Existing datasets: do not duplicate

The current DeepUplift branch already includes:
- Hillstrom randomized email benchmark;
- Criteo randomized uplift loader and recorded benchmark;
- generic retail loader;
- deterministic binary synthetic ground-truth data;
- synthetic continuous-dose data and continuous benchmark runners;
- synthetic coupon-allocation example.

Hillstrom is email marketing, not a push trial or a coupon-redemption experiment. Keep its existing role and do not relabel it as coupon data. Existing Criteo evidence remains a separate advertising benchmark.

## 3. Repository audit and implementation boundaries

The latest frontier branch inspected is `codex/v06-frontier-causal-models` at commit `53b676dfd27aaf57935c56fc2f48881726bb6561`.

Existing capabilities confirmed on that branch:
- public dataset adapters: Hillstrom, Criteo, generic retail;
- observational nuisance estimation with cross-fitting;
- propensity diagnostics, common-support diagnostics, clipping/weighting metadata, ESS and balance reporting;
- controlled_observational_stress_test helper;
- DRNet, VCNet, GIKS continuous benchmark paths and continuous support guards;
- an experimental CCPFN inference adapter;
- evidence bundles and fixed-seed benchmark runners.

Gaps confirmed during the audit:
- the existing controlled_observational_stress_test changes treatment assignment but retains the observed outcome; it is not sufficient to claim known individual counterfactual ground truth for a causal recovery benchmark;
- the current nuisance propensity contract is binary-treatment oriented; push has three assignment arms and requires a multi-arm probability contract or use of known MRT probabilities;
- the current budget optimizer is a sorted Top-K budget simulation, not a primal-dual/Lagrangian solver;
- no PAV or Laplacian monotonic calibration implementation was found in the inspected tree;
- DRNet/VCNet are implemented, but the inspected evidence does not establish a runnable VC-Transformer implementation;
- TransTEE is marked optional/not validated in the existing continuous evidence. Do not silently equate TransTEE, VCNet+Transformer, and the CV-Transformer named in an external resume.

Keep the existing APIs backward-compatible. Add new contracts/adapters where repeated decision data cannot be represented safely by the current one-row-per-unit CausalDataset.

## 4. Work packages

### P0 — Data acquisition, schema audit, and readiness

Implement a local-only myBPmyLife data adapter and audit command.

Suggested public API:
- load_mybpmylife_mrt(path_or_directory, ...);
- audit_mrt_dataset(...);
- export_mrt_causal_snapshot(...), only if a safe snapshot is needed by existing estimators.

The loader must inspect the actual upstream files and data dictionary before hard-coding column names. Preserve:
- participant_id;
- decision_id / decision timestamp;
- availability / eligible-at-decision indicator, if present;
- assigned treatment arm;
- actual notification delivery, if present;
- row-level randomization probability for each arm;
- pre-treatment context/history;
- outcome name, value, and observation window;
- missingness and label-maturity indicators.

Separate assignment from delivery. The primary randomized estimand is assignment-based ITT. Any per-protocol / delivered-treatment analysis is a separate estimand and must address non-compliance.

Data audit must report:
- source file hashes, row/participant counts, duplicate decision keys;
- decision counts per participant and timepoint;
- treatment counts and observed arm proportions;
- declared versus empirical randomization probabilities;
- availability and delivery rates;
- outcome-specific missingness and maturity;
- pre-treatment feature balance by arm;
- time ordering and post-treatment leakage checks;
- participant-level and decision-level split definitions;
- source/license/provenance metadata.

Use known randomization probabilities from the trial protocol/data dictionary. Do not estimate a propensity model and overwrite the known assignment mechanism for the randomized analysis. If the actual row-level probabilities cannot be reconstructed, fail closed with DATA_SCHEMA_UNRESOLVED rather than inferring them from observed treatment proportions.

The MRT split must avoid participant leakage. At minimum, provide a participant-held-out evaluation. A separate temporal split may be provided for within-participant forecasting, clearly labeled as a different target.

### P1 — Real randomized Push baseline

Build a real-data benchmark for each proximal outcome separately.

Required analyses:
- arm-level descriptive outcomes with uncertainty intervals;
- randomized ITT estimates using the known assignment probabilities;
- participant-clustered uncertainty estimates (cluster bootstrap or an appropriate cluster-robust estimator);
- context-conditional treatment-effect analysis using pre-treatment variables only;
- a multi-treatment comparison: no notification vs activity vs dietary notification;
- treatment assignment vs delivery analysis kept separate;
- randomized holdout / evaluation folds that preserve participant-level grouping.

Do not calculate PEHE or individual ITE error on the real MRT dataset. It has no observed per-person counterfactual truth. Qini/AUUC may be reported only with a clearly stated estimator and limitations; prioritize design-based ITT and policy evaluation consistent with the MRT design.

### P2 — Controlled bias and identification benchmark

Create a separate semi-synthetic generator that uses real pre-treatment feature/context distributions from the MRT (and optionally X5) while generating known potential outcomes. It must generate Y(0), Y(activity), and Y(diet) or the corresponding potential-outcome surfaces before treatment assignment. Keep the true effects in a separate evaluation object; never leak them into model features.

Provide independently switchable stress scenarios:

A. Measured confounding:
- treatment assignment depends on pre-treatment X/history/context;
- potential outcomes depend on overlapping and heterogeneous covariates;
- record the exact known propensity mechanism.

B. Sample-selection bias (SSB):
- create a full target population, then a selected/observed sample using a known selection mechanism;
- include MAR selection as the identifiable baseline;
- include an MNAR sensitivity scenario only as a non-identifiability demonstration or oracle-probability upper bound;
- do not claim that an MMOE architecture alone removes sample-selection bias.

C. Positivity / overlap failure:
- create arm-specific propensities near zero or one in selected strata;
- report common-support population, extreme propensity rates, ESS per arm, weight tails, and excluded population;
- compare clipping, trimming, overlap weighting, and no correction;
- report the estimand change caused by trimming/overlap weighting.

D. Hidden confounding:
- introduce an unobserved U that affects treatment and outcome;
- demonstrate that observed-X adjustment does not generally recover the true effect;
- provide a sensitivity/refutation report; do not claim IPW, DR, EFIN, or deep models can identify arbitrary hidden confounding.

For every scenario, save the true DGP parameters, true arm-specific response functions, true CATE, true ATE, assignment probability, selection probability, random seed, and data fingerprint in the benchmark manifest.

Compare:
- unadjusted difference;
- outcome regression / T-Learner;
- IPW and stabilized IPW;
- overlap weighting;
- DR-Learner / R-Learner with cross-fitting;
- existing EFIN/DragonNet/representation candidates only where the adapter is genuinely runnable;
- explicit selection weighting or selection-model correction for SSB.

Evaluate ATE bias/RMSE, CATE PEHE (semi-synthetic only), policy value/regret, confidence-interval coverage, ESS, weight variance, and balance before/after weighting.

### P3 — Multi-treatment and continuous dose

Multi-treatment:
- use the real MRT arms (none/activity/diet) and their known randomization probabilities;
- add a cross-fitted generalized propensity contract for observational multi-arm data only if needed;
- compare a simple multi-arm outcome model, multi-arm DR / IPW where mathematically supported, and runnable deep candidates;
- do not route three-arm data through the existing binary propensity API.

Continuous dose:
- the MRT is not a randomized continuous-dose dataset;
- create a separate semi-synthetic frequency-dose benchmark from real pre-treatment context distributions;
- include at least three DGP shapes: monotone response, diminishing returns/saturation, and non-monotone fatigue;
- retain known response curves and economic optimum separately;
- compare existing DoseResponseGBM, DRNet, VCNet, and GIKS variants under the same split and dose-support guard;
- treat any VC-Transformer implementation as a research candidate only after locating and documenting the exact paper/official implementation. Do not rename TransTEE or a generic Transformer interaction model as an exact reproduction.

Metrics:
- response-curve MISE / integrated curve error;
- dose-effect error and policy regret against known truth;
- support/extrapolation violation;
- calibration and runtime/memory;
- policy value under a declared cost curve.

### P4 — Monotonicity and calibration

Implement and compare:
1. no monotonic correction;
2. post-hoc PAV/isotonic calibration;
3. explicit monotonic constraints or monotonicity penalty;
4. graph/Laplacian smoothness as a separate regularizer.

Important:
- Laplacian smoothness alone does not guarantee monotonicity;
- Push-frequency effects may be non-monotonic because of fatigue;
- enforce monotonicity only for a DGP/domain where the assumption is declared and justified;
- report a non-monotonic fatigue DGP where monotonic correction should be rejected or shown to increase error.

Metrics:
- monotonic violation rate;
- curve error / MISE;
- calibration;
- policy value and policy regret;
- sensitivity to smoothing/constraint strength.

### P5 — Lagrangian and large-scale optimizer benchmark

The current budget module only simulates sorted Top-K allocation. Add a separate constrained optimizer API without breaking it.

Define candidate decision variables x[i,t,a] for user i, decision time t, and action a. Example objective:

maximize sum of predicted incremental business value[i,t,a] * x[i,t,a]

subject to:
- total notification/resource budget;
- per-time-slot capacity;
- per-user rolling frequency cap;
- minimum interval between notifications;
- action/type quota;
- at most one action per user-decision point;
- optional eligibility and guardrail constraints.

The reduced utility under dual prices must be explicit and inspectable, e.g. incremental_value - lambda_budget * resource_cost - lambda_slot * slot_usage - lambda_user * frequency_usage - lambda_action * action_usage.

Compare:
- Greedy Top-K baseline;
- exact MILP/CP-SAT reference on small instances only;
- basic primal-dual/subgradient baseline;
- line-search primal-dual variant;
- heap-based candidate/marginal-gain update variant;
- coordinate-iterator dual-update variant.

These are research/engineering implementations inspired by the named algorithm families. Unless the original source code or paper is located and matched, do not claim exact reproduction of proprietary implementations from a resume.

For each solver report:
- objective value and net incremental value;
- feasibility and maximum constraint violation;
- dual residual / stopping reason / iterations;
- optimality gap against exact small-instance reference;
- runtime, peak memory, candidate count, and scale;
- deterministic seed and full solver config.

Scale ladder: 1K, 10K, 100K, 1M, then 10M only after profiling memory and sparse representation. Do not allocate dense user × time × action tensors at 10M scale.

A single global budget with identical unit costs is a Top-K problem and does not justify a Lagrangian solver. Include multiple coupled constraints before claiming the primal-dual method is needed.

### P6 — Optional X5 SMS external validation

If the user supplies the official X5 files locally:
- build a dedicated loader joining uplift_train, clients, and pre-communication purchases;
- aggregate only pre-campaign purchases into features;
- prevent post-campaign leakage;
- preserve customer ID and source provenance;
- inspect treatment assignment mechanism before assigning randomized/observational status;
- default to observational / assignment-unverified;
- report balance, propensity, overlap, ESS, and sensitivity;
- do not report PEHE or known causal truth;
- do not infer that the communication was a coupon.

This package is optional and must not block the primary Push MRT benchmark.

## 5. Evidence and output contract

All raw data, caches, models, and run outputs stay outside Git. Add a documented local data directory under the existing ignored data convention; never commit raw MRT, X5, Hillstrom, or Criteo files.

Each run must produce a versioned evidence directory outside Git containing:
- config and command;
- dataset manifest, source URL, license status, file hashes, and schema mapping;
- data audit and readiness result;
- randomization/assignment mechanism;
- diagnostics and nuisance estimates;
- model registry/maturity and model configs;
- metrics with estimand labels;
- optimizer config, iteration trace, feasibility and scale report;
- environment, git SHA, seed, runtime, memory;
- limitations and warnings.

Use the existing evidence/artifact patterns wherever possible. Keep aggregate, non-sensitive benchmark summaries in documentation; never commit participant-level data or health-related rows.

Every result must be labeled as one of:
- REAL_RANDOMIZED_MRT;
- REAL_OBSERVATIONAL_ASSIGNMENT_UNVERIFIED;
- SEMI_SYNTHETIC_KNOWN_TRUTH;
- SYNTHETIC_SCALE_STRESS.

Do not merge these evidence classes into one headline score.

## 6. Acceptance gates

P0 passes only if:
- loader schema is validated against the actual data dictionary;
- raw files remain outside Git;
- participant/decision counts, assignment probabilities, outcome windows, and missingness are reported;
- invalid or ambiguous assignment fields fail closed;
- tests cover repeated participant IDs and no-leakage behavior.

P1 passes only if:
- each outcome is analyzed separately;
- known MRT assignment probabilities are used;
- uncertainty respects participant clustering;
- participant-held-out split has no user overlap;
- no individual PEHE claim is made on real data.

P2 passes only if:
- all four bias scenarios have reproducible DGP metadata;
- semi-synthetic potential outcomes are generated before treatment and selection;
- IPW/DR/selection methods are compared against known truth;
- overlap trimming reports target-population changes;
- hidden-confounding limitations are explicit.

P3/P4 pass only if:
- multi-arm assignment probabilities are supported without misusing binary propensity;
- continuous models are evaluated only inside supported dose ranges;
- monotonic and non-monotonic DGPs are both tested;
- Laplacian smoothing is not described as a monotonicity guarantee;
- unvalidated VC-Transformer claims are prohibited.

P5 passes only if:
- exact small-instance reference is available;
- all solvers use the same objective, candidates, costs, and constraints;
- feasibility is checked independently of solver status;
- objective gap, runtime, memory, and scale are recorded;
- the single-budget Top-K special case is included as a baseline.

## 7. Execution order

Implement in gated increments:
1. P0 schema/data audit and a no-raw-data smoke fixture;
2. P1 randomized Push baseline;
3. P2 controlled bias benchmark;
4. P3 multi-treatment and continuous-dose extensions;
5. P4 monotonicity/calibration;
6. P5 constrained optimizer and scale benchmark;
7. P6 optional X5 loader if source files are supplied.

After each phase:
- run focused tests and existing regression tests;
- write an evidence report;
- update the phase status and blockers;
- do not proceed past a failed gate without documenting the reason.

## 8. Existing documentation to read first

- README.md
- docs/architecture/THREE_LAYER_PLATFORM.md
- docs/benchmarks/DATASETS.md
- docs/benchmarks/BENCHMARK_PROTOCOL.md
- docs/models/CONTINUOUS_BENCHMARK_EVIDENCE_V06.md
- docs/data/OBSERVATIONAL_PIPELINE.md
- docs/data/PROPENSITY_AND_OVERLAP.md
- docs/recovery/02_CODEX_RESTORE.md
- deepuplift/data/public/datasets.py
- deepuplift/data/nuisance/propensity.py
- deepuplift/benchmarks/continuous/
- deepuplift/decision/optimization/

## 9. Source references

- myBPmyLife public data repository: https://github.com/WIRED-L-um/MRT_myBPmyLife
- myBPmyLife MRT paper: https://www.jmir.org/2026/1/e78218/
- X5 RetailHero data page: https://ods.ai/competitions/x5-retailhero-uplift-modeling/data
- scikit-uplift X5 dataset documentation: https://www.uplift-modeling.com/en/latest/api/datasets/fetch_x5.html
- Hillstrom and Criteo existing benchmark documentation: docs/benchmarks/DATASETS.md

