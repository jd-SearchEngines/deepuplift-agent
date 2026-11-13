# Experiment A — Causal Uplift Model Comparison

**Status:** SPECIFICATION READY; implementation and new comparative runs are not yet claimed.
**Repository:** `jd-SearchEngines/deepuplift-agent`
**Work branch:** `codex/push-mrt-causal-benchmark`
**Parent research plan:** [PUSH_CAUSAL_BENCHMARK_PLAN.md](PUSH_CAUSAL_BENCHMARK_PLAN.md)

## 1. Research question and boundary

Experiment A compares causal-effect model families for the Shopee CRM Push decision setting:

1. Binary treatment: one eligible campaign vs the defined reference action.
2. Multi-treatment: several eligible campaign/notification actions vs a shared reference action.
3. Continuous treatment: push frequency/intensity dose-response.

The question is: **which model estimates user-level incremental value most accurately and produces the best held-out policy decisions, under a controlled and comparable benchmark?**

This experiment is about treatment-effect modelling. It does not implement or benchmark the separate debiasing interventions (IPW clipping, overlap weighting, SSB correction, monotonic calibration), Thompson Sampling, or constrained optimizers. Those belong to later experiment groups. DR-Learner may appear as a model family, but its inclusion here does not replace the separate debiasing benchmark.

Do not redesign the production CRM scheduler.

## 2. Existing implementation audit — reuse and repair, do not duplicate

The current Push branch already contains:

- P1 real myBPmyLife MRT assignment-ITT baseline, outcome-specific and participant-clustered.
- P2 binary Push-derived semi-synthetic bias scenarios and ATE estimators.
- P3 real three-arm MRT descriptive/conditional Ridge baseline.
- P3 continuous dose-response comparison for DoseResponseGBM, DRNet, VCNet, GIKS-DRNet and GIKS-VCNet.
- P4 monotonicity/calibration comparisons.

Known gaps for Experiment A:

### A1 binary

P2 currently compares unadjusted, outcome regression, IPW, stabilized IPW, overlap weighting and DR for ATE/bias. It does not provide a complete S/T/X/R/DR/Causal-Forest CATE model tournament. The current P2 evaluator also derives the reported CATE prediction/policy surface from the Ridge outcome models for several weighting/DR methods. Therefore those PEHE/policy rows are not independent CATE predictions from each named estimator and must not be used to rank CATE model families. Preserve valid ATE evidence, but create a separate method-specific CATE evaluation path.

### A2 multi-treatment

P3 uses arm-specific Ridge outcome models on the real three-arm MRT. It does not yet compare One-vs-Control learners against a unified multi-treatment learner family.

### A3 continuous

P3 already compares DoseResponseGBM, DRNet, VCNet, GIKS-DRNet and GIKS-VCNet on monotone, saturating and fatigue response functions. This is a useful smoke benchmark, not yet a stable tournament: the recorded comparison uses one seed and 400 rows per response shape. Reuse its contracts and models; extend to multiple seeds and sample sizes before drawing a ranking. Do not claim VC-Transformer reproduction unless the exact paper/official implementation and architecture are located and verified.

## 3. Dataset and evidence design

Keep evidence classes separate. Never combine real randomized results and semi-synthetic known-truth results into one headline score.

### 3.1 Primary Push-context source: myBPmyLife MRT

Reuse the existing loader, data dictionary audit and participant-group split.

- Real randomized assignment arms: none / activity notification / dietary notification.
- Declared assignment probabilities: 0.75 / 0.125 / 0.125, only after verifying the local source fields and P0 audit.
- Preserve the two outcome definitions separately:
  - activity notification and steps in the next 60 minutes;
  - dietary notification and low-sodium choices in the next 24 hours.
- Treat participant as the cluster and split by participant.
- Assignment ITT is primary. Delivery/per-protocol effects are separate estimands.
- Real MRT has no individual counterfactual truth: PEHE and individual ITE error are NOT_AVAILABLE.

For A1 real-data checks, create two outcome-aligned binary comparisons:
- activity vs none for the activity proximal outcome;
- diet vs none for the dietary proximal outcome.

When restricting a three-arm randomized design to one active arm and none, use the correct conditional assignment probability within that pair, derived from the verified trial design. Do not estimate a new propensity model to replace known randomization probabilities. Keep the excluded third arm and the resulting pairwise estimand explicit.

For A2 real-data checks, retain all three arms and analyze each outcome separately. It is permissible to report the effect of each randomized arm on a given outcome, but do not merge steps and low-sodium choices into one generic outcome.

### 3.2 Push-derived semi-synthetic known-truth data

Reuse the MRT pre-treatment context distribution, but generate potential outcomes before treatment assignment. Keep true potential outcomes and CATE arrays outside the model-facing feature frame.

Required DGP families:

- Binary: constant/linear heterogeneous effect and nonlinear heterogeneous effect.
- Multi-treatment: at least three actions including none, with distinct arm-specific effect surfaces and some shared effect structure.
- Continuous: monotone, diminishing-returns/saturating, and non-monotone fatigue dose-response curves.

For each DGP, store true potential outcomes, true CATE/response curves, true ATE, assignment mechanism, seed and configuration in a separate evaluation object.

If rows are sampled from repeated participant data, preserve source participant IDs and use participant-grouped splitting. If the semi-synthetic generator intentionally creates independent synthetic units, document that unit definition explicitly. Do not silently treat repeated decisions from one real participant as independent real-world participants.

### 3.3 Existing external randomized datasets

Reuse the existing Criteo and Hillstrom loaders as optional external binary uplift benchmarks when local data is available and license terms permit use.

- Criteo: large-scale randomized advertising uplift benchmark; external domain, not Push.
- Hillstrom: randomized email-marketing benchmark; external domain, not Push.

Do not download or commit raw data as part of this task. If unavailable locally, report DATA_NOT_AVAILABLE and continue with the Push-derived semi-synthetic suite. Do not relabel these datasets as Push data.

X5 RetailHero remains observational / assignment mechanism unverified and is not a ground-truth model-ranking dataset for this experiment.

## 4. A1 — Binary treatment model comparison

### Estimand

For campaign/action c and eligible decision context X:

`tau_c(X) = E[Y(1)-Y(0) | X, Eligible_c=1]`

The reference action must be explicit. For the real MRT pairwise check, it is the no-notification arm within the declared pairwise randomized comparison. For semi-synthetic data, it is no notification / no CRM push.

Do not compare effects from different outcome definitions or different windows as if they were the same target.

### Required model families

Implement/reuse a common estimator adapter and compare:

1. S-Learner.
2. T-Learner.
3. X-Learner.
4. R-Learner.
5. DR-Learner with cross-fitting.
6. Causal Forest / Generalized Random Forest adapter, where supported.

Optional advanced model:
- DragonNet/TARNet/CFRNet only if an existing implementation or a reproducible, tested adapter is available. Mark unavailable rather than silently substituting another architecture.

Use a simple outcome-regression model as a transparent baseline. Keep nuisance models, feature preprocessing and tuning budget explicit.

### A1 metrics

On semi-synthetic known-truth test data:
- ATE bias and RMSE;
- CATE PEHE / effect RMSE;
- Qini and AUUC, with estimator and treatment prevalence stated;
- policy value and policy regret against the known oracle policy;
- CATE calibration by predicted-effect bins;
- runtime and peak memory.

On real randomized MRT:
- outcome-specific design-based ITT and participant-clustered confidence intervals;
- participant-held-out conditional effect surfaces;
- held-out one-step proximal policy value using known randomization probabilities, with participant-cluster bootstrap;
- no PEHE claim.

Do not use post-treatment variables, delivery, click, purchase, or future behavior as pre-treatment model features.

## 5. A2 — Multi-treatment model comparison

### Estimand

For each eligible action a and a shared reference action 0:

`tau_a(X) = E[Y(a)-Y(0) | X, Eligible_a=1]`

Candidate sets may differ by user. The benchmark must preserve the eligible-action mask. A model must not generate or score an action for a user who is ineligible for it.

### Required comparison families

1. One-vs-Control:
   - fit one binary treatment-effect model per active arm vs none;
   - reuse the same split, feature set, reference definition and tuning budget.
2. Unified multi-treatment outcome model:
   - shared representation with action-specific outcome heads/interactions;
   - predict every eligible arm and the shared reference.
3. Multi-treatment DR / generalized propensity approach:
   - use a correct multi-arm propensity contract;
   - cross-fit nuisance estimates where observational assignment is used;
   - do not route three-arm data through a binary propensity API.
4. Multi-head DragonNet or another deep multi-treatment model only if a real, tested implementation exists.

### A2 data and evaluation

- Real myBPmyLife: none/activity/diet, using known randomization probabilities; analyze activity steps and dietary low-sodium outcomes separately.
- Semi-synthetic: three or more actions with known arm-specific potential outcomes and CATE truth.

On semi-synthetic data report per-arm ATE bias/RMSE, per-arm PEHE, policy value/regret and calibration. On real MRT report outcome-specific ITT, participant-clustered uncertainty and one-step policy value. Do not report real-data PEHE.

Do not compare activity vs diet using a merged outcome. Do not assume that a single user's eligible action set is the same as another user's.

## 6. A3 — Continuous-treatment model comparison

Reuse the existing continuous benchmark and model adapters:

- DoseResponseGBM;
- DRNet;
- VCNet;
- GIKS-DRNet;
- GIKS-VCNet.

VC-Transformer is a candidate only after the exact source paper/official implementation, architecture and loss are verified. Do not rename TransTEE or a generic transformer model as VC-Transformer.

Use three known-truth Push-frequency dose-response shapes:
- monotone;
- diminishing returns / saturation;
- non-monotone fatigue.

Define the dose precisely (for example, number of notifications in a declared rolling window). Hold the campaign/content mix and other treatment-version assumptions fixed in the DGP, or explicitly model them as additional treatment dimensions. Do not treat heterogeneous Campaign content as interchangeable dose units without stating that assumption.

Evaluate only inside observed dose support. Report:
- response-curve MISE / integrated curve error;
- dose-effect RMSE;
- optimal-dose regret / policy regret when the true optimum lies inside support;
- calibration;
- support/extrapolation violations;
- runtime and peak memory.

Extend the current one-seed, 400-row smoke comparison to at least five fixed seeds and at least two sample sizes. Keep the original smoke result for historical reference; do not overwrite it.

## 7. Unified experimental protocol

- Use identical train/validation/test splits for every model within each dataset/DGP/seed.
- Use participant-grouped splits for real MRT and any repeated-participant source data.
- Tune only on training/validation; never select models on the test set.
- Use at least five fixed seeds for the comparative run. Report per-seed results, mean, standard deviation and paired confidence intervals for model differences.
- Use the same pre-treatment feature set, outcome window, eligible population and estimand within each comparison.
- Keep model hyperparameters, preprocessing, nuisance models, early stopping and tuning budget in the manifest.
- Report failures and unsupported models as NOT_AVAILABLE/BLOCKED with reasons; do not drop them silently.
- Do not rank models by one metric alone. Provide separate selections for best CATE accuracy, best calibration, best policy value and fastest runtime.
- For semi-synthetic data, score against held-out known truth. For real randomized data, use design-based estimators and valid one-step OPE; do not fabricate counterfactual truth.
- Keep one-step proximal policy value distinct from full sequential/long-term policy value.

## 8. Important implementation correction

Audit `deepuplift/benchmarks/push_semisynthetic.py` before using its current PEHE/policy metrics for model ranking.

The current evaluator fits Ridge outcome models and uses their full-population effect surface as `policy_effect` for several IPW/DR/overlap estimators. This means the PEHE/policy surface is not necessarily the CATE prediction produced by the named method. Separate:
- ATE estimator comparison (unadjusted, outcome regression, IPW, stabilized IPW, overlap weighting, DR);
- CATE learner comparison (S/T/X/R/DR learners, Causal Forest, etc.).

Every CATE metric must be computed from that method's own held-out effect predictions. Add a test that fails if two method labels reuse the same prediction array without an explicit shared-model declaration.

## 9. Required deliverables

Create a new evidence report:
- `docs/benchmarks/EXPERIMENT_A_MODEL_COMPARISON_RESULTS.md`

Create versioned machine-readable summaries outside Git:
- run config and dataset manifest;
- per-seed metrics;
- model registry/status;
- prediction/effect summaries;
- diagnostics and warnings;
- environment, git SHA and runtime.

Update this specification with measured statuses only after runs complete. Do not commit raw datasets, participant-level rows, model weights, caches or run outputs.

## 10. Acceptance criteria

Experiment A is complete only when:

1. A1 runs the required classical model families on Push-derived semi-synthetic known-truth data with method-specific CATE predictions and at least five seeds.
2. A1 real MRT pairwise checks preserve outcome-specific windows, known assignment probabilities, participant grouping and no-PEHE limitation.
3. A2 compares One-vs-Control against at least one unified multi-treatment model on known-truth semi-synthetic data, and reports the real three-arm MRT design-based validation separately.
4. A3 extends the existing continuous comparison to at least five seeds and two sample sizes, preserving the three DGP shapes and dose-support guards.
5. All models share comparable splits, features, estimands and tuning budgets.
6. Results distinguish REAL_RANDOMIZED_MRT, SEMI_SYNTHETIC_KNOWN_TRUTH, REAL_RANDOMIZED_EXTERNAL and SYNTHETIC_SCALE_STRESS.
7. The report states which model wins by CATE accuracy, calibration, policy value and runtime, and where the evidence is inconclusive.
8. Relevant tests pass and no raw or participant-level data is committed.

A passing software test suite is not evidence that one causal model is statistically superior. Model superiority must be supported by repeated-seed benchmark results and uncertainty intervals.
