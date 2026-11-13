# Codex Executor Prompt — Implement Experiment A Uplift Model Comparison

You are working in the existing repository:

- Repository: `jd-SearchEngines/deepuplift-agent`
- Branch: `codex/push-mrt-causal-benchmark`
- Specification: `docs/benchmarks/EXPERIMENT_A_MODEL_COMPARISON.md`
- Parent plan: `docs/benchmarks/PUSH_CAUSAL_BENCHMARK_PLAN.md`

Your task is to implement and execute **Experiment A: binary, multi-treatment, and continuous-treatment uplift model comparison**. This is a modelling benchmark, not a production CRM redesign.

## 1. Mandatory first actions

1. Read the Experiment A specification completely.
2. Read the existing Push benchmark plan, P1/P2/P3 reports, benchmark protocol, dataset registry, model APIs and relevant tests.
3. Inspect the actual branch HEAD, git status and current local data availability before editing.
4. Reuse the existing loaders, contracts, estimators, continuous benchmark runners, evidence format and tests. Do not create a second platform.
5. Preserve existing public APIs and the existing P0–P5 evidence. Add a separate Experiment A report; do not overwrite historical reports.
6. Keep raw datasets, participant-level records, caches, model weights and run outputs outside Git.

Do not assume the previous reports imply that the local raw datasets are currently available. Verify actual local availability and report DATA_NOT_AVAILABLE when appropriate.

## 2. Critical correctness issue to fix first

Inspect `deepuplift/benchmarks/push_semisynthetic.py`.

The current ATE estimator comparison includes unadjusted, outcome regression, IPW, stabilized IPW, overlap weighting and DR. However, the evaluator currently derives `policy_effect` / PEHE from the Ridge outcome-model effect surface for several method labels. This is not a valid method-specific CATE tournament.

Separate the two tasks:

- ATE estimator benchmark: compare ATE estimates and their bias/RMSE/interval coverage.
- CATE learner benchmark: each learner must produce its own held-out `tau_hat(X)`; calculate PEHE, Qini/AUUC and policy metrics from that learner's own predictions.

Add regression tests proving that each CATE method's prediction path is genuinely method-specific. Do not let IPW/DR ATE labels inherit Ridge CATE predictions unless that shared outcome model is explicitly the CATE learner being evaluated.

## 3. A1 — Binary Uplift model tournament

Implement a common adapter/evaluation path for:

- S-Learner;
- T-Learner;
- X-Learner;
- R-Learner;
- DR-Learner with cross-fitting;
- Causal Forest / GRF, using an existing supported dependency where possible.

Use a transparent outcome-regression baseline. DragonNet/TARNet/CFRNet are optional: include only if an actual runnable implementation or tested adapter exists. Otherwise record NOT_AVAILABLE and the exact reason. Do not silently substitute another model.

### Data

1. Primary known-truth benchmark: reuse the Push-context binary semi-synthetic generator, but create suitable DGP variants (at least constant/linear heterogeneous and nonlinear heterogeneous effects) with randomized assignment for the clean model-capacity comparison.
2. Add observational assignment only as a clearly labeled robustness scenario if it can reuse existing P2 infrastructure; do not let the separate debiasing benchmark take over A1.
3. Real randomized check: use myBPmyLife MRT pairwise comparisons:
   - activity vs none for activity steps in the next 60 minutes;
   - diet vs none for low-sodium choices in the next 24 hours.
   Verify assignment probabilities and use the correct conditional pairwise probabilities. Keep the two outcomes separate.
4. Criteo/Hillstrom may be run as optional external randomized binary benchmarks if local data and license conditions permit. If unavailable, record DATA_NOT_AVAILABLE and continue.

### Evaluation

Known-truth semi-synthetic:
- ATE bias/RMSE;
- CATE PEHE/effect RMSE;
- Qini/AUUC;
- policy value and regret against the known oracle;
- calibration by predicted-effect bins;
- runtime and peak memory.

Real MRT:
- outcome-specific design-based ITT;
- participant-held-out conditional effect surfaces;
- held-out one-step proximal policy value with known randomization probabilities;
- participant-cluster bootstrap confidence intervals;
- PEHE = NOT_AVAILABLE.

Use only pre-treatment features. Participant-group splits are mandatory for real/repeated-participant data.

## 4. A2 — Multi-treatment model tournament

Preserve the real MRT arms none/activity/diet and the two outcome windows separately.

Compare:

1. One-vs-Control: one binary model per active arm vs none.
2. Unified multi-treatment outcome model with shared representation and arm-specific heads/interactions.
3. Multi-treatment DR / generalized propensity model, only with a correct multi-arm propensity contract and cross-fitting where needed.
4. Multi-head DragonNet or another deep multi-treatment model only if genuinely implemented and tested.

Do not pass three-arm data through a binary propensity interface.

Create a Push-context semi-synthetic multi-arm generator with known potential outcomes and arm-specific CATE truth. Use at least three actions including none, heterogeneous arm effects, fixed seeds, and an explicit eligible-action mask.

Metrics:
- per-arm ATE bias/RMSE;
- per-arm PEHE on semi-synthetic test data;
- policy value/regret against the known oracle;
- per-arm calibration;
- runtime and memory.

On real MRT, report outcome-specific ITT and one-step proximal policy value with participant-clustered uncertainty. No real-data PEHE.

One-vs-Control is the required simple baseline. Do not assume a unified deep model wins.

## 5. A3 — Continuous-treatment model tournament

Reuse the existing continuous code and model adapters:

- DoseResponseGBM;
- DRNet;
- VCNet;
- GIKS-DRNet;
- GIKS-VCNet.

Preserve three known-truth response shapes:
- monotone;
- diminishing returns / saturation;
- non-monotone fatigue.

Define dose precisely (e.g. number of pushes in a declared rolling window). State what is held fixed about Campaign/content mix. Do not treat different Campaign versions as interchangeable dose units without declaring that assumption.

Extend the current one-seed, 400-row smoke comparison to:
- at least five fixed seeds;
- at least two sample sizes;
- identical splits and supported-dose grids across models.

Keep all support/extrapolation guards. If the true optimum is outside a model's supported range, report the optimum/policy regret as NOT_AVAILABLE rather than extrapolating.

VC-Transformer may be included only if the exact paper/official implementation and architecture are verified. Do not rename TransTEE or a generic transformer model as VC-Transformer.

Metrics:
- response-curve MISE/integrated curve error;
- dose-effect RMSE;
- optimal-dose regret when identifiable inside support;
- calibration and support violations;
- runtime and peak memory.

## 6. Unified comparison protocol

For each dataset × DGP × sample size:

1. Use identical train/validation/test splits for all models.
2. Use participant-grouped splits whenever source records repeat by participant.
3. Use the same pre-treatment features, outcome window, eligible population and estimand.
4. Tune only on training/validation. Never tune on the test set.
5. Give models comparable tuning budgets and document model-specific hyperparameters.
6. Run at least five fixed seeds; report every seed, mean, standard deviation and paired confidence intervals for differences.
7. Use participant-cluster bootstrap for real MRT and repeated-participant sources.
8. Label each evidence row as REAL_RANDOMIZED_MRT, SEMI_SYNTHETIC_KNOWN_TRUTH, REAL_RANDOMIZED_EXTERNAL or SYNTHETIC_SCALE_STRESS.
9. Keep one-step proximal policy value separate from full sequential/long-term policy value.
10. Do not combine steps and low-sodium outcomes into one generic score.

Do not rank models using PEHE on real data. Do not infer individual counterfactual truth from a randomized trial.

## 7. Execution and engineering requirements

- Prefer existing dependencies and interfaces.
- Add model adapters only where required.
- Keep model implementations small, testable and reversible.
- If a model dependency is unavailable, report it as BLOCKED/NOT_AVAILABLE with a concrete reason; continue with runnable models.
- If raw MRT data is missing locally, run unit tests and available semi-synthetic fixtures, but do not fabricate real-data metrics.
- Do not make a new randomization or production-system change. This task is offline model benchmarking only.
- Do not modify P4 monotonicity or P5 optimizer logic except where a shared API bug blocks Experiment A; if such a change is required, isolate it and add regression tests.

## 8. Required deliverables

1. Implement A1, A2 and A3 comparison paths as far as the available data and dependencies permit.
2. Add focused unit/integration tests for each estimator adapter, data split, eligibility mask, metric and evidence contract.
3. Add runner scripts/CLI commands consistent with the existing project conventions.
4. Write `docs/benchmarks/EXPERIMENT_A_MODEL_COMPARISON_RESULTS.md` with:
   - exact code SHA and environment;
   - data availability, source and license status;
   - model status and configs;
   - per-seed results;
   - aggregate metrics and paired uncertainty intervals;
   - best model by CATE accuracy, calibration, policy value and runtime;
   - limitations, unavailable metrics and blockers;
   - explicit conclusion for A1/A2/A3.
5. Store detailed run artifacts under the existing ignored local runs/evidence directory. Do not commit them.
6. Update `docs/benchmarks/EXPERIMENT_A_MODEL_COMPARISON.md` with measured status only after the corresponding experiments complete.

## 9. Acceptance criteria

Experiment A is complete only if:

- A1 runs at least S/T/X/R/DR learners and Causal Forest on known-truth binary Push-derived data, with method-specific CATE predictions and five seeds.
- A1 real MRT checks preserve known randomization, participant grouping and outcome-specific windows.
- A2 compares One-vs-Control with at least one unified multi-treatment model on known-truth data, and separately validates the real three-arm MRT design.
- A3 reruns the existing continuous model set with at least five seeds and two sample sizes across all three dose-response shapes.
- All models use comparable splits, features, estimands and tuning budgets.
- All metrics are valid for their evidence class; no PEHE on real data and no unsupported extrapolation.
- Relevant focused and regression tests pass.
- The report distinguishes measured results from unavailable or blocked results.

A green test suite is software evidence, not proof of model superiority. Only repeated-seed test results with uncertainty intervals may support a model-ranking claim.

## 10. Work style

Proceed across A1 → A2 → A3 in one coherent implementation effort. Do not stop after producing only a plan or a single tiny prototype if the next independent work is executable. Do not expand into debiasing, monotonicity, Thompson Sampling or optimization in this task.

If one dataset/model is blocked, document the blocker and continue other independent comparisons. Ask the user only if a missing local file, license decision or business estimand makes all further progress impossible.

At completion, report exactly what was implemented and run, what passed, what was unavailable, the measured model comparisons, and the next research step. Do not claim work was executed if only documentation or code scaffolding was added.
