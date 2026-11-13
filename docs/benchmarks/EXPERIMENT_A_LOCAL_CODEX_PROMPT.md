# Codex Prompt — Real-Data-First Experiment A on Local Mac

You are running inside the user's local Codex environment on an Apple Silicon Mac M3 Max, 64 GB unified memory.

**Research correction:** Real public data is the primary benchmark. Shopee CRM Push is only a motivating example, not the required data schema or research boundary. Use real randomized and real observational datasets first. Semi-synthetic data is auxiliary only for known-truth metric checks. Prefer models with accessible source code. If a full run does not fit or takes too long, reduce sample size and run the model; do not abandon it without a measured reason.

Repository: `jd-SearchEngines/deepuplift-agent`  
Branch: `codex/push-mrt-causal-benchmark`

Read:
1. `docs/benchmarks/EXPERIMENT_A_LOCAL_EXECUTION_PLAN.md`
2. `docs/benchmarks/EXPERIMENT_A_REAL_DATA_AND_CODE_REGISTER.md`
3. `docs/benchmarks/EXPERIMENT_A_MODEL_COMPARISON.md`
4. `docs/benchmarks/EXPERIMENT_A_FRONTIER_MODEL_CATALOG.md`
5. `docs/benchmarks/EXPERIMENT_A_EXECUTOR_PROMPT.md`
6. `examples/datasets/manifest.json`

The revised local execution plan and real-data/source-code register supersede any older instruction that makes a synthetic CRM dataset the primary benchmark.

## 1. Inspect before changing anything

- Check `pwd`, `git status --short --branch`, current branch, HEAD SHA.
- Preserve dirty worktree changes. Do not reset, clean, delete, force checkout or overwrite user files.
- Inspect existing P0/P1/P2/P3/P4/P5 code, reports, loaders and tests.
- Inspect local data directories, file sizes, available disk and existing virtual environments.
- Verify every dataset exists before marking it available.
- Inspect .gitignore and make sure raw data/checkpoints/logs are excluded.

## 2. Local environment

- Use native macOS Python, isolated `.venv-experiment-a`.
- Verify arm64, macOS, Python, PyTorch, MPS availability and an actual forward/backward smoke test.
- Use CPU for classical models and unsupported methods; use MPS only for supported PyTorch operations.
- Do not assume CUDA is available.
- Do not blindly install old requirements. For legacy TensorFlow 1.x, old PyTorch, XDL, custom CUDA ops or incompatible Numba, isolate the source environment or port faithfully only when practical.
- Log all versions and device/fallback information.

## 3. Real datasets first

Use the data and source register as the source of truth.

### A1 binary
1. Criteo Uplift v2.1: real randomized advertising data; treatment assignment and exposure are separate. Try 10k/100k, then 500k/1m, then full 13.98m only if feasible.
2. Hillstrom: 64k randomized email customers, three arms. Use pairwise treatment-vs-control comparisons.
3. myBPmyLife MRT: real randomized repeated notification data, participant-clustered.
4. Jobs/NSW and Twins: optional external causal validation with their own design caveats.
5. X5/Lenta/MegaFon: real observational marketing data, only for overlap/confounding/robustness diagnostics unless assignment is verified.

### A2 multi-treatment
1. Hillstrom three-arm randomized marketing experiment.
2. myBPmyLife three-arm MRT, preserving separate outcome definitions.
3. MultiTwins only after source and treatment encoding are verified.
4. Lenta/X5/MegaFon only if true multi-arm assignment fields exist; otherwise keep in binary/observational lane.

### A3 continuous treatment
1. NHEFS real observational data: test baseline smoking intensity (cigarettes/day) as continuous exposure and follow-up weight change as outcome, after verifying raw fields and time ordering.
2. Run GPS/spline and CausalGPS as interpretable baselines.
3. Run source-backed DRNet, VCNet and GIKS where feasible.
4. SCIGAN/TransTEE only if their source code and dependencies can run.
5. Do not claim NHEFS provides randomized dose-response or individual counterfactual truth. Do not report PEHE.

### Auxiliary known-truth validation
Use small IHDP/ACIC/benchmark-provided semi-synthetic data only for validating PEHE, known-truth CATE and dose-response metric implementations. Do not make this the main leaderboard.

## 4. Source-code-first model execution

For each candidate:
- locate the exact official repo first;
- if no official repo exists, classify any third-party code as third-party reproduction;
- record license, commit SHA, dependencies, dataset assumptions and training command;
- do not confuse similarly named repositories or models;
- do not call a simplified architecture the original paper implementation.

Prioritize executable source-backed candidates:
- A1: S/T/X/R/DR via maintained libraries; Causal Forest/GRF; EFIN; DESCN; EUEN/EEUEN; DragonNet/TARNet/CFRNet/FlexTENet where source is verified.
- A2: One-vs-Control; multi-arm shared-head baseline; M3TN source/reproduction; HydraNet if exact repo is verified. MEMENTO/MTMT are not blockers if matching source code is unavailable.
- A3: GPS/spline, CausalGPS, DoseResponseGBM, DRNet, VCNet, GIKS, SCIGAN, TransTEE where exact source is available and runs.

HNet, DR²CE, DRVAE, CRNet/WDBN and any other frontier candidate must remain conditional until source code, dependencies and fidelity are verified. Do not implement paper-only candidates before the source-backed benchmark is running.

## 5. Run ladder: never just say “it doesn't run”

For each dataset × model:
1. inspect README, license, requirements, data schema and official run command;
2. environment import and model-instantiation smoke test;
3. one forward/backward pass;
4. 1k rows or full small dataset, one seed, one epoch/minimum schedule;
5. 5k–10k rows, one seed;
6. 50k–100k rows or full dataset if smaller;
7. full data if memory/time allows;
8. after stability, three seeds for initial results and five for final core comparisons.

If the model is too slow, reduce rows, epochs, batch size, hidden width, tuning budget or seeds. Record each reduction and preserve the original model objective/architecture. If the architecture itself must change, relabel it as an adaptation.

## 6. Mandatory run-failure diagnosis

Every failure must be classified and documented:
- DATA_DOWNLOAD_OR_ACCESS
- LICENSE_OR_TERMS
- DEPENDENCY_CONFLICT
- CUDA_ONLY
- MPS_UNSUPPORTED_OP
- MEMORY_OOM
- RUNTIME_TOO_SLOW
- DATA_SCHEMA_MISMATCH
- REPRODUCTION_GAP
- NUMERICAL_FAILURE
- METRIC_OR_ESTIMAND_INVALID

For each failure include exact command, stack trace excerpt, environment versions, dataset size, backend, elapsed time, peak memory, attempted fixes, small-sample result and final status.

Remediation order:
- verify source and license;
- isolate dependency environment;
- run tiny data and short schedule;
- remove unnecessary multi-GPU launcher assumptions only if the algorithm permits;
- try CPU/MPS-compatible path;
- reduce sample/epochs/batch/seeds;
- make a faithful port only with tests;
- otherwise mark BLOCKED with evidence and continue to the next model.

Never silently substitute a different model or fabricate metrics.

## 7. Fair evaluation

- Identical train/validation/test splits for each model within the same dataset and seed.
- Group by participant/customer when repeated observations exist.
- Do not leak post-treatment variables.
- Real randomized data: ITT, arm-specific estimates, Qini/AUUC/uplift@K with design-aware uncertainty and valid one-step policy value.
- Real observational data: overlap, propensity, balance, ESS, weight tails, sensitivity and robustness. Do not claim Qini alone proves causal validity.
- Real paired-outcome data: pair-level estimands and assumptions; do not claim exact individual counterfactual truth without justification.
- Auxiliary known-truth data only: PEHE, ATE bias, dose-response MISE and oracle regret.
- Keep evidence classes and leaderboards separate.

## 8. Required deliverables

Create/update:
- `docs/benchmarks/LOCAL_MAC_PREFLIGHT.md`
- `docs/benchmarks/DATASET_AUDIT.md`
- `docs/benchmarks/MODEL_SOURCE_AND_RUN_STATUS.md`
- `docs/benchmarks/EXPERIMENT_A_MODEL_COMPARISON_RESULTS.md`

Also produce ignored local artifacts with per-run configs, logs, stack traces, metrics and checkpoints.

Run tests. Do not stop after writing scaffolding. Execute real data and source-backed models in the staged run ladder. Keep raw data out of GitHub. Report actual progress, exact blockers and what successfully ran.


## 9. Mandatory deep dive: expose the mechanism of debiasing and repair

Read `docs/benchmarks/BIAS_DIAGNOSIS_AND_REPAIR_PROTOCOL.md` and execute it as a required experiment layer. Do not merely list IPW/DR/EFIN/VCNet as model names.

For each dataset:
1. Define the estimand and target-trial contract: unit, eligibility, time zero, assignment, delivery/exposure, comparator, outcome window, censoring and target population.
2. Diagnose measured confounding, treatment selection, overlap/positivity, arm imbalance, representation mismatch, nuisance misspecification, continuous-dose support, time-varying confounding, noncompliance, missing outcomes, interference, leakage and treatment-version inconsistency where applicable.
3. Quantify each applicable issue with exact diagnostics: SMD, variance ratios, propensity calibration/distributions, common-support share, max/quantile weights, ESS overall/per arm, dose density, missingness and cluster structure.
4. Run correction ablations with the same split/seed/estimand: unadjusted, outcome regression, IPW, stabilized IPW, clipping grid, matching/stratification, overlap weighting, cross-fitted AIPW/DR and representation balancing where executable. For continuous treatment use GPS/density-ratio and continuous-dose balancing; for repeated decisions use sequential weights/WCLS/longitudinal DR where valid.
5. Expose the actual mathematics: propensity definition, weight formula, clipping threshold, nuisance models, cross-fitting folds, balancing loss and coefficient, generalized propensity or GPS, and target population after trimming/overlap.
6. Report before/after diagnostics and the trade-off: bias/effect estimate, uncertainty, ESS, retained population, runtime/memory and policy value where valid.
7. State residual problems that cannot be fixed by the selected method, especially unmeasured confounding, positivity violations, hidden treatment versions and unsupported dose regions.

Important guardrails:
- Known randomized assignment probabilities are primary for RCT ITT; do not fit a propensity model and replace design probabilities without a justified reason.
- Assignment, delivery/exposure and outcome are separate variables and separate estimands.
- DR is doubly robust only under its identification/regularity assumptions; it does not fix arbitrary hidden confounding or absent overlap.
- EFIN's intervention constraint and DESCN's joint learning are architecture-specific mechanisms, not universal proof of deconfounding.
- Representation balancing can remove effect-modifying information; test the balance penalty and held-out effect quality together.
- Clipping/trimming/overlap weighting can change the target population; report that change.
- Do not use binary propensity formulas for multi-arm or continuous treatments.
- Do not report PEHE on real datasets without individual counterfactual truth.
- Do not call a correction successful based on balance alone; also evaluate effect/decision metrics and uncertainty.

Required deliverables:
- `docs/benchmarks/BIAS_DIAGNOSIS_REPORT.md`
- `docs/benchmarks/BIAS_REPAIR_ABLATIONS.md`
- local machine-readable `bias_diagnostics.json`, safe aggregate weight diagnostics and plots.
Add unit tests for SMD, ESS, propensity/weight construction, clipping, cross-fitting leakage, generalized propensity dimensions, GPS support and pre-treatment feature filtering.
Do not stop after writing the report templates: run at least one complete real-observational correction path (e.g. Lenta/X5/MegaFon if accessible) and the design-based RCT checks. If the data or method is unavailable, document exact blocker and continue with another source-backed dataset.
