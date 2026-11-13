# Codex Prompt — Execute Experiment A on the User's Local Mac (M3 Max, 64 GB)

You are running inside the user's local Codex environment on their Apple Silicon Mac. Execute the work locally. **Do not provision a cloud GPU, rent a server, use GitHub Actions as the training machine, or send the data to an external API.**

Repository: `jd-SearchEngines/deepuplift-agent`  
Branch: `codex/push-mrt-causal-benchmark`

Read these documents completely before changing code:
1. `docs/benchmarks/EXPERIMENT_A_LOCAL_EXECUTION_PLAN.md`
2. `docs/benchmarks/EXPERIMENT_A_MODEL_COMPARISON.md`
3. `docs/benchmarks/EXPERIMENT_A_FRONTIER_MODEL_CATALOG.md`
4. `docs/benchmarks/EXPERIMENT_A_EXECUTOR_PROMPT.md`
5. `docs/benchmarks/PUSH_CAUSAL_BENCHMARK_PLAN.md`
6. `docs/benchmarks/DATASETS.md`
7. `examples/datasets/manifest.json`

The local execution plan is the source of truth for hardware, data, sample sizes, experiment stages and evidence boundaries.

## 1. First inspect; do not overwrite the existing project

Before doing anything:
- run `pwd`, `git status --short --branch`, `git branch --show-current`, `git rev-parse HEAD`;
- confirm the repository is the expected project and branch;
- inspect `.gitignore`, current virtual environments, data directories and available disk space;
- inspect the existing P0/P1/P2/P3 reports and relevant loaders/runners;
- determine whether myBPmyLife, Hillstrom, Criteo, X5, IHDP and ACIC files are actually present locally;
- do not assume a dataset is present because a report from an earlier machine says it was used.

Do not reset, clean, force checkout, overwrite user changes, delete data, or change branches without explicit user approval. If the worktree is dirty, preserve the changes and work around them.

## 2. Native Mac environment and MPS preflight

Use native macOS Python, not a Linux Docker container for the MPS path.

- Prefer a new isolated `.venv-experiment-a`; do not damage the existing `.venv`.
- Inspect `pyproject.toml` and optional dependencies before installing.
- Do not blindly install all of `requirements.txt`; old pinned numba/llvmlite packages may be incompatible with current Apple Silicon/Python.
- Verify `uname -m`, macOS version, Python, RAM, disk, PyTorch version, MPS availability and a real forward/backward smoke test.
- Use CPU for sklearn/EconML/CausalML/tree/forest models unless their implementation explicitly supports another backend.
- Use MPS for supported PyTorch neural models.
- PyTorch MPS does not support float64. Route float64-dependent methods to CPU or use a mathematically safe float32 implementation with a numerical-equivalence test.
- If using `PYTORCH_ENABLE_MPS_FALLBACK=1`, record every model's fallback status; do not report a mixed CPU/MPS run as pure MPS.
- Run neural jobs sequentially to avoid exhausting unified memory. Limit CPU thread counts and log them.
- Save exact package versions and device/backend in every run manifest.

Produce `docs/benchmarks/LOCAL_MAC_PREFLIGHT.md` and a machine-readable local environment manifest before the full benchmark.

## 3. Data — use this exact plan

Do not invent or merge datasets. Follow `EXPERIMENT_A_LOCAL_EXECUTION_PLAN.md`.

### Primary Push dataset: myBPmyLife MRT
- Source: https://github.com/WIRED-L-um/MRT_myBPmyLife
- Published scale: 187,517 decision points, 298 participants (287 with analyzable data).
- Arms: none/activity/diet, declared assignment probabilities 0.75/0.125/0.125 after local schema verification.
- Outcomes stay separate: activity → steps in next 60 minutes; diet → low-sodium choice in next 24 hours.
- Participant-grouped split and participant-cluster uncertainty.
- This is real repeated randomized Push-context evidence, not continuous-dose randomization and not individual counterfactual truth.
- Keep raw data local-only. Treat participant-level health/behavior records as sensitive. Do not commit or upload them.

### Hillstrom RCT
- Source: https://hillstorm1.s3.us-east-2.amazonaws.com/hillstorm_no_indices.csv.gz
- 64,000 customers, approximately one-third each: men's email, women's email, no email.
- Use as the main real three-arm randomized marketing benchmark (A2), and pairwise A1 checks.
- Outcomes visit/conversion/spend remain separate.
- No PEHE on real data.

### Criteo unbiased uplift v2.1
- Official source: https://ailab.criteo.com/criteo-uplift-prediction-dataset/
- Use unbiased v2.1 only: 13,979,592 rows, 12 dense features, treatment, exposure, visit and conversion.
- Run 100k, 500k and 1m samples first. Full data only after memory/runtime preflight.
- Keep treatment assignment and actual exposure separate. Do not infer exact propensity from observed treatment ratio.
- Use uplift ranking metrics; no PEHE.
- License is CC BY-NC-SA 4.0; never commit raw data and review terms before redistribution/commercial use.

### Push-derived semi-synthetic known-truth datasets
This is the primary source for comparing CATE/response-curve accuracy:
- derive pre-treatment context/history from myBPmyLife and optionally Hillstrom;
- generate all potential outcomes before assignment;
- keep truth out of model features in a separate evaluator object;
- binary: constant/linear, nonlinear interaction, sparse high-uplift subgroup;
- multi-treatment: 3-arm and 6-arm, with arm imbalance and per-user eligibility masks;
- continuous: dose 0–5 assigned interventions in a fixed future 7-day policy window, holding content mix fixed in the DGP; include monotone, saturation and inverted-U fatigue curves;
- include pre-decision history: prior 7d/30d push count, time since last push, prior response and activity;
- main sizes 5k and 20k, at least five fixed seeds.
- preserve participant/source-cluster IDs when resampling MRT context and use cluster-aware splitting. Do not claim synthetic repeated rows are new real people.

### X5 RetailHero SMS
Optional observational robustness only. Public assignment randomization is not established. Do not use as randomized truth, do not compute PEHE, and do not use it to rank causal estimators by true effect accuracy.

### IHDP / ACIC
Optional external semi-synthetic checks only if files are already available or can be obtained under verified terms. Do not let them block Push/Hillstrom/Criteo experiments.

Update `examples/datasets/manifest.json` with actual local availability and metadata, but never put private absolute paths, participant rows or raw data into Git. Do not change NOT_AVAILABLE to AVAILABLE unless files are verified locally.

## 4. Execute the full experiment in stages

Proceed across A1 → A2 → A3 in one coherent effort. Do not stop after writing a plan or only implementing scaffolding.

### Stage 0 — preflight
Complete the local environment audit and data availability audit. Do not launch model training until the MPS/CPU smoke tests pass.

### Stage 1 — reproduce existing baselines
Re-run the current P1/P2/P3 smoke commands using the local environment and available data. Verify that historical results are reproducible or document why the current local data/code differs. Preserve old evidence; do not overwrite historical reports.

### Stage 2 — A1 binary
Compare:
- S/T/X/R/DR;
- Causal Forest/GRF where compatible;
- TARNet/CFRNet;
- DragonNet;
- FlexTENet;
- EFIN;
- DESCN;
- EUEN/EEUEN;
- TransTEE/HyperITE if faithful local implementation is practical.

Use Push-derived known-truth binary DGPs (5k/20k, five seeds) for PEHE and oracle policy regret. Use Hillstrom and myBPmyLife for real randomized validation. Use Criteo for 100k/500k/1m scale and Qini/AUUC.

EFIN needs meaningful treatment descriptors. Use actual documented action descriptors where available; otherwise mark the data-contract limitation. Do not invent real descriptors or leak outcomes into descriptors. Test a separate descriptor ablation only in clearly labeled synthetic data.

### Stage 3 — A2 multi-treatment
Compare:
- One-vs-Control;
- multi-arm S/T or shared representation + action-specific heads;
- MEMENTO;
- HydraNet;
- M3TN;
- MTMT;
- EFIN multi-treatment;
- HyperITE if feasible.

Use Hillstrom three-arm RCT, myBPmyLife three-arm MRT (two outcome definitions separately), and Push-derived 3-arm/6-arm known-truth data. Enforce each user's eligible-action mask. Include rare-arm imbalance. Verify whether MTMT's “whether to treat + which treatment type” decomposition matches the action ontology before using it.

### Stage 4 — A3 continuous treatment
Compare:
- GPS/spline;
- DoseResponseGBM;
- DRNet;
- VCNet;
- GIKS-DRNet / GIKS-VCNet;
- SCIGAN;
- TransTEE;
- HNet;
- CRNet/WDBN;
- DR²CE;
- DRVAE as exploratory if available.

Use Push-context semi-synthetic known-truth dose-response only; there is no verified public randomized continuous-frequency Push dataset in the current project. Run monotone, saturation and fatigue DGPs, sizes 5k/20k and five seeds. Enforce support guards; no extrapolated optimum claims.

### Stage 5 — longitudinal overlay
Compare WCLS, static CATE with pre-decision history features and one history-aware GRU/Transformer candidate if feasible. Use repeated-decision known-truth DGPs with carryover, fatigue, cooldown and temporal drift. Keep participant clusters intact. This is not full sequential policy optimization.

## 5. Fair comparison protocol

- identical train/validation/test splits for all models per dataset/DGP/seed;
- participant-grouped splits for MRT and repeated-decision data;
- stratified treatment splits for independent randomized customer datasets;
- same pre-treatment features, outcome windows, eligible population and estimand;
- tune only on train/validation; never tune on test;
- at least five fixed seeds for main semi-synthetic runs;
- paired confidence intervals for model differences;
- report mean and standard deviation across seeds;
- report every model's device/backend, wall time, peak RSS, training epochs and parameter count where available;
- report accuracy, uplift ranking, calibration, policy value and runtime separately;
- every model has a maturity label: official code reproduced, faithful reimplementation, library adapter, architecture-inspired, paper-only, or blocked;
- do not treat test AUC or factual outcome prediction as proof of causal effect accuracy.

Metrics:
- known-truth synthetic: ATE bias/RMSE, PEHE, Qini/AUUC, calibration, oracle policy value/regret; continuous MISE and dose-response error;
- real randomized: design-based ITT, valid randomized one-step policy value, cluster confidence intervals; no PEHE;
- observational: overlap, ESS, balance and sensitivity; no causal ground-truth claim.

## 6. Local resource management

- Run a small 1k–5k smoke test per architecture before larger jobs.
- Main semi-synthetic sizes: 5k and 20k; five seeds.
- Criteo: 100k/500k/1m before full data.
- Run neural models sequentially on MPS.
- Do not parallelize all seeds at once.
- Keep a per-model timeout and save checkpoints where the model supports it.
- If a model is CUDA-only or requires unavailable legacy dependencies, first try a faithful CPU/MPS adapter; otherwise mark BLOCKED_ON_CUDA / BLOCKED_DEPENDENCY and continue.
- Do not silently replace a named model with a simpler model and call it the original.
- Do not rent cloud compute or ask the user to configure a cloud provider during this task.
- At the end, report actual bottlenecks and whether CUDA would materially help. Do not speculate about speedup without a measured profile.

## 7. Data and Git safety

- Keep raw data under ignored local `data/raw/` paths.
- Verify `.gitignore` covers raw files, generated semi-synthetic data, checkpoints, caches and run artifacts.
- Never commit raw myBPmyLife, Hillstrom, Criteo, X5 or participant-level data.
- Do not upload participant-level MRT data to external APIs.
- Commit only code, metadata-only manifests, aggregate results and documentation that cannot reveal participant records.
- Record source URL, license status, file SHA256 and row counts locally.
- If a license is unknown, mark it UNVERIFIED and keep the dataset local-only.

## 8. Required deliverables

1. `docs/benchmarks/LOCAL_MAC_PREFLIGHT.md`
2. updated metadata-only `examples/datasets/manifest.json`
3. reusable local commands / scripts for A1, A2, A3 and longitudinal overlay;
4. `docs/benchmarks/EXPERIMENT_A_MODEL_COMPARISON_RESULTS.md`
5. local ignored run artifacts with per-seed metrics and manifests;
6. tests for dataset contracts, splits, leakage, eligibility masks, model-specific predictions and metrics.

The final report must say precisely:
- what data were found/downloaded and what remained unavailable;
- exact hardware/software/device for each model;
- which models ran on MPS, CPU, or were blocked;
- per-seed results and paired uncertainty intervals;
- separate rankings for CATE accuracy, uplift ranking, calibration, policy value and efficiency;
- which advanced models were statistically tied with simpler baselines;
- whether local resources were sufficient and what measured bottleneck would justify cloud CUDA later.

Do not claim a benchmark is complete merely because the scripts or tests pass. Only completed model runs with valid evidence count as experiment results.
