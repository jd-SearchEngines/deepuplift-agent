# Experiment A — Local Mac Execution Plan and Dataset Contract

**Execution target:** User-controlled Apple Silicon Mac, M3 Max, 64 GB unified memory.
**Execution mode:** Native macOS Python; PyTorch MPS where supported, CPU for classical estimators and unsupported operations.
**Repository:** `jd-SearchEngines/deepuplift-agent`
**Work branch:** `codex/push-mrt-causal-benchmark`
**Related specifications:**
- [Experiment A model comparison](EXPERIMENT_A_MODEL_COMPARISON.md)
- [Frontier model catalogue](EXPERIMENT_A_FRONTIER_MODEL_CATALOG.md)
- [Local Codex executor prompt](EXPERIMENT_A_LOCAL_CODEX_PROMPT.md)

## 1. Decision: run locally first

Use the user's local Mac as the primary development and first full benchmark machine. Do not provision a cloud GPU or paid compute as the default.

Why:
- the current ChatGPT Python execution environment is CPU-only and has limited RAM; it is useful for small analysis, not this full benchmark;
- the user's Mac has 64 GB unified memory and an Apple GPU accessible through PyTorch MPS when the installed macOS/PyTorch/operator combination supports it;
- classical causal estimators run well on CPU; medium tabular neural models can be tested on MPS;
- raw MRT data should remain on the user's machine;
- the benchmark can first establish correctness, model availability, and realistic runtime before any later decision to rent CUDA hardware.

This is not a claim that M3 Max is faster than a dedicated NVIDIA GPU. It is the lowest-friction first execution environment.

## 2. Hardware and runtime constraints

### 2.1 Environment audit is mandatory

Before installing packages or launching training, record:

- macOS version and architecture (`arm64`);
- exact Python version and executable;
- available RAM and disk space;
- PyTorch version;
- `torch.backends.mps.is_available()` and `is_built()`;
- a real forward/backward smoke test on MPS;
- installed versions of NumPy, SciPy, scikit-learn, pandas, LightGBM, EconML, CausalML and optional model dependencies;
- whether each model uses CPU, MPS, or an explicit CPU fallback.

Do not assume MPS is available merely because the machine is Apple Silicon.

### 2.2 Native macOS environment

- Prefer an isolated environment such as `.venv-experiment-a`; do not overwrite an existing working virtual environment.
- Inspect `pyproject.toml` and optional dependency groups first.
- Install the core project and only compatible optional dependencies. Do not blindly install all of `requirements.txt`: it includes older pinned `numba/llvmlite` versions that may create Apple Silicon/Python compatibility problems.
- Do not use a Linux Docker container for the MPS training path; Docker Desktop on Mac does not expose the native Metal/MPS backend as a normal host process would.
- Keep all package versions and installation logs in the run manifest.

### 2.3 MPS-specific constraints

PyTorch MPS can accelerate supported tensor operations, but it is not CUDA. The MPS backend does not support float64 tensors. Models or statistical routines that require float64 must run on CPU or be safely adapted to float32 only when the paper/model assumptions and numerical stability permit it.

For unsupported MPS operators:
- try a documented CPU path first;
- use `PYTORCH_ENABLE_MPS_FALLBACK=1` only as an explicit, logged fallback;
- never label a run as MPS-only if operators silently fell back to CPU;
- record device and fallback status per model.

Run neural training sequentially on MPS to avoid competing for unified memory. Classical models may use a small, controlled CPU thread count. Do not run multiple memory-heavy neural models concurrently.

## 3. Dataset plan — what data is compared, and why

The benchmark uses different datasets for different questions. **No single dataset can validate binary, multi-arm, continuous-dose, repeated-decision, and known individual counterfactual truth simultaneously.**

| Dataset | Type and scale | Treatment / outcome | Role in Experiment A | Important limitation |
|---|---|---|---|---|
| myBPmyLife MRT | Real randomized repeated Push trial; published source reports 187,517 decision points, 298 participants, 287 with analyzable data | none / activity notification / dietary notification; steps in next 60 min and low-sodium choices in next 24 h | Primary real Push-context and longitudinal validation; A1 pairwise and A2 three-arm checks | Only 298 participant clusters; not a continuous randomized-dose trial; no individual counterfactual truth |
| Hillstrom MineThatData | Real randomized email experiment; 64,000 customers | no email / men's email / women's email; post-campaign visit, conversion and spend outcomes | Primary real three-arm randomized benchmark for A2; pairwise A1 comparisons | Email is not Push; no individual counterfactual truth |
| Criteo unbiased uplift v2.1 | Large randomized advertising benchmark; 13,979,592 rows, 12 dense features plus treatment/exposure/visit/conversion fields | treatment assignment vs control; actual exposure recorded separately | A1 marketing uplift and scale benchmark; test 100k/500k/1m samples, full data only if resource audit allows | No individual counterfactual truth; use the official unbiased v2.1 version, not the original leaked version; treatment assignment and exposure must not be conflated |
| Push-derived semi-synthetic | Generated from real pre-treatment context/history distributions, with potential outcomes generated by known functions | binary, multi-arm and continuous dose | **Primary model-accuracy benchmark**: PEHE, ATE bias, response-curve error, oracle policy regret | Results are semi-synthetic, not real Shopee causal effects |
| X5 RetailHero SMS | Observational marketing data; documented training table about 200k customers, client table about 400k and purchase history about 45.8m rows | SMS communication flag and later purchase target | Optional realism/overlap/observational robustness check | Public documentation does not establish random assignment; no known counterfactual truth; do not rank causal models by PEHE or call it randomized |
| IHDP / ACIC 2016 | Public semi-synthetic causal benchmarks, if separately obtained and terms verified | Primarily binary treatment with scenario-specific known potential outcomes | Optional external general-purpose CATE replication | Not Push-specific; local availability and reuse terms must be verified |

### 3.1 myBPmyLife: primary Push dataset

Source: https://github.com/WIRED-L-um/MRT_myBPmyLife

Use the existing project loader and P0/P1 audit. Verify actual local raw-file availability before assuming it exists.

Declared MRT assignment probabilities, only after verification against the data dictionary and row-level assignment fields:
- no notification: 0.75;
- activity notification: 0.125;
- dietary notification: 0.125.

Keep the two outcomes separate:
- activity notification → steps in the following 60 minutes;
- dietary notification → low-sodium food choice in the following 24 hours.

Use participant-grouped splits and participant-clustered uncertainty. The 187k decision rows are repeated decisions, not 187k independent people. Treat the participant-level health/behavior records as sensitive research data: local-only, no public Git commit, no third-party API upload, and no redistribution unless the source terms explicitly allow it. The upstream license is currently recorded as unverified in the project; do not claim it is cleared.

### 3.2 Hillstrom: primary real multi-treatment RCT

Source: https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data-mining-challenge.html

The experiment contains 64,000 customers randomized approximately one-third each to men's email, women's email, or no email. Use:
- all three arms for A2;
- men's vs no email and women's vs no email for pairwise A1 checks;
- visit, conversion and spend as separate outcomes, respecting their documented observation windows.

Known randomization probability is approximately 1/3 per arm. Verify actual arm counts in the local file before use. This is an external marketing RCT, not a Push trial. No PEHE on real data.

### 3.3 Criteo: primary large-scale binary uplift benchmark

Official source: https://ailab.criteo.com/criteo-uplift-prediction-dataset/

Use the **unbiased v2.1 dataset** with 13,979,592 rows, not the original 25.3m version that the source page identifies as affected by advertiser-level leakage. Fields include 12 dense features, treatment, exposure, visit and conversion. The compressed file is about 297 MB and uncompressed data about 3.2 GB.

Local staged sizes:
- smoke: 100,000 rows;
- medium: 500,000 rows;
- scale: 1,000,000 rows;
- full 13,979,592 only after a memory/runtime preflight and explicit local confirmation.

Start with conversion as the primary outcome and visit as a secondary outcome. Keep assignment/treatment and actual exposure separate. Use Qini/AUUC and ranking metrics. Do not use observed treatment share as a substitute for a verified per-row assignment propensity. Do not calculate PEHE. Official terms state CC BY-NC-SA 4.0; do not commit raw data and review non-commercial/share-alike terms before redistribution or commercial use.

### 3.4 Push-derived semi-synthetic data: primary known-truth comparison

This is the only primary source for directly comparing CATE/response-curve accuracy because its potential outcomes are known by construction.

Use real pre-treatment covariates and, where possible, participant histories from myBPmyLife. Optionally use Hillstrom pre-treatment customer features for a second covariate distribution.

Generate all potential outcomes before treatment assignment. Store truth in a separate evaluation object inaccessible to model training. Never include true effects, potential outcomes, future delivery, clicks, purchases or post-treatment history in model features.

Required DGPs:

**Binary**
- DGP-B1: constant/linear heterogeneous treatment effect;
- DGP-B2: nonlinear heterogeneous treatment effect with feature interactions;
- DGP-B3: sparse treatment-effect heterogeneity / rare high-uplift subgroup.

**Multi-treatment**
- DGP-M1: three arms including control, with shared and arm-specific effect components;
- DGP-M2: five active actions plus control, with different effect surfaces and treatment-arm imbalance;
- DGP-M3: user-specific eligibility mask, where only a subset of actions is available to each synthetic user. This is a synthetic eligibility mechanism, not a claim that the public MRT contains Shopee campaign triggers.

**Continuous dose**
- define dose as the number/intensity of Push interventions assigned in a fixed future policy window (e.g. 7 days), with campaign/content mix held fixed in the DGP;
- model pre-decision history (prior 7-day/30-day Push count, time since last Push, previous response and user activity) as covariates;
- generate monotone, saturating/diminishing-return and inverted-U fatigue response curves;
- use dose support 0–5 for the first experiment, and test denser dose grids only if supported;
- keep the historical Push count as a pre-treatment history feature, not as the current randomized dose.

Sample sizes for the first local full comparison:
- binary: 5k and 20k rows;
- multi-treatment: 5k and 20k rows;
- continuous: 5k and 20k rows;
- at least five fixed seeds for the main known-truth runs.

If the local MRT contains too few unique participant clusters for a requested synthetic size, generate repeated synthetic decision contexts while retaining the source-cluster identifier and documenting the resampling method. Do not claim that duplicated/resampled rows are new independent real users. Use cluster-aware splitting wherever a source participant identifier exists.

### 3.5 X5: optional observational realism check

Source: https://ods.ai/competitions/x5-retailhero-uplift-modeling/data

Use only if the user already has the data or can lawfully download it. The public task describes SMS communication and subsequent purchase, but the inspected public metadata does not establish random assignment. Use it for feature/overlap/propensity diagnostics and observational robustness only. No known-truth PEHE, no claim that a model recovered the true effect.

## 4. Dataset-to-experiment matrix

| Experiment | Primary known-truth data | Real randomized validation | Scale / robustness |
|---|---|---|---|
| A1 binary | Push-derived binary semi-synthetic | myBPmyLife pairwise; Hillstrom pairwise | Criteo unbiased v2.1; optional IHDP/ACIC |
| A2 multi-treatment | Push-derived 3-arm and 6-arm semi-synthetic with eligibility masks | Hillstrom 3-arm RCT; myBPmyLife 3-arm MRT | Synthetic arm imbalance; optional X5 only as observational evidence |
| A3 continuous | Push-context semi-synthetic dose-response | No public continuous randomized Push dose dataset in the current registry; myBPmyLife is not a randomized dose experiment | Sample-size, dose-density and support stress |
| Longitudinal overlay | Known-truth repeated-decision Push DGP | myBPmyLife MRT | X5 only for optional observational diagnostics |

Never pool real randomized, observational, and semi-synthetic scores into one leaderboard.

## 5. Local execution sequence

### Stage 0 — Preflight and environment

- Verify local branch and clean/dirty git status.
- Verify data files and available disk space.
- Create an isolated local environment.
- Verify MPS and CPU paths.
- Run a 1-batch forward/backward test on each neural framework.
- Generate an environment report before installing large or legacy dependencies.

### Stage 1 — Data readiness and baseline reproduction

- Reuse the existing dataset registry and loaders.
- Verify source file hashes, schema, row counts, missingness, outcome windows, treatment arms, randomization probabilities and license status.
- Re-run existing P1/P2/P3 smoke baselines before adding new models.
- Record which datasets are present and which are unavailable. Do not silently substitute another dataset.

### Stage 2 — A1 binary

First run small smoke tests (one seed, 1k–5k rows), then the main comparison (five seeds, 5k/20k Push semi-synthetic). Run Criteo at 100k/500k/1m for scale if available. Run Hillstrom and MRT real-data checks separately.

Model families:
- S/T/X/R/DR;
- Causal Forest/GRF if compatible;
- TARNet/CFRNet, DragonNet, FlexTENet;
- EFIN, DESCN, EUEN/EEUEN;
- TransTEE/HyperITE where a faithful local adapter is feasible.

### Stage 3 — A2 multi-treatment

- Hillstrom full 64k three-arm RCT;
- myBPmyLife all three randomized arms with outcome-specific estimands;
- Push-derived known-truth 3-arm and 6-arm data with eligibility masks.

Compare One-vs-Control, multi-arm outcome learner, MEMENTO, HydraNet, M3TN, MTMT and EFIN multi-treatment mode when data contracts and implementation permit. Keep rare-arm performance separate from macro-average.

### Stage 4 — A3 continuous dose

- Push-context known-truth DGPs: monotone, saturation and fatigue;
- sample sizes 5k/20k and five seeds;
- compare GPS/spline, DoseResponseGBM, DRNet, VCNet, GIKS variants, SCIGAN, TransTEE, HNet, CRNet/WDBN and DR²CE as feasible.

Do not claim a real randomized continuous-dose result from MRT.

### Stage 5 — longitudinal validation overlay

Compare WCLS proximal baseline, static CATE with pre-decision history features and one history-aware GRU/Transformer model if local resources permit. Evaluate known-truth carryover, fatigue, cooldown and temporal drift. This is not full sequential policy optimization.

### Stage 6 — report and recommendation

Report:
- exact hardware/software/device for each run;
- dataset source/hash/license/availability;
- model maturity and compatibility;
- all per-seed metrics and paired uncertainty intervals;
- best model by CATE accuracy, uplift ranking, calibration, policy value and runtime/memory;
- which models are statistically tied;
- what cannot be concluded from real randomized or observational data;
- which models are worth moving to CUDA only if local execution is actually bottlenecked.

## 6. Metrics and evidence rules

**Known-truth semi-synthetic**
- ATE bias/RMSE;
- CATE PEHE/effect RMSE;
- Qini/AUUC;
- calibration;
- oracle policy value/regret;
- continuous MISE / dose-response curve error;
- runtime, peak process memory and device.

**Real randomized**
- ITT with design probabilities;
- participant-clustered confidence intervals for MRT;
- arm-specific outcome analysis;
- valid held-out one-step policy value where assignment probabilities are known;
- Qini/AUUC only with the exact metric definition and randomized validation caveats.
- PEHE = NOT_AVAILABLE.

**Observational**
- propensity overlap, ESS, balance, sensitivity;
- no causal truth claim and no PEHE.

## 7. Completion and resource gates

Do not attempt the full Criteo dataset or every frontier model at once.

1. First run one 1k–5k smoke job per model family.
2. Profile wall time, peak RSS, MPS allocation and CPU fallback.
3. Run the core five-seed benchmark at 5k/20k.
4. Increase Criteo to 1m only after the memory preflight.
5. Run the full Criteo file only if the machine remains comfortably below memory pressure and the first scale runs finish reliably.
6. Execute MPS neural models sequentially.
7. If a model is CUDA-only, first check whether a faithful CPU/MPS port is practical. If not, mark it BLOCKED_ON_CUDA and continue. Do not silently replace it with a different model.
8. Recommend cloud CUDA only after reporting the actual local bottleneck and the expected benefit.

## 8. Data safety

- All raw data and participant-level data stay in ignored local directories such as `data/raw/`.
- Verify `.gitignore` covers raw data, generated datasets, caches, model checkpoints, and run artifacts.
- Never commit raw MRT, Hillstrom, Criteo or X5 data.
- Do not upload participant-level MRT data to external model APIs or public services.
- Commit only code, metadata-only manifests, aggregate metrics and documentation that do not expose participant records.
- Keep raw-data SHA256 and source provenance in local run manifests; do not include sensitive row-level data in logs.
