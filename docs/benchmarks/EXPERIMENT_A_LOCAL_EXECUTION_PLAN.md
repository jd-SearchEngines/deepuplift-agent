# Experiment A — Real-Data-First Local Execution Plan

**Decision (2026-11-13):** Real public datasets are the primary evidence. Use datasets with accessible schemas, documented provenance and runnable source code/loaders. Semi-synthetic data is an auxiliary diagnostic for metrics that require known individual counterfactual truth; it is not the main benchmark.
**Execution target:** User's Apple Silicon Mac, M3 Max, 64 GB unified memory.
**Execution mode:** Native macOS; CPU for classical/statistical models and PyTorch MPS where supported.
**Scope:** General causal inference / uplift modelling across advertising, marketing, retail, notifications, education and other public application domains. Shopee CRM Push is a motivating example, not a restriction on dataset or research scope.
**Branch:** `codex/push-mrt-causal-benchmark`

Read together with:
- [Real-data and source-code register](EXPERIMENT_A_REAL_DATA_AND_CODE_REGISTER.md)
- [Model comparison specification](EXPERIMENT_A_MODEL_COMPARISON.md)
- [Frontier model catalogue](EXPERIMENT_A_FRONTIER_MODEL_CATALOG.md)
- [Local Codex prompt](EXPERIMENT_A_LOCAL_CODEX_PROMPT.md)

## 1. Research principles

1. Real public data first. Do not invent a synthetic CRM dataset as the main benchmark.
2. Source code first. Prefer official model implementations and maintained libraries with loaders, scripts, tests and reproducible configs.
3. Scale down before abandoning. If a full dataset or model does not run, test a stratified/appropriate 1k, 5k, 10k, 50k or 100k subset, then increase gradually.
4. Keep evidence classes separate: real randomized, real observational, real paired/proxy-counterfactual, semi-synthetic known-truth and fully synthetic smoke tests.
5. Never claim that a real dataset has known individual treatment effects unless the data design truly supplies that information.
6. Do not force the Shopee CRM data structure onto unrelated public datasets. Preserve each dataset's actual treatment, population, outcome and assignment design.
7. A model without verified source code is a lower priority. Paper-only candidates are recorded, not allowed to block executable comparisons.
8. A run failure must have a diagnostic reason and a remediation attempt, not a vague "could not run".

## 2. Local execution and hardware

Use the local Mac first; do not rent a server or provision cloud GPU by default.

### 2.1 Preflight
Record macOS version, arm64 architecture, Python, PyTorch, MPS availability, memory, disk, dependency versions, and one real forward/backward smoke test. Record CPU/MPS/CPU-fallback for every model.

### 2.2 Runtime constraints
- Create an isolated `.venv-experiment-a`; preserve existing environments and user changes.
- Inspect project dependency files before installing. Do not blindly install old pinned TensorFlow/Numba/llvmlite stacks.
- MPS is not CUDA and does not support float64 tensors. Route float64-dependent methods to CPU unless a safe float32 implementation is validated.
- Do not use a Linux Docker container for the native MPS path.
- Run memory-heavy neural jobs sequentially. Limit CPU threads and record actual device use.
- Do not silently label CPU fallback as GPU execution.

## 3. Real dataset portfolio

The exact sources, code loaders, evidence class and limitations are listed in [EXPERIMENT_A_REAL_DATA_AND_CODE_REGISTER.md](EXPERIMENT_A_REAL_DATA_AND_CODE_REGISTER.md). The first execution priority is:

| Dataset | Evidence class | Treatment structure | Primary role | First run size |
|---|---|---|---|---|
| Criteo Uplift v2.1 | Real randomized advertising | Binary assignment; exposure separate | A1, large-scale marketing uplift | 10k/100k, then 500k/1m |
| Hillstrom | Real randomized email marketing | Three arms: no email / men's email / women's email | A1 pairwise and A2 multi-arm | 5k smoke, then all 64k |
| myBPmyLife MRT | Real randomized repeated notification trial | Three arms, repeated decisions | Repeated-decision / longitudinal and A2 | small participant subset, then all available rows |
| Lenta / X5 / MegaFon | Real observational marketing competition data | Campaign/contact assignment as documented per dataset | Confounding, overlap, selection-bias and robustness diagnostics | 5k/10k, then 50k/100k |
| NHEFS | Real observational cohort | Continuous exposure candidate: baseline smoking intensity (cigarettes/day); outcome weight change | A3 continuous-dose exploratory validation | all complete cases (small); also 500/1k sensitivity |
| Jobs / NSW | Real experimental + observational comparison data | Binary job-training treatment | A1 external causal validation / sensitivity | full small dataset |
| Twins | Real paired birth records | Binary heavier-twin indicator | A1 paired-outcome/proxy validation | full dataset |

**Dataset priority rules:**
- Criteo, Hillstrom and MRT are the first randomized-data runs.
- Lenta/X5/MegaFon are important real marketing observational datasets, but their assignment mechanism must be audited; do not label them randomized.
- NHEFS is a real observational continuous-exposure example. Use `smokeintensity` as the dose only after checking raw schema, missingness, time ordering and covariates. It has no individual counterfactual truth.
- Twins contains paired twins and observed outcomes for both members. It is a real paired-outcome benchmark, not automatically exact same-person counterfactual truth; document the pair-level estimand and assumptions.
- If a dataset source, download, license or loader cannot be verified, mark it unavailable and continue with the next dataset. Do not silently replace it.

## 4. Data-to-experiment matrix

| Experiment | First real randomized / paired evidence | Real observational evidence | Auxiliary known-truth data |
|---|---|---|---|
| A1 binary | Criteo; Hillstrom pairwise; MRT pairwise; Jobs/NSW; Twins with pair-level caveats | X5, Lenta, MegaFon | IHDP/ACIC or controlled semi-synthetic only for PEHE and calibration diagnostics |
| A2 multi-treatment | Hillstrom 3-arm; MRT 3-arm; MultiTwins only after treatment coding is verified | Lenta/X5/MegaFon if true multi-arm fields are present | small multi-arm semi-synthetic only for per-unit truth checks |
| A3 continuous | No verified randomized continuous-dose dataset in the current registry | NHEFS continuous exposure; optionally a verified real environmental-dose dataset | small dose-response semi-synthetic only to test MISE/curve error |
| Repeated decisions | myBPmyLife MRT | longitudinal marketing data only if source and event timestamps are available | small known-truth sequence DGP only for carryover/fatigue diagnostic |

**Important:** A3's real observational NHEFS run tests implementation, overlap, balance and sensitivity. It cannot establish true causal dose-response accuracy without stronger assumptions or randomization. Do not report PEHE on it.

## 5. Model execution policy: source code first

Every model must be registered with:
- paper / publication;
- official source URL, if any;
- source type: official / third-party reproduction / maintained library / paper-only;
- license;
- dependency stack and supported platform;
- data contract;
- local run status;
- fidelity label.

### Priority models

**A1 binary**
- Maintained/library baselines: S/T/X/R/DR learners, Causal Forest/GRF, BART where supported, using EconML/CausalML/scikit-uplift/libuplift where the API matches the estimand.
- Source-backed neural candidates: EFIN, DESCN, EUEN/EEUEN, DragonNet/TARNet/CFRNet, FlexTENet where exact code is verified.
- Run official code first. If legacy dependencies block native Apple Silicon, isolate the original environment or make a faithful minimal port. Do not silently call a simplified model the original.

**A2 multi-treatment**
- Hillstrom and MRT first.
- Compare One-vs-Control and a shared multi-arm learner.
- M3TN source exists as a third-party reproduction; verify exact fidelity and remove CUDA-only/distributed assumptions for a small local run if feasible.
- HydraNet only after its repository and runnable code are verified.
- MEMENTO / MTMT remain paper candidates until a matching official or faithful implementation is located. Do not prioritize paper-only code from unrelated projects.

**A3 continuous**
- NHEFS real observational continuous exposure first.
- Baselines: GPS + spline/GAM, CausalGPS, DoseResponseGBM.
- Source-backed neural candidates: DRNet, VCNet and GIKS (whose repository includes baseline implementations and scripts); SCIGAN and TransTEE if their official code and data contracts run.
- HNet / DR²CE / DRVAE / CRNet-WDBN are conditional candidates: only elevate after locating source, checking license/dependencies and confirming the code actually implements the named paper.

Use [EXPERIMENT_A_REAL_DATA_AND_CODE_REGISTER.md](EXPERIMENT_A_REAL_DATA_AND_CODE_REGISTER.md) as the model/source inventory. The model catalogue's research-only candidates do not become mandatory merely because they appear in a paper list.

## 6. Run ladder — diagnose, then reduce data

For each dataset × model:

1. **Source audit:** inspect README, license, requirements, data format, preprocessing, train/eval commands, CUDA assumptions and custom ops.
2. **Environment smoke:** import package, instantiate model, run one forward pass and one backward pass.
3. **Tiny data:** 1k rows (or all rows for small datasets), one seed, one epoch / minimum training schedule. Verify shapes, loss, gradients and metric.
4. **Small benchmark:** 5k–10k rows, fixed split, one seed.
5. **Medium benchmark:** 50k–100k rows or the full dataset if smaller.
6. **Full benchmark:** only if memory and runtime are acceptable.
7. **Repeated runs:** after the pipeline is stable, run at least three seeds for initial comparison and five seeds for the final core comparison.

If a model is too slow, first reduce rows, epochs, batch size, number of seeds, hidden dimension or hyperparameter-search budget. Record the exact reduction. Do not change the model's defining architecture or objective without relabeling it as an adaptation.

### 6.1 Failure taxonomy (required)

Every failed run must be assigned one or more explicit codes:

- `DATA_DOWNLOAD_OR_ACCESS`: URL 404, authentication, rate limit, unavailable mirror, missing archive.
- `LICENSE_OR_TERMS`: unclear, incompatible or restricted reuse terms.
- `DEPENDENCY_CONFLICT`: incompatible Python / TensorFlow / PyTorch / NumPy / Numba / compiler versions.
- `CUDA_ONLY`: hard-coded CUDA, custom CUDA kernel, distributed CUDA launcher, or no CPU/MPS path.
- `MPS_UNSUPPORTED_OP`: unsupported operator, float64, sparse operation or backend-specific limitation.
- `MEMORY_OOM`: CPU RAM or unified-memory exhaustion.
- `RUNTIME_TOO_SLOW`: measured time exceeds the declared budget.
- `DATA_SCHEMA_MISMATCH`: expected columns, treatment coding, labels, descriptors or tensor shapes differ.
- `REPRODUCTION_GAP`: missing code, configs, checkpoints, preprocessing, exact architecture or undocumented training protocol.
- `NUMERICAL_FAILURE`: NaN/Inf, unstable loss, invalid gradients or degenerate propensity.
- `METRIC_OR_ESTIMAND_INVALID`: no valid target/evaluation metric for the dataset design.

For every failure record: command, stack trace/error excerpt, versions, data size, device, elapsed time, memory peak, attempted fix, reduced-size result and final disposition.

### 6.2 Remediation order

1. Verify source and environment.
2. Try the official documented command in an isolated environment.
3. Run a tiny sample and short training schedule.
4. Remove unnecessary distributed/CUDA launcher requirements if the underlying model supports a single device.
5. Try CPU or MPS-compatible operations; use logged CPU fallback where valid.
6. Reduce sample size / epochs / batch size / seeds.
7. Make a faithful port only if the original method is clear and tests compare its outputs/objective.
8. If still blocked, mark `BLOCKED` with evidence and continue. Never fabricate a result.

## 7. Metrics and evidence boundaries

**Real randomized data**
- ITT / arm-specific effects with design-based uncertainty;
- Qini/AUUC/uplift@K with exact estimator, assignment probabilities and treatment prevalence;
- held-out policy value only with a valid randomized/OPE estimator;
- bootstrap or cluster bootstrap where appropriate;
- no individual PEHE unless the design actually supplies individual counterfactual truth.

**Real observational data**
- propensity distribution and overlap;
- standardized mean differences / covariate balance;
- effective sample size and weight tails;
- outcome-model and propensity-model diagnostics;
- sensitivity to trimming, overlap restrictions, unmeasured confounding and model specification;
- do not treat Qini alone as proof of causal validity; no PEHE.

**Real paired-outcome data**
- report pair-level contrast and the assumptions required to interpret it;
- do not equate paired twin outcomes with exact same-person counterfactuals without qualification.

**Auxiliary semi-synthetic / synthetic data**
- ATE bias/RMSE, CATE PEHE, dose-response MISE, calibration and oracle policy regret;
- use only to verify metric implementations and known-truth recovery, not as the main evidence for model superiority on real applications.

Do not combine scores across evidence classes into one leaderboard.

## 8. Required deliverables

1. `LOCAL_MAC_PREFLIGHT.md`: hardware, MPS, dependencies.
2. `DATASET_AUDIT.md`: datasets found, source, file hash, license, schema, missingness, treatment/outcome counts, download status.
3. `MODEL_SOURCE_AND_RUN_STATUS.md`: official/third-party/library source, dependencies, fidelity, status and blockers.
4. `EXPERIMENT_A_MODEL_COMPARISON_RESULTS.md`: separate results for real randomized, real observational, paired-outcome and auxiliary known-truth datasets.
5. Local ignored run artifacts: per-seed metrics, configs, logs, predictions, checkpoints and stack traces.
6. Tests for loaders, schema, treatment encoding, splits, leakage, metrics and each model adapter.

## 9. Data and Git safety

- Raw data and row-level outputs stay in ignored local directories.
- Never commit raw MRT, Hillstrom, Criteo, X5, Lenta, MegaFon, NHEFS or other participant/customer data.
- Do not upload sensitive participant data to external model APIs.
- Commit code, metadata-only manifests, aggregate non-sensitive metrics and documentation only.
- Record source, version, license, file hash and retrieval date locally.
