# Experiment A — Real Dataset and Source-Code Register

**Policy:** Real data first; executable source code first. This register is a shortlist for reproducible experiments, not a claim that every dataset has already been downloaded or every repository has been tested on the user's Mac.
**Research scope:** General causal inference and uplift modelling. Shopee CRM Push is an example, not a required data schema.

## 1. Real randomized / experimental datasets

| Dataset | Treatment / outcome | Approx. scale | Evidence class | Use | Source / loader |
|---|---|---:|---|---|---|
| Criteo Uplift v2.1 | randomized ad assignment; actual exposure separate; visit/conversion | 13,979,592 | REAL_RANDOMIZED_ADVERTISING | A1 binary; large-scale marketing | Data: https://ailab.criteo.com/criteo-uplift-prediction-dataset/ and https://huggingface.co/datasets/criteo/criteo-uplift. Example runnable benchmark code: https://github.com/kojikojikojikoji/uplift-modeling-benchmark (third-party reference implementation, not Criteo official model code). |
| Hillstrom | no email / men's email / women's email; visit/conversion/spend | 64,000 | REAL_RANDOMIZED_MARKETING | A1 pairwise; A2 three-arm | Original challenge: https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data-mining-challenge.html. Loader: https://github.com/jszymon/libuplift (`fetch_Hillstrom`) or https://github.com/binshuangli/uplift-bench. |
| myBPmyLife MRT | none/activity/diet notifications; proximal activity/diet outcomes | 187,517 decision points; 298 participants | REAL_RANDOMIZED_MRT | repeated decision / longitudinal; A2 | Data/code: https://github.com/WIRED-L-um/MRT_myBPmyLife. Verify exact local schema and license before use. |
| Jobs / NSW | job-training treatment; later employment/earnings outcomes | small (roughly 3k combined experimental/observational rows, dataset variant dependent) | REAL_EXPERIMENT_PLUS_OBSERVATIONAL_COMPARISON | A1 external validation | Load through https://github.com/binshuangli/uplift-bench or documented source from NBER; keep randomized NSW and observational comparison groups separate. |
| Twins | heavier-vs-lighter twin; one-year mortality | about 11,400 twin pairs | REAL_PAIRED_OUTCOME | A1 paired-outcome/proxy validation | Loader/documentation: https://causalml.readthedocs.io/en/stable/_modules/causalml/dataset/_benchmarks.html and source data referenced there. Interpret at pair level; do not overstate as exact same-person ITE truth. |

### Criteo download note
The original historical download endpoint may be unavailable. Use the official publisher page and the Criteo Hugging Face publisher mirror where appropriate. The public benchmark repository reports that its own downloader uses the official mirror first and the historical URL as fallback. Verify file hash, version and schema after download. The dataset terms are CC BY-NC-SA 4.0; do not assume commercial reuse is permitted.

### Hillstrom note
The three randomized arms support true multi-arm assignment comparisons. The experiment does not reveal individual counterfactual outcomes. Use design-aware Qini/AUUC and policy evaluation; no PEHE.

### MRT note
Repeated decision rows are nested within participants. Use participant-level splitting and clustered uncertainty. Keep activity and diet outcomes separate.

## 2. Real observational marketing / retail datasets

| Dataset | Domain | Approx. scale | Evidence class | Use | Source / loader |
|---|---|---:|---|---|---|
| Lenta | retail promotion campaign | ~687k in documented benchmark | REAL_OBSERVATIONAL_ASSIGNMENT_TO_VERIFY | A1 selection/overlap robustness | https://github.com/binshuangli/uplift-bench; original access via scikit-uplift dataset loaders |
| X5 RetailHero | retail SMS/promotion | ~200k train customers; associated customer/purchase tables larger | REAL_OBSERVATIONAL_ASSIGNMENT_TO_VERIFY | A1 propensity, overlap, selection-bias stress | https://ods.ai/competitions/x5-retailhero-uplift-modeling/data and https://github.com/binshuangli/uplift-bench |
| MegaFon | telecom marketing campaign | ~600k in documented benchmark | REAL_OBSERVATIONAL_ASSIGNMENT_TO_VERIFY | A1 observational robustness | https://github.com/binshuangli/uplift-bench; original access via scikit-uplift dataset loaders |

For these datasets, first inspect the original competition/task documentation and actual assignment mechanism. Unless randomization is established, do not label them RCTs. Do not calculate PEHE or claim a model has recovered true individual effects. Report overlap, propensity distributions, covariate balance, ESS, weight tails, trimming sensitivity and model-specification sensitivity.

The UpliftBench dataset documentation reports that its released benchmark caps Lenta/X5/MegaFon to 10,000-row samples for compute, while documenting larger original scales of roughly 687k/200k/600k. Use its source code and loaders as a starting point, but verify the current commit, preprocessing and sample size before reproducing.

## 3. Real observational continuous-exposure dataset

### NHEFS — National Health and Nutrition Examination Follow-up Study

- Public real observational cohort.
- A practical continuous-exposure candidate: baseline `smokeintensity` (cigarettes/day) as dose; `wt82_71` (follow-up weight change) as outcome; use only baseline covariates measured before the exposure/outcome window.
- Typical complete-case variants contain about 1,566–1,629 records; raw source variants may contain about 1,746.
- Source/loader: https://causallib.readthedocs.io/en/stable/api/generated/causallib.datasets.load_nhefs.html and https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/
- Continuous-treatment baseline source: https://github.com/NSAPH-Software/CausalGPS

Important qualification: common NHEFS teaching examples use smoking cessation `qsmk` as a binary treatment. For this benchmark, if using `smokeintensity` as a continuous dose, implement and document that separate exposure definition explicitly; do not mistakenly use the default binary `qsmk` loader output as a continuous treatment. This is observational and has no known individual dose-response truth. Use it to test real-data fitting, support, balance and sensitivity, not PEHE.

## 4. Auxiliary known-truth data — not the primary evidence

Use only when a metric requires ground truth unavailable in real data.

- IHDP / ACIC semi-synthetic variants: PEHE, ATE bias and CATE calibration implementation checks.
- Small in-repository semi-synthetic DGPs: unit tests for known truth, controlled confounding, overlap violations and dose-response curve metrics.
- Keep all such results in a separate `AUXILIARY_KNOWN_TRUTH` leaderboard.

Do not make a synthetic Push/CRM generator the main dataset or spend the first development cycle building a complex business simulator.

## 5. Source-code-first model register

### A1 binary

| Model / family | Source | Source status | Local execution note |
|---|---|---|---|
| S/T/X/R/DR, uplift metrics | https://github.com/uber/causalml ; https://github.com/bookingcom/causalml ; https://github.com/jszymon/libuplift | maintained/library source (verify exact package API/version) | start with scikit-learn CPU-compatible models |
| Causal Forest / GRF | https://github.com/grf-labs/grf ; https://github.com/py-why/EconML | maintained source | use R/Python implementation that matches target estimand |
| EFIN | https://github.com/dgliu/KDD23_EFIN | paper-associated official repository | inspect legacy dependencies and GPL-3.0; port only if necessary |
| DESCN | https://github.com/kailiang-zhong/DESCN | paper-associated official repository | verify old framework and treatment/outcome contracts |
| EUEN / EEUEN | https://github.com/aifor/eeuen | paper-associated source; Apache-2.0 per catalogue | original code has legacy TensorFlow/XDL dependency; isolate/port carefully |
| DragonNet / TARNet / CFRNet | https://github.com/claudiashi57/dragonnet ; https://github.com/clinicalml/cfrnet | source repositories | verify maintenance, dependency and exact paper variant |
| FlexTENet | paper: https://proceedings.neurips.cc/paper_files/paper/2021/hash/8526e0962a844e4a2f158d831d5fddf7-Abstract.html | source availability to verify before implementation | do not count paper-only as completed |

### A2 multi-treatment

| Model / family | Source | Source status | Local execution note |
|---|---|---|---|
| One-vs-Control / multi-arm S/T learner | https://github.com/jszymon/libuplift ; https://github.com/uber/causalml | maintained/library source | first multi-arm baseline |
| M3TN | https://github.com/SZU-AdvTech-2024/082-M3TN-Multi-Gate-Mixture-of-Experts-Based-Multi-Valued-Treatment-Network-for-Uplift-Modeling | third-party reproduction / source implementation, not assumed official | README uses multi-GPU CUDA launch; inspect and create a faithful single-device small-data path if possible |
| HydraNet | https://github.com/BorjaGIH/Hydranet | source repository reported with paper; verify fidelity and runnable commands | use small three-arm benchmark first |
| MEMENTO | paper: https://doi.org/10.1145/3511808.3557125 | no matching official source verified in this register | defer until exact code is located |
| MTMT | paper: https://arxiv.org/abs/2408.12803 | no matching official source verified in this register | defer until exact code is located |

### A3 continuous treatment

| Model / family | Source | Source status | Local execution note |
|---|---|---|---|
| GPS + spline / GAM | https://github.com/NSAPH-Software/CausalGPS | maintained R source | real-data baseline for continuous exposure |
| DRNet | https://github.com/d909b/drnet | original source; legacy TensorFlow 1.13 / Python 3.7 | likely dependency isolation/port needed |
| VCNet / VCNet+TR | https://github.com/nlokeshiisc/GIKS_release | source included in GIKS repository | verify exact implementation and configs |
| GIKS | https://github.com/nlokeshiisc/GIKS_release | official source for AAAI-24 paper | repository includes IHDP/NEWS and baseline scripts; run small first |
| SCIGAN | https://github.com/ioanabica/SCIGAN | source referenced by GIKS | verify TensorFlow dependency |
| TransTEE | https://github.com/hlzhang109/TransTEE | source repository | verify dosage branch and data contract |
| HNet / DR²CE / DRVAE / CRNet-WDBN | paper/source status must be checked individually | conditional candidates | do not prioritize until exact runnable code and fidelity are verified |

### Model maturity labels
Use only:
- `OFFICIAL_CODE_REPRODUCED`
- `FAITHFUL_REIMPLEMENTATION_TESTED`
- `THIRD_PARTY_REPRODUCTION_TESTED`
- `MAINTAINED_LIBRARY_ADAPTER`
- `ARCHITECTURE_INSPIRED_VARIANT`
- `PAPER_ONLY_SOURCE_NOT_FOUND`
- `BLOCKED_WITH_EVIDENCE`

A model appearing in a catalogue is not evidence that its code runs or that its result has been reproduced.

## 6. Why runs fail — record exact cause

The expected failure classes are:
1. Data source unavailable, URL moved/404, authentication, rate limit or archive mismatch.
2. License or terms prohibit download/use/redistribution.
3. Old dependency stack conflicts with current Python, macOS ARM64, NumPy, TensorFlow or PyTorch.
4. CUDA-only custom kernels, distributed launchers or hard-coded GPU calls.
5. MPS unsupported operators, float64 requirements, sparse operations or fallback bugs.
6. CPU RAM / Apple unified memory OOM.
7. Full-data or hyperparameter search runtime too long.
8. Data schema mismatch, treatment coding mismatch, missing descriptors, label-window mismatch.
9. Incomplete reproduction: no preprocessing, configs, checkpoints, code or exact objective.
10. Numerical instability or invalid propensity / metric estimand.

For each blocker, record the exact command, stack trace, environment, dataset size, device, elapsed time, peak memory, fix attempted, result after downsampling and final disposition.

## 7. Download and execution ladder

- Verify license and source first.
- Download or load via documented loader.
- Save source URL, retrieval date, file size and SHA256 locally.
- Start with 1k rows / all rows if dataset is smaller.
- Then 5k–10k, 50k–100k, then full data.
- For each model: import → instantiate → forward/backward → one-epoch run → small run → medium run → full run.
- Reduce rows before altering architecture. Reduce epochs/batch/search budget if needed.
- Run three seeds for early comparison and five seeds for final shortlisted models.
- If a model still cannot run, preserve logs and continue; do not fabricate a result.

## 8. Evaluation guardrails

- Real randomized: ITT, Qini/AUUC/uplift@K, design-aware confidence intervals and valid OPE; no PEHE.
- Real observational: overlap, balance, ESS, weight tails and sensitivity; no causal ground-truth ranking.
- Real paired: pair-level effect/contrast with assumptions.
- Semi-synthetic: PEHE, ATE bias, dose-response curve error and oracle regret only in auxiliary lane.
- No pooled leaderboard across evidence classes.
