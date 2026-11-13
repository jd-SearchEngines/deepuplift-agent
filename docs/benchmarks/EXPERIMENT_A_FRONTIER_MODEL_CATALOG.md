# Experiment A — Frontier Causal Model Catalogue and Benchmark Matrix

**Purpose:** Expand Experiment A beyond classic meta-learners into recent, marketing-specific and research-frontier causal architectures.
**As-of date:** 2026-11-13.
**Repository:** `jd-SearchEngines/deepuplift-agent`
**Branch:** `codex/push-mrt-causal-benchmark`

This catalogue is a model-selection and reproduction specification, not a claim that every candidate is already implemented or that newer models are necessarily better. The benchmark must distinguish (a) the original published method, (b) an official implementation, (c) a faithful reimplementation, and (d) a simplified architecture-inspired variant.

## 1. Benchmark design principles

### 1.1 Three model layers

For every treatment regime, compare three layers:

1. **Statistical / causal baselines:** transparent estimators and tree/forest methods. These establish whether deep architectures actually add value.
2. **Established neural causal architectures:** representation sharing, potential-outcome heads, propensity heads, orthogonal/pseudo-outcome learning, treatment-feature interactions.
3. **Recent domain-specific / frontier architectures:** marketing uplift networks, multi-treatment MMoE, transformer or hypernetwork approaches, and recent continuous-dose balancing models.

Do not call the experiment complete after comparing only S/T/X/R/DR learners. Those remain necessary controls, not the full model set.

### 1.2 Separate estimands and evidence classes

- Binary: `tau_a(x) = E[Y(a)-Y(0) | X=x, eligible for a]`.
- Multi-treatment: `tau_a(x) = E[Y(a)-Y(0) | X=x, eligible for a]` for each action a, retaining the eligible-action mask.
- Continuous: `mu(x,t) = E[Y(t)|X=x]` and contrasts/derivatives relative to a declared reference dose.
- Repeated Push: estimate proximal excursion effects conditional on pre-decision history. Do not silently treat repeated decision rows as independent users or confuse a static ITE with a sequential treatment-regime value.

Evidence classes remain separate:
- `REAL_RANDOMIZED_MRT`;
- `SEMI_SYNTHETIC_KNOWN_TRUTH`;
- `REAL_RANDOMIZED_EXTERNAL`;
- `REAL_OBSERVATIONAL_ASSIGNMENT_UNVERIFIED`;
- `SYNTHETIC_SCALE_STRESS`.

A model may be evaluated in more than one evidence class, but scores must not be pooled across classes.

### 1.3 Fairness of comparison

For each model comparison, freeze:
- same eligible population and action set;
- same pre-treatment feature set;
- same outcome definition and label window;
- same split and seed;
- same nuisance information available to the estimator;
- comparable hyperparameter-search budget;
- same held-out evaluation population.

If an architecture requires treatment descriptors, exposure labels, multiple outcome tasks, a treatment hierarchy, or an eligible-action mask, provide those inputs consistently and document the requirement. Do not manufacture a feature just to make a model run.

## 2. A1 — Binary treatment: complete candidate set

### A1-0. Statistical and meta-learner baselines

| Candidate | Role | Why it stays in the benchmark |
|---|---|---|
| Outcome regression / S-Learner | Simple shared-response baseline | Tests whether treatment interaction is learnable from one response model |
| T-Learner | Separate outcome models | Strong transparent reference under enough samples per arm |
| X-Learner | Imputed treatment effects | Useful when treatment/control sample sizes differ |
| R-Learner | Orthogonal residualized loss | Reduces dependence on first-stage outcome fit under its assumptions |
| DR-Learner | Cross-fitted doubly robust pseudo-outcome | Combines outcome and assignment nuisance estimates |
| Causal Forest / GRF | Nonparametric forest | Strong nonlinear/subgroup baseline with uncertainty support |
| BART / Bayesian Causal Forest | Bayesian tree ensemble | Useful nonlinear baseline and posterior uncertainty where implementation is available |

### A1-1. Established neural causal architectures

| Candidate | Core inductive bias | Required comparison question |
|---|---|---|
| TARNet | Shared representation + treatment/control outcome heads | Does a shared encoder with separate outcome heads beat T-Learner? |
| CFRNet | TARNet plus representation-distribution balancing penalty | Does representation balancing improve CATE under measured confounding/imbalance? |
| DragonNet | Shared representation + outcome heads + propensity head + targeted regularization | Does joint assignment/outcome learning improve effect accuracy or only factual prediction? |
| FlexTENet | Adaptive sharing between common and treatment-specific representation subspaces | Does learned partial sharing help when potential outcomes share most but not all structure? |
| GANITE | Generative counterfactual imputation | Does adversarial counterfactual generation help in low-data / complex-outcome DGPs? |
| TransTEE | Transformer treatment-effect estimator with treatment/covariate attention | Does attention over treatment and covariates help when treatment descriptors and structured inputs exist? |

### A1-2. Marketing-specific and newer uplift architectures — mandatory frontier comparison

| Candidate | Published contribution | Push/CRM relevance | Reproduction notes |
|---|---|---|---|
| **EFIN (KDD 2023)** | Feature encoder includes user/context and treatment features; self-interaction estimates natural response; treatment-aware attention estimates treatment-specific uplift; intervention-constraint module regularizes uplift distribution across intervention groups | Very high when actions have meaningful descriptors: Campaign, message type, channel, content, incentive amount, send-time or frequency policy | Official repo: https://github.com/dgliu/KDD23_EFIN. Repo declares GPL-3.0 and legacy PyTorch/CUDA requirements. Prefer a clean adapter or faithful port; review license before copying code. EFIN is not meaningful if the treatment is only a bare binary flag and there are no treatment descriptors. |
| **DESCN (KDD 2022)** | Entire-space cross network jointly learns treatment propensity, response and pseudo treatment-effect signals; targets treatment bias and arm-size imbalance | High for observational CRM logs with historical targeting/selection and large control/treatment distribution differences | Official repo: https://github.com/kailiang-zhong/DESCN. Verify runtime and dependencies; use mainly in observational/selection-stress evidence as well as a clean RCT reference. |
| **EUEN / EEUEN (ICDM 2021)** | Explicit Uplift Effect Network; EEUEN extends to exposure-uplift and exposure-bias correction | High when assignment, actual push delivery/exposure and downstream outcome are separately logged | Official repo: https://github.com/aifor/eeuen; Apache-2.0. Original code depends on TensorFlow 1.12/XDL; port only the required model logic. Never replace randomized assignment ITT with delivered exposure without declaring a separate compliance/exposure estimand. |
| **HyperITE wrapper (AISTATS 2024)** | Hypernetwork-based soft weight sharing enables dynamic information sharing between treatment-specific learners | High when treatment arms are sparse or have very different sample sizes | Paper: https://proceedings.mlr.press/v238/chauhan24a.html. Implement as a wrapper/ablation over T/X-style learners, not as a completely unrelated base estimator. Test whether gains concentrate in low-sample arms. |

A1 must compare at least EFIN, DESCN, EUEN/EEUEN and FlexTENet/DragonNet in addition to the statistical learners, subject to their required input availability. TransTEE and HyperITE should be included where a faithful adapter is feasible. GANITE and BART are secondary candidates if they add a distinct capability and are executable within the project environment.

**Important model interpretation:**
- EFIN's intervention constraint is an architecture-specific regularizer, not a universal proof of confounding removal.
- DESCN's propensity/response/pseudo-effect joint learning is not a guarantee against unmeasured confounding.
- EUEN/EEUEN requires a precise distinction among treatment assignment, exposure/delivery and outcome.
- FlexTENet and HyperITE test different sharing hypotheses: shared/private subspaces versus generated, treatment-conditioned parameter sharing.
- DragonNet's propensity head does not by itself identify effects under arbitrary hidden confounding.

## 3. A2 — Multi-treatment: complete candidate set

### A2-0. Classical multi-arm controls

- Multi-arm S-Learner: include treatment as a categorical feature; enumerate each eligible action at inference.
- Multi-arm T-Learner: separate outcome head/model per arm.
- One-vs-Control X/R/DR learners: fit each active action against the same reference; use the correct arm-specific/generalized propensity.
- Generalized propensity score / multi-arm IPW and DR estimators as causal-statistical controls.
- Multi-arm causal forest / generalized random forest only if the library supports the exact estimand and the implementation passes synthetic truth tests.

### A2-1. Specialized multi-treatment architectures

| Candidate | Core architecture / hypothesis | Why it matters |
|---|---|---|
| **MEMENTO (CIKM 2022)** | Learns matching confounder representations across multiple discrete treatment groups by minimizing an upper bound on factual and counterfactual losses | Direct multi-treatment counterfactual representation model; Amazon e-commerce application makes it relevant to offers/CRM |
| **HydraNet (2025 journal publication; earlier work cited from 2022)** | Generalizes DragonNet to multi-valued treatments, with shared representation and treatment-specific response heads / multi-arm assignment modelling | Strong neural multi-arm causal baseline; official code: https://github.com/BorjaGIH/Hydranet |
| **M3TN (ICASSP 2024)** | Multi-gate Mixture-of-Experts feature representation plus an explicit uplift reparameterization module | Designed specifically for multi-valued marketing uplift; addresses parameter growth and accumulated error from many independent outcome heads |
| **MTMT (2024 preprint; later journal publication in 2026)** | Multi-task MMoE user encoder; decomposes treatment into a base “whether to treat” effect and an incremental “which treatment type” effect; treatment-user interactions estimate each task's uplift | Especially relevant when CRM has a common Push/offer decision followed by message/offer type, and multiple aligned outcomes |
| **EFIN multi-treatment mode** | Treatment descriptors are embedded and treatment-aware interactions produce action-specific uplift | Tests explicit feature interactions for several related campaign actions |
| **HyperITE multi-arm wrapper** | Dynamic soft parameter sharing across treatment groups | Tests whether scarce treatment arms benefit from borrowing statistical strength |
| **XTNet (2025 arXiv; exploratory)** | Dynamic neural masking and cross-treatment interaction decomposition for multi-category / multi-valued treatment combinations | Frontier candidate when actions interact or combinations are allowed; not a default production choice without verified code and independent replication |

Sources:
- MEMENTO: https://www.amazon.science/publications/memento-neural-model-for-estimating-individual-treatment-effects-for-multiple-treatments
- HydraNet: https://journals.sagepub.com/doi/10.1177/30504554251385053
- M3TN: https://arxiv.org/abs/2401.14426
- MTMT: https://arxiv.org/abs/2408.12803
- XTNet: https://arxiv.org/abs/2511.01641

**MTMT applicability gate:** its decomposition assumes a meaningful hierarchy between “whether to offer any intervention” and “which type of intervention”. If actions are mutually exclusive with no common base-treatment semantics, compare it only after documenting the factorization and testing whether it is valid. Do not impose the decomposition just because the model is recent.

**XTNet maturity gate:** the paper is a recent arXiv candidate. Do not call it a validated state-of-the-art baseline until source code, exact architecture, data protocol and independent reproduction are verified.

### A2-2. Required scenarios

- Balanced randomized assignment across arms.
- Strong arm imbalance / rare treatments.
- Shared and treatment-specific effect components.
- Different eligible-action sets across users.
- Correlated treatment descriptors (similar campaign variants).
- Multi-task outcomes only when the tasks are defined on the same randomized decision unit and have a coherent label window.
- Observational measured-confounding scenario, separate from randomized results.

The core comparison should include One-vs-Control, MEMENTO, HydraNet, M3TN, MTMT, and EFIN multi-treatment mode when their data contracts are satisfied. XTNet is an exploratory research lane.

## 4. A3 — Continuous treatment: complete candidate set

### A3-0. Statistical / flexible response baselines

- Generalized propensity score (GPS) with spline / kernel dose-response regression.
- Generalized additive model or spline-based varying-coefficient regression.
- DoseResponseGBM / tree-based dose-response.
- A continuous-treatment DR / orthogonal pseudo-outcome estimator where the target and nuisance contracts are mathematically valid.
- Continuous-treatment causal forest / generalized random forest only where supported by a tested API and correct continuous-treatment estimand.

### A3-1. Established neural dose-response architectures

| Candidate | Core idea | Benchmark role |
|---|---|---|
| **SCIGAN** | Generative adversarial counterfactual response model for continuous treatment | Generative dose-response baseline; monitor training stability and mode collapse |
| **DRNet** | Shared representation with treatment-dose strata / specialized heads and distribution regularization | Piecewise/local dose-response baseline |
| **VCNet** | Varying-coefficient network using a continuous treatment-dependent function (basis expansion) | Flexible smooth dose-response baseline |
| **TransTEE** | Transformer treatment/covariate interactions; supports discrete, continuous and structured treatment formulations | Transformer-family general-purpose comparison |
| **GIKS (AAAI 2024)** | Gradient interpolation near observed doses plus Gaussian-process kernel smoothing; augments factual training with inferred counterfactual doses | Recent model-agnostic counterfactual augmentation; compare standalone backbones and GIKS-wrapped DRNet/VCNet |
| **E2B (ICML 2022)** | End-to-end entropy balancing learns weights for continuous-treatment causal response estimation | Treat as a balancing/preprocessing wrapper, not as a standalone response architecture |

GIKS official code: https://github.com/nlokeshiisc/GIKS_release  
GIKS paper: https://ojs.aaai.org/index.php/AAAI/article/view/29353  
TransTEE paper: https://arxiv.org/abs/2202.01336  
E2B paper: https://proceedings.mlr.press/v162/bahadori22a.html

### A3-2. Recent 2025–2026 frontier candidates

| Candidate | Publication / contribution | Required treatment |
|---|---|---|
| **CRNet / WDBN (Neural Networks, 2025)** | Contrastive Balancing Representation Network and Weighted Double Balancing Network for heterogeneous dose-response; balances prognostic and treatment-related representations while preserving continuous treatment information | High-priority frontier candidate. Official CRNet repo: https://github.com/euzmin/Contrastive-Balancing-Representation-Network-CRNet. Verify whether WDBN implementation is included before claiming both are reproduced. |
| **DR²CE (International Journal of Approximate Reasoning, 2026)** | Disentangles instrumental, confounding and adjustment factors using mutual-information and IPM constraints for dose-response estimation; uses a varying-coefficient response head | Research candidate; implement only from the paper's exact objective/architecture. Do not equate it with generic DRNet or a simple feature split. |
| **HNet (Frontiers in AI, 2026)** | Hypernetwork maps a user representation to the weights of a per-instance dose-response network, making the dose-response function user-specific | High-priority recent architecture for heterogeneous frequency response. Paper: https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2026.1819009/full. Verify code availability before implementation. |
| **DRVAE (Neurocomputing, 2026)** | VAE-style disentangled covariate representation separates instrumental, confounding, adjustment and external-noise factors for continuous dose response | Exploratory candidate; verify official code, loss and assumptions before spending implementation time. |

Sources:
- CRNet/WDBN: https://doi.org/10.1016/j.neunet.2025.107600
- DR²CE: https://doi.org/10.1016/j.ijar.2025.109601
- HNet: https://doi.org/10.3389/frai.2026.1819009
- DRVAE: https://doi.org/10.1016/j.neucom.2026.132820

### A3-3. Required continuous-treatment scenarios

- Monotone increasing or decreasing response.
- Diminishing returns / saturation.
- Non-monotone fatigue / harm at high frequency.
- Different dose density and sparse tails.
- Dose-support shift between train and test.
- Confounder-dependent dose assignment.
- Hidden-confounding sensitivity, explicitly reported as non-identification rather than a model leaderboard.
- User-level heterogeneity in optimal dose and response curvature.

The core comparison should include GPS/spline, DoseResponseGBM, DRNet, VCNet, TransTEE, GIKS-DRNet, GIKS-VCNet, and HNet. Add CRNet/WDBN and DR²CE when faithfully reproducible. SCIGAN and DRVAE are secondary candidates if stable and executable.

**Do not confuse continuous treatment with repeated decisions:** “number of pushes in the last 7 days” is a derived dose only if the treatment versions and timing assumptions are sufficiently well-defined. If timing, message type and carryover differ, either hold them fixed in the DGP or model them as additional treatment dimensions / longitudinal history.

## 5. Cross-cutting repeated-decision / MRT validation overlay

The target CRM Push use case is naturally repeated and time-varying. A static binary or multi-arm ITE benchmark alone is insufficient.

Add a cross-cutting overlay to A1/A2/A3, not a fourth unrelated leaderboard:

- Use participant-clustered train/validation/test splits.
- Include only pre-decision history: prior push count, time since last push, previous action types, recent activity/purchase history, calendar/time features and eligibility.
- Keep current assignment, actual delivery and outcome separate.
- Include proximal outcome windows and delayed outcomes as distinct tasks/estimands.
- Generate known-truth longitudinal DGPs with zero/short/long carryover, cumulative fatigue, cooldown, heterogeneous history-dependent effects and temporal drift.
- Compare:
  1. existing WCLS-style proximal baseline;
  2. static CATE learner with pre-decision history features;
  3. history-aware sequence model (GRU/Transformer) only if it can be validated against the same known-truth DGP;
  4. a model that ignores history as a negative-control ablation.
- Evaluate excursion-effect bias/RMSE on known truth, proximal policy value, subgroup calibration, sensitivity to history truncation, runtime and participant-cluster bootstrap.
- Real MRT supports randomized proximal ITT and conditional excursion-effect analysis, but it does not reveal per-decision individual counterfactual truth. Do not calculate PEHE on real MRT.

Do not turn this overlay into a full sequential policy optimization benchmark. Long-term dynamic regime value and constrained scheduling remain separate work.

## 6. Implementation priority and maturity gates

### Tier 1 — Must implement or run if the required data contract is satisfied

- Binary: S/T/X/R/DR, Causal Forest, TARNet/CFRNet, DragonNet, FlexTENet, EFIN, DESCN, EUEN/EEUEN.
- Multi-treatment: One-vs-Control, MEMENTO, HydraNet, M3TN, MTMT, EFIN multi-treatment.
- Continuous: GPS/spline, DoseResponseGBM, DRNet, VCNet, TransTEE, GIKS-DRNet, GIKS-VCNet, HNet.
- Longitudinal overlay: WCLS, history-feature model, at least one history-aware neural candidate if data and sample size permit.

### Tier 2 — Implement when source and dependencies are verified

- HyperITE wrapper over T/X/R learners.
- BART / Bayesian Causal Forest.
- GANITE.
- SCIGAN.
- CRNet/WDBN.
- DR²CE.

### Tier 3 — Research-only / do not block the benchmark

- XTNet until code and reproduction protocol are verified.
- DRVAE until code and exact implementation are verified.
- Any claimed “VC-Transformer” until an exact paper and official implementation are identified. TransTEE is a different model and must retain its own name.

### Implementation maturity labels

Every model registry entry must use one of:
- `OFFICIAL_CODE_REPRODUCED`;
- `FAITHFUL_REIMPLEMENTATION_TESTED`;
- `ADAPTER_TO_EXISTING_LIBRARY`;
- `ARCHITECTURE_INSPIRED_VARIANT`;
- `PAPER_ONLY_NOT_IMPLEMENTED`;
- `BLOCKED_DEPENDENCY_OR_DATA`.

A simplified variant must never be reported as the original paper model.

## 7. Evaluation metrics by treatment type

### Binary

- ATE bias / RMSE and confidence-interval coverage on known-truth semi-synthetic data.
- CATE PEHE / effect RMSE on known-truth test data.
- AUUC, Qini, normalized Qini and LIFT@K for uplift ranking, with treatment prevalence and estimator definition recorded.
- CATE calibration slope/intercept and calibration by predicted-effect bins.
- Policy value / regret using known randomization or a valid held-out OPE estimator.
- Runtime, peak memory, parameter count and training stability.

### Multi-treatment

- Per-arm ATE bias/RMSE and per-arm PEHE on known truth.
- Macro- and support-weighted metrics across arms; do not let the largest arm hide rare-arm failure.
- Per-arm Qini/AUUC where the metric definition is valid.
- Best-action accuracy, regret and policy value under the eligible-action mask.
- Cross-arm calibration and treatment-arm confusion / support diagnostics.
- Parameter count, runtime and memory as the number of arms increases.

### Continuous

- MISE / integrated squared error of the full dose-response curve.
- AMSE / pointwise effect RMSE on a fixed dose grid.
- Dose-response calibration and derivative / shape error.
- Optimal-dose error and policy regret only when the oracle optimum lies inside observed support.
- Extrapolation rate, local support density and uncertainty by dose.
- Runtime, memory and stability.

### Repeated Push overlay

- Proximal/excursion effect bias and RMSE on known-truth longitudinal data.
- Clustered uncertainty coverage.
- One-step proximal policy value.
- Carryover/fatigue subgroup calibration and history-window sensitivity.
- Full trajectory policy value is NOT_AVAILABLE unless a valid sequential OPE / continuation-value estimator is supplied.

## 8. Research and reproduction sources

- EFIN KDD 2023 paper: https://arxiv.org/abs/2306.00315
- EFIN official code: https://github.com/dgliu/KDD23_EFIN
- DESCN KDD 2022 paper: https://arxiv.org/abs/2207.09920
- DESCN official code: https://github.com/kailiang-zhong/DESCN
- EUEN/EEUEN ICDM 2021 code: https://github.com/aifor/eeuen
- FlexTENet NeurIPS 2021: https://proceedings.neurips.cc/paper_files/paper/2021/hash/8526e0962a844e4a2f158d831d5fddf7-Abstract.html
- HyperITE AISTATS 2024: https://proceedings.mlr.press/v238/chauhan24a.html
- MEMENTO CIKM 2022: https://www.amazon.science/publications/memento-neural-model-for-estimating-individual-treatment-effects-for-multiple-treatments
- HydraNet 2025: https://journals.sagepub.com/doi/10.1177/30504554251385053
- M3TN ICASSP 2024: https://arxiv.org/abs/2401.14426
- MTMT: https://arxiv.org/abs/2408.12803
- XTNet 2025 preprint: https://arxiv.org/abs/2511.01641
- TransTEE: https://arxiv.org/abs/2202.01336
- GIKS AAAI 2024: https://ojs.aaai.org/index.php/AAAI/article/view/29353
- GIKS official code: https://github.com/nlokeshiisc/GIKS_release
- E2B ICML 2022: https://proceedings.mlr.press/v162/bahadori22a.html
- CRNet/WDBN Neural Networks 2025: https://doi.org/10.1016/j.neunet.2025.107600
- DR²CE 2026: https://doi.org/10.1016/j.ijar.2025.109601
- HNet Frontiers in AI 2026: https://doi.org/10.3389/frai.2026.1819009
- DRVAE Neurocomputing 2026: https://doi.org/10.1016/j.neucom.2026.132820
- Continuous-treatment uplift predict-then-optimize (EJOR 2026): https://doi.org/10.1016/j.ejor.2025.10.025 — this is an estimation-to-decision framework, not an additional response estimator; keep its optimization component outside Experiment A.

## 9. Completion definition

Experiment A is not complete because every model name appears in a table. It is complete when:

1. Tier-1 models have been executed or have a precise, evidence-backed blocker.
2. Tier-2 models are executed where practical and otherwise explicitly deferred with source/dependency reasons.
3. Every reported result maps to an exact model maturity label.
4. At least five fixed seeds are run for the main semi-synthetic comparisons, with paired uncertainty intervals.
5. Sample-size and treatment-arm imbalance stress are included.
6. Each CATE/response metric comes from that method's own predictions.
7. Real randomized data are evaluated with valid design-based estimands and no fabricated individual counterfactual metrics.
8. Model ranking is reported separately for effect accuracy, uplift ranking, calibration, policy value and efficiency.
9. The report identifies where the simpler model is statistically indistinguishable from the frontier model.
10. The result includes a recommendation for the actual CRM Push data regime, not a generic claim that the newest model is best.
