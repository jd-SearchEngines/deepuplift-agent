# Experiment A — Bias Diagnosis, Debiasing and Data-Repair Protocol

**Purpose:** Make every causal-model comparison explain *which bias exists, how it is measured, which correction is applied, what changes mathematically, and whether the correction actually improves estimation or decisions*. This is a required experiment layer, not an appendix of model names.
**Scope:** General causal inference and uplift modelling across randomized and observational marketing, advertising, retail, notification and other public datasets. Shopee CRM Push is a motivating example only.
**Evidence rule:** A correction is not successful merely because a model implements it. It must pass pre/post diagnostics and an appropriate outcome evaluation. No method can generally remove unmeasured confounding from observational data without additional assumptions or design information.

## 1. Required analysis chain

For each dataset and estimand, implement and report:

1. **Define the target trial / estimand:** unit, eligibility, time zero, treatment assignment, actual exposure/delivery, comparator, outcome and outcome window, follow-up, censoring, target population.
2. **Diagnose:** identify which assumptions are threatened and measure the size/location of the problem.
3. **Choose correction:** state the exact formula, fitted nuisance models, target population and assumptions.
4. **Verify:** rerun the same diagnostics after correction and evaluate estimator error or design-based outcomes.
5. **Quantify trade-offs:** bias, variance, effective sample size, support loss, uncertainty, runtime and policy value.
6. **State residual risk:** explicitly list what the correction cannot solve.

Do not label all of these problems as “confounding”. Treatment imbalance, lack of overlap, missing outcomes, noncompliance, time-varying confounding and unmeasured confounding are distinct problems and require different remedies.

## 2. Target-trial and data-contract audit (before modelling)

For every dataset, write a machine-readable estimand record containing:

- observational unit and cluster ID;
- eligibility rule and eligible-action set;
- decision timestamp / time zero;
- assignment variable and assignment mechanism;
- actual delivery/exposure variable, if present;
- treatment version / campaign or intervention descriptor;
- comparator/reference action;
- outcome variable, observation window and delayed-outcome definition;
- pre-treatment covariates and history window;
- missing outcome / censoring indicators;
- randomization probabilities or estimated propensity definition;
- target population (ATE, ATT, overlap population, eligible population, etc.).

**Critical separation:** assignment, delivery/exposure and outcome must be different fields. In randomized marketing datasets, primary ITT uses assignment. A delivered-exposure / per-protocol analysis is a different estimand and needs compliance assumptions or a valid instrument/design.

Do not use post-treatment delivery, click, purchase, downstream behavior or future history as pre-treatment covariates for the same decision.

## 3. Bias taxonomy, diagnostics and correction menu

### 3.1 Measured confounding

**Structure:** pre-treatment covariates (X) affect both treatment (T) and outcome (Y).

**Diagnose**
- Compare covariate distributions by arm before adjustment.
- Calculate standardized mean differences (SMD), variance ratios and distributional differences.
- Fit a propensity model (e(x)=P(T=1\mid X=x)) only when the assignment mechanism is not known by design.
- Inspect whether important confounders are absent, poorly measured or measured after treatment.

**Candidate corrections**
- Stratification / subclassification on propensity score.
- Matching, with explicit caliper, replacement and discarded-population accounting.
- Inverse-propensity weighting (IPW) or stabilized IPW.
- Outcome regression / g-computation.
- Doubly robust AIPW / DR-Learner with cross-fitting.
- For neural representation learning, CFRNet-style IPM/MMD/Wasserstein balance penalties or other explicitly documented balancing objectives.

**Verify**
- SMD and variance ratios before/after.
- Weighted covariate distribution plots.
- ATE/CATE estimate changes and confidence intervals.
- On randomized data or known-truth auxiliary data, check whether correction improves bias/RMSE; do not infer success from balance alone.

**Limit:** balancing measured covariates does not prove that unmeasured confounding is absent.

### 3.2 Treatment-selection bias / non-random assignment

**Structure:** treatment assignment depends on user characteristics, prior outcomes or an existing targeting policy.

**Diagnose**
- Reconstruct the assignment mechanism from documented fields and code where possible.
- Compare arm-level feature and history distributions.
- Estimate propensity only with pre-treatment information.
- Check whether assignment is deterministic for some strata or campaign types.

**Candidate corrections**
- IPW / stabilized IPW using the correct assignment propensity.
- Matching / stratification / overlap weighting for a clearly declared target population.
- Outcome regression or DR with cross-fitted nuisance models.
- In a known randomized trial, use the design probabilities; do not replace known probabilities with an unnecessary fitted propensity model.

**Important deterministic-policy case:** if historical records show only deterministic send/no-send decisions and there is no action variation within relevant contexts, positivity fails. Propensity modelling cannot create missing counterfactual support. The valid remedies are new randomized exploration, a credible natural experiment/instrument, a narrower estimand on supported strata, or an explicit non-identification conclusion.

### 3.3 Lack of overlap / positivity violations

**Problem:** some covariate profiles almost always receive one action and rarely/never receive another.

**Diagnose**
- Propensity-score density/histogram by arm.
- Min/max and quantiles of propensity per arm.
- Fraction outside common support.
- Maximum and upper quantiles of weights.
- Effective sample size (ESS), overall and per arm.
- For multi-arm data, pairwise/common support and arm-specific support.

**Candidate corrections**
- Pre-declared common-support restriction / trimming.
- Weight clipping or truncation, with threshold sensitivity.
- Overlap weights.
- Matching restricted to supported regions.
- Redefine target population explicitly if moving from ATE to an overlap-population estimand.

**Verify**
- Recalculate overlap, weight tails, ESS and covariate balance after correction.
- Report excluded population share and which users/actions were removed.
- Plot estimate and uncertainty against trimming/clipping thresholds.

**Limit:** trimming and overlap weighting change the target population or estimand. They do not recover effects in unsupported regions. Never report an overlap-population effect as the original population ATE.

### 3.4 Arm imbalance / rare treatment groups (multi-treatment)

**Problem:** treatment arms have very different sample counts or some actions are rare.

**Diagnose**
- Count eligible and assigned units per arm.
- Report arm-specific propensity distributions, ESS, outcome rates and uncertainty.
- Check whether rare arms occur only in a narrow covariate region.
- Separate sample-size imbalance from lack of covariate overlap.

**Candidate corrections**
- Generalized propensity score / multi-arm IPW or DR with a correct multi-class assignment model.
- Shared representation and arm-specific heads (e.g. multi-task or MMoE-style architectures).
- Hierarchical shrinkage / parameter sharing for sparse arms.
- Overlap-restricted evaluation.
- Training-only resampling may be tested as an optimization aid, but do not treat it as a causal correction or change evaluation prevalence.

**Do not**
- force all arms to 50/50 and claim this is the natural business population;
- use binary propensity APIs for a genuinely multi-arm assignment without a valid reduction;
- naively duplicate rare-arm rows and treat duplicates as independent observations;
- use SMOTE on treatment/outcome labels as if it solved confounding.

**Verify:** per-arm ATE/CATE error where truth is available, per-arm ESS, macro and support-weighted metrics, policy regret and uncertainty.

### 3.5 Representation imbalance / covariate distribution mismatch

**Problem:** learned representations remain predictive of treatment assignment, so treated/control representations differ.

**Candidate mechanisms**
- CFRNet-style balancing regularizer: penalize an IPM such as MMD or Wasserstein distance between treatment-group representation distributions.
- Domain/adversarial or treatment-invariant representation objectives, only when the exact loss and estimand are documented.
- EFIN treatment-aware interactions and intervention constraints are architecture-specific modelling mechanisms; do not describe them as universal confounding removal.
- DESCN jointly models treatment propensity, response and pseudo-effect signals; it does not guarantee protection against hidden confounders.

**Required ablations**
- factual outcome loss only;
- factual loss + representation balance;
- balance penalty strength sweep;
- shared representation vs treatment-specific representation;
- where applicable, treatment descriptor on/off.

**Verify**
- representation-space IPM/MMD before and after;
- factual outcome fit and held-out CATE metrics;
- rare subgroup performance;
- policy value and uncertainty.

**Limit:** representation balance is only with respect to observed data and the chosen representation metric. Excessive balancing can remove effect-modifying information and increase bias.

### 3.6 Nuisance-model misspecification and finite-sample bias

**Problem:** propensity model (e(x)), outcome models (m_t(x)), or both are misspecified.

**Candidate corrections**
- Compare simple and flexible nuisance learners.
- Cross-fitting: fit nuisance models on other folds and compute scores on held-out folds.
- Orthogonal scores / R-Learner / DR-Learner / AIPW.
- Hyperparameter tuning only on train/validation; never use test outcomes for model selection.
- Calibration checks for propensity probabilities and outcome predictions.

**Binary AIPW pseudo-outcome (illustrative):**

[
\widehat{\tau}_{DR}(X_i)=
\widehat m_1(X_i)-\widehat m_0(X_i)
+\frac{T_i}{\widehat e(X_i)}(Y_i-\widehat m_1(X_i))
-\frac{1-T_i}{1-\widehat e(X_i)}(Y_i-\widehat m_0(X_i)).
]

Here (\widehat e(x)=P(T=1\mid X=x)) and (\widehat m_t(x)=E[Y\mid T=t,X=x]). Under the relevant identification and regularity assumptions, the estimator is consistent if at least one nuisance component is correctly specified (propensity or outcome regression). This does **not** mean arbitrary hidden confounding is removed, nor that finite-sample variance and positivity problems disappear.

**Multi-arm extension:** for action (a) vs reference (0), use the corresponding arm-specific outcome models and generalized propensities (e_a(x),e_0(x)). Document the exact pseudo-outcome and cross-fitting scheme; do not silently reuse the binary formula.

**Verify**
- nuisance calibration and held-out prediction errors;
- cross-fit fold isolation;
- sensitivity to nuisance learner choices;
- ATE/CATE bias on auxiliary known-truth data;
- uncertainty and weight stability on real data.

### 3.7 Continuous-treatment confounding and dose-support problems

**Problem:** treatment is a dose/intensity (D), and dose assignment depends on (X). Sparse dose regions make the response curve unstable.

**Diagnose**
- Dose histogram overall and by key covariates/subgroups.
- Generalized propensity score (GPS) density (r(d,x)=f_{D\mid X}(d\mid x)).
- Conditional dose support and local density by dose.
- Covariate balance across dose strata or under dose weights.
- Extreme density ratios / effective sample size.
- Train/test support shift and extrapolation distance.

**Candidate corrections**
- GPS with spline/GAM/kernel dose-response.
- Continuous-treatment density-ratio weighting with a declared target dose distribution.
- Generalized propensity / balancing methods such as E2B, if exact source and objective are reproduced.
- DRNet / VCNet / GIKS and other dose-response learners, with explicit support guards.
- Restrict curve evaluation to common dose support; separately label extrapolation.

**Verify**
- covariate balance across dose bins / weighted dose groups;
- density and weight tails;
- dose-response curve stability under support restriction;
- MISE/curve error only on known-truth auxiliary data;
- on real observational data, report fit, support and sensitivity—not true PEHE.

**Limit:** a flexible neural response curve cannot identify effects at doses with no data support. Monotonicity constraints are justified only when the domain's causal response is known to be monotone; fatigue or saturation may be non-monotone.

### 3.8 Time-varying confounding, carryover and repeated decisions

**Problem:** prior treatment affects future covariates, and those covariates affect later treatment and outcome. A static propensity model on pooled rows can be biased.

**Diagnose**
- Reconstruct decision-time histories using only information available before each decision.
- Check treatment probability conditional on prior actions, outcomes, activity and time since last treatment.
- Check history-dependent overlap at each decision point.
- Measure carryover and delayed outcome windows separately.
- Cluster repeated decisions by participant/user.

**Candidate corrections**
- Marginal structural models with stabilized inverse-probability-of-treatment weights for sequential decisions.
- WCLS / weighted and centered least squares for proximal excursion effects in MRT settings.
- Sequential doubly robust / longitudinal TMLE when the data contract and implementation are valid.
- History-aware CATE models as predictors, not as a substitute for sequential identification.
- Pre-declared weight truncation and participant-cluster bootstrap.

**Verify**
- per-time-step propensity and cumulative-weight distributions;
- ESS by decision time;
- balance of time-varying histories after weighting;
- proximal effect bias on a known-truth sequence DGP;
- participant-clustered uncertainty on real MRT.

**Limit:** static DR-Learner does not by itself estimate a long-term dynamic treatment regime. Long-term policy value requires a sequential estimand and valid sequential OPE/continuation-value method.

### 3.9 Assignment vs delivery / noncompliance / exposure misclassification

**Problem:** assigned action may not be delivered, delivered action may not be seen, and the outcome may occur after only partial exposure.

**Required separation**
- assignment (Z);
- delivery/exposure (D);
- engagement/receipt if observed;
- outcome (Y).

**Candidate analyses**
- ITT: effect of assignment, preserving randomization.
- Per-protocol / exposure effect: separate estimand, with measured confounding and compliance assumptions.
- IV / CACE: only if assignment is a valid instrument and relevance, exclusion, independence and monotonicity assumptions are defended.
- Exposure-uplift models (e.g. EUEN/EEUEN) only with the required separate labels and faithful implementation.

**Verify:** assignment-to-delivery rate, compliance by arm, ITT vs exposure estimate, instrument diagnostics and sensitivity. Do not replace randomized ITT with a delivered-only comparison.

### 3.10 Missing outcomes, censoring and delayed observation

**Problem:** outcomes are missing or not yet observed within a fixed follow-up window.

**Candidate corrections**
- Define a fixed outcome window and administrative censoring rule.
- Inverse probability of censoring/observation weighting (IPCW) when conditional independent censoring and positivity are plausible.
- Multiple imputation as a sensitivity/estimation option under explicit missingness assumptions.
- Survival/time-to-event estimands when timing itself is the outcome.
- Separate short-term proximal and delayed outcomes; do not collapse them without a justified target.

**Verify:** missingness by arm and covariates, censoring model calibration, IPCW tails/ESS, complete-case vs adjusted sensitivity. State the missingness assumptions.

### 3.11 Interference and shared constraints

**Problem:** one user's treatment may affect another user's outcomes, or a shared campaign budget/frequency cap changes who else can be treated.

**Required handling**
- State the assumed interference structure (none, cluster-level, partial interference, network).
- If assignment is shared at campaign/region/time level, use the true randomization/assignment unit and cluster uncertainty accordingly.
- Keep individual treatment-effect estimands distinct from system-level allocation effects.
- If a global cap or deterministic priority rule changes candidate actions, treat it as part of the policy/assignment mechanism; do not assume independent user-level treatment assignment.

**Candidate designs:** cluster randomization, two-stage saturation designs, switchback/geo experiments, or an explicit interference-aware estimand where supported.

### 3.12 Data leakage and treatment-version inconsistency

**Diagnose**
- Build a feature timestamp audit: every model feature must be available before time zero.
- Check target leakage, post-treatment clicks/orders/delivery, future aggregates and duplicate user/decision records.
- Check whether nominally identical treatment values hide different message content, incentive, channel, send-time or frequency.

**Repair**
- Exclude post-treatment/future variables.
- Define treatment versions and either hold them fixed or model descriptors/interactions.
- Use temporal/group splits and source-level deduplication.
- If treatment versions cannot be made sufficiently consistent, narrow the estimand or report that the treatment is ill-defined.

## 4. Core diagnostics and formulas

### 4.1 Standardized mean difference

For a covariate (X):

[
SMD=\frac{\bar X_1-\bar X_0}
{\sqrt{(s_1^2+s_0^2)/2}}.
]

Report absolute SMD before and after adjustment for every key pre-treatment covariate. An absolute SMD around 0.1 is a commonly used diagnostic reference, not a universal proof of no confounding or a pass/fail causal guarantee.

### 4.2 Effective sample size

For non-negative weights (w_i):

[
ESS=\frac{(\sum_i w_i)^2}{\sum_i w_i^2}.
]

Report ESS overall and per arm, plus the retained population fraction. A method that improves balance while collapsing ESS may produce unstable estimates.

### 4.3 Binary IPW and overlap weights

For binary treatment propensity (e(X)):

- ATE IPW: (w_i=T_i/e(X_i)+(1-T_i)/(1-e(X_i))).
- Stabilized IPW: multiply by marginal treatment probability in the numerator.
- Overlap weights: treated (1-e(X_i)), control (e(X_i)).

Report exact normalization and whether weights are truncated. Overlap weights target the population with treatment equipoise; this is not generally the full-population ATE.

### 4.4 Weight clipping sensitivity

Run a pre-declared grid of clipping thresholds (e.g. no clipping and selected upper quantiles such as 99%, 99.5%, 99.9%). For each threshold report:
- maximum and quantiles of weights;
- ESS;
- SMD;
- estimate and confidence interval;
- retained support;
- policy value where valid.

Do not select a threshold only because it produces the most favorable effect estimate.

### 4.5 Multi-arm generalized propensity

For (K) actions, estimate (e_a(X)=P(T=a\mid X)) for every arm, check probability calibration and arm-specific support, and use arm-specific weights/pseudo-outcomes. Preserve the eligible-action mask when actions are not available to every unit. A softmax probability vector summing to one is not enough: support and calibration must also be checked.

### 4.6 Continuous GPS

For continuous dose (D), estimate the conditional density (r(d,x)=f_{D\mid X}(d\mid x)). If using density-ratio weights, state the target marginal/reference dose density in the numerator and the estimated conditional density in the denominator. Check local support and weight tails. Do not use a binary propensity formula for continuous dose.

## 5. Required correction ablation matrix

Run ablations on the same dataset, estimand, eligible population, split and seed. Keep randomization probabilities fixed when known.

| ID | Estimator / correction | What it tests |
|---|---|---|
| B0 | Raw unadjusted comparison | Baseline observed gap |
| B1 | Outcome regression / g-computation | Outcome-model adjustment alone |
| B2 | Propensity model + diagnostics only | Whether assignment is predictable and overlap is weak |
| B3 | IPW | Reweighting measured covariates to target population |
| B4 | Stabilized IPW | Variance stabilization from numerator choice |
| B5 | IPW clipping grid | Bias/variance/ESS trade-off from extreme weights |
| B6 | Matching / stratification | Restriction to comparable treated/control units |
| B7 | Overlap weighting | Effect in the equipoise population |
| B8 | AIPW / DR with cross-fitting | Orthogonal correction and nuisance-model robustness |
| B9 | Representation balancing (CFRNet/IPM or faithful model) | Learned representation balance |
| B10 | Sensitivity / negative-control analysis | How robust conclusions are to hidden confounding or falsification checks |

Not every correction applies to every dataset. For an RCT, use known design probabilities and focus on design-based estimates; observational propensity corrections are separate stress tests, not replacements for randomization. For continuous treatment, substitute GPS/density-ratio and continuous balancing ablations. For repeated decisions, add sequential weights/WCLS/longitudinal DR as a distinct longitudinal lane.

## 6. Model architecture vs debiasing procedure — do not conflate

- **IPW / matching / overlap weighting:** assignment/population reweighting procedures.
- **AIPW / DR / R-Learner:** orthogonal or doubly robust estimation strategies under explicit assumptions.
- **CFRNet / balancing representation:** representation-distribution balancing objectives.
- **EFIN:** treatment-feature interaction and intervention-constraint architecture; its constraint is not a universal confounding correction.
- **DESCN:** joint propensity/response/pseudo-effect modelling; not a guarantee against unmeasured confounding.
- **EUEN/EEUEN:** distinguishes assignment and exposure uplift; requires separate assignment and exposure fields.
- **VCNet / DRNet / GIKS:** continuous dose-response estimation; they do not automatically repair lack of dose support or hidden confounding.
- **Thompson Sampling / optimizers:** downstream exploration/allocation; they do not repair causal identification.

The report must identify which component is (a) an identification assumption/design, (b) a data repair or restriction, (c) a nuisance estimator, (d) an effect estimator/architecture, and (e) a decision optimizer.

## 7. Required output artifacts

Create/update:

- `BIAS_DIAGNOSIS_REPORT.md`: dataset-specific bias audit and target-trial contract.
- `BIAS_REPAIR_ABLATIONS.md`: method-by-method before/after results and exact configurations.
- `bias_diagnostics.json`: machine-readable propensity, balance, overlap, ESS, missingness, leakage and support summaries.
- `weight_diagnostics.parquet/csv`: aggregate/safe diagnostics only; no raw user identifiers in committed artifacts.
- Plots: propensity distributions, Love plots / SMD, weight histograms, ESS-vs-trimming curves, continuous dose-support plots, arm support maps.
- `MODEL_SOURCE_AND_RUN_STATUS.md`: exact source, maturity, execution and blocker.
- Tests for formulas, weights, propensity shapes, cross-fitting separation, treatment leakage and estimand metadata.

For every correction, report:

1. baseline diagnostic;
2. exact method/formula/configuration;
3. diagnostic after correction;
4. effect estimate / CATE metric / policy metric appropriate to evidence class;
5. variance or confidence interval;
6. ESS and retained population;
7. runtime/memory;
8. assumptions and residual bias.

## 8. Acceptance criteria

The bias-repair layer is complete only when:

1. Each dataset has a documented estimand and assignment/exposure/outcome contract.
2. At least one applicable correction is run end-to-end on a real observational dataset.
3. Randomized datasets preserve known assignment probabilities and ITT as primary.
4. Overlap, SMD, ESS and weight-tail diagnostics are implemented and tested.
5. IPW, stabilized/clipped IPW, overlap weighting and cross-fitted DR/AIPW are compared where the data contract permits.
6. Multi-arm and continuous-treatment corrections use generalized propensity/GPS contracts, not binary shortcuts.
7. At least one representation-balancing ablation is run where a faithful implementation is available.
8. A correction is not called successful unless both its intended diagnostic and relevant outcome/estimation metrics are reported.
9. Unmeasured confounding, unsupported treatment regions, interference and long-term effects are explicitly marked as unresolved where data cannot identify them.
10. The final report distinguishes bias reduction, variance/ESS cost, estimand change and policy-value impact.
