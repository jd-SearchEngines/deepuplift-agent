# Experiment A — Real-data Bias Diagnosis

**Run:** 2026-11-13, local Mac M3 Max, branch `codex/push-mrt-causal-benchmark`
**Evidence class:** real public observational NHEFS; no Push business result is inferred.
**Machine-readable evidence:** [`bias_diagnostics.json`](bias_diagnostics.json)

## Executive status

**PARTIAL.** A complete real-data observational correction path ran on NHEFS for
both a binary treatment (`qsmk`) and a continuous treatment
(`smokeintensity`). The binary lane ran cross-fitted propensity, outcome
regression, IPW, stabilized IPW, clipping, overlap weighting, propensity
matching and AIPW/DR. The continuous lane ran a Gaussian GPS, density-ratio
weights, clipping sensitivity and three existing dose-response models.

The requested myBPmyLife MRT, Hillstrom, Criteo, X5/RetailHero and MegaFon
files were not present in the bounded local audit. They remain
`NOT_AVAILABLE`; no synthetic table was substituted for a real-data claim.

## Data contract and audit

NHEFS is an observational smoking/weight-change study, not a randomized Push
experiment. The binary target-trial contract is: baseline 1971 covariates,
`qsmk` as quitting between the questionnaires, and `wt82_71` as weight change.
The continuous contract is: baseline 1971 covariates, cigarettes/day in 1971
as dose, and `wt82_71` as outcome. Follow-up variables, prices/taxes and
`qsmk` were excluded from the continuous feature set to avoid post-treatment
leakage.

| Lane | Rows | Treatment support | Missingness / selection | Assignment |
|---|---:|---|---|---|
| Binary qsmk | 1,566 | 1,163 control / 403 treated; treated rate 25.7% | causallib `restrict=True` complete-case table; zero missing in analyzed columns; excluded-row selection is not identified here | observational |
| Continuous smokeintensity | 1,566 | 1–80 cigarettes/day; p05=3, p50=20, p95=40 | dose/outcome missing rows excluded; baseline covariates retained and imputed by the project preprocessor; raw table has missing `pregnancies`, `alcoholhowmuch`, birthplace, income, SBP/DBP and cholesterol | observational |

The complete-case binary table has no duplicate rows. Its treatment rate is
unbalanced but not itself a bias proof. A propensity logistic model produced
range 0.0588–0.8142 and 100% of rows inside the 0.05–0.95 screening band.
This is a model-based diagnostic, not a positivity proof. Continuous GPS
conditional density reached `1.36e-7` and the raw marginal/conditional density
ratio reached **690.5**, which is an extreme-weight warning despite ordinary
binary propensity overlap.

## What was actually run

The implementation reuses `estimate_propensity`, `TabularPreprocessor`, the
existing treatment contracts, `effective_sample_size` and continuous support
diagnostics. New code adds explicit weight clipping, matching, cross-fitted
AIPW/DR reporting and continuous GPS density ratios. Every row records the
estimand, assumptions, balance, ESS, weight distribution and bootstrap
interval. The JSON is aggregate-only and contains no participant-level table.

### Internal model debiasing mechanisms

The experiment did not silently label architecture as confounding correction:

| Model family | Mechanism | What it does not guarantee | NHEFS status |
|---|---|---|---|
| CFRNet | balances learned representations with an IPM-style regularizer | does not guarantee exchangeability and can discard effect-modifying information | not run in this real-data lane; existing implementation is a smoke/experimental adapter |
| EFIN | treatment-aware feature interaction and intervention constraint | is not a universal propensity correction or unmeasured-confounding solution | not run; no faithful real-data adapter |
| DESCN | joint propensity/response/pseudo-effect heads and cross-head constraints | learned propensity and shared heads do not prove valid identification | not run; no faithful real-data adapter |
| DRNet / VCNet | continuous dose-conditioned outcome surface | factual prediction is not a GPS correction; VCNet-TR is not implemented | real NHEFS factual holdout run |

## Findings

1. The raw binary arm contrast is 2.541 kg, while cross-fitted outcome
   standardization is 3.578 kg and cross-fitted AIPW/DR is 3.364 kg. These are
   observational estimates under assumptions, not a ground-truth treatment
   effect.
2. IPW and stabilized IPW give the same point estimate (3.461 kg), but the
   stabilized weights have higher overall ESS (1,450 vs 1,017). This is a
   variance diagnostic, not evidence of less bias.
3. Overlap weighting yields 3.450 kg and the smallest maximum SMD among the
   weighting estimators (0.052), changing the target population to the
   overlap population.
4. Matching yields 2.519 kg for a matched ATT-like estimand, not an ATE. It
   retains 604 weighted/matched rows and has maximum SMD 0.066. It must not be
   compared as if it targeted the same population as AIPW.
5. Continuous GPS weights show materially heavier tails than binary
   propensity weights. The p99 density ratio is 5.92, p999 is 41.72 and the
   un-clipped maximum is 690.5. This is the main continuous-support risk.
6. DoseResponseGBM, DRNet and VCNet all produced factual holdout predictions;
   their RMSEs are 7.614, 7.800 and 8.043 respectively. NHEFS has no
   individual-level dose-response truth, so PEHE/ATE bias and causal ranking
   are `NOT_AVAILABLE`.

## Gate status

| Gate | Status | Evidence |
|---|---|---|
| Real observational binary diagnosis | PASS | 1,566 rows, all requested correction estimators executed |
| Real continuous GPS/density-ratio diagnosis | PASS | 1,566 rows, support and clipping sensitivity executed |
| Independent constraint / balance calculations | PASS | SMD, propensity, weights and ESS recomputed in the repair module |
| Causal truth metrics on NHEFS | NOT_AVAILABLE | no individual-level potential outcomes |
| Real Push MRT / randomized Hillstrom/Criteo path | BLOCKED_ON_DATA | local raw files absent |
| Frontier CFRNet/EFIN/DESCN real-data comparison | NOT_RUN | faithful adapters and/or required data contract unavailable |

The results do not establish a causal effect, online policy value, business
lift, or superiority of any model. Next step is to add a lawfully obtained
dataset with treatment assignment/delivery metadata, then rerun the same
contract and compare against randomized ITT where available.
