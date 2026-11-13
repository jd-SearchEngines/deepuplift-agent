# Experiment A — Bias Repair Ablations

This report preserves the first real-data bias-repair run. It does not replace
the earlier synthetic Experiment A report. Values below are from the aggregate
JSON artifact [`bias_diagnostics.json`](bias_diagnostics.json).

## Binary NHEFS ablation

Outcome is `wt82_71` in kg. Intervals are row-bootstrap 95% intervals over
fixed cross-fitted nuisance predictions (`250` replicates). The observational
data has no known ATE, so “estimate” is not a truth-error metric.

| Method | Estimand / repair | Estimate | 95% CI | Max SMD after | ESS overall | ESS treated | ESS control |
|---|---|---:|---:|---:|---:|---:|---:|
| Unadjusted | observed arm contrast | 2.541 | [1.614, 3.526] | 0.282 | 1566.0 | 403.0 | 1163.0 |
| Outcome regression | cross-fitted standardization | 3.578 | [3.351, 3.777] | 0.282 | 1566.0 | 403.0 | 1163.0 |
| IPW | ATE under exchangeability/positivity | 3.461 | [2.486, 4.514] | 0.075 | 1016.6 | 331.3 | 1127.4 |
| Stabilized IPW | same ATE target, marginal stabilization | 3.461 | [2.356, 4.600] | 0.075 | 1449.8 | 331.3 | 1127.4 |
| IPW clip p99 | capped extreme weights | 3.465 | [2.497, 4.335] | 0.075 | 1076.9 | 351.2 | 1127.4 |
| IPW clip p99.5 | capped extreme weights | 3.464 | [2.497, 4.470] | 0.072 | 1060.9 | 346.5 | 1127.4 |
| IPW clip p99.9 | capped extreme weights | 3.468 | [2.556, 4.413] | 0.075 | 1034.2 | 337.7 | 1127.4 |
| Overlap weighting | overlap-population effect | 3.450 | [2.569, 4.427] | 0.052 | 1119.5 | 390.4 | 981.2 |
| AIPW / DR cross-fit | doubly robust pseudo-outcome | 3.364 | [2.376, 4.414] | 0.282 | 1566.0 | 403.0 | 1163.0 |
| Propensity matching | ATT-like matched treated | 2.519 | [1.583, 4.330] | 0.066 | 603.8 | 400.0 | 242.4 |

The before-adjustment maximum absolute SMD is 0.282. The different target
populations matter: overlap weighting and matching are not ATE estimators in
the same finite-sample sense as IPW/AIPW. The wide intervals and point
estimate movement are reported as uncertainty/identification sensitivity, not
as proof that one repair is correct.

## Mathematical and implementation notes

- **IPW:** `w=T/e(X)+(1-T)/(1-e(X))`; arm-normalized weighted means estimate
  an ATE under consistency, exchangeability and positivity.
- **Stabilized IPW:** treated weights multiply by `P(T=1)` and controls by
  `P(T=0)`. It changes variance behavior, not the identification assumptions.
- **Clipping:** `min(w, cap)` at the empirical p99/p99.5/p99.9 of raw IPW.
  This is a finite-sample bias/variance sensitivity and must be disclosed.
- **Overlap weighting:** treated `1-e(X)`, control `e(X)`. It targets the
  overlap population and usually improves balance when propensity tails are
  problematic.
- **Matching:** nearest control on logit propensity with replacement and a
  0.2-SD caliper. The output is ATT-like and reports matched support rather
  than pretending to be ATE.
- **Outcome regression:** cross-fitted estimates of `m1(X)-m0(X)`; it depends
  on outcome-model specification.
- **AIPW/DR:** `m1-m0 + T(Y-m1)/e - (1-T)(Y-m0)/(1-e)`. Cross-fitting keeps
  nuisance predictions out-of-fold. Double robustness is conditional on a
  valid identification model and consistent estimation of at least one
  nuisance component; it does not fix hidden confounding.

## Continuous NHEFS ablation

The GPS model is a HistGradientBoosting conditional mean plus Gaussian
residual density. A marginal KDE divided by the conditional density gives the
density-ratio weight. This is a working GPS model, not a proof of generalized
positivity.

| Density-ratio cap | Cap | ESS | Weighted outcome mean | Weighted dose mean |
|---|---:|---:|---:|---:|
| p99 | 5.920 | 963.0 | 2.577 | 22.136 |
| p99.5 | 11.351 | 729.6 | 2.646 | 22.938 |
| p99.9 | 41.724 | 378.7 | 3.038 | 24.662 |

The raw ratio has p50 0.926, p95 2.105, p99 5.920 and maximum 690.510.
The dose-response surface models are factual prediction models, not a GPS
adjustment: DoseResponseGBM (CPU, RMSE 7.614, 0.26 s), DRNet (MPS, RMSE
7.800, 1.67 s) and VCNet (MPS, RMSE 8.043, 0.12 s) all ran on a deterministic
80/20 holdout. No PEHE, dose-response bias or policy value is computed because
NHEFS provides no individual counterfactual dose-response truth.

## Limitations and next ablations

1. The causallib binary loader is complete-case restricted; selection into
   that table is documented but not repaired. A future run should compare
   explicit missingness indicators/multiple imputation against complete-case
   analysis without changing the held-out target population silently.
2. The binary propensity logistic fit emitted convergence warnings on the
   unscaled engineered features. The estimates were finite and the run
   completed, but the next run should add a documented scaled-logit and
   calibrated GBM sensitivity rather than suppressing the warning.
3. Continuous GPS needs local support diagnostics by covariate strata and a
   flexible conditional density sensitivity. A single Gaussian residual model
   is not sufficient for a production decision policy.
4. Real randomized Hillstrom/Criteo/MRT data and participant-level Push
   history remain required for ITT, longitudinal OPE and policy-value gates.
