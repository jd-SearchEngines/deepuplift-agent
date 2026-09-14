# v0.6 continuous benchmark evidence

These are local CPU runs for implementation and integration evidence. `PASS` means the requested fit, curve prediction, shared `EffectPrediction`, metrics, and continuous decision completed with finite outputs. It does not mean state-of-the-art quality or paper-protocol reproduction.

## Runs and provenance

- Synthetic: 240 observational rows, seed `20260914`, deterministic 75/25 split (180/60), 21 dose points over 0–20, DRNet/VCNet 50 epochs, GIKS factual/augmentation stages 40/20. Bundle: `runs/continuous/synthetic_v06_release/20260914T075743Z_seed20260914/`.
- IHDP: 747 rows, 25 features, official GIKS split 0 (471/276), 65 dose points, DRNet/VCNet 30 epochs and GIKS stages 25/15. Bundle: `runs/continuous/ihdp_v06_final/20260914T075954Z_seed0/`.
- NEWS: 2,993 rows, 498 features, official GIKS split 0 (2,000/993), 65 dose points, DRNet/VCNet 30 epochs and GIKS stages 25/15. Bundle: `runs/continuous/news_v06_final/20260914T080000Z_seed0/`.
- CCPFN: 80 synthetic observational rows, seed `23`, 60/20 split, actual `ccpfn.CEPOEstimator` inference with `ccpfn==0.1.2`, PyTorch `2.5.1`, and the cached published weights. Bundle: `runs/continuous/ccpfn_v06_release/20260914T075756Z_seed23/`. A fresh host needs network access to fetch weights.
- TCGA: `DATA_NOT_AVAILABLE`; no local licensed file was supplied. The CLI returns that status without fabricating metrics.

IHDP and NEWS files were read locally from the official GIKS release checkout and were not copied into this repository. Their upstream redistribution terms are unverified. The synthetic and CCPFN evaluation uses known synthetic response truth. Costs follow the example subsidy curve in the tutorial and are not empirical business costs.

The measured Python 3.11.15 environment used NumPy 1.26.4, pandas 3.0.5, scikit-learn 1.6.1, PyTorch 2.5.1, and psutil 7.2.2; inference and training used CPU. Runtime columns are fit / full-curve prediction / decision seconds. Peak RSS is sampled per model.

## Held-out results

MISE is integrated mean squared response-curve error; dose error is mean absolute maximum-effect-dose error; economic regret is measured against the known economic optimum. Values are rounded for display; JSON bundles retain full precision.

| Dataset | Model | Status | MISE | Dose error | Economic regret | Fit / curve / decision s | Peak RSS MB |
|---|---|---|---:|---:|---:|---:|---:|
| Synthetic | DoseResponseGBM | PASS | 0.0530 | 5.9333 | 0.0281 | 0.065 / 0.009 / 0.002 | 159.2 |
| Synthetic | DRNet | PASS | 0.1045 | 6.8333 | 0.1004 | 0.874 / 0.002 / 0.002 | 362.5 |
| Synthetic | GIKS-DRNet | PASS | 0.0506 | 4.8333 | 0.0427 | 0.183 / 0.005 / 0.004 | 367.3 |
| Synthetic | VCNet | PASS | 0.0623 | 6.9500 | 0.0235 | 0.127 / 0.001 / 0.002 | 367.9 |
| Synthetic | GIKS-VCNet | PASS | 0.0426 | 5.4833 | 0.0745 | 0.144 / 0.017 / 0.003 | 368.5 |
| IHDP | DoseResponseGBM | PASS | 2.2478 | 0.0209 | 0.1814 | 0.099 / 0.035 / 0.027 | 288.9 |
| IHDP | DRNet | PASS | 7.7907 | 0.1250 | 0.2043 | 0.383 / 0.005 / 0.029 | 373.2 |
| IHDP | GIKS-DRNet | PASS | 8.1956 | 0.1246 | 0.2541 | 0.351 / 0.004 / 0.028 | 391.1 |
| IHDP | VCNet | PASS | 15.9814 | 0.1250 | 9.9374 | 0.025 / 0.003 / 0.028 | 397.0 |
| IHDP | GIKS-VCNet | PASS | 15.0696 | 0.1250 | 9.9374 | 0.130 / 0.038 / 0.035 | 410.0 |
| NEWS | DoseResponseGBM | PASS | 1.3360 | 0.1788 | 1.7497 | 1.136 / 0.149 / 0.095 | 373.2 |
| NEWS | DRNet | PASS | 1.4516 | 0.2028 | 2.1285 | 0.668 / 0.040 / 0.100 | 596.7 |
| NEWS | GIKS-DRNet | PASS | 1.4058 | 0.1759 | 2.1665 | 0.876 / 0.027 / 0.100 | 763.9 |
| NEWS | VCNet | PASS | 1.3631 | 0.2156 | 1.2911 | 0.146 / 0.066 / 0.101 | 767.5 |
| NEWS | GIKS-VCNet | PASS | 1.2797 | 0.2038 | 1.3324 | 0.573 / 0.023 / 0.101 | 774.7 |
| Synthetic | CCPFN | PASS | 0.0209 | 8.8000 | 0.0929 | 1.452 / 0.234 / 0.001 | 608.8 |

## GIKS execution and interpretation

Both bases ran the factual-only stage and the separate augmentation stage. On the synthetic split each accepted 180/180 pseudo-labels: GI accepted 38/38 nearby-dose labels and GP smoothing accepted 142/142 far-dose labels. On IHDP each accepted 471/471 (GI 138/138; GP 333/333). On NEWS each accepted 1,998/2,000 (GI 553/553; GP 1,445/1,447). These are execution counts, not verified label quality.

GIKS is an independent implementation of the paper's two-stage mechanism. Its exact RBF Gaussian process uses standardized observed covariates and dose with posterior-variance filtering and weighting; it does not reproduce the official learned embedding/supervision procedure. IHDP scores show that the neural estimators and GIKS variants are poorly calibrated under this compact training budget. No model ranking or quality lift is claimed.

The maximum-effect dose and the maximum-net-value dose are separately stored in the benchmark metrics and shown in each report. The policy uses `Effect(x, dose) * outcome_value - cost(dose)`; `recommended_treatment` remains only the model's maximum-effect reference.

## Model status

| Model | Implementation status | Runnable / evidence | Maturity | License / source notes |
|---|---|---|---|---|
| DoseResponseGBM | Implemented | Fit/predict/decision and all three suite datasets PASS | EXPERIMENTAL | DeepUplift MIT implementation; scikit-learn, no external source code copied |
| DRNet | Implemented native port | Fit/predict/decision and all three suite datasets PASS | OPTIONAL / EXPERIMENTAL | ICLR 2021 architecture; upstream has no declared license; no code copied |
| VCNet | Implemented native port | Fit/predict/decision and all three suite datasets PASS | OPTIONAL / EXPERIMENTAL | ICLR 2021 varying-coefficient base; no VCNet-TR |
| GIKS-DRNet / GIKS-VCNet | Implemented native two-stage estimators | Both ablations PASS on synthetic, IHDP, and NEWS | OPTIONAL / EXPERIMENTAL | Apache-2.0 reference; no code copied; documented methodological differences |
| TransTEE | Not implemented as a runnable adapter | `OPTIONAL_NOT_VALIDATED`; fail-closed marker; no installable dependency | OPTIONAL_NOT_VALIDATED | Official source is MIT; no `transtee` extra |
| CCPFN | External inference adapter | Real fit/context, CEPO curve, and decision PASS | OPTIONAL / FRONTIER_EXPERIMENTAL | `ccpfn==0.1.2`, Apache-2.0; first-use weights require network |
| CausalFM | Deferred | NOT_IMPLEMENTED in this pack | NEXT_VERSION | Separate CATE toolkit is not a continuous-dose estimator |
