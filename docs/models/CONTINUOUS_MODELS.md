# Continuous treatment models

Continuous estimators consume a `CausalDataset` with `treatment_type="continuous"` and return one `EffectPrediction`. The prediction contains a shared `dose_grid`, per-dose outcome and effect arrays, `baseline_dose`, and an optional uncertainty curve. The model's `recommended_treatment` is a per-unit maximum-effect reference. `build_continuous_policy()` makes the business decision by comparing estimated effect times outcome value with dose cost.

| Model | Core idea | Observational use | Dependencies | Maturity in DeepUplift | Evidence |
|---|---|---|---|---|---|
| DoseResponseGBM | Gradient-boosted outcome regression on covariates and dose | Yes, subject to measured-confounding assumptions | Core scikit-learn | EXPERIMENTAL | Runnable reference and synthetic benchmark |
| DRNet | Shared covariate representation with quantile dose strata and dose-conditioned heads | Yes | `deepuplift[continuous]` (PyTorch) | OPTIONAL / EXPERIMENTAL | Native CPU fit, full-curve prediction, and decision smoke |
| VCNet | Learns dose-varying coefficients and combines them with a continuous dose basis | Yes | `deepuplift[continuous]` (PyTorch) | OPTIONAL / EXPERIMENTAL | Native CPU fit, full-curve prediction, and decision smoke |
| GIKS-VCNet / GIKS-DRNet | Factual-only fit followed by counterfactual augmentation | Yes, but overlap and pseudo-label quality remain important | `deepuplift[giks]` (PyTorch) | OPTIONAL / EXPERIMENTAL | Both base-model ablations run on synthetic, IHDP, and NEWS; GI and GP paths are counted in metadata |
| TransTEE | Transformer attention over covariates and treatment | Paper covers continuous and dosage treatments | Official source has no installable PyPI package | OPTIONAL_NOT_VALIDATED | Not runnable through DeepUplift in this release |
| CCPFN | Pretrained in-context CEPO model for continuous treatments | Context-based observational inference | `deepuplift[ccpfn]`; first use may download model weights | OPTIONAL / FRONTIER_EXPERIMENTAL | Real inference smoke passed with `ccpfn==0.1.2` |

## Implementations and limits

DRNet uses a shared representation and a bank of heads selected by quantile dose strata. Each head receives the numeric dose at both hidden layers. This preserves the dose-stratified model family described in the VCNet paper, while remaining an independent implementation; it does not copy code from the unlicensed upstream repository. Stratum boundaries can create response discontinuities.

VCNet uses a learned coefficient vector `beta(x)` and cubic Bernstein dose basis `B(t)`, producing `mu(x,t) = sum_k beta_k(x) B_k(t)`. This makes the estimated response continuous in dose. It is a clean PyTorch base-model implementation, not the authors' code. The paper's original spline construction and VCNet-TR functional targeted regularization are not implemented here.

GIKS performs two distinct training stages. Factual training uses only observed rows. The second stage samples independent counterfactual doses. Near a factual dose, it anchors a first-order Taylor estimate at the observed outcome using the fitted dose gradient. At farther doses, it runs an RBF Gaussian-process posterior over standardized covariates and treatment, rejects labels above a posterior-variance threshold, and uses the remaining posterior uncertainty to reduce sample weight. It then fine-tunes the base estimator on factual plus accepted pseudo rows. The implementation is independent of the official Apache-2.0 source. Its GP kernel acts directly on input covariates and dose, whereas the paper's implementation uses a learned embedding and its own GP supervision procedure. It has not been validated on the authors' IHDP, NEWS, or TCGA protocols.

CCPFN delegates to the maintained `ccpfn.CEPOEstimator` API. `fit()` supplies factual context; it does not train the foundation model. `predict()` queries `estimate_cepo(X, dose)` at each grid point and adapts the returned conditional expected potential outcomes. The optional backend's observed package version is recorded in `EffectPrediction.metadata`. The PyPI extra pins the verified API version and constrains NumPy below 2 to match DeepUplift's current core compatibility range. Weight availability and first-use network requirement are reported by `model_info("CCPFN")`.

TransTEE's official MIT-licensed source supports continuous/dosage settings, but the source tree is not an installable package and no faithful adapter has passed fit/predict validation here. No `transtee` extra is published. CausalFM has a separate Apache-2.0 toolkit and a standard CATE API; it does not estimate continuous dose-response curves. Its optional back-door adapter is recorded for the next version rather than expanding this continuous pack.

## Data and metrics

`synthetic_dose_response()` generates measured confounding, nonlinear dose assignment, heterogeneous concave response curves, true effect curves, and true effect/economic optima. It does not place potential outcomes in the estimator's input frame. `load_ihdp()` and `load_news()` accept a user-provided normalized CSV; `load_giks_continuous_benchmark()` also reads the official GIKS `.pt` matrices, fixed split files, and NumPy response curves using restricted loading. `load_tcga()` accepts a local normalized CSV or GIKS tensor matrix. Raw data is never downloaded implicitly or redistributed. Upstream data terms remain unverified. The smoke benchmark may therefore report measurements from locally supplied official files while marking their license status as unverified; no files are included in this repository.

The benchmark uses MISE (integrated squared response-curve error), dose-response RMSE, integrated absolute error, ADRF and ICTE/CATE curve RMSE, maximum-effect dose error/regret, and economic dose error/regret. Economic regret uses the benchmark's explicit demonstration cost curve; it is an evaluation assumption, not an empirical cost claim. Qini and AUUC are not continuous-treatment metrics here.

## Treatment Support

Each fitted continuous estimator records `observed_dose_min` and `observed_dose_max` from complete factual training rows. By default, its predicted dose curve stays inside that observed interval. The default baseline is dose 0 only when factual training rows actually contain dose 0; otherwise the estimator uses the observed lower endpoint as a supported reference and records `no_treatment_supported: false`. A caller who explicitly supplies an unsupported baseline or dose grid gets a clear error unless `allow_extrapolation=True` is set.

For example, if training doses run from 5 to 20 RMB and a caller asks about 0 to 30 RMB, default fitting and decision support stay within 5 to 20 RMB. The model cannot establish outcomes for the unobserved no-treatment arm at 0 from those rows. Explicit extrapolation adds the requested doses but marks `extrapolation_enabled`, `unsupported_doses`, and a warning in prediction metadata. It does not create empirical overlap. `build_continuous_policy()` filters dose candidates to observed support by default, refuses positive treatment recommendations when the no-treatment baseline is unsupported, and can additionally filter each user's candidates using the local empirical treatment support diagnostic.

`diagnose_continuous_treatment()` reports observed dose quantiles, histogram bins, and dose coverage by selected feature segments. `local_treatment_support()` reports a standardized-feature kNN dose interval and support score per prediction row. These are empirical coverage summaries, not positivity proofs or guarantees of causal identification.

## Next Research Stage

DRNet and VCNet primarily estimate `E[Y | X,T]`; GIKS augments their counterfactual supervision with gradient interpolation and GP smoothing. Stronger continuous-treatment causal identification remains future work: generalized propensity scores (GPS), Continuous DR, and VCNet-TR. They are not implemented or claimed by this release.

Continuous benchmark runs write configuration, dataset provenance, per-model curves and policies, metrics, environment versions, and a Markdown report to `runs/continuous/<run_id>/`. The runner is offline by default. Real raw-data files must already be available locally. Install `deepuplift[benchmark]` to sample per-model peak process RSS; without it, the report records the process high-water RSS method.

Run the offline synthetic matrix with:

```bash
python scripts/run_continuous_benchmark.py --rows 600 --seed 42
```

The default matrix does not include CCPFN, so it never downloads weights implicitly. Artifacts are written under `runs/continuous/<run_id>/`, which is ignored by Git.
