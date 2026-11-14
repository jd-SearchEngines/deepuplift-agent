# Experiment A — Model Source and Run Status

This register separates what is implemented in the repository from what was
actually run in the real-data bias-repair phase. A source citation or a module
name is not treated as evidence of a faithful paper reproduction.

| Family | Internal mechanism / source boundary | Real NHEFS status | Evidence boundary |
|---|---|---|---|
| Propensity logistic + cross-fit | binary `P(T=1|X)` with out-of-fold predictions | **PASS** | cross-fitted nuisance and diagnostics; convergence warnings recorded |
| IPW / stabilized IPW / overlap | explicit weighting in existing nuisance layer plus new ablation reporting | **PASS** | binary observational repair only |
| Matching | logit-propensity NN with replacement/caliper | **PASS** | ATT-like matched estimand; not ATE |
| AIPW / DR | cross-fitted outcome nuisances plus augmented pseudo-outcome | **PASS** | identification assumptions remain untestable |
| GPS / density ratio | conditional dose mean + Gaussian residual density / marginal KDE | **PASS** | working continuous GPS; extreme tail warning |
| DoseResponseGBM | native dose-conditioned GBM | **PASS** | factual holdout only; causal truth unavailable |
| DRNet | native dose-stratified PyTorch network; source metadata points to VCNet project; no code copied | **PASS** | 8-epoch NHEFS factual smoke; not faithful full paper reproduction |
| VCNet | native varying-coefficient Bernstein basis; VCNet-TR not included | **PASS** | 8-epoch NHEFS factual smoke; not VCNet-TR evidence |
| TarNet | shared representation + treatment-specific outcome heads | **PASS** | 15 known-truth synthetic rows across 3 sizes × 5 seeds; MPS; architecture-inspired adapter |
| CFRNet | representation/IPM balance | **PASS / PARTIAL** | balance=0/1 ablation ran on known-truth synthetic data; official TensorFlow source not run; no real NHEFS neural run |
| DragonNet | propensity head + optional targeted regularization | **PASS / PARTIAL** | targeted on/off ablation ran on known-truth synthetic data; official TensorFlow source not run; no real NHEFS neural run |
| EFIN | treatment-aware interaction and intervention constraint | **NOT_RUN** | no faithful real-data adapter in this phase |
| DESCN | propensity/response/pseudo-effect heads and consistency constraints | **NOT_RUN** | no faithful real-data adapter in this phase |
| MEMENTO / HydraNet / M3TN / MTMT | frontier multi-treatment architectures | **NOT_RUN** | required treatment contract/data not available |
| CRNet/WDBN / HNet / GIKS real lane | continuous/frontier candidates | **NOT_RUN** | NHEFS truth and verified adapter gate not met |

The new neural PASS labels mean “the declared architecture-inspired adapter
ran and emitted finite known-truth diagnostics”; they are not official source
reproductions. The real-data PASS labels above mean “the declared computation ran and emitted
finite auditable diagnostics.” They do not mean causal validity, model
superiority or business value. The authoritative aggregate artifact is
[`bias_diagnostics.json`](bias_diagnostics.json).

## Hillstrom randomized execution update (2026-11-14)

The local real-data runner now executes Hillstrom pairwise baselines for
conversion, visit and spend. All three outcomes completed 50/50 PASS rows
(five seeds × two active-arm comparisons × five learners), using the known
randomized probability rather than an estimated propensity model. These are
`REAL_RANDOMIZED_EXTERNAL` results: ITT, Qini/AUUC, calibration and design-
based policy value are reported, while PEHE and individual-effect bias remain
`NOT_AVAILABLE`. See [EXPERIMENT_A_REPRODUCTION_LOG.md](EXPERIMENT_A_REPRODUCTION_LOG.md)
for the source hash, commands, model table and warnings.
