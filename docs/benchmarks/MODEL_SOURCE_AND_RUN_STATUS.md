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
| CFRNet | representation/IPM balance | **NOT_RUN** | existing experimental adapter; no real NHEFS run in this phase |
| EFIN | treatment-aware interaction and intervention constraint | **NOT_RUN** | no faithful real-data adapter in this phase |
| DESCN | propensity/response/pseudo-effect heads and consistency constraints | **NOT_RUN** | no faithful real-data adapter in this phase |
| MEMENTO / HydraNet / M3TN / MTMT | frontier multi-treatment architectures | **NOT_RUN** | required treatment contract/data not available |
| CRNet/WDBN / HNet / GIKS real lane | continuous/frontier candidates | **NOT_RUN** | NHEFS truth and verified adapter gate not met |

The real-data PASS labels above mean “the declared computation ran and emitted
finite auditable diagnostics.” They do not mean causal validity, model
superiority or business value. The authoritative aggregate artifact is
[`bias_diagnostics.json`](bias_diagnostics.json).
