# Observational pipeline

`CausalDataset(assignment_type="observational")` uses a deterministic data-layer gate:

```text
CausalDataset -> diagnostics -> unified nuisance -> overlap/ESS/weighting -> DR/R/IPW -> evaluation -> policy -> evidence
```

The default nuisance path is K-fold cross-fitting. Propensity and outcome nuisance estimates used by DR/R/IPW are out-of-fold; each learner can receive `fit(dataset, nuisance=result)`. A full propensity model is retained only to score new rows and is separately recorded as prediction provenance.

Readiness is `READY`, `REVIEW`, or `NOT_RECOMMENDED` based on overlap, extreme propensity, balance, leakage warnings, sample size, and ESS. It is not a test for hidden confounding. All observational causal estimates depend on conditional ignorability/unconfoundedness and positivity assumptions.
