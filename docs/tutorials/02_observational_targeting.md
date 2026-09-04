# Observational Targeting

When historical campaigns were not randomized, the pipeline automatically
estimates nuisance quantities. You can make the protocol explicit:

```python
from deepuplift.application import run_uplift_pipeline

result = run_uplift_pipeline(
    data=df,
    feature_cols=["recency_days", "orders_30d", "spend_90d"],
    treatment_col="campaign_received",
    outcome_col="conversion",
    assignment_type="observational",
    model_names=["IPW-Learner", "DR-Learner", "R-Learner"],
    nuisance_config={
        "estimator": "logistic", "cross_fit": True, "n_splits": 5,
        "weighting": "overlap", "trim_threshold": 0.05,
    },
)
```

Review propensity distributions, common support, trimming, effective sample
size (ESS), and balance before/after weighting. Cross-fitting reduces reuse of
the same rows for nuisance prediction, but these diagnostics cannot prove
absence of hidden confounding. A causal claim still needs a defensible
assignment story and sensitivity analysis.

```bash
python examples/observational_targeting/run.py
```
