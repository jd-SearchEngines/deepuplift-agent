# DeepUplift Agent Offline-To-Online Validation Playbook

Offline uplift metrics are decision support, not the final proof of business impact. This playbook explains how to move from model comparison and policy value simulation to an online experiment.

## Why Offline Is Not Enough

QINI, AUUC, Top-K uplift, calibration, bootstrap CI, and policy value help select a promising policy, but they still depend on historical data quality and causal assumptions. Real business impact should be validated with an online experiment whenever possible.

## Offline Readiness Gates

Before online testing:

| Gate | Requirement | Why |
| --- | --- | --- |
| Treatment design | treatment/control definition is stable | Prevents policy from learning a moving target |
| Overlap | weak-overlap rate is low or risk accepted | Ensures comparable treatment/control support |
| Selection bias | SSB proxy is not high, or robust learner is chosen | Reduces confounding risk |
| Model ranking | QINI/AUUC and Top-K are positive enough | Confirms useful uplift ordering |
| Uncertainty | bootstrap lower bound is acceptable | Avoids deploying noise |
| Calibration | uplift score buckets are directionally monotonic | Makes Top-K thresholds more reliable |
| Business value | policy value is positive after cost | Aligns offline metric with business outcome |
| Evidence | run artifacts and model card are complete | Keeps the decision auditable |

## Online Experiment Design

Recommended A/B design:

1. Choose a policy threshold, for example Top 10% or Top 20% by uplift score.
2. Create an eligible population that passes business constraints and exclusion rules.
3. Randomize eligible users into:
   - treatment arm: receive the campaign/action
   - holdout arm: do not receive the action
4. Keep the score threshold fixed during the experiment window.
5. Measure primary business outcome and guardrail metrics.
6. Estimate incremental lift, confidence interval, and ROI.

## Metrics To Report

| Metric | Meaning |
| --- | --- |
| Incremental conversion | treatment conversion minus holdout conversion inside targeted population |
| Incremental revenue | incremental conversion times business value |
| Contact cost | targeted users times unit treatment cost |
| Net value | incremental revenue minus contact cost |
| ROI | net value divided by contact cost |
| Reach | share of eligible population targeted |
| Guardrails | unsubscribe, complaint, churn, fatigue, fairness or budget metrics |

## Common Pitfalls

- Changing model thresholds mid-experiment.
- Comparing targeted treatment users against non-random historical controls.
- Reusing users across overlapping campaigns without conflict rules.
- Optimizing QINI while ignoring cost and user fatigue.
- Reporting average conversion instead of incremental conversion.
- Ignoring negative-uplift segments.

## Interview Talk Track

I treat offline policy value as a pre-launch decision estimate. The production decision still needs an online holdout experiment. The platform helps choose the threshold, estimate expected value, export the audience, and preserve the evidence needed to compare offline expectation with online incremental impact.
