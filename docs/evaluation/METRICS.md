# Evaluation metrics

Known-effect datasets report PEHE, ATE bias, CATE correlation, and top-10% true effect. Randomized datasets report Qini/AUUC and uplift at target fractions. Calibration reports bucket predicted versus observed incremental outcome and MAE. Policy reports targets, incremental outcome/value, cost, net value, iROAS, and budget utilization. These are distinct views: strong ranking does not imply calibration or online lift.

## V1.1 benchmark definitions

`ranking_metrics` now reports two related but distinct areas:

- `auuc`: area under the cumulative uplift-gain curve
  `prefix_count * (mean_treatment - mean_control)` over the ranked support
  where both treatment arms are observed; prefixes with a missing arm are not
  assigned a zero uplift. If fewer than two ranked support points remain, AUUC
  is reported as unavailable rather than as a fabricated zero.
- `qini`: area under the cumulative Qini gain curve minus the area of the
  straight random-targeting baseline joining zero to the observed total gain.
- Both areas are restricted to the ranked fraction where both treatment arms
  have been observed; the machine-readable result exposes that support start.
- `qini_raw_area` and `random_baseline_area` are retained so the decomposition
  is auditable rather than treating Qini and AUUC as aliases.

The ranking bootstrap is an original-row bootstrap. Its seed, requested count,
valid count, and percentile 95% intervals are stored in
`ranking.bootstrap_stability`. Bootstrap uncertainty does not correct design or
confounding bias.

`fixed_budget_policy_value` implements a fixed-budget Top-K policy. It selects
the stable descending-score prefix that fits `floor(budget / treatment_cost)`
and estimates the policy value using known or estimated binary propensities:

```text
V(pi) = mean[ pi(X) * T*Y/e(X) + (1-pi(X)) * (1-T)*Y/(1-e(X)) ]
```

The reported interval bootstraps the original per-row IPW contributions. This
is suitable for a binary randomized or explicitly modeled assignment lane; it
is not a claim of sequential MRT value or identification under hidden
confounding.

`bootstrap_cluster_ci` resamples independent participant/unit IDs and keeps
all repeated decision rows from a selected cluster together. It is the
uncertainty primitive for repeated MRT rows; row bootstrap remains available
for independent-row fixtures. The report records the bootstrap unit and
cluster count.

`multiarm_policy_value` accepts an `(n_rows, n_arms)` matrix of logged
assignment probabilities. It estimates `I(A=d(X))Y/p(A=d(X)|X)` and subtracts
the cost of the selected arm. It rejects non-positive/non-normalized
probabilities and does not infer assignment probabilities from observed arm
counts. Pass participant IDs through `clusters=` for cluster bootstrap.

Continuous benchmarks report both `response_mise` and `effect_mise`:

- `response_mise` integrates squared error between predicted and true potential
  response curves.
- `effect_mise` integrates squared error between predicted and true incremental
  effect curves.

The historical `mise` field remains as an alias for `response_mise` for
backward compatibility. `effect_mise` is only available when an auxiliary
known-truth effect curve exists; it is not fabricated for real observational
data.
