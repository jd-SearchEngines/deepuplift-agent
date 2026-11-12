# Push offline policy evaluation

**Status:** `PASS` for per-decision/trajectory IPS and cluster bootstrap API;
full sequential DR is `NOT_AVAILABLE` until a continuation-value nuisance model
is supplied.

The public entry points are:

- `evaluate_ope_policy()` for single-decision or row-level OPE;
- `evaluate_sequential_ope_policy()` for repeated decisions with complete
  participant trajectories.

The sequential evaluator reports:

- one-step contextual IPS/SNIPS;
- cumulative per-decision IPS/SNIPS using `W_t = product_{s<=t} rho_s`;
- trajectory IPS and SNIPS;
- per-decision DR when direct-method predictions are supplied;
- step, cumulative, and trajectory weight maximum/variance/ESS diagnostics;
- cluster bootstrap intervals.

It rejects duplicate `(cluster_id, decision_time)` rows and sorts each
trajectory by decision time. Logged propensities must be
`P(observed logged action | history)`; raw invalid values fail before clipping.
A stochastic target policy may pass `target_logged_action_probability` for the
probability assigned to each logged action. The evaluator does not label a
one-step DR correction as a full sequential DR estimator. A continuation-value
model is required before that estimator can be enabled.

## Synthetic sequential smoke

Configuration: fatigue DGP, 100 users, 8 decision points, seed `20261113`;
logged policy evaluated against the fatigue-aware target policy; 200 cluster
bootstrap repetitions. Report stored outside Git at
`/tmp/deepuplift-sequential-ope-smoke.json`.

| Metric | Value |
|---|---:|
| True target mean trajectory value | 3.725000 |
| Per-decision IPS | 0.977640 |
| Trajectory IPS | 543.484981 |
| Trajectory SNIPS | 3.702522 |
| Cluster count | 100 |
| Maximum trajectory importance weight | 14,678.778144 |

The trajectory IPS point estimate is visibly unstable because cumulative
importance weights explode. Trajectory SNIPS is closer to the generated truth
in this smoke, but it is not unbiased by construction and cannot be promoted
without multi-seed bias/coverage analysis. These are semi-synthetic values,
not real MRT policy value or online business lift.

## Five-seed support stress

Configuration: 100 users, seeds `11,22,33,44,55`, fatigue DGP, horizons 8 and
16, 200 cluster bootstrap repetitions. Full JSON:
`/tmp/deepuplift-sequential-ope-5seed-v2.json`.

| Horizon | Non-zero trajectory SNIPS seeds | Mean SNIPS bias on finite seeds | Trajectory ESS range |
|---:|---:|---:|---:|
| 8 | 4/5 | 0.023671 | 0–2 |
| 16 | 0/5 | NOT_AVAILABLE | 0–0 |

The horizon-16 result is an explicit support failure, not a successful policy
evaluation. Per-decision estimators remain the appropriate next comparison;
full-trajectory IS must be reported as unsupported at this overlap level.

Next gate: evaluate five seeds and positivity-stress scenarios, then add a
continuation-value contract for sequential DR only where its assumptions and
diagnostics are satisfied.
