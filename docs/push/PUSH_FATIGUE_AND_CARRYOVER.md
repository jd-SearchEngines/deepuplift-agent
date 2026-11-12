# Push fatigue and carryover laboratory

**Status:** `PASS` for the semi-synthetic laboratory implementation; `NOT_AVAILABLE`
for real MRT long-term causal effects.

`deepuplift.benchmarks.push_longitudinal` samples pre-treatment context from a
local MRT source and simulates a known-truth sequential world. It records
potential outcomes, assignment probability, availability, prior push count,
time since last push, fatigue state, carryover state, and policy value. The
world is binary push versus no-push and is separate from the real randomized
MRT proximal ITT analysis.

Implemented scenarios:

- no fatigue / no carryover
- homogeneous and heterogeneous fatigue
- cumulative, rolling-window, and decayed fatigue state
- spacing-dependent effects
- carryover
- temporal drift
- negative-effect users
- positivity failure

## Reproducible smoke evidence

Run configuration: 100 synthetic users, 8 decision points per user, seed
`20261113`, policies `logged`, `no_push`, `random_push`, `rule_based`,
`fatigue_aware`, and `oracle`. Full JSON was written outside the repository to
`/tmp/deepuplift-longitudinal-smoke-v2.json`.

| Scenario | Policy | Cumulative value | Incremental vs no-push | Pushes | Negative-effect pushes |
|---|---:|---:|---:|---:|---:|
| fatigue | logged | 348.721117 | 24.721117 | 199 | 0 |
| fatigue | fatigue-aware | 352.500000 | 28.500000 | 200 | 0 |
| heterogeneous fatigue | logged | 345.662757 | 21.662757 | 205 | 11 |
| heterogeneous fatigue | fatigue-aware | 350.000000 | 26.000000 | 200 | 0 |
| spacing | logged | 355.948160 | 31.948160 | 217 | 2 |
| spacing | fatigue-aware | 361.500000 | 37.500000 | 200 | 0 |
| carryover | logged | 368.083180 | 44.083180 | 192 | 0 |
| carryover | fatigue-aware | 372.500000 | 48.500000 | 200 | 0 |

The values above are generated-truth policy values, not observed MRT outcomes,
not OPE estimates, and not online business lift. The `oracle` policy is an
upper-bound myopic diagnostic and is not a deployable policy. The longitudinal
module also exposes `dynamic_oracle`, which searches the full finite-horizon
action sequence under the generated state transition; it is a truth-only upper
bound, not a learned policy.

## Remaining research gate

The current laboratory validates state transitions, scenario isolation, and
policy comparison. It does not yet establish a sequential DR estimator or a
production candidate builder. The next gate is to connect the canonical
decision-point frame to a learned proximal effect surface, then compare
replay/OPE estimates with the known cumulative truth under multiple seeds and
positivity stress.
