# Push candidate builder and constrained optimization

**Status:** `PASS` for the semi-synthetic model-to-optimizer smoke.

`build_push_candidate_bundle()` consumes fitted effect predictions and creates
sparse candidates with user, synthetic decision slot, action, eligibility,
predicted incremental outcome, utility mapping, resource cost, and model
provenance. `optimize_push_candidate_bundle()` reuses the existing Greedy,
Heap, Primal-Dual, Line Search, Coordinate Iterator, and Exact Reference
contracts and independently checks every selected set.

The current real MRT source does not provide monetary Push cost. Therefore the
default resource unit is `push_capacity_units`, not currency. If predicted
effects are in outcome units, a `value_multiplier` must be explicitly declared
before subtracting capacity cost in the existing net-value objective. No such
mapping is a claim about revenue or profit.

## Measured model-to-optimizer smoke

Configuration: fatigue DGP, 100 users, 8 synthetic slots, seed `20261113`, 800
candidates, participant-group 5-fold OOF WCLS-style model, capacity budget 80,
per-user frequency cap 2, minimum interval 2, `value_multiplier=10`.
Report: `/tmp/deepuplift-model-optimizer-smoke-v2.json`.

| Solver | Selected | Objective net value | Independent feasibility |
|---|---:|---:|---|
| Greedy | 80 | 61.170575 | PASS |
| Heap | 80 | 61.170575 | PASS |
| Primal-Dual | 80 | 61.170575 | PASS |
| Line Search | 80 | 61.170575 | PASS |
| Coordinate | 80 | 61.170575 | PASS |

The candidate provenance records `model_version=wcls_group_oof_v1`; values are
not independently randomized. With the same model predictions and default
unit capacity cost but no value mapping, all predicted effects are below cost
and the optimizer correctly selects zero candidates. That is a unit mismatch
warning, not a reason to lower constraints.

The one-command entry point is:

```bash
python scripts/run_push_end_to_end.py /path/to/Data_MRT.zip \
  --users 100 --horizon 8 --seed 20261113 \
  --budget 80 --value-multiplier 10 \
  --output /tmp/deepuplift-model-optimizer.json
```
