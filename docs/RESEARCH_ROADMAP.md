# DeepUplift research roadmap

This roadmap is an execution record, not a promise that an unrun method is supported.

## Current status

| Workstream | Status | Evidence |
|---|---|---|
| P0 Push MRT audit | PASS | `docs/benchmarks/PUSH_CAUSAL_P0_AUDIT.md` |
| P1 randomized Push baseline | PASS | `docs/benchmarks/PUSH_CAUSAL_P1_BASELINE.md` |
| P2 semi-synthetic bias/overlap | PASS | `docs/benchmarks/PUSH_CAUSAL_P2_SEMISYNTHETIC.md` |
| P3 multi-treatment/continuous | PASS | `docs/benchmarks/PUSH_CAUSAL_P3_CONTINUOUS.md` |
| P4 calibration/monotonicity | PASS | `docs/benchmarks/PUSH_CAUSAL_P4_CALIBRATION.md` |
| P5 optimizer V1.1 | PASS feasibility; quality/convergence separated | `docs/benchmarks/PUSH_CAUSAL_P5_OPTIMIZER.md` |
| Public dataset registry | PASS metadata smoke | `examples/datasets/manifest.json`, `scripts/build_dataset_registry.py` |
| Cluster-aware OPE | PASS reusable API smoke | `deepuplift/core/ope.py`, `tests/decision/test_ope.py` |

## Next executable sequence

1. Complete GitHub CI artifacts for the five-seed optimizer workflow.
2. Add a unified benchmark matrix linking dataset ground-truth status to allowed metrics; never emit PEHE for RCT/observational data without counterfactual truth.
3. Run a multi-seed semi-synthetic identification matrix: naive, outcome regression, IPW, overlap weighting, DR, and cross-fitting under measured confounding, SSB, and overlap failure.
4. Extend the policy layer with OPE confidence gates and policy-value/regret comparisons on held-out data.
5. Optimize Coordinate Iterator and implement a valid Lagrangian upper bound before reporting primal-dual gaps.
6. Expand model comparisons only after baseline and failure-mode evidence are stable.

## Execution rules

- Real data, randomized data, semi-synthetic truth, and observational stress data remain separate evidence classes.
- `PASS`, `FAIL`, `NOT_RUN`, `BLOCKED`, and `NOT_AVAILABLE` are not interchangeable.
- Raw data and model weights stay outside Git; registry entries record local preparation requirements and license status.
- Offline policy value is not online business lift.
