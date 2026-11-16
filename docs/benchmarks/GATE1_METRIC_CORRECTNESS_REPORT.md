# Gate 1 Metric Correctness Report

Date: 2026-11-16  
Branch: `codex/batch1-metric-correctness`  
Implementation commit: `25a93b8`

## Gate status

**PASS for deterministic metric correctness; independent SDK review: NOT_COMPLETED due to timeout.** The deterministic gate is sufficient for this bounded implementation task, but the review timeout is retained and must not be represented as a reviewer PASS.

Evidence class: hand-calculated fixture plus synthetic repeated-unit fixture. No raw MRT rows or business result are used.

## Changes

- Added a hand-checkable Qini/AUUC fixture with explicit area convention.
- Added `bootstrap_cluster_ci`, which resamples participant IDs and retains all rows from sampled clusters.
- Added `multiarm_policy_value_contributions` and `multiarm_policy_value` using the logged per-row arm probability matrix.
- Added strict checks for arm indices, positive normalized propensities, dimensions, finite values and cost vectors.
- Added cluster-vs-row bootstrap metadata and documentation.

## Hand-calculated fixture

Rows are ranked by uplift with treatment/outcome `(1,4), (0,1), (1,3), (0,2)`.

| ranked fraction | Qini gain | uplift gain used by AUUC |
|---:|---:|---:|
| 0.50 | 3.0 | 6.0 |
| 0.75 | 5.0 | 7.5 |
| 1.00 | 4.0 | 8.0 |

The implementation returns `qini_raw_area=2.125`, `random_baseline_area=1.5`, `qini=0.625`, and `auuc=3.625`. These values are asserted numerically in the test suite.

## Cluster bootstrap

The repeated-row fixture contains two participants with two rows each. The cluster bootstrap returns estimate `6.5`, `cluster_count=2`, and 32 deterministic draws at seed 11. Rows are sampled through participant indices, not independently, so within-participant dependence is retained.

## Multi-arm policy value

The four-row, three-arm fixture passes the actual `(rows, arms)` assignment probability matrix. With arm costs `[0,1,2]`, the IPW policy-value estimate is `8.3333333333`. The cluster-bootstrap result records `bootstrap_unit=cluster` and `cluster_count=2`. A non-normalized propensity matrix is rejected.

## Tests

- Focused Gate 1 tests: `10 passed`.
- Full project suite, Python 3.11, serial BLAS: `114 passed, 2 skipped, 432 warnings`.
- `git diff --check` and Python bytecode compilation: PASS.

## Review boundary

An independent official SDK read-only reviewer was started in a separate thread but exceeded the local execution timeout and was terminated. No reviewer verdict was used as evidence. The remaining validator evidence is deterministic pytest plus the exact hand-calculated values above. A future review may re-run after the branch is pushed.

## Limitations and next action

This closes the metric-layer correctness gaps for the implemented estimators. It does not reconcile myBPmyLife provenance, establish real MRT policy value, or prove causal identification. Next eligible task is Gate 2 data provenance reconciliation; do not treat these synthetic metrics as business lift.
