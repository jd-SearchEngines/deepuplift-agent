# Push-derived P5 constrained optimizer benchmark

**Phase:** P5 — Lagrangian and scale-aware allocation solvers
**Evidence class:** `SYNTHETIC_SCALE_STRESS`
**Status:** `PASS` for correctness gates and the requested 100K/500K/1M runs; Coordinate Iterator remains a deliberately slower research baseline.

This report covers the remediation specified in `P5_OPTIMIZER_REMEDIATION_PROMPT.md`. It does not claim business uplift, causal validity on the real Push data, or production readiness.

## Scope and objective

All solvers use the same sparse candidate arrays, objective, seed, and constraints:

`sum(incremental_value - resource_cost)`

The constraints are global budget, per-slot capacity, per-user frequency, minimum interval, optional action quota, at-most-one action per user-time, and eligibility. `simulate_budget` and existing public APIs remain unchanged.

The independent feasibility checker is O(K log K): it groups selected rows by user, sorts each user's time slots, and checks adjacent intervals. It does not enumerate selected-row pairs. Per-index eligibility checks use a cached boolean mask. Candidate construction validates one-dimensional equal-length finite arrays, non-negative costs, and boolean/integer eligibility.

## Exact small-instance acceptance

Seed 42, 18 candidates, exhaustive enumeration. The corrected solvers all returned the exact objective `4.9375443709` and a feasible selected set.

| Solver | Net objective | Gap to exact | Feasible | Stopping / mode |
|---|---:|---:|---|---|
| Exact enumeration | 4.937544 | 0.000000 | yes | enumeration complete |
| Greedy Top-K | 4.937544 | 0.000000 | yes | static net order |
| Primal-Dual | 4.937544 | 0.000000 | yes | projected dual tolerance |
| Line Search | 4.937544 | 0.000000 | yes | merit-based line search |
| Heap | 4.937544 | 0.000000 | yes | static net heap |
| Coordinate Iterator | 4.937544 | 0.000000 | yes | sequential coordinate updates |

Five additional random seeds were checked in unit tests; every heuristic output was feasible and no heuristic objective exceeded its exact reference. Ties are deterministic through stable sorting.

## Dual and solver remediation

- Budget, slot, and action residuals are signed `(used - limit) / max(limit, 1)` values.
- Dual ascent is projected: `lambda <- max(0, lambda + step * signed_residual)`. Under-used resources can reduce prices, and prices remain non-negative.
- Results expose actual selected-set `primal_constraint_violation`, relaxed-selection residuals, signed residuals, dual prices, complementarity, dual step, stopping reason, iteration count, and incumbent fallback use.
- Objective accounting applies the budget multiplier to `resource_cost`; the base objective is not double-counted.
- Repair is independently feasibility-preserving and compares against the Greedy incumbent. If a candidate update is worse, the incumbent is retained and recorded.
- Line Search evaluates a merit function containing primal violation, complementarity, and dual-step size; accepted steps do not worsen that merit.
- Coordinate Iterator recomputes selection and residuals after every coordinate. At 100K+ scale, the benchmark budget is two cycles at 100K and one cycle at 500K/1M; this is recorded in each JSON report and is not presented as full convergence.

## Scale results — seed 42

Every row below had selected-set feasibility `yes` and actual max constraint violation `0.0`. Gap is relative to the best feasible heuristic objective within that scale. RSS is peak RSS from the Solver's isolated process, not cumulative parent RSS. Candidate-array memory is the raw sparse array footprint.

| Candidates | Solver | Objective | Relative gap | Time (s) | Peak RSS (MB) | Candidates/s |
|---:|---|---:|---:|---:|---:|---:|
| 100,000 | Greedy | 36,212.6341 | 0.0159% | 1.946 | 29.02 | 51,381 |
| 100,000 | Primal-Dual | 36,218.4055 | 0.0000% | 8.195 | 86.03 | 12,202 |
| 100,000 | Line Search | 36,212.6341 | 0.0159% | 3.567 | 94.58 | 28,039 |
| 100,000 | Heap | 36,212.6341 | 0.0159% | 0.228 | 80.08 | 438,596 |
| 100,000 | Coordinate | 36,212.6341 | 0.0159% | 32.219 | 123.28 | 3,104 |
| 500,000 | Greedy | 181,154.4305 | 0.0182% | 41.510 | 119.94 | 12,045 |
| 500,000 | Primal-Dual | 181,187.4076 | 0.0000% | 74.479 | 226.47 | 6,713 |
| 500,000 | Line Search | 181,154.4305 | 0.0182% | 50.090 | 278.66 | 9,982 |
| 500,000 | Heap | 181,154.4305 | 0.0182% | 1.773 | 213.66 | 282,011 |
| 500,000 | Coordinate | 181,154.4305 | 0.0182% | 120.221 | 333.03 | 4,159 |
| 1,000,000 | Greedy | 362,688.7656 | 0.0164% | 225.466 | 195.25 | 4,435 |
| 1,000,000 | Primal-Dual | 362,748.2474 | 0.0000% | 294.168 | 411.78 | 3,399 |
| 1,000,000 | Line Search | 362,688.7656 | 0.0164% | 230.536 | 518.42 | 4,338 |
| 1,000,000 | Heap | 362,688.7656 | 0.0164% | 5.038 | 385.92 | 198,492 |
| 1,000,000 | Coordinate | 362,688.7656 | 0.0164% | 395.577 | 629.61 | 2,528 |

Raw candidate-array memory was 6.4 MB, 32.0 MB, and 64.0 MB respectively. Resource guard status was `PASS` for all three scales. The 1M run was completed with one Coordinate cycle and a 20-iteration Primal-Dual budget; see the machine-readable report generated by the benchmark script for signed residuals, prices, complementarity, and stopping diagnostics.

## Multi-seed evidence required by the plan

Seeds 1, 2, and 3 were run at 10K and 100K. All 30 Solver-scale rows were feasible. At 100K, the Primal-Dual objective was the best heuristic on all three seeds; the other four methods were between `0.01396%` and `0.02348%` below the best. At 10K, Primal-Dual or Coordinate was best on each seed; the non-best gap ranged from `0.02158%` to `0.03643%`.

## Tests and limitations

The focused optimizer suite passed 8 tests, including exact comparisons, candidate validation, eligibility and zero-budget edges, negative net value, duplicate user-time, interval boundaries, signed price decrease, coordinate recomputation, and a 20K independent checker run. The full repository test result is recorded in the release handoff.

The Heap and Coordinate implementations are static-data-structure and sequential-coordinate baselines respectively. They are useful comparison points but are not claimed to be dynamic sublinear optimizers. Coordinate's repeated full sorting is O(I·N log N), not pairwise O(N²); its 1M runtime is the main follow-up performance target. No raw MRT data, model weights, or generated machine-specific benchmark artifacts are committed.
