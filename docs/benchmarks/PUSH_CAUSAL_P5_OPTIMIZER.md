# Push-derived P5 constrained optimizer benchmark

**Phase:** P5 — Lagrangian and scale-aware allocation solvers
**Evidence class:** `SYNTHETIC_SCALE_STRESS`
**V1.1 status:** `PASS` for feasibility and benchmark execution; solution quality is exact on the small reference and measured against the best feasible heuristic at large scale; convergence is reported separately and is not universally achieved.

This report covers the V1.1 remediation. It makes no business-uplift, causal-validity, or production-readiness claim.

## 1. Contracts and complexity

All solvers use identical sparse candidate arrays and objective:

`sum(incremental_value - resource_cost)`

Constraints are global budget, slot capacity, user frequency, minimum interval, optional action quota, at-most-one action per user-time, and eligibility. `simulate_budget` and existing public APIs remain unchanged.

The independent feasibility checker is O(K log K): selected rows are grouped by user, each user's slots are sorted, and only adjacent intervals are checked. No selected-row pair enumeration remains. Eligibility is cached per candidate. Candidate arrays are validated for one-dimensional equal length, finite values, non-negative costs, and boolean/integer eligibility.

The main V1.1 O(N²) defect was repeated full-array `net_value` computation inside selection and incumbent loops. The affected solvers now cache `net_value` and `resource_cost` once per solve. A regression test uses a property spy to verify Greedy reads `net_value` once.

## 2. Separate acceptance dimensions

Each result now reports:

- `feasibility_status` and actual selected-set `primal_residual`;
- `quality_status`, `relative_optimality_gap`, `relative_heuristic_gap`, and reference type;
- `convergence_status`, `stopping_reason`, `convergence_metrics`;
- `dual_step_norm` for price-change norm;
- `dual_feasibility_residual` and `dual_feasibility` for non-negative prices;
- `complementarity`, signed residuals, incumbent fallback, and `primal_dual_gap_valid`.

The historical `dual_residual` key is retained for compatibility but now means dual-feasibility violation. It no longer stores price-change norm; that value is `dual_step_norm`. `max_iterations` and `line_search_no_improvement` are `NOT_CONVERGED`, never convergence PASS.

For large instances, no exact upper bound is available. Therefore `relative_heuristic_gap` is reported against the best feasible heuristic in that run, while `relative_optimality_gap` and `primal_dual_gap` remain unavailable. This distinction is encoded in the machine-readable JSON.

## 3. Exact reference

The exact small reference now prefers `scipy.optimize.milp`; if SciPy MILP is unavailable or fails, it falls back to exhaustive enumeration. Seed 42, 18 candidates, returned method `scipy_milp`, objective `4.9375443709`, and all five heuristics matched it exactly.

| Solver | Objective | Relative optimality gap | Feasibility | Convergence |
|---|---:|---:|---|---|
| Exact MILP | 4.937544 | 0 | PASS | N/A |
| Greedy | 4.937544 | 0 | PASS | N/A |
| Primal-Dual | 4.937544 | 0 | PASS | PASS |
| Line Search | 4.937544 | 0 | PASS | NOT_CONVERGED (`line_search_no_improvement`) |
| Heap | 4.937544 | 0 | PASS | N/A |
| Coordinate Iterator | 4.937544 | 0 | PASS | PASS |

## 4. V1.1 scale results

All rows below used isolated solver processes. Actual selected-set feasibility was PASS with max constraint violation 0. RSS is process-local peak RSS. Gaps are relative to the best feasible heuristic at that scale, not an exact optimality claim.

| Scale | Solver | Objective | Relative heuristic gap | Time | Peak RSS | Convergence |
|---:|---|---:|---:|---:|---:|---|
| 100K* | Greedy | 36,316.6113 | 0.01396% | 0.164s | 79.59MB | N/A |
| 100K* | Primal-Dual | 36,321.6825 | 0 | 5.948s | 107.92MB | NOT_CONVERGED / max_iterations |
| 100K* | Line Search | 36,316.6113 | 0.01396% | 1.580s | 118.59MB | NOT_CONVERGED |
| 100K* | Heap | 36,316.6113 | 0.01396% | 0.216s | 44.58MB | N/A |
| 100K* | Coordinate | 36,316.6113 | 0.01396% | 22.948s | 146.50MB | NOT_CONVERGED / max_iterations |
| 500K | Greedy | 181,154.4305 | 0.01820% | 0.876s | 135.52MB | N/A |
| 500K | Primal-Dual | 181,187.4076 | 0 | 35.670s | 250.11MB | NOT_CONVERGED / max_iterations |
| 500K | Line Search | 181,154.4305 | 0.01820% | 10.660s | 300.25MB | NOT_CONVERGED |
| 500K | Heap | 181,154.4305 | 0.01820% | 2.154s | 231.25MB | N/A |
| 500K | Coordinate | 181,154.4305 | 0.01820% | 81.367s | 360.23MB | NOT_CONVERGED / max_iterations |
| 1M | Greedy | 362,688.7656 | 0.01640% | 1.943s | 295.55MB | N/A |
| 1M | Primal-Dual | 362,748.2474 | 0 | 75.658s | 504.55MB | NOT_CONVERGED / max_iterations |
| 1M | Line Search | 362,688.7656 | 0.01640% | 20.779s | 623.47MB | NOT_CONVERGED |
| 1M | Heap | 362,688.7656 | 0.01640% | 4.621s | 476.20MB | N/A |
| 1M | Coordinate | 362,688.7656 | 0.01640% | 185.404s | 714.48MB | NOT_CONVERGED / coordinate budget |

`*` 100K row is seed 1 from the five-seed run. The 500K/1M rows are seed 42. Candidate-array memory was 6.4MB, 32.0MB, and 64.0MB respectively. Resource guard passed for all requested scales.

## 5. Five-seed stability run

Seeds 1–5 were run at 1K, 10K, and 100K. All 15 scale runs and all 75 Solver rows were feasible. Exact-reference quality was PASS for every seed. At 100K, Primal-Dual had the best heuristic objective for all five seeds; the non-best heuristic gap range was `0.01267%–0.02148%`. Convergence was not universal: 100K Primal-Dual and Coordinate runs used the scale iteration budget, and Line Search frequently stopped with `line_search_no_improvement`.

The JSON was generated by:

```bash
python scripts/run_push_optimizer_benchmark.py \
  --seed 1 --seed 2 --seed 3 --seed 4 --seed 5 \
  --scale 1000 --scale 10000 --scale 100000 --isolated \
  --output optimizer-seeds1-5-small-scales.json
```

Machine-specific JSON is intentionally not committed. GitHub Actions uploads the JSON artifact for each scale.

## 6. Historical pre-V1.1 measurements

The following numbers are retained for comparison and are not overwritten. They were produced before the repeated `net_value` fix and before the separated convergence semantics:

| Scale | Greedy (s) | Primal-Dual (s) | Line Search (s) | Heap (s) | Coordinate (s) |
|---:|---:|---:|---:|---:|---:|
| 1K | 0.0037 | 0.0041 | 0.0170 | 0.0040 | 0.0109 |
| 10K | 0.1552 | 0.0466 | 0.1924 | 0.1610 | 0.3577 |
| 100K | 12.8147 | 1.2351 | 3.8101 | 12.9816 | 17.4252 |
| 500K | not run | not run | not run | not run | not run |
| 1M | not run | not run | not run | not run | not run |

For the comparable Greedy path, V1.1 is approximately 78x faster at 100K, 47x faster at 500K, and 116x faster at 1M. Primal-Dual is approximately 2.1x faster at 500K and 3.9x faster at 1M under the recorded iteration budgets. These are local benchmark comparisons, not CI measurements.

## 7. Remaining limitations

Heap is a static net-order baseline, not a dynamic incremental optimizer. Coordinate remains O(I·N log N) because each coordinate recomputes a full utility order; it is no longer using pairwise O(N²) feasibility checks, but its 1M runtime remains the main follow-up target. Large-scale convergence requires a principled dual upper-bound/duality-gap implementation rather than simply increasing iteration limits.
