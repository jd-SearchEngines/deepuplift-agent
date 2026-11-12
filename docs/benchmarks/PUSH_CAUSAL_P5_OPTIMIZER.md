# Push-derived P5 constrained optimizer benchmark

**Phase:** P5 — Lagrangian and scale-aware allocation solvers
**Evidence class:** `SYNTHETIC_SCALE_STRESS`
**Status:** `PASS` for exact small reference and 1K/10K/100K sparse stress runs

The existing Top-K budget simulator remains unchanged. The new research API uses sparse candidate rows with identical values, costs, users, slots, actions, eligibility, and constraints for every solver:

- global budget;
- per-slot capacity;
- per-user frequency cap;
- minimum interval;
- action quota;
- at most one action per user-time.

The inspectable reduced utility is `incremental_value - resource_cost - lambda_budget*resource_cost - lambda_slot*slot_usage - lambda_action*action_usage`. The named primal-dual variants are transparent engineering implementations inspired by the algorithm families; they are not claims of reproducing proprietary implementations.

## Exact small instance

The 18-candidate enumeration reference produced net objective `4.93754`. All solvers passed an independent feasibility check:

| Solver | Net objective | Gap to exact | Feasible |
|---|---:|---:|---|
| Exact enumeration | 4.93754 | 0.00000 | yes |
| Greedy Top-K | 4.82454 | 0.11301 | yes |
| Primal-dual | 0.00000 | 4.93754 | yes |
| Line-search primal-dual | 2.06259 | 2.87496 | yes |
| Heap | 4.82454 | 0.11301 | yes |
| Coordinate iterator | 3.07654 | 1.86101 | yes |

The gap table is intentionally not promoted as a quality win for primal-dual. It shows the current baseline behavior and gives the next optimization target.

## Sparse scale results

| Candidates | Candidate-array memory | Peak RSS MB | Greedy s | Primal-dual s | Line-search s | Heap s | Coordinate s | All feasible |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1,000 | 64 KB | 81.09 | 0.0037 | 0.0041 | 0.0170 | 0.0040 | 0.0109 | yes |
| 10,000 | 640 KB | 82.64 | 0.1552 | 0.0466 | 0.1924 | 0.1610 | 0.3577 | yes |
| 100,000 | 6.4 MB | 117.20 | 12.8147 | 1.2351 | 3.8101 | 12.9816 | 17.4252 | yes |

The 100K run took 56.01 s wall time for the full 1K/10K/100K suite. A 1M run was not started because the current 100K profile shows that further profiling and vectorized state updates are needed before claiming 1M readiness; 10M remains out of scope until sparse memory and runtime gates pass. No dense tensor was allocated.

`P5 PASS` means exact small-instance comparison, independent feasibility checks, common candidate/objective definitions, dual traces, runtime, memory, and scale evidence exist. It does not mean the current primal-dual variants dominate greedy or heap.
