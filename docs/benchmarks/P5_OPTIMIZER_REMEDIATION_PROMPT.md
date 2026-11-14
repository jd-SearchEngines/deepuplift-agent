# P5 Optimizer Remediation & Scale Benchmark — Codex Executor Prompt

You are continuing the existing DeepUplift Push Causal Benchmark in this repository.

## Objective

Repair and validate the P5 constrained allocation solvers, then establish reproducible scale evidence at 100K, 500K, and 1M candidate decisions. A 10M run is optional and must be guarded by a memory preflight.

Work on the existing branch:
codex/push-mrt-causal-benchmark

Do not create a second causal platform. Preserve the existing Data → Models → Decision architecture and public APIs unless a compatibility-preserving extension is required.

## Current evidence and known defects

Read these files first:

- deepuplift/decision/optimization/constrained.py
- tests/decision/test_constrained_optimization.py
- scripts/run_push_optimizer_benchmark.py
- docs/benchmarks/PUSH_CAUSAL_P5_OPTIMIZER.md
- docs/benchmarks/PUSH_CAUSAL_BENCHMARK_PLAN.md

Current recorded 18-candidate exact-reference run:

| Solver | Net objective | Gap to exact |
|---|---:|---:|
| Exact enumeration | 4.93754 | 0 |
| Greedy Top-K | 4.82454 | 0.11301 |
| Primal-Dual | 0.00000 | 4.93754 |
| Line-search Primal-Dual | 2.06259 | 2.87496 |
| Heap | 4.82454 | 0.11301 |
| Coordinate iterator | 3.07654 | 1.86101 |

The 100K report shows feasibility but weak objective quality for the dual variants. Do not describe the current implementation as an optimizer quality win.

The following implementation risks have been identified and must be checked against the current code before editing:

1. AllocationCandidates.eligible_mask() creates a full-size all-ones array when eligibility is omitted. Calling it inside per-candidate loops can cause repeated O(N) allocations and effectively O(N²) memory traffic.
2. check_feasibility() checks minimum-interval violations with pairwise combinations of selected rows. This is O(K²) and is not acceptable for 100K–1M selected rows.
3. _dual_violations() currently exposes only positive constraint excess. Dual prices are only increased; they cannot decrease when a relaxed solution underuses a resource. This can lock the solver into excessive prices and poor/empty repaired solutions.
4. The primal-dual stopping rule must not declare convergence solely because positive primal violation is zero. Track the relevant primal feasibility and complementary-slackness / dual residual information.
5. Coordinate updates must be genuinely sequential: after changing a coordinate price, recompute the relaxed selection and residual before updating the next coordinate. Otherwise label the method as a simultaneous update, not coordinate descent.
6. Heap currently sorts static net values and returns the same selection as Greedy. If utilities do not change after each selection, document it as a static heap implementation/data-structure baseline; do not claim it is a dynamic marginal-value optimizer.

## Required implementation

### A. Correctness and complexity

1. Add an O(1) per-index eligibility check. Never construct a new N-length mask inside candidate loops. If a mask is needed, construct it once and reuse it.
2. Replace pairwise minimum-interval validation with an O(K log K) or better method:
   - group selected slots by user;
   - sort each user's selected slots;
   - compare adjacent slots to validate the minimum interval;
   - independently detect duplicate (user, slot) decisions.
3. Validate candidate arrays:
   - one-dimensional and equal length;
   - finite numeric value/cost;
   - non-negative resource costs;
   - eligibility length and dtype/shape;
   - deterministic behavior for ties.
4. Keep the independent feasibility checker separate from the solver's incremental state. The final selected set must be rechecked independently.
5. Avoid dense user × time × action tensors. Keep sparse candidate rows.

### B. Dual optimization

1. Implement signed residuals for inequality constraints:
   - budget: (used_budget - budget_limit) / max(budget_limit, 1);
   - each slot: (used_slot - slot_capacity) / max(slot_capacity, 1);
   - each action quota: (used_action - action_quota) / max(action_quota, 1).
2. Update prices with projected dual ascent:
   lambda = max(0, lambda + step_size * signed_residual).
   Prices must be able to decrease when a resource is underused.
3. Report separately:
   - primal constraint violation;
   - signed residuals;
   - dual prices;
   - complementarity / stopping diagnostics;
   - iteration count and stopping reason.
4. Do not use a single positive-excess residual as the only convergence condition.
5. Revisit the reduced utility and dual objective derivation. Document the exact primal objective and Lagrangian. Ensure resource cost is not accidentally counted twice: the base objective is currently sum(incremental_value - resource_cost), while the budget multiplier applies to resource cost as a constraint price.
6. Add a repair/improvement stage that always preserves feasibility. Compare the repaired dual solution with the feasible Greedy incumbent and retain the better objective. Record whether fallback/incumbent selection was used; do not hide a poor dual result.
7. For line search, optimize a mathematically meaningful dual objective or a clearly defined residual/merit function. Document what is minimized and test that accepted steps do not worsen that merit function.

### C. Coordinate and heap semantics

1. Implement true sequential coordinate updates (recompute selection/residual after each coordinate), or rename/document the method honestly if it is not coordinate descent.
2. Keep Heap and Greedy as separate data-structure baselines only if their semantics are identical. State that identical static scores should produce identical selected sets/objectives.
3. If implementing dynamic heap updates, prove the update rule is consistent with the objective and constraints; do not invent marginal gains.

### D. Tests

Add unit and property-style tests for:

- exact reference versus every heuristic on many deterministic small seeds;
- all solver outputs independently feasible;
- eligibility omitted and partially false;
- zero budget / zero slot capacity;
- negative net-value candidates;
- action quota absent and present;
- duplicate user-time candidates;
- minimum interval edge cases;
- underused resources causing dual prices to decrease;
- dual prices remaining non-negative;
- coordinate update actually recomputing between coordinates;
- deterministic tie handling;
- large selected sets where feasibility checking completes without pairwise O(K²) behavior.

For small instances, exact enumeration is the quality oracle. Report optimality gap. A solver that returns a feasible empty set must not automatically receive a quality PASS.

### E. Benchmark protocol

Use the exact same candidate arrays, objective, eligibility, and constraints for every solver within a run.

Run:

- 10K and 100K for at least 3 fixed seeds;
- 500K and 1M for at least 1 fixed seed;
- 10M only after a memory preflight and only if the estimated peak fits a conservative fraction of available RAM.

Record for every solver and scale:

- candidate count and selected count;
- net objective and incremental value;
- total cost and each constraint utilization;
- independent feasibility result and maximum violation;
- optimality gap where exact truth is available;
- relative gap to the best feasible heuristic for large scales (clearly label this as heuristic-relative, not optimality gap);
- dual prices, primal residual, stopping diagnostics and iterations;
- wall time, candidates/second, peak RSS, candidate-array memory;
- seed, Python, NumPy, OS and hardware/runtime metadata.

Use separate processes or reset peak-RSS measurements between scale runs where practical. Do not report process-wide cumulative peak RSS as if it were each solver's isolated memory use.

### F. Resource guard

Before 1M/10M runs:

- estimate base arrays, sort/index arrays, utility arrays and expected Python state overhead;
- inspect available RAM where supported;
- abort or skip a scale with an explicit RESOURCE_GUARD_SKIPPED status if the conservative estimate exceeds the configured memory budget;
- never silently OOM or silently reduce the candidate count.

Keep 1M as the default maximum automated CI scale. Keep 10M opt-in/manual only.

## Required outputs

1. Updated implementation and tests.
2. Updated scripts/run_push_optimizer_benchmark.py (or a dedicated scale runner) with repeatable CLI options.
3. Machine-readable JSON report and concise Markdown report under an output directory; generated artifacts must not be committed if large or machine-specific.
4. Update docs/benchmarks/PUSH_CAUSAL_P5_OPTIMIZER.md with actual new results, limitations, and clear evidence labels.
5. Update the execution table in docs/benchmarks/PUSH_CAUSAL_BENCHMARK_PLAN.md.
6. Keep the existing Top-K simulator API unchanged.
7. Add no fabricated numbers. If a run is skipped or fails, report that explicitly.

## Acceptance gates

- Unit tests pass.
- Exact small-instance comparisons are reproducible across seeds.
- All selected sets pass the independent feasibility checker.
- No O(N²) eligibility-mask allocation or pairwise interval validation remains.
- Dual prices are non-negative and can decrease when constraints are underused.
- Solver quality is reported separately from feasibility and runtime.
- 100K and 500K benchmarks complete with machine-readable reports.
- 1M completes within the configured time and memory budget, or returns an explicit resource-guard skip with evidence.
- 10M is not required for this task; it must remain opt-in until 1M is stable.

## Execution discipline

- Inspect before changing; preserve compatible interfaces.
- Run focused tests after each change, then the full relevant test set.
- Run the scale benchmark only after correctness and complexity fixes.
- Commit changes to the existing branch with a descriptive commit message.
- Do not merge PR #3.
- Finish with a concise report: files changed, tests run, scale results, unresolved risks, and the exact next action.
