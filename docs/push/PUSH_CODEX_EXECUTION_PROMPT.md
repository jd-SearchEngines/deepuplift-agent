# Codex Execution Prompt — Continue Long-Term Push Research

Repository: jd-SearchEngines/deepuplift-agent  
Branch: codex/push-mrt-causal-benchmark  
Existing PR: #3  
Plan: docs/push/PUSH_LONG_TERM_EXECUTION_PLAN.md

You are being asked to EXECUTE the plan in the linked document, not merely summarize it or create another proposal.

## Mission

Build the long-term Push causal decision pipeline around the existing myBPmyLife MRT work:

data audit → proximal effect model → history-aware Push policy → frequency/capacity optimizer → OPE → reproducible evidence.

Read the full execution plan first. Then inspect the current branch HEAD, PR diff, implementation, tests, and existing progress artifacts. The plan is grounded in the current code audit and calls out specific correctness risks; verify each one before editing.

## Required first actions

1. Stay on codex/push-mrt-causal-benchmark and update PR #3. Do not create a competing branch or PR.
2. Reproduce the current targeted Push tests, full test suite, longitudinal smoke, and sequential OPE smoke.
3. Record current Git SHA, exact commands, results, and failures.
4. Inspect these files first:
   - deepuplift/benchmarks/push_longitudinal.py
   - deepuplift/benchmarks/push_mrt.py
   - deepuplift/data/push.py
   - deepuplift/core/ope.py
   - deepuplift/decision/optimization/constrained.py
   - docs/push/PUSH_RESEARCH_ARCHITECTURE.md
   - docs/push/PUSH_FATIGUE_AND_CARRYOVER.md
   - docs/push/PUSH_OPE.md
5. Verify, do not assume, the findings in the plan.

## Implementation priority

Do not start by adding more model families. Fix the validity of the current experiments first.

### Priority 1 — Correct OPE

- Make logged propensity mean P(observed logged action | history), not merely P(Push).
- Support deterministic and stochastic target policies.
- Separate one-step contextual IPS from true sequential per-decision IS.
- Implement cumulative weights W_t = product of step ratios through t.
- Keep full-trajectory IS as a separate estimator.
- Validate raw propensities before clipping; invalid values must fail or return BLOCKED.
- Add ESS, support/coverage, max and variance of step/cumulative/trajectory weights.
- Make reward horizon and value units consistent.
- Add exact toy tests and at least five-seed known-truth tests.
- Keep full sequential DR NOT_AVAILABLE until a valid continuation-value estimator is implemented and verified.

### Priority 2 — Correct the longitudinal DGP

- Ensure no-carryover scenarios truly have zero carryover.
- Separate cumulative, rolling-window, and decayed fatigue.
- Add recovery/spacing and heterogeneous fatigue.
- Use common exogenous draws across policy comparisons.
- Distinguish myopic effect oracle from a true dynamic oracle.
- Add scenario-isolation tests and known-truth checks.
- Preserve the evidence label SEMI_SYNTHETIC_KNOWN_TRUTH.

### Priority 3 — Learn proximal effects and build a policy

- Start with robust simple baselines and MRT-appropriate WCLS or documented equivalent.
- Use pre-treatment state and history only.
- Split and cross-fit by participant, never random decision rows.
- Evaluate true effect error only in known-truth synthetic data.
- On real MRT, do not claim individual CATE accuracy.
- Build a policy that consumes fitted model predictions; distinguish it from hand-written rules.

### Priority 4 — Connect policy to the real optimizer

- Build candidates per user × decision slot × action from model predictions.
- Reuse AllocationCandidates and AllocationConstraints if they fit.
- Enforce global capacity, per-slot capacity, per-user frequency, minimum interval, action quotas, and one action per user/decision point.
- Independently validate feasibility.
- Compare greedy/heap/primal-dual/coordinate and exact small-instance reference.
- Do not feed the optimizer random synthetic net values when model predictions are available.
- Keep Activity/Steps and Diet/Low-sodium outcomes separate unless a justified utility mapping is explicitly configured.

### Priority 5 — Validate real MRT and policy evaluation

- Re-audit arm probabilities, assignment vs delivery, participant counts, and outcome missingness.
- Review the current complete-case HT denominator; state the actual estimand and missingness assumptions.
- Report Activity vs None, Diet vs None, and Diet vs Activity separately by outcome.
- Use participant-clustered uncertainty.
- Keep real MRT proximal ITT separate from synthetic long-term cumulative policy value.
- Do not infer real elapsed Push intervals or Availability from ordinal sequence or delivery fields.

## Experiment requirements

- Smoke: 5 seeds.
- Final comparative evidence: at least 20 seeds, subject to runtime/resource preflight.
- Initial sizes: 100, 500, 1,000 users; horizon 8 and 16. Larger scales only after resource preflight.
- Use common random numbers for policy comparisons.
- Report per-user mean cumulative value, uncertainty, push count, user coverage, frequency violations, negative-effect pushes, ESS, weight diagnostics, and optimizer feasibility.
- Keep each research track's own estimand and metrics. Do not force a single metric across causal identification, effect estimation, optimization, and OPE.
- Save configs, Git SHA, dataset/DGP version, seed, evidence class, metrics, and exact command.

## Execution behavior

- Continue through the plan in dependency order; parallelize independent tests where safe.
- Commit each coherent implementation batch to the existing branch.
- Update the progress report and relevant Push docs with measured evidence.
- Run targeted tests after each change and the full test suite before milestone completion.
- Do not stop after docs, interfaces, or smoke tests.
- If a real-data field is absent, mark it NOT_AVAILABLE and continue the semi-synthetic lane.
- If OPE has high variance or low ESS, report the failure; do not hide it with clipping.
- Do not commit raw MRT or participant-level data.
- Do not merge PR #3.

## Required response when you finish a work batch

Report:
1. Commit SHA(s) and files changed.
2. Tests run and pass/fail counts.
3. Experiments run, seeds, users, horizon, and exact output paths.
4. Real MRT results separately from semi-synthetic results.
5. OPE bias/coverage/ESS/weight diagnostics.
6. Optimizer feasibility, quality gap, and runtime.
7. Remaining blockers with precise evidence.
8. The next executable task.

Start now with the baseline reproduction and Priority 1 OPE correctness work. Continue beyond that whenever dependencies and resources permit.