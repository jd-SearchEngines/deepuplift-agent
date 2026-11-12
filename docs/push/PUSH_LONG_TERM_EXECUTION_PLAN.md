# Long-Term Push Causal Decision System — Detailed Execution Plan

**Status:** Active implementation plan  
**Updated:** 2026-11-13  
**Repository:** jd-SearchEngines/deepuplift-agent  
**Execution branch:** codex/push-mrt-causal-benchmark  
**Tracking PR:** #3  
**Primary dataset:** myBPmyLife Micro-Randomized Trial (MRT)

---

## 1. Product and research objective

DeepUplift will use long-term Push intervention as its primary end-to-end research scenario.

The business question is:

> Given a user's current state and intervention history, should we send a Push now, which intervention arm should be selected, and when should we stop or wait — while maximizing incremental user outcome and respecting contact-capacity and frequency constraints?

The first version studies intervention selection and timing/frequency. It does not generate Push copy or optimize message text. The public MRT has Activity and Diet intervention arms, so these are the concrete treatment choices.

The research system must connect:

1. Data and randomization audit.
2. Causal identification.
3. Proximal treatment-effect estimation.
4. History-aware policy learning.
5. Frequency- and capacity-constrained optimization.
6. Off-policy policy evaluation (OPE).
7. Evidence-based comparison against simple baselines.

The intended engineering chain is:

    user state + prior assignment history
        -> valid decision-point dataset
        -> proximal treatment-effect model
        -> candidate actions and predicted incremental value
        -> timing / frequency-aware policy
        -> constrained optimizer
        -> OPE and known-truth validation
        -> reproducible evidence report

The system must distinguish three evidence types throughout:

- REAL_RANDOMIZED_MRT: conclusions supported by the observed randomized MRT and its actual outcome windows.
- SEMI_SYNTHETIC_KNOWN_TRUTH: controlled experiments where potential outcomes and policy values are generated and known.
- ENGINEERING_ONLY: schema, solver, performance, or API tests that do not establish causal or business effects.

Never mix these evidence classes in one result table without an explicit label.

## 2. Fixed dataset decision

For this development cycle, use myBPmyLife MRT as the only primary public dataset.

Do not expand the primary scope to Hillstrom, Criteo, X5, IHDP, or ACIC in this cycle. They may remain in the general registry, but they are not required to complete the Push research path.

### 2.1 What the MRT can support

The current repository audit identifies:

- Repeated decision records clustered within approximately 294 participants.
- Three assignment arms: none, activity, diet.
- Declared arm probabilities in the current adapter: none 0.75, activity 0.125, diet 0.125. Re-verify these against upstream study documentation and the raw table before every causal run.
- Activity outcome: steps measured in a 60-minute post-decision window.
- Diet outcome: low-sodium choices measured in a 24-hour post-decision window.
- Assignment and notification/delivery fields are distinct in the raw schema.
- Repeated decision rows are not independent users.

The current repository audit also reports that the raw table does not expose a reliable calendar timestamp or an explicit availability field in the canonical adapter. These must remain NOT_AVAILABLE unless upstream documentation or a newly verified raw field proves otherwise.

### 2.2 What the MRT cannot currently establish

Do not claim that the real MRT directly identifies:

- Actual elapsed time since the previous delivered Push, unless a validated timestamp exists.
- A production daily or weekly frequency cap, unless the time unit and decision eligibility support it.
- A commercial revenue or profit effect.
- Individual counterfactual truth or real-data PEHE.
- A long-term cumulative business lift merely by summing short-window outcomes.
- A per-protocol delivery effect by treating delivery as randomized assignment.

The real dataset's strongest immediate evidence is assignment-based proximal ITT for the documented outcome windows, with participant-clustered uncertainty. History-conditional estimates may be studied where the pre-treatment history and randomization mechanism support them, but they must be labeled with their actual estimand and assumptions.

### 2.3 Data access and provenance

- Keep the raw data local; never commit raw participant-level data to Git.
- Record the upstream source URL, retrieval/version metadata, file SHA256, license/reuse statement, raw row count, participant count, and loader version.
- The experiment must stop with BLOCKED or NOT_AVAILABLE if the local data file is absent or the license cannot be verified.
- Do not report a dataset as runnable merely because it appears in a metadata registry.

## 3. Current implementation audit and concrete issues to resolve

The following findings are based on the current branch code and progress report. Re-check them before changing code, because the branch may advance.

### 3.1 Existing implementation to preserve

- MRT loader and assignment audit: deepuplift/data/public/mrt.py
- Canonical repeated-decision schema: deepuplift/data/push.py
- Randomized proximal baseline: deepuplift/benchmarks/push_mrt.py
- Static known-truth identification stress: deepuplift/benchmarks/push_semisynthetic.py
- Longitudinal known-truth lab: deepuplift/benchmarks/push_longitudinal.py
- Sequential OPE implementation: deepuplift/core/ope.py
- Existing constrained optimizer: deepuplift/decision/optimization/constrained.py
- Existing scripts: scripts/run_push_decision_points.py, scripts/run_push_mrt_baseline.py, scripts/run_push_semisynthetic.py, scripts/run_push_longitudinal.py, scripts/run_push_optimizer_benchmark.py

Do not rewrite working modules without a measured reason. Extend their contracts and tests.

### 3.2 OPE propensity semantics need correction

In sequential OPE, the denominator must be the probability of the action that was actually logged:

    mu_t = P(A_t = observed_action | H_t, eligible_t = 1)

For a binary push/no-push policy:

- If the logged action is Push, mu_t = p_push.
- If the logged action is No Push, mu_t = 1 - p_push.
- If the decision is genuinely ineligible and No Push is forced, the action probability is 1 for the forced action, provided eligibility is observed and the estimand conditions on that decision process.

For a three-arm MRT action, use the probability of the observed arm, not a generic Push probability.

The current longitudinal generator records an assignment_probability representing P(Push), even when the logged action is No Push. Do not pass that value as the logged-action propensity. Add separate, unambiguous fields such as:

- push_probability
- logged_action_probability
- assignment_probability_by_arm where relevant
- eligibility / availability status, only when genuinely observed or explicitly synthetic

The OPE API should make it difficult to accidentally pass P(Push) where P(observed action) is required. Validate this contract with tests.

### 3.3 The current per-decision metric is not sequential per-decision importance sampling

The current sequential evaluator computes a row-level ratio and averages ratio_t * reward_t. That is a one-step / stepwise estimator; it does not use the cumulative history ratio required by per-decision importance sampling.

For each participant trajectory, define:

    rho_t = pi_t(A_t | H_t) / mu_t(A_t | H_t)
    W_t = product_{k=1..t} rho_k

Then the undiscounted per-decision IS estimate of mean trajectory return is based on:

    mean_over_users [ sum_t W_t * reward_t ]

For discounted return, include gamma^(t-1).

Trajectory IS is a separate estimator:

    mean_over_users [ (product_{t=1..T} rho_t) * G ]

where G is the full trajectory return and complete trajectories / assumptions are valid.

Keep one-step contextual IPS, per-decision sequential IS, and full-trajectory IS as separate metrics with explicit names and units. Do not label a one-step row estimator as sequential PDIS.

For stochastic target policies, use pi_t(a|H_t), not only a deterministic target_action label. Preserve deterministic policy support as a special case.

### 3.4 Raw propensity validation must happen before clipping

Do not silently turn invalid propensities into valid ones by clipping.

Before any clipping or stabilization, fail or mark BLOCKED on:

- NaN or infinite propensity.
- Probability <= 0 or > 1.
- Missing propensity for an evaluated decision.
- Impossible logged action under the declared logging policy.
- Target action outside logging support.

After validation, clipping/stabilization may be evaluated as a sensitivity variant. Report unclipped and clipped estimates, clipping rate, max weight, weight variance, ESS, and policy support/coverage.

### 3.5 OPE unit consistency requires a known-truth test

The current smoke report shows a large gap between generated mean trajectory value and trajectory IPS. The current report itself flags weight explosion. Rebuild the test so every estimator and truth use the same unit:

- Mean reward per decision.
- Mean undiscounted cumulative reward per user trajectory.
- Mean discounted cumulative reward per user trajectory.

Never compare a sum over users with a per-user mean. Verify all estimator outputs against exact or high-precision Monte Carlo truth over multiple seeds.

### 3.6 Longitudinal DGP has scenario-isolation and oracle-definition risks

Audit deepuplift/benchmarks/push_longitudinal.py carefully.

Specific issues visible in the current implementation:

- The baseline outcome includes a previous_push contribution even in the nominal no-carryover scenario. This means the no-carryover scenario may still contain carryover. Make the scenario controls orthogonal and add invariant tests.
- Fatigue is currently represented primarily by cumulative prior_pushes. Add separate, explicitly defined fatigue mechanisms: cumulative exposure, rolling-window count, and decayed fatigue state. Do not claim they are interchangeable.
- The current oracle action selects on the current true proximal effect. This is a myopic effect oracle, not necessarily the globally optimal dynamic policy over the full horizon. Rename and label it accordingly.
- A true dynamic oracle may only be called an oracle if it optimizes the full known-truth trajectory objective under the same constraints. Use exhaustive search or dynamic programming for small horizons; use it as a diagnostic upper bound only.
- Ensure all policies are evaluated on common exogenous user contexts, availability draws, and paired noise (common random numbers) to reduce comparison variance.
- Potential outcomes must be evaluated under the same current history state; future state transitions must then evolve according to each policy's chosen actions.
- Record the exact DGP parameters and generated truth separately from model-facing features.

Add tests proving that a no-fatigue/no-carryover DGP has exactly zero fatigue and carryover parameters/effects, and that each scenario activates only its declared mechanisms.

### 3.7 Real MRT outcome missingness needs a correct estimand

The current proximal baseline drops outcome-missing rows before calculating HT contrasts. Audit whether the resulting estimator still targets the full randomized decision-point population.

Do not automatically call a complete-case contrast an unbiased full-population ITT.

For each outcome, report:

- randomized/assigned rows by arm;
- outcome-observed rows by arm;
- missingness rate by arm and participant;
- whether missingness differs by arm or pre-treatment state;
- exact denominator and estimand for every reported contrast.

If outcomes are missing, use a justified missingness strategy only when its assumptions can be stated and tested. Otherwise report complete-case/descriptive results and sensitivity bounds, and mark full-population ITT as NOT_IDENTIFIED or BLOCKED as appropriate.

### 3.8 Time and Availability must remain honest

The current canonical schema uses an ordinal decision_sequence derived from study fields. It does not provide a validated calendar timestamp.

- Keep decision_sequence distinct from decision_time.
- Do not call ordinal sequence a number of hours or days.
- Do not infer time_since_last_push from sequence difference.
- Do not infer Availability from whether a notification was delivered or from whether assignment was recorded.
- A synthetic time slot is allowed inside the semi-synthetic lab, but must be labeled synthetic.
- If upstream source documents establish a valid time mapping, add it with provenance and tests.

## 4. Canonical data and model contracts

### 4.1 Real MRT decision-point contract

Maintain the existing canonical schema and add fields only with explicit provenance.

Required concepts:

- participant_id / user_id
- decision_sequence
- decision_time: actual timestamp or null
- pre-treatment state
- pre-treatment history
- availability: observed or null
- assigned treatment arm
- assignment probability for each possible arm or the observed arm
- actual delivery/exposure, separately observed
- outcome and documented outcome window
- missingness indicator
- source row ID and source fields for audit

Every model feature must be available before the decision. Post-treatment outcomes and delivery-derived fields cannot leak into pre-treatment state.

### 4.2 Synthetic decision-point contract

For semi-synthetic experiments, explicitly record:

- synthetic user ID
- synthetic time index
- generated availability
- logging policy probability for every action
- probability of the actually logged action
- logged action
- target policy action probabilities
- pre-treatment state and history
- per-step reward
- potential outcome for every action at the current history
- fatigue state and parameters
- carryover state and parameters
- next-state transition
- complete trajectory return
- DGP scenario, seed, and evidence class

The DGP must expose truth only to evaluators and oracle diagnostics, never to model training or policy features.

### 4.3 Treatment and outcome separation

Real MRT has three arms: none, activity, diet. Keep Activity and Diet outcomes separate:

- Activity → Steps in the 60-minute window.
- Diet → Low-sodium choices in the 24-hour window.

Do not add steps and food-choice counts into one scalar reward without a pre-declared, justified utility conversion and sensitivity analysis.

For synthetic engineering, a binary Push/No Push setup is acceptable for first validating the sequential machinery. After it is correct, extend to the three-arm action space.

## 5. Execution roadmap

Execute the phases in dependency order, while allowing independent tests and documentation to run in parallel. Do not stop after producing documents.

### Phase P0 — Reproduce current state and freeze the baseline

Tasks:

1. Read the current PR diff, current branch HEAD, existing progress report, and relevant tests.
2. Run the existing targeted Push tests and full test suite.
3. Reproduce the current longitudinal and OPE smoke outputs.
4. Save the exact command, config, Git SHA, environment, and result path.
5. Create a baseline issue list with severity and affected files.
6. Do not overwrite or delete earlier evidence.

Acceptance:

- Existing tests reproduce or failures are recorded.
- Current baseline report identifies all results as real, semi-synthetic, or engineering-only.
- No unverified raw data is added to Git.

### Phase P1 — Correct OPE semantics before extending OPE

Tasks:

1. Separate P(Push) from P(observed action).
2. Add target action probability support for stochastic target policies.
3. Implement stepwise/contextual IPS as a clearly named one-step estimator.
4. Implement true per-decision sequential IS using cumulative trajectory weights.
5. Keep full-trajectory IS separate.
6. Define consistent per-user mean trajectory-return units.
7. Validate raw propensities before clipping.
8. Add overlap/support diagnostics, ESS, max and variance of step/cumulative/trajectory weights.
9. Add clipping sensitivity analysis; never report clipped-only results without unclipped comparison.
10. Cluster bootstrap complete participant trajectories, not individual decision rows.
11. Add hand-calculated toy examples with exact answers.
12. Add known-truth Monte Carlo tests across at least five seeds in the initial pass.

Required test cases:

- Binary Push/No Push with unequal logging probabilities.
- Three-arm None/Activity/Diet with unequal arm probabilities.
- Deterministic target policy.
- Stochastic target policy.
- Target policy matching logging policy.
- Target policy outside logging support.
- Ineligible decision with a forced action, only in synthetic data where eligibility is known.
- Invalid zero, negative, >1, NaN and infinite propensities.
- Unequal trajectory lengths and missing outcome handling.
- Cluster bootstrap that resamples whole users.
- Clipping sensitivity with an explicit bias/variance comparison.

Acceptance:

- Exact toy cases pass.
- On known-truth simulations, estimates have interpretable units and multi-seed bias/coverage results.
- Weight explosion is reported as a failure/limitation, not hidden by clipping.
- Sequential DR remains NOT_AVAILABLE until a valid continuation-value estimator is implemented and tested.

### Phase P2 — Make the longitudinal DGP scientifically controlled

Tasks:

1. Refactor DGP parameters into explicit, orthogonal mechanisms.
2. Fix no-carryover and no-fatigue invariants.
3. Add cumulative, rolling-window, and decayed fatigue definitions.
4. Add spacing effects and recovery.
5. Add delayed/carryover effects.
6. Add user-level fatigue heterogeneity.
7. Add temporal drift and negative-effect users.
8. Add overlap/positivity stress with known action probabilities.
9. Add optional measured and hidden confounding scenarios where appropriate.
10. Ensure potential outcomes are known at each history state.
11. Add common random numbers across policies.
12. Rename the existing myopic oracle and add a true dynamic oracle only for small-horizon known-truth cases.
13. Emit a complete DGP manifest and truth file separate from model-facing data.

Initial experiment matrix:

- Users: 100, 500, 1,000; larger runs only after resource preflight.
- Horizon: 8 and 16 decision points.
- Seeds: 5 for smoke, then at least 20 for final comparisons.
- Scenarios: no fatigue/no carryover, homogeneous fatigue, heterogeneous fatigue, spacing/recovery, carryover, temporal drift, negative effect, positivity failure.
- Policies: no push, logged/random, fixed frequency, rule-based, fatigue-aware rule, learned policy when available, myopic oracle, dynamic oracle for small horizons.

Acceptance:

- Mechanism-isolation tests pass.
- Policy comparisons use common exogenous draws.
- Results report mean per-user cumulative value, standard error/CI, push count, user coverage, frequency violations, and negative-effect pushes.
- The DGP can reproduce known truth exactly when noise is zero.
- All outputs are marked SEMI_SYNTHETIC_KNOWN_TRUTH.

### Phase P3 — Learn proximal effects from decision-point history

Tasks:

1. Establish a simple design-based and pooled regression baseline.
2. Add MRT-appropriate WCLS or a documented equivalent for proximal effects.
3. Add regularized treatment-by-context interaction models.
4. Compare simple S/T learners only where sample size and arm support permit.
5. Use participant-grouped cross-fitting or held-out participants.
6. Keep all features pre-treatment.
7. Evaluate on held-out users, not randomly held-out decision rows.
8. On semi-synthetic data, compare predicted effects with known proximal truth.
9. On real MRT, report conditional model estimates as model-based heterogeneity, not individual CATE accuracy.
10. Report uncertainty and shrink unstable subgroup effects toward the population estimate.

Acceptance:

- At least one robust proximal effect baseline runs end-to-end.
- Synthetic PEHE/PEHE-like metrics are emitted only where individual truth exists.
- Real MRT reports no PEHE or individual counterfactual error claim.
- Participant overlap between train/validation/test is zero.
- A simple model remains the default if more complex models do not demonstrate stable held-out improvement.

### Phase P4 — Connect learned effects to Policy and the existing optimizer

Tasks:

1. Create a Candidate Builder from model predictions.
2. Generate candidates per user × decision slot × eligible treatment.
3. Include predicted incremental outcome, action/resource cost, user, time slot, action, eligibility, and provenance.
4. Use the existing AllocationCandidates and AllocationConstraints contracts where possible.
5. Map constraints to the Push scenario:
   - total Push capacity/budget;
   - per-slot capacity;
   - per-user total frequency cap;
   - action quota for Activity/Diet;
   - minimum interval in synthetic time slots;
   - at most one action per user per decision point.
6. Ensure the optimizer consumes model-produced incremental effects, not independently generated random net values.
7. Compare Greedy/Top-K, Heap, Primal-Dual, Line Search, Coordinate, and exact small-instance reference where supported.
8. Independently validate every selected set for feasibility.
9. Report objective value, resource usage, constraint violations, runtime, and optimality/heuristic gap.
10. Add a no-model and random-score baseline to detect whether the model contributes value.
11. Keep separate optimization runs for Steps and Low-sodium outcomes unless a justified utility mapping is explicitly configured.
12. If real monetary cost is unavailable, use unit Push capacity only as an engineering constraint and label it non-monetary.

Acceptance:

- Candidate Builder → Optimizer runs from one command.
- The candidate values demonstrably originate from the fitted effect model.
- All selected policies pass independent feasibility checks.
- Exact small-instance comparisons validate solver quality.
- No synthetic action cost is presented as observed business cost.

### Phase P5 — Policy learning and frequency-aware strategy

Tasks:

1. Implement baseline policies: no Push, randomized/logged policy, fixed frequency, fixed spacing, context rule.
2. Implement a fatigue-aware rule using only observable pre-decision history.
3. Implement a learned proximal-effect policy from P3.
4. Implement constrained policy selection from P4.
5. Compare action choice and policy value under the same DGP/user contexts.
6. Tune policy thresholds on validation users only.
7. Freeze the selected policy before final test evaluation.
8. Report treatment mix, push frequency distribution, user coverage, wait/skip decisions, and fatigue-state distribution.
9. Distinguish myopic proximal-effect optimization from full-horizon cumulative policy optimization.
10. Do not call a rule "learned" unless its parameters or decision function are actually fit from training data.

Acceptance:

- The learned policy beats or explains its performance relative to simple baselines on held-out known-truth simulations.
- Policy performance is stable across seeds and DGP variants, or failure conditions are clearly documented.
- Oracle is a diagnostic upper bound only and is never treated as a deployable policy.

### Phase P6 — Real MRT proximal policy evaluation

Tasks:

1. Re-run the raw MRT audit and confirm assignment probabilities against upstream design documentation.
2. Report participant count, decision-row count, arm counts, unresolved assignments, delivery rates, outcome missingness, and outcome windows.
3. Revisit the complete-case HT denominator and define the actual estimand.
4. Report design-based Activity vs None, Diet vs None, and Diet vs Activity contrasts separately for each supported outcome.
5. Use participant-clustered uncertainty.
6. Estimate history-conditional proximal effects only from pre-treatment variables and verified decision order.
7. Compare a simple fixed/rule policy with a model-based proximal policy only where the MRT design supports the target estimand.
8. Keep assignment-based ITT distinct from delivery/per-protocol effects.
9. Do not claim long-term Push fatigue from real MRT unless actual exposure history and elapsed time are validated.
10. If the time/availability data are insufficient, publish a precise NOT_AVAILABLE statement and continue the long-term work in the semi-synthetic lane.

Acceptance:

- A reproducible real-data report exists for each supported outcome.
- All denominators, missingness, randomization probabilities, and uncertainty units are explicit.
- No full long-term causal claim is made from unsupported fields.

### Phase P7 — Sequential OPE and final end-to-end evidence

Tasks:

1. Complete P1 OPE correctness before using OPE to compare learned policies.
2. Run sequential OPE on known-truth simulations first.
3. Compare one-step IPS, true per-decision IS, trajectory IS, self-normalized variants, and sequential DR only when continuation-value models are available.
4. Fit continuation-value nuisance models with participant-level cross-fitting.
5. Compare unclipped and clipping/stabilization variants.
6. Measure multi-seed bias, variance, CI coverage, ESS, max cumulative weight, and failure rate.
7. Run positivity stress to show when OPE becomes unreliable.
8. Only evaluate real MRT policies at the proximal/decision horizon supported by actual assignment and eligibility information.
9. Produce one end-to-end run: data → effect model → policy → optimizer → OPE → known-truth comparison.
10. Keep true policy value, OPE estimate, and online business lift as three separate concepts.

Acceptance:

- Known-truth policy value and OPE estimate use identical reward horizon and units.
- OPE uncertainty is user-clustered.
- The report identifies unsupported target policies and low-ESS estimates as unreliable/BLOCKED.
- Sequential DR is PASS only after the continuation-value implementation passes known-truth multi-seed tests.
- End-to-end results include simple baselines, learned policy, constrained policy, and known-truth oracle diagnostics.

## 6. Experiment and reporting protocol

There is no requirement to force every research track to use the same metric. Each track must define its own estimand and metrics. The shared protocol is about reproducibility and evidence provenance.

Every run must record:

- experiment ID;
- Git SHA;
- config hash and full config;
- dataset version and SHA256, or DGP version;
- evidence class;
- outcome and outcome window;
- estimand;
- user-level train/validation/test split;
- random seed;
- model and policy version;
- assignment and target policy probabilities;
- sample sizes at both decision-row and participant levels;
- metric definitions and units;
- runtime and peak memory where relevant;
- result artifact path;
- limitations and status.

Required statuses:

- PASS: implementation and stated acceptance checks pass.
- FAIL: an acceptance check failed.
- NOT_RUN: not executed.
- BLOCKED: required data/assumption/resource is unavailable.
- NOT_AVAILABLE: the source does not contain the required field or estimand.
- NOT_IDENTIFIED: available data and assumptions do not identify the target causal quantity.

Do not use a single overall PASS if a required sub-stage is blocked. Report stage-level status.

## 7. Required final result artifacts

Create or update:

- docs/push/PUSH_RESEARCH_ARCHITECTURE.md
- docs/push/PUSH_DATA_DICTIONARY.md
- docs/push/PUSH_FATIGUE_AND_CARRYOVER.md
- docs/push/PUSH_OPE.md
- docs/push/PUSH_END_TO_END_RESULTS.md
- docs/benchmarks/PROGRESS_REPORT.md

Machine-readable outputs should be written under the repository's existing ignored results directory or a documented local output path. Do not commit raw participant data or large generated files. Small aggregate result tables and reproducible configs may be committed.

The final report must contain:

1. What is genuinely implemented.
2. What was actually run and exact commands.
3. Real MRT evidence, separately by outcome.
4. Semi-synthetic known-truth results, separately by scenario and seed.
5. OPE bias/variance/coverage and weight diagnostics.
6. Optimizer feasibility and quality.
7. End-to-end policy value comparison.
8. What remains NOT_AVAILABLE / BLOCKED / NOT_IDENTIFIED.
9. Recommended next technical step and why.

## 8. Engineering execution rules

- Continue on codex/push-mrt-causal-benchmark and update PR #3.
- Do not create a competing branch or duplicate PR.
- Do not merge PR #3.
- Read existing code before editing and reuse valid implementations.
- Make small, coherent commits with meaningful messages.
- Add unit tests, integration tests, and reproducible experiment scripts with each implementation batch.
- Run targeted tests after each module and the full test suite before reporting a milestone.
- Run multi-seed experiments; do not promote a one-seed smoke result as evidence of algorithmic improvement.
- Use resource preflight before large experiments. Never launch 10M-scale work without an explicit resource check.
- Keep raw MRT files and participant-level records out of Git.
- Never fabricate timestamps, availability, action propensities, costs, or outcomes.
- If a dependency is blocked, continue independent work and document the blocker.
- Do not stop after documentation; the goal is working code and measured results.
- Do not ask the user to approve ordinary reversible implementation choices. Ask only when a real data license, credential, destructive operation, or business utility definition is required.

## 9. Definition of done for this research cycle

The cycle is complete only when all of the following are true:

1. Real MRT proximal ITT analysis is reproducible and its missingness/denominator assumptions are explicit.
2. The semi-synthetic sequential DGP has verified no-fatigue/no-carryover invariants and known potential outcomes.
3. OPE uses correct action propensities and correct cumulative sequential weights.
4. Multi-seed OPE bias, variance, confidence-interval coverage, ESS, and positivity failure are reported.
5. At least one learned proximal-effect model is evaluated on held-out participants.
6. Its predictions feed the actual Push candidate builder and constrained optimizer.
7. Frequency, spacing, capacity, and action constraints are independently validated.
8. Learned, rule-based, random, no-Push, and oracle-diagnostic policies are compared on the same known-truth worlds.
9. Real MRT conclusions are clearly separated from semi-synthetic long-term conclusions.
10. A reproducible end-to-end command and final evidence report are available.

The central deliverable is a working and honestly evaluated Push decision pipeline — not a larger collection of disconnected model demos.
