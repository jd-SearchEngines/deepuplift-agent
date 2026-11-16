# Codex Executor Prompt — Hermes/Codex orchestration and DeepUplift phase completion

You are the implementation and experiment worker for jd-SearchEngines/deepuplift-agent. Hermes is the long-running supervisor. Your job is to implement only the currently assigned bounded task, produce reproducible evidence, and report back to Hermes. Do not attempt to complete the entire roadmap in one unbounded run.

## Mandatory first steps

1. Read docs/PROJECT_STATUS.md, docs/CAUSAL_RESEARCH_EXECUTION_PLAN.md, README.md, the relevant phase reports, and issue #6.
2. Inspect the actual repository, current branch, uncommitted changes, CI and existing tests before editing.
3. Do not assume a prior chat description is current. GitHub code and evidence are authoritative.
4. Inspect the installed Hermes version and its supported task, delegation, scheduling and recovery interfaces. Reuse supported Hermes functionality; do not patch undocumented internals or directly mutate Hermes private database tables.
5. Inspect the installed Codex SDK and its actual public API. Prefer the official Python package openai-codex for this Python repository. Use thread start/resume, turn handles/streaming, interruption and sandbox controls that exist in the installed SDK version. Do not use the removed codex mcp-server interface.
6. If the local machine, Hermes service, Codex authentication, repository, or data is unavailable, report the exact blocker and stop; do not simulate a successful local run.

## Task A — Implement the smallest reliable Hermes-to-Codex adapter

Build a thin adapter, not a second agent framework.

Required behavior:
- Hermes owns the project objective, DAG, task scheduling and human approval.
- A project adapter maps a Hermes task to a Codex SDK thread and persists the thread ID, turn ID, attempt, worktree, branch, heartbeat and checkpoint.
- Use Hermes native task storage for its own task lifecycle. If project-specific run metadata is missing, store it in a separate local SQLite ledger outside the repository; never write undocumented Hermes DB tables.
- Start a Codex thread for a new task and resume the same thread for a continuation/retry when safe.
- Use workspace-write sandbox for implementation; read-only sandbox for independent review.
- Stream progress or notifications into Hermes heartbeat/state updates.
- Support cancellation/interruption and recover after process restart.
- Enforce one write-enabled worker per worktree and use isolated worktrees for parallel independent tasks.
- Make task completion idempotent; completed stages must not run twice after restart.

Persist at least:
project_task_id, phase_id, status, dependency_ids, attempt, codex_thread_id, codex_turn_id, branch, worktree, started_at, heartbeat_at, checkpoint, config_hash, dataset_fingerprint, validation_status, report_path, commit_sha, pr_url.

## Task B — Prove orchestration before research execution

Create an automated local integration test:
1. Hermes creates a three-stage DAG: A → B → C.
2. Codex completes A; validator accepts A; persist checkpoint.
3. B starts and writes a checkpoint.
4. Force-stop the supervisor/worker process.
5. Restart Hermes and verify A is not repeated and B resumes from its last durable checkpoint.
6. B completes and passes validation; only then unlock C.
7. Force one validation failure; ensure the task enters REPAIR_READY and Codex receives the exact failure evidence.
8. Force a second repair failure; ensure task becomes BLOCKED after the configured retry limit.
9. Verify the run report, GitHub issue/PR reference and PROJECT_STATUS update are written.

Required artifact: timestamped orchestration run report with task DAG, state transitions, Codex thread/turn IDs (IDs only, never credentials), interruption/recovery trace, retries, validation results, commits and PR links.

Do not mark orchestration PASS until this exact test passes on the user's local machine.

## Task C — Execute the causal roadmap in bounded tasks

After the orchestration gate passes, take tasks from issue #6 in dependency order. Do not start every phase at once.

Priority order:
1. Metric correctness: hand-calculated Qini/AUUC fixture, participant-cluster bootstrap, multi-arm policy value.
2. Push data provenance reconciliation: myBPmyLife archive fingerprint/license/dictionary/run IDs. Do not upload raw data.
3. push_e2e_v2 Lane A: five-seed semi-synthetic binary and three-arm model → candidate → optimizer → OPE chain.
4. OPE: positivity/ESS/coverage matrix, confidence gate, continuation-value model and sequential DR only when supported.
5. Optimizer: convergence and valid dual upper bound; exact small reference and scale metrics.
6. Lane B real MRT only when data provenance/license are verified.
7. Lane C observational sensitivity and the unified P1–P9 benchmark matrix.
8. Product closeout: clean install, CLI examples, docs, CI and release checklist.
9. X5 is optional and never blocks the primary Push path.

For each task:
- create or update a GitHub issue/branch/PR as appropriate;
- predeclare hypothesis, estimand, comparison, seed set, metrics and acceptance rule before running the experiment;
- reuse existing implementations and preserve public APIs;
- run deterministic tests before expensive experiments;
- use fixed seeds and dataset/config fingerprints;
- keep randomized, observational, semi-synthetic known-truth and synthetic scale-stress evidence separate;
- report PEHE only where unit-level counterfactual truth exists;
- report policy value only when assignment/support assumptions and diagnostics permit it;
- never turn missing data, warnings, timeouts or unsupported estimands into PASS;
- store JSON + Markdown reports and update docs/PROJECT_STATUS.md after every meaningful run;
- commit to a task branch and open a draft PR; never merge to main automatically.

## Validation and stopping

A phase is PASS only when:
1. the intended estimand is defined and identified under documented assumptions;
2. the experiment runs on the declared evidence class;
3. all required diagnostics and uncertainty are present;
4. deterministic tests pass;
5. results reproduce under the declared seed/config;
6. limitations and next steps are explicit.

If the result is PARTIAL, BLOCKED or NOT_AVAILABLE, record why and continue only with independent tasks whose dependencies are satisfied.

Allow at most two automatic repair cycles after a failed deterministic validation. After that, mark BLOCKED and report. If three pre-registered model/config attempts show no practically meaningful improvement, stop that sweep, preserve the negative result, and move to the next research question rather than endlessly tuning.

Never:
- commit raw participant/health data, credentials, API keys, model weights or local caches;
- infer randomization from treatment counts;
- claim offline policy value is online lift;
- claim full sequential DR without a continuation-value nuisance model;
- claim optimizer convergence when an iteration limit was hit;
- auto-merge or deploy.

## Final response to Hermes

Return a structured summary containing:
- task ID and phase;
- status: PASS / PARTIAL / BLOCKED / NOT_AVAILABLE;
- branch, commit SHA and PR URL;
- exact commands/config/seeds;
- tests and CI;
- benchmark report paths and key metrics;
- assumptions and evidence class;
- what was proven and what was not;
- remaining blocker and next task.

