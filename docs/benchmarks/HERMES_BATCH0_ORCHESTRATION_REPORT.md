# Hermes + Codex SDK Batch 0 Orchestration Report

Run ID: `hermes-batch0-20261116`

Evidence class: synthetic engineering fixture only; no causal or business result.

## Gate result

- Hermes native Kanban board created the DAG and stored task lifecycle events.
- Official `openai_codex.Codex` created/resumed worker threads; thread and turn IDs are recorded below.
- Independent deterministic validator ran outside the worker thread.
- A → B → C completed with a forced B interruption and restart recovery.
- A was idempotently skipped after restart; B resumed using its persisted Codex thread ID.
- One validation failure repaired successfully; repeated failure entered BLOCKED after the limit.

## Stage summary

| Stage | Hermes task | Status | Attempts | Validation | Checkpoint |
|---|---|---:|---:|---|---|
| A | `t_ae69ce17` | ACCEPTED | 1 | PASS | validated |
| B | `t_f6ef6f10` | ACCEPTED | 2 | PASS | validated |
| BLOCKED_REPAIR | `t_61f4c54f` | BLOCKED | 3 | BLOCKED | blocked_cycle_3 |
| C | `t_65aa13a9` | ACCEPTED | 1 | PASS | validated |
| REPAIR | `t_3a83b46e` | ACCEPTED | 2 | PASS | validated |

## Event trace

| Time | Phase | Event | Details |
|---|---|---|---|
| 2026-11-16T13:04:31+00:00 | - | dag_created | `{"task_ids": {"A": "t_ae69ce17", "B": "t_f6ef6f10", "BLOCKED_REPAIR": "t_61f4c54f", "C": "t_65aa13a9", "REPAIR": "t_3a83b46e"}}` |
| 2026-11-16T13:04:31+00:00 | A | claimed | `{"task_id": "t_ae69ce17"}` |
| 2026-11-16T13:04:32+00:00 | A | heartbeat | `{"note": "Codex SDK worker started"}` |
| 2026-11-16T13:05:16+00:00 | A | codex_turn_completed | `{"duration_ms": 40333, "status": "completed", "thread_id": "01a1e475-91ce-7211-87df-dbc5cd3e5547", "turn_id": "01a1e475-9ed7-7e01-8436-1fe28abcdc57"}` |
| 2026-11-16T13:05:16+00:00 | A | validated | `{"actual": "A_COMPLETED_ONCE", "expected": "A_COMPLETED_ONCE", "path_role": "A_COMPLETED_ONCE.txt"}` |
| 2026-11-16T13:05:17+00:00 | B | claimed | `{"task_id": "t_f6ef6f10"}` |
| 2026-11-16T13:05:18+00:00 | B | heartbeat | `{"note": "Codex SDK worker writing durable checkpoint"}` |
| 2026-11-16T13:05:36+00:00 | B | codex_turn_completed | `{"duration_ms": 18057, "status": "completed", "thread_id": "01a1e476-447a-7d51-8f42-8d3aecde449f", "turn_id": "01a1e476-4494-70f2-a5fb-50b9192c8218"}` |
| 2026-11-16T13:05:36+00:00 | B | forced_interruption | `{"checkpoint": "b_checkpoint_written"}` |
| 2026-11-16T13:05:58+00:00 | A | idempotent_skip | `{}` |
| 2026-11-16T13:05:59+00:00 | B | reclaimed_after_restart | `{}` |
| 2026-11-16T13:06:00+00:00 | B | claimed | `{"task_id": "t_f6ef6f10"}` |
| 2026-11-16T13:06:00+00:00 | B | heartbeat | `{"note": "resuming same Codex thread from durable checkpoint"}` |
| 2026-11-16T13:06:13+00:00 | B | codex_turn_completed | `{"duration_ms": 11234, "status": "completed", "thread_id": "01a1e476-447a-7d51-8f42-8d3aecde449f", "turn_id": "01a1e476-f004-7700-9f49-76d272e75df8"}` |
| 2026-11-16T13:06:13+00:00 | B | validated | `{"actual": "B_COMPLETED", "expected": "B_COMPLETED", "path_role": "B_COMPLETED.txt"}` |
| 2026-11-16T13:06:14+00:00 | C | claimed | `{"task_id": "t_65aa13a9"}` |
| 2026-11-16T13:06:15+00:00 | C | heartbeat | `{"note": "dependency B accepted; unlocking C"}` |
| 2026-11-16T13:07:07+00:00 | C | codex_turn_completed | `{"duration_ms": 51507, "status": "completed", "thread_id": "01a1e477-2331-7060-8a4a-1c6254e437da", "turn_id": "01a1e477-2349-7c21-b484-b5ad76977c79"}` |
| 2026-11-16T13:07:07+00:00 | C | validated | `{"actual": "C_COMPLETED", "expected": "C_COMPLETED", "path_role": "C_COMPLETED.txt"}` |
| 2026-11-16T13:07:44+00:00 | REPAIR | claimed | `{"task_id": "t_3a83b46e"}` |
| 2026-11-16T13:08:04+00:00 | REPAIR | codex_turn_completed | `{"duration_ms": 14124, "status": "completed", "thread_id": "01a1e478-80ae-7c82-98a8-32c84b659585", "turn_id": "01a1e478-946a-7420-9031-962a3f5049b0"}` |
| 2026-11-16T13:08:04+00:00 | REPAIR | validation_rejected | `{"actual": "WRONG", "expected": "FIXED", "path_role": "repair.txt", "reason": "marker mismatch"}` |
| 2026-11-16T13:08:14+00:00 | REPAIR | codex_turn_completed | `{"duration_ms": 8824, "status": "completed", "thread_id": "01a1e478-80ae-7c82-98a8-32c84b659585", "turn_id": "01a1e478-d037-7c42-be98-bda8abd7e35d"}` |
| 2026-11-16T13:08:14+00:00 | REPAIR | validated | `{"actual": "FIXED", "expected": "FIXED", "path_role": "repair.txt"}` |
| 2026-11-16T13:08:15+00:00 | BLOCKED_REPAIR | claimed | `{"task_id": "t_61f4c54f"}` |
| 2026-11-16T13:08:23+00:00 | BLOCKED_REPAIR | codex_turn_completed | `{"duration_ms": 8436, "status": "completed", "thread_id": "01a1e478-f810-7e22-b6da-55873df4b231", "turn_id": "01a1e478-f829-7eb1-a7bf-4d9a54ca7445"}` |
| 2026-11-16T13:08:23+00:00 | BLOCKED_REPAIR | validation_rejected | `{"actual": "WRONG", "expected": "FIXED", "path_role": "blocked.txt", "reason": "marker mismatch"}` |
| 2026-11-16T13:08:34+00:00 | BLOCKED_REPAIR | codex_turn_completed | `{"duration_ms": 9650, "status": "completed", "thread_id": "01a1e478-f810-7e22-b6da-55873df4b231", "turn_id": "01a1e479-1df1-7fe1-a21e-de47e3f2aca9"}` |
| 2026-11-16T13:08:34+00:00 | BLOCKED_REPAIR | validation_rejected | `{"actual": "WRONG", "expected": "FIXED", "path_role": "blocked.txt", "reason": "marker mismatch"}` |
| 2026-11-16T13:08:51+00:00 | BLOCKED_REPAIR | codex_turn_completed | `{"duration_ms": 15387, "status": "completed", "thread_id": "01a1e478-f810-7e22-b6da-55873df4b231", "turn_id": "01a1e479-48b5-7252-ba67-1d21672e7b4b"}` |
| 2026-11-16T13:08:51+00:00 | BLOCKED_REPAIR | validation_rejected | `{"actual": "WRONG", "expected": "FIXED", "path_role": "blocked.txt", "reason": "marker mismatch"}` |
| 2026-11-16T13:08:52+00:00 | BLOCKED_REPAIR | retry_limit_reached | `{"max_repair_cycles": 2}` |

## Limits

This report proves the local bounded orchestration path only. It does not prove long-running causal validity, real MRT data availability, online lift, or unattended operation after the machine is offline.

## Deterministic test evidence

- Adapter tests: `pytest -q tests/test_hermes_codex_sdk_adapter.py` → `2 passed`.
- Full project suite: `pytest -q` with Python 3.11 and serial BLAS settings → `111 passed, 2 skipped, 432 warnings`.
- The first local adapter attempt exposed a multi-line Hermes `--json` parsing bug. The failure was preserved in the local execution history, the parser was corrected to parse the complete JSON document, and the run was resumed through the same idempotency keys without creating duplicate tasks.
