# Autonomous Research Runner Acceptance Report

Run: `batch3-front-end-20261116`  
Board: `deepuplift-batch3-20261116`  
Date: 2026-11-16  
Evidence class: local engineering + semi-synthetic research evidence

## Architecture

```text
Research documents
        |
        v
Hermes Planner --stable idempotency keys--> Hermes Kanban DAG
        |                                      |
        |                              public claim/heartbeat
        v                                      v
SQLite project ledger <--- Supervisor <--- Codex SDK worker process
        |                       |             |
        |                       +-- stream heartbeat/thread metadata
        v                                     v
Independent validator ---> bounded repair / BLOCKED ---> GitHub evidence
```

Hermes remains the task-board owner. The SQLite ledger stores only
DeepUplift-specific run metadata outside the repository. The Worker uses the
public asynchronous `openai-codex` API: `AsyncCodex`, `thread_start` or
`thread_resume`, `AsyncThread.turn`, `AsyncTurnHandle.stream`, and
`ThreadReadResponse` after the stream. No private Hermes database table is
modified.

## Acceptance evidence

| Capability | Evidence | Status |
|---|---|---|
| Dynamic DAG and idempotency | Planner created 3 stable task keys; repeated planning returned the same keys | PASS |
| Task claim/dependency unlock | `t_e7126026` → `t_2b94bd22` → `t_57462294` | PASS |
| Long-task heartbeat | Hermes heartbeat events arrived during SDK streams; 514/244/405/876 stream events were recorded on real turns | PASS |
| SDK thread persistence | Engineering retry reused `01a1e4c2-6b4c-7830-bf97-99f1ae69897f` | PASS |
| Forced interruption and recovery | Worker killed with `-9`; same thread `01a1e4c7-1365-7560-a8f0-877b1c0c1683` resumed and wrote `COMPLETE` | PASS |
| Validation/repair | Missing `python` command produced return 127 and `REPAIR_READY`; bounded retry later passed | PASS |
| BLOCKED boundary | Unit test exercises two repair cycles then `BLOCKED`; no infinite retry | PASS |
| Independent review | Read-only SDK Worker returned a PARTIAL evidence verdict; no source files changed | PASS |
| Launchd front-end | Scripts and plist template syntax checked; not installed | NOT_INSTALLED |
| GitHub sync | Code/report branch and Draft PR handoff are pending final commit/push in this run | PENDING |

## Real task sequence

1. Runner engineering audit and focused tests. One real shell failure was
   retained; the same SDK thread performed the repair and passed.
2. `push_e2e_v2` executed the five-seed benchmark and wrote the Markdown/JSON
   evidence.
3. Independent read-only review checked the report, code, provenance and
   focused tests, and rejected an unconditional PASS until the report SHA and
   persisted test gate were repaired.

## Boundaries

This proves a local, bounded autonomous development loop while the Mac,
Hermes and Codex authentication are available. It does not prove unattended
operation while the machine is offline, online business lift, real MRT causal
validity, or full Sequential DR. The real myBPmyLife lane remains blocked by
missing local archive and unverified license.
