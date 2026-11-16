# Issue #12 Runner Evidence Ledger Extract

This is a sanitized, durable extract of the local SQLite run ledger. The full runtime database remains outside Git at `~/Library/Application Support/DeepUplift/runner/runner.sqlite3`; no credentials, raw participant data or private Hermes database content is copied here.

## Commands and observed transitions

| Observation | Evidence |
|---|---|
| Initial launch | `scripts/autonomous_runner/install_launchd.sh` returned 0; `launchctl print gui/501/com.deepuplift.autonomous-runner` showed `state = running`, `properties = keepalive | runatload`, PID 20724 |
| Dynamic planning | Ledger planner events selected existing report-backed stages and scheduled `research:optimizer-evidence:v1` without rerunning accepted Sequential DR/Multi-arm reports |
| Worker start | `worker_started`, optimizer attempt 1, PID 20769; metadata persisted Codex Thread `01a1e509-bf9b-7d02-b539-3e0ead9859ac` |
| Forced interruption | Supervisor PID 21437 received `SIGKILL`; no ledger or report file was deleted |
| launchd recovery | `launchctl print` after restart showed `runs = 2`, new Supervisor PID 21662 |
| Thread recovery | `recovery_thread_checkpoint` persisted Thread `01a1e50e-0df2-7d01-baf5-0df46bfbce78`; the optimizer stage was not duplicated |
| Failed-turn routing | `recovery_result_consumed` followed by `validation_rejected`; `repair_count=1`, stage `REPAIR_READY` |
| Bounded retry | Same Hermes task ID was claimed again, attempt 3, with `--thread-id 01a1e50e-0df2-7d01-baf5-0df46bfbce78` |
| Current observation | Optimizer Worker remains `RUNNING` with fresh Heartbeat; no terminal result or acceptance is claimed yet |

## Current durable state at report time

```text
ACCEPTED=1, RUNNING=1
accepted: research:multi-arm-chain:v1 (report_reconciled)
running:  research:optimizer-evidence:v1
attempt: 3
repair_count: 1
real MRT: BLOCKED
online lift: NOT_RUN
```

## Interpretation

The launchd and recovery gates are evidenced. The research-runner gate is deliberately `PARTIAL` until the current Worker returns a result and the independent acceptance command passes. Heartbeat, process liveness, and a Codex Thread ID are not substitutes for validation.
