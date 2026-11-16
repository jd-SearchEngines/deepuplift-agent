# Issue #12 Long-Term Autonomous Research Runner Report

Observed: 2026-11-17 (Asia/Shanghai)  
Branch: `codex/issue12-long-term-runner`  
Draft PR: [#13](https://github.com/jd-SearchEngines/deepuplift-agent/pull/13)

## Current verdict

`RUNNER_ENGINEERING=PASS` for installation, dynamic planning, stable state, and the observed Supervisor kill/restart path. `AUTONOMOUS_CONTINUOUS=PARTIAL`: an actual Codex SDK Worker is still running the next optimizer phase and has not yet produced a terminal result, so this report does not claim full multi-phase completion.

The real myBPmyLife MRT lane remains `BLOCKED`; the two new research reports are semi-synthetic known-truth evidence only and do not claim business lift.

## Environment audit

| Component | Observation | Gate |
|---|---|---|
| Hermes | Hermes Agent v0.23.1, user-local installation present | PASS |
| Codex SDK | `openai_codex` imports in the validated Python 3.11.15 SDK environment | PASS |
| Project runner | Python 3.11.15 under the SDK venv; system `python3` is 3.9.6 | PASS |
| GitHub CLI | authenticated; credential values omitted | PASS |
| State | `~/Library/Application Support/DeepUplift/runner/` with SQLite WAL, checkpoints and logs | PASS |
| Raw MRT | no locally authorized raw myBPmyLife archive used | BLOCKED |

## Actual launchd evidence

`scripts/autonomous_runner/install_launchd.sh` installed and loaded:

```text
gui/501/com.deepuplift.autonomous-runner
state = running
properties = keepalive | runatload
program = .../codex-sdk-runtime/.venv/bin/python
state_root = ~/Library/Application Support/DeepUplift/runner/
```

The observed initial Supervisor PID was `20724`. After the controlled `kill -9 21437`, launchd restarted the service with PID `21662` and `runs = 2`. No root launch was used.

## Planner and phase transition

The Planner no longer emits the historical fixed Batch 3 DAG. It reads project status, research plan, persisted stages and machine-readable reports, then selects the next eligible task. At observation time it recognized the existing Sequential DR and Multi-arm reports, did not rerun them, and advanced to:

`research:optimizer-evidence:v1`

The Ledger contained one accepted reconciled research stage and one active optimizer stage. This is evidence of dynamic phase selection, not evidence that the optimizer phase has passed.

## Recovery evidence

The forced Supervisor interruption produced these persisted events:

- `recovery_thread_checkpoint`
- `recovery_result_consumed`
- `validation_rejected`
- `worker_started`

The same Codex Thread ID `01a1e50e-0df2-7d01-baf5-0df46bfbce78` was retained when the interrupted optimizer task was routed into bounded repair. The completed prior phase was not duplicated. The recovery path records `repair_count=1` and will enter `BLOCKED` after the configured retry limit if the worker/validator cannot complete.

This is stronger than an in-process unit test because the Supervisor was actually killed and launchd restarted the local service. It is still not a full success claim: the current optimizer worker must produce a result and pass independent validation before the overall continuous gate can become PASS.

## New research work

- [Sequential DR report](SEQUENTIAL_DR_REPORT.md): 5 fixed seeds, known-truth trajectories, exact continuation values, ESS, weight tails and cluster bootstrap. The report separates ordinary per-decision DR from true Sequential DR. Synthetic gate `PASS`; real MRT `BLOCKED`; online lift `NOT_RUN`.
- [Multi-arm chain report](MULTIARM_PUSH_CHAIN_REPORT.md): 5 fixed seeds, three logged-arm probabilities, cluster bootstrap, candidate construction, constrained optimizer and multi-arm OPE. This is explicitly an oracle-effect diagnostic, not fitted-model quality or business lift.

## Tests

- Full local suite before launchd: `126 passed, 2 skipped`.
- Recovery-focused orchestration tests: `10 passed`.
- Sequential DR tests: `3 passed`.
- Multi-arm chain test: `1 passed`.

## Remaining state and next action

The service is intentionally left loaded under launchd. At the latest observation the optimizer SDK thread still had an active writer, with fresh Heartbeat but no result file after the bounded-recovery attempt. The next evidence required is the optimizer Worker result, independent acceptance command, and a subsequent Planner event. If the bounded retries are exhausted, the correct outcome is `BLOCKED` with the SDK/validation error preserved; a running process alone will never be promoted to `PASS`.
