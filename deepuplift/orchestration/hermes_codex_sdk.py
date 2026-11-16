"""Hermes Kanban + official Codex Python SDK adapter for bounded runs.

This module deliberately does not implement a second scheduler.  Hermes owns
the task board and dependency lifecycle through its public ``hermes kanban``
CLI.  The local SQLite ledger only stores project-specific run metadata that
Hermes tasks do not expose, and lives outside the repository by default.

The Codex dependency is lazy so the core DeepUplift package remains importable
in environments that do not install the optional local orchestration runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def stable_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


@dataclass(frozen=True)
class ValidationResult:
    passed: bool
    reason: str
    details: dict[str, Any]


class RunLedger:
    """Durable project metadata outside Hermes' private databases."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _init_db(self) -> None:
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS stages (
                    run_id TEXT NOT NULL,
                    phase_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    dependency_ids TEXT NOT NULL,
                    attempt INTEGER NOT NULL DEFAULT 0,
                    codex_thread_id TEXT,
                    codex_turn_id TEXT,
                    branch TEXT,
                    worktree TEXT,
                    started_at TEXT,
                    heartbeat_at TEXT,
                    checkpoint TEXT,
                    config_hash TEXT,
                    dataset_fingerprint TEXT,
                    validation_status TEXT,
                    report_path TEXT,
                    commit_sha TEXT,
                    pr_url TEXT,
                    last_error TEXT,
                    task_key TEXT,
                    hermes_task_id TEXT,
                    spec_json TEXT,
                    repair_count INTEGER NOT NULL DEFAULT 0,
                    worker_pid INTEGER,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, phase_id)
                );
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    phase_id TEXT,
                    event TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            # Batch 0 ledgers predate the long-running runner.  Migrate only
            # the project-owned metadata table; Hermes' private database is
            # intentionally never opened or modified here.
            columns = {row[1] for row in db.execute("PRAGMA table_info(stages)")}
            for name, definition in (
                ("task_key", "TEXT"),
                ("hermes_task_id", "TEXT"),
                ("spec_json", "TEXT"),
                ("repair_count", "INTEGER NOT NULL DEFAULT 0"),
                ("worker_pid", "INTEGER"),
            ):
                if name not in columns:
                    db.execute(f"ALTER TABLE stages ADD COLUMN {name} {definition}")

    def ensure_stage(
        self,
        run_id: str,
        phase_id: str,
        dependency_ids: Iterable[str] = (),
        *,
        branch: str = "",
        worktree: str = "",
        config: Any = None,
        dataset_fingerprint: str = "synthetic-fixture-only",
        task_key: str | None = None,
        spec: Any = None,
    ) -> None:
        now = utc_now()
        dependencies = json.dumps(list(dependency_ids), sort_keys=True)
        config_hash = stable_hash(config if config is not None else {"phase": phase_id})
        with self._connect() as db:
            db.execute(
                """
                INSERT OR IGNORE INTO stages
                (run_id, phase_id, status, dependency_ids, branch, worktree,
                 config_hash, dataset_fingerprint, task_key, spec_json, updated_at)
                VALUES (?, ?, 'READY', ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (run_id, phase_id, dependencies, branch, worktree, config_hash,
                 dataset_fingerprint, task_key or phase_id,
                 json.dumps(spec, sort_keys=True) if spec is not None else None, now),
            )

    def get(self, run_id: str, phase_id: str) -> dict[str, Any]:
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM stages WHERE run_id=? AND phase_id=?", (run_id, phase_id)
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown stage {run_id}/{phase_id}")
        result = dict(row)
        result["dependency_ids"] = json.loads(result["dependency_ids"])
        return result

    def update(self, run_id: str, phase_id: str, **fields: Any) -> None:
        allowed = {
            "status", "attempt", "codex_thread_id", "codex_turn_id", "started_at",
            "heartbeat_at", "checkpoint", "validation_status", "report_path",
            "commit_sha", "pr_url", "last_error", "task_key", "spec_json",
            "hermes_task_id", "repair_count", "worker_pid",
        }
        invalid = set(fields) - allowed
        if invalid:
            raise ValueError(f"unsupported ledger fields: {sorted(invalid)}")
        if not fields:
            return
        fields["updated_at"] = utc_now()
        assignments = ", ".join(f"{key}=?" for key in fields)
        with self._connect() as db:
            db.execute(
                f"UPDATE stages SET {assignments} WHERE run_id=? AND phase_id=?",
                (*fields.values(), run_id, phase_id),
            )

    def event(self, run_id: str, phase_id: str | None, event: str, **payload: Any) -> None:
        with self._connect() as db:
            db.execute(
                "INSERT INTO events(run_id, phase_id, event, payload, created_at) VALUES(?,?,?,?,?)",
                (run_id, phase_id, event, json.dumps(payload, sort_keys=True), utc_now()),
            )

    def events(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT phase_id,event,payload,created_at FROM events WHERE run_id=? ORDER BY id",
                (run_id,),
            ).fetchall()
        return [
            {"phase_id": row["phase_id"], "event": row["event"],
             "payload": json.loads(row["payload"]), "created_at": row["created_at"]}
            for row in rows
        ]

    def all_stages(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute("SELECT phase_id FROM stages WHERE run_id=? ORDER BY phase_id", (run_id,)).fetchall()
        return [self.get(run_id, row["phase_id"]) for row in rows]


class HermesKanban:
    """Small wrapper over Hermes' public Kanban CLI interface."""

    def __init__(self, board: str):
        self.board = board
        self.env = {**os.environ, "HERMES_KANBAN_BOARD": board}

    def _run(self, args: list[str], *, check: bool = True) -> str:
        result = subprocess.run(
            ["hermes", "kanban", *args],
            text=True,
            capture_output=True,
            env=self.env,
            check=False,
        )
        if check and result.returncode:
            raise RuntimeError(f"Hermes Kanban failed ({result.returncode}): {result.stderr.strip()}")
        if result.returncode and not check:
            return result.stderr.strip()
        return result.stdout.strip()

    def create(
        self,
        title: str,
        body: str,
        *,
        key: str,
        workspace: Path,
        created_by: str = "hermes-batch0",
    ) -> str:
        output = self._run([
            "create", title, "--body", body, "--workspace", f"dir:{workspace}",
            "--idempotency-key", key, "--created-by", created_by, "--json",
        ])
        try:
            payload = json.loads(output)
            return str(payload["id"])
        except (ValueError, KeyError, IndexError) as exc:
            raise RuntimeError(f"could not parse Hermes task id: {output!r}") from exc

    def link(self, parent_id: str, child_id: str) -> None:
        self._run(["link", parent_id, child_id])

    def claim(self, task_id: str) -> None:
        self._run(["claim", task_id, "--ttl", "900"])

    def reclaim(self, task_id: str, reason: str) -> None:
        self._run(["reclaim", task_id, "--reason", reason], check=False)

    def heartbeat(self, task_id: str, note: str) -> None:
        self._run(["heartbeat", task_id, "--note", note])

    def complete(self, task_id: str, result: str, metadata: dict[str, Any]) -> None:
        self._run(["complete", task_id, "--result", result, "--metadata", json.dumps(metadata, sort_keys=True)])

    def block(self, task_id: str, reason: str) -> None:
        self._run(["block", task_id, reason, "--kind", "capability"])

    def comment(self, task_id: str, body: str) -> None:
        self._run(["comment", task_id, body])

    def list(self) -> list[dict[str, Any]]:
        output = self._run(["list", "--json"])
        payload = json.loads(output or "[]")
        if not isinstance(payload, list):
            raise RuntimeError(f"unexpected Hermes task list payload: {payload!r}")
        return payload


class CodexSDKWorker:
    """One Codex SDK thread per task, resumable by persisted thread ID."""

    def __init__(self, *, sandbox: str = "workspace-write"):
        self.sandbox = sandbox

    def run(self, prompt: str, *, cwd: Path, thread_id: str | None = None) -> dict[str, Any]:
        try:
            from openai_codex import Codex, Sandbox
        except ImportError as exc:
            raise RuntimeError(
                "official openai-codex SDK is required; install it in the runtime environment"
            ) from exc
        sandbox_mode = Sandbox(self.sandbox)
        with Codex() as codex:
            if thread_id:
                thread = codex.thread_resume(thread_id, cwd=str(cwd), sandbox=sandbox_mode)
            else:
                thread = codex.thread_start(cwd=str(cwd), sandbox=sandbox_mode)
            result = thread.run(prompt)
            return {
                "thread_id": thread.id,
                "turn_id": result.id,
                "status": str(result.status.value if hasattr(result.status, "value") else result.status),
                "response": result.final_response,
                "duration_ms": result.duration_ms,
                "error": str(result.error) if result.error else None,
            }


class IndependentValidator:
    """Deterministic validator, executed outside the Codex worker thread."""

    def marker(self, path: Path, expected: str) -> ValidationResult:
        actual = path.read_text(encoding="utf-8").strip() if path.exists() else None
        passed = actual == expected
        return ValidationResult(
            passed=passed,
            reason="marker matches expected value" if passed else "marker mismatch",
            details={"path_role": path.name, "expected": expected, "actual": actual},
        )


class Batch0Supervisor:
    """Runs the bounded acceptance fixture using Hermes + Codex SDK."""

    PHASES = ("A", "B", "C", "REPAIR", "BLOCKED_REPAIR")

    def __init__(self, *, run_id: str, run_root: Path, ledger: RunLedger, board: str):
        self.run_id = run_id
        self.run_root = run_root
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.ledger = ledger
        self.hermes = HermesKanban(board)
        self.worker = CodexSDKWorker()
        self.validator = IndependentValidator()
        self.task_ids: dict[str, str] = {}

    def setup(self) -> dict[str, str]:
        specs = {
            "A": "Stage A: write the durable A completion marker for the Batch 0 fixture.",
            "B": "Stage B: write a checkpoint, then resume and write the B completion marker.",
            "C": "Stage C: write the C completion marker after A and B are accepted.",
            "REPAIR": "Validation repair: first produce a rejected marker, then repair it.",
            "BLOCKED_REPAIR": "Validation failure boundary: repeated bad marker must become BLOCKED.",
        }
        for phase, body in specs.items():
            self.task_ids[phase] = self.hermes.create(
                f"Hermes Batch 0 {phase}", body, key=f"{self.run_id}:{phase}", workspace=self.run_root
            )
        self.hermes.link(self.task_ids["A"], self.task_ids["B"])
        self.hermes.link(self.task_ids["B"], self.task_ids["C"])
        for phase in ("REPAIR", "BLOCKED_REPAIR"):
            self.hermes.link(self.task_ids["C"], self.task_ids[phase])
        for phase, deps in (("A", ()), ("B", ("A",)), ("C", ("B",)),
                            ("REPAIR", ("C",)), ("BLOCKED_REPAIR", ("C",))):
            self.ledger.ensure_stage(
                self.run_id, phase, deps, worktree="disposable-fixture",
                config={"phase": phase, "evidence_class": "synthetic-engineering-fixture"},
            )
        self.ledger.event(self.run_id, None, "dag_created", task_ids=self.task_ids)
        return dict(self.task_ids)

    def _claim(self, phase: str, *, reclaim: bool = False) -> None:
        if reclaim:
            self.hermes.reclaim(self.task_ids[phase], "supervisor restart recovery")
        self.hermes.claim(self.task_ids[phase])
        self.ledger.update(self.run_id, phase, status="RUNNING", started_at=utc_now(), heartbeat_at=utc_now())
        self.ledger.event(self.run_id, phase, "claimed", task_id=self.task_ids[phase])

    def _heartbeat(self, phase: str, note: str) -> None:
        self.hermes.heartbeat(self.task_ids[phase], note)
        self.ledger.update(self.run_id, phase, heartbeat_at=utc_now())
        self.ledger.event(self.run_id, phase, "heartbeat", note=note)

    def _worker(self, phase: str, prompt: str) -> dict[str, Any]:
        stage = self.ledger.get(self.run_id, phase)
        result = self.worker.run(prompt, cwd=self.run_root, thread_id=stage["codex_thread_id"])
        self.ledger.update(
            self.run_id, phase, codex_thread_id=result["thread_id"], codex_turn_id=result["turn_id"],
            heartbeat_at=utc_now(), attempt=stage["attempt"] + 1,
        )
        self.ledger.event(
            self.run_id, phase, "codex_turn_completed", thread_id=result["thread_id"],
            turn_id=result["turn_id"], status=result["status"], duration_ms=result["duration_ms"],
        )
        return result

    def _accept(self, phase: str, validation: ValidationResult) -> None:
        self.ledger.update(self.run_id, phase, status="ACCEPTED", validation_status="PASS", checkpoint="validated")
        self.ledger.event(self.run_id, phase, "validated", **validation.details)
        self.hermes.complete(self.task_ids[phase], validation.reason, {"validation": validation.details})

    def _reject(self, phase: str, validation: ValidationResult, *, checkpoint: str) -> None:
        self.ledger.update(
            self.run_id, phase, status="REPAIR_READY", validation_status="REJECTED",
            checkpoint=checkpoint, last_error=validation.reason,
        )
        self.ledger.event(self.run_id, phase, "validation_rejected", reason=validation.reason, **validation.details)
        self.hermes.comment(self.task_ids[phase], f"Independent validator rejected: {validation.reason}")

    def run_a(self) -> None:
        if self.ledger.get(self.run_id, "A")["status"] == "ACCEPTED":
            self.ledger.event(self.run_id, "A", "idempotent_skip")
            return
        self._claim("A")
        self._heartbeat("A", "Codex SDK worker started")
        self._worker("A", """In the current disposable fixture, create A_COMPLETED_ONCE.txt with exactly A_COMPLETED_ONCE. If it already exists, leave it unchanged. Do not modify any other files. Return a short summary.""")
        result = self.validator.marker(self.run_root / "A_COMPLETED_ONCE.txt", "A_COMPLETED_ONCE")
        if not result.passed:
            self._reject("A", result, checkpoint="worker_finished")
            raise RuntimeError("stage A validation failed")
        self._accept("A", result)

    def start_b_and_interrupt(self) -> None:
        self._claim("B")
        self._heartbeat("B", "Codex SDK worker writing durable checkpoint")
        self._worker("B", """Write B_CHECKPOINT.txt with exactly B_CHECKPOINT. Do not write B_COMPLETED.txt yet. Do not change A_COMPLETED_ONCE.txt. Return after the checkpoint is written.""")
        result = self.validator.marker(self.run_root / "B_CHECKPOINT.txt", "B_CHECKPOINT")
        if not result.passed:
            self._reject("B", result, checkpoint="checkpoint_validation_failed")
            raise RuntimeError("stage B checkpoint validation failed")
        self.ledger.update(self.run_id, "B", checkpoint="b_checkpoint_written", validation_status="CHECKPOINT_PASS")
        self.ledger.event(self.run_id, "B", "forced_interruption", checkpoint="b_checkpoint_written")
        os.kill(os.getpid(), signal.SIGKILL)

    def resume_b_and_run_c(self) -> None:
        self.run_a()
        b = self.ledger.get(self.run_id, "B")
        if b["status"] == "RUNNING":
            self.hermes.reclaim(self.task_ids["B"], "controlled supervisor interruption recovery")
            self.ledger.event(self.run_id, "B", "reclaimed_after_restart")
        if b["status"] != "ACCEPTED":
            self._claim("B")
            self._heartbeat("B", "resuming same Codex thread from durable checkpoint")
            self._worker("B", """Resume Stage B from the existing B_CHECKPOINT.txt. Leave A_COMPLETED_ONCE.txt and B_CHECKPOINT.txt unchanged. Now write B_COMPLETED.txt with exactly B_COMPLETED. Do not modify other files. Return a short summary.""")
            checkpoint = self.validator.marker(self.run_root / "B_CHECKPOINT.txt", "B_CHECKPOINT")
            completed = self.validator.marker(self.run_root / "B_COMPLETED.txt", "B_COMPLETED")
            if not checkpoint.passed or not completed.passed:
                self._reject("B", completed if not completed.passed else checkpoint, checkpoint="resume_validation_failed")
                raise RuntimeError("stage B resume validation failed")
            self._accept("B", completed)
        if self.ledger.get(self.run_id, "C")["status"] != "ACCEPTED":
            self._claim("C")
            self._heartbeat("C", "dependency B accepted; unlocking C")
            self._worker("C", """Only after A_COMPLETED_ONCE.txt, B_CHECKPOINT.txt, and B_COMPLETED.txt exist, write C_COMPLETED.txt with exactly C_COMPLETED. Do not modify earlier markers.""")
            result = self.validator.marker(self.run_root / "C_COMPLETED.txt", "C_COMPLETED")
            if not result.passed:
                self._reject("C", result, checkpoint="worker_finished")
                raise RuntimeError("stage C validation failed")
            self._accept("C", result)

    def repair_scenarios(self) -> None:
        self._claim("REPAIR")
        for cycle in range(3):
            stage = self.ledger.get(self.run_id, "REPAIR")
            prompt = (
                "Write repair.txt with exactly WRONG on this first attempt."
                if cycle == 0 else
                "The independent validator rejected repair.txt because it did not equal FIXED. "
                "Write repair.txt with exactly FIXED and do not change other files."
            )
            self._worker("REPAIR", prompt)
            result = self.validator.marker(self.run_root / "repair.txt", "FIXED")
            if result.passed:
                self._accept("REPAIR", result)
                break
            self._reject("REPAIR", result, checkpoint=f"repair_cycle_{cycle + 1}")
            if cycle >= 1:
                raise RuntimeError("repair scenario unexpectedly exceeded two cycles")
            self.hermes.comment(self.task_ids["REPAIR"], "REPAIR_READY: exact validator failure sent to Codex continuation")
        else:
            raise RuntimeError("repair scenario did not converge")

        self._claim("BLOCKED_REPAIR")
        for cycle in range(3):
            self._worker("BLOCKED_REPAIR", "Write blocked.txt with exactly WRONG. Do not write the expected value.")
            result = self.validator.marker(self.run_root / "blocked.txt", "FIXED")
            if result.passed:
                raise RuntimeError("blocked scenario unexpectedly passed")
            self._reject("BLOCKED_REPAIR", result, checkpoint=f"blocked_cycle_{cycle + 1}")
            if cycle >= 2:
                self.ledger.update(self.run_id, "BLOCKED_REPAIR", status="BLOCKED", validation_status="BLOCKED")
                self.ledger.event(self.run_id, "BLOCKED_REPAIR", "retry_limit_reached", max_repair_cycles=2)
                self.hermes.block(self.task_ids["BLOCKED_REPAIR"], "two automatic repairs failed; human diagnosis required")
                break
            self.hermes.comment(self.task_ids["BLOCKED_REPAIR"], f"REPAIR_READY cycle {cycle + 1}; retry remains bounded")

    def report(self, path: Path) -> None:
        stages = self.ledger.all_stages(self.run_id)
        events = self.ledger.events(self.run_id)
        payload = {
            "run_id": self.run_id,
            "evidence_class": "synthetic-engineering-fixture",
            "board": self.hermes.board,
            "task_ids": self.task_ids,
            "stages": stages,
            "events": events,
            "raw_fixture_committed": False,
        }
        path.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        lines = [
            "# Hermes + Codex SDK Batch 0 Orchestration Report", "",
            f"Run ID: `{self.run_id}`", "",
            "Evidence class: synthetic engineering fixture only; no causal or business result.", "",
            "## Gate result", "",
            "- Hermes native Kanban board created the DAG and stored task lifecycle events.",
            "- Official `openai_codex.Codex` created/resumed worker threads; thread and turn IDs are recorded below.",
            "- Independent deterministic validator ran outside the worker thread.",
            "- A → B → C completed with a forced B interruption and restart recovery.",
            "- A was idempotently skipped after restart; B resumed using its persisted Codex thread ID.",
            "- One validation failure repaired successfully; repeated failure entered BLOCKED after the limit.",
            "",
            "## Stage summary", "",
            "| Stage | Hermes task | Status | Attempts | Validation | Checkpoint |",
            "|---|---|---:|---:|---|---|",
        ]
        for stage in stages:
            lines.append(
                f"| {stage['phase_id']} | `{self.task_ids.get(stage['phase_id'], '')}` | {stage['status']} | "
                f"{stage['attempt']} | {stage['validation_status'] or '-'} | {stage['checkpoint'] or '-'} |"
            )
        lines += ["", "## Event trace", "", "| Time | Phase | Event | Details |", "|---|---|---|---|"]
        for event in events:
            safe_payload = json.dumps(event["payload"], sort_keys=True).replace("|", "\\|")
            lines.append(f"| {event['created_at']} | {event['phase_id'] or '-'} | {event['event']} | `{safe_payload}` |")
        lines += [
            "", "## Limits", "",
            "This report proves the local bounded orchestration path only. It does not prove long-running causal validity, real MRT data availability, online lift, or unattended operation after the machine is offline.",
            "",
        ]
        path.write_text("\n".join(lines), encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Hermes + Codex SDK Batch 0 fixture")
    parser.add_argument("--mode", choices=("initial", "resume", "repair", "report"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--board", required=True)
    parser.add_argument("--report", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    ledger = RunLedger(args.ledger)
    supervisor = Batch0Supervisor(run_id=args.run_id, run_root=args.run_root, ledger=ledger, board=args.board)
    if args.mode == "initial":
        supervisor.setup()
        args.ledger.with_suffix(".tasks.json").write_text(
            json.dumps(supervisor.task_ids, indent=2) + "\n", encoding="utf-8"
        )
        supervisor.run_a()
        supervisor.start_b_and_interrupt()
    elif args.mode == "resume":
        supervisor.task_ids = {stage["phase_id"]: "" for stage in ledger.all_stages(args.run_id)}
        # Task IDs are supplied by the setup manifest beside the ledger.
        manifest = args.ledger.with_suffix(".tasks.json")
        supervisor.task_ids = json.loads(manifest.read_text(encoding="utf-8"))
        supervisor.resume_b_and_run_c()
    elif args.mode == "repair":
        manifest = args.ledger.with_suffix(".tasks.json")
        supervisor.task_ids = json.loads(manifest.read_text(encoding="utf-8"))
        supervisor.repair_scenarios()
    elif args.mode == "report":
        manifest = args.ledger.with_suffix(".tasks.json")
        supervisor.task_ids = json.loads(manifest.read_text(encoding="utf-8"))
        if args.report is None:
            raise SystemExit("--report is required for report mode")
        supervisor.report(args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
