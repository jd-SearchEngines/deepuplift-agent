"""Long-running Hermes -> official Codex SDK research runner.

The runner is deliberately a thin project adapter. Hermes owns the task board
and dependency lifecycle; this module owns only DeepUplift-specific evidence,
checkpoint and recovery metadata in the existing external SQLite ledger.

The worker process uses the public asynchronous ``openai-codex`` API. Streaming
notifications update a heartbeat file while the supervisor remains alive in a
different process, so a long Codex turn does not look idle to Hermes.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .hermes_codex_sdk import HermesKanban, IndependentValidator, RunLedger, ValidationResult, utc_now


TERMINAL = {"ACCEPTED", "BLOCKED"}


@dataclass(frozen=True)
class TaskSpec:
    key: str
    title: str
    body: str
    dependencies: tuple[str, ...] = ()
    acceptance_command: str = ""
    evidence_class: str = "ENGINEERING"
    prompt: str = ""
    max_repair_cycles: int = 2
    sandbox: str = "workspace-write"


class ResearchPlanner:
    """Deterministic planner backed by the repository's research documents.

    The planner is intentionally transparent and idempotent. It reads the
    status/plan/reports and emits a stable task key; an optional LLM may later
    propose tasks, but it must still pass through this persisted task contract.
    """

    REQUIRED_DOCS = (
        "docs/PROJECT_STATUS.md",
        "docs/CAUSAL_RESEARCH_EXECUTION_PLAN.md",
    )

    def __init__(self, repo_root: Path):
        self.repo_root = Path(repo_root).resolve()

    def _read(self, relative: str) -> str:
        path = self.repo_root / relative
        return path.read_text(encoding="utf-8") if path.exists() else "MISSING: " + relative

    def context(self) -> dict[str, Any]:
        status = self._read("docs/PROJECT_STATUS.md")
        plan = self._read("docs/CAUSAL_RESEARCH_EXECUTION_PLAN.md")
        reports = sorted((self.repo_root / "docs/benchmarks").glob("*.md"))
        report_names = [path.name for path in reports[-12:]]
        blocked = [line.strip() for line in status.splitlines() if "BLOCKED" in line or "NOT_AVAILABLE" in line]
        return {
            "status_excerpt": status[-12000:],
            "plan_excerpt": plan[-12000:],
            "recent_reports": report_names,
            "blocked_items": blocked[-30:],
        }

    def build_plan(self) -> list[TaskSpec]:
        context = self.context()
        status = context["status_excerpt"]
        runner_needed = "Autonomous Research Runner" not in status or "NOT VERIFIED" in status
        prefix = "batch3" if runner_needed else "batch3-followup"
        runner_task = TaskSpec(
            key=f"{prefix}:runner-engineering",
            title="DeepUplift Autonomous Research Runner engineering validation",
            body=(
                "Implement and validate the persistent Hermes Supervisor, official Codex SDK worker, "
                "independent validator, heartbeat/checkpoint/recovery and launchd front-end. "
                "Use the repository documents as the acceptance contract."
            ),
            acceptance_command="python3 -m pytest -q tests/orchestration",
            evidence_class="ENGINEERING",
            prompt=(
                "Audit the current Autonomous Research Runner implementation in this repository. "
                "Run the focused orchestration tests and fix only concrete failures. Preserve all "
                "historical reports and do not touch raw data or credentials. Summarize commands and results."
            ),
        )
        experiment = TaskSpec(
            key=f"{prefix}:push-e2e-v2",
            title="Run Push semi-synthetic push_e2e_v2",
            body=(
                "Run the known-truth semi-synthetic Push benchmark with binary and three-arm actions, "
                "five fixed seeds, policy/OPE/optimizer diagnostics, and explicit limitations."
            ),
            dependencies=(runner_task.key,),
            acceptance_command="python3 -m pytest -q tests/benchmarks/test_push_e2e_v2.py",
            evidence_class="SEMI_SYNTHETIC_KNOWN_TRUTH",
            prompt=(
                "Run exactly `python3 scripts/run_push_e2e_v2.py --users 300 --horizon 4 --seeds 42 52 62 72 82 "
                "--output-json docs/benchmarks/BATCH3_PUSH_E2E_REPORT.json --output-md docs/benchmarks/BATCH3_PUSH_E2E_REPORT.md`. "
                "Use only synthetic data or already authorized local fixtures. Do not download the blocked "
                "myBPmyLife archive. Preserve historical outputs and report unsupported real-data and full "
                "sequential-DR gates honestly."
            ),
        )
        report = TaskSpec(
            key=f"{prefix}:independent-report",
            title="Independently validate and publish Batch 3 evidence",
            body=(
                "Read the Batch 3 code, tests and generated report in a read-only validation context; "
                "check provenance, metrics, seeds, constraints and unsupported claims, then update project status."
            ),
            dependencies=(experiment.key,),
            acceptance_command="python3 -m pytest -q tests/benchmarks/test_push_e2e_v2.py tests/orchestration",
            evidence_class="REVIEW",
            prompt=(
                "Act as an independent read-only reviewer. Inspect the Batch 3 report, code diff, test "
                "logs and project status. Run the focused tests without modifying code. Reject claims "
                "not supported by the evidence and return a concise verdict."
            ),
            sandbox="read-only",
        )
        # Keep the planner's evidence visible in the ledger; it is useful when
        # a later run explains why a blocked task was not scheduled.
        if context["blocked_items"]:
            return [runner_task, experiment, report]
        return [runner_task, experiment, report]

    def sync(self, *, run_id: str, board: str, ledger: RunLedger) -> list[TaskSpec]:
        specs = self.build_plan()
        hermes = HermesKanban(board)
        ids: dict[str, str] = {}
        for spec in specs:
            ids[spec.key] = hermes.create(
                spec.title,
                spec.body,
                key=f"deepuplift:{spec.key}",
                workspace=self.repo_root,
                created_by="hermes-autonomous-runner",
            )
            ledger.ensure_stage(
                run_id,
                spec.key,
                spec.dependencies,
                branch="codex/batch3-autonomous-runner",
                worktree=str(self.repo_root),
                config={"task_key": spec.key, "acceptance_command": spec.acceptance_command},
                dataset_fingerprint="synthetic-only-unless-provenance-gate-opens",
                task_key=spec.key,
                spec=asdict(spec),
            )
        for spec in specs:
            for dependency in spec.dependencies:
                hermes.link(ids[dependency], ids[spec.key])
        ledger.event(run_id, None, "planner_synced", task_ids=ids, context=self.context())
        return specs


class SingleInstanceLock:
    """Portable POSIX lock for the macOS foreground/launchd entry point."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"runner already active: {self.path}") from exc
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(f"pid={os.getpid()} started={utc_now()}\n")
        self.handle.flush()
        return self

    def __exit__(self, *_args):
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()


class AsyncCodexWorker:
    """Worker implementation using only the installed public async SDK API."""

    def __init__(self, *, sandbox: str = "workspace-write"):
        self.sandbox = sandbox

    async def run(
        self,
        prompt: str,
        *,
        cwd: Path,
        thread_id: str | None,
        heartbeat_path: Path,
        metadata_path: Path,
    ) -> dict[str, Any]:
        try:
            from openai_codex import AsyncCodex, Sandbox
            from openai_codex.errors import TransportClosedError
        except ImportError as exc:
            raise RuntimeError("official openai-codex SDK is required in the runner runtime") from exc
        heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
        sandbox_mode = Sandbox(self.sandbox)
        async with AsyncCodex() as codex:
            thread = (
                await codex.thread_resume(thread_id, cwd=str(cwd), sandbox=sandbox_mode)
                if thread_id
                else await codex.thread_start(cwd=str(cwd), sandbox=sandbox_mode)
            )
            metadata_path.write_text(json.dumps({"thread_id": thread.id, "pid": os.getpid()}) + "\n", encoding="utf-8")
            heartbeat_path.touch()
            handle = await thread.turn(prompt)
            event_count = 0
            stream_error = None
            try:
                async for event in handle.stream():
                    event_count += 1
                    heartbeat_path.touch()
                    payload = getattr(event, "payload", None)
                    turn = getattr(payload, "turn", None)
                    if turn is not None and getattr(turn, "id", None):
                        metadata_path.write_text(
                            json.dumps({"thread_id": thread.id, "turn_id": turn.id, "pid": os.getpid()}) + "\n",
                            encoding="utf-8",
                        )
            except TransportClosedError as exc:
                # The stream can close at terminal state.  Thread.read below
                # is the authoritative completion check.
                stream_error = str(exc)
            read = await thread.read(include_turns=True)
            turns = list(getattr(read.thread, "turns", []) or [])
            last_turn = turns[-1] if turns else None
            status = getattr(getattr(last_turn, "status", None), "value", None) or str(getattr(last_turn, "status", "unknown"))
            error = getattr(last_turn, "error", None)
            final_response = ""
            if last_turn is not None:
                for item in reversed(getattr(last_turn, "items", []) or []):
                    root = getattr(item, "root", item)
                    if getattr(root, "type", None) == "agentMessage":
                        final_response = getattr(root, "text", "") or ""
                        break
            result = {
                "thread_id": thread.id,
                "turn_id": getattr(last_turn, "id", None),
                "status": status,
                "response": final_response,
                "duration_ms": getattr(last_turn, "duration_ms", None),
                "error": str(error) if error else None,
                "stream_event_count": event_count,
                "stream_error": stream_error,
            }
            metadata_path.write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")
            heartbeat_path.touch()
            if status not in {"completed", "complete", "idle"}:
                raise RuntimeError(f"Codex turn did not complete: {result}")
            return result


def _worker_entry(args: argparse.Namespace) -> int:
    try:
        result = asyncio.run(
            AsyncCodexWorker(sandbox=args.sandbox).run(
                args.prompt,
                cwd=Path(args.cwd),
                thread_id=args.thread_id or None,
                heartbeat_path=Path(args.heartbeat),
                metadata_path=Path(args.metadata),
            )
        )
        Path(args.result).write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")
        return 0
    except BaseException as exc:  # persist failure evidence before process exit
        failure = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        Path(args.result).write_text(json.dumps(failure, sort_keys=True) + "\n", encoding="utf-8")
        return 1


class AutonomousResearchRunner:
    """Persistent supervisor with bounded retries and independent validation."""

    def __init__(
        self,
        *,
        run_id: str,
        repo_root: Path,
        ledger: RunLedger,
        board: str,
        state_root: Path,
        heartbeat_timeout: float = 120.0,
        poll_interval: float = 2.0,
    ):
        self.run_id = run_id
        self.repo_root = Path(repo_root).resolve()
        self.ledger = ledger
        self.hermes = HermesKanban(board)
        self.state_root = Path(state_root).expanduser()
        self.state_root.mkdir(parents=True, exist_ok=True)
        self.heartbeat_timeout = heartbeat_timeout
        self.poll_interval = poll_interval
        self.planner = ResearchPlanner(self.repo_root)
        self.validator = IndependentValidator()
        self.stop_requested = False
        self.task_ids: dict[str, str] = {}

    def plan(self) -> list[TaskSpec]:
        specs = self.planner.sync(run_id=self.run_id, board=self.hermes.board, ledger=self.ledger)
        # Resolve ids from the public board snapshot without touching Hermes DB.
        snapshot = self.hermes.list()
        self.task_ids = {
            spec.key: next(
                row["id"] for row in snapshot
                if row.get("title") == spec.title and row.get("created_by") == "hermes-autonomous-runner"
            )
            for spec in specs
        }
        self.ledger.event(self.run_id, None, "planner_task_ids_resolved", task_ids=self.task_ids)
        return specs

    def _spec(self, stage: dict[str, Any]) -> TaskSpec:
        raw = json.loads(stage["spec_json"] or "{}")
        return TaskSpec(**raw)

    def _ready(self, stage: dict[str, Any]) -> bool:
        if stage["status"] in TERMINAL:
            return False
        for dep in stage["dependency_ids"]:
            if self.ledger.get(self.run_id, dep)["status"] != "ACCEPTED":
                return False
        return stage["status"] in {"READY", "REPAIR_READY"}

    def _claim(self, stage: dict[str, Any]) -> None:
        task_id = self.task_ids[stage["phase_id"]]
        self.hermes.claim(task_id)
        now = utc_now()
        self.ledger.update(
            self.run_id,
            stage["phase_id"],
            status="RUNNING",
            started_at=now,
            heartbeat_at=now,
            attempt=stage["attempt"] + 1,
            worker_pid=None,
        )
        self.ledger.event(self.run_id, stage["phase_id"], "claimed", task_id=task_id)

    def _launch_worker(self, stage: dict[str, Any], spec: TaskSpec) -> tuple[subprocess.Popen[str], dict[str, Path]]:
        key = stage["phase_id"].replace("/", "_").replace(":", "_")
        paths = {
            "heartbeat": self.state_root / f"{key}.heartbeat",
            "metadata": self.state_root / f"{key}.worker.json",
            "result": self.state_root / f"{key}.result.json",
        }
        for path in paths.values():
            path.unlink(missing_ok=True)
        command = [
            os.environ.get("CODEX_PYTHON", sys.executable),
            "-m",
            "deepuplift.orchestration.autonomous_runner",
            "--worker",
            "--prompt",
            spec.prompt,
            "--cwd",
            str(self.repo_root),
            "--thread-id",
            stage["codex_thread_id"] or "",
            "--heartbeat",
            str(paths["heartbeat"]),
            "--metadata",
            str(paths["metadata"]),
            "--result",
            str(paths["result"]),
            "--sandbox",
            spec.sandbox,
        ]
        process = subprocess.Popen(command, cwd=self.repo_root, text=True)
        self.ledger.update(self.run_id, stage["phase_id"], worker_pid=process.pid)
        self.ledger.event(self.run_id, stage["phase_id"], "worker_started", pid=process.pid)
        return process, paths

    def _monitor(self, stage: dict[str, Any], process: subprocess.Popen[str], paths: dict[str, Path]) -> dict[str, Any] | None:
        phase = stage["phase_id"]
        task_id = self.task_ids[phase]
        while process.poll() is None and not self.stop_requested:
            now = time.time()
            if paths["heartbeat"].exists():
                age = now - paths["heartbeat"].stat().st_mtime
                if age > self.heartbeat_timeout:
                    self.hermes.comment(task_id, f"worker heartbeat stale for {age:.1f}s; supervisor will recover")
                    self.ledger.event(self.run_id, phase, "heartbeat_stale", age_seconds=age)
                    process.terminate()
                    break
                self.hermes.heartbeat(task_id, f"Codex worker alive; heartbeat_age={age:.1f}s")
                self.ledger.update(self.run_id, phase, heartbeat_at=utc_now())
                self.ledger.event(self.run_id, phase, "heartbeat", age_seconds=age)
            time.sleep(self.poll_interval)
        if self.stop_requested and process.poll() is None:
            process.terminate()
            return None
        if process.poll() != 0:
            self.ledger.event(self.run_id, phase, "worker_interrupted_or_failed", returncode=process.returncode)
            return None
        if not paths["result"].exists():
            return None
        result = json.loads(paths["result"].read_text(encoding="utf-8"))
        self.ledger.update(
            self.run_id,
            phase,
            codex_thread_id=result.get("thread_id") or stage["codex_thread_id"],
            codex_turn_id=result.get("turn_id"),
            heartbeat_at=utc_now(),
            worker_pid=None,
        )
        self.ledger.event(self.run_id, phase, "worker_finished", result={k: v for k, v in result.items() if k != "response"})
        return result

    def _validate(self, stage: dict[str, Any], spec: TaskSpec) -> ValidationResult:
        if not spec.acceptance_command:
            return ValidationResult(True, "no command validator configured", {})
        result = subprocess.run(
            spec.acceptance_command,
            cwd=self.repo_root,
            shell=True,
            text=True,
            capture_output=True,
            check=False,
        )
        details = {
            "command": spec.acceptance_command,
            "returncode": result.returncode,
            "stdout_tail": result.stdout[-4000:],
            "stderr_tail": result.stderr[-4000:],
        }
        return ValidationResult(result.returncode == 0, "acceptance command passed" if result.returncode == 0 else "acceptance command failed", details)

    def _finish(self, stage: dict[str, Any], spec: TaskSpec, validation: ValidationResult) -> None:
        phase = stage["phase_id"]
        if validation.passed:
            self.ledger.update(self.run_id, phase, status="ACCEPTED", validation_status="PASS", checkpoint="validated", last_error=None)
            self.ledger.event(self.run_id, phase, "validated", **validation.details)
            self.hermes.complete(self.task_ids[phase], validation.reason, {"validation": validation.details, "evidence_class": spec.evidence_class})
            return
        repair_count = stage["repair_count"] + 1
        self.ledger.update(self.run_id, phase, repair_count=repair_count, validation_status="REJECTED", last_error=validation.reason, checkpoint=f"validation_failed_{repair_count}")
        self.ledger.event(self.run_id, phase, "validation_rejected", repair_count=repair_count, **validation.details)
        if repair_count > spec.max_repair_cycles:
            self.ledger.update(self.run_id, phase, status="BLOCKED", validation_status="BLOCKED")
            self.ledger.event(self.run_id, phase, "retry_limit_reached", max_repair_cycles=spec.max_repair_cycles)
            self.hermes.block(self.task_ids[phase], f"automatic repair limit reached: {validation.reason}")
        else:
            self.ledger.update(self.run_id, phase, status="REPAIR_READY")
            with contextlib.suppress(Exception):
                self.hermes.reclaim(self.task_ids[phase], "validation failed; bounded repair requested")
            self.hermes.comment(self.task_ids[phase], f"REPAIR_READY cycle={repair_count}: {validation.reason}")

    def run_once(self) -> bool:
        progressed = False
        for stage in self.ledger.all_stages(self.run_id):
            if not self._ready(stage):
                continue
            spec = self._spec(stage)
            self._claim(stage)
            process, paths = self._launch_worker(stage, spec)
            result = self._monitor(stage, process, paths)
            if result is None:
                self.ledger.update(self.run_id, stage["phase_id"], status="READY", last_error="worker interrupted or failed")
                with contextlib.suppress(Exception):
                    self.hermes.reclaim(self.task_ids[stage["phase_id"]], "supervisor recovery after worker interruption")
            else:
                current = self.ledger.get(self.run_id, stage["phase_id"])
                self._finish(current, spec, self._validate(current, spec))
            progressed = True
        return progressed

    def run(self, *, max_cycles: int = 1) -> None:
        with SingleInstanceLock(self.state_root / "runner.lock"):
            for _ in range(max_cycles):
                if self.stop_requested:
                    break
                if not self.run_once():
                    break


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DeepUplift Autonomous Research Runner")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--prompt")
    parser.add_argument("--cwd", type=Path)
    parser.add_argument("--thread-id", default="")
    parser.add_argument("--heartbeat", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--sandbox", default="workspace-write")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.worker:
        required = (args.prompt, args.cwd, args.heartbeat, args.metadata, args.result)
        if any(value is None for value in required):
            raise SystemExit("worker mode requires --prompt, --cwd, --heartbeat, --metadata and --result")
        return _worker_entry(args)
    raise SystemExit("supervisor is started through scripts/autonomous_runner/start.sh")


if __name__ == "__main__":
    raise SystemExit(main())

