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

    def _report_passed(self, relative: str) -> bool:
        path = self.repo_root / relative
        if not path.exists():
            return False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        return payload.get("status") == "PASS" or payload.get("gate_status") == "PASS"

    def build_plan(self, existing: dict[str, dict[str, Any]] | None = None) -> list[TaskSpec]:
        """Select the next eligible research task from evidence, not a fixed batch."""
        existing = existing or {}
        sequential = TaskSpec(
            key="research:sequential-dr:v1",
            title="Implement and evaluate Sequential DR continuation values",
            body="Add a continuation-value model and a true sequential doubly robust estimator on known-truth trajectories. Keep ordinary per-decision DR labelled separately.",
            acceptance_command="python3 -m pytest -q tests/decision/test_sequential_dr.py",
            evidence_class="SEMI_SYNTHETIC_KNOWN_TRUTH",
            prompt=(
                "Run exactly `python3 scripts/run_sequential_dr_benchmark.py --output-json docs/benchmarks/SEQUENTIAL_DR_REPORT.json --output-md docs/benchmarks/SEQUENTIAL_DR_REPORT.md`. "
                "Use only the known-truth synthetic trajectory DGP. Verify the continuation-value estimator, compare IPS/SNIPS/ordinary per-decision DR/true sequential DR, and record ESS, weight tails, cluster bootstrap and coverage. Do not claim real MRT or online lift."
            ),
        )
        multiarm = TaskSpec(
            key="research:multi-arm-chain:v1",
            title="Complete the three-arm treatment-to-policy chain",
            body="Connect the existing three-arm effect models to candidate construction, constrained optimization, multi-arm OPE and policy evaluation.",
            dependencies=(sequential.key,),
            acceptance_command="python3 -m pytest -q tests/benchmarks/test_push_multiarm_chain.py",
            evidence_class="SEMI_SYNTHETIC_KNOWN_TRUTH",
            prompt=(
                "Run exactly `python3 scripts/run_push_multiarm_chain.py --output-json docs/benchmarks/MULTIARM_PUSH_CHAIN_REPORT.json --output-md docs/benchmarks/MULTIARM_PUSH_CHAIN_REPORT.md`. "
                "Use known-truth synthetic data, actual logged arm probabilities, the existing candidate/optimizer APIs and cluster bootstrap. Keep three-arm integrated evidence separate from the earlier one-vs-control comparison."
            ),
        )
        optimizer = TaskSpec(
            key="research:optimizer-evidence:v1",
            title="Complete optimizer convergence and dual-bound evidence",
            body="Compare constrained solvers against exact small references with valid feasibility, stopping and dual-bound diagnostics; do not claim a strict gap without a valid bound.",
            dependencies=(multiarm.key,),
            acceptance_command="python3 -m pytest -q tests/decision/test_optimizer_evidence.py",
            evidence_class="SYNTHETIC_SCALE_STRESS",
            prompt="Audit the existing optimizer evidence and run the pre-registered small-reference and scale diagnostics. Add only mathematically valid convergence or dual-bound evidence, preserving negative results.",
        )
        matrix = TaskSpec(
            key="research:benchmark-matrix:v1",
            title="Build the unified P1-P9 and Push P0-P6 benchmark matrix",
            body="Reconcile every phase status, evidence class, seed/config, report and blocker into a machine-readable matrix.",
            dependencies=(optimizer.key,),
            acceptance_command="python3 -m pytest -q tests/benchmarks/test_benchmark_manifest.py",
            evidence_class="RESEARCH_EVIDENCE_INDEX",
            prompt="Build and validate the unified benchmark matrix from existing reports. Do not rerun completed experiments or convert BLOCKED/NOT_AVAILABLE evidence into PASS.",
        )
        candidates = (sequential, multiarm, optimizer, matrix)
        report_paths = {
            sequential.key: "docs/benchmarks/SEQUENTIAL_DR_REPORT.json",
            multiarm.key: "docs/benchmarks/MULTIARM_PUSH_CHAIN_REPORT.json",
        }
        for spec in candidates:
            stage = existing.get(spec.key)
            if stage and stage["status"] in TERMINAL:
                continue
            if spec.key in report_paths and self._report_passed(report_paths[spec.key]):
                continue
            return [spec]
        return []

    def sync(self, *, run_id: str, board: str, ledger: RunLedger) -> list[TaskSpec]:
        existing = {stage["phase_id"]: stage for stage in ledger.all_stages(run_id)}
        specs = self.build_plan(existing)
        hermes = HermesKanban(board)
        ids: dict[str, str] = {
            phase_id: str(stage["hermes_task_id"])
            for phase_id, stage in existing.items()
            if stage.get("hermes_task_id")
        }
        for spec in specs:
            for dependency in spec.dependencies:
                if dependency not in existing:
                    report_path = {
                        "research:sequential-dr:v1": "docs/benchmarks/SEQUENTIAL_DR_REPORT.json",
                        "research:multi-arm-chain:v1": "docs/benchmarks/MULTIARM_PUSH_CHAIN_REPORT.json",
                    }.get(dependency)
                    if report_path and self._report_passed(report_path):
                        ledger.ensure_stage(run_id, dependency, (), task_key=dependency)
                        ledger.update(run_id, dependency, status="ACCEPTED", validation_status="PASS", checkpoint="report_reconciled")
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
                branch=subprocess.check_output(["git", "branch", "--show-current"], cwd=self.repo_root, text=True).strip(),
                worktree=str(self.repo_root),
                config={"task_key": spec.key, "acceptance_command": spec.acceptance_command},
                dataset_fingerprint="synthetic-only-unless-provenance-gate-opens",
                task_key=spec.key,
                spec=asdict(spec),
            )
            ledger.update(run_id, spec.key, hermes_task_id=ids[spec.key])
        for spec in specs:
            for dependency in spec.dependencies:
                # A dependency may have been accepted in an earlier planner
                # invocation. Its public Hermes id is persisted in our ledger;
                # never reach into Hermes' private database to rediscover it.
                dependency_id = ids.get(dependency)
                if dependency_id and ids.get(spec.key):
                    hermes.link(dependency_id, ids[spec.key])
        ledger.event(run_id, None, "planner_synced", task_ids=ids, context=self.context(), selected_keys=[spec.key for spec in specs])
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
        resolved = {
            spec.key: next(
                row["id"] for row in snapshot
                if row.get("title") == spec.title and row.get("created_by") == "hermes-autonomous-runner"
            )
            for spec in specs
        }
        for stage in self.ledger.all_stages(self.run_id):
            if stage.get("hermes_task_id"):
                resolved[stage["phase_id"]] = stage["hermes_task_id"]
        self.task_ids = resolved
        self.ledger.event(self.run_id, None, "planner_task_ids_resolved", task_ids=self.task_ids)
        return specs

    def _spec(self, stage: dict[str, Any]) -> TaskSpec:
        raw = json.loads(stage["spec_json"] or "{}")
        return TaskSpec(**raw)

    def _worker_paths(self, phase: str) -> dict[str, Path]:
        key = phase.replace("/", "_").replace(":", "_")
        return {
            "heartbeat": self.state_root / f"{key}.heartbeat",
            "metadata": self.state_root / f"{key}.worker.json",
            "result": self.state_root / f"{key}.result.json",
        }

    @staticmethod
    def _pid_alive(pid: int | None) -> bool:
        if not pid:
            return False
        try:
            os.kill(int(pid), 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

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
        paths = self._worker_paths(stage["phase_id"])
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

    def _recover_running(self) -> bool:
        """Adopt an orphaned Worker after Supervisor restart.

        The process ID and deterministic checkpoint paths are project-owned
        metadata. If the worker is still alive, the new supervisor waits for
        it instead of starting a duplicate SDK turn. If it has exited, the
        result file is consumed and independently validated. A missing result
        is routed through the same bounded repair policy as any other failure.
        """
        progressed = False
        for stage in self.ledger.all_stages(self.run_id):
            if stage["status"] != "RUNNING" or stage["phase_id"] not in self.task_ids:
                continue
            phase = stage["phase_id"]
            spec = self._spec(stage)
            paths = self._worker_paths(phase)
            if paths["metadata"].exists():
                with contextlib.suppress(OSError, ValueError):
                    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
                    if metadata.get("thread_id") and metadata.get("thread_id") != stage.get("codex_thread_id"):
                        self.ledger.update(self.run_id, phase, codex_thread_id=metadata["thread_id"], heartbeat_at=utc_now())
                        stage = self.ledger.get(self.run_id, phase)
                        self.ledger.event(self.run_id, phase, "recovery_thread_checkpoint", thread_id=metadata["thread_id"])
            pid = stage.get("worker_pid")
            if self._pid_alive(pid):
                self.ledger.event(self.run_id, phase, "recovery_adopted_worker", pid=pid)
                while self._pid_alive(pid) and not self.stop_requested:
                    now = time.time()
                    if paths["heartbeat"].exists():
                        age = now - paths["heartbeat"].stat().st_mtime
                        self.ledger.update(self.run_id, phase, heartbeat_at=utc_now())
                        if age > self.heartbeat_timeout:
                            self.ledger.event(self.run_id, phase, "recovery_heartbeat_stale", age_seconds=age, pid=pid)
                            with contextlib.suppress(ProcessLookupError):
                                os.kill(int(pid), signal.SIGTERM)
                            break
                    time.sleep(self.poll_interval)
                if self.stop_requested:
                    return progressed
            result = None
            if paths["result"].exists():
                with contextlib.suppress(OSError, ValueError):
                    result = json.loads(paths["result"].read_text(encoding="utf-8"))
            self.ledger.update(
                self.run_id,
                phase,
                codex_thread_id=(result or {}).get("thread_id") or stage.get("codex_thread_id"),
                codex_turn_id=(result or {}).get("turn_id"),
                worker_pid=None,
                heartbeat_at=utc_now(),
            )
            self.ledger.event(self.run_id, phase, "recovery_result_consumed", result={k: v for k, v in (result or {}).items() if k != "response"})
            if result and result.get("status") in {"completed", "complete", "idle"}:
                validation = self._validate(self.ledger.get(self.run_id, phase), spec)
                self._finish(self.ledger.get(self.run_id, phase), spec, validation)
            else:
                self._finish(
                    self.ledger.get(self.run_id, phase),
                    spec,
                    ValidationResult(False, "worker exited without a completed result", {"recovered": True, "result": result}),
                )
            progressed = True
        return progressed

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
            if not self._ready(stage) or stage["phase_id"] not in self.task_ids:
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
            cycles = 0
            self.plan()
            while max_cycles == 0 or cycles < max_cycles:
                if self.stop_requested:
                    break
                progressed = self._recover_running()
                if not progressed:
                    progressed = self.run_once()
                # Re-plan after every task transition and also on idle. This
                # is the key difference from the bounded Batch 3 runner: an
                # accepted phase can unlock a new research phase without a
                # human or process restart.
                self.plan()
                cycles += 1
                if not progressed:
                    self.ledger.event(self.run_id, None, "planner_idle", next_plan=self.planner.context())
                    if max_cycles > 0:
                        break
                    time.sleep(max(self.poll_interval, 5.0))


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

