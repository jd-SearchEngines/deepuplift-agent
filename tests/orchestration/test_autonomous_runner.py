from __future__ import annotations

import json
from pathlib import Path

import pytest

from deepuplift.orchestration.autonomous_runner import (
    AutonomousResearchRunner,
    ResearchPlanner,
    SingleInstanceLock,
    TaskSpec,
)
from deepuplift.orchestration.hermes_codex_sdk import RunLedger, ValidationResult


def test_planner_reads_status_and_emits_stable_dag(tmp_path: Path):
    (tmp_path / "docs/benchmarks").mkdir(parents=True)
    (tmp_path / "docs/PROJECT_STATUS.md").write_text("P5 PASS\nreal Push BLOCKED\n", encoding="utf-8")
    (tmp_path / "docs/CAUSAL_RESEARCH_EXECUTION_PLAN.md").write_text("Gate 3 push_e2e_v2\n", encoding="utf-8")
    planner = ResearchPlanner(tmp_path)
    first = planner.build_plan()
    second = planner.build_plan()
    assert [item.key for item in first] == [item.key for item in second]
    assert len(first) == 1
    assert first[0].key == "research:sequential-dr:v1"
    report = tmp_path / "docs/benchmarks/SEQUENTIAL_DR_REPORT.json"
    report.write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
    next_plan = planner.build_plan()
    assert [item.key for item in next_plan] == ["research:multi-arm-chain:v1"]
    assert next_plan[0].dependencies == (first[0].key,)
    assert "BLOCKED" in planner.context()["blocked_items"][0]


def test_ledger_migrates_runner_fields_and_preserves_idempotency(tmp_path: Path):
    ledger = RunLedger(tmp_path / "ledger.sqlite3")
    spec = TaskSpec(key="a", title="A", body="A")
    ledger.ensure_stage("run", "a", config={"x": 1}, task_key=spec.key, spec=vars(spec))
    ledger.ensure_stage("run", "a", config={"x": 2}, task_key=spec.key, spec=vars(spec))
    stage = ledger.get("run", "a")
    assert stage["status"] == "READY"
    assert stage["repair_count"] == 0
    assert json.loads(stage["spec_json"])["key"] == "a"


def test_single_instance_lock_rejects_second_owner(tmp_path: Path):
    path = tmp_path / "runner.lock"
    with SingleInstanceLock(path):
        with pytest.raises(RuntimeError, match="already active"):
            with SingleInstanceLock(path):
                pass


def test_validation_failure_is_bounded_and_becomes_blocked(tmp_path: Path, monkeypatch):
    ledger = RunLedger(tmp_path / "ledger.sqlite3")
    spec = TaskSpec(key="a", title="A", body="A", max_repair_cycles=2)
    ledger.ensure_stage("run", "a", spec=vars(spec))
    runner = AutonomousResearchRunner(
        run_id="run",
        repo_root=tmp_path,
        ledger=ledger,
        board="test-board",
        state_root=tmp_path / "state",
    )
    runner.task_ids["a"] = "task-a"
    monkeypatch.setattr(runner.hermes, "block", lambda *_args: None)
    monkeypatch.setattr(runner.hermes, "comment", lambda *_args: None)
    for expected_status in ("REPAIR_READY", "REPAIR_READY", "BLOCKED"):
        stage = ledger.get("run", "a")
        runner._finish(stage, spec, ValidationResult(False, "intentional failure", {"test": True}))
        assert ledger.get("run", "a")["status"] == expected_status


def test_runner_does_not_accept_unmet_dependency(tmp_path: Path):
    ledger = RunLedger(tmp_path / "ledger.sqlite3")
    ledger.ensure_stage("run", "a")
    ledger.ensure_stage("run", "b", ("a",), spec=vars(TaskSpec(key="b", title="B", body="B")))
    runner = AutonomousResearchRunner(
        run_id="run", repo_root=tmp_path, ledger=ledger, board="test-board", state_root=tmp_path / "state"
    )
    assert runner._ready(ledger.get("run", "b")) is False
