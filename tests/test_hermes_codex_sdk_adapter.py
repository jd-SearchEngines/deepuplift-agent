from __future__ import annotations

import sqlite3

from deepuplift.orchestration.hermes_codex_sdk import IndependentValidator, RunLedger


def test_ledger_persists_checkpoint_and_events(tmp_path):
    ledger = RunLedger(tmp_path / "run.sqlite")
    ledger.ensure_stage("run", "A", config={"seed": 1})
    ledger.update("run", "A", status="RUNNING", checkpoint="started", attempt=1)
    ledger.event("run", "A", "checkpoint", value="started")
    assert ledger.get("run", "A")["checkpoint"] == "started"
    assert ledger.events("run")[0]["event"] == "checkpoint"
    with sqlite3.connect(tmp_path / "run.sqlite") as db:
        assert db.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"


def test_validator_rejects_then_accepts(tmp_path):
    validator = IndependentValidator()
    marker = tmp_path / "marker.txt"
    marker.write_text("WRONG\n", encoding="utf-8")
    rejected = validator.marker(marker, "FIXED")
    assert not rejected.passed
    marker.write_text("FIXED\n", encoding="utf-8")
    assert validator.marker(marker, "FIXED").passed
