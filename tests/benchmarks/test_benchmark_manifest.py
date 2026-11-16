from __future__ import annotations

import json
from pathlib import Path


def test_long_term_research_reports_have_explicit_evidence_gates():
    root = Path(__file__).resolve().parents[2]
    sequential = json.loads((root / "docs/benchmarks/SEQUENTIAL_DR_REPORT.json").read_text(encoding="utf-8"))
    multiarm = json.loads((root / "docs/benchmarks/MULTIARM_PUSH_CHAIN_REPORT.json").read_text(encoding="utf-8"))
    assert sequential["status"] == "PASS"
    assert sequential["gates"]["real_mrt"].startswith("BLOCKED")
    assert sequential["gates"]["online_lift"] == "NOT_RUN"
    assert multiarm["status"] == "PASS"
    assert multiarm["gates"]["three_arm_logged_probability_ope"] == "PASS"
