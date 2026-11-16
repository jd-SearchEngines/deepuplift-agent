#!/usr/bin/env python3
"""Run and persist the independent Batch 3 report validation gate."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from deepuplift.benchmarks.push_e2e_v2 import render_push_e2e_report


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    report_path = root / "docs/benchmarks/BATCH3_PUSH_E2E_REPORT.json"
    markdown_path = root / "docs/benchmarks/BATCH3_PUSH_E2E_REPORT.md"
    evidence_path = root / "docs/benchmarks/evidence/BATCH3_PUSH_E2E_VALIDATION.md"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    command = "python3 -m pytest -q tests/orchestration tests/benchmarks/test_push_e2e_v2.py"
    result = subprocess.run(command, cwd=root, shell=True, text=True, capture_output=True, check=False)
    report["gates"]["test_pass"] = "PASS" if result.returncode == 0 else "FAIL"
    report["gates"]["code_pass"] = "PASS" if report.get("status") in {"PASS", "PARTIAL"} else "FAIL"
    report["validation"] = {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    markdown_path.write_text(render_push_e2e_report(report), encoding="utf-8")
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(
        "# Batch 3 Validation Evidence\n\n"
        f"Command: `{command}`\n\n"
        f"Return code: `{result.returncode}`\n\n"
        "```text\n" + result.stdout + result.stderr + "```\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": report["gates"]["test_pass"], "returncode": result.returncode}))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())

