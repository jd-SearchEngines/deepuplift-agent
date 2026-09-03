from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def latest(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def read_json(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def format_ready_sources(payload: dict) -> str:
    counts = payload.get("ready_source_counts") or {}
    if not counts:
        return "NA"
    return ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))


def build_snapshot(reports_dir: Path) -> tuple[str, dict]:
    audit_path = latest(reports_dir, "model_catalog_audit_*.json")
    regression_path = latest(reports_dir, "agent_regression_*.json")
    compare_path = latest(reports_dir, "no_ui_compare_manifest_*.json")
    screenshot_dir = latest(reports_dir, "ui_screenshots_*")

    audit = read_json(audit_path)
    regression = read_json(regression_path)
    compare = read_json(compare_path)
    regression_rows = regression.get("results") or []
    compare_rows = compare.get("ranking") or []
    ready_sources = format_ready_sources(audit)

    best_row = {}
    if regression_rows:
        best_row = max(
            regression_rows,
            key=lambda row: row.get("qini") if isinstance(row, dict) and row.get("qini") is not None else float("-inf"),
        )

    registered = audit.get("registered_models", "NA")
    ready = audit.get("ready_models", "NA")
    generated_at = time.strftime("%Y-%m-%d %H:%M:%S %z")
    proof = {
        "generated_at": generated_at,
        "registered_models": registered,
        "ready_models": ready,
        "ready_sources": audit.get("ready_source_counts") or {},
        "audit_path": str(audit_path) if audit_path else "",
        "regression_path": str(regression_path) if regression_path else "",
        "compare_path": str(compare_path) if compare_path else "",
        "screenshot_dir": str(screenshot_dir) if screenshot_dir else "",
        "regression_runs": len(regression_rows),
        "compare_candidates": len(compare_rows),
        "best_regression_model": best_row.get("model", ""),
        "best_regression_qini": best_row.get("qini"),
        "best_regression_auuc": best_row.get("auuc"),
    }

    best_metric = ""
    if best_row:
        best_metric = f"{best_row.get('model')} (QINI={best_row.get('qini'):.4f}, AUUC={best_row.get('auuc'):.4f})"
    else:
        best_metric = "NA"

    markdown = f"""# DeepUplift Agent Live Resume Snapshot

Generated at: `{generated_at}`

## Proof Metrics

| Metric | Value |
| --- | --- |
| Registered models | {registered} |
| Runnable models | {ready} |
| Ready backends | {ready_sources} |
| Latest regression runs | {len(regression_rows)} |
| Latest no-UI compare candidates | {len(compare_rows)} |
| Example regression winner | {best_metric} |
| Model audit | `{proof["audit_path"]}` |
| Regression report | `{proof["regression_path"]}` |
| Compare manifest | `{proof["compare_path"]}` |
| UI screenshot directory | `{proof["screenshot_dir"]}` |

## Resume Bullets With Live Numbers

- Built an end-to-end uplift/causal decision workbench for growth targeting, covering diagnostics, model recommendation, multi-model comparison, policy-value simulation, scoring export, evidence packaging, and regression validation.
- Designed a unified model registry, model-card, and parameter-preset system covering **{registered} uplift/causal models**, with **{ready} runnable locally** across `{ready_sources}`.
- Implemented an Agent workflow that automates Diagnostics -> Candidate Recommendation -> Model Compare -> Readiness Scoring -> Evidence Export.
- Added decision-grade evaluation: QINI, AUUC, Top-K uplift, policy value, calibration, overlap diagnostics, bootstrap CI, sensitivity checks, segment analysis, and decision-readiness scoring.
- Built reproducibility infrastructure including model catalog audit, preset audit, no-UI compare smoke, browser screenshot smoke, artifact validation, run notes, tags, evidence summaries, and downloadable evidence bundles.

## Interview Sound Bite

DeepUplift Agent turns uplift modeling from a model zoo into a causal decision workflow. The key design choice is that every model, from neural uplift networks to EconML and scikit-uplift adapters, must pass through the same registry, preset, trainer, evaluator, readiness, and artifact contracts before it can be compared or recommended.
"""
    return markdown, proof


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a resume-ready live snapshot from DeepUplift regression artifacts.")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--output", default="docs/RESUME_DEEPUplift_AGENT_SNAPSHOT.md")
    parser.add_argument("--check-doc", default="docs/RESUME_DEEPUplift_AGENT.md")
    parser.add_argument("--check", action="store_true", help="Fail if the main resume brief does not include the live registered/ready counts.")
    args = parser.parse_args()

    markdown, proof = build_snapshot(Path(args.reports_dir))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")

    if args.check:
        doc_path = Path(args.check_doc)
        doc = doc_path.read_text(encoding="utf-8") if doc_path.exists() else ""
        missing = [
            str(proof["registered_models"]),
            str(proof["ready_models"]),
        ]
        missing = [value for value in missing if value not in doc]
        if missing:
            raise AssertionError(f"{doc_path} is missing live resume counts: {missing}")

    print(json.dumps({"status": "ok", "output": str(output), **proof}, ensure_ascii=False))


if __name__ == "__main__":
    main()
