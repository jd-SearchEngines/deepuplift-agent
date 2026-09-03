from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepuplift.core.artifacts import save_json
from deepuplift.core.evidence_store import build_evidence_store, evidence_store_frame, latest_run_diff


def write_markdown(path: Path, payload: dict, run_diff: dict) -> None:
    records = payload.get("records") or []
    summary = payload.get("summary") or {}
    lines = [
        "# DeepUplift Experiment Evidence Store",
        "",
        f"Generated at: `{time.strftime('%Y-%m-%d %H:%M:%S %z')}`",
        "",
        "## Summary",
        "",
        f"- Evidence manifests indexed: {summary.get('manifest_count')}",
        f"- Models represented: {summary.get('models')}",
        f"- Readiness levels: `{summary.get('ready_levels')}`",
        f"- Promotion levels: `{summary.get('promotion_levels')}`",
        "",
        "## Latest Run Diff",
        "",
        f"- Status: `{run_diff.get('status')}`",
    ]
    diff = run_diff.get("diff") or {}
    if diff:
        lines.extend(
            [
                f"- Previous run: `{diff.get('previous_run_id')}` ({diff.get('previous_model')})",
                f"- Current run: `{diff.get('current_run_id')}` ({diff.get('current_model')})",
                f"- Same data hash: `{diff.get('same_data_hash')}`",
                f"- Readiness: `{diff.get('readiness_transition')}`",
                f"- Promotion: `{diff.get('promotion_transition')}`",
                "",
                "| Metric | Previous | Current | Delta |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for metric, row in (diff.get("metric_deltas") or {}).items():
            lines.append(f"| {metric} | {row.get('previous')} | {row.get('current')} | {row.get('delta')} |")
    lines.extend(
        [
            "",
            "## Latest Evidence Records",
            "",
            "| Run | Model | Readiness | Promotion | QINI | AUUC | Data Rows | Manifest |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in records[:30]:
        lines.append(
            "| {run} | {model} | {readiness} | {promotion} | {qini} | {auuc} | {rows} | `{manifest}` |".format(
                run=row.get("run_id"),
                model=row.get("model"),
                readiness=row.get("readiness_level"),
                promotion=row.get("promotion_level"),
                qini=row.get("qini_score"),
                auuc=row.get("auuc_score"),
                rows=row.get("data_rows"),
                manifest=row.get("manifest_path"),
            )
        )
    lines.extend(
        [
            "",
            "## Review Use",
            "",
            "Use this store as the bridge from local run folders to an industrial experiment registry.",
            "Each row is backed by an `evidence_manifest.json` containing data fingerprint, environment, artifact hashes, readiness, and promotion evidence.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Index DeepUplift evidence manifests into a lightweight experiment store.")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--json-output", default="reports/evidence_store_latest.json")
    parser.add_argument("--csv-output", default="reports/evidence_store_latest.csv")
    parser.add_argument("--diff-output", default="reports/evidence_run_diff_latest.json")
    parser.add_argument("--markdown-output", default="docs/DEEPUplift_EXPERIMENT_EVIDENCE_STORE.md")
    args = parser.parse_args()

    payload = build_evidence_store(ROOT, limit=args.limit)
    diff = latest_run_diff(payload)
    json_path = ROOT / args.json_output
    csv_path = ROOT / args.csv_output
    diff_path = ROOT / args.diff_output
    md_path = ROOT / args.markdown_output
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(json_path, payload)
    save_json(diff_path, diff)
    evidence_store_frame(payload).to_csv(csv_path, index=False)
    write_markdown(md_path, payload, diff)
    print(
        json.dumps(
            {
                "status": "ok",
                "json": args.json_output,
                "csv": args.csv_output,
                "diff": args.diff_output,
                "markdown": args.markdown_output,
                "manifest_count": payload.get("summary", {}).get("manifest_count"),
                "run_diff_status": diff.get("status"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
