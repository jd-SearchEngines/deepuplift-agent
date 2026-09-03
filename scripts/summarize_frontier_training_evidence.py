from __future__ import annotations

import json
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


TARGET_DATASETS = {
    "synthetic_ads_full_funnel_ecup_7k": "Full-funnel ECUP synthetic training smoke",
    "synthetic_growth_delayed_feedback_7k": "Delayed-feedback synthetic training smoke",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize no-UI compare evidence for frontier synthetic datasets.")
    parser.add_argument("--allow-missing", action="store_true", help="Write a report even when frontier compare manifests are not present yet.")
    args = parser.parse_args()

    reports_dir = ROOT / "reports"
    manifests = sorted(reports_dir.glob("no_ui_compare_manifest_*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    latest_by_dataset: dict[str, dict] = {}
    for path in manifests:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        dataset_id = payload.get("dataset_id")
        if dataset_id in TARGET_DATASETS and dataset_id not in latest_by_dataset:
            payload["_manifest_path"] = str(path.relative_to(ROOT))
            latest_by_dataset[dataset_id] = payload

    failures = [dataset_id for dataset_id in TARGET_DATASETS if dataset_id not in latest_by_dataset]
    rows = []
    for dataset_id, payload in latest_by_dataset.items():
        ranking = payload.get("ranking") or []
        best = ranking[0] if ranking else {}
        best_run = next((run for run in payload.get("runs", []) if run.get("model") == best.get("model")), {})
        metrics = {}
        run_dir = ROOT / best_run.get("run_dir", "")
        metrics_path = run_dir / "metrics.json"
        run_note_path = run_dir / "run_note.md"
        if metrics_path.is_file():
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                metrics = {}
        full_funnel = metrics.get("full_funnel") or {}
        delayed_feedback = metrics.get("delayed_feedback") or {}
        metric_evidence = []
        if full_funnel.get("top_k"):
            metric_evidence.append(f"full_funnel_top_k={len(full_funnel.get('top_k') or [])}")
        if delayed_feedback.get("top_k"):
            metric_evidence.append(f"delayed_feedback_top_k={len(delayed_feedback.get('top_k') or [])}")
        rows.append(
            {
                "dataset_id": dataset_id,
                "label": TARGET_DATASETS[dataset_id],
                "dataset": payload.get("dataset"),
                "rows": payload.get("rows"),
                "models": ", ".join(payload.get("models", [])),
                "best_model": best.get("model"),
                "best_qini": best.get("qini"),
                "best_auuc": best.get("auuc"),
                "best_readiness": best.get("readiness_level"),
                "sensitivity": best.get("sensitivity_verdict"),
                "metric_evidence": ", ".join(metric_evidence) or "missing",
                "run_note_has_frontier_section": (
                    "yes"
                    if run_note_path.is_file()
                    and (
                        "Full-Funnel ECUP Metrics" in run_note_path.read_text(encoding="utf-8")
                        or "Delayed Feedback Uplift" in run_note_path.read_text(encoding="utf-8")
                    )
                    else "no"
                ),
                "run_dir": best_run.get("run_dir"),
                "manifest": payload.get("_manifest_path"),
            }
        )

    output = {
        "status": "fail" if failures else "ok",
        "rows": rows,
        "failures": failures,
    }
    out_json = reports_dir / "frontier_training_evidence_latest.json"
    out_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        "# Frontier Dataset Training Evidence",
        "",
        "This report records no-UI compare evidence for source-gated frontier datasets.",
        "",
        "| Dataset | Rows | Models | Best model | QINI | AUUC | Readiness | Sensitivity | Frontier metrics | Run note | Manifest |",
        "|---|---:|---|---|---:|---:|---|---|---|---|---|",
    ]
    for row in rows:
        md_lines.append(
            "| {dataset} | {rows} | {models} | {best_model} | {best_qini} | {best_auuc} | {best_readiness} | {sensitivity} | {metric_evidence} | {run_note_has_frontier_section} | `{manifest}` |".format(
                **row
            )
        )
    if failures:
        md_lines.append("")
        md_lines.append("Missing evidence: " + ", ".join(failures))
    out_md = ROOT / "docs" / "FRONTIER_DATASET_TRAINING_EVIDENCE.md"
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": output["status"], "json": str(out_json.relative_to(ROOT)), "markdown": str(out_md.relative_to(ROOT)), "rows": len(rows), "failures": failures}, ensure_ascii=False))
    if failures and not args.allow_missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
