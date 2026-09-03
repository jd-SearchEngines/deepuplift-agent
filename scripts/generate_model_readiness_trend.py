from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any


def load_audits(reports_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(reports_dir.glob("model_catalog_audit_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        rows.append(
            {
                "path": str(path),
                "mtime": path.stat().st_mtime,
                "generated_at": payload.get("generated_at") or "",
                "registered_models": int(payload.get("registered_models") or 0),
                "ready_models": int(payload.get("ready_models") or 0),
                "missing_descriptions": int(payload.get("missing_descriptions") or 0),
                "missing_model_cards": int(payload.get("missing_model_cards") or 0),
                "missing_model_presets": int(payload.get("missing_model_presets") or 0),
                "ready_source_counts": payload.get("ready_source_counts") or {},
                "ready_family_counts": payload.get("ready_family_counts") or {},
            }
        )
    rows.sort(key=lambda row: row["mtime"])
    return rows


def write_csv(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    sources = sorted({source for row in rows for source in row["ready_source_counts"]})
    families = sorted({family for row in rows for family in row["ready_family_counts"]})
    fieldnames = [
        "generated_at",
        "registered_models",
        "ready_models",
        "missing_descriptions",
        "missing_model_cards",
        "missing_model_presets",
        "path",
    ] + [f"source_{source}" for source in sources] + [f"family_{family}" for family in families]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat = {
                "generated_at": row["generated_at"],
                "registered_models": row["registered_models"],
                "ready_models": row["ready_models"],
                "missing_descriptions": row["missing_descriptions"],
                "missing_model_cards": row["missing_model_cards"],
                "missing_model_presets": row["missing_model_presets"],
                "path": row["path"],
            }
            flat.update({f"source_{source}": row["ready_source_counts"].get(source, 0) for source in sources})
            flat.update({f"family_{family}": row["ready_family_counts"].get(family, 0) for family in families})
            writer.writerow(flat)


def build_markdown(rows: list[dict[str, Any]], csv_path: Path, json_path: Path) -> str:
    generated_at = time.strftime("%Y-%m-%d %H:%M:%S %z")
    if not rows:
        return f"""# DeepUplift Agent Model Readiness Trend

Generated at: `{generated_at}`

No model catalog audits found.
"""
    first = rows[0]
    last = rows[-1]
    registered_delta = last["registered_models"] - first["registered_models"]
    ready_delta = last["ready_models"] - first["ready_models"]
    audit_rows = "\n".join(
        f"| {row['generated_at'] or 'NA'} | {row['registered_models']} | {row['ready_models']} | {row['missing_descriptions']} | {row['missing_model_cards']} | {row['missing_model_presets']} | `{row['path']}` |"
        for row in rows[-12:]
    )
    source_rows = "\n".join(
        f"| {source} | {count} |"
        for source, count in sorted(last["ready_source_counts"].items(), key=lambda item: (-item[1], item[0]))
    )
    family_rows = "\n".join(
        f"| {family} | {count} |"
        for family, count in sorted(last["ready_family_counts"].items(), key=lambda item: (-item[1], item[0]))
    )
    return f"""# DeepUplift Agent Model Readiness Trend

Generated at: `{generated_at}`

## Summary

| Metric | Value |
| --- | ---: |
| Audit snapshots | {len(rows)} |
| First registered models | {first["registered_models"]} |
| Latest registered models | {last["registered_models"]} |
| Registered delta | {registered_delta:+d} |
| First ready models | {first["ready_models"]} |
| Latest ready models | {last["ready_models"]} |
| Ready delta | {ready_delta:+d} |
| Latest missing descriptions | {last["missing_descriptions"]} |
| Latest missing model cards | {last["missing_model_cards"]} |
| Latest missing presets | {last["missing_model_presets"]} |

## Latest Ready Sources

| Source | Ready Models |
| --- | ---: |
{source_rows or "| NA | 0 |"}

## Latest Ready Families

| Family | Ready Models |
| --- | ---: |
{family_rows or "| NA | 0 |"}

## Recent Audit Snapshots

| Generated At | Registered | Ready | Missing Descriptions | Missing Cards | Missing Presets | Path |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
{audit_rows}

## Artifacts

- CSV: `{csv_path}`
- JSON: `{json_path}`

## Interview Talk Track

The model catalog is not a hard-coded claim. It is audited repeatedly, and the readiness trend shows how many models are registered, how many are runnable, which backends are ready, and whether model cards or presets are missing.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate model readiness trend reports from catalog audit snapshots.")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--csv-output", default="reports/model_readiness_trend_latest.csv")
    parser.add_argument("--json-output", default="reports/model_readiness_trend_latest.json")
    parser.add_argument("--markdown-output", default="docs/DEEPUplift_AGENT_MODEL_READINESS_TREND.md")
    args = parser.parse_args()

    rows = load_audits(Path(args.reports_dir))
    csv_path = Path(args.csv_output)
    json_path = Path(args.json_output)
    markdown_path = Path(args.markdown_output)
    write_csv(rows, csv_path)
    payload = {
        "status": "ok",
        "snapshots": len(rows),
        "latest": rows[-1] if rows else {},
        "first": rows[0] if rows else {},
        "registered_delta": (rows[-1]["registered_models"] - rows[0]["registered_models"]) if rows else 0,
        "ready_delta": (rows[-1]["ready_models"] - rows[0]["ready_models"]) if rows else 0,
        "csv": str(csv_path),
        "markdown": str(markdown_path),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(build_markdown(rows, csv_path, json_path), encoding="utf-8")
    print(json.dumps({"status": "ok", "snapshots": len(rows), "csv": str(csv_path), "json": str(json_path), "markdown": str(markdown_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
