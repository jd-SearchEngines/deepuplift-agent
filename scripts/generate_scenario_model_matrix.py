from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepuplift.core.scenario_models import scenario_model_matrix
from scripts.audit_model_catalog import extract_model_descriptions


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def table(rows: list[dict[str, Any]], columns: list[str], limit: int = 40) -> str:
    view = rows[:limit]
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in view:
        body.append("| " + " | ".join(str(row.get(col, "")).replace("\n", " ") for col in columns) + " |")
    suffix = f"\n\nShowing {len(view)} of {len(rows)} models. See CSV/JSON for the complete matrix." if len(rows) > limit else ""
    return "\n".join([header, sep, *body]) + suffix


def build_markdown(generated_at: str, rows: list[dict[str, Any]]) -> str:
    ready = [row for row in rows if row["status"] == "ready"]
    scenario_counts: dict[str, int] = {}
    for row in rows:
        for tag in str(row["scenario_tags"]).split(";"):
            tag = tag.strip()
            if tag:
                scenario_counts[tag] = scenario_counts.get(tag, 0) + 1
    scenario_rows = [{"scenario_tag": key, "models": value} for key, value in sorted(scenario_counts.items())]
    return f"""# Uplift Scenario Model Matrix

Generated at: `{generated_at}`

This matrix connects the model catalog to business scenarios. It is designed for interview and review settings where the key question is not "how many models are registered?" but "which models should be considered for coupon, ads, growth, recommendation, or marketplace decisions, and why?"

## Summary

- Registered models: `{len(rows)}`
- Ready models: `{len(ready)}`
- Scenario tags: `{len(scenario_counts)}`

## Scenario Tag Coverage

{table(scenario_rows, ["scenario_tag", "models"], limit=20)}

## Model Matrix Preview

{table(rows, ["model", "status", "source", "family", "stage", "tasks", "scenario_tags", "recommended_preset"], limit=45)}

## How To Use In An Interview

1. Start from the business scenario: coupon, ads, growth, recommendation, marketplace.
2. Filter by `scenario_tags`.
3. Keep only `ready` models for live demo; mention guarded/optional models as designed extension points.
4. Explain tradeoffs via `stage`, `assumptions`, `risks`, and `decision_use`.
5. Move from model selection to Compare, Policy Value, and online incrementality validation.
"""


def main() -> None:
    generated_at = time.strftime("%Y-%m-%d %H:%M:%S %z")
    descriptions = extract_model_descriptions(Path("app.py"))
    rows = scenario_model_matrix(descriptions)
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reports_dir / "scenario_model_matrix_latest.csv"
    json_path = reports_dir / "scenario_model_matrix_latest.json"
    markdown_path = Path("docs/UPLIFT_SCENARIO_MODEL_MATRIX.md")
    write_csv(csv_path, rows)
    payload = {"status": "ok", "generated_at": generated_at, "rows": rows}
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(build_markdown(generated_at, rows), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "ok",
                "rows": len(rows),
                "csv": str(csv_path),
                "json": str(json_path),
                "markdown": str(markdown_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
