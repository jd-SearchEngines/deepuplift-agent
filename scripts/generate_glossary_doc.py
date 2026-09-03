from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepuplift.core.glossary import glossary_context_names, glossary_rows


def build_markdown() -> str:
    lines = [
        "# DeepUplift Glossary",
        "",
        "This glossary explains the terms that appear in the UI, benchmark reports and interview materials.",
        "",
        "## All Terms",
        "",
        "| Term | Category | Plain Meaning | Formula | Why It Matters | Common Pitfall | UI Location | Evidence |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in glossary_rows():
        lines.append(
            "| {term} | {category} | {short_cn} | `{formula}` | {why_it_matters} | {common_pitfall} | {ui_location} | {evidence} |".format(
                **row
            )
        )
    for context in glossary_context_names():
        lines.extend(
            [
                "",
                f"## Context: {context}",
                "",
                "| Term | Plain Meaning | Evidence |",
                "| --- | --- | --- |",
            ]
        )
        for row in glossary_rows(context):
            lines.append(f"| {row['term']} | {row['short_cn']} | {row['evidence']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the DeepUplift glossary markdown and JSON artifacts.")
    parser.add_argument("--output", default="docs/DEEPUplift_GLOSSARY.md")
    parser.add_argument("--json-output", default="reports/glossary_latest.json")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_markdown(), encoding="utf-8")

    json_output = Path(args.json_output)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps({"status": "ok", "terms": glossary_rows(), "contexts": glossary_context_names()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"status": "ok", "output": str(output), "json_output": str(json_output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
