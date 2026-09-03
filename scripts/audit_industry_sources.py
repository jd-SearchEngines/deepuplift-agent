from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepuplift.core.industry_playbook import industry_references


TRUSTED_QUALITY = {
    "official_product_doc",
    "official_product_article",
    "official_tech_blog",
    "official_developer_article",
    "official_open_source",
    "official_research_paper",
    "official_product_help",
    "WWW_2024_paper",
    "ECML_PKDD_2024_industry_paper",
    "AAAI_2026_paper",
    "Stanford_GSB_working_paper",
    "JSAI_2020_paper",
    "journal_paper",
    "EJOR_2026_paper",
    "arXiv_paper",
    "arXiv_industrial_recommendation",
    "review_paper",
}

SECONDARY_ALLOWED = {"industry_talk_mirror"}


def quality_bucket(source_type: str) -> str:
    lowered = source_type.lower()
    if lowered.startswith("official"):
        return "official"
    if "journal" in lowered or "review" in lowered or any(token in lowered for token in ["www_", "ecml", "aaai", "jsai", "ejor", "pmlr"]):
        return "peer_reviewed_or_conference"
    if "arxiv" in lowered or "working_paper" in lowered:
        return "research_preprint"
    if "open_source" in lowered:
        return "open_source"
    if source_type in SECONDARY_ALLOWED:
        return "secondary_context"
    return "needs_review"


def check_url(url: str, timeout: int) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        method="GET",
        headers={"User-Agent": "DeepUplift-Agent-Source-Audit/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return {"ok": True, "status": response.status, "final_url": response.geturl()}
    except urllib.error.HTTPError as exc:
        return {"ok": 200 <= exc.code < 400, "status": exc.code, "final_url": url, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "status": None, "final_url": url, "error": str(exc)}


def table(rows: list[dict[str, Any]]) -> str:
    columns = ["id", "company", "source_type", "quality", "scenario", "url"]
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(col, "")).replace("\n", " ") for col in columns) + " |")
    return "\n".join([header, sep, *body])


def build_audit(check_network: bool, timeout: int) -> dict[str, Any]:
    refs = industry_references()
    failures: list[str] = []
    warnings: list[str] = []
    ids = [ref.get("id") for ref in refs]
    duplicates = [item for item, count in Counter(ids).items() if count > 1]
    if duplicates:
        failures.append(f"Duplicate reference ids: {duplicates}")
    if len(refs) < 18:
        failures.append(f"Need at least 18 industrial references, found {len(refs)}")

    audited_rows = []
    for ref in refs:
        missing = [key for key in ["id", "company", "source_type", "scenario", "title", "url", "lesson", "interview_hook"] if not ref.get(key)]
        if missing:
            failures.append(f"{ref.get('id', '<missing-id>')} missing fields: {missing}")
        source_type = str(ref.get("source_type", ""))
        quality = quality_bucket(source_type)
        if source_type not in TRUSTED_QUALITY and source_type not in SECONDARY_ALLOWED:
            failures.append(f"{ref.get('id')} has untrusted source_type: {source_type}")
        if quality == "secondary_context":
            warnings.append(f"{ref.get('id')} is secondary-context only; keep it paired with stronger evidence.")
        url = str(ref.get("url", ""))
        if not url.startswith("https://"):
            failures.append(f"{ref.get('id')} URL should use https: {url}")
        row = {**ref, "quality": quality}
        if check_network and url.startswith("https://"):
            row["network"] = check_url(url, timeout)
            if not row["network"].get("ok"):
                warnings.append(f"{ref.get('id')} network check did not pass: {row['network']}")
        audited_rows.append(row)

    quality_counts = Counter(row["quality"] for row in audited_rows)
    if quality_counts["official"] < 6:
        failures.append(f"Need at least 6 official sources, found {quality_counts['official']}")
    if quality_counts["peer_reviewed_or_conference"] + quality_counts["research_preprint"] < 8:
        failures.append("Need at least 8 paper/preprint references.")

    return {
        "status": "fail" if failures else "ok",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "references": len(refs),
        "quality_counts": dict(sorted(quality_counts.items())),
        "failures": failures,
        "warnings": warnings,
        "rows": audited_rows,
    }


def write_markdown(payload: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = payload.get("rows") or []
    quality_counts = payload.get("quality_counts") or {}
    quality_text = "\n".join(f"- `{key}`: {value}" for key, value in quality_counts.items())
    warnings = "\n".join(f"- {warning}" for warning in payload.get("warnings") or []) or "- None"
    output.write_text(
        f"""# Industry Source Quality Audit

Generated at: `{payload.get("generated_at")}`

This audit keeps the industrial uplift knowledge base defensible for interviews and demos. It accepts official docs, official engineering/research pages, peer-reviewed or conference papers, arXiv research, official open-source repositories, and clearly labeled secondary context.

## Summary

- Status: `{payload.get("status")}`
- References: `{payload.get("references")}`

## Quality Mix

{quality_text}

## Warnings

{warnings}

## Reference Table

{table(rows)}
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit source quality for industrial uplift references.")
    parser.add_argument("--json", default="reports/industry_source_audit_latest.json")
    parser.add_argument("--markdown", default="docs/INDUSTRY_SOURCE_QUALITY_AUDIT.md")
    parser.add_argument("--check-network", action="store_true")
    parser.add_argument("--timeout", type=int, default=8)
    args = parser.parse_args()

    payload = build_audit(check_network=args.check_network, timeout=args.timeout)
    json_path = Path(args.json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(payload, Path(args.markdown))
    print(json.dumps({key: payload[key] for key in ["status", "references", "quality_counts", "failures", "warnings"]}, ensure_ascii=False))
    if payload["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
