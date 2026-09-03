from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepuplift.core.scenario_models import scenario_model_matrix
from scripts.audit_model_catalog import extract_model_descriptions


REQUIRED_TAGS = [
    "Coupon / subsidy allocation",
    "Ads / audience targeting / bidding",
    "Growth / recall / push / SMS",
    "CRM lifecycle intervention",
    "Recommendation intervention",
    "Marketplace pricing / incentive",
    "LLM routing / cost-quality escalation",
    "Industrial tabular",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit scenario-to-model matrix coverage.")
    parser.add_argument("--app", default="app.py")
    parser.add_argument("--json", default="reports/scenario_model_matrix_audit_latest.json")
    parser.add_argument("--min-models", type=int, default=90)
    parser.add_argument("--min-ready", type=int, default=35)
    args = parser.parse_args()

    descriptions = extract_model_descriptions(Path(args.app))
    rows = scenario_model_matrix(descriptions)
    failures: list[str] = []
    if len(rows) < args.min_models:
        failures.append(f"Too few matrix rows: {len(rows)} < {args.min_models}")
    ready = [row for row in rows if row["status"] == "ready"]
    if len(ready) < args.min_ready:
        failures.append(f"Too few ready rows: {len(ready)} < {args.min_ready}")

    missing_tags = [row["model"] for row in rows if not str(row.get("scenario_tags", "")).strip()]
    if missing_tags:
        failures.append(f"Models missing scenario tags: {missing_tags[:20]}")

    tag_counts: Counter[str] = Counter()
    for row in rows:
        for tag in str(row.get("scenario_tags", "")).split(";"):
            tag = tag.strip()
            if tag:
                tag_counts[tag] += 1
    for tag in REQUIRED_TAGS:
        if tag_counts[tag] == 0:
            failures.append(f"Missing required scenario tag: {tag}")

    payload = {
        "status": "fail" if failures else "ok",
        "models": len(rows),
        "ready_models": len(ready),
        "tag_counts": dict(sorted(tag_counts.items())),
        "failures": failures,
    }
    output = Path(args.json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
