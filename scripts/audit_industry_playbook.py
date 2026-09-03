from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepuplift.core.industry_playbook import business_scenarios, industry_references


REQUIRED_DOCS = [
    "docs/INDUSTRIAL_UPLIFT_CASE_STUDIES.md",
    "docs/UPLIFT_BUSINESS_SCENARIOS.md",
    "docs/UPLIFT_INTERVIEW_INDUSTRY_DEEP_DIVE.md",
    "docs/UPLIFT_MODEL_SELECTION_PLAYBOOK.md",
    "docs/UPLIFT_COUPON_ADS_GROWTH_PLAYBOOK.md",
    "docs/UPLIFT_LLM_CAUSAL_FRONTIER.md",
    "docs/LLM_ROUTING_POLICY_PLAYBOOK.md",
    "docs/UPLIFT_LLM_INTERVIEW_QA.md",
    "docs/UPLIFT_ONLINE_EXPERIMENT_AND_INCREMENTALITY.md",
    "docs/UPLIFT_INDUSTRY_REFERENCES.md",
]

REQUIRED_APP_TEXT = [
    "Industrial Case Studies",
    "Industry Mode",
    "Business Scenarios",
    "Interview Hard Questions",
    "Business Scenario Model Recommendation",
    "Coupon / Ads / Growth ROI Templates",
    "LLM + Uplift Frontier",
    "LLM Routing ROI Simulator",
    "LLM Interview Q&A",
]

REQUIRED_SCENARIOS = ["coupon", "ads", "growth", "crm", "recommendation", "marketplace", "llm_routing"]
REQUIRED_COMPANIES = ["Tencent Ads", "Meituan", "Alibaba Entertainment", "DiDi", "ByteDance", "Uber", "Google", "Amazon", "Microsoft Research", "LMSYS RouteLLM"]


def read(path: Path) -> str:
    if not path.exists():
        raise AssertionError(f"Missing required file: {path}")
    return path.read_text(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit industrial uplift playbook coverage.")
    parser.add_argument("--app", default="app.py")
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    failures: list[str] = []
    docs = {}
    for doc in REQUIRED_DOCS:
        try:
            docs[doc] = read(Path(doc))
        except AssertionError as exc:
            failures.append(str(exc))

    app_text = read(Path(args.app))
    for phrase in REQUIRED_APP_TEXT:
        if phrase not in app_text:
            failures.append(f"App is missing phrase: {phrase}")

    scenario_ids = {row["id"] for row in business_scenarios()}
    for scenario in REQUIRED_SCENARIOS:
        if scenario not in scenario_ids:
            failures.append(f"Missing business scenario: {scenario}")

    companies = {row["company"] for row in industry_references()}
    for company in REQUIRED_COMPANIES:
        if company not in companies:
            failures.append(f"Missing company/source: {company}")

    if len(industry_references()) < 12:
        failures.append("Need at least 12 industrial references.")

    combined_docs = "\n".join(docs.values()).lower()
    for keyword in ["coupon", "pctr", "pcvr", "qini", "auuc", "policy value", "roi", "selection bias", "multi-treatment", "dose-response", "llm routing", "text treatment", "persuadable queries", "budget pacing"]:
        if keyword not in combined_docs:
            failures.append(f"Industry docs missing keyword: {keyword}")

    payload = {
        "status": "fail" if failures else "ok",
        "references": len(industry_references()),
        "scenarios": sorted(scenario_ids),
        "companies": sorted(companies),
        "required_docs": REQUIRED_DOCS,
        "failures": failures,
    }
    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
