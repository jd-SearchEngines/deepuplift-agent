from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_DIFFICULTIES = [
    ("D01", "uplift"),
    ("D02", "Treatment"),
    ("D03", "Selection Bias"),
    ("D04", "registry"),
    ("D05", "Prediction"),
    ("D06", "Evaluation"),
    ("D07", "Policy"),
    ("D08", "Agent"),
    ("D09", "Evidence"),
    ("D10", "Model"),
    ("D11", "Guarded"),
    ("D12", "Production"),
    ("D13", "LLM"),
    ("D14", "Deep uplift loss"),
    ("D15", "Differentiable CFRNet"),
    ("D16", "Contrastive uplift"),
]


REQUIRED_FILES = [
    "docs/INTERVIEW_DIFFICULTIES_AND_SOLUTIONS.md",
    "docs/INTERVIEW_DEEPUplift_AGENT_DEEP_DIVE.md",
    "docs/AGENT_INTERVIEW_TRANSCRIPT_SAMPLE.md",
    "docs/DEEPUplift_AGENT_INTERVIEW_EVIDENCE_PACK.md",
    "docs/DEEPUplift_MODEL_DECONSTRUCTION_INDEX.md",
    "docs/INTERVIEW_QA_REHEARSAL_CN.md",
    "docs/DEEPUplift_AGENT_PRODUCTION_ROADMAP.md",
    "docs/DEEPUplift_AGENT_MODEL_BACKEND_STRATEGY.md",
    "docs/DEEPUplift_AGENT_MODEL_READINESS_TREND.md",
    "docs/DEEPUplift_AGENT_ONEPAGER_CN.md",
    "docs/INTERVIEW_DEMO_CHECKLIST_CN.md",
    "docs/DEEPUplift_AGENT_OFFLINE_ONLINE_PLAYBOOK.md",
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
    "docs/INDUSTRY_SOURCE_QUALITY_AUDIT.md",
    "docs/UPLIFT_INDUSTRY_SOURCE_ADOPTION_BACKLOG.md",
    "docs/UPLIFT_SCENARIO_MODEL_MATRIX.md",
    "docs/UPLIFT_SCENARIO_MODEL_MATRIX_INTERVIEW_QA.md",
    "docs/UPLIFT_AGENT_INTERVIEW_RUNBOOK.md",
    "docs/RESUME_DEEPUplift_AGENT.md",
    "docs/RESUME_BULLETS_CN.md",
    "docs/DEEPUplift_AGENT_ARCHITECTURE.md",
    "docs/DEEPUplift_AGENT_2_0_ARCHITECTURE.md",
    "docs/DEEPUplift_AGENT_EVIDENCE_INDEX.md",
]


REQUIRED_STORY_TEXT = [
    "Interview Difficulties",
    "Interview Deep Dive",
    "Interview Evidence Pack",
    "Interview Difficulty Proof Map",
    "Download complete interview pack ZIP",
    "Chinese Interview Q&A Rehearsal",
    "Production & Backend Tradeoffs",
    "Productionization Roadmap",
    "Model Backend Strategy",
    "Model Readiness Trend",
    "Guarded Backend Runtime Guide",
    "Chinese Project One-Pager",
    "Architecture Flow",
    "Interview Demo Checklist",
    "Chinese Interview Demo Checklist",
    "Offline-To-Online Validation Playbook",
    "Selection Bias / SSB Proxy",
    "Agent Interview Coach",
    "Agent Interview Transcript Sample",
    "Industry Mode",
    "Interview Runbook",
    "Industrial Case Studies",
    "Industry Source Quality Audit",
    "Industry Source Adoption Backlog",
    "Business Scenarios",
    "Interview Hard Questions",
    "Business Scenario Model Recommendation",
    "Coupon / Ads / Growth ROI Templates",
    "LLM + Uplift Frontier",
    "LLM Routing ROI Simulator",
    "LLM Interview Q&A",
    "Scenario ROI Helper",
    "Scenario Model Matrix",
    "Evidence Dashboard",
    "Regression Gate Artifacts",
]


def read(path: Path) -> str:
    if not path.exists():
        raise AssertionError(f"Missing required file: {path}")
    return path.read_text(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit DeepUplift interview materials for required hard-topic coverage.")
    parser.add_argument("--app", default="app.py")
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    docs = {path: read(Path(path)) for path in REQUIRED_FILES}
    difficulty_doc = docs["docs/INTERVIEW_DIFFICULTIES_AND_SOLUTIONS.md"]
    deep_dive_doc = docs["docs/INTERVIEW_DEEPUplift_AGENT_DEEP_DIVE.md"]
    app_text = read(Path(args.app))

    failures: list[str] = []
    rows = []
    for diff_id, keyword in REQUIRED_DIFFICULTIES:
        in_difficulty = diff_id in difficulty_doc and keyword.lower() in difficulty_doc.lower()
        in_deep_dive = diff_id in deep_dive_doc and keyword.lower() in deep_dive_doc.lower()
        rows.append(
            {
                "id": diff_id,
                "keyword": keyword,
                "difficulty_doc": in_difficulty,
                "deep_dive_doc": in_deep_dive,
            }
        )
        if not in_difficulty:
            failures.append(f"{diff_id} missing or weak in INTERVIEW_DIFFICULTIES_AND_SOLUTIONS.md")
        if not in_deep_dive:
            failures.append(f"{diff_id} missing or weak in INTERVIEW_DEEPUplift_AGENT_DEEP_DIVE.md")

    for phrase in REQUIRED_STORY_TEXT:
        if phrase not in app_text:
            failures.append(f"Story UI is missing phrase: {phrase}")

    payload = {
        "status": "fail" if failures else "ok",
        "required_files": REQUIRED_FILES,
        "difficulties": rows,
        "story_text": REQUIRED_STORY_TEXT,
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
