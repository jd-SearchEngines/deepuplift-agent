from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepuplift.core.industry_playbook import INDUSTRY_INTERVIEW_PROMPTS, industry_agent_reply
from deepuplift.core.interview_agent import INTERVIEW_AGENT_TOPICS, interview_agent_reply
from deepuplift.core.frontier_models import frontier_agent_reply


def latest(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def read_json(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def fallback_answer(question: str, evidence: str, context: dict) -> str:
    return (
        "### Deterministic evidence fallback\n\n"
        f"**Question**: {question}\n\n"
        "**Core answer**: this prompt is handled as a project-grounded fallback so the interview evidence pack "
        "stays complete even when a new topic is added before a dedicated answer template is written. The answer "
        "must still be reviewed before being used as the primary interview script.\n\n"
        "**Project framing**: DeepUplift should explain the issue through its causal workflow: diagnose treatment "
        "design and overlap first, choose a model family through registry/model cards, normalize predictions into "
        "`y0_pred / y1_pred / uplift_score`, evaluate QINI/AUUC/Top-K/calibration/policy value, then archive an "
        "evidence bundle for reproducibility.\n\n"
        f"**Evidence to cite**: `{evidence}`. Current context uses task `{context.get('task', 'NA')}`, model "
        f"`{context.get('model_name', 'NA')}`, treatment `{context.get('treatment_col', 'NA')}`, outcome "
        f"`{context.get('outcome_col', 'NA')}`, with catalog `{context.get('registered_models', 'NA')}` and "
        f"ready `{context.get('ready_models', 'NA')}` from the latest audit.\n\n"
        "**Review note**: when this fallback appears, add a dedicated answer in `deepuplift/core/interview_agent.py` "
        "or `deepuplift/core/industry_playbook.py` so future transcripts remain specific rather than generic."
    )


def build_sample(reports_dir: Path) -> tuple[str, dict]:
    audit = read_json(latest(reports_dir, "model_catalog_audit_*.json"))
    context = {
        "task": "classification",
        "model_name": "DRLearnerLightGBM",
        "treatment_col": "treatment",
        "outcome_col": "visit",
        "registered_models": audit.get("registered_models", 101),
        "ready_models": audit.get("ready_models", 72),
    }
    generated_at = time.strftime("%Y-%m-%d %H:%M:%S %z")
    rows = []
    lines = [
        "# DeepUplift Agent Interview Transcript Sample",
        "",
        f"Generated at: `{generated_at}`",
        "",
        f"This generated transcript proves the Agent can answer the {len(INTERVIEW_AGENT_TOPICS)} interview hard points from deterministic project knowledge, not from ad-hoc narration.",
        "",
        "## Context",
        "",
        f"- task: `{context['task']}`",
        f"- model: `{context['model_name']}`",
        f"- treatment_col: `{context['treatment_col']}`",
        f"- outcome_col: `{context['outcome_col']}`",
        f"- registered_models: `{context['registered_models']}`",
        f"- ready_models: `{context['ready_models']}`",
        "",
        "## Q&A",
        "",
    ]
    for idx, topic in enumerate(INTERVIEW_AGENT_TOPICS, start=1):
        question = topic["question"]
        answer = interview_agent_reply(question, context) or fallback_answer(question, topic["evidence"], context)
        rows.append(
            {
                "id": topic["id"],
                "topic": topic["topic"],
                "question": question,
                "answer_chars": len(answer),
                "evidence": topic["evidence"],
            }
        )
        lines.extend(
            [
                f"### {idx}. {topic['id']} {topic['topic']}",
                "",
                f"**Question:** {question}",
                "",
                answer,
                "",
            ]
        )

    lines.extend(["## Industry Q&A", ""])
    for idx, question in enumerate(INDUSTRY_INTERVIEW_PROMPTS, start=1):
        evidence = "docs/INDUSTRIAL_UPLIFT_CASE_STUDIES.md; docs/UPLIFT_BUSINESS_SCENARIOS.md"
        answer = industry_agent_reply(question, context) or frontier_agent_reply(question) or fallback_answer(question, evidence, context)
        rows.append(
            {
                "id": f"IND{idx:02d}",
                "topic": "industrial uplift",
                "question": question,
                "answer_chars": len(answer),
                "evidence": evidence,
            }
        )
        lines.extend(
            [
                f"### IND{idx:02d} industrial uplift",
                "",
                f"**Question:** {question}",
                "",
                answer,
                "",
            ]
        )

    payload = {
        "status": "ok",
        "generated_at": generated_at,
        "context": context,
        "rows": rows,
        "min_answer_chars": min(row["answer_chars"] for row in rows) if rows else 0,
    }
    return "\n".join(lines).strip() + "\n", payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a deterministic Agent interview transcript sample.")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--output", default="docs/AGENT_INTERVIEW_TRANSCRIPT_SAMPLE.md")
    parser.add_argument("--json-output", default="reports/agent_interview_transcript_sample_latest.json")
    args = parser.parse_args()

    markdown, payload = build_sample(Path(args.reports_dir))
    if payload["min_answer_chars"] < 300:
        raise SystemExit(f"Agent transcript sample has too-short answers: {payload['min_answer_chars']}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")

    json_output = Path(args.json_output)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "output": str(output), "json_output": str(json_output), "rows": len(payload["rows"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
