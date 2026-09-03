from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepuplift.core.industry_playbook import (
    INDUSTRY_INTERVIEW_PROMPTS,
    business_scenarios,
    industry_references,
    llm_interview_qa,
    policy_templates,
    scenario_model_rows,
)


OUTPUT_FILES = {
    "case_studies": Path("docs/INDUSTRIAL_UPLIFT_CASE_STUDIES.md"),
    "business_scenarios": Path("docs/UPLIFT_BUSINESS_SCENARIOS.md"),
    "industry_deep_dive": Path("docs/UPLIFT_INTERVIEW_INDUSTRY_DEEP_DIVE.md"),
    "model_selection": Path("docs/UPLIFT_MODEL_SELECTION_PLAYBOOK.md"),
    "coupon_ads_growth": Path("docs/UPLIFT_COUPON_ADS_GROWTH_PLAYBOOK.md"),
    "llm_uplift_frontier": Path("docs/UPLIFT_LLM_CAUSAL_FRONTIER.md"),
    "llm_routing_policy": Path("docs/LLM_ROUTING_POLICY_PLAYBOOK.md"),
    "llm_interview_qa": Path("docs/UPLIFT_LLM_INTERVIEW_QA.md"),
    "online_incrementality": Path("docs/UPLIFT_ONLINE_EXPERIMENT_AND_INCREMENTALITY.md"),
    "references": Path("docs/UPLIFT_INDUSTRY_REFERENCES.md"),
}


def table(rows: list[dict], columns: list[str]) -> str:
    if not rows:
        return ""
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        values = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, list):
                value = ", ".join(str(item) for item in value)
            values.append(str(value).replace("\n", " "))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, sep, *body])


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")


def industry_mini_board_rows() -> list[dict[str, str]]:
    return [
        {
            "board": "China growth and marketing",
            "companies": "Tencent Ads, Meituan, Alibaba, DiDi, ByteDance, Kuaishou, JD.com",
            "what_to_say": "国内业务重点讲发券/补贴/广告/推荐干预，突出随机实验、X-Learner/DR/R、全链路 ROI、延迟反馈和多 treatment。",
            "ui_proof": "Story Fresh Additions + Knowledge Industrial Knowledge Map + Models Scenario Matrix",
        },
        {
            "board": "Global experimentation",
            "companies": "Uber, Google, Amazon, Airbnb, Spotify",
            "what_to_say": "海外材料重点讲 A/B holdout、geo/marketplace experiment、incremental LTV、个性化和实验平台分层。",
            "ui_proof": "Policy Scenario ROI Lab + Knowledge Source Adoption Drilldown + Evidence Pack",
        },
        {
            "board": "Research frontier",
            "companies": "PMLR, arXiv, journal/conference papers",
            "what_to_say": "前沿问题包括 delayed feedback、multi-treatment、continuous treatment、combinatorial treatment、budget-constrained causal bandit 和 journey overlap。",
            "ui_proof": "Models Scenario Tag + Agent industrial hard questions + docs/UPLIFT_MODEL_SELECTION_PLAYBOOK.md",
        },
        {
            "board": "LLM serving and causal copilot",
            "companies": "Microsoft Research, LMSYS, FrugalGPT, BEST-Route, causal-copilot research",
            "what_to_say": "LLM 不替代因果估计器，而是作为 copilot/orchestrator；LLM routing 可表述为 strong-model treatment 的增量质量、成本和时延决策。",
            "ui_proof": "Story LLM + Uplift Frontier + Knowledge LLM map + Policy LLM Routing ROI Simulator",
        },
    ]


def case_studies_md(generated_at: str) -> str:
    refs = industry_references()
    rows = [
        {
            "company": ref["company"],
            "scenario": ref["scenario"],
            "source_type": ref["source_type"],
            "lesson": ref["lesson"],
            "url": ref["url"],
        }
        for ref in refs
    ]
    details = "\n".join(
        f"### {ref['company']} - {ref['title']}\n\n"
        f"- Scenario: `{ref['scenario']}`\n"
        f"- Source type: `{ref['source_type']}`\n"
        f"- Link: {ref['url']}\n"
        f"- Industrial lesson: {ref['lesson']}\n"
        f"- Interview hook: {ref['interview_hook']}\n"
        for ref in refs
    )
    return f"""# Industrial Uplift Case Studies

Generated at: `{generated_at}`

This document collects trusted public references for industrial uplift, incrementality, coupon allocation, advertising lift, marketplace experimentation, and growth intervention.

## Case Study Index

{table(rows, ["company", "scenario", "source_type", "lesson", "url"])}

## China / Global Industry Mini-Board

{table(industry_mini_board_rows(), ["board", "companies", "what_to_say", "ui_proof"])}

## Detailed Notes

{details}
"""


def business_scenarios_md(generated_at: str) -> str:
    scenarios = business_scenarios()
    rows = [
        {
            "id": row["id"],
            "name": row["name"],
            "business_question": row["business_question"],
            "treatments": row["treatments"],
            "primary_metrics": row["primary_metrics"],
            "risks": row["risks"],
            "ready_models": row["ready_models"],
        }
        for row in scenarios
    ]
    return f"""# Uplift Business Scenarios

Generated at: `{generated_at}`

The workbench should recommend uplift models by business design, not by algorithm popularity.

{table(rows, ["id", "name", "business_question", "treatments", "primary_metrics", "risks", "ready_models"])}

## Interview Framing

- Coupon and subsidy: optimize incremental gross profit after coupon cost, not raw conversion.
- Ads: distinguish pCTR/pCVR from causal incrementality, and use holdout/geo/ghost-ad style evidence.
- Growth and CRM: include delayed feedback, fatigue, opt-out uplift, and long-term holdout.
- Recommendation and marketplace: watch for position bias, interference, spillover, and feedback loops.
"""


def industry_deep_dive_md(generated_at: str) -> str:
    prompts = "\n".join(f"- {prompt}" for prompt in INDUSTRY_INTERVIEW_PROMPTS)
    scenario_rows = scenario_model_rows()
    return f"""# Uplift Interview Industry Deep Dive

Generated at: `{generated_at}`

## Questions The Interviewer May Ask

{prompts}

## How To Answer

1. Start from business intervention: coupon, ad exposure, push, CRM journey, recommendation exposure, subsidy.
2. State why ordinary response prediction is insufficient.
3. Identify treatment design: binary, multi-treatment, continuous dose, delayed/long-term treatment, marketplace interference.
4. Name the right model family: T/X for randomized baselines, DR/R/DML for observational bias, causal forest for non-linear heterogeneity, multi-treatment/dose-response for action selection.
5. Tie offline metrics to online value: QINI/AUUC -> Top-K -> policy value -> holdout/A/B/geo experiment.
6. Name risks: overlap, selection bias, SUTVA violation, fatigue, arbitrage, attribution bias, delayed feedback.

## Scenario-To-Model Cheat Sheet

{table(scenario_rows, ["scenario", "business_question", "recommended_models", "model_families", "primary_metrics", "major_risks", "evidence"])}
"""


def model_selection_md(generated_at: str) -> str:
    return f"""# Uplift Model Selection Playbook

Generated at: `{generated_at}`

## Decision Rules

| Data / business condition | Recommended family | Why |
| --- | --- | --- |
| Randomized binary treatment, balanced arms | T-Learner, transformed outcome, scikit-uplift baselines | Fast and transparent baseline |
| Treatment/control imbalance or observational assignment | DR-Learner, R-Learner, DML, Domain Adaptation | Orthogonalization and propensity correction reduce selection bias |
| Strong non-linear heterogeneity | Causal forest, DR forest, LightGBM learners | Captures segment-level treatment-effect variation |
| Multiple coupons/channels/actions | Multi-treatment T/DR learner | Learns action-vs-control uplift and recommended action |
| Coupon amount, bid, discount, frequency | Dose-response learner | Treatment is continuous rather than binary |
| Need business-ready Top-K targeting | Calibrated DR/R, GGBM uplift, policy value | Ranking and threshold stability matter more than raw outcome accuracy |
| Delayed activation or long-term retention | Delayed-feedback uplift, survival/long-window evaluation | Labels can mature after the campaign window |
| LLM routing or model escalation | DR/R learner, calibrated uplift ranking, contextual bandit | Treatment is strong-model call; optimize incremental quality net of cost |
| LLM-generated text/creative/offer | Multi-treatment learner, structured/text treatment CATE | Treatment is the content representation, not merely send/no-send |

## Workbench Model Catalog Usage

The `Models` page exposes scenario tags so interview demos can filter the registry by coupon/subsidy, ads/bidding, growth/CRM, recommendation, marketplace, open-source benchmark, and industrial tabular use cases. This keeps the model list tied to treatment design instead of becoming a flat algorithm dump.

## Interview One-Liner

I do not choose uplift models by leaderboard habit. I first classify the treatment design and bias risk, then select the simplest model family that can answer the business decision with credible evidence.
"""


def coupon_ads_growth_md(generated_at: str) -> str:
    templates = policy_templates()
    return f"""# Coupon, Ads, And Growth Uplift Playbook

Generated at: `{generated_at}`

## Policy Value Templates

{table(templates, ["scenario", "conversion_value_hint", "contact_cost_hint", "extra_guardrails"])}

## Workbench Scenario ROI Helper

The Policy page includes a scenario helper that converts business assumptions into the curve inputs:

| Scenario | Value input | Cost input |
| --- | --- | --- |
| Coupon | gross margin per order | coupon face value * redemption probability + fatigue / abuse penalty |
| Ads | incremental conversion value | CPM / 1000 * exposures per user + attribution / spillover buffer |
| Growth Push | activation or LTV value | message cost + fatigue penalty + opt-out risk penalty |
| Marketplace Subsidy | marketplace balance value or gross profit | subsidy payout + ops cost + spillover buffer |

The final Top-K rule is `uplift_score * value_per_conversion - contact_cost`. Offline output still needs holdout, geo, switchback, or long-term experiment validation before full rollout.

## Coupon / Subsidy

- Do not target only high conversion probability users.
- Estimate incremental gross profit: uplift * margin - coupon cost - channel cost.
- Suppress sleeping dogs and negative uplift users.
- Use budget caps, user-level subsidy caps, and abuse/fraud filters.
- Keep long-term holdout for fatigue and subsidy dependency.

## Ads / Bidding

- pCTR and pCVR predict probability; incrementality estimates the causal effect of showing the ad.
- Prefer randomized holdout, ghost-ad style measurement, or geo experiments when user-level randomization is hard.
- Report incremental conversions, incremental value, and iROAS lower bounds.
- Watch marketplace interference and cross-channel spillover.

## Growth / CRM

- Include fatigue cost and opt-out uplift in policy value.
- Use delayed feedback windows for retention and activation.
- Keep frequency caps and long-term holdouts.
- Treat channel choice as multi-treatment when push/email/SMS/call coexist.
"""


def llm_uplift_frontier_md(generated_at: str) -> str:
    llm_refs = [
        ref
        for ref in industry_references()
        if str(ref.get("scenario", "")).startswith("llm")
        or "text_treatment" in str(ref.get("scenario", ""))
        or "cost_aware_llm" in str(ref.get("scenario", ""))
    ]
    return f"""# Uplift + LLM Causal Frontier

Generated at: `{generated_at}`

## Positioning

The project treats LLMs as a causal workflow layer, not as a replacement for treatment-effect estimators. The safest architecture is:

`LLM causal copilot -> deterministic diagnostics/model registry/evaluator -> policy value -> evidence card/regression gate`

## Three High-Value Integration Patterns

| Pattern | Treatment | Outcome | Recommended Method | Demo Surface |
| --- | --- | --- | --- | --- |
| LLM as causal copilot | analysis workflow step | correct treatment design, adjustment set, model choice | source-grounded Agent + deterministic audits | Story / Knowledge / Agent |
| Text or creative as treatment | prompt, ad copy, push text, offer description | incremental conversion, opt-out, LTV | structured/text treatment CATE, multi-treatment uplift | Models / Agent / Policy |
| Uplift for LLM routing | strong model vs cheap model | incremental quality or task success | DR/R learner, calibrated uplift, contextual bandit | Policy LLM Routing ROI Simulator |
| Budget-paced LLM serving | strong model, RAG, tool call, long-thinking compute | cost-adjusted task success under budget/SLO | calibrated uplift -> contextual bandit -> budget pacing | Policy / Models / Evidence |

## Why This Is Not Just Prompt Engineering

- LLM text generation can create candidate interventions, but randomization and uplift evaluation decide whether a creative has incremental value.
- LLM routing is not model ranking; it is an incremental decision: whether the strong model adds enough quality to justify added cost and latency.
- LLM causal explanations must be backed by source quality, UI evidence, generated docs, and smoke/regression tests.

## Reference Map

{table(llm_refs, ["id", "company", "source_type", "scenario", "title", "url", "lesson"])}

## Interview One-Liner

I used LLMs where they are strongest: orchestration, explanation, evidence synthesis, and structured treatment representation. The actual causal decision still goes through uplift/CATE estimators, policy value, and online incrementality validation.
"""


def llm_routing_policy_md(generated_at: str) -> str:
    return f"""# LLM Routing Policy Playbook

Generated at: `{generated_at}`

## Causal Framing

- Control: route the query to a cheap/small model.
- Treatment: escalate the query to a stronger or more expensive model.
- Outcome: judge score, task success, saved human-review cost, user satisfaction, or business value.
- Uplift: `E[quality | strong_model, X] - E[quality | cheap_model, X]`.
- Policy value: `quality_uplift * value_per_quality_point - incremental_model_cost - latency_penalty`.

## Offline Workflow

1. Randomly route a slice of traffic across small and strong models.
2. Collect judge score, task success, latency, and cost.
3. Train DR/R learner or causal forest on query/task/model features.
4. Calibrate uplift scores and choose a Top-K escalation threshold.
5. Stress test with budget caps, latency SLO and drift checks.

## Online Workflow

1. Start with fixed threshold and holdout.
2. Track cost-adjusted quality, latency, fallback and hallucination/evidence failures.
3. Move to contextual bandit only after offline calibration and guardrails are stable.
4. Use budget pacing because model prices, traffic mix and model quality are non-stationary.

## Demo Mapping

- Policy page: `LLM Routing ROI Simulator`.
- Knowledge page: `LLM + Causal Frontier Map`.
- Agent page: ask `LLM routing 为什么是 uplift 问题？`
- Story page: `LLM + Uplift Frontier`.
"""


def llm_interview_qa_md(generated_at: str) -> str:
    rows = llm_interview_qa()
    details = "\n".join(
        f"## {index}. {row['question']}\n\n"
        f"**Short answer:** {row['answer']}\n\n"
        f"**Engineering design:** {row['engineering_design']}\n\n"
        f"**Evidence:** `{row['evidence']}`\n\n"
        f"**Interview line:** {row['interview_line']}\n"
        for index, row in enumerate(rows, start=1)
    )
    return f"""# Uplift + LLM Interview Q&A

Generated at: `{generated_at}`

This document turns the LLM frontier work into interview-ready answers. The repeated pattern is: use LLMs for orchestration and representation, keep causal identification and policy value in deterministic, testable components.

## Quick Matrix

{table(rows, ["question", "answer", "engineering_design", "evidence", "interview_line"])}

## Deep Answers

{details}
"""


def online_incrementality_md(generated_at: str) -> str:
    return f"""# Online Experiment And Incrementality Playbook

Generated at: `{generated_at}`

## Offline-To-Online Ladder

1. Offline diagnostics: treatment design, overlap, feature balance, leakage.
2. Model comparison: QINI, AUUC, Top-K uplift, calibration, bootstrap CI.
3. Policy simulation: convert uplift scores into cost-adjusted Top-K thresholds.
4. Holdout design: user-level RCT if possible; geo/switchback/cluster when interference exists.
5. Online metrics: incremental conversion, incremental gross profit, iROAS/ROI, opt-out/negative actions, long-term retention.
6. Rollout guard: fixed threshold, pre-registered metric, budget cap, and monitoring of drift/fatigue.

## Why A/B Is Still Needed

Offline uplift estimates can rank candidates, but online treatment assignment verifies the actual incrementality under current budget, channel, creative, competition, and user fatigue conditions.
"""


def references_md(generated_at: str) -> str:
    rows = industry_references()
    return f"""# Uplift Industry References

Generated at: `{generated_at}`

{table(rows, ["id", "company", "source_type", "scenario", "title", "url", "lesson"])}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DeepUplift industrial uplift playbook docs.")
    parser.add_argument("--reports-dir", default="reports")
    args = parser.parse_args()

    generated_at = time.strftime("%Y-%m-%d %H:%M:%S %z")
    outputs = {
        "case_studies": case_studies_md(generated_at),
        "business_scenarios": business_scenarios_md(generated_at),
        "industry_deep_dive": industry_deep_dive_md(generated_at),
        "model_selection": model_selection_md(generated_at),
        "coupon_ads_growth": coupon_ads_growth_md(generated_at),
        "llm_uplift_frontier": llm_uplift_frontier_md(generated_at),
        "llm_routing_policy": llm_routing_policy_md(generated_at),
        "llm_interview_qa": llm_interview_qa_md(generated_at),
        "online_incrementality": online_incrementality_md(generated_at),
        "references": references_md(generated_at),
    }
    for key, content in outputs.items():
        write(OUTPUT_FILES[key], content)

    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "ok",
        "generated_at": generated_at,
        "references": industry_references(),
        "scenarios": business_scenarios(),
        "outputs": {key: str(path) for key, path in OUTPUT_FILES.items()},
    }
    latest_path = reports_dir / "industry_playbook_latest.json"
    latest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "outputs": payload["outputs"], "json_output": str(latest_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
