from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepuplift.core.model_deconstruction import MODEL_DECONSTRUCTIONS  # noqa: E402


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def by_model(rows: list[dict[str, Any]], key: str = "model") -> dict[str, dict[str, Any]]:
    return {str(row.get(key)): row for row in rows if isinstance(row, dict) and row.get(key)}


def group_by_model(rows: list[dict[str, Any]], key: str = "model") -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if isinstance(row, dict) and row.get(key):
            grouped.setdefault(str(row.get(key)), []).append(row)
    return grouped


def compact_paths(items: list[str] | tuple[str, ...], limit: int = 4) -> list[str]:
    return [str(item) for item in list(items)[:limit] if item]


def metric(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.4g}"
    except (TypeError, ValueError):
        return str(value)


PROOF_RANK = {
    "reference_only": 0,
    "source_deconstruction": 1,
    "training_smoke": 2,
    "objective_and_loss_diagnostics": 3,
    "tuned_benchmark_candidate": 4,
    "paper_level_benchmark_candidate": 5,
}


def strongest_proof(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "source_deconstruction"
    return max((str(row.get("proof_level") or "reference_only") for row in rows), key=lambda item: PROOF_RANK.get(item, -1))


def license_gate(rows: list[dict[str, Any]], license_policy: str) -> str:
    text = " ".join(
        [license_policy]
        + [str(row.get("source_license", "")) for row in rows]
        + [str(row.get("license_risk", "")) for row in rows]
        + [str(row.get("adoption_decision", "")) for row in rows]
    ).lower()
    if "gpl" in text:
        return "idea_only_or_isolated_plugin"
    if "unknown" in text or "read_only" in text or "not verified" in text:
        return "read_only_until_verified"
    if "guarded" in text:
        return "guarded_adapter"
    return "permissive_or_paper_reference"


def promotion_decisions_for_model(model_id: str, promotion_payload: dict[str, Any]) -> list[str]:
    needle = model_id.lower()
    decisions: set[str] = set()
    for card in promotion_payload.get("cards") or []:
        haystack = " ".join(str(card.get(field, "")) for field in ["entity", "primary_blocker", "primary_reason", "safe_claim", "unsafe_claim"]).lower()
        haystack += " " + " ".join(str(item) for item in card.get("signals") or []).lower()
        if needle in haystack:
            decisions.add(str(card.get("overall_decision") or "review_before_promotion"))
    return sorted(decisions)


def evidence_stage(
    proof: str,
    paper_card: dict[str, Any],
    tuned_card: dict[str, Any],
    objective_card: dict[str, Any],
    training_card: dict[str, Any],
    gate: str,
) -> str:
    paper_verdict = paper_card.get("verdict")
    tuned_verdict = tuned_card.get("verdict")
    if gate in {"idea_only_or_isolated_plugin", "read_only_until_verified"} and proof not in {"paper_level_benchmark_candidate", "tuned_benchmark_candidate"}:
        return "research_only_license_guarded"
    if paper_verdict in {"accuracy_and_ranking_candidate", "ranking_candidate"} or proof == "paper_level_benchmark_candidate":
        return "paper_level_review_candidate"
    if tuned_verdict in {"tuned_candidate", "ranking_candidate"} or proof == "tuned_benchmark_candidate":
        return "tuned_benchmark_candidate"
    if objective_card:
        return "objective_diagnostics_ready"
    if training_card.get("status") == "ok":
        return "training_smoke_ready"
    return "source_deconstruction_ready"


def promotion_tier(stage: str, launch_decision: str) -> str:
    if launch_decision == "block_promotion":
        return "T3_blocked_for_launch"
    if stage == "paper_level_review_candidate":
        return "T0_interview_and_benchmark_anchor"
    if stage == "tuned_benchmark_candidate":
        return "T1_tuned_candidate"
    if stage in {"objective_diagnostics_ready", "training_smoke_ready"}:
        return "T2_architecture_evidence"
    return "T3_research_backlog"


def evidence_score(
    proof: str,
    paper_card: dict[str, Any],
    tuned_card: dict[str, Any],
    objective_card: dict[str, Any],
    training_card: dict[str, Any],
    claim_rows: list[dict[str, Any]],
    gap_rows: list[dict[str, Any]],
    gate: str,
    launch_decision: str,
) -> int:
    score = 20 + PROOF_RANK.get(proof, 0) * 8
    if paper_card:
        score += 12
    if tuned_card:
        score += 8
    if objective_card:
        score += 10
    if training_card.get("status") == "ok":
        score += 6
    score += min(len(claim_rows), 4) * 2
    score += min(len(gap_rows), 3)
    if gate in {"idea_only_or_isolated_plugin", "read_only_until_verified"}:
        score -= 8
    if launch_decision == "block_promotion":
        score -= 10
    return max(0, min(score, 100))


def launch_decision(decisions: list[str], stage: str) -> str:
    if "block_promotion" in decisions:
        return "block_promotion"
    if "review_before_promotion" in decisions:
        return "review_before_promotion"
    if stage in {"paper_level_review_candidate", "tuned_benchmark_candidate"}:
        return "review_before_pilot"
    return "demo_only"


def safe_claim(stage: str, model_id: str, paper_card: dict[str, Any], tuned_card: dict[str, Any], objective_card: dict[str, Any], model_line: str) -> str:
    if stage == "paper_level_review_candidate":
        return paper_card.get("interview_line") or f"{model_id} has paper-level benchmark candidate evidence; keep production claims behind OPE/holdout."
    if stage == "tuned_benchmark_candidate":
        return tuned_card.get("interview_line") or f"{model_id} has tuned benchmark evidence; claim the dataset and metric boundary explicitly."
    if stage == "objective_diagnostics_ready":
        return objective_card.get("safe_claim") or f"{model_id} can be explained from loss/source to diagnostics, without claiming benchmark superiority."
    if stage == "training_smoke_ready":
        return f"{model_id} has executable training smoke evidence and model-card explanation."
    return model_line


def blocked_claim(gate: str, stage: str) -> str:
    blocks = ["No production lift claim without online incrementality, OPE/holdout and rollback guardrails."]
    if gate in {"idea_only_or_isolated_plugin", "read_only_until_verified"}:
        blocks.append("No direct source-code integration claim until license is approved or the adapter is isolated.")
    if stage not in {"paper_level_review_candidate", "tuned_benchmark_candidate"}:
        blocks.append("No SOTA or paper-level win claim from source docs, objective diagnostics or smoke evidence alone.")
    return " ".join(blocks)


def next_step(model: Any, stage: str, paper_card: dict[str, Any], tuned_card: dict[str, Any], objective_card: dict[str, Any], gap_rows: list[dict[str, Any]]) -> str:
    if stage == "paper_level_review_candidate":
        return paper_card.get("watchout") or "Run business holdout/OPE and stress benchmark before pilot promotion."
    if stage == "tuned_benchmark_candidate":
        return tuned_card.get("watchout") or "Promote only after multi-seed paper benchmark and metric-conflict review."
    if objective_card:
        tasks = objective_card.get("next_tasks") or []
        if tasks:
            return str(tasks[0])
    for row in gap_rows:
        tasks = row.get("validation_needed") or []
        if tasks:
            return str(tasks[0])
    return (model.next_tasks or ("add benchmark evidence",))[0]


def build_rows(root: Path) -> list[dict[str, Any]]:
    claims = read_json(root / "reports" / "algorithm_claim_ledger_latest.json")
    gaps = read_json(root / "reports" / "paper_reproduction_gap_ledger_latest.json")
    objective = read_json(root / "reports" / "deep_model_objective_diagnostics_latest.json")
    tuned = read_json(root / "reports" / "deep_model_tuned_benchmark_latest.json")
    paper = read_json(root / "reports" / "paper_benchmark_interpretation_latest.json")
    training = read_json(root / "reports" / "deep_model_training_evidence_latest.json")
    promotion = read_json(root / "reports" / "promotion_launch_cards_latest.json")

    claim_rows = group_by_model(claims.get("rows") or [])
    gap_rows = group_by_model(gaps.get("rows") or [])
    objective_cards = by_model(objective.get("cards") or [])
    tuned_cards = by_model(tuned.get("model_scorecards") or [])
    paper_cards = by_model(paper.get("model_scorecards") or [])
    training_cards = by_model(training.get("leaderboard") or [])

    rows: list[dict[str, Any]] = []
    for model_id, model in MODEL_DECONSTRUCTIONS.items():
        model_claims = claim_rows.get(model_id, [])
        model_gaps = gap_rows.get(model_id, [])
        objective_card = objective_cards.get(model_id, {})
        tuned_card = tuned_cards.get(model_id, {})
        paper_card = paper_cards.get(model_id, {})
        training_card = training_cards.get(model_id, {})
        proof = strongest_proof(model_gaps)
        gate = license_gate(model_gaps, model.license_policy)
        stage = evidence_stage(proof, paper_card, tuned_card, objective_card, training_card, gate)
        decisions = promotion_decisions_for_model(model_id, promotion)
        launch = launch_decision(decisions, stage)
        tier = promotion_tier(stage, launch)
        score = evidence_score(proof, paper_card, tuned_card, objective_card, training_card, model_claims, model_gaps, gate, launch)
        evidence_refs = [
            f"docs/model_deconstruction/{model_id}.md",
            "reports/model_deconstruction_catalog_latest.json",
            "reports/algorithm_claim_ledger_latest.json",
            "reports/paper_reproduction_gap_ledger_latest.json",
        ]
        if paper_card:
            evidence_refs.append("reports/paper_benchmark_interpretation_latest.json")
        if tuned_card:
            evidence_refs.append("reports/deep_model_tuned_benchmark_latest.json")
        if objective_card:
            evidence_refs.append("reports/deep_model_objective_diagnostics_latest.json")
        if training_card:
            evidence_refs.append("reports/deep_model_training_evidence_latest.json")

        rows.append(
            {
                "model": model_id,
                "display_name": model.display_name,
                "family": model.family,
                "role": model.role,
                "evidence_stage": stage,
                "promotion_tier": tier,
                "evidence_score": score,
                "paper_proof_level": proof,
                "paper_verdict": paper_card.get("verdict") or "not_in_latest_paper_benchmark",
                "tuned_verdict": tuned_card.get("verdict") or "not_in_latest_tuned_benchmark",
                "objective_status": "ready" if objective_card else "not_applicable_or_missing",
                "training_status": training_card.get("status") or "not_in_latest_training_smoke",
                "license_gate": gate,
                "launch_decision": launch,
                "claim_rows": len(model_claims),
                "source_rows": len(model_gaps),
                "loss_terms": len(objective_card.get("loss_terms") or []),
                "best_pehe": paper_card.get("pehe_mean_over_datasets") or tuned_card.get("best_pehe"),
                "best_qini": paper_card.get("qini_mean_over_datasets") or tuned_card.get("best_qini"),
                "policy_value": paper_card.get("policy_top10_oracle_value_mean_over_datasets") or tuned_card.get("best_policy_top10_oracle_value"),
                "safe_claim": safe_claim(stage, model_id, paper_card, tuned_card, objective_card, model.interview_line),
                "blocked_claim": blocked_claim(gate, stage),
                "next_evidence_step": next_step(model, stage, paper_card, tuned_card, objective_card, model_gaps),
                "demo_route": "Model Deconstruction -> Model Review Card -> Evidence Dashboard -> Interview Evidence Pack",
                "interview_30s": f"{model_id}: {stage}; score={score}; claim boundary is `{tier}`.",
                "evidence_refs": evidence_refs,
                "code_refs": compact_paths(model.code_paths),
            }
        )
    return rows


def build_payload(root: Path) -> dict[str, Any]:
    rows = build_rows(root)
    by_stage: dict[str, int] = {}
    by_tier: dict[str, int] = {}
    by_gate: dict[str, int] = {}
    by_launch: dict[str, int] = {}
    for row in rows:
        by_stage[row["evidence_stage"]] = by_stage.get(row["evidence_stage"], 0) + 1
        by_tier[row["promotion_tier"]] = by_tier.get(row["promotion_tier"], 0) + 1
        by_gate[row["license_gate"]] = by_gate.get(row["license_gate"], 0) + 1
        by_launch[row["launch_decision"]] = by_launch.get(row["launch_decision"], 0) + 1
    top_rows = sorted(rows, key=lambda row: (-int(row.get("evidence_score", 0)), str(row.get("model"))))[:3]
    return {
        "schema_version": 1,
        "status": "ok",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "summary": {
            "models": len(rows),
            "stage_counts": by_stage,
            "promotion_tier_counts": by_tier,
            "license_gate_counts": by_gate,
            "launch_decision_counts": by_launch,
            "top_models": [row.get("model") for row in top_rows],
        },
        "rows": rows,
        "interview_tracks": {
            "30s": "我用 promotion matrix 把每个模型能讲到什么程度、缺什么证据、是否能进入试点前评审统一起来。",
            "5min": "矩阵把源码拆解、论文复现、benchmark、objective diagnostics、license gate 和 launch decision 串成一张表，避免把 smoke 证据包装成 paper win 或 production lift。",
            "15min": "评审时先看 T0/T1 候选，再看 blocked claim 和 next evidence step；这就是从模型 zoo 升级成 causal decision platform 的证据治理层。",
        },
    }


def flatten(value: Any) -> str:
    if isinstance(value, list):
        return "; ".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return "" if value is None else str(value)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model",
        "display_name",
        "family",
        "evidence_stage",
        "promotion_tier",
        "evidence_score",
        "paper_proof_level",
        "paper_verdict",
        "tuned_verdict",
        "objective_status",
        "training_status",
        "license_gate",
        "launch_decision",
        "claim_rows",
        "source_rows",
        "loss_terms",
        "best_pehe",
        "best_qini",
        "policy_value",
        "safe_claim",
        "blocked_claim",
        "next_evidence_step",
        "demo_route",
        "interview_30s",
        "evidence_refs",
        "code_refs",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: flatten(row.get(field)) for field in fields})


def build_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary") or {}
    rows = payload.get("rows") or []
    tracks = payload.get("interview_tracks") or {}
    overview = "\n".join(
        "| {model} | {stage} | {tier} | {score} | {paper} | {tuned} | {gate} | {launch} | {next_step} |".format(
            model=row.get("model", ""),
            stage=str(row.get("evidence_stage", "")).replace("|", "/"),
            tier=str(row.get("promotion_tier", "")).replace("|", "/"),
            score=row.get("evidence_score", ""),
            paper=str(row.get("paper_verdict", "")).replace("|", "/"),
            tuned=str(row.get("tuned_verdict", "")).replace("|", "/"),
            gate=str(row.get("license_gate", "")).replace("|", "/"),
            launch=str(row.get("launch_decision", "")).replace("|", "/"),
            next_step=str(row.get("next_evidence_step", "")).replace("|", "/"),
        )
        for row in sorted(rows, key=lambda item: (-int(item.get("evidence_score", 0)), str(item.get("model"))))
    )
    sections = []
    for row in rows:
        sections.append(
            f"""## {row.get("model")}

| Field | Value |
| --- | --- |
| Evidence stage | `{row.get("evidence_stage")}` |
| Promotion tier | `{row.get("promotion_tier")}` |
| Evidence score | `{row.get("evidence_score")}` |
| Paper proof level | `{row.get("paper_proof_level")}` |
| Paper verdict | `{row.get("paper_verdict")}` |
| Tuned verdict | `{row.get("tuned_verdict")}` |
| Objective / training | `{row.get("objective_status")}` / `{row.get("training_status")}` |
| License gate | `{row.get("license_gate")}` |
| Launch decision | `{row.get("launch_decision")}` |
| Best PEHE / QINI / policy | `{metric(row.get("best_pehe"))}` / `{metric(row.get("best_qini"))}` / `{metric(row.get("policy_value"))}` |

Safe claim: {row.get("safe_claim")}

Blocked claim: {row.get("blocked_claim")}

Next evidence step: {row.get("next_evidence_step")}

Evidence refs: {flatten(row.get("evidence_refs"))}
"""
        )
    return f"""# DeepUplift Model Evidence Promotion Matrix

Generated at: `{payload.get("generated_at")}`

This matrix answers the review question: for each model, what can we safely claim today, what evidence supports it, what is still blocked, and what is the next step before a stronger interview or pilot claim?

## Summary

| Metric | Value |
| --- | ---: |
| Models | {summary.get("models", 0)} |

Stage counts: `{json.dumps(summary.get("stage_counts") or {}, ensure_ascii=False, sort_keys=True)}`

Promotion tier counts: `{json.dumps(summary.get("promotion_tier_counts") or {}, ensure_ascii=False, sort_keys=True)}`

License gate counts: `{json.dumps(summary.get("license_gate_counts") or {}, ensure_ascii=False, sort_keys=True)}`

## Interview Tracks

- 30s: {tracks.get("30s", "")}
- 5min: {tracks.get("5min", "")}
- 15min: {tracks.get("15min", "")}

## Matrix

| Model | Evidence stage | Promotion tier | Score | Paper verdict | Tuned verdict | License gate | Launch | Next step |
| --- | --- | --- | ---: | --- | --- | --- | --- | --- |
{overview}

{chr(10).join(sections)}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate model evidence promotion matrix for DeepUplift model review.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--json-output", default="reports/model_evidence_promotion_matrix_latest.json")
    parser.add_argument("--csv-output", default="reports/model_evidence_promotion_matrix_latest.csv")
    parser.add_argument("--doc-output", default="docs/DEEPUplift_MODEL_EVIDENCE_PROMOTION_MATRIX.md")
    args = parser.parse_args()

    root = Path(args.root)
    payload = build_payload(root)
    json_output = Path(args.json_output)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(Path(args.csv_output), payload.get("rows") or [])
    doc_output = Path(args.doc_output)
    doc_output.parent.mkdir(parents=True, exist_ok=True)
    doc_output.write_text(build_markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "ok",
                "json": str(json_output),
                "csv": args.csv_output,
                "markdown": str(doc_output),
                "models": (payload.get("summary") or {}).get("models"),
                "stages": (payload.get("summary") or {}).get("stage_counts"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
