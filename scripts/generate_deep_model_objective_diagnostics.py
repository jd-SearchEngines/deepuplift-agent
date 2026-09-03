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


DEEP_MODELS = ("CFRNet", "DragonNet", "EFIN", "DESCN")

LOSS_TERMS: dict[str, list[dict[str, str]]] = {
    "CFRNet": [
        {
            "term": "factual outcome loss",
            "formula": "L_y=(1-T)*ell(y0,Y)+T*ell(y1,Y)",
            "diagnostic": "train_outcome_loss / valid_outcome_loss / PEHE",
            "failure_signal": "loss falls but PEHE or policy value does not improve",
        },
        {
            "term": "representation balance / IPM",
            "formula": "alpha * IPM(phi_t, phi_c), e.g. MMD or energy distance",
            "diagnostic": "train_treatment_loss, aux_loss_last, post-trim QINI",
            "failure_signal": "over-balancing erases true effect modifiers or under-balancing leaves selection bias",
        },
    ],
    "DragonNet": [
        {
            "term": "factual outcome loss",
            "formula": "L_y=(1-T)*ell(y0,Y)+T*ell(y1,Y)",
            "diagnostic": "train_outcome_loss / PEHE / ATE error",
            "failure_signal": "good factual loss but poor treatment-effect ranking",
        },
        {
            "term": "propensity auxiliary head",
            "formula": "alpha * BCE(e_hat(X), T)",
            "diagnostic": "train_treatment_loss / overlap / ESS",
            "failure_signal": "propensity head learns logging bias but support is still weak",
        },
        {
            "term": "targeted regularization",
            "formula": "beta * (Y - (y_hat + eps * (T/e-(1-T)/(1-e))))^2",
            "diagnostic": "tarreg ablation, QINI delta, PEHE delta",
            "failure_signal": "near-zero or near-one propensity makes the targeted residual unstable",
        },
    ],
    "EFIN": [
        {
            "term": "base response path",
            "formula": "y0=base(x)",
            "diagnostic": "factual outcome loss / calibration",
            "failure_signal": "base response dominates and uplift path contributes little",
        },
        {
            "term": "treatment-aware feature interaction",
            "formula": "a_j=softmax(v^T relu(sigmoid(W_t t)+sigmoid(W_x x_j)))",
            "diagnostic": "attention export backlog / treatment-feature ablation / QINI",
            "failure_signal": "attention overfits sparse campaign or treatment IDs",
        },
        {
            "term": "uplift path",
            "formula": "y1=base(x)+u(x,t), tau=u(x,t)",
            "diagnostic": "policy value / uplift@K / treatment auxiliary loss",
            "failure_signal": "interaction signal is strong offline but not stable across seeds",
        },
    ],
    "DESCN": [
        {
            "term": "propensity head",
            "formula": "p=P(T=1|phi)",
            "diagnostic": "propensity/overlap future metric, readiness warning",
            "failure_signal": "biased exposure is modeled but missing overlap remains missing overlap",
        },
        {
            "term": "entire-space response heads",
            "formula": "estr=p*mu1, escr=(1-p)*mu0",
            "diagnostic": "head consistency backlog / QINI / AUUC",
            "failure_signal": "head semantics are hard to audit unless exported separately",
        },
        {
            "term": "cross-head treatment effect constraints",
            "formula": "cross_tr=sigmoid(mu0_logit+tau_logit), cross_cr=sigmoid(mu1_logit-tau_logit)",
            "diagnostic": "ablation battle card / train_loss_delta / policy value",
            "failure_signal": "many loss terms can fit tiny data while CATE accuracy stays weak",
        },
    ],
}


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def by_model(rows: list[dict[str, Any]], key: str = "model") -> dict[str, dict[str, Any]]:
    return {str(row.get(key)): row for row in rows if isinstance(row, dict) and row.get(key)}


def scorecard_for(model_id: str, tuned: dict[str, Any]) -> dict[str, Any]:
    for row in tuned.get("model_scorecards") or []:
        if row.get("model") == model_id:
            return row
    return {}


def training_for(model_id: str, training: dict[str, Any]) -> dict[str, Any]:
    for row in training.get("leaderboard") or []:
        if row.get("model") == model_id:
            return row
    return {}


def claim_policy(model_id: str, scorecard: dict[str, Any], training: dict[str, Any]) -> dict[str, str]:
    verdict = str(scorecard.get("verdict") or "")
    if verdict in {"tuned_candidate", "ranking_candidate"}:
        safe = "可以说该模型在当前 tuned benchmark 至少有候选证据，但仍需按业务 metric 和线上实验确认。"
        unsafe = "不要说它已经全面 SOTA 或可以直接上线。"
    elif verdict:
        safe = "可以说该模型提供架构深度、loss 解释和 failure attribution 证据。"
        unsafe = "不要把 smoke 或单轮 tuned 结果包装成 paper-level win。"
    elif training.get("status") == "ok":
        safe = "可以说该模型训练链路可运行，并且能产出 evaluator/artifact evidence。"
        unsafe = "不要把 smoke training 当成严肃 benchmark 结论。"
    else:
        safe = "只能作为模型结构和源码拆解候选。"
        unsafe = "不要宣称该模型已通过训练或 benchmark 验证。"
    return {"safe_claim": safe, "unsafe_claim": unsafe}


def build_cards(root: Path) -> dict[str, Any]:
    tuned = read_json(root / "reports" / "deep_model_tuned_benchmark_latest.json")
    training = read_json(root / "reports" / "deep_model_training_evidence_latest.json")
    cards: list[dict[str, Any]] = []
    for model_id in DEEP_MODELS:
        model = MODEL_DECONSTRUCTIONS[model_id]
        scorecard = scorecard_for(model_id, tuned)
        training_row = training_for(model_id, training)
        policy = claim_policy(model_id, scorecard, training_row)
        loss_terms = LOSS_TERMS[model_id]
        diagnostics = {
            "training_status": training_row.get("status"),
            "qini_score": training_row.get("qini_score"),
            "auuc_score": training_row.get("auuc_score"),
            "oracle_top10_recall": training_row.get("oracle_top10_recall"),
            "train_loss_last": training_row.get("train_loss_last"),
            "aux_loss_last": training_row.get("aux_loss_last"),
            "tuned_verdict": scorecard.get("verdict"),
            "best_pehe": scorecard.get("best_pehe"),
            "best_qini": scorecard.get("best_qini"),
            "avg_delta_pehe_vs_baseline": scorecard.get("avg_delta_pehe_vs_baseline"),
            "avg_delta_qini_vs_baseline": scorecard.get("avg_delta_qini_vs_baseline"),
        }
        cards.append(
            {
                "model": model_id,
                "family": model.family,
                "role": model.role,
                "objective": list(model.formulas),
                "loss_terms": loss_terms,
                "code_paths": list(model.code_paths),
                "diagnostics": diagnostics,
                "observability_gaps": list(model.implementation_gaps),
                "failure_modes": list(model.failure_modes),
                "next_tasks": list(model.next_tasks),
                "safe_claim": policy["safe_claim"],
                "unsafe_claim": policy["unsafe_claim"],
                "interview_line": model.interview_line,
                "source_license_policy": model.license_policy,
                "evidence_refs": [
                    "reports/deep_model_training_evidence_latest.json",
                    "reports/deep_model_tuned_benchmark_latest.json",
                    f"docs/model_deconstruction/{model_id}.md",
                ],
            }
        )
    return {
        "schema_version": 1,
        "status": "ok",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "summary": {
            "models": len(cards),
            "loss_terms": sum(len(card["loss_terms"]) for card in cards),
            "tuned_candidates": sum(1 for card in cards if (card.get("diagnostics") or {}).get("tuned_verdict") == "tuned_candidate"),
            "models_with_training_smoke": sum(1 for card in cards if (card.get("diagnostics") or {}).get("training_status") == "ok"),
        },
        "cards": cards,
        "interview_tracks": {
            "30s": "我不是只列深度模型名，而是把每个模型拆成 objective、loss term、诊断指标和 safe/unsafe claim。",
            "5min": "CFRNet 讲 representation balance，DragonNet 讲 propensity/tarreg，EFIN 讲 treatment-aware interaction，DESCN 讲 entire-space cross heads；每个都有训练或 tuned benchmark 证据。",
            "15min": "把 loss term 和可观测诊断连起来：factual loss 看拟合，IPM/propensity/interaction/cross-head 看因果结构，PEHE/QINI/policy value 看是否真的帮助决策。",
        },
    }


def write_csv(path: Path, cards: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model",
        "family",
        "role",
        "loss_term_count",
        "training_status",
        "tuned_verdict",
        "best_pehe",
        "best_qini",
        "safe_claim",
        "unsafe_claim",
        "interview_line",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for card in cards:
            diagnostics = card.get("diagnostics") or {}
            writer.writerow(
                {
                    "model": card.get("model"),
                    "family": card.get("family"),
                    "role": card.get("role"),
                    "loss_term_count": len(card.get("loss_terms") or []),
                    "training_status": diagnostics.get("training_status"),
                    "tuned_verdict": diagnostics.get("tuned_verdict"),
                    "best_pehe": diagnostics.get("best_pehe"),
                    "best_qini": diagnostics.get("best_qini"),
                    "safe_claim": card.get("safe_claim"),
                    "unsafe_claim": card.get("unsafe_claim"),
                    "interview_line": card.get("interview_line"),
                }
            )


def build_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary") or {}
    cards = payload.get("cards") or []
    rows = []
    for card in cards:
        diagnostics = card.get("diagnostics") or {}
        rows.append(
            "| {model} | {terms} | {training} | {verdict} | {pehe} | {qini} | {claim} |".format(
                model=card.get("model", ""),
                terms=len(card.get("loss_terms") or []),
                training=diagnostics.get("training_status", ""),
                verdict=diagnostics.get("tuned_verdict", ""),
                pehe=diagnostics.get("best_pehe", ""),
                qini=diagnostics.get("best_qini", ""),
                claim=str(card.get("safe_claim", "")).replace("|", "/"),
            )
        )

    detail_sections = []
    for card in cards:
        term_rows = "\n".join(
            "| {term} | {formula} | {diagnostic} | {failure_signal} |".format(
                term=str(term.get("term", "")).replace("|", "/"),
                formula=str(term.get("formula", "")).replace("|", "/"),
                diagnostic=str(term.get("diagnostic", "")).replace("|", "/"),
                failure_signal=str(term.get("failure_signal", "")).replace("|", "/"),
            )
            for term in card.get("loss_terms") or []
        )
        detail_sections.append(
            f"""## {card.get("model")} Objective Card

Family: `{card.get("family")}`

Role: {card.get("role")}

### Loss Terms

| Term | Formula | Observable diagnostic | Failure signal |
| --- | --- | --- | --- |
{term_rows}

### Claims

- Safe claim: {card.get("safe_claim")}
- Unsafe claim: {card.get("unsafe_claim")}
- Interview line: {card.get("interview_line")}
- License policy: {card.get("source_license_policy")}

### Evidence

{chr(10).join(f"- `{item}`" for item in card.get("evidence_refs") or [])}
"""
        )
    tracks = payload.get("interview_tracks") or {}
    return f"""# DeepUplift Deep Model Objective Diagnostics

Generated at: `{payload.get("generated_at")}`

This artifact connects each deep uplift architecture to the exact loss terms, observable diagnostics, benchmark evidence, failure signals and safe interview claims.

## Summary

| Metric | Value |
| --- | ---: |
| Models | {summary.get("models", 0)} |
| Loss terms | {summary.get("loss_terms", 0)} |
| Models with training smoke | {summary.get("models_with_training_smoke", 0)} |
| Tuned candidates | {summary.get("tuned_candidates", 0)} |

## Leaderboard

| Model | Loss terms | Training smoke | Tuned verdict | Best PEHE | Best QINI | Safe claim |
| --- | ---: | --- | --- | ---: | ---: | --- |
{chr(10).join(rows)}

## Interview Tracks

- 30s: {tracks.get("30s", "")}
- 5min: {tracks.get("5min", "")}
- 15min: {tracks.get("15min", "")}

{chr(10).join(detail_sections)}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate objective/loss diagnostics for deep uplift models.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--json-output", default="reports/deep_model_objective_diagnostics_latest.json")
    parser.add_argument("--csv-output", default="reports/deep_model_objective_diagnostics_latest.csv")
    parser.add_argument("--doc-output", default="docs/DEEPUplift_DEEP_MODEL_OBJECTIVE_DIAGNOSTICS.md")
    args = parser.parse_args()

    root = Path(args.root)
    payload = build_cards(root)
    json_output = Path(args.json_output)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(Path(args.csv_output), payload.get("cards") or [])
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
                "loss_terms": (payload.get("summary") or {}).get("loss_terms"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
