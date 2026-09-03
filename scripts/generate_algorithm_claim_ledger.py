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


def compact_paths(items: list[str] | tuple[str, ...], limit: int = 3) -> list[str]:
    return [str(item) for item in list(items)[:limit] if item]


def metric(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.4g}"
    except (TypeError, ValueError):
        return str(value)


def promotion_cards_for_model(model_id: str, promotion_payload: dict[str, Any]) -> list[dict[str, Any]]:
    needle = model_id.lower()
    matched: list[dict[str, Any]] = []
    for card in promotion_payload.get("cards") or []:
        haystack = " ".join(
            str(card.get(field, ""))
            for field in ["entity", "primary_blocker", "primary_reason", "safe_claim", "unsafe_claim", "interview_line"]
        ).lower()
        haystack += " " + " ".join(str(item) for item in card.get("signals") or []).lower()
        if needle in haystack:
            matched.append(card)
    return matched


def model_evidence_level(
    model_id: str,
    paper_card: dict[str, Any],
    tuned_card: dict[str, Any],
    objective_card: dict[str, Any],
) -> str:
    if paper_card.get("verdict") in {"accuracy_and_ranking_candidate", "ranking_candidate"}:
        return "paper_level_candidate"
    if tuned_card.get("verdict") in {"tuned_candidate", "ranking_candidate"}:
        return "tuned_candidate"
    if objective_card:
        return "objective_diagnostics"
    model = MODEL_DECONSTRUCTIONS[model_id]
    if model.smoke_evidence:
        return "source_and_smoke"
    return "source_only"


def build_claim_rows(root: Path) -> list[dict[str, Any]]:
    paper = read_json(root / "reports" / "paper_benchmark_interpretation_latest.json")
    tuned = read_json(root / "reports" / "deep_model_tuned_benchmark_latest.json")
    objective = read_json(root / "reports" / "deep_model_objective_diagnostics_latest.json")
    promotion = read_json(root / "reports" / "promotion_launch_cards_latest.json")
    training = read_json(root / "reports" / "deep_model_training_evidence_latest.json")
    model_catalog = read_json(root / "reports" / "model_deconstruction_catalog_latest.json")

    paper_cards = by_model(paper.get("model_scorecards") or [])
    tuned_cards = by_model(tuned.get("model_scorecards") or [])
    objective_cards = by_model(objective.get("cards") or [])
    training_cards = by_model(training.get("leaderboard") or [])
    catalog_cards = {str(row.get("model_id")): row for row in model_catalog.get("models") or [] if row.get("model_id")}

    rows: list[dict[str, Any]] = []
    for model_id, model in MODEL_DECONSTRUCTIONS.items():
        paper_card = paper_cards.get(model_id, {})
        tuned_card = tuned_cards.get(model_id, {})
        objective_card = objective_cards.get(model_id, {})
        training_card = training_cards.get(model_id, {})
        catalog_card = catalog_cards.get(model_id, {})
        promo_cards = promotion_cards_for_model(model_id, promotion)
        promo_decisions = sorted({str(card.get("overall_decision")) for card in promo_cards if card.get("overall_decision")})
        evidence_level = model_evidence_level(model_id, paper_card, tuned_card, objective_card)
        evidence_refs = [
            "reports/model_deconstruction_catalog_latest.json",
            f"docs/model_deconstruction/{model_id}.md",
        ]
        if paper_card:
            evidence_refs.append("reports/paper_benchmark_interpretation_latest.json")
        if tuned_card:
            evidence_refs.append("reports/deep_model_tuned_benchmark_latest.json")
        if objective_card:
            evidence_refs.append("reports/deep_model_objective_diagnostics_latest.json")
        if training_card:
            evidence_refs.append("reports/deep_model_training_evidence_latest.json")
        if promo_cards:
            evidence_refs.append("reports/promotion_launch_cards_latest.json")

        formula = model.formulas[0] if model.formulas else ""
        rows.append(
            {
                "claim_id": f"{model_id}.algorithm_definition",
                "model": model_id,
                "family": model.family,
                "claim_type": "algorithm_definition",
                "claim": f"{model.display_name} is implemented as {model.role}.",
                "formula_or_test": formula,
                "code_refs": compact_paths(model.code_paths, 3),
                "evidence_refs": evidence_refs,
                "evidence_level": evidence_level,
                "benchmark_verdict": paper_card.get("verdict") or tuned_card.get("verdict") or "not_in_latest_benchmark",
                "promotion_decision": "; ".join(promo_decisions) or "not_model_specific",
                "safe_claim": model.interview_line,
                "unsafe_claim": "Do not claim production lift without launch guardrail, OPE and online incrementality evidence.",
                "interview_line": model.interview_line,
                "ui_route": "Model Deconstruction -> Model Review Card",
                "next_action": (model.next_tasks or ("refresh model deconstruction doc",))[0],
            }
        )

        rows.append(
            {
                "claim_id": f"{model_id}.training_objective",
                "model": model_id,
                "family": model.family,
                "claim_type": "training_objective",
                "claim": "The training path has an explicit causal objective or pseudo-outcome contract that can be explained from formula to code.",
                "formula_or_test": " ; ".join(model.formulas[:3]),
                "code_refs": compact_paths(model.code_paths, 4),
                "evidence_refs": evidence_refs,
                "evidence_level": "objective_diagnostics" if objective_card else evidence_level,
                "benchmark_verdict": (objective_card.get("diagnostics") or {}).get("tuned_verdict")
                or tuned_card.get("verdict")
                or paper_card.get("verdict")
                or "not_in_latest_objective_diagnostics",
                "promotion_decision": "; ".join(promo_decisions) or "not_model_specific",
                "safe_claim": objective_card.get("safe_claim")
                or "Can explain objective, call chain and evaluator contract; keep benchmark claims separate.",
                "unsafe_claim": objective_card.get("unsafe_claim")
                or "Do not equate source-level objective explanation with benchmark superiority.",
                "interview_line": objective_card.get("interview_line") or model.interview_line,
                "ui_route": "Model Deconstruction -> Deep Model Objective Diagnostics",
                "next_action": (objective_card.get("next_tasks") or model.next_tasks or ["add objective diagnostics"])[0],
            }
        )

        rows.append(
            {
                "claim_id": f"{model_id}.benchmark_evidence",
                "model": model_id,
                "family": model.family,
                "claim_type": "benchmark_evidence",
                "claim": "Benchmark evidence is separated into paper-level effect accuracy/ranking evidence and tuned deep-model battle evidence.",
                "formula_or_test": (
                    f"PEHE={metric(paper_card.get('pehe_mean_over_datasets'))}; "
                    f"ATE error={metric(paper_card.get('ate_error_mean_over_datasets'))}; "
                    f"QINI={metric(paper_card.get('qini_mean_over_datasets') or tuned_card.get('best_qini'))}"
                ),
                "code_refs": ["scripts/run_paper_benchmark_suite.py", "scripts/run_deep_model_tuned_benchmark.py"],
                "evidence_refs": evidence_refs,
                "evidence_level": evidence_level,
                "benchmark_verdict": paper_card.get("verdict") or tuned_card.get("verdict") or "not_in_latest_benchmark",
                "promotion_decision": "; ".join(promo_decisions) or "not_model_specific",
                "safe_claim": paper_card.get("interview_line")
                or tuned_card.get("interview_line")
                or "Use smoke evidence and model-card explanation until leaderboard evidence exists.",
                "unsafe_claim": "Do not claim SOTA from one smoke, one seed, or a metric that conflicts with policy value.",
                "interview_line": paper_card.get("interview_line") or tuned_card.get("interview_line") or model.interview_line,
                "ui_route": "Evidence Dashboard -> Paper-Level Benchmark Evidence / Deep Model Battle Report",
                "next_action": paper_card.get("watchout") or tuned_card.get("watchout") or "Run multi-seed benchmark before stronger claim.",
            }
        )

        license_sources = []
        for source in model.external_sources:
            license_sources.append(f"{source.name} ({source.license}; {source.license_risk})")
        rows.append(
            {
                "claim_id": f"{model_id}.license_and_adoption",
                "model": model_id,
                "family": model.family,
                "claim_type": "license_and_adoption",
                "claim": "External papers/repos are used as cited references or guarded adapters; code-copy risk is separated from idea adoption.",
                "formula_or_test": "license gate: " + model.license_policy,
                "code_refs": compact_paths(catalog_card.get("code_paths") or model.code_paths, 3),
                "evidence_refs": evidence_refs,
                "evidence_level": "license_gate",
                "benchmark_verdict": paper_card.get("verdict") or tuned_card.get("verdict") or "not_benchmark_claim",
                "promotion_decision": "; ".join(promo_decisions) or "not_model_specific",
                "safe_claim": model.license_policy,
                "unsafe_claim": "Do not copy GPL or unknown-license research code into default runtime; use clean-room implementation or guarded isolation.",
                "interview_line": "我把 paper idea、official repo、local implementation 和 license risk 分开讲，避免把参考代码误包装成可直接商用集成。",
                "ui_route": "Model Deconstruction -> Source map / License gate",
                "next_action": "; ".join(license_sources[:2]) if license_sources else "Keep license policy in model card.",
            }
        )

    return rows


def build_payload(root: Path) -> dict[str, Any]:
    rows = build_claim_rows(root)
    by_claim_type: dict[str, int] = {}
    by_evidence_level: dict[str, int] = {}
    by_promotion: dict[str, int] = {}
    for row in rows:
        by_claim_type[row["claim_type"]] = by_claim_type.get(row["claim_type"], 0) + 1
        by_evidence_level[row["evidence_level"]] = by_evidence_level.get(row["evidence_level"], 0) + 1
        by_promotion[row["promotion_decision"]] = by_promotion.get(row["promotion_decision"], 0) + 1
    return {
        "schema_version": 1,
        "status": "ok",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "summary": {
            "models": len(MODEL_DECONSTRUCTIONS),
            "claims": len(rows),
            "claim_type_counts": by_claim_type,
            "evidence_level_counts": by_evidence_level,
            "promotion_decision_counts": by_promotion,
        },
        "rows": rows,
        "interview_tracks": {
            "30s": "我给每个算法 claim 建了台账：公式、源码、benchmark、license、safe/unsafe claim 都能追到 artifact。",
            "5min": "面试里我不是只说模型名，而是打开 claim ledger：T/DR 作为强基线，CFR/Dragon/EFIN/DESCN 作为深度结构，分别看 objective、PEHE/QINI/policy value 和上线边界。",
            "15min": "这个 ledger 是工程治理层：它把模型拆解、paper benchmark、deep tuned benchmark、objective diagnostics、promotion cards 串起来，防止把 smoke 证据包装成生产结论。",
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
        "claim_id",
        "model",
        "family",
        "claim_type",
        "claim",
        "formula_or_test",
        "code_refs",
        "evidence_refs",
        "evidence_level",
        "benchmark_verdict",
        "promotion_decision",
        "safe_claim",
        "unsafe_claim",
        "interview_line",
        "ui_route",
        "next_action",
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
    overview_rows = []
    for row in rows:
        if row.get("claim_type") == "benchmark_evidence":
            overview_rows.append(
                "| {model} | {level} | {verdict} | {formula} | {safe} |".format(
                    model=row.get("model", ""),
                    level=row.get("evidence_level", ""),
                    verdict=str(row.get("benchmark_verdict", "")).replace("|", "/"),
                    formula=str(row.get("formula_or_test", "")).replace("|", "/"),
                    safe=str(row.get("safe_claim", "")).replace("|", "/"),
                )
            )

    sections = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("model")), []).append(row)
    for model, model_rows in grouped.items():
        claim_rows = "\n".join(
            "| {claim_type} | {level} | {verdict} | {ui} | {safe} | {unsafe} |".format(
                claim_type=str(row.get("claim_type", "")).replace("|", "/"),
                level=str(row.get("evidence_level", "")).replace("|", "/"),
                verdict=str(row.get("benchmark_verdict", "")).replace("|", "/"),
                ui=str(row.get("ui_route", "")).replace("|", "/"),
                safe=str(row.get("safe_claim", "")).replace("|", "/"),
                unsafe=str(row.get("unsafe_claim", "")).replace("|", "/"),
            )
            for row in model_rows
        )
        sections.append(
            f"""## {model}

| Claim type | Evidence level | Benchmark verdict | UI route | Safe claim | Unsafe claim |
| --- | --- | --- | --- | --- | --- |
{claim_rows}
"""
        )

    return f"""# DeepUplift Algorithm Claim Ledger

Generated at: `{payload.get("generated_at")}`

This ledger is the review bridge between algorithm theory, source deconstruction, benchmark evidence, promotion safety and interview talk tracks.

## Summary

| Metric | Value |
| --- | ---: |
| Models | {summary.get("models", 0)} |
| Claim rows | {summary.get("claims", 0)} |

Claim type counts: `{json.dumps(summary.get("claim_type_counts") or {}, ensure_ascii=False, sort_keys=True)}`

Evidence level counts: `{json.dumps(summary.get("evidence_level_counts") or {}, ensure_ascii=False, sort_keys=True)}`

## Interview Tracks

- 30s: {tracks.get("30s", "")}
- 5min: {tracks.get("5min", "")}
- 15min: {tracks.get("15min", "")}

## Benchmark Claim Overview

| Model | Evidence level | Verdict | Metric proof | Safe claim |
| --- | --- | --- | --- | --- |
{chr(10).join(overview_rows)}

{chr(10).join(sections)}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an algorithm claim ledger that ties model claims to code, formulas and evidence.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--json-output", default="reports/algorithm_claim_ledger_latest.json")
    parser.add_argument("--csv-output", default="reports/algorithm_claim_ledger_latest.csv")
    parser.add_argument("--doc-output", default="docs/DEEPUplift_ALGORITHM_CLAIM_LEDGER.md")
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
                "claims": (payload.get("summary") or {}).get("claims"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
