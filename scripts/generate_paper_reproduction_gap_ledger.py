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


def by_model(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("model")): row for row in rows if isinstance(row, dict) and row.get("model")}


def proof_level(model_id: str, paper_card: dict[str, Any], tuned_card: dict[str, Any], objective_card: dict[str, Any], training_card: dict[str, Any]) -> str:
    if paper_card.get("verdict") in {"accuracy_and_ranking_candidate", "ranking_candidate"}:
        return "paper_level_benchmark_candidate"
    if tuned_card.get("verdict") in {"tuned_candidate", "ranking_candidate"}:
        return "tuned_benchmark_candidate"
    if objective_card:
        return "objective_and_loss_diagnostics"
    if training_card.get("status") == "ok":
        return "training_smoke"
    model = MODEL_DECONSTRUCTIONS[model_id]
    if model.code_paths:
        return "source_deconstruction"
    return "reference_only"


def adoption_decision(license_text: str, license_risk: str, source_type: str) -> str:
    lowered = f"{license_text} {license_risk} {source_type}".lower()
    if "gpl" in lowered:
        return "idea_only_clean_room_or_isolated_plugin"
    if "unknown" in lowered or "not verified" in lowered:
        return "read_only_until_license_verified"
    if "paper" in lowered and "code" not in lowered:
        return "paper_formula_reference"
    return "permissive_reference_or_guarded_adapter"


def safe_claim_for_level(level: str, model_id: str) -> str:
    if level == "paper_level_benchmark_candidate":
        return f"{model_id} has local source deconstruction plus paper-level benchmark evidence; still needs business holdout/OPE before production claims."
    if level == "tuned_benchmark_candidate":
        return f"{model_id} has tuned benchmark candidate evidence; claim the dataset/metric boundary explicitly."
    if level == "objective_and_loss_diagnostics":
        return f"{model_id} can be explained from formula/loss to diagnostics, but benchmark superiority is not established."
    if level == "training_smoke":
        return f"{model_id} has executable training smoke evidence, not a paper-level reproduction win."
    if level == "source_deconstruction":
        return f"{model_id} has source-level explanation and implementation notes; run smoke/benchmark before stronger claims."
    return f"{model_id} is currently a cited research reference only."


def build_rows(root: Path) -> list[dict[str, Any]]:
    paper = read_json(root / "reports" / "paper_benchmark_interpretation_latest.json")
    tuned = read_json(root / "reports" / "deep_model_tuned_benchmark_latest.json")
    objective = read_json(root / "reports" / "deep_model_objective_diagnostics_latest.json")
    training = read_json(root / "reports" / "deep_model_training_evidence_latest.json")

    paper_cards = by_model(paper.get("model_scorecards") or [])
    tuned_cards = by_model(tuned.get("model_scorecards") or [])
    objective_cards = by_model(objective.get("cards") or [])
    training_cards = by_model(training.get("leaderboard") or [])

    rows: list[dict[str, Any]] = []
    for model_id, model in MODEL_DECONSTRUCTIONS.items():
        paper_card = paper_cards.get(model_id, {})
        tuned_card = tuned_cards.get(model_id, {})
        objective_card = objective_cards.get(model_id, {})
        training_card = training_cards.get(model_id, {})
        level = proof_level(model_id, paper_card, tuned_card, objective_card, training_card)
        common_evidence = [
            f"docs/model_deconstruction/{model_id}.md",
            "reports/model_deconstruction_catalog_latest.json",
        ]
        if paper_card:
            common_evidence.append("reports/paper_benchmark_interpretation_latest.json")
        if tuned_card:
            common_evidence.append("reports/deep_model_tuned_benchmark_latest.json")
        if objective_card:
            common_evidence.append("reports/deep_model_objective_diagnostics_latest.json")
        if training_card:
            common_evidence.append("reports/deep_model_training_evidence_latest.json")

        for source in model.external_sources:
            adoption = adoption_decision(source.license, source.license_risk, source.source_type)
            rows.append(
                {
                    "model": model_id,
                    "display_name": model.display_name,
                    "family": model.family,
                    "source_name": source.name,
                    "source_url": source.url,
                    "source_type": source.source_type,
                    "source_license": source.license,
                    "license_risk": source.license_risk,
                    "adoption_decision": adoption,
                    "paper_or_repo_claim": source.adoption,
                    "local_reproduction_scope": model.current_status,
                    "proof_level": level,
                    "matched_formula": " ; ".join(model.formulas[:2]),
                    "matched_code_refs": list(model.code_paths[:4]),
                    "benchmark_verdict": paper_card.get("verdict") or tuned_card.get("verdict") or "not_in_latest_benchmark",
                    "best_pehe": paper_card.get("pehe_mean_over_datasets") or tuned_card.get("best_pehe"),
                    "best_qini": paper_card.get("qini_mean_over_datasets") or tuned_card.get("best_qini"),
                    "implementation_gaps": list(model.implementation_gaps),
                    "validation_needed": list(model.next_tasks),
                    "evidence_refs": common_evidence,
                    "safe_claim": safe_claim_for_level(level, model_id),
                    "unsafe_claim": "Do not describe a cited paper or external repo as fully reproduced unless local code, benchmark evidence, and license gate all support it.",
                    "interview_line": model.interview_line,
                }
            )
    return rows


def build_payload(root: Path) -> dict[str, Any]:
    rows = build_rows(root)
    by_level: dict[str, int] = {}
    by_adoption: dict[str, int] = {}
    by_license: dict[str, int] = {}
    for row in rows:
        by_level[row["proof_level"]] = by_level.get(row["proof_level"], 0) + 1
        by_adoption[row["adoption_decision"]] = by_adoption.get(row["adoption_decision"], 0) + 1
        by_license[row["source_license"]] = by_license.get(row["source_license"], 0) + 1
    return {
        "schema_version": 1,
        "status": "ok",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "summary": {
            "models": len(MODEL_DECONSTRUCTIONS),
            "source_rows": len(rows),
            "proof_level_counts": by_level,
            "adoption_decision_counts": by_adoption,
            "license_counts": by_license,
        },
        "rows": rows,
        "interview_tracks": {
            "30s": "我把论文复现讲成证据等级：paper idea、source deconstruction、training smoke、paper-level benchmark、上线 guardrail。",
            "5min": "每个外部来源都有 license 和 adoption decision；GPL/unknown 只做思想参考，permissive repo 才能考虑 guarded adapter。",
            "15min": "复现不是照搬代码，而是把公式、loss、源码、benchmark 和 failure mode 全部映射到本仓库 evidence；没有 evidence 的 claim 就保守表达。",
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
        "source_name",
        "source_url",
        "source_type",
        "source_license",
        "license_risk",
        "adoption_decision",
        "paper_or_repo_claim",
        "local_reproduction_scope",
        "proof_level",
        "matched_formula",
        "matched_code_refs",
        "benchmark_verdict",
        "best_pehe",
        "best_qini",
        "implementation_gaps",
        "validation_needed",
        "evidence_refs",
        "safe_claim",
        "unsafe_claim",
        "interview_line",
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
    overview = []
    for row in rows:
        overview.append(
            "| {model} | {source} | {license} | {adoption} | {level} | {verdict} | {safe} |".format(
                model=row.get("model", ""),
                source=str(row.get("source_name", "")).replace("|", "/"),
                license=str(row.get("source_license", "")).replace("|", "/"),
                adoption=str(row.get("adoption_decision", "")).replace("|", "/"),
                level=str(row.get("proof_level", "")).replace("|", "/"),
                verdict=str(row.get("benchmark_verdict", "")).replace("|", "/"),
                safe=str(row.get("safe_claim", "")).replace("|", "/"),
            )
        )
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("model")), []).append(row)
    sections = []
    for model, model_rows in grouped.items():
        source_rows = "\n".join(
            "| {source} | {type} | {license} | {adoption} | {risk} |".format(
                source=f"[{str(row.get('source_name', '')).replace('|', '/')}]({row.get('source_url', '')})",
                type=str(row.get("source_type", "")).replace("|", "/"),
                license=str(row.get("source_license", "")).replace("|", "/"),
                adoption=str(row.get("adoption_decision", "")).replace("|", "/"),
                risk=str(row.get("license_risk", "")).replace("|", "/"),
            )
            for row in model_rows
        )
        first = model_rows[0]
        sections.append(
            f"""## {model}

Proof level: `{first.get("proof_level")}`
Benchmark verdict: `{first.get("benchmark_verdict")}`

Formula/code scope: {first.get("matched_formula")}

| Source | Type | License | Adoption decision | Risk |
| --- | --- | --- | --- | --- |
{source_rows}

Safe claim: {first.get("safe_claim")}

Unsafe claim: {first.get("unsafe_claim")}

Implementation gaps: {flatten(first.get("implementation_gaps"))}
"""
        )
    return f"""# DeepUplift Paper Reproduction Gap Ledger

Generated at: `{payload.get("generated_at")}`

This artifact separates paper/reference adoption from verified local reproduction. It is designed for interview and architecture review questions like "Did you really reproduce this paper, or did you only cite it?"

## Summary

| Metric | Value |
| --- | ---: |
| Models | {summary.get("models", 0)} |
| Source rows | {summary.get("source_rows", 0)} |

Proof level counts: `{json.dumps(summary.get("proof_level_counts") or {}, ensure_ascii=False, sort_keys=True)}`

Adoption decision counts: `{json.dumps(summary.get("adoption_decision_counts") or {}, ensure_ascii=False, sort_keys=True)}`

## Interview Tracks

- 30s: {tracks.get("30s", "")}
- 5min: {tracks.get("5min", "")}
- 15min: {tracks.get("15min", "")}

## Reproduction Overview

| Model | Source | License | Adoption | Proof level | Benchmark verdict | Safe claim |
| --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(overview)}

{chr(10).join(sections)}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper/repo reproduction gap ledger for DeepUplift model claims.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--json-output", default="reports/paper_reproduction_gap_ledger_latest.json")
    parser.add_argument("--csv-output", default="reports/paper_reproduction_gap_ledger_latest.csv")
    parser.add_argument("--doc-output", default="docs/DEEPUplift_PAPER_REPRODUCTION_GAP_LEDGER.md")
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
                "source_rows": (payload.get("summary") or {}).get("source_rows"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
