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


def metric(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.4g}"
    except (TypeError, ValueError):
        return str(value)


def recipe_for_row(row: dict[str, Any]) -> dict[str, Any]:
    model_id = str(row.get("model"))
    model = MODEL_DECONSTRUCTIONS.get(model_id)
    stage = str(row.get("evidence_stage") or "")
    tier = str(row.get("promotion_tier") or "")
    refs = list(row.get("evidence_refs") or [])
    if "reports/model_evidence_promotion_matrix_latest.json" not in refs:
        refs.append("reports/model_evidence_promotion_matrix_latest.json")

    if stage == "paper_level_review_candidate":
        return {
            "recipe_id": f"{model_id}.pilot_readiness",
            "model": model_id,
            "current_stage": stage,
            "promotion_tier": tier,
            "upgrade_goal": "from paper-level interview anchor to pilot-readiness candidate",
            "hypothesis": "Paper-level CATE/ranking evidence is stable enough to justify an offline-to-shadow review, but not enough for production lift claims.",
            "command": "PYTHON_BIN=python3 PAPER_BENCH_PRESET=benchmark-v3 PAPER_BENCH_ROWS=400 scripts/full_regression_check.sh",
            "optional_env": "Use SKIP_UI=1 for no-browser regression; keep online pilot evidence outside default smoke.",
            "dataset_scope": "IHDP / ACIC-style / known-CATE synthetic plus one business scenario holdout before shadow scoring.",
            "metrics_to_watch": ["PEHE", "ATE error", "QINI", "AUUC", "policy value", "bootstrap CI", "warning audit", "promotion card"],
            "pass_gate": "multi-seed metrics remain stable, metric conflicts are explained, no high-severity launch blocker is model-specific, and OPE/holdout plan is documented.",
            "fail_action": "Keep as T0 interview/benchmark anchor, open a promotion card, and do not claim online incrementality.",
            "expected_artifacts": [
                "reports/paper_benchmark_latest.json",
                "reports/paper_benchmark_interpretation_latest.json",
                "reports/regression_warning_audit_latest.json",
                "reports/promotion_launch_cards_latest.json",
            ],
            "owner_role": "ML lead + experimentation owner",
            "ui_route": "Evidence -> Model Evidence Promotion Matrix -> Paper-Level Benchmark Evidence",
            "interview_line": f"{model_id} is strong enough to discuss as a paper-level candidate, but I still require OPE/holdout before any production lift claim.",
            "evidence_refs": refs,
        }

    if stage == "tuned_benchmark_candidate":
        return {
            "recipe_id": f"{model_id}.tuned_to_paper_candidate",
            "model": model_id,
            "current_stage": stage,
            "promotion_tier": tier,
            "upgrade_goal": "from tuned/ranking candidate to paper-level benchmark candidate",
            "hypothesis": "The tuned variant improves ranking or policy value on selected datasets; the next step is to prove the effect is not metric cherry-picking.",
            "command": "python3 scripts/run_deep_model_tuned_benchmark.py --preset tuned-v1 --rows 400 --known-effect-bootstrap-samples 16",
            "optional_env": "Keep heavy external repos guarded; do not copy GPL/unknown code into the default runtime.",
            "dataset_scope": "same datasets as benchmark-v3, with per-dataset win/loss interpretation and baseline deltas against T/DR.",
            "metrics_to_watch": ["delta PEHE vs baseline", "delta QINI vs baseline", "policy top10 value", "loss delta", "license gate"],
            "pass_gate": "at least one stable dataset win plus no severe license/adoption blocker; paper benchmark interpretation records safe/unsafe claim.",
            "fail_action": "Keep as T1 ranking candidate or T2 architecture evidence; document why T/DR wins on small tabular data.",
            "expected_artifacts": [
                "reports/deep_model_tuned_benchmark_latest.json",
                "reports/deep_model_tuned_benchmark_battle_cards_latest.csv",
                "docs/DEEPUplift_DEEP_MODEL_TUNING_BENCHMARK.md",
            ],
            "owner_role": "ML researcher + license reviewer",
            "ui_route": "Model Deconstruction -> Deep Model Battle Report -> License gate",
            "interview_line": f"{model_id} is not marketed as a universal winner; the recipe asks whether tuning survives baseline and license review.",
            "evidence_refs": refs,
        }

    if stage == "objective_diagnostics_ready":
        return {
            "recipe_id": f"{model_id}.objective_to_benchmark",
            "model": model_id,
            "current_stage": stage,
            "promotion_tier": tier,
            "upgrade_goal": "from loss/objective explanation to benchmark evidence",
            "hypothesis": "The source and loss terms are explainable, but the model needs telemetry that links each loss term to PEHE/QINI/policy movement.",
            "command": "python3 scripts/smoke_deep_model_training_evidence.py && python3 scripts/generate_deep_model_objective_diagnostics.py",
            "optional_env": "Add model-specific probes as guarded scripts before adding new default dependencies.",
            "dataset_scope": "known-CATE synthetic for PEHE/ATE plus deep training smoke for loss curves and evidence manifest.",
            "metrics_to_watch": ["train loss", "valid loss", "aux loss", "PEHE", "QINI", "policy value", "objective card gaps"],
            "pass_gate": "loss terms are separately observable, run manifest exists, and at least one benchmark metric improves or failure attribution is explicit.",
            "fail_action": "Keep as architecture evidence; add missing telemetry such as balance curve, propensity calibration or cross-head diagnostics.",
            "expected_artifacts": [
                "reports/deep_model_training_evidence_latest.json",
                "reports/deep_model_objective_diagnostics_latest.json",
                "reports/model_evidence_promotion_matrix_latest.json",
            ],
            "owner_role": "deep-model owner",
            "ui_route": "Model Deconstruction -> Deep Model Objective Diagnostics",
            "interview_line": f"{model_id} is valuable only if I can tie its loss terms to metrics; this recipe prevents loss-name storytelling.",
            "evidence_refs": refs,
        }

    return {
        "recipe_id": f"{model_id}.source_to_smoke",
        "model": model_id,
        "current_stage": stage,
        "promotion_tier": tier,
        "upgrade_goal": "from source deconstruction to executable smoke/benchmark evidence",
        "hypothesis": "The model is explainable at source level, but it needs a runnable smoke or benchmark row before stronger interview claims.",
        "command": "python3 scripts/generate_model_deconstruction_docs.py && python3 scripts/smoke_model_deconstruction_agent.py",
        "optional_env": "Use guarded adapters for optional backends; keep default demo runnable even when dependencies are absent.",
        "dataset_scope": "small synthetic smoke first; promote to benchmark-v3 only after the adapter has stable inputs, outputs and model card.",
        "metrics_to_watch": ["golden question pass", "model card completeness", "evidence manifest", "QINI/AUUC smoke", "license gate"],
        "pass_gate": "source docs, code refs, formulas, license policy and at least one runnable smoke artifact are present.",
        "fail_action": "Keep as research backlog; do not describe it as implemented benchmark evidence.",
        "expected_artifacts": [
            "docs/model_deconstruction",
            "reports/model_deconstruction_agent_latest.json",
            "reports/algorithm_claim_ledger_latest.json",
        ],
        "owner_role": "model-platform owner",
        "ui_route": "Model Deconstruction -> Source Call Chain -> Algorithm Claim Ledger",
        "interview_line": f"{model_id} is currently source-level unless the recipe creates smoke/benchmark artifacts.",
        "evidence_refs": refs,
    }


def build_payload(root: Path) -> dict[str, Any]:
    matrix = read_json(root / "reports" / "model_evidence_promotion_matrix_latest.json")
    rows = [recipe_for_row(row) for row in matrix.get("rows") or []]
    if not rows:
        rows = [
            recipe_for_row(
                {
                    "model": model_id,
                    "evidence_stage": "source_deconstruction_ready",
                    "promotion_tier": "T3_research_backlog",
                    "evidence_refs": [f"docs/model_deconstruction/{model_id}.md"],
                }
            )
            for model_id in MODEL_DECONSTRUCTIONS
        ]
    by_goal: dict[str, int] = {}
    by_owner: dict[str, int] = {}
    for row in rows:
        by_goal[row["upgrade_goal"]] = by_goal.get(row["upgrade_goal"], 0) + 1
        by_owner[row["owner_role"]] = by_owner.get(row["owner_role"], 0) + 1
    return {
        "schema_version": 1,
        "status": "ok",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "summary": {
            "recipes": len(rows),
            "models": len({row["model"] for row in rows}),
            "upgrade_goal_counts": by_goal,
            "owner_role_counts": by_owner,
        },
        "rows": rows,
        "interview_tracks": {
            "30s": "promotion matrix 说明缺什么，upgrade recipe card 说明下一步怎么补证据。",
            "5min": "每个模型都有 hypothesis、命令、指标、pass gate、fail action 和 expected artifacts，评审时能直接变成迭代计划。",
            "15min": "这层把算法深度和工程治理接起来：loss telemetry、paper benchmark、license guard、OPE/holdout 不再是口头计划，而是可执行 recipe。",
        },
    }


def flatten(value: Any) -> str:
    if isinstance(value, list):
        return "; ".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return "" if value is None else str(value)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "recipe_id",
        "model",
        "current_stage",
        "promotion_tier",
        "upgrade_goal",
        "hypothesis",
        "command",
        "optional_env",
        "dataset_scope",
        "metrics_to_watch",
        "pass_gate",
        "fail_action",
        "expected_artifacts",
        "owner_role",
        "ui_route",
        "interview_line",
        "evidence_refs",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: flatten(row.get(field)) for field in fields})


def build_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary") or {}
    rows = payload.get("rows") or []
    tracks = payload.get("interview_tracks") or {}
    table = "\n".join(
        "| {model} | {goal} | {tier} | {owner} | {gate} |".format(
            model=row.get("model", ""),
            goal=str(row.get("upgrade_goal", "")).replace("|", "/"),
            tier=str(row.get("promotion_tier", "")).replace("|", "/"),
            owner=str(row.get("owner_role", "")).replace("|", "/"),
            gate=str(row.get("pass_gate", "")).replace("|", "/"),
        )
        for row in rows
    )
    sections = []
    for row in rows:
        sections.append(
            f"""## {row.get("model")}

Recipe: `{row.get("recipe_id")}`
Current stage: `{row.get("current_stage")}`
Promotion tier: `{row.get("promotion_tier")}`

Hypothesis: {row.get("hypothesis")}

Command:

```bash
{row.get("command")}
```

Dataset scope: {row.get("dataset_scope")}

Metrics to watch: {flatten(row.get("metrics_to_watch"))}

Pass gate: {row.get("pass_gate")}

Fail action: {row.get("fail_action")}

Expected artifacts: {flatten(row.get("expected_artifacts"))}

Interview line: {row.get("interview_line")}
"""
        )
    return f"""# DeepUplift Model Upgrade Recipe Cards

Generated at: `{payload.get("generated_at")}`

This artifact turns the promotion matrix into executable next experiments. It is for review questions like "What exactly would you do next to upgrade this model's evidence?"

## Summary

| Metric | Value |
| --- | ---: |
| Models | {summary.get("models", 0)} |
| Recipes | {summary.get("recipes", 0)} |

Upgrade goal counts: `{json.dumps(summary.get("upgrade_goal_counts") or {}, ensure_ascii=False, sort_keys=True)}`

## Interview Tracks

- 30s: {tracks.get("30s", "")}
- 5min: {tracks.get("5min", "")}
- 15min: {tracks.get("15min", "")}

## Recipe Overview

| Model | Upgrade goal | Promotion tier | Owner | Pass gate |
| --- | --- | --- | --- | --- |
{table}

{chr(10).join(sections)}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate model upgrade recipe cards from the DeepUplift promotion matrix.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--json-output", default="reports/model_upgrade_recipe_cards_latest.json")
    parser.add_argument("--csv-output", default="reports/model_upgrade_recipe_cards_latest.csv")
    parser.add_argument("--doc-output", default="docs/DEEPUplift_MODEL_UPGRADE_RECIPE_CARDS.md")
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
                "recipes": (payload.get("summary") or {}).get("recipes"),
                "models": (payload.get("summary") or {}).get("models"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
