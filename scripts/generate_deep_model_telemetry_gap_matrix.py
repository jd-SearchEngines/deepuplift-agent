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


TELEMETRY_SPECS: dict[str, dict[str, dict[str, str]]] = {
    "CFRNet": {
        "factual outcome loss": {
            "telemetry_family": "outcome_fit",
            "current_signal": "train_loss_last, valid_loss, PEHE and QINI are available through training/tuned benchmark artifacts.",
            "missing_telemetry": "outcome calibration by treatment arm and PEHE-by-segment curve.",
            "why_needed": "CFRNet can reduce factual loss while still ranking heterogeneous treatment effects poorly.",
            "implementation_hook": "trainer history export plus evaluator segment metrics for treated/control validation slices.",
            "pass_gate": "valid outcome loss does not diverge and segment PEHE does not degrade versus T/DR baseline.",
            "fail_action": "keep as architecture evidence and inspect sample size, feature scaling and treatment imbalance.",
            "expected_metric_link": "PEHE, ATE error, QINI and policy value should move in the same direction or the conflict must be explained.",
            "priority": "P1",
        },
        "representation balance / IPM": {
            "telemetry_family": "representation_balance",
            "current_signal": "CFR balance smoke has differentiable MMD/energy loss and training balance terms.",
            "missing_telemetry": "per-epoch IPM/MMD curve, treated-control representation distance histogram, alpha sweep and post-trim ranking delta.",
            "why_needed": "Balance is CFRNet's core causal claim; without representation telemetry, a worse PEHE result is hard to attribute.",
            "implementation_hook": "export phi_x from forward, compute treated/control MMD or mean-L2 per epoch, persist to run manifest and UI chart.",
            "pass_gate": "balance metric improves without erasing effect modifiers, and high/low-alpha ablation has an explained PEHE/QINI tradeoff.",
            "fail_action": "freeze CFRNet at tuned-candidate or source-level claim; do not call it paper-level unless representation telemetry supports the story.",
            "expected_metric_link": "IPM down plus stable PEHE/QINI; IPM down with PEHE worse means over-balancing or sample-size failure.",
            "priority": "P0",
        },
    },
    "DragonNet": {
        "factual outcome loss": {
            "telemetry_family": "outcome_fit",
            "current_signal": "training smoke records total/outcome loss, and tuned benchmark records PEHE/QINI.",
            "missing_telemetry": "factual loss split by treatment arm and CATE calibration by propensity bucket.",
            "why_needed": "DragonNet can fit outcomes while its treatment-effect ranking is weak under poor overlap.",
            "implementation_hook": "add arm-level validation loss and bucketed effect residuals to trainer/evaluator artifacts.",
            "pass_gate": "both arms have stable validation loss and no high-propensity bucket dominates PEHE.",
            "fail_action": "compare against DRLearner nuisance diagnostics before promoting DragonNet.",
            "expected_metric_link": "arm-level fit should support PEHE/ATE error and not just total binary outcome loss.",
            "priority": "P1",
        },
        "propensity auxiliary head": {
            "telemetry_family": "propensity_overlap",
            "current_signal": "aux_loss_last is logged and objective diagnostics mention ESS/overlap.",
            "missing_telemetry": "propensity calibration, overlap histogram, clipped ESS and tail-risk warning by seed.",
            "why_needed": "A propensity head cannot repair missing support; it only exposes the logging policy if telemetry is calibrated.",
            "implementation_hook": "export e_hat, compute calibration bins, ESS after clipping and overlap warnings in evaluator.",
            "pass_gate": "calibration error is bounded, ESS stays usable after clipping, and PEHE/QINI are not driven by near-0/near-1 propensity tails.",
            "fail_action": "fall back to DR/R/DML with explicit nuisance diagnostics or require randomized/holdout data.",
            "expected_metric_link": "better propensity calibration should reduce instability in PEHE, QINI and DR policy value.",
            "priority": "P0",
        },
        "targeted regularization": {
            "telemetry_family": "targeted_regularization",
            "current_signal": "tuned battle can compare tarreg/no-tarreg variants, but the tarreg term is not logged separately.",
            "missing_telemetry": "separate targeted-reg loss curve, epsilon magnitude, beta sweep and instability flag under clipped propensity.",
            "why_needed": "Targeted regularization is DragonNet's signature objective; if it is bundled into total loss, the claim is hard to defend.",
            "implementation_hook": "split dragonnet_loss into outcome, propensity and tarreg components, persist per epoch and expose beta ablation.",
            "pass_gate": "tarreg improves or explains PEHE/QINI/policy movement without exploding under overlap clipping.",
            "fail_action": "keep tarreg as guarded ablation evidence and do not claim targeted regularization helped.",
            "expected_metric_link": "beta sweep should show a stable region where PEHE/QINI do not collapse.",
            "priority": "P0",
        },
    },
    "EFIN": {
        "base response path": {
            "telemetry_family": "outcome_fit",
            "current_signal": "training smoke records shared evaluator metrics and total loss.",
            "missing_telemetry": "base-response calibration and uplift-path contribution ratio.",
            "why_needed": "EFIN can look strong because the base response path dominates, not because treatment interaction is useful.",
            "implementation_hook": "persist base path prediction, uplift path prediction and their variance/contribution ratio.",
            "pass_gate": "uplift path contributes stable signal beyond base response and improves ranking/policy metrics.",
            "fail_action": "describe EFIN as response-model-plus-interaction architecture, not proven uplift superiority.",
            "expected_metric_link": "base calibration plus uplift contribution should explain QINI/policy changes.",
            "priority": "P1",
        },
        "treatment-aware feature interaction": {
            "telemetry_family": "interaction_attention",
            "current_signal": "objective diagnostics document the attention formula and tuned benchmark records ranking evidence.",
            "missing_telemetry": "attention weights, feature-treatment interaction stability, interaction ablation and campaign-sparsity sensitivity.",
            "why_needed": "EFIN's interview value is the treatment-aware interaction mechanism; without exported attention it is just a black-box deep net.",
            "implementation_hook": "return interaction attention from forward under a guarded flag, save top features per seed and compare with no-interaction ablation.",
            "pass_gate": "attention is stable across seeds and interaction ablation changes QINI/policy value in the expected direction.",
            "fail_action": "keep EFIN as GPL/clean-room architecture reference and avoid claiming interpretable attention evidence.",
            "expected_metric_link": "stable interaction attention should align with uplift@K, QINI and business policy value.",
            "priority": "P0",
        },
        "uplift path": {
            "telemetry_family": "uplift_path",
            "current_signal": "policy value, QINI and uplift ranking metrics exist in tuned benchmark.",
            "missing_telemetry": "uplift-path variance, treatment auxiliary loss split and seed stability by top-K segment.",
            "why_needed": "A strong offline interaction signal may be unstable across campaigns, seeds or sparse treatment IDs.",
            "implementation_hook": "export tau contribution statistics and bootstrap top-K policy variance in tuned benchmark runner.",
            "pass_gate": "top-K uplift/policy value remains positive across seeds with confidence interval not dominated by one slice.",
            "fail_action": "limit the claim to ranking-candidate and ask for larger campaign logs or holdout validation.",
            "expected_metric_link": "uplift-path stability should support AUUC/QINI and policy value with bootstrap CI.",
            "priority": "P1",
        },
    },
    "DESCN": {
        "propensity head": {
            "telemetry_family": "propensity_overlap",
            "current_signal": "objective diagnostics list propensity head and training smoke records aux loss.",
            "missing_telemetry": "propensity calibration, overlap histogram and exposure-head drift by seed.",
            "why_needed": "DESCN models exposure, but biased exposure remains a causal risk unless support is measured.",
            "implementation_hook": "export p_hat from DESCN forward, compute calibration/ESS/overlap and attach warnings to run card.",
            "pass_gate": "calibrated propensity and sufficient overlap before DESCN is compared as an uplift candidate.",
            "fail_action": "use DESCN as classification/entire-space architecture evidence only.",
            "expected_metric_link": "propensity telemetry should explain when DESCN ranking gains are credible versus exposure leakage.",
            "priority": "P0",
        },
        "entire-space response heads": {
            "telemetry_family": "head_consistency",
            "current_signal": "tuned benchmark has ranking/policy rows but does not persist per-head semantics.",
            "missing_telemetry": "mu0/mu1/tau/estr/escr head distributions, arm-level calibration and consistency residuals.",
            "why_needed": "Entire-space heads are powerful but easy to overfit on small tabular data without head-level observability.",
            "implementation_hook": "persist head outputs from forward and compute consistency residuals on validation batches.",
            "pass_gate": "mu0/mu1/tau heads are calibrated and cross-head residuals remain bounded across seeds.",
            "fail_action": "keep DESCN as ranking candidate, not clean CATE estimation evidence.",
            "expected_metric_link": "head consistency should align with PEHE/ATE error, while ranking-only gains are explicitly labeled.",
            "priority": "P0",
        },
        "cross-head treatment effect constraints": {
            "telemetry_family": "cross_head_constraint",
            "current_signal": "objective diagnostics document cross_tr/cross_cr and tuned ablation records some tradeoff evidence.",
            "missing_telemetry": "cross-head loss component, constraint residual curve, ablation over constraint weight and task-boundary guard.",
            "why_needed": "Many heads can fit tiny data while CATE accuracy remains weak; constraints need their own evidence.",
            "implementation_hook": "split esx_loss components and export per-term curves plus no-cross-constraint ablation.",
            "pass_gate": "constraint term improves at least one decision metric without worsening PEHE beyond an explained boundary.",
            "fail_action": "state that DESCN is classification/entire-space guarded until regression CATE evidence is stronger.",
            "expected_metric_link": "constraint residual should help explain PEHE-vs-QINI tradeoffs in battle cards.",
            "priority": "P0",
        },
    },
}


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def by_model(rows: list[dict[str, Any]], key: str = "model") -> dict[str, dict[str, Any]]:
    return {str(row.get(key)): row for row in rows if isinstance(row, dict) and row.get(key)}


def flatten(value: Any) -> str:
    if isinstance(value, list):
        return "; ".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return "" if value is None else str(value)


def metric(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.4g}"
    except (TypeError, ValueError):
        return str(value)


def telemetry_status(spec: dict[str, str]) -> str:
    text = f"{spec.get('current_signal', '')} {spec.get('missing_telemetry', '')}".lower()
    if "missing" in text or "not logged" in text or "not persist" in text:
        return "partial_needs_instrumentation"
    return "observable"


def build_rows(root: Path) -> list[dict[str, Any]]:
    objective = read_json(root / "reports" / "deep_model_objective_diagnostics_latest.json")
    training = read_json(root / "reports" / "deep_model_training_evidence_latest.json")
    tuned = read_json(root / "reports" / "deep_model_tuned_benchmark_latest.json")
    recipes = read_json(root / "reports" / "model_upgrade_recipe_cards_latest.json")
    cfr_balance = read_json(root / "reports" / "cfrnet_differentiable_balance_smoke_latest.json")
    cfr_training = read_json(root / "reports" / "cfrnet_training_balance_smoke_latest.json")

    objective_cards = by_model(objective.get("cards") or [])
    training_cards = by_model(training.get("leaderboard") or [])
    tuned_cards = by_model(tuned.get("model_scorecards") or [])
    recipe_cards = by_model(recipes.get("rows") or [])

    rows: list[dict[str, Any]] = []
    for model_id in DEEP_MODELS:
        model = MODEL_DECONSTRUCTIONS[model_id]
        objective_card = objective_cards.get(model_id, {})
        training_card = training_cards.get(model_id, {})
        tuned_card = tuned_cards.get(model_id, {})
        recipe_card = recipe_cards.get(model_id, {})
        loss_terms = objective_card.get("loss_terms") or []
        for loss in loss_terms:
            term = str(loss.get("term") or "")
            spec = TELEMETRY_SPECS.get(model_id, {}).get(term, {})
            evidence_refs = [
                "reports/deep_model_objective_diagnostics_latest.json",
                "reports/deep_model_training_evidence_latest.json",
                "reports/deep_model_tuned_benchmark_latest.json",
                "reports/model_upgrade_recipe_cards_latest.json",
                f"docs/model_deconstruction/{model_id}.md",
            ]
            if model_id == "CFRNet":
                if cfr_balance:
                    evidence_refs.append("reports/cfrnet_differentiable_balance_smoke_latest.json")
                if cfr_training:
                    evidence_refs.append("reports/cfrnet_training_balance_smoke_latest.json")
            rows.append(
                {
                    "telemetry_id": f"{model_id}.{term.lower().replace(' / ', '_').replace(' ', '_')}",
                    "model": model_id,
                    "display_name": model.display_name,
                    "family": model.family,
                    "loss_term": term,
                    "formula": loss.get("formula"),
                    "objective_diagnostic": loss.get("diagnostic"),
                    "failure_signal": loss.get("failure_signal"),
                    "telemetry_family": spec.get("telemetry_family", "general"),
                    "telemetry_status": telemetry_status(spec),
                    "current_signal": spec.get("current_signal", "objective diagnostics only"),
                    "current_metrics": {
                        "training_status": training_card.get("status"),
                        "train_loss_last": training_card.get("train_loss_last"),
                        "aux_loss_last": training_card.get("aux_loss_last"),
                        "tuned_verdict": tuned_card.get("verdict"),
                        "best_pehe": tuned_card.get("best_pehe"),
                        "best_qini": tuned_card.get("best_qini"),
                        "recipe_goal": recipe_card.get("upgrade_goal"),
                    },
                    "missing_telemetry": spec.get("missing_telemetry", "model-specific telemetry spec missing"),
                    "why_needed": spec.get("why_needed", "needed to connect the loss term to benchmark and policy outcomes"),
                    "implementation_hook": spec.get("implementation_hook", "add guarded trainer/evaluator artifact export"),
                    "pass_gate": spec.get("pass_gate", "telemetry is persisted and explains at least one benchmark movement"),
                    "fail_action": spec.get("fail_action", "keep as source/objective evidence only"),
                    "expected_metric_link": spec.get("expected_metric_link", "PEHE/QINI/policy value movement should be explained"),
                    "priority": spec.get("priority", "P2"),
                    "next_command": "python3 scripts/smoke_deep_model_training_evidence.py && python3 scripts/generate_deep_model_objective_diagnostics.py",
                    "ui_route": "Model Deconstruction -> Deep Model Telemetry Gap Matrix -> Evidence Dashboard",
                    "interview_line": (
                        f"{model_id} / {term}: current metrics "
                        f"PEHE={metric(tuned_card.get('best_pehe'))}, QINI={metric(tuned_card.get('best_qini'))}; "
                        f"next telemetry is {spec.get('telemetry_family', 'general')}."
                    ),
                    "safe_claim": "可以说该 loss term 有源码/公式/现有指标证据，并且缺失 telemetry 已被显式列入工程门禁。",
                    "blocked_claim": "不能把 loss 名称或单次 smoke 指标包装成 paper-level 或 production lift 结论。",
                    "code_refs": list(model.code_paths),
                    "evidence_refs": evidence_refs,
                }
            )
    return rows


def build_payload(root: Path) -> dict[str, Any]:
    rows = build_rows(root)
    by_priority: dict[str, int] = {}
    by_family: dict[str, int] = {}
    by_status: dict[str, int] = {}
    for row in rows:
        by_priority[str(row.get("priority"))] = by_priority.get(str(row.get("priority")), 0) + 1
        by_family[str(row.get("telemetry_family"))] = by_family.get(str(row.get("telemetry_family")), 0) + 1
        by_status[str(row.get("telemetry_status"))] = by_status.get(str(row.get("telemetry_status")), 0) + 1
    return {
        "schema_version": 1,
        "status": "ok",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "summary": {
            "models": len({row["model"] for row in rows}),
            "telemetry_rows": len(rows),
            "p0_rows": by_priority.get("P0", 0),
            "priority_counts": by_priority,
            "telemetry_family_counts": by_family,
            "telemetry_status_counts": by_status,
        },
        "rows": rows,
        "interview_tracks": {
            "30s": "这张表回答深度模型为什么输或赢：不是只看 PEHE/QINI，而是看对应 loss term 还缺什么 telemetry。",
            "5min": "CFRNet 看 representation balance，DragonNet 看 propensity/tarreg，EFIN 看 interaction attention，DESCN 看 head consistency；每行都有 hook、pass gate 和 fail action。",
            "15min": "它把 paper-level benchmark 失败归因推进到工程实现：导出 phi/e_hat/head/attention，做 per-epoch 曲线、seed 稳定性、ablation，再回写 promotion matrix。",
        },
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "telemetry_id",
        "model",
        "loss_term",
        "formula",
        "telemetry_family",
        "telemetry_status",
        "current_signal",
        "missing_telemetry",
        "why_needed",
        "implementation_hook",
        "pass_gate",
        "fail_action",
        "expected_metric_link",
        "priority",
        "ui_route",
        "interview_line",
        "safe_claim",
        "blocked_claim",
        "code_refs",
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
    overview = "\n".join(
        "| {model} | {term} | {family} | {priority} | {missing} | {gate} |".format(
            model=str(row.get("model", "")).replace("|", "/"),
            term=str(row.get("loss_term", "")).replace("|", "/"),
            family=str(row.get("telemetry_family", "")).replace("|", "/"),
            priority=str(row.get("priority", "")).replace("|", "/"),
            missing=str(row.get("missing_telemetry", "")).replace("|", "/"),
            gate=str(row.get("pass_gate", "")).replace("|", "/"),
        )
        for row in rows
    )
    sections = []
    for row in rows:
        sections.append(
            f"""## {row.get("telemetry_id")}

Model: `{row.get("model")}`
Loss term: `{row.get("loss_term")}`
Formula: `{row.get("formula")}`
Telemetry family: `{row.get("telemetry_family")}`
Priority: `{row.get("priority")}`

Current signal: {row.get("current_signal")}

Missing telemetry: {row.get("missing_telemetry")}

Why needed: {row.get("why_needed")}

Implementation hook: {row.get("implementation_hook")}

Pass gate: {row.get("pass_gate")}

Fail action: {row.get("fail_action")}

Expected metric link: {row.get("expected_metric_link")}

Interview line: {row.get("interview_line")}
"""
        )
    return f"""# DeepUplift Deep Model Telemetry Gap Matrix

Generated at: `{payload.get("generated_at")}`

This artifact turns deep uplift objectives into an observability plan. It connects each loss term to current evidence, missing telemetry, implementation hooks, pass gates, failure actions and interview-safe claims.

## Summary

| Metric | Value |
| --- | ---: |
| Models | {summary.get("models", 0)} |
| Telemetry rows | {summary.get("telemetry_rows", 0)} |
| P0 rows | {summary.get("p0_rows", 0)} |

Priority counts: `{json.dumps(summary.get("priority_counts") or {}, ensure_ascii=False, sort_keys=True)}`

Telemetry families: `{json.dumps(summary.get("telemetry_family_counts") or {}, ensure_ascii=False, sort_keys=True)}`

## Interview Tracks

- 30s: {tracks.get("30s", "")}
- 5min: {tracks.get("5min", "")}
- 15min: {tracks.get("15min", "")}

## Gap Overview

| Model | Loss term | Telemetry family | Priority | Missing telemetry | Pass gate |
| --- | --- | --- | --- | --- | --- |
{overview}

{chr(10).join(sections)}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deep model telemetry gap matrix from objective and benchmark evidence.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--json-output", default="reports/deep_model_telemetry_gap_matrix_latest.json")
    parser.add_argument("--csv-output", default="reports/deep_model_telemetry_gap_matrix_latest.csv")
    parser.add_argument("--doc-output", default="docs/DEEPUplift_DEEP_MODEL_TELEMETRY_GAP_MATRIX.md")
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
                "rows": (payload.get("summary") or {}).get("telemetry_rows"),
                "models": (payload.get("summary") or {}).get("models"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
