from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def flatten(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return "" if value is None else str(value)


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug[:96] or "unknown"


def group_triage_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if not isinstance(row, dict):
            continue
        grouped[(str(row.get("domain") or "unknown"), str(row.get("entity") or "unknown"))].append(row)
    return grouped


def score_penalty_for(row: dict[str, Any]) -> int:
    family = str(row.get("risk_family") or "")
    boundary = str(row.get("claim_boundary") or "")
    severity = str(row.get("severity") or "")
    penalty = {"high": 24, "medium": 10, "low": 3}.get(severity, 5)
    if row.get("launch_blocking"):
        penalty += 12
    if family == "business_value":
        penalty += 10
    elif family == "support_and_overlap":
        penalty += 8
    elif family == "refutation_stability":
        penalty += 6
    elif family in {"model_quality", "deep_model_failure_attribution"}:
        penalty += 5
    elif family == "evidence_governance":
        penalty += 4
    if boundary == "smoke_path_only_not_quality_claim":
        penalty = min(penalty, 4)
    return penalty


def evidence_grade(
    launch_blockers: int,
    review_rows: int,
    smoke_only_rows: int,
    promotion_decision: str,
    risk_families: set[str],
) -> str:
    if launch_blockers or promotion_decision == "block_promotion":
        return "D_blocked_for_pilot"
    if {"business_value", "support_and_overlap"} & risk_families:
        return "C_shadow_only_with_guardrails"
    if review_rows:
        return "B_offline_review_candidate"
    if smoke_only_rows:
        return "B_smoke_path_validated"
    return "A_shadow_candidate"


def pilot_gate_for(
    launch_blockers: int,
    review_rows: int,
    promotion_decision: str,
    domain: str,
    risk_families: set[str],
) -> str:
    if launch_blockers or promotion_decision == "block_promotion":
        return "blocked"
    if domain in {"paper_benchmark", "deep_model_battle"} or risk_families & {"model_quality", "deep_model_failure_attribution"}:
        return "paper_review_candidate"
    if review_rows or promotion_decision == "review_before_promotion":
        return "shadow_candidate"
    return "monitor"


def pilot_path_for(gate: str) -> str:
    paths = {
        "blocked": "offline_only -> fix blocker -> rerun benchmark/warning triage -> promotion review",
        "paper_review_candidate": "paper-level benchmark -> multi-seed CI -> failure attribution -> shadow review",
        "shadow_candidate": "offline benchmark/OPE -> shadow scoring -> small holdout/A/B -> ramp-up",
        "monitor": "shadow monitor -> guardrail dashboard -> small holdout/A/B if business owner accepts risk",
    }
    return paths.get(gate, paths["blocked"])


def next_gate_for(gate: str) -> str:
    next_gates = {
        "blocked": "do_not_shadow_until_p0_blockers_clear",
        "paper_review_candidate": "paper_review_then_shadow_candidate",
        "shadow_candidate": "shadow_scoring_with_holdout_design",
        "monitor": "guarded_shadow_or_low_risk_holdout",
    }
    return next_gates.get(gate, "owner_review")


def rollback_conditions_for(gate: str, risk_families: set[str]) -> list[str]:
    conditions = ["negative online incremental profit", "guardrail metric regression", "artifact manifest or run diff missing"]
    if gate == "blocked":
        conditions.insert(0, "any launch-blocking row remains unresolved")
    if "support_and_overlap" in risk_families:
        conditions.append("weak overlap or post-trim metric sign flip")
    if "refutation_stability" in risk_families:
        conditions.append("refutation or bootstrap CI fails")
    if "business_value" in risk_families:
        conditions.append("ROI/iROAS or policy value is non-positive")
    if "model_quality" in risk_families or "deep_model_failure_attribution" in risk_families:
        conditions.append("benchmark metric conflict worsens against baseline")
    return conditions


def required_artifacts_for(gate: str, risk_families: set[str]) -> list[str]:
    artifacts = [
        "reports/promotion_launch_cards_latest.json",
        "reports/regression_warning_triage_latest.json",
        "reports/evidence_store_latest.json",
    ]
    if gate in {"paper_review_candidate", "shadow_candidate"} or risk_families & {"model_quality", "deep_model_failure_attribution"}:
        artifacts += [
            "reports/paper_benchmark_latest.json",
            "reports/paper_benchmark_interpretation_latest.json",
            "reports/deep_model_tuned_benchmark_latest.json",
        ]
    if risk_families & {"business_value", "support_and_overlap", "refutation_stability"}:
        artifacts += [
            "reports/ope_engine_smoke_latest.json",
            "reports/industrial_scenario_compare_smoke_latest.json",
            "reports/regression_warning_audit_latest.json",
        ]
    return sorted(set(artifacts))


def safe_claim_for(gate: str) -> str:
    claims = {
        "blocked": "可以说平台能识别上线 blocker 并给出 owner/实验路线；不能说该策略可以试点。",
        "paper_review_candidate": "可以说模型进入 paper-level/机制评审候选；不能说已具备生产上线证据。",
        "shadow_candidate": "可以说具备 shadow/offline review 候选资格；A/B 前仍要补 holdout/OPE/rollback 证据。",
        "monitor": "可以说风险较低且可进入监控式 shadow；不能把 monitor 解释成正式上线批准。",
    }
    return claims.get(gate, "可以说已进入 owner review；不能绕过证据门禁。")


def unsafe_claim_for(gate: str) -> str:
    claims = {
        "blocked": "不要把 demo/smoke/leaderboard 包装成可上线 ROI 或稳定增量。",
        "paper_review_candidate": "不要把单次 paper benchmark 或局部机制证据包装成 SOTA 或生产效果。",
        "shadow_candidate": "不要跳过 shadow/OPE/holdout 直接宣称能进正式流量。",
        "monitor": "不要把无 blocker 当成无风险，线上 incrementality 仍需实验验证。",
    }
    return claims.get(gate, "不要把 review 状态包装成 launch approval。")


def build_scorecard(card: dict[str, Any], triage_rows: list[dict[str, Any]]) -> dict[str, Any]:
    domain = str(card.get("domain") or "unknown")
    entity = str(card.get("entity") or "unknown")
    promotion_decision = str(card.get("overall_decision") or "monitor")
    risk_families = {str(row.get("risk_family") or "unknown") for row in triage_rows}
    claim_boundaries = {str(row.get("claim_boundary") or "unknown") for row in triage_rows}
    action_lanes = {str(row.get("action_lane") or "unknown") for row in triage_rows}
    launch_blockers = sum(1 for row in triage_rows if row.get("launch_blocking"))
    review_rows = sum(1 for row in triage_rows if row.get("triage_status") == "requires_review_before_shadow")
    smoke_only_rows = sum(1 for row in triage_rows if row.get("smoke_only"))
    penalty = sum(score_penalty_for(row) for row in triage_rows)
    if promotion_decision == "block_promotion":
        penalty += 20
    elif promotion_decision == "review_before_promotion":
        penalty += 8
    readiness_score = max(0, min(100, 100 - penalty))
    gate = pilot_gate_for(launch_blockers, review_rows, promotion_decision, domain, risk_families)
    grade = evidence_grade(launch_blockers, review_rows, smoke_only_rows, promotion_decision, risk_families)
    next_experiments = []
    for row in triage_rows:
        experiment = str(row.get("next_experiment") or "")
        if experiment and experiment not in next_experiments:
            next_experiments.append(experiment)
    for experiment in card.get("next_experiments") or []:
        experiment = str(experiment)
        if experiment and experiment not in next_experiments:
            next_experiments.append(experiment)

    scorecard_id = f"PRS-{slugify(domain)}-{slugify(entity)}"
    return {
        "scorecard_id": scorecard_id,
        "card_id": card.get("card_id"),
        "domain": domain,
        "entity": entity,
        "pilot_gate": gate,
        "readiness_score": readiness_score,
        "evidence_grade": grade,
        "promotion_decision": promotion_decision,
        "promotion_stage": card.get("promotion_stage"),
        "owner_role": card.get("owner_role") or "Platform owner",
        "warning_count": card.get("warning_count") or len(triage_rows),
        "launch_blockers": launch_blockers,
        "review_rows": review_rows,
        "smoke_only_rows": smoke_only_rows,
        "risk_families": sorted(risk_families),
        "claim_boundaries": sorted(claim_boundaries),
        "action_lanes": sorted(action_lanes),
        "primary_blocker": card.get("primary_blocker"),
        "primary_reason": card.get("primary_reason"),
        "next_gate": next_gate_for(gate),
        "pilot_path": pilot_path_for(gate),
        "next_experiments": next_experiments[:6],
        "rollback_conditions": rollback_conditions_for(gate, risk_families),
        "required_artifacts": required_artifacts_for(gate, risk_families),
        "safe_claim": safe_claim_for(gate),
        "unsafe_claim": unsafe_claim_for(gate),
        "demo_route": "Evidence Dashboard -> Pilot Readiness Scorecard -> Promotion Cards -> Warning Triage",
        "interview_line": (
            f"`{entity}` gate=`{gate}` score=`{readiness_score}`: "
            f"我把 offline score、warning triage 和 launch card 合成 pilot decision，而不是只看 leaderboard。"
        ),
    }


def build_payload(cards_payload: dict[str, Any], triage_payload: dict[str, Any]) -> dict[str, Any]:
    cards = [card for card in cards_payload.get("cards") or [] if isinstance(card, dict)]
    triage_rows = [row for row in triage_payload.get("rows") or [] if isinstance(row, dict)]
    grouped = group_triage_rows(triage_rows)
    scorecards = [build_scorecard(card, grouped.get((str(card.get("domain")), str(card.get("entity"))), [])) for card in cards]
    scorecards.sort(
        key=lambda row: (
            {"blocked": 0, "paper_review_candidate": 1, "shadow_candidate": 2, "monitor": 3}.get(
                str(row.get("pilot_gate")), 9
            ),
            int(row.get("readiness_score") or 0),
            str(row.get("domain") or ""),
            str(row.get("entity") or ""),
        )
    )
    gate_counts = Counter(str(row.get("pilot_gate") or "unknown") for row in scorecards)
    grade_counts = Counter(str(row.get("evidence_grade") or "unknown") for row in scorecards)
    owner_counts = Counter(str(row.get("owner_role") or "unknown") for row in scorecards)
    blocked = gate_counts.get("blocked", 0)
    shadow = gate_counts.get("shadow_candidate", 0)
    monitor = gate_counts.get("monitor", 0)
    paper = gate_counts.get("paper_review_candidate", 0)
    pilot_gate = "blocked" if blocked else ("review_required" if paper or shadow else "monitor")
    return {
        "schema_version": 1,
        "status": "ok" if scorecards else "fail",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "summary": {
            "pilot_gate": pilot_gate,
            "scorecards": len(scorecards),
            "blocked": blocked,
            "paper_review_candidates": paper,
            "shadow_candidates": shadow,
            "monitor": monitor,
            "gate_counts": dict(gate_counts),
            "evidence_grade_counts": dict(grade_counts),
            "owner_counts": dict(owner_counts),
            "source_launch_gate": (cards_payload.get("summary") or {}).get("launch_gate"),
            "source_triage_gate": (triage_payload.get("summary") or {}).get("triage_gate"),
            "top_pilot_candidates": [
                row.get("scorecard_id")
                for row in sorted(scorecards, key=lambda item: int(item.get("readiness_score") or 0), reverse=True)
                if row.get("pilot_gate") in {"shadow_candidate", "monitor"}
            ][:5],
        },
        "scorecards": scorecards,
        "gate_definitions": {
            "blocked": "Do not enter shadow or A/B until launch-blocking rows are cleared.",
            "paper_review_candidate": "Good for model/paper review; needs stronger benchmark and failure attribution before shadow.",
            "shadow_candidate": "Eligible for shadow scoring with holdout/OPE/rollback plan.",
            "monitor": "Low current warning pressure, but still needs guardrail monitoring and online incrementality evidence.",
        },
        "interview_tracks": {
            "30s": f"Pilot readiness turns launch review into a gate: {blocked} blocked, {paper} paper-review, {shadow} shadow candidates, {monitor} monitor rows.",
            "5min": "Open one blocked row and one candidate row. Explain score, primary blocker, next gate, required artifacts and rollback conditions.",
            "15min": "Trace benchmark -> warning audit -> warning triage -> promotion card -> pilot readiness. This proves the project is a causal decision platform, not a model zoo.",
            "30min": "Use the scorecard as a launch review agenda: owner, risk family, claim boundary, OPE/holdout plan, rollback rule, and which artifact proves each step.",
        },
        "source_artifacts": {
            "promotion_launch_cards": "reports/promotion_launch_cards_latest.json",
            "regression_warning_triage": "reports/regression_warning_triage_latest.json",
            "regression_warning_audit": "reports/regression_warning_audit_latest.json",
        },
    }


CSV_FIELDS = [
    "scorecard_id",
    "card_id",
    "domain",
    "entity",
    "pilot_gate",
    "readiness_score",
    "evidence_grade",
    "promotion_decision",
    "promotion_stage",
    "owner_role",
    "warning_count",
    "launch_blockers",
    "review_rows",
    "smoke_only_rows",
    "risk_families",
    "claim_boundaries",
    "action_lanes",
    "primary_blocker",
    "primary_reason",
    "next_gate",
    "pilot_path",
    "next_experiments",
    "rollback_conditions",
    "required_artifacts",
    "safe_claim",
    "unsafe_claim",
    "demo_route",
    "interview_line",
]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: flatten(row.get(field)) for field in CSV_FIELDS})


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    summary = payload.get("summary") or {}
    rows = payload.get("scorecards") or []
    tracks = payload.get("interview_tracks") or {}
    lines = [
        "# DeepUplift Pilot Readiness Scorecard",
        "",
        f"Generated at: `{payload.get('generated_at')}`",
        "",
        "This report converts launch warnings into a pilot decision board. It answers whether a model/scenario should stay offline, enter paper review, move to shadow scoring, or only be monitored.",
        "",
        "## Summary",
        "",
        f"- Pilot gate: `{summary.get('pilot_gate')}`",
        f"- Scorecards: `{summary.get('scorecards')}`",
        f"- Blocked: `{summary.get('blocked')}`",
        f"- Paper review candidates: `{summary.get('paper_review_candidates')}`",
        f"- Shadow candidates: `{summary.get('shadow_candidates')}`",
        f"- Monitor: `{summary.get('monitor')}`",
        f"- Evidence grades: `{json.dumps(summary.get('evidence_grade_counts') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Gate Definitions",
        "",
    ]
    for gate, definition in (payload.get("gate_definitions") or {}).items():
        lines.append(f"- `{gate}`: {definition}")
    lines.extend(
        [
            "",
            "## Interview Tracks",
            "",
            f"- 30 seconds: {tracks.get('30s', '')}",
            f"- 5 minutes: {tracks.get('5min', '')}",
            f"- 15 minutes: {tracks.get('15min', '')}",
            f"- 30 minutes: {tracks.get('30min', '')}",
            "",
            "## Scorecard Table",
            "",
            "| Gate | Score | Grade | Domain | Entity | Owner | Primary blocker | Next gate |",
            "| --- | ---: | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in rows[:80]:
        lines.append(
            f"| {row.get('pilot_gate')} | {row.get('readiness_score')} | {row.get('evidence_grade')} | "
            f"{row.get('domain')} | {row.get('entity')} | {row.get('owner_role')} | "
            f"{row.get('primary_blocker')} | {row.get('next_gate')} |"
        )
    lines.extend(["", "## Demo Script", ""])
    lines.append("1. Open Evidence Dashboard -> Pilot Readiness Scorecard.")
    lines.append("2. Filter `blocked` and explain why the system refuses pilot despite runnable benchmark evidence.")
    lines.append("3. Filter `shadow_candidate` or `monitor` and explain the exact next gate, rollback conditions and required artifacts.")
    lines.append("4. Drill down into Promotion Launch Cards and Regression Warning Triage to show the evidence chain.")
    lines.extend(["", "## Source Artifacts", ""])
    for name, artifact in (payload.get("source_artifacts") or {}).items():
        lines.append(f"- {name}: `{artifact}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate pilot readiness scorecards from launch cards and warning triage.")
    parser.add_argument("--promotion-cards", default="reports/promotion_launch_cards_latest.json")
    parser.add_argument("--warning-triage", default="reports/regression_warning_triage_latest.json")
    parser.add_argument("--json-output", default="reports/pilot_readiness_scorecard_latest.json")
    parser.add_argument("--csv-output", default="reports/pilot_readiness_scorecard_latest.csv")
    parser.add_argument("--doc-output", default="docs/DEEPUplift_PILOT_READINESS_SCORECARD.md")
    args = parser.parse_args()

    payload = build_payload(read_json(ROOT / args.promotion_cards), read_json(ROOT / args.warning_triage))
    write_json(ROOT / args.json_output, payload)
    write_csv(ROOT / args.csv_output, payload.get("scorecards") or [])
    write_markdown(ROOT / args.doc_output, payload)
    print(
        json.dumps(
            {
                "status": payload.get("status"),
                "json": args.json_output,
                "csv": args.csv_output,
                "markdown": args.doc_output,
                "scorecards": (payload.get("summary") or {}).get("scorecards"),
                "blocked": (payload.get("summary") or {}).get("blocked"),
                "shadow_candidates": (payload.get("summary") or {}).get("shadow_candidates"),
                "pilot_gate": (payload.get("summary") or {}).get("pilot_gate"),
            },
            ensure_ascii=False,
        )
    )
    if payload.get("status") != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
