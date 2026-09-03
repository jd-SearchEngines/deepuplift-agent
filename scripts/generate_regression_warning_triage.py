from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
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


def risk_family_for(domain: str, signal: str) -> str:
    signal_l = signal.lower()
    if "policy_value" in signal_l or "roi" in signal_l or "observed_uplift" in signal_l:
        return "business_value"
    if "overlap" in signal_l or "trim" in signal_l or "coverage" in signal_l or "ess" in signal_l:
        return "support_and_overlap"
    if "sensitivity" in signal_l or "refutation" in signal_l:
        return "refutation_stability"
    if "readiness" in signal_l:
        return "launch_readiness"
    if any(term in signal_l for term in ["qini", "auuc", "pehe", "ate", "baseline", "metric_conflict"]):
        return "model_quality"
    if domain == "evidence_governance":
        return "evidence_governance"
    if domain == "deep_model_battle":
        return "deep_model_failure_attribution"
    return "general_review"


def is_smoke_only(domain: str, signal: str) -> bool:
    signal_l = signal.lower()
    return domain == "regression_smoke" and any(
        term in signal_l for term in ["qini", "auuc", "sensitivity", "run_status"]
    )


def claim_boundary_for(severity: str, domain: str, signal: str, family: str) -> str:
    if is_smoke_only(domain, signal):
        return "smoke_path_only_not_quality_claim"
    if severity == "high" and family in {"business_value", "support_and_overlap", "launch_readiness"}:
        return "production_blocker"
    if domain in {"paper_benchmark", "deep_model_battle"}:
        return "benchmark_or_research_blocker"
    if severity == "medium":
        return "review_before_pilot"
    return "monitor_with_evidence"


def action_lane_for(family: str, smoke_only: bool) -> str:
    if smoke_only:
        return "interpret_smoke_as_path_validation"
    if family == "business_value":
        return "cost_threshold_and_holdout_ablation"
    if family == "support_and_overlap":
        return "overlap_trim_or_randomized_exploration"
    if family == "refutation_stability":
        return "multi_seed_refutation_and_bootstrap"
    if family == "launch_readiness":
        return "shadow_gate_and_readiness_owner"
    if family in {"model_quality", "deep_model_failure_attribution"}:
        return "benchmark_rerun_and_failure_attribution"
    if family == "evidence_governance":
        return "artifact_manifest_refresh"
    return "owner_review"


def triage_status_for(severity: str, smoke_only: bool, claim_boundary: str) -> str:
    if smoke_only:
        return "expected_smoke_caveat"
    if claim_boundary == "production_blocker":
        return "blocks_pilot_until_experiment"
    if severity == "medium":
        return "requires_review_before_shadow"
    return "monitor"


def safe_claim_for(claim_boundary: str) -> str:
    claims = {
        "smoke_path_only_not_quality_claim": "可以说默认训练/评估路径可执行；不能说该 smoke split 证明模型有效。",
        "production_blocker": "可以说系统识别出了上线 blocker；不能说该模型/策略已经可试点。",
        "benchmark_or_research_blocker": "可以说该模型需要更多 benchmark 或失败归因；不能把研究证据包装成生产证据。",
        "review_before_pilot": "可以说这是 shadow/offline review 候选；进入 A/B 前要补证据。",
        "monitor_with_evidence": "可以说当前只需监控，但仍要保留 evidence manifest 和 rollback 条件。",
    }
    return claims.get(claim_boundary, "可以说该风险已被记录；不能跳过 owner review。")


def unsafe_claim_for(claim_boundary: str) -> str:
    claims = {
        "smoke_path_only_not_quality_claim": "不要把小样本 smoke 的 QINI/AUUC 当成模型质量结论。",
        "production_blocker": "不要说已经能上线、稳定提升 ROI 或已通过 incrementality 验证。",
        "benchmark_or_research_blocker": "不要只凭单次 leaderboard 或消融覆盖率宣称 paper-level 复现完成。",
        "review_before_pilot": "不要绕过 shadow/OPE/holdout 直接进入正式 A/B。",
        "monitor_with_evidence": "不要把 monitor 解释成正式上线批准。",
    }
    return claims.get(claim_boundary, "不要把 warning 消失等同于风险消失。")


def card_lookup(cards_payload: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for card in cards_payload.get("cards") or []:
        if isinstance(card, dict):
            lookup[(str(card.get("domain") or ""), str(card.get("entity") or ""))] = card
    return lookup


def build_rows(audit: dict[str, Any], cards_payload: dict[str, Any]) -> list[dict[str, Any]]:
    cards = card_lookup(cards_payload)
    rows: list[dict[str, Any]] = []
    for source in audit.get("warnings") or []:
        if not isinstance(source, dict):
            continue
        domain = str(source.get("domain") or "unknown")
        entity = str(source.get("entity") or "unknown")
        signal = str(source.get("signal") or "unknown")
        severity = str(source.get("severity") or "low")
        family = risk_family_for(domain, signal)
        smoke_only = is_smoke_only(domain, signal)
        claim_boundary = claim_boundary_for(severity, domain, signal, family)
        action_lane = action_lane_for(family, smoke_only)
        triage_status = triage_status_for(severity, smoke_only, claim_boundary)
        card = cards.get((domain, entity)) or {}
        rows.append(
            {
                "severity": severity,
                "domain": domain,
                "entity": entity,
                "signal": signal,
                "value": source.get("value"),
                "risk_family": family,
                "triage_status": triage_status,
                "claim_boundary": claim_boundary,
                "action_lane": action_lane,
                "smoke_only": smoke_only,
                "launch_blocking": bool(severity == "high" and not smoke_only),
                "promotion_decision": card.get("overall_decision") or source.get("promotion_decision"),
                "promotion_stage": card.get("promotion_stage"),
                "owner_role": card.get("owner_role") or source.get("owner_role"),
                "safe_claim": safe_claim_for(claim_boundary),
                "unsafe_claim": unsafe_claim_for(claim_boundary),
                "recommended_action": source.get("recommended_action"),
                "next_experiment": source.get("next_experiment") or (card.get("next_experiments") or [""])[0],
                "ui_route": source.get("ui_route"),
                "artifact": source.get("artifact"),
                "why_it_matters": source.get("why_it_matters"),
                "interview_line": source.get("interview_line"),
                "card_id": card.get("card_id"),
            }
        )
    rows.sort(
        key=lambda row: (
            0 if row.get("launch_blocking") else 1,
            {"high": 0, "medium": 1, "low": 2}.get(str(row.get("severity")), 9),
            str(row.get("risk_family")),
            str(row.get("entity")),
        )
    )
    return rows


def build_payload(audit: dict[str, Any], cards_payload: dict[str, Any]) -> dict[str, Any]:
    rows = build_rows(audit, cards_payload)
    family_counts = Counter(str(row.get("risk_family") or "unknown") for row in rows)
    action_counts = Counter(str(row.get("action_lane") or "unknown") for row in rows)
    claim_counts = Counter(str(row.get("claim_boundary") or "unknown") for row in rows)
    status_counts = Counter(str(row.get("triage_status") or "unknown") for row in rows)
    owner_counts = Counter(str(row.get("owner_role") or "unknown") for row in rows)
    smoke_only_rows = sum(1 for row in rows if row.get("smoke_only"))
    launch_blocking_rows = sum(1 for row in rows if row.get("launch_blocking"))
    review_rows = sum(1 for row in rows if row.get("triage_status") == "requires_review_before_shadow")
    triage_gate = "blocked_for_pilot" if launch_blocking_rows else ("review_required" if review_rows else "monitor")
    return {
        "schema_version": 1,
        "status": "ok" if rows else "fail",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "summary": {
            "warning_rows": len(rows),
            "smoke_only_rows": smoke_only_rows,
            "launch_blocking_rows": launch_blocking_rows,
            "review_rows": review_rows,
            "triage_gate": triage_gate,
            "risk_family_counts": dict(family_counts),
            "action_lane_counts": dict(action_counts),
            "claim_boundary_counts": dict(claim_counts),
            "triage_status_counts": dict(status_counts),
            "owner_counts": dict(owner_counts),
            "source_launch_gate": (audit.get("summary") or {}).get("launch_gate"),
            "promotion_launch_gate": (cards_payload.get("summary") or {}).get("launch_gate"),
        },
        "rows": rows,
        "interview_tracks": {
            "30s": f"I triage warnings instead of hiding them: {smoke_only_rows} rows are smoke-only caveats, {launch_blocking_rows} rows block pilot, and the gate is `{triage_gate}`.",
            "5min": "Start with a smoke-only caveat, then a production blocker. This shows the system separates executable proof from quality proof and launch proof.",
            "15min": "Walk risk_family -> claim_boundary -> action_lane -> owner -> next_experiment, then connect the row to promotion cards and online validation.",
            "30min": "Use the triage table as a launch review agenda: business value, support/overlap, refutation stability, model quality, readiness, and evidence governance each has a different owner and experiment.",
        },
        "source_artifacts": {
            "regression_warning_audit": "reports/regression_warning_audit_latest.json",
            "promotion_launch_cards": "reports/promotion_launch_cards_latest.json",
        },
    }


CSV_FIELDS = [
    "severity",
    "domain",
    "entity",
    "signal",
    "value",
    "risk_family",
    "triage_status",
    "claim_boundary",
    "action_lane",
    "smoke_only",
    "launch_blocking",
    "promotion_decision",
    "promotion_stage",
    "owner_role",
    "safe_claim",
    "unsafe_claim",
    "recommended_action",
    "next_experiment",
    "ui_route",
    "artifact",
    "card_id",
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
    rows = payload.get("rows") or []
    tracks = payload.get("interview_tracks") or {}
    lines = [
        "# DeepUplift Regression Warning Triage",
        "",
        f"Generated at: `{payload.get('generated_at')}`",
        "",
        "This report does not suppress warnings. It classifies each warning into a claim boundary and an action lane so reviewers can distinguish smoke-only caveats from pilot-blocking launch risks.",
        "",
        "## Summary",
        "",
        f"- Warning rows: `{summary.get('warning_rows')}`",
        f"- Smoke-only caveats: `{summary.get('smoke_only_rows')}`",
        f"- Launch-blocking rows: `{summary.get('launch_blocking_rows')}`",
        f"- Review rows: `{summary.get('review_rows')}`",
        f"- Triage gate: `{summary.get('triage_gate')}`",
        f"- Risk families: `{json.dumps(summary.get('risk_family_counts') or {}, ensure_ascii=False, sort_keys=True)}`",
        f"- Claim boundaries: `{json.dumps(summary.get('claim_boundary_counts') or {}, ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Interview Tracks",
        "",
        f"- 30 seconds: {tracks.get('30s', '')}",
        f"- 5 minutes: {tracks.get('5min', '')}",
        f"- 15 minutes: {tracks.get('15min', '')}",
        f"- 30 minutes: {tracks.get('30min', '')}",
        "",
        "## Triage Table",
        "",
        "| Severity | Domain | Entity | Signal | Risk family | Boundary | Action lane | Launch blocking |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows[:80]:
        lines.append(
            f"| {row.get('severity')} | {row.get('domain')} | {row.get('entity')} | {row.get('signal')} | "
            f"{row.get('risk_family')} | {row.get('claim_boundary')} | {row.get('action_lane')} | {row.get('launch_blocking')} |"
        )
    lines.extend(["", "## Evidence Links", ""])
    for name, artifact in (payload.get("source_artifacts") or {}).items():
        lines.append(f"- {name}: `{artifact}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify regression warning audit rows into launch-review triage lanes.")
    parser.add_argument("--warning-audit", default="reports/regression_warning_audit_latest.json")
    parser.add_argument("--promotion-cards", default="reports/promotion_launch_cards_latest.json")
    parser.add_argument("--json-output", default="reports/regression_warning_triage_latest.json")
    parser.add_argument("--csv-output", default="reports/regression_warning_triage_latest.csv")
    parser.add_argument("--doc-output", default="docs/DEEPUplift_REGRESSION_WARNING_TRIAGE.md")
    args = parser.parse_args()

    payload = build_payload(read_json(ROOT / args.warning_audit), read_json(ROOT / args.promotion_cards))
    write_json(ROOT / args.json_output, payload)
    write_csv(ROOT / args.csv_output, payload.get("rows") or [])
    write_markdown(ROOT / args.doc_output, payload)
    print(
        json.dumps(
            {
                "status": payload.get("status"),
                "json": args.json_output,
                "csv": args.csv_output,
                "markdown": args.doc_output,
                "warning_rows": (payload.get("summary") or {}).get("warning_rows"),
                "smoke_only_rows": (payload.get("summary") or {}).get("smoke_only_rows"),
                "launch_blocking_rows": (payload.get("summary") or {}).get("launch_blocking_rows"),
                "triage_gate": (payload.get("summary") or {}).get("triage_gate"),
            },
            ensure_ascii=False,
        )
    )
    if payload.get("status") != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
