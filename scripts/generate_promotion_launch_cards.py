from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DECISION_RANK = {"block_promotion": 0, "review_before_promotion": 1, "monitor": 2}
SEVERITY_RANK = {"high": 0, "medium": 1, "low": 2}


def read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def latest(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug[:80] or "unknown"


def decision_from_rows(rows: list[dict[str, Any]]) -> str:
    decisions = [str(row.get("promotion_decision") or "monitor") for row in rows]
    return min(decisions, key=lambda item: DECISION_RANK.get(item, 9)) if decisions else "monitor"


def promotion_stage(decision: str, severity_counts: dict[str, int]) -> str:
    if decision == "block_promotion":
        return "research_only_blocked_for_online"
    if decision == "review_before_promotion":
        return "shadow_or_offline_review_only"
    if severity_counts.get("low", 0):
        return "monitor_with_evidence"
    return "candidate_for_shadow"


def card_safe_claim(decision: str, domain: str) -> str:
    if decision == "block_promotion":
        return "可以作为风险识别、failure attribution 和补实验路线证据；不能作为可上线效果 claim。"
    if domain == "deep_model_battle":
        return "可以讲深度模型架构、loss、ablation 和与 T/DR baseline 的差距；上线 claim 需要更强 benchmark。"
    if decision == "review_before_promotion":
        return "可以作为离线候选或 shadow scoring 候选；需要补充 holdout/OPE/多 seed 证据后再试点。"
    return "可以进入低风险 shadow 观察，但仍需在线实验确认 incrementality。"


def card_unsafe_claim(decision: str) -> str:
    if decision == "block_promotion":
        return "不要说它已经能上线、稳定提升 ROI，或已经通过因果有效性验证。"
    if decision == "review_before_promotion":
        return "不要只凭 leaderboard 第一名或单 seed 指标宣称生产可用。"
    return "不要把 monitor 状态解释成正式上线批准。"


def before_shadow(decision: str, signals: list[str]) -> str:
    if decision == "block_promotion":
        return "先修复 high-severity blocker，刷新 warning audit，并给出 owner/root cause。"
    if any("overlap" in signal for signal in signals):
        return "补 overlap/support 诊断并确认 trim 后核心指标不翻转。"
    return "补充多 seed/bootstrapped smoke，并确认 artifact manifest、readiness 和 run diff 完整。"


def before_ab(decision: str, signals: list[str]) -> str:
    if decision == "block_promotion":
        return "不得进入 A/B；先完成 shadow 或 holdout 前置验证。"
    if any(term in " ".join(signals) for term in ["policy_value", "roi", "observed_uplift"]):
        return "完成成本阈值/ROI ablation，并锁定 rollback 指标。"
    return "完成 shadow scoring、OPE/ESS/coverage 检查，并定义实验 primary metric 与 guardrail。"


def before_ramp(decision: str) -> str:
    if decision != "monitor":
        return "需要小流量 A/B 或 holdout 显著为正，并清除 P0/P1 blocker。"
    return "需要持续监控 calibration、ROI、负向 uplift 人群和 drift。"


def metrics_to_watch(domain: str, signals: list[str]) -> list[str]:
    metrics = ["QINI/AUUC", "policy value", "bootstrap CI", "readiness level"]
    if domain in {"paper_benchmark", "deep_model_battle"}:
        metrics += ["PEHE", "ATE error", "baseline delta", "loss curve"]
    if any("overlap" in signal for signal in signals):
        metrics += ["weak overlap rate", "post-trim QINI"]
    if any("sensitivity" in signal for signal in signals):
        metrics += ["refutation p-value", "permutation null gap"]
    if any("roi" in signal or "policy_value" in signal for signal in signals):
        metrics += ["ROI/iROAS", "budget used", "incremental profit"]
    return sorted(set(metrics))


def group_warning_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        domain = str(row.get("domain") or "unknown")
        entity = str(row.get("entity") or "unknown")
        grouped[(domain, entity)].append(row)
    return grouped


def build_card(domain: str, entity: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = sorted(rows, key=lambda row: SEVERITY_RANK.get(str(row.get("severity")), 9))
    severity_counts = Counter(str(row.get("severity") or "low") for row in rows)
    owner_counts = Counter(str(row.get("owner_role") or "Platform owner") for row in rows)
    decision = decision_from_rows(rows)
    signals = sorted({str(row.get("signal") or "unknown") for row in rows})
    experiments = []
    routes = []
    evidence_refs = []
    for row in rows:
        action = str(row.get("next_experiment") or row.get("recommended_action") or "")
        if action and action not in experiments:
            experiments.append(action)
        route = str(row.get("ui_route") or "")
        if route and route not in routes:
            routes.append(route)
        artifact = str(row.get("artifact") or "")
        if artifact and artifact not in evidence_refs:
            evidence_refs.append(artifact)
    top_row = rows[0] if rows else {}
    owner = owner_counts.most_common(1)[0][0] if owner_counts else "Platform owner"
    stage = promotion_stage(decision, dict(severity_counts))
    card_id = f"PLC-{slugify(domain)}-{slugify(entity)}"

    return {
        "card_id": card_id,
        "domain": domain,
        "entity": entity,
        "warning_count": len(rows),
        "severity_counts": dict(severity_counts),
        "overall_decision": decision,
        "promotion_stage": stage,
        "owner_role": owner,
        "signals": signals,
        "primary_blocker": str(top_row.get("signal") or "none"),
        "primary_blocker_value": top_row.get("value"),
        "primary_reason": str(top_row.get("why_it_matters") or "No blocker found."),
        "safe_claim": card_safe_claim(decision, domain),
        "unsafe_claim": card_unsafe_claim(decision),
        "before_shadow": before_shadow(decision, signals),
        "before_ab": before_ab(decision, signals),
        "before_ramp": before_ramp(decision),
        "next_experiments": experiments[:5],
        "metrics_to_watch": metrics_to_watch(domain, signals),
        "ui_routes": routes[:3],
        "evidence_refs": evidence_refs[:5],
        "interview_line": str(top_row.get("interview_line") or "我会把风险变成 promotion card，而不是只展示 leaderboard。"),
        "demo_script_30s": (
            f"`{entity}` 当前 decision=`{decision}`，stage=`{stage}`。"
            f" 主 blocker 是 `{top_row.get('signal', 'none')}`；下一步不是上线，而是 {experiments[0] if experiments else '补充验证证据'}"
        ),
        "warning_rows": rows,
    }


def decision_counts(cards: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(card.get("overall_decision") or "monitor") for card in cards)
    return {key: counts.get(key, 0) for key in ["block_promotion", "review_before_promotion", "monitor"]}


def owner_counts(cards: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(card.get("owner_role") or "Platform owner") for card in cards)
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def build_payload(reports_dir: Path) -> dict[str, Any]:
    source_path = latest(reports_dir, "regression_warning_audit_latest.json")
    audit = read_json(source_path)
    warnings = audit.get("warnings") or []
    if not isinstance(warnings, list):
        warnings = []
    grouped = group_warning_rows([row for row in warnings if isinstance(row, dict)])
    cards = [
        build_card(domain, entity, rows)
        for (domain, entity), rows in grouped.items()
    ]
    cards.sort(
        key=lambda card: (
            DECISION_RANK.get(str(card.get("overall_decision")), 9),
            -int(card.get("warning_count") or 0),
            str(card.get("domain") or ""),
            str(card.get("entity") or ""),
        )
    )
    summary = audit.get("summary") or {}
    blocked_cards = [card for card in cards if card.get("overall_decision") == "block_promotion"]
    review_cards = [card for card in cards if card.get("overall_decision") == "review_before_promotion"]
    launch_gate = "blocked" if blocked_cards else ("review_required" if review_cards else "monitor")
    return {
        "schema_version": 1,
        "status": "ok",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "source_warning_audit": str(source_path) if source_path else None,
        "summary": {
            "launch_gate": launch_gate,
            "source_launch_gate": summary.get("launch_gate"),
            "total_cards": len(cards),
            "total_warning_rows": len(warnings),
            "decision_counts": decision_counts(cards),
            "owner_counts": owner_counts(cards),
            "domain_counts": dict(Counter(str(card.get("domain") or "unknown") for card in cards)),
            "top_blocked_cards": [card.get("card_id") for card in blocked_cards[:5]],
            "review_cards": len(review_cards),
        },
        "cards": cards,
        "interview_tracks": {
            "30s": "我把 warning audit 聚合成 promotion card：每个模型/场景都有 decision、owner、下一步实验和不能宣传的 claim。",
            "5min": "先看 blocked/review/monitor，再挑一个 high blocker 展示 primary reason、before shadow/A/B/ramp 条件，证明项目不是 demo 截图，而是 launch review 流程。",
            "15min": "展开 QINI/AUUC/PEHE/policy value 冲突、OPE/overlap/refutation 证据和 owner 分工，说明如何从 offline benchmark 走到 shadow、holdout、A/B。",
        },
    }


def write_csv(path: Path, cards: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "card_id",
        "domain",
        "entity",
        "warning_count",
        "overall_decision",
        "promotion_stage",
        "owner_role",
        "primary_blocker",
        "primary_blocker_value",
        "primary_reason",
        "safe_claim",
        "unsafe_claim",
        "before_shadow",
        "before_ab",
        "before_ramp",
        "next_experiments",
        "metrics_to_watch",
        "ui_routes",
        "evidence_refs",
        "interview_line",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for card in cards:
            row = {field: card.get(field) for field in fields}
            for field in ["next_experiments", "metrics_to_watch", "ui_routes", "evidence_refs"]:
                row[field] = "; ".join(str(item) for item in (card.get(field) or []))
            writer.writerow(row)


def markdown_table(cards: list[dict[str, Any]], limit: int = 30) -> str:
    rows = []
    for card in cards[:limit]:
        rows.append(
            "| {card_id} | {decision} | {stage} | {owner} | {entity} | {blocker} | {next_exp} |".format(
                card_id=card.get("card_id", ""),
                decision=card.get("overall_decision", ""),
                stage=card.get("promotion_stage", ""),
                owner=str(card.get("owner_role", "")).replace("|", "/"),
                entity=str(card.get("entity", "")).replace("|", "/"),
                blocker=str(card.get("primary_blocker", "")).replace("|", "/"),
                next_exp=str((card.get("next_experiments") or [""])[0]).replace("|", "/"),
            )
        )
    return "\n".join(rows) or "| none | monitor | candidate_for_shadow | Platform owner | none | none | none |"


def build_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary") or {}
    decisions = summary.get("decision_counts") or {}
    owners = summary.get("owner_counts") or {}
    cards = payload.get("cards") or []
    owner_table = "\n".join(f"| {owner} | {count} |" for owner, count in owners.items())
    top_cards = markdown_table(cards)
    detail_sections = []
    for card in cards[:8]:
        detail_sections.append(
            f"""### {card.get("card_id")} - {card.get("entity")}

- Decision: `{card.get("overall_decision")}`; stage: `{card.get("promotion_stage")}`; owner: `{card.get("owner_role")}`.
- Primary blocker: `{card.get("primary_blocker")}` = `{card.get("primary_blocker_value")}`.
- Safe claim: {card.get("safe_claim")}
- Unsafe claim: {card.get("unsafe_claim")}
- Before shadow: {card.get("before_shadow")}
- Before A/B: {card.get("before_ab")}
- Before ramp: {card.get("before_ramp")}
- Metrics to watch: {", ".join(card.get("metrics_to_watch") or [])}
- Interview line: {card.get("interview_line")}
"""
        )

    tracks = payload.get("interview_tracks") or {}
    return f"""# DeepUplift Promotion Launch Cards

Generated at: `{payload.get("generated_at")}`

This artifact converts `regression_warning_audit_latest.json` into launch-review cards. It answers: which result can be shown as research evidence, which result can enter shadow review, which result must not be promoted, who owns the fix, and what experiment is required next.

## Summary

| Metric | Value |
| --- | ---: |
| Launch gate | {summary.get("launch_gate", "NA")} |
| Cards | {summary.get("total_cards", 0)} |
| Warning rows | {summary.get("total_warning_rows", 0)} |
| Block promotion | {decisions.get("block_promotion", 0)} |
| Review before promotion | {decisions.get("review_before_promotion", 0)} |
| Monitor | {decisions.get("monitor", 0)} |

## Owner Load

| Owner role | Cards |
| --- | ---: |
{owner_table or "| Platform owner | 0 |"}

## Launch Review Board

| Card | Decision | Stage | Owner | Entity | Primary blocker | Next experiment |
| --- | --- | --- | --- | --- | --- | --- |
{top_cards}

## How To Explain In Interviews

- 30s: {tracks.get("30s", "")}
- 5min: {tracks.get("5min", "")}
- 15min: {tracks.get("15min", "")}

## Card Details

{chr(10).join(detail_sections) if detail_sections else "No cards generated."}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate launch-review promotion cards from warning audit rows.")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--json-output", default="reports/promotion_launch_cards_latest.json")
    parser.add_argument("--csv-output", default="reports/promotion_launch_cards_latest.csv")
    parser.add_argument("--doc-output", default="docs/DEEPUplift_PROMOTION_LAUNCH_CARDS.md")
    args = parser.parse_args()

    payload = build_payload(Path(args.reports_dir))
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
                "cards": len(payload.get("cards") or []),
                "launch_gate": (payload.get("summary") or {}).get("launch_gate"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
