from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any


def latest(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def owner_role_for(domain: str) -> str:
    if domain == "industrial_scenario":
        return "ML owner + business owner"
    if domain in {"deep_model_battle", "paper_benchmark", "regression_smoke"}:
        return "ML / benchmark owner"
    if domain == "ope":
        return "Experimentation owner"
    if domain == "evidence_governance":
        return "Platform / evidence owner"
    return "Platform owner"


def ui_route_for(domain: str) -> str:
    routes = {
        "regression_smoke": "Evidence -> Regression Gate Artifacts",
        "industrial_scenario": "场景工作台 -> Scenario Compare; Evidence -> Regression Warning Audit",
        "deep_model_battle": "Model Deconstruction -> Deep Model Battle Report",
        "paper_benchmark": "Model Deconstruction -> Paper-Level Benchmark Evidence",
        "ope": "Policy -> OPE mini-lab; Evidence -> Regression Warning Audit",
        "evidence_governance": "Evidence -> Experiment Evidence Store",
    }
    return routes.get(domain, "Evidence -> Regression Warning Audit")


def next_experiment_for(domain: str, signal: str) -> str:
    signal = signal.lower()
    if "readiness" in signal:
        return "Run shadow scoring, inspect readiness blockers, then require holdout/A/B before promotion."
    if any(term in signal for term in ["policy_value", "roi", "observed_uplift"]):
        return "Re-run threshold/cost ablation and validate the selected audience with holdout or A/B."
    if any(term in signal for term in ["qini", "auuc", "sensitivity"]):
        return "Rerun multi-seed benchmark with bootstrap/refutation and compare run diff."
    if any(term in signal for term in ["overlap", "coverage", "effective_sample", "ess"]):
        return "Add support diagnostics, trim unsupported users, or collect randomized exploration logs."
    if domain == "deep_model_battle":
        return "Run tuned ablation with loss curves, representation balance, and baseline deltas."
    if domain == "paper_benchmark":
        return "Review metric conflict by deployment goal and rerun larger known-CATE benchmark if needed."
    if domain == "evidence_governance":
        return "Refresh evidence store, run diff, manifest index, and interview evidence pack."
    return "Assign owner, document root cause, and add a targeted validation run before promotion."


def promotion_decision_for(severity: str) -> str:
    if severity == "high":
        return "block_promotion"
    if severity == "medium":
        return "review_before_promotion"
    return "monitor"


def add_warning(
    rows: list[dict[str, Any]],
    *,
    severity: str,
    domain: str,
    artifact: str,
    entity: str,
    signal: str,
    value: Any,
    why_it_matters: str,
    recommended_action: str,
    interview_line: str,
) -> None:
    promotion_decision = promotion_decision_for(severity)
    rows.append(
        {
            "severity": severity,
            "domain": domain,
            "artifact": artifact,
            "entity": entity,
            "signal": signal,
            "value": value,
            "why_it_matters": why_it_matters,
            "recommended_action": recommended_action,
            "interview_line": interview_line,
            "promotion_decision": promotion_decision,
            "owner_role": owner_role_for(domain),
            "next_experiment": next_experiment_for(domain, signal),
            "ui_route": ui_route_for(domain),
            "interview_priority": "P0" if promotion_decision == "block_promotion" else ("P1" if promotion_decision == "review_before_promotion" else "P2"),
        }
    )


def audit_agent_regression(rows: list[dict[str, Any]], payload: dict[str, Any], artifact: str) -> None:
    for row in payload.get("results") or []:
        if not isinstance(row, dict):
            continue
        entity = f"{row.get('dataset_id', 'dataset')} / {row.get('model', 'model')}"
        qini = as_float(row.get("qini"))
        auuc = as_float(row.get("auuc"))
        if row.get("status") != "ok":
            add_warning(
                rows,
                severity="high",
                domain="regression_smoke",
                artifact=artifact,
                entity=entity,
                signal="run_status",
                value=row.get("status"),
                why_it_matters="Smoke regression should prove the default training path can finish.",
                recommended_action="Inspect run error and keep the model out of demo ranking until fixed.",
                interview_line="我把 smoke 失败当成工程阻塞，不会把失败模型包装成可上线能力。",
            )
        if qini is not None and qini <= 0:
            add_warning(
                rows,
                severity="medium",
                domain="regression_smoke",
                artifact=artifact,
                entity=entity,
                signal="qini_non_positive",
                value=qini,
                why_it_matters="Non-positive QINI means the ranking is not beating random targeting in this small smoke split.",
                recommended_action="Treat this as path validation only; rerun with more rows/seeds before claiming model quality.",
                interview_line="我会明确说 smoke 只证明链路，不证明策略有效，质量 claim 必须看 paper benchmark 或线上 holdout。",
            )
        if auuc is not None and auuc <= 0:
            add_warning(
                rows,
                severity="medium",
                domain="regression_smoke",
                artifact=artifact,
                entity=entity,
                signal="auuc_non_positive",
                value=auuc,
                why_it_matters="AUUC below zero is a ranking-quality warning for the current validation split.",
                recommended_action="Check calibration, Top-K lift and sensitivity before using this run in an interview demo.",
                interview_line="AUUC 和 QINI 都会被我放进 readiness gate，而不是只展示成功截图。",
            )
        sensitivity = row.get("sensitivity") or {}
        verdict = str(sensitivity.get("verdict") or "").lower()
        if verdict in {"fail", "caution"}:
            add_warning(
                rows,
                severity="high" if verdict == "fail" else "medium",
                domain="regression_smoke",
                artifact=artifact,
                entity=entity,
                signal="sensitivity_verdict",
                value=verdict,
                why_it_matters="Sensitivity/refutation checks catch rankings that may be split noise or leakage-sensitive.",
                recommended_action="Increase sensitivity samples and inspect null tests before promotion.",
                interview_line="我会主动讲 refutation：如果过不了随机分数或置换检验，就只能算研究证据。",
            )
        propensity = ((row.get("overlap_trim") or {}).get("propensity") or {})
        weak_overlap = as_float(propensity.get("weak_overlap_rate"))
        if weak_overlap is not None and weak_overlap > 0.10:
            add_warning(
                rows,
                severity="high",
                domain="regression_smoke",
                artifact=artifact,
                entity=entity,
                signal="weak_overlap_rate",
                value=weak_overlap,
                why_it_matters="Weak overlap means the model is extrapolating treatment effects outside common support.",
                recommended_action="Trim unsupported users or collect more randomized exploration before deployment.",
                interview_line="我不会让 uplift 模型跨 support 外推，overlap 是上线前硬门槛。",
            )


def audit_scenario_compare(rows: list[dict[str, Any]], payload: dict[str, Any], artifact: str) -> None:
    for row in payload.get("best_by_dataset") or []:
        if not isinstance(row, dict):
            continue
        entity = f"{row.get('dataset_id', 'dataset')} / {row.get('model', 'model')}"
        readiness = str(row.get("readiness_level") or "")
        if readiness in {"Not ready", "Research only"}:
            add_warning(
                rows,
                severity="high" if readiness == "Not ready" else "medium",
                domain="industrial_scenario",
                artifact=artifact,
                entity=entity,
                signal="readiness_level",
                value=readiness,
                why_it_matters="Scenario winner still may not be launch-ready once calibration, overlap and policy value are considered.",
                recommended_action="Use this row for model comparison only; require shadow scoring or holdout before rollout.",
                interview_line="我把“赢了 leaderboard”和“能上线”分开，readiness 是单独的上线证据。",
            )
        for signal, label in [
            ("top10_observed_uplift", "Top-K observed uplift"),
            ("industrial_top10_policy_value_sum", "Top-K policy value"),
            ("industrial_top10_roi_proxy", "ROI proxy"),
        ]:
            value = as_float(row.get(signal))
            if value is not None and value <= 0:
                add_warning(
                    rows,
                    severity="high",
                    domain="industrial_scenario",
                    artifact=artifact,
                    entity=entity,
                    signal=signal,
                    value=value,
                    why_it_matters=f"{label} is non-positive, so the chosen audience may destroy value despite model ranking.",
                    recommended_action="Re-optimize threshold/cost constraints or pick a cost-aware policy before launch.",
                    interview_line="业务 ROI 是最终门槛：QINI 好但 policy value 坏时，我会选择不投放。",
                )
        weak_overlap = as_float(row.get("weak_overlap_rate"))
        if weak_overlap is not None and weak_overlap > 0.05:
            add_warning(
                rows,
                severity="high" if weak_overlap > 0.10 else "medium",
                domain="industrial_scenario",
                artifact=artifact,
                entity=entity,
                signal="weak_overlap_rate",
                value=weak_overlap,
                why_it_matters="Scenario compare is using users with weak common support.",
                recommended_action="Add overlap trimming and compare post-trim QINI/policy value before pilot.",
                interview_line="工业场景里我优先处理 selection bias 和 weak overlap，再讨论模型复杂度。",
            )
        trim_qini = as_float(row.get("trim05_qini"))
        qini = as_float(row.get("qini"))
        if qini is not None and trim_qini is not None and qini > 0 and trim_qini <= 0:
            add_warning(
                rows,
                severity="medium",
                domain="industrial_scenario",
                artifact=artifact,
                entity=entity,
                signal="trim05_qini_sign_flip",
                value=trim_qini,
                why_it_matters="Ranking quality disappears after overlap trimming, suggesting weak support drives the signal.",
                recommended_action="Report pre/post-trim metrics together and avoid policy claims until stable.",
                interview_line="如果 trim 后指标翻转，我会把它作为 failure attribution，而不是隐藏掉。",
            )


def audit_deep_benchmark(rows: list[dict[str, Any]], payload: dict[str, Any], artifact: str) -> None:
    for row in payload.get("model_scorecards") or []:
        if not isinstance(row, dict):
            continue
        model = str(row.get("model") or "model")
        verdict = str(row.get("verdict") or "")
        if verdict in {"needs_more_tuning", "guarded_or_task_limited"}:
            add_warning(
                rows,
                severity="medium",
                domain="deep_model_battle",
                artifact=artifact,
                entity=model,
                signal="model_verdict",
                value=verdict,
                why_it_matters="A deep architecture can be valuable to explain, but it should not be overclaimed as a paper-level winner.",
                recommended_action=str(row.get("watchout") or "Run larger tuned benchmark before promotion."),
                interview_line=str(row.get("interview_line") or "把深度模型讲成架构深度和 failure attribution，而不是无条件优于基线。"),
            )
        delta_pehe = as_float(row.get("avg_delta_pehe_vs_baseline"))
        if delta_pehe is not None and delta_pehe > 0:
            add_warning(
                rows,
                severity="medium",
                domain="deep_model_battle",
                artifact=artifact,
                entity=model,
                signal="avg_delta_pehe_vs_baseline",
                value=delta_pehe,
                why_it_matters="Positive PEHE delta means the deep variant is less accurate than the T/DR baseline on known-CATE data.",
                recommended_action="Keep T/DR as mandatory baseline and inspect loss/representation diagnostics.",
                interview_line="深度模型输了 PEHE 时，我会解释样本规模、loss 权重和树模型强基线，而不是硬吹 SOTA。",
            )
        delta_qini = as_float(row.get("avg_delta_qini_vs_baseline"))
        if delta_qini is not None and delta_qini < 0:
            add_warning(
                rows,
                severity="medium",
                domain="deep_model_battle",
                artifact=artifact,
                entity=model,
                signal="avg_delta_qini_vs_baseline",
                value=delta_qini,
                why_it_matters="Negative QINI delta means the tuned deep variant is not improving ranking over the baseline.",
                recommended_action="Run ablations for learning rate, representation balance, tarreg and interaction width.",
                interview_line="我用 battle card 证明自己知道为什么输，以及下一步怎么调。",
            )
    failure = payload.get("failure_attribution") or {}
    for reason, count in (failure.get("reason_counts") or {}).items():
        add_warning(
            rows,
            severity="low",
            domain="deep_model_battle",
            artifact=artifact,
            entity=str(reason),
            signal="guarded_skip_reason",
            value=count,
            why_it_matters="Guarded skips define task boundaries and avoid pretending every model supports every dataset.",
            recommended_action="Keep the skip explicit in docs/UI and add adapter only when task support is real.",
            interview_line="task-aware skip 是工程成熟度，不是失败遮掩。",
        )


def audit_paper_interpretation(rows: list[dict[str, Any]], payload: dict[str, Any], artifact: str) -> None:
    for row in payload.get("metric_conflicts") or []:
        if not isinstance(row, dict):
            continue
        add_warning(
            rows,
            severity="medium",
            domain="paper_benchmark",
            artifact=artifact,
            entity=str(row.get("dataset_id") or "dataset"),
            signal="metric_conflict",
            value=row.get("conflict"),
            why_it_matters="Effect accuracy, ranking and policy value can disagree; a single metric can mislead model selection.",
            recommended_action="Present PEHE/ATE/QINI/AUUC/policy value together and choose by deployment goal.",
            interview_line=str(row.get("interview_reading") or "我会把 metric conflict 当成 causal decision platform 的核心价值。"),
        )
    for row in payload.get("model_scorecards") or []:
        if not isinstance(row, dict):
            continue
        verdict = str(row.get("verdict") or "")
        if "needs" in verdict:
            add_warning(
                rows,
                severity="medium",
                domain="paper_benchmark",
                artifact=artifact,
                entity=str(row.get("model") or "model"),
                signal="paper_model_verdict",
                value=verdict,
                why_it_matters="Paper-level scorecard says this model needs more data/training review before strong claims.",
                recommended_action=str(row.get("watchout") or "Attach run manifests and rerun with more seeds before promotion."),
                interview_line=str(row.get("interview_line") or "模型没有赢时也要能讲清楚假设、证据和边界。"),
            )


def audit_ope(rows: list[dict[str, Any]], payload: dict[str, Any], artifact: str) -> None:
    diagnostics = ((payload.get("ope") or {}).get("diagnostics") or {})
    coverage = as_float(diagnostics.get("coverage_rate"))
    ess_fraction = as_float(diagnostics.get("effective_sample_fraction"))
    clipped = as_float(diagnostics.get("clipped_rate"))
    if coverage is not None and coverage < 0.10:
        add_warning(
            rows,
            severity="high",
            domain="ope",
            artifact=artifact,
            entity="general_ope_engine",
            signal="coverage_rate",
            value=coverage,
            why_it_matters="Low OPE coverage means the proposed policy is mostly unsupported by logged actions.",
            recommended_action="Collect randomized exploration or shrink the candidate policy before trusting OPE.",
            interview_line="OPE 只能评估历史日志支持过的策略，coverage 低时必须转 shadow/探索。",
        )
    if ess_fraction is not None and ess_fraction < 0.10:
        add_warning(
            rows,
            severity="high",
            domain="ope",
            artifact=artifact,
            entity="general_ope_engine",
            signal="effective_sample_fraction",
            value=ess_fraction,
            why_it_matters="Low ESS means IPS/SNIPS/DR estimates have high variance.",
            recommended_action="Use clipping sensitivity and larger logged samples before launch decisions.",
            interview_line="我会用 ESS 解释 OPE 可信度，而不是只报一个 DR value。",
        )
    if clipped is not None and clipped > 0.10:
        add_warning(
            rows,
            severity="medium",
            domain="ope",
            artifact=artifact,
            entity="general_ope_engine",
            signal="clipped_rate",
            value=clipped,
            why_it_matters="Heavy clipping indicates unstable propensities and biased OPE sensitivity.",
            recommended_action="Report clipping sensitivity and diagnose propensity model support.",
            interview_line="clipping 是稳定性工具，不是消除 bias 的魔法。",
        )


def audit_evidence_store(rows: list[dict[str, Any]], payload: dict[str, Any], artifact: str) -> None:
    records = payload.get("records") or []
    if not records:
        add_warning(
            rows,
            severity="high",
            domain="evidence_governance",
            artifact=artifact,
            entity="evidence_store",
            signal="records",
            value=0,
            why_it_matters="No evidence records means claims cannot be traced to run artifacts.",
            recommended_action="Run generate_evidence_store before interviews or reviews.",
            interview_line="没有 evidence manifest，我不会把它当作可信 claim。",
        )
        return
    not_ready = [
        row
        for row in records
        if isinstance(row, dict) and str(row.get("readiness_level") or "") in {"Not ready", "Research only"}
    ]
    if not_ready:
        add_warning(
            rows,
            severity="low",
            domain="evidence_governance",
            artifact=artifact,
            entity="evidence_store",
            signal="research_or_not_ready_records",
            value=len(not_ready),
            why_it_matters="The evidence store preserves non-promotable runs, which is useful for audit but should not be sold as production-ready.",
            recommended_action="Filter by readiness/promotion level when creating launch claims.",
            interview_line="我保留失败和研究态 run，是为了能解释边界，而不是只展示好看的结果。",
        )


def sanitize(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    return value


def severity_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"high": 0, "medium": 0, "low": 0}
    for row in rows:
        severity = str(row.get("severity") or "low")
        counts[severity] = counts.get(severity, 0) + 1
    return counts


def domain_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        domain = str(row.get("domain") or "unknown")
        counts[domain] = counts.get(domain, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def build_markdown(payload: dict[str, Any]) -> str:
    summary = payload.get("summary") or {}
    rows = payload.get("warnings") or []
    counts = summary.get("severity_counts") or {}
    domains = summary.get("domain_counts") or {}
    top_rows = rows[:12]

    top_table = "\n".join(
        "| {severity} | {domain} | {decision} | {owner} | {entity} | {signal} | {value} | {next_experiment} |".format(
            severity=row.get("severity", ""),
            domain=row.get("domain", ""),
            decision=row.get("promotion_decision", ""),
            owner=row.get("owner_role", ""),
            entity=str(row.get("entity", "")).replace("|", "/"),
            signal=row.get("signal", ""),
            value=str(row.get("value", "")).replace("|", "/"),
            next_experiment=str(row.get("next_experiment", row.get("recommended_action", ""))).replace("|", "/"),
        )
        for row in top_rows
    )
    domain_table = "\n".join(f"| {domain} | {count} |" for domain, count in domains.items())
    rows_table = "\n".join(
        "| {severity} | {priority} | {domain} | {decision} | {owner} | {artifact} | {entity} | {signal} | {why} | {route} | {line} |".format(
            severity=row.get("severity", ""),
            priority=row.get("interview_priority", ""),
            domain=row.get("domain", ""),
            decision=row.get("promotion_decision", ""),
            owner=row.get("owner_role", ""),
            artifact=row.get("artifact", ""),
            entity=str(row.get("entity", "")).replace("|", "/"),
            signal=row.get("signal", ""),
            why=str(row.get("why_it_matters", "")).replace("|", "/"),
            route=str(row.get("ui_route", "")).replace("|", "/"),
            line=str(row.get("interview_line", "")).replace("|", "/"),
        )
        for row in rows
    )

    return f"""# DeepUplift Regression Warning Audit

Generated at: `{payload.get("generated_at")}`

## Summary

| Metric | Value |
| --- | ---: |
| Total warning rows | {summary.get("total_warnings", 0)} |
| High | {counts.get("high", 0)} |
| Medium | {counts.get("medium", 0)} |
| Low | {counts.get("low", 0)} |

## Domain Counts

| Domain | Warnings |
| --- | ---: |
{domain_table or "| none | 0 |"}

## Top Launch Guardrails

| Severity | Domain | Promotion decision | Owner role | Entity | Signal | Value | Next experiment |
| --- | --- | --- | --- | --- | --- | --- | --- |
{top_table or "| low | all | monitor | Platform owner | none | clean |  | No launch guardrail warnings found. |"}

## How To Explain In Interviews

- 这个 audit 的定位不是“证明所有模型都好”，而是把不能上线、只能研究、需要 shadow 或需要 holdout 的证据显式化，并转成 owner、next experiment 和 promotion decision。
- 如果 QINI/AUUC、PEHE、policy value 冲突，我会按上线目标选择指标，并把冲突作为 causal decision platform 的价值点讲。
- 如果 deep model 输给 T/DR baseline，我会展示 battle card、loss/ablation、task boundary 和下一步调参，而不是过度包装。
- 如果 OPE coverage/ESS 不够，我会要求更多 randomized exploration 或线上小流量实验。

## Warning Rows

| Severity | Priority | Domain | Promotion decision | Owner role | Artifact | Entity | Signal | Why it matters | UI route | Interview line |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
{rows_table or "| low | P2 | all | monitor | Platform owner | none | none | clean | No risks found. | Evidence | Evidence gate is clean. |"}
"""


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "severity",
        "domain",
        "artifact",
        "entity",
        "signal",
        "value",
        "why_it_matters",
        "recommended_action",
        "promotion_decision",
        "owner_role",
        "next_experiment",
        "ui_route",
        "interview_priority",
        "interview_line",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def build_audit(reports_dir: Path) -> dict[str, Any]:
    artifact_paths = {
        "agent_regression": latest(reports_dir, "agent_regression_*.json"),
        "industrial_scenario_compare": latest(reports_dir, "industrial_scenario_compare_smoke_latest.json"),
        "deep_model_tuned_benchmark": latest(reports_dir, "deep_model_tuned_benchmark_latest.json"),
        "paper_benchmark_interpretation": latest(reports_dir, "paper_benchmark_interpretation_latest.json"),
        "ope_engine": latest(reports_dir, "ope_engine_smoke_latest.json"),
        "evidence_store": latest(reports_dir, "evidence_store_latest.json"),
    }
    warnings: list[dict[str, Any]] = []
    audit_agent_regression(warnings, read_json(artifact_paths["agent_regression"]), str(artifact_paths["agent_regression"]))
    audit_scenario_compare(
        warnings,
        read_json(artifact_paths["industrial_scenario_compare"]),
        str(artifact_paths["industrial_scenario_compare"]),
    )
    audit_deep_benchmark(
        warnings,
        read_json(artifact_paths["deep_model_tuned_benchmark"]),
        str(artifact_paths["deep_model_tuned_benchmark"]),
    )
    audit_paper_interpretation(
        warnings,
        read_json(artifact_paths["paper_benchmark_interpretation"]),
        str(artifact_paths["paper_benchmark_interpretation"]),
    )
    audit_ope(warnings, read_json(artifact_paths["ope_engine"]), str(artifact_paths["ope_engine"]))
    audit_evidence_store(warnings, read_json(artifact_paths["evidence_store"]), str(artifact_paths["evidence_store"]))
    warnings.sort(key=lambda row: {"high": 0, "medium": 1, "low": 2}.get(str(row.get("severity")), 3))
    return sanitize(
        {
            "schema_version": 1,
            "status": "ok",
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
            "inputs": {name: str(path) if path else None for name, path in artifact_paths.items()},
            "summary": {
                "total_warnings": len(warnings),
                "severity_counts": severity_counts(warnings),
                "domain_counts": domain_counts(warnings),
                "top_high": sum(1 for row in warnings if row.get("severity") == "high"),
                "launch_gate": "review_required" if any(row.get("severity") == "high" for row in warnings) else "monitor",
            },
            "warnings": warnings,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DeepUplift regression warning and launch guardrail audit.")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--json-output", default="reports/regression_warning_audit_latest.json")
    parser.add_argument("--csv-output", default="reports/regression_warning_audit_latest.csv")
    parser.add_argument("--markdown-output", default="docs/DEEPUplift_REGRESSION_WARNING_AUDIT.md")
    args = parser.parse_args()

    payload = build_audit(Path(args.reports_dir))
    json_path = Path(args.json_output)
    csv_path = Path(args.csv_output)
    markdown_path = Path(args.markdown_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    write_csv(csv_path, payload.get("warnings") or [])
    markdown_path.write_text(build_markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload.get("status"),
                "json": str(json_path),
                "csv": str(csv_path),
                "markdown": str(markdown_path),
                "warnings": (payload.get("summary") or {}).get("total_warnings"),
                "severity_counts": (payload.get("summary") or {}).get("severity_counts"),
                "launch_gate": (payload.get("summary") or {}).get("launch_gate"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
