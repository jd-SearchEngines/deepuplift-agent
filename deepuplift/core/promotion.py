from __future__ import annotations

from typing import Any

import pandas as pd


def _as_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _top_fraction_row(metrics: dict[str, Any], fraction: float) -> dict[str, Any]:
    for row in metrics.get("top_k", []) or []:
        if abs(float(row.get("top_fraction") or 0) - fraction) < 1e-6:
            return row
    return {}


def promotion_gate(metrics: dict[str, Any], readiness: dict[str, Any]) -> dict[str, Any]:
    checks = []

    def add_check(name: str, status: str, value: Any, guidance: str, critical: bool = False) -> None:
        checks.append(
            {
                "check": name,
                "status": status,
                "value": value,
                "critical": critical,
                "guidance": guidance,
            }
        )

    readiness_score = int(readiness.get("score") or 0)
    blockers = readiness.get("blockers") or []
    if readiness_score >= 80 and not blockers:
        add_check("readiness", "pass", readiness_score, "Readiness score supports a controlled pilot.")
    elif readiness_score >= 60:
        add_check("readiness", "warn", readiness_score, "Readiness supports shadow scoring before live treatment.")
    elif readiness_score >= 40:
        add_check("readiness", "warn", readiness_score, "Keep this run in research review.", critical=True)
    else:
        add_check("readiness", "blocker", readiness_score, "Readiness is too low for promotion.", critical=True)

    if blockers:
        add_check("readiness_blockers", "blocker", len(blockers), "; ".join(blockers[:3]), critical=True)

    qini = _as_float(metrics.get("qini_score"))
    if qini is None:
        add_check("qini", "missing", None, "QINI is required before promotion.", critical=True)
    elif qini > 0:
        add_check("qini", "pass", qini, "Uplift ranking beats random directionally.")
    else:
        add_check("qini", "blocker", qini, "Non-positive QINI blocks promotion.", critical=True)

    top10 = _top_fraction_row(metrics, 0.1)
    top10_uplift = _as_float(top10.get("observed_uplift"))
    if top10_uplift is None:
        add_check("top10_uplift", "missing", None, "Top 10% observed uplift is required for targeting review.", critical=True)
    elif top10_uplift > 0:
        add_check("top10_uplift", "pass", top10_uplift, "Top audience has positive observed uplift.")
    else:
        add_check("top10_uplift", "blocker", top10_uplift, "Top audience does not lift.", critical=True)

    policy_best = metrics.get("policy_best") or {}
    policy_net = _as_float(policy_best.get("observed_net_value", policy_best.get("predicted_net_value")))
    if policy_net is None:
        add_check("policy_value", "warn", None, "Policy value is missing; keep decision offline until cost/value is configured.")
    elif policy_net > 0:
        add_check("policy_value", "pass", policy_net, "Best policy threshold has positive net value.")
    else:
        add_check("policy_value", "blocker", policy_net, "No positive net-value threshold.", critical=True)

    calibration = _as_float(metrics.get("calibration_mae"))
    if calibration is None:
        add_check("calibration", "warn", None, "Calibration is missing; do not use score magnitude for budget pacing.")
    elif calibration <= 0.15:
        add_check("calibration", "pass", calibration, "Calibration is strong enough for threshold review.")
    elif calibration <= 0.30:
        add_check("calibration", "warn", calibration, "Calibration is usable but needs monitoring.")
    else:
        add_check("calibration", "warn", calibration, "Calibration is weak; use ranking only.")

    propensity = ((metrics.get("overlap_trim") or {}).get("propensity") or {})
    weak_overlap = _as_float(propensity.get("weak_overlap_rate"))
    if weak_overlap is None:
        add_check("overlap", "warn", None, "Overlap diagnostics are missing.")
    elif weak_overlap <= 0.02:
        add_check("overlap", "pass", weak_overlap, "Common support looks healthy.")
    elif weak_overlap <= 0.10:
        add_check("overlap", "warn", weak_overlap, "Restrict deployment to common support.")
    else:
        add_check("overlap", "blocker", weak_overlap, "Weak overlap is too high for promotion.", critical=True)

    sensitivity_verdict = str((metrics.get("sensitivity") or {}).get("verdict") or "").lower()
    if sensitivity_verdict == "fail":
        add_check("sensitivity", "blocker", sensitivity_verdict, "Refutation checks failed.", critical=True)
    elif sensitivity_verdict in {"pass", "caution"}:
        add_check("sensitivity", "pass" if sensitivity_verdict == "pass" else "warn", sensitivity_verdict, "Sensitivity check ran.")
    else:
        add_check("sensitivity", "warn", "not_run", "Run sensitivity checks before final promotion.")

    qini_ci = (((metrics.get("bootstrap") or {}).get("metrics") or {}).get("qini_score") or {})
    qini_ci_low = _as_float(qini_ci.get("low"))
    uncertainty_ready = False
    if qini_ci_low is None:
        add_check("bootstrap_ci", "warn", "not_run", "Bootstrap CI is required before live pilot.")
    elif qini_ci_low > 0:
        uncertainty_ready = True
        add_check("bootstrap_ci", "pass", qini_ci_low, "QINI lower bound is positive.")
    else:
        add_check("bootstrap_ci", "warn", qini_ci_low, "QINI lower bound is not positive.")

    critical_blockers = [row for row in checks if row["critical"] and row["status"] == "blocker"]
    if critical_blockers:
        level = "not_ready"
        recommendation = "Do not promote. Fix the blocking data, metric, overlap, or policy-value issue first."
    elif readiness_score >= 80 and uncertainty_ready:
        level = "pilot_candidate"
        recommendation = "Eligible for a small controlled online pilot with holdout and guardrail monitoring."
    elif readiness_score >= 60:
        level = "shadow_candidate"
        recommendation = "Run shadow scoring or offline audience review; add bootstrap CI before live treatment."
    else:
        level = "research_only"
        recommendation = "Keep as research evidence and compare against stronger baselines."

    next_required_evidence = []
    if not uncertainty_ready:
        next_required_evidence.append("bootstrap_ci")
    if not sensitivity_verdict:
        next_required_evidence.append("sensitivity_check")
    if policy_net is None:
        next_required_evidence.append("cost_value_policy_config")
    if level in {"shadow_candidate", "pilot_candidate"}:
        next_required_evidence.append("online_holdout_plan")

    return {
        "schema_version": 1,
        "level": level,
        "recommendation": recommendation,
        "readiness_score": readiness_score,
        "checks": checks,
        "critical_blockers": [row["check"] for row in critical_blockers],
        "next_required_evidence": list(dict.fromkeys(next_required_evidence)),
    }
