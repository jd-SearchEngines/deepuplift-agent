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


def _top_fraction_row(metrics: dict, fraction: float) -> dict:
    for row in metrics.get("top_k", []) or []:
        if abs((row.get("top_fraction") or 0) - fraction) < 1e-6:
            return row
    return {}


def decision_readiness(metrics: dict) -> dict:
    checks = []
    blockers = []

    def add_check(name: str, value, status: str, points: int, max_points: int, guidance: str) -> None:
        checks.append(
            {
                "check": name,
                "value": value,
                "status": status,
                "points": points,
                "max_points": max_points,
                "guidance": guidance,
            }
        )
        if status == "blocker":
            blockers.append(f"{name}: {guidance}")

    qini = _as_float(metrics.get("qini_score"))
    if qini is None:
        add_check("QINI", None, "missing", 0, 12, "QINI not computed.")
    elif qini > 0:
        add_check("QINI", f"{qini:.4f}", "pass", 12, 12, "Ranking has positive QINI.")
    else:
        add_check("QINI", f"{qini:.4f}", "blocker", 0, 12, "QINI is non-positive; treat as exploratory.")

    auuc = _as_float(metrics.get("auuc_score"))
    if auuc is None:
        add_check("AUUC", None, "missing", 0, 8, "AUUC not computed.")
    elif auuc > 0:
        add_check("AUUC", f"{auuc:.4f}", "pass", 8, 8, "Ranking has positive AUUC.")
    else:
        add_check("AUUC", f"{auuc:.4f}", "warn", 0, 8, "AUUC is non-positive.")

    top10 = _top_fraction_row(metrics, 0.1)
    top10_uplift = _as_float(top10.get("observed_uplift"))
    if top10_uplift is None:
        add_check("Top 10% uplift", None, "missing", 0, 15, "Top-K observed uplift not available.")
    elif top10_uplift > 0:
        add_check("Top 10% uplift", f"{top10_uplift:.4f}", "pass", 15, 15, "Top audience shows positive observed uplift.")
    else:
        add_check("Top 10% uplift", f"{top10_uplift:.4f}", "blocker", 0, 15, "Top audience is not lifting; do not deploy this threshold.")

    top10_rows = _as_float(top10.get("rows"))
    top10_treated = _as_float(top10.get("treated_rows"))
    top10_control = _as_float(top10.get("control_rows"))
    top10_min_cell = min(
        [value for value in [top10_treated, top10_control] if value is not None],
        default=None,
    )
    if top10_rows is None:
        add_check("Top 10% support", None, "missing", 0, 8, "Top-K support counts are not available.")
    elif top10_rows < 20 or (top10_min_cell is not None and top10_min_cell < 5):
        add_check(
            "Top 10% support",
            f"rows={top10_rows:.0f}, min_cell={top10_min_cell if top10_min_cell is not None else 'NA'}",
            "blocker",
            0,
            8,
            "Top-K cell counts are too small for deployment evidence; increase validation size or use bootstrap CI.",
        )
    elif top10_rows < 100 or (top10_min_cell is not None and top10_min_cell < 25):
        add_check(
            "Top 10% support",
            f"rows={top10_rows:.0f}, min_cell={top10_min_cell:.0f}" if top10_min_cell is not None else f"rows={top10_rows:.0f}",
            "warn",
            4,
            8,
            "Top-K evidence is directionally useful but sample support is still thin.",
        )
    else:
        add_check(
            "Top 10% support",
            f"rows={top10_rows:.0f}, min_cell={top10_min_cell:.0f}" if top10_min_cell is not None else f"rows={top10_rows:.0f}",
            "pass",
            8,
            8,
            "Top-K bucket has enough treated/control support for a first decision read.",
        )

    policy_best = metrics.get("policy_best") or {}
    policy_net = _as_float(policy_best.get("observed_net_value", policy_best.get("predicted_net_value")))
    if policy_net is None:
        add_check("Policy net value", None, "missing", 0, 15, "Policy value not computed.")
    elif policy_net > 0:
        add_check("Policy net value", f"{policy_net:.4f}", "pass", 15, 15, "Best targeting threshold has positive net value.")
    else:
        add_check("Policy net value", f"{policy_net:.4f}", "blocker", 0, 15, "No positive net-value threshold under current cost/value settings.")

    calibration_mae = _as_float(metrics.get("calibration_mae"))
    if calibration_mae is None:
        add_check("Calibration", None, "missing", 0, 12, "Calibration not computed.")
    elif calibration_mae <= 0.15:
        add_check("Calibration", f"{calibration_mae:.4f}", "pass", 12, 12, "Uplift buckets are reasonably calibrated.")
    elif calibration_mae <= 0.30:
        add_check("Calibration", f"{calibration_mae:.4f}", "warn", 7, 12, "Calibration is usable but should be monitored.")
    else:
        add_check("Calibration", f"{calibration_mae:.4f}", "warn", 2, 12, "Calibration error is high; use ranking cautiously.")

    propensity = ((metrics.get("overlap_trim") or {}).get("propensity") or {})
    weak_overlap = _as_float(propensity.get("weak_overlap_rate"))
    if weak_overlap is None:
        add_check("Overlap", None, "missing", 0, 15, "Overlap diagnostics not available.")
    elif weak_overlap <= 0.02:
        add_check("Overlap", f"{weak_overlap:.1%}", "pass", 15, 15, "Common support looks healthy.")
    elif weak_overlap <= 0.10:
        add_check("Overlap", f"{weak_overlap:.1%}", "warn", 8, 15, "Some weak-overlap users exist; cap deployment to common support.")
    else:
        add_check("Overlap", f"{weak_overlap:.1%}", "blocker", 0, 15, "Weak overlap is high; deployment needs trimming or more data.")

    sensitivity = metrics.get("sensitivity") or {}
    verdict = str(sensitivity.get("verdict", "")).lower()
    if verdict == "pass":
        add_check("Sensitivity", "pass", "pass", 15, 15, "Ranking beats null/refutation checks.")
    elif verdict == "caution":
        add_check("Sensitivity", "caution", "warn", 8, 15, "Ranking beats only part of the null checks.")
    elif verdict == "fail":
        add_check("Sensitivity", "fail", "blocker", 0, 15, "Ranking does not beat refutation checks.")
    else:
        add_check("Sensitivity", "not run", "missing", 4, 15, "Run sensitivity samples before deployment.")

    qini_ci = (((metrics.get("bootstrap") or {}).get("metrics") or {}).get("qini_score") or {})
    qini_ci_low = _as_float(qini_ci.get("low"))
    if qini_ci_low is None:
        add_check("Bootstrap CI", "not run", "missing", 4, 8, "Run bootstrap before model selection is final.")
    elif qini_ci_low > 0:
        add_check("Bootstrap CI", f"QINI low {qini_ci_low:.4f}", "pass", 8, 8, "QINI lower confidence bound is positive.")
    else:
        add_check("Bootstrap CI", f"QINI low {qini_ci_low:.4f}", "warn", 2, 8, "QINI lower bound is not positive.")

    max_points = sum(row["max_points"] for row in checks) or 1
    raw_score = round(100 * sum(row["points"] for row in checks) / max_points)
    score = max(0, min(100, raw_score))
    if blockers:
        score = min(score, 59)
    if score >= 80:
        level = "Ready for controlled rollout"
        recommendation = "可以进入小流量或灰度投放，但仍建议保留 holdout 和监控。"
    elif score >= 60:
        level = "Pilot / shadow deploy"
        recommendation = "适合离线通过后做 shadow scoring 或小流量试验。"
    elif score >= 40:
        level = "Research only"
        recommendation = "适合继续实验对比，不建议直接上线。"
    else:
        level = "Not ready"
        recommendation = "当前证据不足或存在关键风险，先修数据/模型/评估。"

    return {
        "score": score,
        "level": level,
        "recommendation": recommendation,
        "checks": checks,
        "blockers": blockers,
    }
