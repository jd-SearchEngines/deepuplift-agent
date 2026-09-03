from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd


def _clean_float(value: Any):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(number) or np.isinf(number):
        return None
    return number


def default_policy_fractions(max_fraction: float = 1.0, step: float = 0.05) -> List[float]:
    max_fraction = min(max(float(max_fraction), step), 1.0)
    fractions = np.arange(step, max_fraction + step / 2, step)
    return [round(float(item), 4) for item in fractions if item <= max_fraction + 1e-9]


def policy_value_curve(
    df: pd.DataFrame,
    uplift_col: str = "uplift_score",
    outcome_col: str | None = None,
    treatment_col: str | None = None,
    conversion_value: float = 1.0,
    contact_cost: float = 0.0,
    fractions: Iterable[float] | None = None,
) -> Dict[str, Any]:
    if uplift_col not in df.columns:
        raise ValueError(f"{uplift_col} is required for policy value.")

    fractions = list(fractions or default_policy_fractions())
    sorted_df = df.sort_values(uplift_col, ascending=False).reset_index(drop=True)
    rows = []

    has_observed = (
        outcome_col is not None
        and treatment_col is not None
        and outcome_col in sorted_df.columns
        and treatment_col in sorted_df.columns
    )

    for fraction in fractions:
        n = max(1, int(len(sorted_df) * fraction))
        top = sorted_df.head(n)
        predicted_incremental = float(top[uplift_col].mean()) * n
        predicted_gross = predicted_incremental * conversion_value
        cost = n * contact_cost
        row = {
            "top_fraction": float(fraction),
            "rows": int(n),
            "mean_predicted_uplift": _clean_float(top[uplift_col].mean()),
            "predicted_incremental_outcomes": _clean_float(predicted_incremental),
            "predicted_gross_value": _clean_float(predicted_gross),
            "contact_cost": _clean_float(cost),
            "predicted_net_value": _clean_float(predicted_gross - cost),
        }

        if has_observed:
            treated = top[top[treatment_col] == 1]
            control = top[top[treatment_col] == 0]
            treated_rate = treated[outcome_col].mean() if len(treated) else np.nan
            control_rate = control[outcome_col].mean() if len(control) else np.nan
            observed_uplift = treated_rate - control_rate
            observed_incremental = observed_uplift * n
            observed_gross = observed_incremental * conversion_value
            row.update(
                {
                    "treated_rows": int(len(treated)),
                    "control_rows": int(len(control)),
                    "observed_uplift": _clean_float(observed_uplift),
                    "observed_incremental_outcomes": _clean_float(observed_incremental),
                    "observed_gross_value": _clean_float(observed_gross),
                    "observed_net_value": _clean_float(observed_gross - cost),
                }
            )
        rows.append(row)

    ranking_col = "observed_net_value" if has_observed else "predicted_net_value"
    valid_rows = [row for row in rows if row.get(ranking_col) is not None]
    best = max(valid_rows, key=lambda row: row[ranking_col]) if valid_rows else None
    return {
        "rows": rows,
        "best": best,
        "ranking_col": ranking_col,
        "has_observed": has_observed,
        "conversion_value": float(conversion_value),
        "contact_cost": float(contact_cost),
    }


def _numeric_policy_series(df: pd.DataFrame, col: str | None, default: float) -> pd.Series:
    if col and col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default).astype(float)
    return pd.Series(float(default), index=df.index)


def budget_policy_optimizer(
    df: pd.DataFrame,
    *,
    uplift_col: str = "uplift_score",
    value_col: str | None = None,
    cost_col: str | None = None,
    risk_col: str | None = None,
    group_col: str | None = None,
    budget: float | None = None,
    max_contacts: int | None = None,
    conversion_value: float = 1.0,
    default_cost: float = 0.0,
    risk_weight: float = 1.0,
    min_net_value: float = 0.0,
    rank_by: str = "net_value",
) -> Dict[str, Any]:
    """Greedy budget-constrained policy optimizer for uplift decisions.

    The objective is:

        maximize sum_i uplift_i * value_i - cost_i - risk_weight * risk_i
        subject to sum_i cost_i <= budget and selected rows <= max_contacts

    This intentionally stays dependency-free and deterministic. It is the
    production policy layer between CATE/uplift scores and campaign/routing
    decisions, not a replacement for online holdout validation.
    """

    if uplift_col not in df.columns:
        raise ValueError(f"{uplift_col} is required for budget policy optimization.")
    if rank_by not in {"net_value", "roi"}:
        raise ValueError("rank_by must be either 'net_value' or 'roi'.")

    view = df.copy().reset_index(drop=False).rename(columns={"index": "source_index"})
    uplift = pd.to_numeric(view[uplift_col], errors="coerce").fillna(0.0).astype(float)
    value = _numeric_policy_series(view, value_col, conversion_value).clip(lower=0.0)
    cost = _numeric_policy_series(view, cost_col, default_cost).clip(lower=0.0)
    risk = _numeric_policy_series(view, risk_col, 0.0).clip(lower=0.0)
    gross = uplift * value
    risk_penalty = risk_weight * risk
    net = gross - cost - risk_penalty
    efficiency = net / np.where(cost.to_numpy() > 1e-12, cost.to_numpy(), 1.0)

    view["_policy_gross_value"] = gross
    view["_policy_cost"] = cost
    view["_policy_risk_penalty"] = risk_penalty
    view["_policy_net_value"] = net
    view["_policy_roi"] = efficiency
    view["_policy_rank_score"] = view["_policy_roi"] if rank_by == "roi" else view["_policy_net_value"]

    eligible = view[view["_policy_net_value"] > float(min_net_value)].copy()
    eligible = eligible.sort_values(["_policy_rank_score", "_policy_net_value"], ascending=False).reset_index(drop=True)
    if max_contacts is not None:
        eligible = eligible.head(max(0, int(max_contacts))).copy()

    if budget is None:
        selected = eligible.copy()
        budget_value = float(eligible["_policy_cost"].sum())
    else:
        budget_value = max(0.0, float(budget))
        selected_rows = []
        spend = 0.0
        for _, row in eligible.iterrows():
            row_cost = float(row["_policy_cost"])
            if spend + row_cost <= budget_value + 1e-12:
                selected_rows.append(row)
                spend += row_cost
        selected = pd.DataFrame(selected_rows, columns=eligible.columns)

    total_cost = float(selected["_policy_cost"].sum()) if len(selected) else 0.0
    total_gross = float(selected["_policy_gross_value"].sum()) if len(selected) else 0.0
    total_risk = float(selected["_policy_risk_penalty"].sum()) if len(selected) else 0.0
    total_net = float(selected["_policy_net_value"].sum()) if len(selected) else 0.0
    source_indices = [int(item) for item in selected.get("source_index", pd.Series(dtype=int)).tolist()]

    group_rows: list[dict[str, Any]] = []
    if group_col and group_col in selected.columns and len(selected):
        grouped = selected.groupby(group_col, dropna=False)
        for group, group_df in grouped:
            group_rows.append(
                {
                    "group": str(group),
                    "rows": int(len(group_df)),
                    "cost": _clean_float(group_df["_policy_cost"].sum()),
                    "gross_value": _clean_float(group_df["_policy_gross_value"].sum()),
                    "risk_penalty": _clean_float(group_df["_policy_risk_penalty"].sum()),
                    "net_value": _clean_float(group_df["_policy_net_value"].sum()),
                }
            )

    return {
        "schema_version": 1,
        "objective": "maximize_sum_incremental_net_value_under_budget",
        "formula": "sum(uplift_i * value_i - cost_i - risk_weight * risk_i), subject to sum(cost_i)<=budget",
        "rank_by": rank_by,
        "budget": _clean_float(budget_value),
        "max_contacts": max_contacts,
        "eligible_rows": int(len(eligible)),
        "selected_rows": int(len(selected)),
        "selected_fraction": _clean_float(len(selected) / len(view)) if len(view) else None,
        "budget_used": _clean_float(total_cost),
        "budget_remaining": _clean_float(max(0.0, budget_value - total_cost)),
        "gross_value": _clean_float(total_gross),
        "risk_penalty": _clean_float(total_risk),
        "net_value": _clean_float(total_net),
        "roi": _clean_float(total_net / total_cost) if total_cost > 1e-12 else None,
        "iroas": _clean_float(total_gross / total_cost) if total_cost > 1e-12 else None,
        "mean_selected_uplift": _clean_float(selected[uplift_col].mean()) if len(selected) else None,
        "min_selected_net_value": _clean_float(selected["_policy_net_value"].min()) if len(selected) else None,
        "selected_source_indices": source_indices,
        "group_summary": group_rows,
        "guardrails": {
            "min_net_value": float(min_net_value),
            "risk_weight": float(risk_weight),
            "default_cost": float(default_cost),
            "note": "Offline budget policy is a launch candidate generator; final validation still requires holdout or randomized online experiment.",
        },
    }


def budget_policy_sensitivity(
    df: pd.DataFrame,
    *,
    budgets: Iterable[float],
    **kwargs,
) -> list[dict[str, Any]]:
    rows = []
    for budget in budgets:
        result = budget_policy_optimizer(df, budget=float(budget), **kwargs)
        rows.append(
            {
                "budget": _clean_float(budget),
                "selected_rows": result.get("selected_rows"),
                "budget_used": result.get("budget_used"),
                "gross_value": result.get("gross_value"),
                "net_value": result.get("net_value"),
                "roi": result.get("roi"),
                "iroas": result.get("iroas"),
                "mean_selected_uplift": result.get("mean_selected_uplift"),
            }
        )
    return rows


def exploration_ope_guardrail_rows() -> List[Dict[str, str]]:
    """Production guardrails for moving from offline uplift to online adaptive policies."""

    return [
        {
            "stage": "1. Randomized exploration bucket",
            "purpose": "create overlap across treatment arms so the router/subsidy policy can estimate incremental effect instead of copying the old policy",
            "method": "small stratified random bucket, epsilon exploration or interleaved holdout",
            "formula_or_check": "P(A=a|X=x) must be bounded away from 0 for candidate actions",
            "applies_to": "LLM routing, coupon action choice, ad audience, driver/passenger subsidy",
            "risk_if_missing": "off-policy estimates inherit selection bias; high-value actions may never be observed for similar contexts",
        },
        {
            "stage": "2. Off-policy evaluation",
            "purpose": "estimate the value of a new policy before sending all traffic to it",
            "method": "IPS / self-normalized IPS / doubly robust OPE",
            "formula_or_check": "V_DR(pi)=mean[mu_hat(pi(x),x)+1{a=pi(x)}/p(a|x)*(y-mu_hat(a,x))]",
            "applies_to": "budgeted targeting, routing threshold, multi-action policy",
            "risk_if_missing": "offline compare may overstate policy value when historical action probabilities are uneven",
        },
        {
            "stage": "3. Calibrated fixed threshold",
            "purpose": "deploy a stable, explainable uplift policy before using online adaptive bandits",
            "method": "Top-K threshold with calibration, bootstrap CI, overlap trimming and cost/value guardrails",
            "formula_or_check": "treat if uplift_hat * value - cost - risk_penalty > 0 and CI/overlap gates pass",
            "applies_to": "coupon ROI, push frequency, LLM strong/RAG/tool/human escalation",
            "risk_if_missing": "positive QINI can still map to negative net value or unstable launch buckets",
        },
        {
            "stage": "4. Budget pacing",
            "purpose": "keep the policy inside daily/weekly spend, latency or labor constraints",
            "method": "rank by marginal policy value, reserve budget for high-uncertainty exploration, monitor burn rate",
            "formula_or_check": "maximize sum gain_i subject to sum cost_i <= B and action capacity constraints",
            "applies_to": "DiDi city-time budget, ad spend, human review queue, LLM model-cost budget",
            "risk_if_missing": "policy works on average but exhausts budget early or overloads a constrained action arm",
        },
        {
            "stage": "5. Contextual bandit upgrade",
            "purpose": "adapt when model prices, traffic mix, creative quality or subsidy response drift over time",
            "method": "Thompson/UCB/epsilon-greedy with causal logging, guardrail metrics and holdout",
            "formula_or_check": "use bandit only after logging, OPE, calibration and safety guardrails are stable",
            "applies_to": "LLM routing, ad bidding, marketplace subsidy, CRM action optimization",
            "risk_if_missing": "fixed uplift policy becomes stale; if added too early, bandit optimizes noisy or biased rewards",
        },
    ]


def _llm_action_net_values(df: pd.DataFrame) -> pd.DataFrame:
    user_value = df.get("user_value", pd.Series(1.0, index=df.index)).astype(float)
    risk_penalty = (
        df.get("hallucination_risk", pd.Series(0.0, index=df.index)).astype(float) * 0.10
        + df.get("evidence_failure_risk", pd.Series(0.0, index=df.index)).astype(float) * 0.08
        + df.get("judge_noise_risk", pd.Series(0.0, index=df.index)).astype(float) * 0.04
        + df.get("budget_pressure", pd.Series(0.0, index=df.index)).astype(float) * 0.03
    )
    latency = df.get("latency_penalty", pd.Series(0.0, index=df.index)).astype(float)
    action_cost = df.get("incremental_cost", pd.Series(0.0, index=df.index)).astype(float)

    nets = pd.DataFrame(index=df.index)
    nets["cheap_model"] = df["cheap_quality"].astype(float) * user_value
    for action, col in [
        ("strong_model", "strong_quality"),
        ("rag", "rag_quality"),
        ("tool", "tool_quality"),
        ("human_review", "human_quality"),
    ]:
        nets[action] = df[col].astype(float) * user_value - action_cost - latency - risk_penalty
    return nets


def _target_action(policy: str, df: pd.DataFrame, nets: pd.DataFrame) -> pd.Series:
    if policy == "all_cheap":
        return pd.Series("cheap_model", index=df.index)
    if policy == "all_strong":
        return pd.Series("strong_model", index=df.index)
    if policy == "oracle_best":
        return nets.idxmax(axis=1)
    if policy == "positive_gain_router":
        planned = df["route_action"].astype(str)
        planned_net = nets.lookup(df.index, planned) if hasattr(nets, "lookup") else np.array([nets.at[idx, action] for idx, action in planned.items()])
        cheap_net = nets["cheap_model"].to_numpy()
        return pd.Series(np.where(planned_net > cheap_net, planned, "cheap_model"), index=df.index)
    if policy == "conservative_router":
        planned = df["route_action"].astype(str)
        planned_net = np.array([nets.at[idx, action] for idx, action in planned.items()])
        gain = planned_net - nets["cheap_model"].to_numpy()
        cutoff = float(np.nanquantile(gain, 0.75))
        return pd.Series(np.where(gain > max(0.0, cutoff), planned, "cheap_model"), index=df.index)
    raise ValueError(f"Unknown policy: {policy}")


def llm_routing_ope_summary(
    dataset_path: str | Path = "examples/datasets/synthetic_llm_multi_action_routing_9k.csv",
    max_rows: int = 3000,
) -> Dict[str, Any]:
    """Small deterministic OPE lab for LLM multi-action routing.

    This is intentionally a didactic calculator, not a production OPE engine.
    It shows how logged propensity and action overlap control whether a new
    router can be evaluated offline.
    """

    path = Path(dataset_path)
    if not path.exists():
        return {"status": "missing", "path": str(path), "rows": [], "message": "dataset not found"}

    df = pd.read_csv(path, nrows=max_rows)
    required = {
        "treatment",
        "route_action",
        "propensity",
        "cheap_quality",
        "strong_quality",
        "rag_quality",
        "tool_quality",
        "human_quality",
        "selected_action_quality",
        "incremental_cost",
        "latency_penalty",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        return {"status": "missing_columns", "path": str(path), "rows": [], "missing": missing}

    nets = _llm_action_net_values(df)
    logged_action = pd.Series(np.where(df["treatment"].astype(int) == 1, df["route_action"].astype(str), "cheap_model"), index=df.index)
    propensity = df["propensity"].astype(float).clip(0.02, 0.98)
    logged_propensity = pd.Series(np.where(logged_action == "cheap_model", 1.0 - propensity, propensity), index=df.index).clip(0.02, 0.98)
    observed_reward = np.array([nets.at[idx, action] for idx, action in logged_action.items()])
    global_bias = float(np.nanmean(observed_reward)) * 0.05
    logged_direct = np.array([nets.at[idx, action] for idx, action in logged_action.items()]) * 0.95 + global_bias

    rows: list[dict[str, Any]] = []
    for policy in ["all_cheap", "all_strong", "positive_gain_router", "conservative_router", "oracle_best"]:
        target = _target_action(policy, df, nets)
        target_net = np.array([nets.at[idx, action] for idx, action in target.items()])
        direct_prediction = target_net * 0.95 + global_bias
        match = (target == logged_action).to_numpy()
        weights = match.astype(float) / logged_propensity.to_numpy()
        ips = float(np.nanmean(weights * observed_reward))
        snips = float(np.nansum(weights * observed_reward) / max(np.nansum(weights), 1e-9))
        dr = float(np.nanmean(direct_prediction + weights * (observed_reward - logged_direct)))
        rows.append(
            {
                "policy": policy,
                "coverage_rate": _clean_float(np.mean(match)),
                "effective_sample_size": _clean_float((np.nansum(weights) ** 2) / max(np.nansum(weights**2), 1e-9)),
                "mean_logged_propensity": _clean_float(np.nanmean(logged_propensity[match]) if np.any(match) else np.nan),
                "direct_method_value": _clean_float(np.nanmean(direct_prediction)),
                "ips_value": _clean_float(ips),
                "snips_value": _clean_float(snips),
                "doubly_robust_value": _clean_float(dr),
                "oracle_net_value": _clean_float(np.nanmean(target_net)),
                "launch_readiness": (
                    "high" if np.mean(match) >= 0.30 and ((np.nansum(weights) ** 2) / max(np.nansum(weights**2), 1e-9)) >= 200 else
                    "medium" if np.mean(match) >= 0.12 else
                    "low"
                ),
                "interview_note": "DR OPE is more stable when either reward model or logged propensity is usable; low coverage means collect exploration first.",
            }
        )

    return {
        "status": "ok",
        "path": str(path),
        "rows_used": int(len(df)),
        "rows": rows,
        "formula": "V_DR(pi)=mean[mu_hat(pi(x),x)+1{a=pi(x)}/p(a|x)*(r-mu_hat(a,x))]",
    }


def llm_routing_action_coverage_rows(
    dataset_path: str | Path = "examples/datasets/synthetic_llm_multi_action_routing_9k.csv",
    max_rows: int = 3000,
) -> list[dict[str, Any]]:
    path = Path(dataset_path)
    if not path.exists():
        return []
    df = pd.read_csv(path, nrows=max_rows)
    required = {"treatment", "route_action", "propensity"}
    if not required.issubset(df.columns):
        return []
    logged_action = pd.Series(np.where(df["treatment"].astype(int) == 1, df["route_action"].astype(str), "cheap_model"), index=df.index)
    propensity = df["propensity"].astype(float).clip(0.02, 0.98)
    logged_propensity = pd.Series(np.where(logged_action == "cheap_model", 1.0 - propensity, propensity), index=df.index).clip(0.02, 0.98)
    rows: list[dict[str, Any]] = []
    total = max(len(df), 1)
    for action in ["cheap_model", "strong_model", "rag", "tool", "human_review"]:
        mask = logged_action == action
        count = int(mask.sum())
        share = count / total
        rows.append(
            {
                "logged_action": action,
                "rows": count,
                "share": _clean_float(share),
                "mean_logged_propensity": _clean_float(logged_propensity[mask].mean() if count else np.nan),
                "support_status": "strong" if share >= 0.15 else "thin" if share >= 0.05 else "deficient",
                "recommendation": (
                    "usable for OPE"
                    if share >= 0.15
                    else "keep target policy conservative or add exploration"
                    if share >= 0.05
                    else "do not trust OPE for this action before collecting exploration"
                ),
            }
        )
    return rows


def llm_routing_bandit_drift_rows() -> list[dict[str, Any]]:
    periods = [
        {
            "period": "P1 stable launch",
            "traffic": 10000,
            "quality_lift": 0.075,
            "value_per_quality": 2.0,
            "unit_cost": 0.006,
            "latency_penalty": 0.002,
            "risk_penalty": 0.003,
            "budget": 18.0,
            "event": "offline uplift threshold is freshly calibrated",
        },
        {
            "period": "P2 strong price cut",
            "traffic": 10000,
            "quality_lift": 0.078,
            "value_per_quality": 2.0,
            "unit_cost": 0.0025,
            "latency_penalty": 0.002,
            "risk_penalty": 0.003,
            "budget": 18.0,
            "event": "provider price drops; static policy under-routes valuable queries",
        },
        {
            "period": "P3 silent quality regression",
            "traffic": 10000,
            "quality_lift": 0.035,
            "value_per_quality": 2.0,
            "unit_cost": 0.0025,
            "latency_penalty": 0.002,
            "risk_penalty": 0.004,
            "budget": 18.0,
            "event": "strong model quality regresses silently",
        },
        {
            "period": "P4 evidence risk spike",
            "traffic": 10000,
            "quality_lift": 0.060,
            "value_per_quality": 2.0,
            "unit_cost": 0.003,
            "latency_penalty": 0.003,
            "risk_penalty": 0.030,
            "budget": 18.0,
            "event": "RAG evidence failures increase risk penalty",
        },
        {
            "period": "P5 new tool arm onboarding",
            "traffic": 10000,
            "quality_lift": 0.095,
            "value_per_quality": 2.0,
            "unit_cost": 0.005,
            "latency_penalty": 0.003,
            "risk_penalty": 0.006,
            "budget": 18.0,
            "event": "new tool action needs forced exploration then adoption",
        },
    ]
    fixed_fraction = 0.20
    rows: list[dict[str, Any]] = []
    for item in periods:
        traffic = float(item["traffic"])
        gain_per_route = (
            float(item["quality_lift"]) * float(item["value_per_quality"])
            - float(item["unit_cost"])
            - float(item["latency_penalty"])
            - float(item["risk_penalty"])
        )
        unit_spend = max(float(item["unit_cost"]) + float(item["latency_penalty"]), 1e-9)
        budget_fraction = min(0.60, float(item["budget"]) / max(traffic * unit_spend, 1e-9))
        adaptive_fraction = 0.02 if gain_per_route <= 0 else min(0.60, max(0.05, budget_fraction))
        if "quality regression" in item["event"] or "regress" in item["event"]:
            adaptive_fraction = min(adaptive_fraction, 0.10)
        if "evidence failures" in item["event"]:
            adaptive_fraction = min(adaptive_fraction, 0.08)
        if "onboarding" in item["event"]:
            adaptive_fraction = max(0.08, min(adaptive_fraction, 0.35))

        fixed_net = traffic * fixed_fraction * gain_per_route
        adaptive_net = traffic * adaptive_fraction * gain_per_route
        all_strong_net = traffic * gain_per_route
        rows.append(
            {
                "period": item["period"],
                "event": item["event"],
                "gain_per_route": _clean_float(gain_per_route),
                "fixed_fraction": fixed_fraction,
                "adaptive_fraction": _clean_float(adaptive_fraction),
                "fixed_threshold_net": _clean_float(fixed_net),
                "adaptive_bandit_net": _clean_float(adaptive_net),
                "all_strong_net": _clean_float(all_strong_net),
                "adaptive_minus_fixed": _clean_float(adaptive_net - fixed_net),
                "budget": item["budget"],
                "interview_line": "fixed uplift is the launch policy; contextual bandit helps when price, quality, risk or action inventory drifts.",
            }
        )
    return rows
