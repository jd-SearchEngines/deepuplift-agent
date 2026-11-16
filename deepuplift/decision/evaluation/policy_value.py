from __future__ import annotations

from typing import Any

import numpy as np

from .uncertainty import bootstrap_ci


def fixed_budget_topk(
    scores: Any,
    *,
    budget: float,
    treatment_cost: float,
) -> dict[str, Any]:
    """Select the highest-scoring rows subject to a fixed treatment budget."""
    score_array = np.asarray(scores, dtype="float64").reshape(-1)
    if not np.isfinite(float(treatment_cost)) or treatment_cost <= 0:
        raise ValueError("treatment_cost must be positive for a fixed-budget policy.")
    if not np.isfinite(float(budget)) or budget < 0:
        raise ValueError("budget must be non-negative.")
    if not np.isfinite(score_array).all():
        raise ValueError("scores must be finite for a fixed-budget policy.")
    order = np.argsort(-score_array, kind="stable")
    capacity = float(budget) / float(treatment_cost)
    nearest_capacity = round(capacity)
    if np.isclose(capacity, nearest_capacity, rtol=0.0, atol=1e-12 * max(1.0, abs(capacity))):
        capacity = float(nearest_capacity)
    count = min(len(score_array), int(np.floor(capacity)))
    selected = np.zeros(len(score_array), dtype=bool)
    selected[order[:count]] = True
    return {
        "selected": selected.tolist(),
        "selected_indices": order[:count].astype(int).tolist(),
        "target_count": int(count),
        "budget": float(budget),
        "treatment_cost": float(treatment_cost),
        "budget_utilization": float(count * treatment_cost / budget) if budget > 0 else 0.0,
    }


def policy_value_contributions(
    treatment: Any,
    outcome: Any,
    policy_treatment: Any,
    propensity: Any,
    *,
    treatment_cost: float = 0.0,
    outcome_value: float = 1.0,
) -> np.ndarray:
    """Return per-row IPW contributions for a binary policy value."""
    t = np.asarray(treatment, dtype="float64").reshape(-1)
    y = np.asarray(outcome, dtype="float64").reshape(-1)
    policy_values = np.asarray(policy_treatment, dtype="float64").reshape(-1)
    if not np.isfinite(policy_values).all() or not np.isin(policy_values, (0.0, 1.0)).all():
        raise ValueError("policy_treatment must contain only finite binary values 0 or 1.")
    policy = policy_values.astype(bool)
    e = np.asarray(propensity, dtype="float64")
    if e.ndim == 0:
        e = np.full(len(t), float(e))
    e = e.reshape(-1)
    if not (len(t) == len(y) == len(policy) == len(e)):
        raise ValueError("treatment, outcome, policy_treatment and propensity must have equal length.")
    if not np.isfinite(t).all() or not np.isin(t, (0.0, 1.0)).all():
        raise ValueError("treatment must contain only finite binary values 0 or 1.")
    if not np.isfinite(y).all():
        raise ValueError("outcome must contain only finite values.")
    if not np.isfinite(float(treatment_cost)) or not np.isfinite(float(outcome_value)):
        raise ValueError("treatment_cost and outcome_value must be finite.")
    if not np.isfinite(e).all() or np.any((e <= 0) | (e >= 1)):
        raise ValueError("propensity must lie strictly between zero and one.")
    selected_outcome = np.where(policy, t * y / e, (1.0 - t) * y / (1.0 - e))
    return selected_outcome * float(outcome_value) - policy.astype("float64") * float(treatment_cost)


def fixed_budget_policy_value(
    scores: Any,
    treatment: Any,
    outcome: Any,
    propensity: Any,
    *,
    budget: float,
    treatment_cost: float,
    outcome_value: float = 1.0,
    bootstrap_samples: int = 1000,
    random_state: int = 42,
) -> dict[str, Any]:
    """Evaluate a fixed-budget Top-K policy with an original-row bootstrap CI."""
    selection = fixed_budget_topk(scores, budget=budget, treatment_cost=treatment_cost)
    contributions = policy_value_contributions(
        treatment,
        outcome,
        selection["selected"],
        propensity,
        treatment_cost=treatment_cost,
        outcome_value=outcome_value,
    )
    interval = bootstrap_ci(contributions, samples=bootstrap_samples, random_state=random_state)
    return {
        **selection,
        "policy_value": interval["estimate"],
        "policy_value_ci": {"low": interval["low"], "high": interval["high"]},
        "bootstrap": interval,
        "estimand": "IPW value of the fixed-budget binary policy",
    }
