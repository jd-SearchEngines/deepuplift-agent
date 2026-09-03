from __future__ import annotations

from typing import Any

import numpy as np

from deepuplift.decision import build_binary_policy


def _summary(selected: np.ndarray, true_effect: np.ndarray, *, cost: float, outcome_value: float, budget: float | None = None) -> dict[str, Any]:
    selected = np.asarray(selected, dtype=bool)
    total_cost = float(selected.sum() * cost)
    value = float(true_effect[selected].sum() * outcome_value)
    return {"target_count": int(selected.sum()), "total_cost": total_cost, "incremental_outcome": float(true_effect[selected].sum()), "incremental_value": value, "net_value": value - total_cost, "iROAS": value / total_cost if total_cost else None, "budget_utilization": total_cost / budget if budget else None, "offline_only": True}


def coupon_policy_benchmark(dataset, prediction, *, budget: float = 500.0, treatment_cost: float = 10.0, outcome_value: float = 35.0) -> dict[str, Any]:
    frame = dataset.to_pandas().reset_index(drop=True)
    truth = frame["true_uplift"].to_numpy(dtype="float64") if "true_uplift" in frame else np.zeros(len(frame))
    n = len(frame)
    rng = np.random.default_rng(42)
    max_targets = min(n, int(budget // treatment_cost))
    top = np.argsort(np.asarray(prediction.uplift))[::-1]
    random_selected = np.zeros(n, dtype=bool); random_selected[rng.choice(n, max_targets, replace=False)] = True
    top_selected = np.zeros(n, dtype=bool); top_selected[top[:max_targets]] = True
    policy = build_binary_policy(prediction, treatment_cost=treatment_cost, outcome_value=outcome_value, budget=budget)
    net_selected = policy.to_frame()["recommended_treatment"].to_numpy(dtype=int) == 1
    oracle_order = np.argsort(truth)[::-1]
    oracle_selected = np.zeros(n, dtype=bool)
    profitable = oracle_order[(truth[oracle_order] * outcome_value - treatment_cost) > 0]
    oracle_selected[profitable[:max_targets]] = True
    policies = {"A_NO_TREATMENT": _summary(np.zeros(n, dtype=bool), truth, cost=treatment_cost, outcome_value=outcome_value, budget=budget), "B_RANDOM_TREATMENT": _summary(random_selected, truth, cost=treatment_cost, outcome_value=outcome_value, budget=budget), "C_TOP_PREDICTED_UPLIFT": _summary(top_selected, truth, cost=treatment_cost, outcome_value=outcome_value, budget=budget), "D_NET_VALUE_BUDGET": _summary(net_selected, truth, cost=treatment_cost, outcome_value=outcome_value, budget=budget), "ORACLE": _summary(oracle_selected, truth, cost=treatment_cost, outcome_value=outcome_value, budget=budget)}
    oracle_value = policies["ORACLE"]["net_value"]
    for name, row in policies.items():
        row["policy_regret_vs_oracle"] = float(oracle_value - row["net_value"])
    return policies
