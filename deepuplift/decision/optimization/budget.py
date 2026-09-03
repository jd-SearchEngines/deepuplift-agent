from __future__ import annotations

from typing import Iterable

import pandas as pd


def simulate_budget(
    predictions: pd.DataFrame,
    budgets: Iterable[float],
    *,
    effect_col: str = "uplift",
    outcome_value: float = 1.0,
    treatment_cost: float = 0.0,
) -> pd.DataFrame:
    frame = predictions.sort_values(effect_col, ascending=False).copy()
    rows = []
    for budget in budgets:
        count = len(frame) if treatment_cost <= 0 else min(len(frame), int(float(budget) // treatment_cost))
        selected = frame.head(count)
        value = float(selected[effect_col].sum()) * outcome_value
        cost = float(count) * treatment_cost
        rows.append({
            "budget": float(budget),
            "target_count": int(count),
            "incremental_outcome": float(selected[effect_col].sum()),
            "incremental_value": value,
            "cost": cost,
            "net_value": value - cost,
            "iroas": value / cost if cost > 0 else None,
        })
    return pd.DataFrame(rows)
