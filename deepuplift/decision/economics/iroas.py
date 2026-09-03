from __future__ import annotations


def calculate_iroas(incremental_value: float, treatment_cost: float) -> float | None:
    if treatment_cost <= 0:
        return None
    return float(incremental_value) / float(treatment_cost)
