from __future__ import annotations

import numpy as np


def expected_incremental_value(effect, outcome_value: float) -> np.ndarray:
    return np.asarray(effect, dtype="float64") * float(outcome_value)


def expected_net_value(effect, outcome_value: float, treatment_cost) -> np.ndarray:
    return expected_incremental_value(effect, outcome_value) - np.asarray(treatment_cost, dtype="float64")
