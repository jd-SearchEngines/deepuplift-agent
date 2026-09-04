from __future__ import annotations

import numpy as np
import pandas as pd


def generate_coupon_data(rows: int = 10_000, *, random_state: int = 42) -> pd.DataFrame:
    """Generate private-data-free coupon allocation data with heterogeneous effects."""

    if rows < 200:
        raise ValueError("Coupon demo needs at least 200 rows for a train/test split.")
    rng = np.random.default_rng(random_state)
    affinity = rng.beta(2.0, 2.0, rows)
    price_sensitivity = rng.beta(2.0, 3.0, rows)
    sessions = rng.poisson(5, rows)
    orders_30d = rng.poisson(1.5, rows)
    spend_90d = np.maximum(0, rng.gamma(2.0, 35.0, rows))
    recency_days = rng.integers(1, 90, rows)
    region = rng.choice(["east", "north", "south", "west"], rows, p=[0.32, 0.24, 0.22, 0.22])
    treatment = rng.binomial(1, 0.5, rows)
    coupon_cost = np.where(treatment == 1, 10.0, 0.0)
    base_logit = -2.4 + 1.4 * affinity + 0.12 * sessions + 0.18 * orders_30d + 0.004 * spend_90d - 0.012 * recency_days
    true_uplift = 0.035 + 0.11 * affinity - 0.075 * price_sensitivity + 0.018 * (sessions >= 6)
    outcome_probability = 1.0 / (1.0 + np.exp(-(base_logit + treatment * true_uplift * 4.0)))
    outcome = rng.binomial(1, np.clip(outcome_probability, 0.01, 0.99))
    gmv = outcome * (25.0 + rng.gamma(2.0, 18.0, rows))
    margin = gmv * 0.35 - coupon_cost
    return pd.DataFrame(
        {
            "user_id": [f"user_{index:07d}" for index in range(rows)],
            "affinity": affinity,
            "price_sensitivity": price_sensitivity,
            "sessions_7d": sessions,
            "orders_30d": orders_30d,
            "spend_90d": spend_90d,
            "recency_days": recency_days,
            "region": region,
            "treatment": treatment,
            "coupon_received": treatment,
            "outcome": outcome,
            "conversion": outcome,
            "gmv": gmv,
            "margin": margin,
            "coupon_cost": coupon_cost,
            "true_uplift": true_uplift,
        }
    )
