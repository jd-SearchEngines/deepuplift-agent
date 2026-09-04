# Multi Coupon Amount

For `0 / 5 / 10 / 20` currency-unit coupons, the largest uplift is not always
the best action. The decision layer evaluates:

```text
net value = treatment-specific effect * outcome value - treatment cost
```

```python
from deepuplift.decision import build_multi_policy

policy = build_multi_policy(
    prediction,
    treatment_costs={5: 1.0, 10: 3.0, 20: 10.0},
    outcome_value=100.0,
    budget=2_000,
    optimizer="value_per_cost",
)
```

The result includes the recommended treatment, expected incremental outcome,
cost, net value, and budget utilization. `NO_TREATMENT` is always available.

```bash
python examples/coupon_amount/run.py
```

This path is experimental and offline-only.
