# Binary Coupon Targeting

Question: which users should receive a coupon?

This is a randomized binary treatment example. Features must be measured before
the coupon is assigned.

```python
import pandas as pd
from deepuplift.application import run_uplift_pipeline

df = pd.read_csv("coupon_rct.csv")
result = run_uplift_pipeline(
    data=df,
    feature_cols=["recency_days", "orders_30d", "spend_90d"],
    treatment_col="coupon_received",
    outcome_col="conversion",
    id_column="user_id",
    assignment_type="randomized",
    treatment_cost=10,
    outcome_value=35,
    budget=50_000,
)
```

Inspect `result.diagnostics`, `result.benchmark`, `result.prediction`,
`result.policy`, and `result.experiment_plan`. The policy answers “who” under
the supplied cost/value/budget assumptions; its value is not online lift.

Run the reproducible local demo with:

```bash
python examples/coupon_allocation/run.py
```
