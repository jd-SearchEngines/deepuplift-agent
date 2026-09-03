# Continuous Subsidy

For a dose in `0~20`, the raw effect-maximizing dose may not maximize economics.
`DoseResponseGBM` preserves the dose grid and dose-specific outcome/effect
predictions. The decision layer evaluates every dose, including dose `0`:

```text
net value(dose) = effect(x, dose) * outcome value - cost(dose)
```

```python
from deepuplift.decision import build_continuous_policy

policy = build_continuous_policy(
    prediction,
    dose_cost=lambda dose: 0.01 * dose,
    outcome_value=1.0,
)
```

The output is marked `EXPERIMENTAL` and `OFFLINE_ONLY`; it is a reference
economic decision example, not a production dose-response claim.

```bash
python examples/subsidy_optimization/run.py
```
