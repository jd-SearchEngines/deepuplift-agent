# Core Contracts

DeepUplift uses four shared objects to keep the layers independent.

| Contract | Owner | Consumer | Meaning |
| --- | --- | --- | --- |
| `CausalDataset` | Data | Models, Decision | Data plus column semantics, treatment type, assignment type and metadata |
| `DataDiagnostics` | Data | Application, UI, routing | Balance, missingness, propensity, overlap, leakage and readiness |
| `EffectPrediction` | Models | Decision | Unit-level `y0`, `y1`, uplift/CATE, treatment effects, propensity and uncertainty |
| `PolicyResult` | Decision | Application, UI, experiment system | Recommended treatment, economics, constraints, reasons and summary |

`EffectPrediction` is the most important boundary: the decision layer never
needs to know whether the source was a meta-learner, neural network, EconML,
CausalML or scikit-uplift.

## Treatment types

- `binary`: one treatment versus control; the reference coupon pipeline is complete.
- `multi_discrete`: several discrete actions such as 0/5/10/20 RMB coupons; the prediction contract can hold action-specific effects.
- `continuous`: dose or discount intensity; the contract reserves recommended dose and effect bounds for dose-response backends.

## Assignment types

- `randomized`: inspect arm ratio, balance, missingness and leakage.
- `observational`: additionally estimate propensity, inspect common support and warn about ignorability/positivity assumptions.

The contracts accept backend data objects and only materialize pandas at the
reference adapter boundary. This keeps future Polars or distributed adapters
possible without putting a dataframe dependency into every public object.
