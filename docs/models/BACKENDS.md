# Model Backends

## Native reference backends

The v0.2 reference runtime includes four real binary-treatment baselines:

- `S-Learner`
- `T-Learner`
- `X-Learner`
- `DR-Learner`

They use scikit-learn outcome/effect estimators and all return the same
`EffectPrediction` contract. The coupon E2E uses these four models only.

## Optional adapters

`deepuplift.models.adapters` defines explicit boundaries for:

- EconML;
- CausalML;
- scikit-uplift.

These adapters are optional and do not make a missing package look runnable.
The existing legacy catalog in `deepuplift.core.registry` remains available
for the Streamlit workbench and has its own dependency guards.

## Capability vocabulary

`ModelCapabilities` records binary/multi/continuous support, task support,
sample-weight support, out-of-core support, propensity requirements and
backend name. A future adapter should declare these fields before entering
model discovery or business-mode routing.
