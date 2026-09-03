# Model Backends

## Native reference backends

The v0.3 reference runtime includes real binary-treatment baselines:

- `S-Learner`
- `T-Learner`
- `X-Learner`
- `DR-Learner`
- `R-Learner`
- `IPW-Learner`
- `S-Learner-RF`, `T-Learner-RF`, `X-Learner-RF`, `DR-Learner-RF`

They use scikit-learn outcome/effect estimators and all return the same
`EffectPrediction` contract. Observational DR/R/IPW reuse a cross-fitted
`NuisanceResult`; they do not silently fit a second propensity model.

## Optional adapters

`deepuplift.models.adapters` defines explicit boundaries for:

- EconML;
- CausalML;
- scikit-uplift.

These adapters are optional/interface-only in v0.3 until a configured estimator
can fit and predict. They do not make a missing package look runnable.
The existing legacy catalog in `deepuplift.core.registry` remains available
for the Streamlit workbench and has its own dependency guards.

## Capability vocabulary

`ModelCapabilities` records binary/multi/continuous support, task support,
sample-weight support, out-of-core support, propensity requirements and
backend name. A future adapter should declare these fields before entering
model discovery or business-mode routing.
