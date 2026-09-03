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

The EconML and CausalML entries are real adapters: they import the configured
third-party estimator, fit it, normalize predictions to `EffectPrediction`, and
record backend/nuisance provenance. They remain dependency-gated and do not make
a missing package look runnable. scikit-uplift remains an explicit
`INTERFACE_ONLY` compatibility boundary until a real adapter is added.
The existing legacy catalog in `deepuplift.core.registry` remains available
for the Streamlit workbench and has its own dependency guards.

## Capability vocabulary

`ModelCapabilities` records binary/multi/continuous support, task support,
sample-weight support, out-of-core support, propensity requirements and
backend name. A future adapter should declare these fields before entering
model discovery or business-mode routing.
