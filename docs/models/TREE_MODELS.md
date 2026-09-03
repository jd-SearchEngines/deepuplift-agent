# Tree and forest models

Stable native tree-backed meta learners are `S-Learner-RF`, `T-Learner-RF`, `X-Learner-RF`, and `DR-Learner-RF`. They share the new `EffectPrediction` contract and, for observational data, the unified nuisance output.

`CausalForestDML`, `CausalMLUpliftTree`, and `CausalMLUpliftRandomForest` now have
real optional adapters. They are dependency-gated and are reported as
`NOT_INSTALLED` until EconML or CausalML is installed and exercised in the
optional-backend workflow. `SkLiftTwoModels` remains explicitly
`INTERFACE_ONLY` with no silent fallback. Deep models remain
experimental/research; legacy implementations remain available through the
compatibility layer.
