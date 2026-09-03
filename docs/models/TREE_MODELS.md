# Tree and forest models

Stable native tree-backed meta learners are `S-Learner-RF`, `T-Learner-RF`, `X-Learner-RF`, and `DR-Learner-RF`. They share the new `EffectPrediction` contract and, for observational data, the unified nuisance output.

`EconMLCausalForest`, `CausalMLUpliftTree`, and `SkLiftTwoModels` are explicitly `OPTIONAL`/interface-only in v0.3 until a configured adapter can fit and predict. Deep models remain experimental/research; legacy implementations remain available through the compatibility layer.
