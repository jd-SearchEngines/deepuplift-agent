# Propensity, overlap, weighting

Use `estimate_nuisance()` for one auditable contract. Supported propensity estimators are logistic regression, gradient boosting, histogram gradient boosting, and optional LightGBM. Weighting is explicit: `ipw`, `stabilized_ipw`, `overlap`, or `none`.

Clipping and trimming are opt-in. When enabled, the evidence records strategy, threshold/range, rows removed, percentage removed, ESS before/after, and SMD balance before/after. ESS is `(sum weights)^2 / sum(weights^2)`, reported overall and per arm. Extreme propensities and common support are reported using p01/p05/p25/median/p75/p95/p99.
