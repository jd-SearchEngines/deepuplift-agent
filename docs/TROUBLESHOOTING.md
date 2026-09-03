# Troubleshooting

- **Post-treatment leakage**: remove features created after assignment; they
  can make offline results look unrealistically strong.
- **No overlap / extreme propensity**: inspect common support and trim or
  redesign the target population; do not extrapolate outside support.
- **Too few treated/control rows**: reduce model complexity or collect more
  randomized data; nuisance cross-fitting also needs enough rows per arm.
- **Optional backend not installed**: install its extra in an isolated
  environment; release status is `NOT_INSTALLED`, not a native fallback.
- **Qini is negative**: ranking may be worse than the random policy; check
  treatment coding, leakage, sample size, and the outcome definition.
- **Good Qini but poor calibration**: ranking and effect magnitude are different
  objectives; calibrate or use policy validation before assigning costs.
- **Positive uplift but negative net value**: effect alone does not pay for the
  treatment; inspect `effect * outcome_value - cost` and budget constraints.

## When not to use DeepUplift

Do not use it as a causal shortcut when there is no treatment/control contrast,
the outcome happens before treatment, most features are post-treatment,
positivity is severely violated, or an observational analysis omits important
confounders. Uplift output cannot prove absolute causality or production safety.
