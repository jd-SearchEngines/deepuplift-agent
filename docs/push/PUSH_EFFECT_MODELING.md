# Push proximal effect modeling

**Status:** `PASS` for the participant-held-out synthetic baseline;
`NOT_AVAILABLE` on the repository fixture because the complete-outcome sample
is intentionally too small. Real MRT results must be generated only after the
local source and license are verified.

The reference model is a WCLS-style centered treatment regression:

```text
Y_t = alpha(H_t) + (A_t - p_t) * beta(H_t) + error_t
```

where `p_t` is the known randomized probability for the binary arm-vs-none
comparison. The implementation uses inverse treatment-variance weights and
participant-group cross-fitting. Features are pre-treatment state, ordinal
study fields, and prior assignment history; delivery and future outcomes are
not model features.

The real MRT adapter fits Activity-vs-None or Diet-vs-None separately. It does
not combine Steps with Low-sodium Food Choices and does not emit real-data PEHE
or individual counterfactual accuracy.

## Missingness contract

The current baseline reports a complete-outcome held-out model. It records
rows before and after outcome filtering and missingness by arm. It does not
call the complete-case estimate a full-population ITT without a missingness
assumption. If there are too few participant groups or rows, the status is
`NOT_AVAILABLE`.

## Synthetic use

The longitudinal known-truth laboratory exposes `x0`, `x1`, `x2`, prior push
count, time since last push, and fatigue state as model-facing pre-decision
features. A fitted model can be evaluated against `true_proximal_effect` only
in the `SEMI_SYNTHETIC_KNOWN_TRUTH` evidence class.
