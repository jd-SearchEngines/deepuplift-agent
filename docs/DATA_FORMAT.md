# Data Format

Every input needs an identifier (optional but recommended), pre-treatment
features, one treatment column, and one outcome column.

Binary:

```text
user_id | features... | treatment | outcome
```

Multi-discrete:

```text
treatment ∈ {0, 5, 10, 20}
```

Continuous:

```text
dose ∈ R
```

Features must be measured before treatment assignment. Do not include outcomes,
exposure, clicks, purchases, post-treatment engagement, or any other variable
caused by treatment. Declare `assignment_type="randomized"` for an RCT and
`assignment_type="observational"` for historical non-random assignment.
