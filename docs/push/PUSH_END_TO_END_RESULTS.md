# Push end-to-end evidence

This file records the current executable chain and its evidence boundaries.

```text
MRT-context DGP
  -> participant-group OOF proximal effect model
  -> model-produced Push candidates
  -> constrained optimizer
  -> sequential OPE diagnostics
```

## Current status

| Link | Status | Evidence class |
|---|---|---|
| Decision-point schema | PASS | REAL_RANDOMIZED_MRT contract / engineering |
| Longitudinal fatigue/carryover DGP | PASS | SEMI_SYNTHETIC_KNOWN_TRUTH |
| Participant-group proximal model | PASS synthetic; real fixture NOT_AVAILABLE | synthetic / real-data gate |
| Candidate Builder → optimizer | PASS | SEMI_SYNTHETIC_KNOWN_TRUTH + engineering |
| Sequential IPS/SNIPS | PASS API; horizon-16 support BLOCKED | SEMI_SYNTHETIC_KNOWN_TRUTH |
| Sequential DR | NOT_AVAILABLE | continuation-value model absent |
| Real MRT long-term policy value | NOT_AVAILABLE | time/availability/exposure support absent |

The optimizer chain is not a business-value claim. Steps and low-sodium food
choices remain separate outcomes, and the current binary synthetic chain does
not yet cover the real MRT three-arm Activity/Diet policy space.
