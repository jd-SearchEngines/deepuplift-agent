# Push causal decision architecture

DeepUplift treats long-term Push as a repeated-decision problem. The unit of
analysis is `user × decision point`, while participant-level clustering and
time order remain explicit throughout the pipeline.

```text
local MRT source
    -> audit raw assignment / delivery / outcome fields
    -> canonical decision points
    -> participant-held-out model split
    -> proximal effect model or known-truth synthetic DGP
    -> timing / frequency / fatigue-aware policy
    -> constrained candidate builder and optimizer
    -> cluster-aware OPE and policy evidence
```

## Evidence boundaries

1. **Real randomized MRT**: assignment-based ITT and proximal outcomes only.
   Repeated rows are clustered by participant. Assignment is not exposure, and
   offline value is not online business lift.
2. **Semi-synthetic known truth**: generated potential outcomes, fatigue,
   carryover, overlap and policy value may be evaluated against truth. Results
   must be labeled `SEMI_SYNTHETIC_KNOWN_TRUTH`.
3. **Observational stress**: hidden confounding and MNAR remain sensitivity
   problems. No weighting method is allowed to claim identification without
   its assumptions.

## Layer contracts

- **Data layer**: `build_mrt_decision_points()` produces the canonical schema,
  validates assignment probabilities, retains unresolved rows for audit, and
  refuses to fabricate availability or timestamps.
- **Model layer**: consumes pre-treatment state and prior history only; reports
  proximal effects separately from cumulative effects.
- **Decision layer**: consumes predicted incremental effects, creates explicit
  candidate costs/constraints, and evaluates policies with cluster-aware OPE.

## Current implementation status

| Component | Status | Evidence |
|---|---|---|
| MRT raw schema and assignment audit | PASS | `deepuplift/data/public/mrt.py` |
| Canonical decision-point schema | PASS | `deepuplift/data/push.py` |
| Participant-held-out split | PASS | `participant_group_split()` |
| Proximal randomized baseline | PASS | `deepuplift/benchmarks/push_mrt.py` |
| Known-truth static identification stress | PASS | `deepuplift/benchmarks/push_semisynthetic.py` |
| Repeated-decision bootstrap OPE | PASS API smoke | `deepuplift/core/ope.py` |
| Longitudinal fatigue/carryover policy lab | NOT_RUN | next implementation gate |

The architecture is deliberately fail-closed: a missing upstream field is
reported as `NOT_AVAILABLE`, not replaced by an inferred production value.
