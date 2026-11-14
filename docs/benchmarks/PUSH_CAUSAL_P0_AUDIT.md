# Push MRT P0 data audit

**Phase:** P0 — loader, schema audit, and readiness
**Evidence class:** `REAL_RANDOMIZED_MRT`
**Status:** `PASS` for the supplied upstream snapshot; P1 and later phases remain pending.
**Audit command:** `python scripts/audit_mybpmylife_mrt.py <local Data_MRT.zip> --output <external-report.json>`

## Provenance and licensing

The audit read the local snapshot from the upstream [MRT_myBPmyLife repository](https://github.com/WIRED-L-um/MRT_myBPmyLife). No raw file, participant row, extracted CSV, cache, or model artifact was added to this repository. The upstream repository snapshot did not contain a `LICENSE` file; the loader therefore records `LICENSE_UNVERIFIED` and does not make a redistribution claim.

| Source | Bytes | SHA-256 |
|---|---:|---|
| `Data_MRT.zip` | 13,389,920 | `172296f6ec08e75b9e3b1a86452ed4b0c09628aaa464eccab7802131810d2251` |
| `Data_MRT/MRT Data.csv` | 31,618,914 | `84fd53d83d311b8eb60cf6113de32e4a5d76e443d84bfcae0a6f7ebf4a357cc0` |
| `Data_MRT/Moderator.csv` | 37,793,803 | `d3aab33261b7ce439c49237c24acbd095196199d7ca04a4aa1a8ffb81a387d03` |

## Schema and assignment resolution

The data dictionary was read from the upstream workbook. The loader preserves `ParticipantIdentifier`, `PrefDatetimeRowID`, `DiceRollRowID`, study day, time-of-day, phase, pre-decision history, dice assignment, notification delivery, and outcome-specific fields. There is no calendar timestamp in the upstream table; ordinal study day and time-of-day fields are retained rather than synthesizing a timestamp.

Assignment is resolved only from explicit fields:

- `IsDiceRollSuccess == 0` → `assignment_arm = none`;
- `IsDiceRollSuccess == 1` and `DiceRollType == activity/diet` → that notification arm;
- anything else → unresolved and excluded from the analysis snapshot.

The declared MRT assignment probabilities are `none=0.75`, `activity=0.125`, and `diet=0.125`. They are not estimated from observed counts and are not overwritten by a propensity model. `NotificationType` is kept as a separate delivery field for future per-protocol analysis.

## Audit results

| Check | Result |
|---|---:|
| Raw rows | 193,723 |
| Analysis rows with explicit assignment | 189,241 |
| Unresolved assignment rows, retained only for audit | 4,482 |
| Participants | 294 |
| Exact `(participant, preference, dice)` duplicate rows | 0 |
| Duplicate preference keys (repeated dice events) | 216 rows |
| Duplicate non-null dice IDs | 0 |
| Invalid success values | 0 |
| Invalid dice types | 0 |
| Within-participant ordinal ordering violations | 0 |
| Participant-held-out split | 235 train / 59 test participants; 0 overlap |
| License status | `LICENSE_UNVERIFIED` |

### Assignment balance

| Arm | Declared probability | Empirical analysis-row proportion | Difference |
|---|---:|---:|---:|
| none | 0.7500 | 0.751285 | +0.001285 |
| activity | 0.1250 | 0.125116 | +0.000116 |
| diet | 0.1250 | 0.123599 | -0.001401 |

### Outcome-specific readiness

The two proximal outcomes remain separate:

- `SumValues_Steps_60minAfter`: activity notification outcome, 60-minute window; missingness 5.7732%.
- `NumValues_LowSaltChoices_24hrAfter`: dietary notification outcome, 24-hour window; missingness 2.3136%.

Delivery is not treated as assignment. Delivery is observed on 23.7277% of all raw rows and 97.6608% of assigned notification-arm rows. The observed delivery rate is descriptive only; it is not a causal compliance estimate.

## Gate decision

`P0 PASS` for the supplied snapshot. The loader, data dictionary mapping, fail-closed assignment handling, repeated-participant metadata, outcome windows, missingness, balance, provenance hashes, and participant-held-out split contract are implemented and tested. P1 randomized ITT analysis has not been run in this P0 report, and no real-data causal effect or business lift is claimed.
