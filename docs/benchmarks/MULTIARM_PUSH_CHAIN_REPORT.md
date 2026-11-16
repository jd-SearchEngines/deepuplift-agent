# Multi-arm Push Candidate Chain

Status: **PASS** (semi-synthetic known-truth only)

| Seed | Solver | Feasible | Selected | Truth value | IPW value | Abs error | Bootstrap |
|---:|---|---|---:|---:|---:|---:|---|
| 42 | greedy | True | 0 | 0.348242 | 0.322805 | 0.025437 | cluster |
| 42 | heap | True | 0 | 0.348242 | 0.322805 | 0.025437 | cluster |
| 52 | greedy | True | 0 | 0.353511 | 0.376948 | 0.023437 | cluster |
| 52 | heap | True | 0 | 0.353511 | 0.376948 | 0.023437 | cluster |
| 62 | greedy | True | 0 | 0.347455 | 0.377347 | 0.029893 | cluster |
| 62 | heap | True | 0 | 0.347455 | 0.377347 | 0.029893 | cluster |
| 72 | greedy | True | 0 | 0.349958 | 0.345036 | 0.004922 | cluster |
| 72 | heap | True | 0 | 0.349958 | 0.345036 | 0.004922 | cluster |
| 82 | greedy | True | 0 | 0.353490 | 0.383255 | 0.029765 | cluster |
| 82 | heap | True | 0 | 0.353490 | 0.383255 | 0.029765 | cluster |

## Gates

- three_arm_logged_probability_ope: PASS
- candidate_optimizer_chain: PASS
- real_mrt: BLOCKED; local authorized raw myBPmyLife archive is not available
- online_lift: NOT_RUN

## Limitations

- Candidates use known truth as an oracle diagnostic; this is not a fitted-model quality claim.
- Policy value uses logged arm probabilities and cluster bootstrap but remains offline semi-synthetic evidence.
