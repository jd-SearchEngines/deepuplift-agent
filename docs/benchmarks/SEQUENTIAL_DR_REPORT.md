# Sequential DR Benchmark

Status: **PASS** (known-truth semi-synthetic only)

This report validates a finite-horizon sequential doubly robust estimator using supplied target-policy V and logged-action Q continuation values. It does not claim real MRT performance or online lift.

| Seed | Truth | Ordinary per-decision DR | Sequential DR | Sequential abs error | Trajectory ESS | Max trajectory weight |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 2.029473 | 0.510506 | 2.042025 | 0.012552 | 77.00 | 16.00 |
| 52 | 2.028156 | 0.513405 | 2.053621 | 0.025465 | 88.00 | 16.00 |
| 62 | 2.044159 | 0.507931 | 2.031722 | 0.012437 | 73.00 | 16.00 |
| 72 | 2.016878 | 0.505685 | 2.022738 | 0.005861 | 82.00 | 16.00 |
| 82 | 2.057332 | 0.518320 | 2.073281 | 0.015949 | 70.00 | 16.00 |

## Gates

- Known-truth Sequential DR: PASS
- Ordinary DR and Sequential DR are reported as separate estimands: PASS
- Real MRT: BLOCKED; local authorized raw myBPmyLife archive is not available
- Online lift: NOT_RUN
