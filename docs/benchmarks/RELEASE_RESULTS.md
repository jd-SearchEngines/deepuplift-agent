# Release Results

This page is a compact index; the full evidence bundle is generated outside
Git under `release_runs/<run_id>/` and is intentionally not committed.

Final validation evidence for the `0.5.0b1` Beta candidate (2026-09-04):

- Run: `20260903T175621Z-42`; commit: `40b32dfdab84917ed3ebb1089a3ba91859faf545`.
- Verdict: `READY_FOR_BETA_RELEASE`.
- Hillstrom: 64,000 randomized rows; all native baselines and EconML passed.
  DR-Learner Qini was `32.2305`; X-Learner had the lowest calibration MAE,
  `0.00371`; S-Learner-RF was fastest at about `1.84s` in the isolated run.
- Criteo: 100,000 randomized rows sampled uniformly from the official v2.1
  stream; T/DR/RF models passed. DR-Learner-RF Qini was `21.9049` and
  S-Learner-RF was fastest at about `3.63s` in the isolated run.
- Optional adapters: CausalForestDML, CausalMLUpliftTree, and
  CausalMLUpliftRandomForest each passed real fit, predict, and metric smoke.
- Scale: 10K `4.24s`, 100K `41.69s`, 1M `571.50s`; 1M peak process RSS was
  `1386.9MB`. These are measured local runs, not extrapolations.
- Packaging: wheel and sdist built; fresh Python 3.11 venv install, import, and
  synthetic quickstart passed.

The full evidence bundle is generated outside Git under
`release_runs/<run_id>/`. Never copy raw Hillstrom or Criteo data into this
repository.
