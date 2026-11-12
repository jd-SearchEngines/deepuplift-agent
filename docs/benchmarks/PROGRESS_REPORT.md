# DeepUplift progress report

## 2026-11-13 execution batch

### Delivered

- Added a metadata-only registry for IHDP, ACIC 2016, Hillstrom, Criteo, myBPmyLife MRT, and X5 RetailHero.
- Added license/reuse policy, assignment type, ground-truth status, evaluation metrics, and local availability fields.
- Added `scripts/build_dataset_registry.py` to produce JSON and CSV without downloading data.
- Added optional participant/cluster bootstrap to generic OPE. Row bootstrap remains explicitly labeled `row`.
- Added prediction/cluster length validation for OPE.
- Added canonical repeated-decision schema with explicit `NOT_AVAILABLE` fields
  for MRT availability, calendar time, and time-since-last-push.
- Added a known-truth longitudinal laboratory for fatigue, spacing, carryover,
  heterogeneous effects, temporal drift, negative effects, and positivity stress.

### Measured evidence

- Dataset registry smoke: 6 datasets, 0 manifest validation failures, all raw-data availability explicitly `NOT_AVAILABLE` in the repository-only run.
- OPE repeated-decision smoke: 200 users × 5 decision rows, 300 bootstrap repetitions, cluster count 200. Row-bootstrap IPS CI: `[0.25095, 0.30603]`; cluster-bootstrap IPS CI: `[0.25195, 0.30362]`. The result is an uncertainty-method comparison, not a causal business claim.
- Full test suite before this batch: `76 passed, 2 skipped`.
- Decision schema and longitudinal laboratory tests: `5 passed`.
- Longitudinal smoke: 4 scenarios × 6 policies, 100 users × 8 decision points,
  seed `20261113`; report stored outside Git at `/tmp/deepuplift-longitudinal-smoke.json`.

### Not claimed

- No IHDP, ACIC, Hillstrom, Criteo, X5, or MRT raw file was downloaded or committed in this batch.
- No PEHE/CATE metric was emitted for the OPE smoke.
- Dataset registration does not mean the data is locally available or license-cleared.
- Longitudinal policy values are generated-truth evidence, not real MRT or online business outcomes.

### Next gate

Connect the longitudinal lab to learned proximal effects and sequential OPE, then build the unified semi-synthetic identification matrix with explicit DGP truth, five seeds, measured confounding/SSB/overlap scenarios, and bias/variance/coverage/ESS/policy-value outputs. The matrix must stop at `BLOCKED` when a method or dataset cannot support a valid estimand.
