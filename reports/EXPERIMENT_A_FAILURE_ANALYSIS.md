# Experiment A — Failure and Blocker Analysis

## No silent failures in completed lanes

The Criteo runner produced 45/45 PASS rows for conversion and 45/45 PASS rows
for visit. Hillstrom produced 50/50 PASS rows for each of conversion, visit and
spend. The existing synthetic A1/A2/A3 and neural lanes also completed their
declared rows. No row was converted from an exception or timeout into PASS.

## Warnings retained

The visit lane emitted scikit-learn LogisticRegression convergence warnings in
the cross-fitted nuisance path. Outputs remained finite and the rows were
marked PASS for execution, but this is a model-quality warning, not proof of
nuisance convergence. A future sensitivity run should compare scaled logistic,
calibrated tree and regularization settings under a pre-registered validation
criterion.

The Criteo learned policy target rates vary substantially by learner and size
(for example, conversion at 500k ranged from about 0.1% for S-Learner to about
99.8% for DR-Learner). This is a useful diagnostic of score calibration and
decision threshold behavior; it is not a recommendation to deploy any of the
policies.

## Data blockers

1. **myBPmyLife MRT:** no raw locally licensed archive was available. The
   semi-synthetic generator and fixture tests do not satisfy the real Push
   gate. Participant grouping, delivery/exposure, eligibility and delayed
   outcome semantics still need an audited local archive.
2. **MT-LIFT:** the official repository was inspected, but the public download
   path redirected to Google Drive login in this environment. No fake or
   partial multi-treatment result was produced.
3. **UVET/CUVET:** the official train artifact is about 10.3GB. A 50MB range
   was sufficient for schema/support audit but is not a uniform sample and was
   not used for a formal performance or model ranking claim.
4. **Frontier models:** source repositories frequently require legacy
   TensorFlow/CUDA/distributed stacks, special treatment descriptors or data
   contracts. Native experimental adapters are not described as faithful paper
   reproductions.

## Engineering issues addressed

The semi-synthetic benchmark previously evaluated CATE and policy metrics using
the ATE estimator's Ridge effect surface. That path is now separated and
covered by a regression test. Large Criteo ranking runs previously materialized
an O(N²) treatment-balance curve through repeated DataFrame prefix slicing; the
curve is now generated with NumPy cumulative sums and can be omitted for large
machine-readable runs while retaining scalar metrics and bootstrap stability.

## Remaining risks

- Criteo metrics are randomized external-data diagnostics, not individual
  ground-truth CATE metrics.
- Hillstrom and Criteo policy values are offline design-based estimates and
  require a deployment policy contract before business interpretation.
- The continuous A3 data-generating process has known truth, but not the
  longitudinal MRT history, fatigue, delay and participant dependence required
  for a real Push claim.
- The current Criteo execution uses three seeds for a bounded local run; the
  five-seed requirement is satisfied by the synthetic and Hillstrom lanes, not
  by this large real-data lane.
