# Experiment A — Local Model Comparison Results

**Run date:** 2026-11-13  
**Repository:** `jd-SearchEngines/deepuplift-agent`  
**Branch:** `codex/push-mrt-causal-benchmark`  
**Validated code SHA:** `18e1cfe32315686ba97e1f30460e7314dad5a8d0` (Experiment A implementation; this report is added in the following documentation commit)  
**Evidence scope:** local Mac execution only; no cloud GPU, GitHub Actions training, external model API, or raw-data upload.

## Executive status

**PARTIAL: synthetic execution PASS; required real-data lanes BLOCKED_ON_DATA.**

The deterministic binary, multi-treatment and continuous software paths were actually executed. The local environment passed its native arm64/MPS checks. The required myBPmyLife, Hillstrom, Criteo, X5, IHDP and ACIC files were not present in the bounded local audit, so no real-data metric is reported and no synthetic result is described as a Push business result.

The detailed machine-readable result, including every per-seed row, model status, device, runtime, RSS, configuration and metric is kept outside Git at:

```text
/tmp/deepuplift-experiment-a/experiment_a_results_v1_slim.json
```

It is intentionally not committed because it is a generated run artifact and may contain large prediction previews. The repository report contains aggregate evidence and representative per-seed tables.

## Environment

| Item | Measured value |
|---|---|
| Host | Mac15,8, arm64, 16 logical CPUs |
| Memory | 64 GiB unified memory |
| Disk | about 2.7 TiB free at preflight |
| macOS | 26.6.2 (25G83) |
| Python | 3.11.15 in `.venv-experiment-a` |
| PyTorch | 2.15.0 |
| MPS | built and available; real float32 forward/backward PASS |
| NumPy / pandas / SciPy | 1.26.4 / 3.0.6 / 1.17.1 |
| scikit-learn / EconML | 1.6.1 / 0.16.0 |
| LightGBM | installed but import BLOCKED by missing macOS `libomp.dylib`; not used for reported rows |

The system Python 3.9.6 does not satisfy the project requirement `Python >=3.10`; it was not used. Full preflight evidence is in [LOCAL_MAC_PREFLIGHT.md](LOCAL_MAC_PREFLIGHT.md). Classical estimators run on CPU. DRNet, VCNet and GIKS use explicit MPS in the final A3 run; DoseResponseGBM and GIKS's NumPy/GP stage remain CPU operations. Neural jobs were sequential.

Peak memory is sampled absolute process RSS using psutil. It includes the Python interpreter and allocator state; it is not a causal attribution of bytes to a single layer. This avoids incorrectly mixing macOS lifetime `ru_maxrss` into per-model measurements.

## Dataset availability and evidence class

| Dataset | Local status | Experiment use | License / evidence boundary |
|---|---|---|---|
| myBPmyLife MRT | `NOT_AVAILABLE` | real Push A1/A2 and Push-context A3 not run | upstream terms remain unverified; no raw archive found |
| Hillstrom | `NOT_AVAILABLE` | real external A1/A2 not run | upstream challenge terms require review |
| Criteo unbiased v2.1 | `NOT_AVAILABLE` | 100k/500k/1m scale not run | CC BY-NC-SA 4.0 terms require review; no official file found |
| X5 RetailHero | `NOT_AVAILABLE` | optional observational check not run | assignment randomization unverified |
| IHDP / ACIC | `NOT_AVAILABLE` | optional external truth checks not run | upstream terms/access not verified |
| DeepUplift deterministic binary fixture | `AVAILABLE` | A1 software validation | `SYNTHETIC_SCALE_STRESS`, not MRT-derived |
| DeepUplift deterministic continuous fixture | `AVAILABLE` | A3 software validation | `SYNTHETIC_SCALE_STRESS`, not MRT-derived |

No unrelated Criteo-named files under another project directory were substituted. No raw or participant-level data was added to Git.

## A1 — binary treatment

The reported synthetic lane ran 6 models × 2 sizes × 5 seeds = **60/60 PASS**. Splits were deterministic and stratified; every model produced its own held-out effect prediction. All 10 size×seed groups had six distinct prediction fingerprints.

### PEHE by seed

Column order: S, T, X, R, DR, CausalForestDML.

| n | seed | PEHE |
|---:|---:|---|
| 5,000 | 42 | 0.0311 / 0.0381 / 0.0309 / 0.2024 / 0.1214 / 0.0321 |
| 5,000 | 52 | 0.0254 / 0.0320 / 0.0252 / 0.2069 / 0.0343 / 0.0272 |
| 5,000 | 62 | 0.0303 / 0.0327 / 0.0271 / 0.2005 / 0.0383 / 0.0278 |
| 5,000 | 72 | 0.0291 / 0.0358 / 0.0280 / 0.1962 / 0.0366 / 0.0298 |
| 5,000 | 82 | 0.0311 / 0.0372 / 0.0307 / 0.2007 / 0.0354 / 0.0319 |
| 20,000 | 42 | 0.0258 / 0.0255 / 0.0216 / 0.1996 / 0.0250 / 0.0261 |
| 20,000 | 52 | 0.0258 / 0.0275 / 0.0236 / 0.2005 / 0.0258 / 0.0280 |
| 20,000 | 62 | 0.0260 / 0.0259 / 0.0219 / 0.1978 / 0.0244 / 0.0277 |
| 20,000 | 72 | 0.0255 / 0.0259 / 0.0223 / 0.1974 / 0.0243 / 0.0263 |
| 20,000 | 82 | 0.0259 / 0.0267 / 0.0230 / 0.1973 / 0.0241 / 0.0277 |

### A1 aggregate

| n | Model | PEHE mean ± SD | Qini mean | mean runtime (s) | mean RSS (MB) |
|---:|---|---:|---:|---:|---:|
| 5k | S-Learner | 0.0294 ± 0.0024 | 99.58 | 1.013 | 571.5 |
| 5k | T-Learner | 0.0352 ± 0.0025 | 99.52 | 1.037 | 571.8 |
| 5k | X-Learner | 0.0284 ± 0.0024 | 100.15 | 1.328 | 572.0 |
| 5k | R-Learner | 0.2013 ± 0.0039 | 80.63 | 0.347 | 572.0 |
| 5k | DR-Learner | 0.0532 ± 0.0380 | 99.25 | 0.622 | 572.0 |
| 5k | CausalForestDML | 0.0298 ± 0.0019 | 99.74 | 1.883 | 707.6 |
| 20k | S-Learner | 0.0258 ± 0.0002 | 403.82 | 4.548 | 1487.3 |
| 20k | T-Learner | 0.0263 ± 0.0008 | 406.24 | 4.540 | 1482.1 |
| 20k | X-Learner | 0.0225 ± 0.0008 | 406.48 | 5.509 | 1482.2 |
| 20k | R-Learner | 0.1985 ± 0.0013 | 300.61 | 1.417 | 1482.3 |
| 20k | DR-Learner | 0.0247 ± 0.0007 | 406.34 | 2.370 | 1479.4 |
| 20k | CausalForestDML | 0.0271 ± 0.0007 | 404.88 | 7.144 | 1669.5 |

These are synthetic-effect metrics only. The Qini values are not normalized across sample sizes. The strongest observed CATE accuracy in this fixture is X-Learner; CausalForestDML and S-Learner are close at 5k. This is not evidence that X-Learner is best for real Push.

## A2 — multi-treatment

The 3-arm synthetic lane ran One-vs-Control and the existing unified `MultiTreatmentOutcome` adapter for 2 sizes × 5 seeds = **20/20 PASS**. The generated data included explicit per-user eligibility masks. Independent evaluation found **0 eligibility violations** across all rows.

| n | Model | Arm-1 PEHE | Arm-2 PEHE | Policy regret | Best-action accuracy | Runtime (s) | RSS (MB) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 5k | One-vs-Control | 0.0460 | 0.0762 | 0.00192 | 0.9322 | 0.726 | 1761.7 |
| 5k | MultiTreatmentOutcome | 0.0291 | 0.0376 | 0.00171 | 0.9354 | 0.406 | 1761.7 |
| 20k | One-vs-Control | 0.0396 | 0.0703 | 0.00073 | 0.9586 | 2.595 | 1753.2 |
| 20k | MultiTreatmentOutcome | 0.0172 | 0.0240 | 0.00056 | 0.9640 | 1.298 | 1750.4 |

The unified adapter is an existing shared API outcome model, not MEMENTO, HydraNet, M3TN, MTMT or EFIN. The lower error in this synthetic DGP is therefore evidence about this adapter and fixture, not those frontier papers.

## A3 — continuous treatment

The continuous lane ran 5 models × 2 sizes × 5 seeds × 3 shapes = **150/150 PASS**. Models used the same split and dose grid. The final run used CPU for DoseResponseGBM and MPS for DRNet, VCNet, GIKS-DRNet and GIKS-VCNet. The generic fixture is not derived from myBPmyLife because the MRT file was unavailable.

Values below are five-seed means. MISE is lower-is-better; regret is lower-is-better.

| Shape / n | DoseResponseGBM MISE | DRNet MISE | VCNet MISE | GIKS-DRNet MISE | GIKS-VCNet MISE |
|---|---:|---:|---:|---:|---:|
| monotone / 5k | 0.24799 | 0.24252 | **0.21823** | 0.24069 | 0.24275 |
| monotone / 20k | 0.23415 | 0.23759 | 0.22567 | 0.22600 | **0.20959** |
| saturating / 5k | 0.12972 | 0.11676 | **0.10316** | 0.12406 | 0.12266 |
| saturating / 20k | 0.14729 | 0.15511 | 0.13797 | 0.12778 | **0.12245** |
| fatigue / 5k | **0.04084** | 0.06269 | 0.04556 | 0.04852 | 0.04707 |
| fatigue / 20k | 0.02542 | 0.03189 | **0.02402** | 0.03257 | 0.03048 |

Overall, VCNet/GIKS-VCNet are competitive on curve MISE, while DoseResponseGBM and VCNet are efficient on this local fixture. GIKS has materially higher training time because of its augmentation/GP stage. No extrapolated optimum claim is made; support guards were active.

Representative mean training cost at 20k: DoseResponseGBM about 1.6 s, VCNet about 0.7 s, DRNet about 6.3 s, GIKS-VCNet about 1.6 s and GIKS-DRNet about 10.5–10.9 s. Absolute sampled RSS reached about 8 GB in the largest sequential MPS run, below the 64 GB host capacity but high enough to prohibit concurrent jobs.

## Frontier model execution status

The registry was audited rather than treating source files as adapters. The following Tier-1 catalogue names were not run as the original paper models because the repository has no tested `EffectPrediction` adapter, exact dependency path, or verified data contract for them:

- A1: TARNet, CFRNet, DragonNet, FlexTENet, EFIN, DESCN, EUEN/EEUEN, TransTEE, HyperITE;
- A2: MEMENTO, HydraNet, M3TN, MTMT and multi-treatment EFIN/HyperITE;
- A3: SCIGAN, CRNet/WDBN, HNet and DR²CE.

They are recorded as `PAPER_ONLY_NOT_IMPLEMENTED` / `BLOCKED_DEPENDENCY_OR_DATA` in the local JSON. The existing registry marks TransTEE `OPTIONAL_NOT_VALIDATED`. The legacy deep modules are not silently called faithful adapters. This is a known completion blocker for the normative frontier-comparison scope.

## Acceptance gates

| Gate | Status | Evidence |
|---|---|---|
| Environment and MPS preflight | PASS | [LOCAL_MAC_PREFLIGHT.md](LOCAL_MAC_PREFLIGHT.md), MPS smoke |
| A1 method-specific synthetic execution | PASS | 60/60 rows, distinct prediction fingerprints, five seeds and two sizes |
| A2 eligibility and unified-vs-One-vs-Control synthetic execution | PASS | 20/20 rows, 0 mask violations |
| A3 three-shape continuous synthetic execution | PASS | 150/150 rows, five seeds and two sizes |
| Real MRT/Hillstrom/Criteo validation | BLOCKED_ON_DATA | all target files absent locally |
| Frontier Tier-1 reproduction | PARTIAL / BLOCKED | exact adapters/data contracts are not available |
| Longitudinal MRT overlay | NOT_RUN | requires local MRT data |
| Full regression tests | PASS | `95 passed, 2 skipped` |
| Experiment A overall | PARTIAL | synthetic lane complete; real and frontier lanes remain incomplete |

## Main limitations and next steps

1. Supply lawfully obtained local files and their source hashes, beginning with myBPmyLife. Re-run P0 audit, then real MRT A1/A2 separately by outcome window and participant cluster.
2. Add Hillstrom and Criteo loaders/runs only after local availability and terms are verified; run Criteo 100k/500k/1m sequentially.
3. Implement or import only license-compatible, faithful adapters for the frontier Tier-1 models, with treatment descriptors/exposure fields where required. Keep maturity labels and add model-specific tests before ranking.
4. Extend the existing continuous report to record per-seed confidence intervals and the full monotone/saturating/fatigue Push-context generator once MRT context is available.
5. Run the longitudinal overlay (WCLS, history-feature model and a validated sequence candidate) only with participant-grouped MRT data. Real MRT PEHE remains unavailable.

This report does not claim online uplift, real business value, causal superiority on myBPmyLife, or that MPS is faster than CUDA. It reports only the measured local synthetic execution and the concrete data/implementation blockers.
