# Experiment A — Model Comparison Results

Run date: 2026-11-14
Branch: `codex/push-mrt-causal-benchmark`
Hardware: local Mac M3 Max, 64GB unified memory; classical lanes on CPU; MPS neural lanes sequential.

## Status summary

| Lane | Executed rows | Status | Evidence |
|---|---:|---|---|
| Synthetic A1 binary | 60/60 | `PASS` | 6 learners × 2 sizes × 5 seeds; known truth |
| Synthetic A2 multi-treatment | 20/20 | `PASS` | One-vs-Control and unified outcome adapter; eligibility checked |
| Synthetic A3 continuous | 150/150 | `PASS` | 5 models × 2 sizes × 5 seeds × 3 response shapes |
| Synthetic A1 neural | 75/75 | `PASS` | TarNet/CFRNet/DragonNet architecture-inspired adapters on MPS |
| Hillstrom randomized real | 150/150 | `PASS` | 3 outcomes × 2 pairwise arms × 5 learners × 5 seeds |
| Criteo randomized real | 90/90 | `PASS` | conversion and visit × 3 sizes × 3 seeds × 5 learners |
| myBPmyLife real MRT | 0 | `NOT_AVAILABLE` | no local raw archive |
| MT-LIFT real multi-treatment | 0 | `ACCESS_BLOCKED` | official download requires authenticated Drive access |

## Criteo real randomized A1

The runner uses the published randomization probability 0.85, cross-fitted
outcome nuisances for R/DR, and design-aware IPW policy value. It reports
factual prediction metrics and uplift ranking separately; PEHE and individual
counterfactual bias are not computed on this randomized aggregate dataset.

### Conversion

Entries are mean over three seeds; `inc` is design-based policy increment over
the observed control mean. RSS is the maximum sampled process RSS within the
size group, so it includes interpreter/allocator state.

| n | Model | Qini/AUUC | inc | runtime s | RSS MB |
|---:|---|---:|---:|---:|---:|
| 100k | S / T / X / R / DR | 1.316 / 4.862 / -2.606 / 4.281 / 10.209 | -0.000171 / 0.000155 / -0.000139 / 0.000253 / 0.000685 | 0.52 / 0.57 / 1.11 / 1.02 / 1.31 | 471 |
| 500k | S / T / X / R / DR | 61.191 / 99.151 / 89.456 / 70.856 / 101.851 | 0.000028 / 0.001257 / 0.001286 / 0.000627 / 0.001407 | 1.66 / 1.93 / 3.12 / 3.03 / 3.66 | 1,122 |
| 1m | S / T / X / R / DR | 154.838 / 143.490 / 149.593 / 95.127 / 159.596 | 0.000703 / 0.000893 / 0.000958 / 0.000842 / 0.001169 | 3.26 / 3.44 / 5.06 / 5.73 / 6.51 | 1,782 |

DR-Learner has the largest conversion Qini and policy increment at all three
reported sizes in this local run. This is a local model-selection observation,
not a causal superiority claim.

### Visit

| n | Model | Qini/AUUC | inc | runtime s | RSS MB |
|---:|---|---:|---:|---:|---:|
| 100k | S / T / X / R / DR | 83.665 / 72.585 / 53.593 / 34.651 / 90.918 | 0.005946 / 0.004685 / 0.003188 / 0.001521 / 0.004907 | 1.07 / 1.32 / 1.83 / 1.55 / 2.69 | 471 |
| 500k | S / T / X / R / DR | 794.048 / 716.961 / 830.606 / 499.225 / 719.761 | 0.010440 / 0.009442 / 0.010808 / 0.006983 / 0.009531 | 2.82 / 3.29 / 4.66 / 5.41 / 7.56 | 1,122 |
| 1m | S / T / X / R / DR | 1,524.095 / 1,376.281 / 1,389.155 / 934.765 / 1,448.575 | 0.010468 / 0.009164 / 0.009825 / 0.007417 / 0.010076 | 5.15 / 5.57 / 7.25 / 10.49 / 14.14 | 1,826 |

S-Learner leads visit policy increment at 500k/1m, while X-Learner leads Qini
at 500k. The leaders are outcome- and scale-dependent.

## Continuous Treatment A3

The established local suite ran DoseResponseGBM, DRNet, VCNet, GIKS-DRNet and
GIKS-VCNet on monotone, saturating and fatigue response surfaces at 5k/20k,
five seeds each. These are known-truth synthetic results. Five-seed MISE means
are shown below; lower is better.

| Shape / n | DoseResponseGBM | DRNet | VCNet | GIKS-DRNet | GIKS-VCNet |
|---|---:|---:|---:|---:|---:|
| monotone / 5k | 0.2480 | 0.2425 | **0.2182** | 0.2407 | 0.2427 |
| monotone / 20k | 0.2341 | 0.2376 | 0.2257 | 0.2260 | **0.2096** |
| saturating / 5k | 0.1297 | 0.1168 | **0.1032** | 0.1241 | 0.1227 |
| saturating / 20k | 0.1473 | 0.1551 | 0.1380 | 0.1278 | **0.1224** |
| fatigue / 5k | **0.0408** | 0.0627 | 0.0456 | 0.0485 | 0.0471 |
| fatigue / 20k | 0.0254 | 0.0319 | **0.0240** | 0.0326 | 0.0305 |

This supports the current continuous code path and support diagnostics. It is
not an official reproduction of every paper architecture and not a UVET or
Push business result.

## Method and metric separation

The Push semi-synthetic runner now separates `ate_estimators` from
`cate_learners`. ATE estimators report only ATE, bias, weights, ESS and balance;
they explicitly return `cate_policy_surface=NOT_EVALUATED`. S/T/X/R/DR learners
fit their own held-out CATE paths and report PEHE, CATE-derived ATE bias and
oracle policy diagnostics. A regression test verifies distinct prediction
fingerprints and prevents the old Ridge surface from being reused as every
method's CATE.

## Frontier model status

EFIN, DESCN, MEMENTO, HydraNet, M3TN, MTMT, CRNet/WDBN, HNet, SCIGAN and
TransTEE were source-audited, but no faithful, licensed, data-compatible local
reproduction was claimed. The existing native adapters are explicitly labelled
architecture-inspired or experimental where applicable.

## Overall gate

**PARTIAL.** The reproducible local benchmark lanes are executed and tested;
real myBPmyLife Push, MT-LIFT and full UVET lanes remain incomplete.
