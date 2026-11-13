# Experiment A — Reproduction Log

**Run date:** 2026-11-14
**Repository:** `jd-SearchEngines/deepuplift-agent`
**Branch:** `codex/push-mrt-causal-benchmark`
**Evidence class:** `SYNTHETIC_SCALE_STRESS` for the new neural run; this is not a real-data or Push result.

This log records what was actually inspected and executed. It distinguishes
official-source availability from the runnable adapter that lives in this
repository.

## Local run

Command:

```bash
.venv-experiment-a/bin/python scripts/run_experiment_a_local.py \
  --neural-only --sizes 1000 5000 10000 \
  --seeds 42 52 62 72 82 --neural-epochs 8 \
  --neural-device auto \
  --output /tmp/deepuplift-experiment-a/a1_neural_v1.json
```

The output remains outside Git. It contains 75 aggregate/per-run rows: 3
sample sizes × 5 seeds × 5 variants. All 75 completed successfully. The
device recorded by every row was MPS; no CPU fallback was needed.

## A1 neural adapter benchmark

The DGP is the existing known-truth binary fixture. The adapter produces its
own held-out `EffectPrediction` for every variant. `CFRNet_balance0` is an
intentional ablation whose objective reduces to the TARNet loss; its equality
with TarNet is therefore expected and documented, not hidden prediction reuse.

| Variant | PEHE mean ± SD | Paired PEHE delta vs TarNet | ATE bias | Calibration MAE | Runtime (s) |
|---|---:|---:|---:|---:|---:|
| TarNet | 0.0673 ± 0.0287 | 0.0000 | -0.0138 | 0.0483 | 0.489 |
| CFRNet, balance=0 | 0.0673 ± 0.0287 | 0.0000 | -0.0138 | 0.0483 | 0.876 |
| CFRNet, balance=1 | 0.0730 ± 0.0196 | +0.0057, 95% CI [-0.0028, 0.0142] | +0.0120 | 0.0551 | 0.809 |
| DragonNet, targeted off | 0.0682 ± 0.0253 | +0.0008, 95% CI [-0.0018, 0.0034] | -0.0106 | 0.0475 | 0.514 |
| DragonNet, targeted on | 2.8896 ± 1.6812 | +2.8223, 95% CI [1.9681, 3.6765] | -2.5212 | 2.8981 | 0.523 |

These results do not establish that the mechanisms are harmful in general:
they show that this short, native adapter run and this tuning choice did not
support a targeted-regularization benefit, while CFR's balance penalty
increased PEHE slightly in this small known-truth fixture. Full architecture
reproduction requires the official preprocessing, objective, optimizer and
training schedule.

## Adapter and source boundary

The new adapter is [`torch_causal.py`](../../deepuplift/models/binary/torch_causal.py).
It reuses the retained network/loss definitions, adds the stable
`CausalDataset → fit → EffectPrediction` contract, records MPS/CPU device, and
exposes the CFR balance coefficient and Dragon targeted-regularization toggle.
Its maturity is explicitly `ARCHITECTURE_INSPIRED_VARIANT`; no external source
code was copied into the repository.

## Official source audit and blockers

The following HEAD commits were read or cloned into `/tmp` only. They are not
dependencies of the project and no raw data or third-party source was
committed.

| Model | Official repository HEAD | Source stack / contract | Local disposition |
|---|---|---|---|
| CFRNet | `e5080335b1d0b66ff30de7129ce4c096e84816a6` | TensorFlow 0.12.0-rc1, NumPy 1.11.3; IHDP files and legacy scripts | `BLOCKED_WITH_EVIDENCE`: legacy stack; native adapter run is architecture-inspired only |
| DragonNet | `2e51b2478854df389893bac85430bca5b70f5a26` | TensorFlow 1.13, Keras 2.2.4, NumPy 1.15; IHDP/ACIC scripts | `BLOCKED_WITH_EVIDENCE`: incompatible with current native MPS environment; native adapter run is not official reproduction |
| EFIN | `4c98210271c618e550fa3673fd7dd543c62bf577` | Python 3.8.5, PyTorch 1.13.1+cu117, Optuna 2.10, 8-GPU distributed command; GPL-3.0 | `BLOCKED_WITH_EVIDENCE`: CUDA/distributed and license boundary; treatment descriptors/Criteo data also unavailable |
| DESCN | `80fd11e02d05af241eeaf67d701be599da528cb1` | Python 3.7, PyTorch 1.10.1, CUDA 11.4; Lazada/ACIC schema | `BLOCKED_WITH_EVIDENCE`: legacy CUDA stack and no supplied Lazada/ACIC data |
| EUEN/EEUEN | `d85f24a2df32f8cfb974864046aa203461ee9e4b` | XDL/TensorFlow 1.12; requires separate assignment, exposure and outcome fields | `BLOCKED_WITH_EVIDENCE`: no XDL runtime or exposure-labeled local data |
| HydraNet | `63c5614b954ee550766163085724487a19096f42` | large pinned Python/Linux stack; IHDP/synthetic data; CPU/GPU flag exists | `NOT_RUN`: no faithful multi-arm adapter or verified local data contract |
| M3TN | `a4700f180746fe5132c68c0794d41ed7fc0ac81f` | Criteo-specific code; README invokes 8-process CUDA distributed launch | `BLOCKED_WITH_EVIDENCE`: no Criteo file and no single-device validated path |
| GIKS | `5b51a1e8eacf01ee7e0b5798ff0b35b8bac8b76f` | official AAAI-24 repository; current project has a native contract adapter | existing synthetic continuous evidence; official source parity not claimed |
| SCIGAN | `acdbcae0e317a5a6ce15d50990e16cdad30b2d1c` | official repository inspected; no current project adapter/data contract | `NOT_RUN`, source audit retained for next A3 lane |
| TransTEE | `3e6883a7edd7d780185d524553b13ae633b6a229` | official repository inspected; continuous integration remains unvalidated | `BLOCKED_WITH_EVIDENCE` in current registry |

Source URLs are maintained in the [frontier model catalogue](EXPERIMENT_A_FRONTIER_MODEL_CATALOG.md)
and [real-data/source register](EXPERIMENT_A_REAL_DATA_AND_CODE_REGISTER.md).

## Reproduction status

- `PASS`: runnable native TarNet/CFRNet/DragonNet architecture adapters with
  independent predictions, mechanism toggles, five seeds and three sizes.
- `PARTIAL`: CFR/Dragon official repositories are verified but not executed in
  their original legacy stacks; measured results are only the native adapter.
- `BLOCKED_WITH_EVIDENCE`: EFIN, DESCN and EUEN/EEUEN official runs due to
  CUDA/legacy runtime, data contract and/or license constraints.
- `NOT_RUN`: HydraNet, M3TN, SCIGAN and TransTEE faithful unified benchmarks.

The next implementation target is a source-backed A2 small three-arm lane
after a verified Hillstrom or Criteo file is present, followed by a faithful
continuous GPS/spline versus GIKS/SCIGAN/TransTEE audit. No frontier model is
promoted based only on appearing in the catalogue.

## Hillstrom real randomized lane (2026-11-14)

The Hillstrom CSV was retrieved locally from the public MineThatData source;
the raw file remains under `.local/experiment-a/raw/` and is ignored by Git.
SHA-256: `0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece`.
This is a real randomized email experiment, not a Push/MRT file and not a
counterfactual ground-truth fixture. The script uses the known 0.5 probability
for each pairwise active-arm-versus-control comparison, fits only outcome
nuisances, and reports design-based test ITT and policy-value estimates. PEHE
and individual-effect bias are deliberately `NOT_AVAILABLE`.

Command template (aggregate outputs remain outside Git):

```bash
.venv-experiment-a/bin/python scripts/run_hillstrom_real_benchmark.py \
  --path .local/experiment-a/raw/hillstrom.csv \
  --outcome conversion --seeds 42 52 62 72 82 \
  --output /tmp/deepuplift-experiment-a/hillstrom_conversion_v1.json
```

The conversion, visit and spend runs each produced **50/50 PASS** rows:
two pairwise comparisons × five seeds × five learners (`S`, `T`, `X`, `R`,
`DR`). The exact per-seed raw aggregate JSON files are retained in
`/tmp/deepuplift-experiment-a/` and are not committed.

| Outcome / active arm | ITT mean | Best mean Qini/AUUC learner | Best mean policy increment | Notes |
|---|---:|---|---:|---|
| conversion / Mens (1) | 0.007734 | T-Learner, 21.5415 | S-Learner, 0.007622 | randomized design; no PEHE |
| conversion / Womens (2) | 0.002784 | S-Learner, 9.2951 | S-Learner, 0.002463 | randomized design; no PEHE |
| visit / Mens (1) | 0.077830 | S-Learner, 217.8461 | S-Learner, 0.077755 | randomized design; no PEHE |
| visit / Womens (2) | 0.044156 | X-Learner, 165.0595 | X-Learner, 0.044435 | randomized design; no PEHE |
| spend / Mens (1) | 0.812912 | R-Learner, 2394.4343 | DR-Learner, 0.611201 | continuous outcome; Ridge nuisance warnings retained |
| spend / Womens (2) | 0.454647 | S/T/X tie, 1223.2044 | DR-Learner, 0.360620 | continuous outcome; Ridge nuisance warnings retained |

These are randomized external-data baselines and are not claims about Push
business value. The next real-data requirement remains a licensed myBPmyLife
MRT file with participant/time/eligibility/delivery/outcome semantics.
