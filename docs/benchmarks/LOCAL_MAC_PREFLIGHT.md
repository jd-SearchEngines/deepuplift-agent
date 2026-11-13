# Experiment A — Local Mac Preflight

**Audit date:** 2026-11-13 (Asia/Shanghai)  
**Repository:** `jd-SearchEngines/deepuplift-agent`  
**Branch:** `codex/push-mrt-causal-benchmark`  
**Evidence class:** environment and data-readiness audit; this is not model-performance evidence.

## Result

| Gate | Status | Evidence |
|---|---|---|
| Native Apple Silicon | PASS | `arm64`, Mac15,8, macOS 26.6.2 |
| Memory / disk | PASS | 64 GiB unified memory; approximately 2.7 TiB free at audit |
| Isolated Python | PASS | `.venv-experiment-a`, Python 3.11.15 |
| PyTorch MPS build/availability | PASS | PyTorch 2.15.0, `is_built=True`, `is_available=True` |
| Real MPS forward/backward | PASS | float32 Linear/ReLU smoke completed successfully |
| Target raw datasets | NOT_AVAILABLE | No local myBPmyLife, Hillstrom, Criteo, X5, IHDP or ACIC raw files found in the bounded audit |
| Experiment A full data comparison | BLOCKED_ON_DATA | Push-derived MRT-context runs and real-data validation require user-local files |

## Hardware and runtime

```text
Model:       Mac15,8
Architecture: arm64
CPU:         16 logical CPUs
Memory:      68,719,476,736 bytes (64 GiB)
macOS:       26.6.2 (25G83)
Python used for audit: /Users/jingdong/.homebrew/bin/python3.11
Project environment: .venv-experiment-a
```

The system `/usr/bin/python3` is Python 3.9.6 and does not satisfy the project requirement `Python >=3.10`; it was not used for Experiment A training. The isolated environment was created with the existing native Homebrew Python 3.11 installation. It is ignored by Git.

Installed core versions in the isolated environment include:

```text
deepuplift 0.5.0b1
numpy 1.26.4
pandas 3.0.6
scipy 1.17.1
scikit-learn 1.6.1
lightgbm 4.7.0
torch 2.15.0
econml 0.16.0
psutil 7.2.2
```

The exact environment manifest and package freeze are kept in the ignored local evidence area, not committed with participant or raw data.

## MPS smoke

A float32 two-layer network completed one forward/backward pass on both CPU and MPS. The MPS run completed with a finite loss and gradient. This validates the host/backend path only; it does not imply that every model operator is MPS-compatible. The existing continuous estimators remain CPU-routed unless a model-specific device path is explicitly enabled and validated.

PyTorch MPS does not support float64. Float64-dependent statistical estimators therefore remain on CPU. Neural jobs will be run sequentially, and any explicit MPS fallback will be recorded per model.

## Dataset availability audit

The metadata-only registry still correctly reports `NOT_AVAILABLE` for:

- myBPmyLife MRT (`Data_MRT.zip`): upstream source and license/reuse terms require verification; no local raw archive found;
- Hillstrom email RCT: no local CSV/CSV.GZ found;
- Criteo unbiased v2.1: no local file found;
- X5 RetailHero: no local competition files found;
- IHDP / ACIC: no local files found.

The bounded search also found unrelated Criteo-named artifacts under another project directory. They are not used or relabeled as the official Criteo uplift v2.1 dataset.

No download was initiated. No raw data was uploaded or added to Git. The available deterministic DeepUplift synthetic fixtures can be used for software and metric validation, but they must be labeled `SYNTHETIC_SCALE_STRESS`, not as myBPmyLife-derived semi-synthetic evidence.

## Stage 0 acceptance

**PARTIAL / BLOCKED_ON_DATA.** The native environment and MPS path pass. The required raw-data readiness gate does not pass because the target files are not present locally. Experiment A can proceed with deterministic synthetic software validation and model-availability audits; real MRT/Hillstrom/Criteo/X5 metrics remain `DATA_NOT_AVAILABLE` until the user supplies lawfully obtained local files.
