# DeepUplift API Guide

Normal users should use the five public namespaces:

- `deepuplift.data`: `create_causal_dataset`, public loaders, diagnostics,
  propensity/nuisance, overlap, and preprocessing.
- `deepuplift.models`: `build_model`, `model_info`, model discovery, and the
  unified `EffectPrediction` output.
- `deepuplift.decision`: ranking/calibration metrics, targeting, policy,
  budget, multi/continuous treatment selection, and experiment plans.
- `deepuplift.application`: `run_uplift_pipeline` for the binary end-to-end
  path, including automatic observational nuisance estimation.
- `deepuplift.benchmarks`: benchmark runners and release benchmark commands.

The shared contracts live in `deepuplift.contracts`. `deepuplift.core` is kept
for legacy application services and compatibility; new integrations should not
need to import it directly.

Install the local package with `pip install -e .`. A PyPI release is pending;
`pip install deepuplift` is not yet presented as an available release.
