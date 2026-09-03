# Benchmark protocol

`run_benchmark()` uses a fixed seed and held-out split, records dataset fingerprint, git SHA, model metadata, nuisance configuration, environment, runtime, warnings, and selection views: `BEST_RANKING`, `BEST_CALIBRATION`, `BEST_POLICY_VALUE`, `FASTEST`, and `RECOMMENDED`.

Evidence bundles contain `config.json`, `dataset_manifest.json`, `diagnostics.json`, `nuisance.json`, `model.json`, `metrics.json`, `policy.json`, `environment.json`, and a hashed `evidence_manifest.json`. Missing ground truth means PEHE is `NOT_AVAILABLE`, never fabricated.
