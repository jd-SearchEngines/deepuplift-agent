# Benchmark protocol

`run_benchmark()` uses a fixed seed and held-out split, records dataset fingerprint, git SHA, model metadata, nuisance configuration, environment, runtime, warnings, and selection views: `BEST_RANKING`, `BEST_CALIBRATION`, `BEST_POLICY_VALUE`, `FASTEST`, and `RECOMMENDED`.

Evidence bundles contain `config.json`, `dataset_manifest.json`, `diagnostics.json`, `nuisance.json`, `model.json`, `metrics.json`, `policy.json`, `environment.json`, and a hashed `evidence_manifest.json`. Missing ground truth means PEHE is `NOT_AVAILABLE`, never fabricated.

## Offline policy evaluation and repeated decisions

`evaluate_ope_policy()` reports IPS, SNIPS, DR (when direct-method predictions are supplied), propensity clipping, effective sample size, and bootstrap intervals. For repeated decision data such as MRT, pass `cluster_id` to resample complete users/participants rather than treating decision rows as iid. The JSON result records `diagnostics.bootstrap_unit=cluster`, `cluster_count`, and `bootstrap.bootstrap_unit=cluster`. Without `cluster_id`, bootstrap is explicitly labeled `row` and must not be presented as participant-level uncertainty.
