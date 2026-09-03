# Public Dataset Benchmark

Raw public datasets are not redistributed by this repository. Download them
under their upstream terms and pass local paths to the release runner:

```bash
python scripts/run_release_benchmark.py \
  --real-data \
  --hillstrom-path /path/to/hillstorm_no_indices.csv.gz \
  --criteo-path /path/to/criteo-uplift-v2.1.csv.gz \
  --scale 10000 100000 1000000
```

The runner records dataset rows, assignment/treatment type, provenance,
Qini/AUUC, Uplift@10%, calibration, runtime, memory, model selection, and
release-gate status. PEHE is emitted only for datasets with unit-level true
effects, such as the deterministic synthetic DGP; it is not calculated for
Hillstrom or Criteo.

Hillstrom is a randomized email dataset. Criteo is a randomized uplift dataset
with upstream license/terms. Always inspect the generated report rather than
interpreting a single Qini number as a production guarantee.
