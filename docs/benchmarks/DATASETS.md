# Public benchmark datasets

The loaders are provenance-only adapters and do not commit raw data. `synthetic_ground_truth` is a deterministic DGP with X, T, Y, and true effect. `load_hillstrom` targets the MineThatData email RCT; the public reference file is available from the upstream URL used by scikit-uplift: `https://hillstorm1.s3.us-east-2.amazonaws.com/hillstorm_no_indices.csv.gz`. `load_criteo` supports a caller-controlled sample or full local file; use the [Criteo AI Lab dataset page](https://ailab.criteo.com/criteo-uplift-prediction-dataset/) and its linked unbiased v2.1 download. The Criteo page states the dataset is under CC BY-NC-SA 4.0 terms, so users must review those terms before redistribution or commercial use. `load_retail` accepts a local Lenta/X5-style file and preserves upstream licensing responsibility.

## Unified registry

`examples/datasets/manifest.json` is the metadata-only registry for the current research program. It records source URL, license/reuse policy, assignment type, ground-truth status, recommended models, evaluation metrics, and a local path hint for IHDP, ACIC 2016, Hillstrom, Criteo, myBPmyLife MRT, and X5 RetailHero. The manifest deliberately contains no raw data path that resolves in the repository.

Build a machine-readable registry without downloading data:

```bash
python scripts/build_dataset_registry.py \
  --output-json /tmp/deepuplift-dataset-registry.json \
  --output-csv /tmp/deepuplift-dataset-registry.csv
```

The command separates schema validation from data availability. A manifest can be `ok` while `availability_status=NOT_AVAILABLE`; that means the metadata contract is valid but the user must prepare the upstream file locally. Raw data remains local-only, and `ground_truth_status=none` or `assignment_probability_only` prevents the benchmark layer from emitting PEHE/CATE error claims.
