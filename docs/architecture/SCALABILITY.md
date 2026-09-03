# Scalability Boundary

The framework is designed to make scale decisions explicit.

## Current support

- `PandasBackend` is the reference in-memory backend.
- Native S/T/X/DR examples materialize numeric arrays and therefore target small/medium datasets that fit memory.
- Optional `PolarsBackend` supports lazy CSV/Parquet scanning and explicit `to_pandas()` materialization.
- Existing third-party adapters declare their dependency and capability status; missing packages fail with an actionable message.

## What is not claimed

The project does not claim that every estimator is out-of-core, distributed,
or production-ready for 10M+ rows. A lazy data scan alone does not make a
model distributed. Before running a large job, inspect the adapter capability,
memory boundary, feature encoding cost, and artifact volume.

## Extension path

1. Keep ingestion and profiling lazy in a backend implementation.
2. Add chunked feature transformation with deterministic schema alignment.
3. Add explicit materialization guards before estimators that require NumPy.
4. Add benchmark evidence for distributed LightGBM/XGBoost, Ray, Dask or Spark only when a real adapter and reproducible test exist.
