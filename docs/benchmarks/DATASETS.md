# Public benchmark datasets

The loaders are provenance-only adapters and do not commit raw data. `synthetic_ground_truth` is a deterministic DGP with X, T, Y, and true effect. `load_hillstrom` targets the MineThatData email RCT. `load_criteo` supports a caller-controlled sample or full local file. `load_retail` accepts a local Lenta/X5-style file and preserves upstream licensing responsibility.
