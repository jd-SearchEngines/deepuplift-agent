from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ReleaseConfig:
    seed: int = 42
    synthetic_rows: int = 500
    scale_sizes: tuple[int, ...] = (10_000, 100_000)
    hillstrom_path: str | Path | None = None
    criteo_path: str | Path | None = None
    output_dir: str | Path = "release_runs"
    criteo_sample_rows: int = 100_000
    packaging_python: str | None = None
    packaging_status: str = "EXTERNAL_CHECK"
    ci_status: str = "EXTERNAL_CHECK"
    optional_backend_report_path: str | Path | None = None
    benchmark_models: tuple[str, ...] = ("S-Learner", "T-Learner", "X-Learner", "DR-Learner", "R-Learner", "S-Learner-RF", "DR-Learner-RF", "CausalForestDML")
