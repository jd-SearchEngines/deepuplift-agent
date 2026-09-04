from __future__ import annotations

from .runner import run_benchmark


def run_smoke_suite(dataset, *, output_dir=None, seed=42):
    return run_benchmark(dataset, models=["S-Learner", "T-Learner", "DR-Learner"], output_dir=output_dir, seed=seed)


__all__ = ["run_smoke_suite"]
