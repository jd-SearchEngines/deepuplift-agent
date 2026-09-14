"""Offline continuous-treatment benchmark data, metrics, runner, and reports."""

from .datasets import (
    SyntheticContinuousData,
    continuous_dose_cost,
    load_continuous_local,
    load_giks_continuous_benchmark,
    load_ihdp,
    load_news,
    load_tcga,
    synthetic_dose_response,
)
from .metrics import continuous_curve_metrics
from .report import continuous_benchmark_report
from .runner import run_continuous_dataset, run_continuous_model
from .suite import run_continuous_suite

__all__ = [
    "SyntheticContinuousData", "continuous_dose_cost", "load_continuous_local",
    "load_giks_continuous_benchmark", "load_ihdp", "load_news", "load_tcga", "synthetic_dose_response",
    "continuous_curve_metrics", "continuous_benchmark_report",
    "run_continuous_dataset", "run_continuous_model", "run_continuous_suite",
]
