from __future__ import annotations

from typing import Any

from .datasets import SyntheticContinuousData, synthetic_dose_response
from .report import continuous_benchmark_report
from .runner import run_continuous_dataset


def run_continuous_suite(
    *,
    data: SyntheticContinuousData | None = None,
    rows: int = 600,
    seed: int = 42,
    model_names: list[str] | None = None,
    test_fraction: float = 0.25,
    outcome_value: float = 1.0,
    budget: float | None = None,
    model_options: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    benchmark = data or synthetic_dose_response(rows=rows, seed=seed, outcome_value=outcome_value)
    result = run_continuous_dataset(
        benchmark,
        model_names=model_names,
        test_fraction=test_fraction,
        seed=seed,
        outcome_value=outcome_value,
        budget=budget,
        model_options=model_options,
    )
    result["report"] = continuous_benchmark_report(result)
    return result


__all__ = ["run_continuous_suite"]
