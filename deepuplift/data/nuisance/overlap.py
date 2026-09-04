from __future__ import annotations

from .contracts import NuisanceResult


def overlap_report(result: NuisanceResult) -> dict:
    return {key: value for key, value in result.diagnostics.items() if key in {"min", "p01", "p05", "p25", "p50", "median", "p75", "p95", "p99", "max", "common_support_rate", "extreme_propensity_rate", "warning", "treated_distribution", "control_distribution"}}


__all__ = ["overlap_report"]
