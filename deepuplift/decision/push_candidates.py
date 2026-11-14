"""Candidate construction from fitted Push effect predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .optimization import (
    AllocationCandidates,
    AllocationConstraints,
    check_feasibility,
    exact_small_reference,
    greedy_topk,
    solve_coordinate_iterator,
    solve_heap,
    solve_line_search_primal_dual,
    solve_primal_dual,
)


@dataclass(frozen=True)
class PushCandidateBundle:
    """Sparse model-produced Push candidates and their optimizer contract."""

    frame: pd.DataFrame
    candidates: AllocationCandidates
    metadata: dict[str, Any]


def build_push_candidate_bundle(
    predictions: pd.DataFrame,
    *,
    outcome: str,
    model_version: str,
    resource_cost_col: str | None = None,
    default_resource_cost: float = 1.0,
    value_multiplier: float = 1.0,
    eligibility_col: str | None = None,
    evidence_class: str = "SEMI_SYNTHETIC_KNOWN_TRUTH",
) -> PushCandidateBundle:
    """Create optimizer candidates from predicted incremental outcomes.

    Without an explicit monetary mapping, `value_multiplier=1` means the
    objective is in outcome units and `resource_cost` is in abstract Push
    capacity units. The function never labels that as revenue or profit.
    """

    required = {"user_id", "time_slot", "action", "predicted_incremental_effect"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Push predictions are missing required columns: {missing}")
    if not np.isfinite(pd.to_numeric(predictions["predicted_incremental_effect"], errors="coerce")).all():
        raise ValueError("predicted_incremental_effect must be finite.")
    if resource_cost_col is not None and resource_cost_col not in predictions.columns:
        raise ValueError(f"resource_cost_col {resource_cost_col!r} is not present.")
    if eligibility_col is not None and eligibility_col not in predictions.columns:
        raise ValueError(f"eligibility_col {eligibility_col!r} is not present.")
    if default_resource_cost < 0 or not np.isfinite(default_resource_cost):
        raise ValueError("default_resource_cost must be finite and non-negative.")
    if not np.isfinite(value_multiplier):
        raise ValueError("value_multiplier must be finite.")

    frame = predictions.copy().reset_index(drop=True)
    frame["predicted_incremental_outcome"] = pd.to_numeric(frame["predicted_incremental_effect"], errors="coerce")
    frame["predicted_incremental_value"] = frame["predicted_incremental_outcome"] * float(value_multiplier)
    frame["resource_cost"] = (
        pd.to_numeric(frame[resource_cost_col], errors="coerce")
        if resource_cost_col is not None
        else float(default_resource_cost)
    )
    if not np.isfinite(frame["resource_cost"]).all() or (frame["resource_cost"] < 0).any():
        raise ValueError("resource costs must be finite and non-negative.")
    frame["eligible"] = frame[eligibility_col].astype(bool) if eligibility_col is not None else True
    frame["outcome"] = outcome
    frame["model_version"] = model_version
    frame["evidence_class"] = evidence_class
    frame["utility_unit"] = "outcome_units" if value_multiplier == 1.0 else "mapped_value_units"
    frame["resource_cost_unit"] = "push_capacity_units"
    candidates = AllocationCandidates(
        incremental_value=frame["predicted_incremental_value"].to_numpy(dtype="float64"),
        resource_cost=frame["resource_cost"].to_numpy(dtype="float64"),
        user_id=frame["user_id"].to_numpy(),
        time_slot=frame["time_slot"].to_numpy(),
        action=frame["action"].to_numpy(),
        eligible=frame["eligible"].to_numpy(dtype=bool),
    )
    metadata = {
        "source": "fitted_model_predictions",
        "model_version": model_version,
        "outcome": outcome,
        "evidence_class": evidence_class,
        "utility_mapping": "predicted_incremental_effect * value_multiplier",
        "value_multiplier": float(value_multiplier),
        "utility_unit": frame["utility_unit"].iloc[0] if len(frame) else None,
        "resource_cost_unit": "push_capacity_units",
        "raw_business_cost_observed": False,
        "candidate_count": int(len(frame)),
    }
    return PushCandidateBundle(frame=frame, candidates=candidates, metadata=metadata)


def optimize_push_candidate_bundle(
    bundle: PushCandidateBundle,
    constraints: AllocationConstraints,
    *,
    solvers: Iterable[str] = ("greedy", "heap", "primal_dual", "line_search", "coordinate"),
    include_exact: bool = False,
) -> dict[str, Any]:
    """Run existing constrained solvers on the same model-produced candidates."""

    solver_map = {
        "greedy": greedy_topk,
        "heap": solve_heap,
        "primal_dual": solve_primal_dual,
        "line_search": solve_line_search_primal_dual,
        "coordinate": solve_coordinate_iterator,
    }
    results: dict[str, Any] = {}
    if include_exact:
        results["exact"] = exact_small_reference(bundle.candidates, constraints)
    for name in solvers:
        if name not in solver_map:
            raise ValueError(f"Unknown Push optimizer {name!r}; expected {sorted(solver_map)}")
        result = solver_map[name](bundle.candidates, constraints)
        result["candidate_provenance"] = bundle.metadata
        result["independent_feasibility"] = check_feasibility(
            bundle.candidates,
            constraints,
            result.get("selected_indices", []),
        )
        results[name] = result
    feasibility_rows = [
        row.get("independent_feasibility", row.get("feasibility", {}))
        for row in results.values()
    ]
    return {
        "status": "PASS" if all(row.get("feasible") for row in feasibility_rows) else "FAIL",
        "metadata": bundle.metadata,
        "solvers": results,
    }


__all__ = ["PushCandidateBundle", "build_push_candidate_bundle", "optimize_push_candidate_bundle"]
