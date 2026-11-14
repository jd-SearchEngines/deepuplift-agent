"""Research constrained allocation solvers for multi-coupled candidates.

The existing ``simulate_budget`` Top-K helper remains unchanged.  These
solvers operate on sparse candidate rows and expose reduced utilities, dual
prices, feasibility checks, and an exact small-instance reference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import multiprocessing as mp
import os
import platform
import resource
import sys
import time
from typing import Any, Iterator, Mapping

import numpy as np


@dataclass(frozen=True)
class AllocationCandidates:
    incremental_value: np.ndarray
    resource_cost: np.ndarray
    user_id: np.ndarray
    time_slot: np.ndarray
    action: np.ndarray
    eligible: np.ndarray | None = None
    _eligible_mask: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        arrays = [self.incremental_value, self.resource_cost, self.user_id, self.time_slot, self.action]
        converted = [np.asarray(value) for value in arrays]
        if any(value.ndim != 1 for value in converted):
            raise ValueError("Candidate arrays must be one-dimensional.")
        lengths = {len(value) for value in converted}
        if len(lengths) != 1:
            raise ValueError("All candidate arrays must have the same length.")
        n = next(iter(lengths))
        if n == 0:
            raise ValueError("At least one candidate is required.")
        values = np.asarray(self.incremental_value, dtype="float64")
        costs = np.asarray(self.resource_cost, dtype="float64")
        if not np.isfinite(values).all() or not np.isfinite(costs).all():
            raise ValueError("Candidate value and cost arrays must be finite.")
        if np.any(costs < 0):
            raise ValueError("resource_cost must be non-negative.")
        if self.eligible is None:
            eligible = np.ones(n, dtype=bool)
        else:
            raw_eligible = np.asarray(self.eligible)
            if raw_eligible.ndim != 1 or len(raw_eligible) != n:
                raise ValueError("eligible must be a one-dimensional mask matching the candidate row count.")
            if raw_eligible.dtype.kind not in {"b", "i", "u"}:
                raise ValueError("eligible must be boolean or integer-valued.")
            eligible = raw_eligible.astype(bool, copy=True)
        object.__setattr__(self, "_eligible_mask", eligible)

    @property
    def size(self) -> int:
        return len(self.incremental_value)

    @property
    def net_value(self) -> np.ndarray:
        return np.asarray(self.incremental_value, dtype="float64") - np.asarray(self.resource_cost, dtype="float64")

    def eligible_mask(self) -> np.ndarray:
        """Return the cached eligibility mask; no per-index allocation occurs."""

        return self._eligible_mask

    def is_eligible(self, index: int) -> bool:
        return bool(self._eligible_mask[int(index)])


@dataclass(frozen=True)
class AllocationConstraints:
    budget: float
    slot_capacity: Mapping[Any, int]
    user_frequency_cap: int = 1
    action_quota: Mapping[Any, int] | None = None
    minimum_interval: int = 0

    def __post_init__(self) -> None:
        if self.budget < 0 or self.user_frequency_cap < 1 or self.minimum_interval < 0:
            raise ValueError("budget must be nonnegative; frequency cap must be positive; interval cannot be negative.")


def _constraint_usage(candidates: AllocationCandidates, selected: list[int]) -> dict[str, Any]:
    ids = np.asarray(selected, dtype="int64")
    if not len(ids):
        return {"cost": 0.0, "slot": {}, "user": {}, "action": {}, "decision_key": set()}
    slot = {}
    user = {}
    action = {}
    keys = set()
    for idx in ids:
        slot[candidates.time_slot[idx]] = slot.get(candidates.time_slot[idx], 0) + 1
        user[candidates.user_id[idx]] = user.get(candidates.user_id[idx], 0) + 1
        action[candidates.action[idx]] = action.get(candidates.action[idx], 0) + 1
        keys.add((candidates.user_id[idx], candidates.time_slot[idx]))
    return {"cost": float(np.sum(np.asarray(candidates.resource_cost)[ids])), "slot": slot, "user": user, "action": action, "decision_key": keys}


def _new_state() -> dict[str, Any]:
    return {"cost": 0.0, "slot": {}, "user": {}, "action": {}, "decision_key": set(), "user_slots": {}}


def _state_can_add(candidates: AllocationCandidates, constraints: AllocationConstraints, state: dict[str, Any], idx: int) -> bool:
    if not candidates.is_eligible(idx):
        return False
    slot = candidates.time_slot[idx]
    user = candidates.user_id[idx]
    action = candidates.action[idx]
    if state["cost"] + float(candidates.resource_cost[idx]) > constraints.budget + 1e-9:
        return False
    if state["slot"].get(slot, 0) + 1 > int(constraints.slot_capacity.get(slot, 0)):
        return False
    if state["user"].get(user, 0) + 1 > constraints.user_frequency_cap:
        return False
    if constraints.action_quota is not None and state["action"].get(action, 0) + 1 > int(constraints.action_quota.get(action, 0)):
        return False
    if (user, slot) in state["decision_key"]:
        return False
    if constraints.minimum_interval > 0 and any(abs(int(previous) - int(slot)) < constraints.minimum_interval for previous in state["user_slots"].get(user, ())):
        return False
    return True


def _state_add(candidates: AllocationCandidates, state: dict[str, Any], idx: int) -> None:
    slot = candidates.time_slot[idx]
    user = candidates.user_id[idx]
    action = candidates.action[idx]
    state["cost"] += float(candidates.resource_cost[idx])
    state["slot"][slot] = state["slot"].get(slot, 0) + 1
    state["user"][user] = state["user"].get(user, 0) + 1
    state["action"][action] = state["action"].get(action, 0) + 1
    state["decision_key"].add((user, slot))
    state["user_slots"].setdefault(user, []).append(slot)


def _can_add(candidates: AllocationCandidates, constraints: AllocationConstraints, selected: list[int], idx: int) -> bool:
    if not candidates.is_eligible(idx):
        return False
    usage = _constraint_usage(candidates, selected)
    if usage["cost"] + float(candidates.resource_cost[idx]) > constraints.budget + 1e-9:
        return False
    slot = candidates.time_slot[idx]
    if usage["slot"].get(slot, 0) + 1 > int(constraints.slot_capacity.get(slot, 0)):
        return False
    user = candidates.user_id[idx]
    if usage["user"].get(user, 0) + 1 > constraints.user_frequency_cap:
        return False
    action = candidates.action[idx]
    if constraints.action_quota is not None and usage["action"].get(action, 0) + 1 > int(constraints.action_quota.get(action, 0)):
        return False
    if (user, slot) in usage["decision_key"]:
        return False
    if constraints.minimum_interval > 0:
        for chosen in selected:
            if candidates.user_id[chosen] == user and abs(int(candidates.time_slot[chosen]) - int(slot)) < constraints.minimum_interval:
                return False
    return True


def check_feasibility(candidates: AllocationCandidates, constraints: AllocationConstraints, selected: list[int]) -> dict[str, Any]:
    """Independently check a selected set in O(K log K), not pairwise O(K²)."""

    selected = [int(index) for index in selected]
    invalid_indices = [index for index in selected if index < 0 or index >= candidates.size]
    if invalid_indices:
        raise ValueError(f"Selected candidate indices are out of range: {invalid_indices[:5]}")
    usage = _constraint_usage(candidates, selected)
    violations = [max(0.0, usage["cost"] - constraints.budget)]
    ineligible_count = sum(not candidates.is_eligible(index) for index in selected)
    violations.append(float(ineligible_count))
    for slot, count in usage["slot"].items():
        violations.append(max(0.0, float(count - int(constraints.slot_capacity.get(slot, 0)))))
    for user, count in usage["user"].items():
        violations.append(max(0.0, float(count - constraints.user_frequency_cap)))
    if constraints.action_quota is not None:
        for action, count in usage["action"].items():
            violations.append(max(0.0, float(count - int(constraints.action_quota.get(action, 0)))))
    duplicates = len(selected) - len(usage["decision_key"])
    violations.append(float(max(0, duplicates)))
    user_slots: dict[Any, list[int]] = {}
    for index in selected:
        user_slots.setdefault(candidates.user_id[index], []).append(int(candidates.time_slot[index]))
    interval_violations = 0
    if constraints.minimum_interval > 0:
        for slots in user_slots.values():
            ordered = sorted(slots)
            interval_violations += sum(
                right - left < constraints.minimum_interval
                for left, right in zip(ordered, ordered[1:])
            )
    violations.append(float(interval_violations))
    return {
        "feasible": max(violations, default=0.0) <= 1e-9,
        "max_constraint_violation": float(max(violations, default=0.0)),
        "violations": violations,
        "usage": usage,
        "invalid_index_count": len(invalid_indices),
        "ineligible_selected_count": int(ineligible_count),
        "minimum_interval_violation_count": int(interval_violations),
    }


def _result(
    name: str,
    candidates: AllocationCandidates,
    constraints: AllocationConstraints,
    selected: list[int],
    *,
    iterations: int,
    dual_prices: dict[str, Any],
    stopping_reason: str,
    relaxed_residual: float,
    started: float,
    trace: list[dict[str, Any]] | None = None,
    diagnostics: dict[str, Any] | None = None,
    incumbent_fallback_used: bool = False,
    incumbent_objective: float | None = None,
    net_value: np.ndarray | None = None,
) -> dict[str, Any]:
    feasibility = check_feasibility(candidates, constraints, selected)
    usage = feasibility["usage"]
    diagnostics = diagnostics or {}
    net_value = candidates.net_value if net_value is None else net_value
    convergence_solver = name in {"primal_dual", "line_search_primal_dual", "coordinate_iterator"}
    convergence_metrics = {
        "primal_residual": float(diagnostics.get("primal_violation", feasibility["max_constraint_violation"])),
        "dual_step_norm": float(diagnostics.get("dual_step_norm", float("inf"))),
        "dual_feasibility_residual": float(diagnostics.get("dual_feasibility_residual", 0.0)),
        "complementarity": float(diagnostics.get("complementarity", 0.0)),
    }
    convergence_criteria_met = all(value <= 1e-6 for value in convergence_metrics.values())
    convergence_status = (
        "NOT_APPLICABLE"
        if not convergence_solver
        else "PASS" if convergence_criteria_met and stopping_reason in {"primal_dual_tolerance", "line_search_tolerance", "coordinate_tolerance"} else "NOT_CONVERGED"
    )
    return {
        "solver": name,
        "status": "PASS" if feasibility["feasible"] else "INFEASIBLE",
        "selected_indices": [int(i) for i in selected],
        "selected_count": int(len(selected)),
        "incremental_value": float(np.sum(np.asarray(candidates.incremental_value)[selected])) if selected else 0.0,
        "total_cost": float(usage["cost"]),
        "objective_net_value": float(np.sum(net_value[selected])) if selected else 0.0,
        "feasibility": {key: value for key, value in feasibility.items() if key != "usage"},
        "feasibility_status": "PASS" if feasibility["feasible"] else "FAIL",
        "quality_status": "UNASSESSED",
        "quality_reference_type": None,
        "quality_reference_objective": None,
        "relative_optimality_gap": None,
        "relative_heuristic_gap": None,
        "usage": {key: value for key, value in usage.items() if key != "decision_key"},
        "dual_prices": dual_prices,
        "primal_residual": float(feasibility["max_constraint_violation"]),
        "primal_constraint_violation": float(feasibility["max_constraint_violation"]),
        "relaxed_primal_residual": float(diagnostics.get("primal_violation", feasibility["max_constraint_violation"])),
        "relaxed_primal_constraint_violation": float(diagnostics.get("primal_violation", feasibility["max_constraint_violation"])),
        "signed_residuals": diagnostics.get("signed_residuals", {}),
        "dual_step_norm": float(diagnostics.get("dual_step_norm", float("inf"))),
        "dual_feasibility": diagnostics.get("dual_feasibility", {"nonnegative": True, "residual": 0.0}),
        "dual_feasibility_residual": float(diagnostics.get("dual_feasibility_residual", 0.0)),
        "dual_residual": float(diagnostics.get("dual_feasibility_residual", 0.0)),
        "dual_residual_semantics": "dual feasibility violation; price-change norm is reported separately as dual_step_norm",
        "complementarity": float(diagnostics.get("complementarity", 0.0)),
        "primal_dual_gap": diagnostics.get("primal_dual_gap"),
        "primal_dual_gap_valid": bool(diagnostics.get("primal_dual_gap_valid", False)),
        "convergence_metrics": convergence_metrics,
        "convergence_status": convergence_status,
        "stopping_diagnostics": diagnostics,
        "stopping_reason": stopping_reason,
        "iterations": int(iterations),
        "runtime_seconds": float(time.perf_counter() - started),
        "candidate_count": int(candidates.size),
        "incumbent_fallback_used": bool(incumbent_fallback_used),
        "incumbent_objective": incumbent_objective,
        "trace": trace or [],
    }


def _greedy_selection(candidates: AllocationCandidates, constraints: AllocationConstraints, net_value: np.ndarray | None = None) -> list[int]:
    net_value = candidates.net_value if net_value is None else net_value
    order = np.argsort(-net_value, kind="mergesort")
    selected = []
    state = _new_state()
    for idx in order:
        idx = int(idx)
        if net_value[idx] <= 0:
            continue
        if _state_can_add(candidates, constraints, state, idx):
            selected.append(idx)
            _state_add(candidates, state, idx)
    return selected


def greedy_topk(candidates: AllocationCandidates, constraints: AllocationConstraints) -> dict[str, Any]:
    started = time.perf_counter()
    net_value = candidates.net_value
    selected = _greedy_selection(candidates, constraints, net_value)
    return _result("greedy_topk", candidates, constraints, selected, iterations=1, dual_prices={}, stopping_reason="sorted_net_value", relaxed_residual=0.0, started=started, net_value=net_value)


def _reduced_utilities(candidates: AllocationCandidates, prices: dict[str, Any], net_value: np.ndarray | None = None, resource_cost: np.ndarray | None = None) -> np.ndarray:
    slot_price = prices.get("slot", {})
    action_price = prices.get("action", {})
    net_value = candidates.net_value if net_value is None else net_value
    resource_cost = np.asarray(candidates.resource_cost, dtype="float64") if resource_cost is None else resource_cost
    return net_value - prices.get("budget", 0.0) * resource_cost - np.fromiter((slot_price.get(value, 0.0) for value in candidates.time_slot), dtype="float64", count=candidates.size) - np.fromiter((action_price.get(value, 0.0) for value in candidates.action), dtype="float64", count=candidates.size)


def _relaxed_select(candidates: AllocationCandidates, constraints: AllocationConstraints, utilities: np.ndarray) -> list[int]:
    selected: list[int] = []
    # Relax resource/slot/action capacities for dual pricing, but preserve
    # eligibility, one action per user-time, frequency, and interval guards.
    user_counts: dict[Any, int] = {}
    keys: set[tuple[Any, Any]] = set()
    user_slots: dict[Any, list[Any]] = {}
    for idx in np.argsort(-utilities, kind="mergesort"):
        idx = int(idx)
        if not candidates.is_eligible(idx) or utilities[idx] <= 0:
            continue
        user = candidates.user_id[idx]
        slot = candidates.time_slot[idx]
        if (user, slot) in keys or user_counts.get(user, 0) >= constraints.user_frequency_cap:
            continue
        if constraints.minimum_interval > 0 and any(abs(int(previous) - int(slot)) < constraints.minimum_interval for previous in user_slots.get(user, ())):
            continue
        selected.append(idx)
        keys.add((user, slot))
        user_counts[user] = user_counts.get(user, 0) + 1
        user_slots.setdefault(user, []).append(slot)
    return selected


def _dual_violations(
    candidates: AllocationCandidates,
    constraints: AllocationConstraints,
    selected: list[int],
    prices: dict[str, Any] | None = None,
) -> tuple[float, dict[str, Any]]:
    """Return signed normalized residuals and primal/dual diagnostics."""

    usage = _constraint_usage(candidates, selected)
    budget_residual = (usage["cost"] - constraints.budget) / max(constraints.budget, 1.0)
    slot_keys = set(constraints.slot_capacity) | set(usage["slot"])
    slot_residual = {
        slot: (usage["slot"].get(slot, 0) - int(constraints.slot_capacity.get(slot, 0))) / max(int(constraints.slot_capacity.get(slot, 0)), 1)
        for slot in sorted(slot_keys, key=str)
    }
    action_residual: dict[Any, float] = {}
    if constraints.action_quota is not None:
        action_keys = set(constraints.action_quota) | set(usage["action"])
        action_residual = {
            action: (usage["action"].get(action, 0) - int(constraints.action_quota.get(action, 0))) / max(int(constraints.action_quota.get(action, 0)), 1)
            for action in sorted(action_keys, key=str)
        }
    signed = {"budget": float(budget_residual), "slot": slot_residual, "action": action_residual}
    all_residuals = [budget_residual, *slot_residual.values(), *action_residual.values()]
    primal_violation = float(np.sqrt(sum(max(value, 0.0) ** 2 for value in all_residuals)))
    complementarity = 0.0
    dual_values = []
    if prices is not None:
        dual_values = [float(prices.get("budget", 0.0)), *[float(value) for value in prices.get("slot", {}).values()], *[float(value) for value in prices.get("action", {}).values()]]
        complementarity = float(
            prices.get("budget", 0.0) * max(-budget_residual, 0.0)
            + sum(prices.get("slot", {}).get(slot, 0.0) * max(-value, 0.0) for slot, value in slot_residual.items())
            + sum(prices.get("action", {}).get(action, 0.0) * max(-value, 0.0) for action, value in action_residual.items())
        )
    dual_feasibility_residual = float(np.sqrt(sum(max(-value, 0.0) ** 2 for value in dual_values))) if dual_values else 0.0
    diagnostics = {
        "usage": usage,
        "signed_residuals": signed,
        "primal_violation": primal_violation,
        "complementarity": complementarity,
        "dual_feasibility": {"nonnegative": dual_feasibility_residual <= 1e-12, "residual": dual_feasibility_residual},
        "dual_feasibility_residual": dual_feasibility_residual,
    }
    return primal_violation, diagnostics


def _initial_prices(constraints: AllocationConstraints) -> dict[str, Any]:
    return {
        "budget": 0.0,
        "slot": {slot: 0.0 for slot in constraints.slot_capacity},
        "action": {action: 0.0 for action in (constraints.action_quota or {})},
    }


def _projected_dual_update(prices: dict[str, Any], signed_residuals: dict[str, Any], step_size: float) -> dict[str, Any]:
    updated = {
        "budget": max(0.0, prices.get("budget", 0.0) + step_size * signed_residuals.get("budget", 0.0)),
        "slot": dict(prices.get("slot", {})),
        "action": dict(prices.get("action", {})),
    }
    for key, residual in signed_residuals.get("slot", {}).items():
        updated["slot"][key] = max(0.0, updated["slot"].get(key, 0.0) + step_size * residual)
    for key, residual in signed_residuals.get("action", {}).items():
        updated["action"][key] = max(0.0, updated["action"].get(key, 0.0) + step_size * residual)
    return updated


def _dual_price_delta(left: dict[str, Any], right: dict[str, Any]) -> float:
    values = [(left.get("budget", 0.0) - right.get("budget", 0.0)) ** 2]
    keys = set(left.get("slot", {})) | set(right.get("slot", {}))
    values.extend((left.get("slot", {}).get(key, 0.0) - right.get("slot", {}).get(key, 0.0)) ** 2 for key in keys)
    keys = set(left.get("action", {})) | set(right.get("action", {}))
    values.extend((left.get("action", {}).get(key, 0.0) - right.get("action", {}).get(key, 0.0)) ** 2 for key in keys)
    return float(np.sqrt(sum(values)))


def _dual_merit(diagnostics: dict[str, Any], dual_step: float = 0.0) -> float:
    return float(diagnostics.get("primal_violation", 0.0) + 0.1 * diagnostics.get("complementarity", 0.0) + 0.01 * dual_step)


def _best_feasible_incumbent(candidates: AllocationCandidates, constraints: AllocationConstraints, selected: list[int], incumbent: list[int], net_value: np.ndarray | None = None) -> tuple[list[int], bool]:
    net_value = candidates.net_value if net_value is None else net_value
    selected_value = float(np.sum(net_value[selected])) if selected else 0.0
    incumbent_value = float(np.sum(net_value[incumbent])) if incumbent else 0.0
    return (incumbent, True) if incumbent_value > selected_value else (selected, False)


def _copy_prices(prices: dict[str, Any]) -> dict[str, Any]:
    return {
        "budget": float(prices.get("budget", 0.0)),
        "slot": dict(prices.get("slot", {})),
        "action": dict(prices.get("action", {})),
    }


def _solver_diagnostics(
    candidates: AllocationCandidates,
    constraints: AllocationConstraints,
    selected: list[int],
    prices: dict[str, Any],
    dual_step: float,
    *,
    accepted_steps: int = 0,
    stopping_reason: str = "",
) -> tuple[float, dict[str, Any]]:
    primal, diagnostics = _dual_violations(candidates, constraints, selected, prices)
    diagnostics = dict(diagnostics)
    diagnostics.update(
        {
            "dual_step_norm": float(dual_step),
            "accepted_steps": int(accepted_steps),
            "stopping_reason": stopping_reason,
        }
    )
    return primal, diagnostics


def _repair(candidates: AllocationCandidates, constraints: AllocationConstraints, utilities: np.ndarray) -> list[int]:
    selected: list[int] = []
    state = _new_state()
    for idx in np.argsort(-utilities, kind="mergesort"):
        idx = int(idx)
        if utilities[idx] <= 0:
            continue
        if _state_can_add(candidates, constraints, state, idx):
            selected.append(idx)
            _state_add(candidates, state, idx)
    return selected


def solve_primal_dual(candidates: AllocationCandidates, constraints: AllocationConstraints, *, max_iterations: int = 40, step_size: float = 0.8) -> dict[str, Any]:
    started = time.perf_counter()
    net_value = candidates.net_value
    resource_cost = np.asarray(candidates.resource_cost, dtype="float64")
    prices = _initial_prices(constraints)
    incumbent = _greedy_selection(candidates, constraints, net_value)
    selected = list(incumbent)
    trace: list[dict[str, Any]] = []
    dual_step = float("inf")
    relaxed_residual = float("inf")
    stopping_reason = "max_iterations"
    fallback_used = False
    for iteration in range(1, max_iterations + 1):
        utilities = _reduced_utilities(candidates, prices, net_value, resource_cost)
        relaxed = _relaxed_select(candidates, constraints, utilities)
        _, before = _dual_violations(candidates, constraints, relaxed, prices)
        rate = step_size / np.sqrt(iteration)
        updated = _projected_dual_update(prices, before["signed_residuals"], rate)
        dual_step = _dual_price_delta(updated, prices)
        updated_relaxed = _relaxed_select(candidates, constraints, _reduced_utilities(candidates, updated, net_value, resource_cost))
        relaxed_residual, after = _dual_violations(candidates, constraints, updated_relaxed, updated)
        repaired = _repair(candidates, constraints, _reduced_utilities(candidates, updated, net_value, resource_cost))
        selected, used_incumbent = _best_feasible_incumbent(candidates, constraints, repaired, incumbent, net_value)
        fallback_used = fallback_used or used_incumbent
        if not used_incumbent and float(np.sum(net_value[repaired])) > float(np.sum(net_value[incumbent])):
            incumbent = list(repaired)
        merit = _dual_merit(after, dual_step)
        trace.append(
            {
                "iteration": iteration,
                "relaxed_residual": float(relaxed_residual),
                "primal_violation": float(after["primal_violation"]),
                "signed_residuals": after["signed_residuals"],
                "dual_step": float(dual_step),
                "complementarity": float(after["complementarity"]),
                "merit": merit,
                "selected": len(updated_relaxed),
                "accepted_step": True,
            }
        )
        prices = updated
        if after["primal_violation"] <= 1e-6 and dual_step <= 1e-6 and after["complementarity"] <= 1e-6:
            stopping_reason = "primal_dual_tolerance"
            break
    primal, diagnostics = _solver_diagnostics(candidates, constraints, updated_relaxed, prices, dual_step, accepted_steps=len(trace), stopping_reason=stopping_reason)
    return _result("primal_dual", candidates, constraints, selected, iterations=iteration, dual_prices=prices, stopping_reason=stopping_reason, relaxed_residual=relaxed_residual, started=started, trace=trace, diagnostics=diagnostics, incumbent_fallback_used=fallback_used, incumbent_objective=float(np.sum(net_value[incumbent])) if incumbent else 0.0, net_value=net_value)


def solve_line_search_primal_dual(candidates: AllocationCandidates, constraints: AllocationConstraints, *, max_iterations: int = 40) -> dict[str, Any]:
    started = time.perf_counter()
    net_value = candidates.net_value
    resource_cost = np.asarray(candidates.resource_cost, dtype="float64")
    prices = _initial_prices(constraints)
    incumbent = _greedy_selection(candidates, constraints, net_value)
    trace: list[dict[str, Any]] = []
    residual = float("inf")
    dual_step = float("inf")
    stopping_reason = "max_iterations"
    for iteration in range(1, max_iterations + 1):
        current_selection = _relaxed_select(candidates, constraints, _reduced_utilities(candidates, prices, net_value, resource_cost))
        current, current_diagnostics = _dual_violations(candidates, constraints, current_selection, prices)
        current_merit = _dual_merit(current_diagnostics, 0.0)
        best = (current_merit, _copy_prices(prices), current_selection, current_diagnostics, 0.0, False)
        for multiplier in (0.25, 0.5, 1.0, 2.0):
            trial = _projected_dual_update(prices, current_diagnostics["signed_residuals"], multiplier / np.sqrt(iteration))
            trial_selection = _relaxed_select(candidates, constraints, _reduced_utilities(candidates, trial, net_value, resource_cost))
            _, trial_diagnostics = _dual_violations(candidates, constraints, trial_selection, trial)
            trial_step = _dual_price_delta(trial, prices)
            trial_merit = _dual_merit(trial_diagnostics, trial_step)
            if trial_merit < best[0] - 1e-12:
                best = (trial_merit, trial, trial_selection, trial_diagnostics, trial_step, True)
        merit, next_prices, relaxed, diagnostics, dual_step, accepted = best
        prices = next_prices
        residual = float(diagnostics["primal_violation"])
        repaired = _repair(candidates, constraints, _reduced_utilities(candidates, prices, net_value, resource_cost))
        if float(np.sum(net_value[repaired])) > float(np.sum(net_value[incumbent])):
            incumbent = list(repaired)
        trace.append({"iteration": iteration, "line_search_residual": residual, "primal_violation": residual, "dual_step": dual_step, "complementarity": diagnostics["complementarity"], "merit": merit, "accepted_step": accepted})
        if residual <= 1e-6 and dual_step <= 1e-6 and diagnostics["complementarity"] <= 1e-6:
            stopping_reason = "line_search_tolerance"
            break
        if not accepted:
            stopping_reason = "line_search_no_improvement"
            break
    selected = _repair(candidates, constraints, _reduced_utilities(candidates, prices, net_value, resource_cost))
    selected, fallback = _best_feasible_incumbent(candidates, constraints, selected, incumbent, net_value)
    primal, diagnostics = _solver_diagnostics(candidates, constraints, relaxed, prices, dual_step, accepted_steps=sum(bool(row["accepted_step"]) for row in trace), stopping_reason=stopping_reason)
    return _result("line_search_primal_dual", candidates, constraints, selected, iterations=iteration, dual_prices=prices, stopping_reason=stopping_reason, relaxed_residual=residual, started=started, trace=trace, diagnostics=diagnostics, incumbent_fallback_used=fallback, incumbent_objective=float(np.sum(net_value[incumbent])) if incumbent else 0.0, net_value=net_value)


def solve_heap(candidates: AllocationCandidates, constraints: AllocationConstraints) -> dict[str, Any]:
    started = time.perf_counter()
    net_value = candidates.net_value
    heap = [(-float(value), int(idx)) for idx, value in enumerate(net_value)]
    heapq.heapify(heap)
    selected: list[int] = []
    state = _new_state()
    while heap:
        negative_value, idx = heapq.heappop(heap)
        if -negative_value <= 0:
            continue
        if _state_can_add(candidates, constraints, state, idx):
            selected.append(idx)
            _state_add(candidates, state, idx)
    return _result("heap", candidates, constraints, selected, iterations=1, dual_prices={}, stopping_reason="heap_exhausted_static_net_order", relaxed_residual=0.0, started=started, net_value=net_value)


def coordinate_iterator(candidates: AllocationCandidates) -> Iterator[tuple[str, Any]]:
    yield ("budget", None)
    for value in sorted(set(candidates.time_slot.tolist()), key=str):
        yield ("slot", value)
    for value in sorted(set(candidates.action.tolist()), key=str):
        yield ("action", value)


def solve_coordinate_iterator(candidates: AllocationCandidates, constraints: AllocationConstraints, *, max_iterations: int = 40) -> dict[str, Any]:
    started = time.perf_counter()
    net_value = candidates.net_value
    resource_cost = np.asarray(candidates.resource_cost, dtype="float64")
    prices = _initial_prices(constraints)
    incumbent = _greedy_selection(candidates, constraints, net_value)
    residual = float("inf")
    dual_step = float("inf")
    trace: list[dict[str, Any]] = []
    stopping_reason = "max_iterations"
    for iteration in range(1, max_iterations + 1):
        cycle_max_step = 0.0
        for coordinate, key in coordinate_iterator(candidates):
            selected = _relaxed_select(candidates, constraints, _reduced_utilities(candidates, prices, net_value, resource_cost))
            _, before = _dual_violations(candidates, constraints, selected, prices)
            trial = _copy_prices(prices)
            step = 0.5 / np.sqrt(iteration)
            if coordinate == "budget":
                trial["budget"] = max(0.0, trial["budget"] + step * before["signed_residuals"]["budget"])
            elif coordinate == "slot":
                trial["slot"][key] = max(0.0, trial["slot"].get(key, 0.0) + step * before["signed_residuals"]["slot"].get(key, 0.0))
            else:
                trial["action"][key] = max(0.0, trial["action"].get(key, 0.0) + step * before["signed_residuals"]["action"].get(key, 0.0))
            next_selected = _relaxed_select(candidates, constraints, _reduced_utilities(candidates, trial, net_value, resource_cost))
            _, after = _dual_violations(candidates, constraints, next_selected, trial)
            dual_step = _dual_price_delta(trial, prices)
            cycle_max_step = max(cycle_max_step, dual_step)
            prices = trial
            repaired = _repair(candidates, constraints, _reduced_utilities(candidates, prices, net_value, resource_cost))
            if float(np.sum(net_value[repaired])) > float(np.sum(net_value[incumbent])):
                incumbent = list(repaired)
            trace.append({"iteration": iteration, "coordinate": coordinate, "key": key, "before_primal_violation": before["primal_violation"], "after_primal_violation": after["primal_violation"], "selected_before": len(selected), "selected_after": len(next_selected), "dual_step": dual_step, "recomputed": True})
            residual = float(after["primal_violation"])
        if residual <= 1e-6 and dual_step <= 1e-6:
            stopping_reason = "coordinate_tolerance"
            break
        if cycle_max_step <= 1e-6:
            stopping_reason = "coordinate_fixed_point"
            break
    selected = _repair(candidates, constraints, _reduced_utilities(candidates, prices, net_value, resource_cost))
    selected, fallback = _best_feasible_incumbent(candidates, constraints, selected, incumbent, net_value)
    relaxed = _relaxed_select(candidates, constraints, _reduced_utilities(candidates, prices, net_value, resource_cost))
    residual, diagnostics = _solver_diagnostics(candidates, constraints, relaxed, prices, dual_step, accepted_steps=len(trace), stopping_reason=stopping_reason)
    return _result("coordinate_iterator", candidates, constraints, selected, iterations=iteration, dual_prices=prices, stopping_reason=stopping_reason, relaxed_residual=residual, started=started, trace=trace, diagnostics=diagnostics, incumbent_fallback_used=fallback, incumbent_objective=float(np.sum(net_value[incumbent])) if incumbent else 0.0, net_value=net_value)


def _exact_enumeration(candidates: AllocationCandidates, constraints: AllocationConstraints) -> dict[str, Any]:
    started = time.perf_counter()
    net_value = candidates.net_value
    best: list[int] = []
    best_value = 0.0
    for mask in range(1 << candidates.size):
        selected = [idx for idx in range(candidates.size) if mask & (1 << idx)]
        feasibility = check_feasibility(candidates, constraints, selected)
        if feasibility["feasible"]:
            value = float(np.sum(net_value[selected])) if selected else 0.0
            if value > best_value:
                best_value, best = value, selected
    row = _result("exact_small_reference", candidates, constraints, best, iterations=1 << candidates.size, dual_prices={}, stopping_reason="enumeration_complete", relaxed_residual=0.0, started=started, net_value=net_value)
    row["reference_method"] = "exhaustive_enumeration"
    return row


def _milp_small_reference(candidates: AllocationCandidates, constraints: AllocationConstraints, *, max_candidates: int) -> dict[str, Any] | None:
    if candidates.size > max_candidates:
        return None
    try:
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import coo_matrix
    except Exception:
        return None
    n = candidates.size
    net_value = candidates.net_value
    row_indices: list[int] = []
    col_indices: list[int] = []
    data: list[float] = []
    upper: list[float] = []

    def add_row(indices: list[int], limit: float, values: list[float] | None = None) -> None:
        row = len(upper)
        row_indices.extend([row] * len(indices))
        col_indices.extend(indices)
        data.extend([1.0] * len(indices) if values is None else values)
        upper.append(float(limit))

    add_row(list(range(n)), constraints.budget, [float(value) for value in candidates.resource_cost])
    for slot in sorted(set(candidates.time_slot.tolist()), key=str):
        indices = [idx for idx in range(n) if candidates.time_slot[idx] == slot]
        add_row(indices, constraints.slot_capacity.get(slot, 0))
    for user in sorted(set(candidates.user_id.tolist()), key=str):
        indices = [idx for idx in range(n) if candidates.user_id[idx] == user]
        add_row(indices, constraints.user_frequency_cap)
    if constraints.action_quota is not None:
        for action in sorted(set(candidates.action.tolist()), key=str):
            indices = [idx for idx in range(n) if candidates.action[idx] == action]
            add_row(indices, constraints.action_quota.get(action, 0))
    for left in range(n):
        for right in range(left + 1, n):
            same_key = candidates.user_id[left] == candidates.user_id[right] and candidates.time_slot[left] == candidates.time_slot[right]
            close = constraints.minimum_interval > 0 and candidates.user_id[left] == candidates.user_id[right] and abs(int(candidates.time_slot[left]) - int(candidates.time_slot[right])) < constraints.minimum_interval
            if same_key or close:
                row = len(upper)
                row_indices.extend([row, row])
                col_indices.extend([left, right])
                data.extend([1.0, 1.0])
                upper.append(1.0)
    matrix = coo_matrix((data, (row_indices, col_indices)), shape=(len(upper), n)).tocsr()
    upper_bounds = np.ones(n, dtype="float64")
    upper_bounds[~candidates.eligible_mask()] = 0.0
    started = time.perf_counter()
    result = milp(c=-net_value, integrality=np.ones(n), bounds=Bounds(np.zeros(n), upper_bounds), constraints=LinearConstraint(matrix, -np.inf, np.asarray(upper)), options={"disp": False})
    if not result.success or result.x is None:
        return None
    selected = [idx for idx, value in enumerate(result.x) if value >= 0.5]
    if not check_feasibility(candidates, constraints, selected)["feasible"]:
        return None
    row = _result("exact_small_reference", candidates, constraints, selected, iterations=int(getattr(result, "mip_node_count", 0) or 0), dual_prices={}, stopping_reason="milp_optimal", relaxed_residual=0.0, started=started, net_value=net_value)
    row["reference_method"] = "scipy_milp"
    return row


def exact_small_reference(candidates: AllocationCandidates, constraints: AllocationConstraints, *, max_candidates: int = 20) -> dict[str, Any]:
    if candidates.size > max_candidates:
        return {"solver": "exact_small_reference", "status": "NOT_RUN", "reason": f"exact reference is limited to <= {max_candidates} candidates", "candidate_count": candidates.size}
    return _milp_small_reference(candidates, constraints, max_candidates=max_candidates) or _exact_enumeration(candidates, constraints)


def _synthetic_candidates(size: int, seed: int) -> tuple[AllocationCandidates, AllocationConstraints]:
    rng = np.random.default_rng(seed)
    users = np.arange(size) // 4
    slots = rng.integers(0, 24, size=size)
    actions = rng.choice(np.array(["activity", "diet", "generic"]), size=size)
    cost = rng.choice(np.array([1.0, 1.5, 2.0]), size=size)
    value = rng.uniform(0.2, 4.0, size=size) + (actions == "activity") * 0.2
    candidates = AllocationCandidates(value, cost, users, slots, actions)
    constraints = AllocationConstraints(budget=max(10.0, size * 0.35), slot_capacity={slot: max(1, size // 80) for slot in range(24)}, user_frequency_cap=2, action_quota={"activity": max(1, size // 20), "diet": max(1, size // 20), "generic": max(1, size // 20)}, minimum_interval=2)
    return candidates, constraints


def _peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024.0 * 1024.0) if sys.platform == "darwin" else value / 1024.0


def _available_memory_mb() -> float | None:
    try:
        import psutil

        return float(psutil.virtual_memory().available) / (1024.0 * 1024.0)
    except Exception:
        try:
            pages = os.sysconf("SC_AVPHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return float(pages * page_size) / (1024.0 * 1024.0)
        except (AttributeError, OSError, ValueError):
            return None


def _resource_guard(size: int, *, memory_budget_mb: float | None = None, allow_oversize: bool = False) -> dict[str, Any]:
    available = _available_memory_mb()
    estimated = 256.0 + float(size) * 0.0005
    budget = float(memory_budget_mb) if memory_budget_mb is not None else (available * 0.6 if available is not None else None)
    if size > 1_000_000 and not allow_oversize:
        return {"status": "RESOURCE_GUARD_SKIPPED", "reason": "sizes above 1M require explicit allow_oversize=True", "estimated_peak_mb": estimated, "available_mb": available, "budget_mb": budget}
    if budget is not None and estimated > budget:
        return {"status": "RESOURCE_GUARD_SKIPPED", "reason": "estimated peak memory exceeds the configured guard budget", "estimated_peak_mb": estimated, "available_mb": available, "budget_mb": budget}
    return {"status": "PASS", "reason": "resource estimate within guard budget", "estimated_peak_mb": estimated, "available_mb": available, "budget_mb": budget}


def _compact_scale_result(row: dict[str, Any]) -> dict[str, Any]:
    compact = dict(row)
    compact["selected_indices"] = []
    compact["selected_indices_omitted"] = True
    return compact


_SOLVER_FUNCTIONS = {
    "greedy_topk": greedy_topk,
    "primal_dual": solve_primal_dual,
    "line_search_primal_dual": solve_line_search_primal_dual,
    "heap": solve_heap,
    "coordinate_iterator": solve_coordinate_iterator,
}


def _scale_iteration_budgets(size: int) -> dict[str, int]:
    return {
        "primal_dual": 20 if size >= 100_000 else 40,
        "line_search_primal_dual": 20 if size >= 100_000 else 40,
        "coordinate_iterator": 1 if size >= 500_000 else (2 if size >= 100_000 else 40),
    }


def _scale_solver_call(name: str, candidates: AllocationCandidates, constraints: AllocationConstraints, size: int) -> dict[str, Any]:
    budgets = _scale_iteration_budgets(size)
    if name == "primal_dual":
        return solve_primal_dual(candidates, constraints, max_iterations=budgets["primal_dual"])
    if name == "line_search_primal_dual":
        return solve_line_search_primal_dual(candidates, constraints, max_iterations=budgets["line_search_primal_dual"])
    if name == "coordinate_iterator":
        return solve_coordinate_iterator(candidates, constraints, max_iterations=budgets["coordinate_iterator"])
    return _SOLVER_FUNCTIONS[name](candidates, constraints)


def _isolated_solver_worker(name: str, candidates: AllocationCandidates, constraints: AllocationConstraints, size: int, connection: Any) -> None:
    try:
        row = _scale_solver_call(name, candidates, constraints, size)
        row["process_peak_rss_mb"] = _peak_rss_mb()
        connection.send(_compact_scale_result(row))
    except Exception as exc:  # pragma: no cover - exercised by process failure, not solver logic
        connection.send({"solver": name, "status": "FAIL", "error": f"{type(exc).__name__}: {exc}"})
    finally:
        connection.close()


def _run_isolated_solver(name: str, candidates: AllocationCandidates, constraints: AllocationConstraints, size: int) -> dict[str, Any]:
    context = mp.get_context("fork" if "fork" in mp.get_all_start_methods() else "spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_isolated_solver_worker, args=(name, candidates, constraints, size, child))
    process.start()
    child.close()
    row = parent.recv() if parent.poll(3600) else {"solver": name, "status": "FAIL", "error": "isolated solver timed out"}
    process.join(timeout=5)
    if process.is_alive():
        process.terminate()
        process.join()
    parent.close()
    return row


def _benchmark_row(row: dict[str, Any], best_objective: float | None) -> dict[str, Any]:
    row = _compact_scale_result(row)
    runtime = float(row.get("runtime_seconds", 0.0))
    row["candidates_per_second"] = float(row.get("candidate_count", 0) / runtime) if runtime > 0 else None
    if best_objective is not None and row.get("status") == "PASS":
        row["relative_heuristic_gap"] = float(max(0.0, best_objective - row.get("objective_net_value", 0.0)) / max(abs(best_objective), 1.0))
    else:
        row["relative_heuristic_gap"] = None
    return row


def _annotate_exact_quality(row: dict[str, Any], exact_objective: float) -> dict[str, Any]:
    objective = float(row.get("objective_net_value", 0.0))
    gap = max(0.0, float(exact_objective - objective))
    relative_gap = gap / max(abs(float(exact_objective)), 1.0)
    row["quality_reference_type"] = "exact_small_reference"
    row["quality_reference_objective"] = float(exact_objective)
    row["relative_optimality_gap"] = relative_gap
    row["relative_heuristic_gap"] = None
    row["quality_status"] = "PASS" if relative_gap <= 1e-9 else "MEASURED_GAP"
    row["primal_dual_gap"] = None
    row["primal_dual_gap_valid"] = False
    return row


def _aggregate_convergence(rows: list[dict[str, Any]]) -> str:
    applicable = [row for row in rows if row.get("convergence_status") != "NOT_APPLICABLE"]
    if not applicable:
        return "NOT_APPLICABLE"
    if all(row.get("convergence_status") == "PASS" for row in applicable):
        return "PASS"
    return "PARTIAL" if any(row.get("convergence_status") == "PASS" for row in applicable) else "NOT_CONVERGED"


def run_constrained_optimizer_suite(*, seed: int = 42, scales: tuple[int, ...] = (1000, 10000), isolate_scales: bool = False, memory_budget_mb: float | None = None, allow_oversize: bool = False) -> dict[str, Any]:
    small_candidates, small_constraints = _synthetic_candidates(18, seed)
    exact = exact_small_reference(small_candidates, small_constraints, max_candidates=20)
    solvers = [greedy_topk, solve_primal_dual, solve_line_search_primal_dual, solve_heap, solve_coordinate_iterator]
    small = [exact, *[solver(small_candidates, small_constraints) for solver in solvers]]
    exact_objective = exact["objective_net_value"]
    for row in small:
        _annotate_exact_quality(row, exact_objective)
        row["optimality_gap_vs_exact"] = float(exact_objective - row.get("objective_net_value", 0.0)) if row.get("status") == "PASS" else None
    scale_reports = []
    for size in scales:
        guard = _resource_guard(size, memory_budget_mb=memory_budget_mb, allow_oversize=allow_oversize)
        if guard["status"] != "PASS":
            scale_reports.append({"size": int(size), "status": "RESOURCE_GUARD_SKIPPED", "resource_guard": guard, "solvers": []})
            continue
        candidates, constraints = _synthetic_candidates(size, seed + size)
        if isolate_scales:
            raw_rows = [_run_isolated_solver(name, candidates, constraints, size) for name in _SOLVER_FUNCTIONS]
        else:
            raw_rows = [_scale_solver_call(name, candidates, constraints, size) for name in _SOLVER_FUNCTIONS]
        best = max((row.get("objective_net_value", 0.0) for row in raw_rows if row.get("status") == "PASS"), default=None)
        rows = [_benchmark_row(row, best) for row in raw_rows]
        scale_reports.append(
            {
                "size": int(size),
                "status": "PASS" if all(row.get("status") == "PASS" for row in rows) else "FAIL",
                "feasibility_status": "PASS" if all(row.get("feasibility_status") == "PASS" for row in rows) else "FAIL",
                "quality_status": "REFERENCE_ONLY",
                "convergence_status": _aggregate_convergence(rows),
                "fallback_rate": float(sum(bool(row.get("incumbent_fallback_used")) for row in rows) / len(rows)) if rows else 0.0,
                "candidate_memory_bytes": int(sum(np.asarray(value).nbytes for value in (candidates.incremental_value, candidates.resource_cost, candidates.user_id, candidates.time_slot, candidates.action))),
                "peak_rss_mb": max((row.get("process_peak_rss_mb", 0.0) for row in rows), default=0.0) if isolate_scales else None,
                "resource_guard": guard,
                "isolated_processes": bool(isolate_scales),
                "iteration_budgets": _scale_iteration_budgets(size),
                "solvers": rows,
            }
        )
    scale_statuses = [report["status"] for report in scale_reports]
    return {
        "status": "PASS" if exact["status"] == "PASS" and all(row["status"] == "PASS" for row in small + [solver for report in scale_reports for solver in report["solvers"]]) and all(status in {"PASS", "RESOURCE_GUARD_SKIPPED"} for status in scale_statuses) else "FAIL",
        "benchmark_version": "deepuplift-v1.1-optimizer-remediation",
        "feasibility_status": "PASS" if exact["status"] == "PASS" and all(row.get("feasibility_status") == "PASS" for row in small + [solver for report in scale_reports for solver in report["solvers"]]) else "FAIL",
        "quality_status": "PASS" if all(row.get("quality_status") == "PASS" for row in small) else "MEASURED_GAP",
        "convergence_status": _aggregate_convergence(small),
        "objective_definition": "sum(incremental_value - resource_cost)",
        "constraint_definition": "global budget + slot capacity + user frequency cap + minimum interval + action quota + at-most-one action per user-time + eligibility",
        "complexity_notes": {
            "greedy_topk": "O(N log N) static net sort plus incremental feasibility state",
            "heap": "O(N log N) static heap plus incremental feasibility state",
            "primal_dual": "O(I * N log N) with cached candidate arrays and independent O(K log K) checks",
            "line_search_primal_dual": "O(I * L * N log N) where L is the bounded line-search trial count",
            "coordinate_iterator": "O(I * C * N log N), sequentially recomputing between coordinates; not a dynamic sublinear claim",
            "feasibility_checker": "O(K log K) by per-user slot sorting; no pairwise combinations",
        },
        "environment": {"python": sys.version, "numpy": np.__version__, "platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor(), "cpu_count": os.cpu_count(), "seed": int(seed), "isolated_processes": bool(isolate_scales)},
        "small_instance": {"candidate_count": small_candidates.size, "exact_reference": exact, "solvers": small},
        "scales": scale_reports,
        "limitations": [
            "The named primal-dual variants are transparent research implementations, not claims of exact reproduction of proprietary algorithms.",
            "The exact reference is enumeration and is intentionally limited to small instances.",
            "Sparse candidate rows are used; no dense user x time x action tensor is allocated.",
        ],
    }


__all__ = ["AllocationCandidates", "AllocationConstraints", "check_feasibility", "coordinate_iterator", "exact_small_reference", "greedy_topk", "run_constrained_optimizer_suite", "solve_coordinate_iterator", "solve_heap", "solve_line_search_primal_dual", "solve_primal_dual"]
