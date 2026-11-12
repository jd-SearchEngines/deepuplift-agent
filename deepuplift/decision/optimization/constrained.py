"""Research constrained allocation solvers for multi-coupled candidates.

The existing ``simulate_budget`` Top-K helper remains unchanged.  These
solvers operate on sparse candidate rows and expose reduced utilities, dual
prices, feasibility checks, and an exact small-instance reference.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import itertools
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

    def __post_init__(self) -> None:
        arrays = [self.incremental_value, self.resource_cost, self.user_id, self.time_slot, self.action]
        lengths = {len(np.asarray(value)) for value in arrays}
        if len(lengths) != 1:
            raise ValueError("All candidate arrays must have the same length.")
        n = next(iter(lengths))
        if n == 0:
            raise ValueError("At least one candidate is required.")
        if self.eligible is not None and len(self.eligible) != n:
            raise ValueError("eligible must match the candidate row count.")
        if not np.isfinite(np.asarray(self.incremental_value, dtype="float64")).all() or not np.isfinite(np.asarray(self.resource_cost, dtype="float64")).all():
            raise ValueError("Candidate value and cost arrays must be finite.")

    @property
    def size(self) -> int:
        return len(self.incremental_value)

    @property
    def net_value(self) -> np.ndarray:
        return np.asarray(self.incremental_value, dtype="float64") - np.asarray(self.resource_cost, dtype="float64")

    def eligible_mask(self) -> np.ndarray:
        return np.ones(self.size, dtype=bool) if self.eligible is None else np.asarray(self.eligible, dtype=bool)


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
    if not candidates.eligible_mask()[idx]:
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
    if not candidates.eligible_mask()[idx]:
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
    usage = _constraint_usage(candidates, selected)
    violations = [max(0.0, usage["cost"] - constraints.budget)]
    for slot, count in usage["slot"].items():
        violations.append(max(0.0, float(count - int(constraints.slot_capacity.get(slot, 0)))))
    for user, count in usage["user"].items():
        violations.append(max(0.0, float(count - constraints.user_frequency_cap)))
    if constraints.action_quota is not None:
        for action, count in usage["action"].items():
            violations.append(max(0.0, float(count - int(constraints.action_quota.get(action, 0)))))
    duplicates = len(selected) - len(usage["decision_key"])
    violations.append(float(max(0, duplicates)))
    interval_violations = 0
    for left, right in itertools.combinations(selected, 2):
        if candidates.user_id[left] == candidates.user_id[right] and abs(int(candidates.time_slot[left]) - int(candidates.time_slot[right])) < constraints.minimum_interval:
            interval_violations += 1
    violations.append(float(interval_violations))
    return {"feasible": max(violations, default=0.0) <= 1e-9, "max_constraint_violation": float(max(violations, default=0.0)), "violations": violations, "usage": usage}


def _result(name: str, candidates: AllocationCandidates, constraints: AllocationConstraints, selected: list[int], *, iterations: int, dual_prices: dict[str, Any], stopping_reason: str, relaxed_residual: float, started: float, trace: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    feasibility = check_feasibility(candidates, constraints, selected)
    usage = feasibility["usage"]
    return {
        "solver": name,
        "status": "PASS" if feasibility["feasible"] else "INFEASIBLE",
        "selected_indices": [int(i) for i in selected],
        "selected_count": int(len(selected)),
        "incremental_value": float(np.sum(np.asarray(candidates.incremental_value)[selected])) if selected else 0.0,
        "total_cost": float(usage["cost"]),
        "objective_net_value": float(np.sum(candidates.net_value[selected])) if selected else 0.0,
        "feasibility": {key: value for key, value in feasibility.items() if key != "usage"},
        "usage": {key: value for key, value in usage.items() if key != "decision_key"},
        "dual_prices": dual_prices,
        "dual_residual": float(relaxed_residual),
        "stopping_reason": stopping_reason,
        "iterations": int(iterations),
        "runtime_seconds": float(time.perf_counter() - started),
        "candidate_count": int(candidates.size),
        "trace": trace or [],
    }


def greedy_topk(candidates: AllocationCandidates, constraints: AllocationConstraints) -> dict[str, Any]:
    started = time.perf_counter()
    order = np.argsort(-candidates.net_value, kind="mergesort")
    selected = []
    state = _new_state()
    for idx in order:
        idx = int(idx)
        if _state_can_add(candidates, constraints, state, idx):
            selected.append(idx)
            _state_add(candidates, state, idx)
    return _result("greedy_topk", candidates, constraints, selected, iterations=1, dual_prices={}, stopping_reason="sorted_net_value", relaxed_residual=0.0, started=started)


def _reduced_utilities(candidates: AllocationCandidates, prices: dict[str, Any]) -> np.ndarray:
    slot_price = prices.get("slot", {})
    action_price = prices.get("action", {})
    return candidates.net_value - prices.get("budget", 0.0) * np.asarray(candidates.resource_cost, dtype="float64") - np.asarray([slot_price.get(value, 0.0) for value in candidates.time_slot]) - np.asarray([action_price.get(value, 0.0) for value in candidates.action])


def _relaxed_select(candidates: AllocationCandidates, constraints: AllocationConstraints, utilities: np.ndarray) -> list[int]:
    selected: list[int] = []
    # Relax resource/slot/action capacities for dual pricing, but preserve
    # eligibility, one action per user-time, frequency, and interval guards.
    user_counts: dict[Any, int] = {}
    keys: set[tuple[Any, Any]] = set()
    user_slots: dict[Any, list[Any]] = {}
    for idx in np.argsort(-utilities, kind="mergesort"):
        idx = int(idx)
        if not candidates.eligible_mask()[idx] or utilities[idx] <= 0:
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


def _dual_violations(candidates: AllocationCandidates, constraints: AllocationConstraints, selected: list[int]) -> tuple[float, dict[str, Any]]:
    usage = _constraint_usage(candidates, selected)
    budget_excess = max(0.0, usage["cost"] - constraints.budget) / max(constraints.budget, 1.0)
    slot_excess = {slot: max(0.0, count - int(constraints.slot_capacity.get(slot, 0))) / max(int(constraints.slot_capacity.get(slot, 0)), 1) for slot, count in usage["slot"].items()}
    action_excess = {}
    if constraints.action_quota is not None:
        action_excess = {action: max(0.0, count - int(constraints.action_quota.get(action, 0))) / max(int(constraints.action_quota.get(action, 0)), 1) for action, count in usage["action"].items()}
    residual = float(np.sqrt(budget_excess**2 + sum(value * value for value in slot_excess.values()) + sum(value * value for value in action_excess.values())))
    return residual, {"budget": budget_excess, "slot": slot_excess, "action": action_excess}


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
    prices: dict[str, Any] = {"budget": 0.0, "slot": {}, "action": {}}
    trace = []
    relaxed_residual = float("inf")
    for iteration in range(1, max_iterations + 1):
        utilities = _reduced_utilities(candidates, prices)
        relaxed = _relaxed_select(candidates, constraints, utilities)
        relaxed_residual, violation = _dual_violations(candidates, constraints, relaxed)
        trace.append({"iteration": iteration, "relaxed_residual": relaxed_residual, "selected": len(relaxed)})
        if relaxed_residual <= 1e-6:
            break
        rate = step_size / np.sqrt(iteration)
        prices["budget"] += rate * violation["budget"]
        for key, value in violation["slot"].items():
            prices["slot"][key] = prices["slot"].get(key, 0.0) + rate * value
        for key, value in violation["action"].items():
            prices["action"][key] = prices["action"].get(key, 0.0) + rate * value
    selected = _repair(candidates, constraints, _reduced_utilities(candidates, prices))
    return _result("primal_dual", candidates, constraints, selected, iterations=iteration, dual_prices=prices, stopping_reason="residual_tolerance" if relaxed_residual <= 1e-6 else "max_iterations", relaxed_residual=relaxed_residual, started=started, trace=trace)


def solve_line_search_primal_dual(candidates: AllocationCandidates, constraints: AllocationConstraints, *, max_iterations: int = 40) -> dict[str, Any]:
    started = time.perf_counter()
    prices: dict[str, Any] = {"budget": 0.0, "slot": {}, "action": {}}
    trace = []
    residual = float("inf")
    for iteration in range(1, max_iterations + 1):
        current = _dual_violations(candidates, constraints, _relaxed_select(candidates, constraints, _reduced_utilities(candidates, prices)))[0]
        best = (current, prices)
        for multiplier in (0.25, 0.5, 1.0, 2.0):
            trial = {"budget": prices["budget"], "slot": dict(prices["slot"]), "action": dict(prices["action"])}
            relaxed = _relaxed_select(candidates, constraints, _reduced_utilities(candidates, trial))
            residual_trial, violation = _dual_violations(candidates, constraints, relaxed)
            rate = multiplier / np.sqrt(iteration)
            trial["budget"] += rate * violation["budget"]
            for key, value in violation["slot"].items():
                trial["slot"][key] = trial["slot"].get(key, 0.0) + rate * value
            for key, value in violation["action"].items():
                trial["action"][key] = trial["action"].get(key, 0.0) + rate * value
            after = _dual_violations(candidates, constraints, _relaxed_select(candidates, constraints, _reduced_utilities(candidates, trial)))[0]
            if after < best[0]:
                best = (after, trial)
        residual, prices = best
        trace.append({"iteration": iteration, "line_search_residual": residual})
        if residual <= 1e-6:
            break
    selected = _repair(candidates, constraints, _reduced_utilities(candidates, prices))
    return _result("line_search_primal_dual", candidates, constraints, selected, iterations=iteration, dual_prices=prices, stopping_reason="residual_tolerance" if residual <= 1e-6 else "max_iterations", relaxed_residual=residual, started=started, trace=trace)


def solve_heap(candidates: AllocationCandidates, constraints: AllocationConstraints) -> dict[str, Any]:
    started = time.perf_counter()
    heap = [(-float(value), int(idx)) for idx, value in enumerate(candidates.net_value)]
    heapq.heapify(heap)
    selected: list[int] = []
    state = _new_state()
    while heap:
        _, idx = heapq.heappop(heap)
        if _state_can_add(candidates, constraints, state, idx):
            selected.append(idx)
            _state_add(candidates, state, idx)
    return _result("heap", candidates, constraints, selected, iterations=1, dual_prices={}, stopping_reason="heap_exhausted", relaxed_residual=0.0, started=started)


def coordinate_iterator(candidates: AllocationCandidates) -> Iterator[tuple[str, Any]]:
    yield ("budget", None)
    for value in sorted(set(candidates.time_slot.tolist()), key=str):
        yield ("slot", value)
    for value in sorted(set(candidates.action.tolist()), key=str):
        yield ("action", value)


def solve_coordinate_iterator(candidates: AllocationCandidates, constraints: AllocationConstraints, *, max_iterations: int = 40) -> dict[str, Any]:
    started = time.perf_counter()
    prices: dict[str, Any] = {"budget": 0.0, "slot": {}, "action": {}}
    residual = float("inf")
    trace = []
    for iteration in range(1, max_iterations + 1):
        selected = _relaxed_select(candidates, constraints, _reduced_utilities(candidates, prices))
        residual, violation = _dual_violations(candidates, constraints, selected)
        for coordinate, key in coordinate_iterator(candidates):
            if coordinate == "budget":
                prices["budget"] += 0.5 * violation["budget"]
            elif coordinate == "slot":
                prices["slot"][key] = prices["slot"].get(key, 0.0) + 0.5 * violation["slot"].get(key, 0.0)
            else:
                prices["action"][key] = prices["action"].get(key, 0.0) + 0.5 * violation["action"].get(key, 0.0)
        trace.append({"iteration": iteration, "coordinate_residual": residual})
        if residual <= 1e-6:
            break
    selected = _repair(candidates, constraints, _reduced_utilities(candidates, prices))
    return _result("coordinate_iterator", candidates, constraints, selected, iterations=iteration, dual_prices=prices, stopping_reason="residual_tolerance" if residual <= 1e-6 else "max_iterations", relaxed_residual=residual, started=started, trace=trace)


def exact_small_reference(candidates: AllocationCandidates, constraints: AllocationConstraints, *, max_candidates: int = 20) -> dict[str, Any]:
    if candidates.size > max_candidates:
        return {"solver": "exact_small_reference", "status": "NOT_RUN", "reason": f"exact reference is limited to <= {max_candidates} candidates", "candidate_count": candidates.size}
    started = time.perf_counter()
    best: list[int] = []
    best_value = 0.0
    for mask in range(1 << candidates.size):
        selected = [idx for idx in range(candidates.size) if mask & (1 << idx)]
        feasibility = check_feasibility(candidates, constraints, selected)
        if feasibility["feasible"]:
            value = float(np.sum(candidates.net_value[selected])) if selected else 0.0
            if value > best_value:
                best_value, best = value, selected
    return _result("exact_small_reference", candidates, constraints, best, iterations=1 << candidates.size, dual_prices={}, stopping_reason="enumeration_complete", relaxed_residual=0.0, started=started)


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


def run_constrained_optimizer_suite(*, seed: int = 42, scales: tuple[int, ...] = (1000, 10000)) -> dict[str, Any]:
    small_candidates, small_constraints = _synthetic_candidates(18, seed)
    exact = exact_small_reference(small_candidates, small_constraints, max_candidates=20)
    solvers = [greedy_topk, solve_primal_dual, solve_line_search_primal_dual, solve_heap, solve_coordinate_iterator]
    small = [exact, *[solver(small_candidates, small_constraints) for solver in solvers]]
    exact_objective = exact["objective_net_value"]
    for row in small:
        row["optimality_gap_vs_exact"] = float(exact_objective - row.get("objective_net_value", 0.0)) if row.get("status") == "PASS" else None
    scale_reports = []
    for size in scales:
        candidates, constraints = _synthetic_candidates(size, seed + size)
        rows = [solver(candidates, constraints) for solver in solvers]
        scale_reports.append({"size": size, "candidate_memory_bytes": int(sum(np.asarray(value).nbytes for value in (candidates.incremental_value, candidates.resource_cost, candidates.user_id, candidates.time_slot, candidates.action))), "peak_rss_mb": _peak_rss_mb(), "solvers": rows})
    return {
        "status": "PASS" if exact["status"] == "PASS" and all(row["status"] == "PASS" for row in small + [solver for report in scale_reports for solver in report["solvers"]]) else "FAIL",
        "objective_definition": "sum(incremental_value - resource_cost)",
        "constraint_definition": "global budget + slot capacity + user frequency cap + minimum interval + action quota + at-most-one action per user-time + eligibility",
        "small_instance": {"candidate_count": small_candidates.size, "exact_reference": exact, "solvers": small},
        "scales": scale_reports,
        "limitations": [
            "The named primal-dual variants are transparent research implementations, not claims of exact reproduction of proprietary algorithms.",
            "The exact reference is enumeration and is intentionally limited to small instances.",
            "Sparse candidate rows are used; no dense user x time x action tensor is allocated.",
        ],
    }


__all__ = ["AllocationCandidates", "AllocationConstraints", "check_feasibility", "coordinate_iterator", "exact_small_reference", "greedy_topk", "run_constrained_optimizer_suite", "solve_coordinate_iterator", "solve_heap", "solve_line_search_primal_dual", "solve_primal_dual"]
