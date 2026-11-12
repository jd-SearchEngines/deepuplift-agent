from __future__ import annotations

import numpy as np

from deepuplift.decision.optimization import AllocationCandidates, AllocationConstraints, exact_small_reference, greedy_topk, run_constrained_optimizer_suite, solve_coordinate_iterator, solve_heap, solve_line_search_primal_dual, solve_primal_dual


def _small():
    candidates = AllocationCandidates(
        incremental_value=np.array([4.0, 3.0, 2.0, 1.0]),
        resource_cost=np.ones(4),
        user_id=np.array([0, 0, 1, 1]),
        time_slot=np.array([0, 1, 0, 2]),
        action=np.array(["a", "a", "b", "b"]),
    )
    constraints = AllocationConstraints(budget=2, slot_capacity={0: 1, 1: 1, 2: 1}, user_frequency_cap=1, action_quota={"a": 1, "b": 1}, minimum_interval=0)
    return candidates, constraints


def test_all_solvers_are_feasible_and_exact_reference_dominates():
    candidates, constraints = _small()
    exact = exact_small_reference(candidates, constraints)
    rows = [greedy_topk(candidates, constraints), solve_primal_dual(candidates, constraints), solve_line_search_primal_dual(candidates, constraints), solve_heap(candidates, constraints), solve_coordinate_iterator(candidates, constraints)]
    assert exact["status"] == "PASS"
    assert all(row["feasibility"]["feasible"] for row in rows)
    assert all(row["objective_net_value"] <= exact["objective_net_value"] + 1e-9 for row in rows)


def test_optimizer_suite_reports_exact_gap_and_sparse_scales():
    report = run_constrained_optimizer_suite(seed=3, scales=(50,))
    assert report["status"] == "PASS"
    assert report["small_instance"]["exact_reference"]["status"] == "PASS"
    assert report["scales"][0]["candidate_memory_bytes"] > 0
