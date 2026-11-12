from __future__ import annotations

import numpy as np

from deepuplift.decision.optimization import AllocationCandidates, AllocationConstraints, check_feasibility, exact_small_reference, greedy_topk, run_constrained_optimizer_suite, solve_coordinate_iterator, solve_heap, solve_line_search_primal_dual, solve_primal_dual
from deepuplift.decision.optimization.constrained import _projected_dual_update


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


def test_heuristics_match_exact_on_multiple_small_seeds_and_are_deterministic():
    for seed in range(5):
        rng = np.random.default_rng(seed)
        candidates = AllocationCandidates(
            incremental_value=rng.uniform(-1, 4, 12),
            resource_cost=rng.choice([0.0, 1.0, 2.0], 12),
            user_id=np.arange(12) // 3,
            time_slot=np.arange(12) % 6,
            action=np.array(["a", "b", "c"] * 4),
            eligible=np.array([1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1], dtype=np.int8),
        )
        constraints = AllocationConstraints(4, {slot: 2 for slot in range(6)}, 2, {"a": 2, "b": 2, "c": 2}, 1)
        exact = exact_small_reference(candidates, constraints)
        solvers = [greedy_topk, solve_primal_dual, solve_line_search_primal_dual, solve_heap, solve_coordinate_iterator]
        rows = [solver(candidates, constraints) for solver in solvers]
        assert all(row["feasibility"]["feasible"] for row in rows)
        assert all(row["objective_net_value"] <= exact["objective_net_value"] + 1e-9 for row in rows)
        assert all(row["selected_indices"] == solver(candidates, constraints)["selected_indices"] for row, solver in zip(rows, solvers))


def test_candidate_validation_eligibility_and_edge_constraints():
    with np.testing.assert_raises(ValueError):
        AllocationCandidates(np.ones((2, 1)), np.ones(2), np.arange(2), np.arange(2), np.array(["a", "b"]))
    with np.testing.assert_raises(ValueError):
        AllocationCandidates(np.ones(2), np.array([-1.0, 1.0]), np.arange(2), np.arange(2), np.array(["a", "b"]))
    with np.testing.assert_raises(ValueError):
        AllocationCandidates(np.ones(2), np.ones(2), np.arange(2), np.arange(2), np.array(["a", "b"]), eligible=np.array([0.1, 1.0]))

    candidates = AllocationCandidates(np.array([-2.0, 3.0, 4.0]), np.ones(3), np.array([0, 0, 1]), np.array([0, 1, 1]), np.array(["a", "b", "b"]), eligible=np.array([0, 1, 1]))
    zero = AllocationConstraints(0, {0: 0, 1: 0}, 1, action_quota=None, minimum_interval=2)
    row = greedy_topk(candidates, zero)
    assert row["selected_indices"] == []
    assert not check_feasibility(candidates, zero, [0])["feasible"]
    assert check_feasibility(candidates, zero, [0, 1])["minimum_interval_violation_count"] == 1
    assert check_feasibility(candidates, AllocationConstraints(10, {0: 2, 1: 2}, 2, None, 0), [1, 1])["feasible"] is False
    assert greedy_topk(candidates, AllocationConstraints(10, {0: 2, 1: 2}, 2, None, 0))["objective_net_value"] == 5.0
    negative = AllocationCandidates(np.array([-2.0]), np.array([1.0]), np.array([0]), np.array([0]), np.array(["a"]))
    assert greedy_topk(negative, AllocationConstraints(10, {0: 1}, 1))["selected_indices"] == []


def test_signed_dual_update_allows_prices_to_decrease_but_never_negative():
    prices = {"budget": 1.0, "slot": {0: 0.5}, "action": {"a": 0.25}}
    updated = _projected_dual_update(prices, {"budget": -2.0, "slot": {0: -2.0}, "action": {"a": -2.0}}, 1.0)
    assert updated == {"budget": 0.0, "slot": {0: 0.0}, "action": {"a": 0.0}}
    assert all(value >= 0 for value in [updated["budget"], *updated["slot"].values(), *updated["action"].values()])


def test_coordinate_iterator_recomputes_between_coordinates_and_reports_diagnostics():
    candidates, constraints = _small()
    row = solve_coordinate_iterator(candidates, constraints)
    assert row["trace"]
    assert all(item["recomputed"] for item in row["trace"])
    assert row["signed_residuals"] is not None
    assert "complementarity" in row


def test_large_independent_feasibility_checker_does_not_use_pairwise_selection_checks():
    size = 20_000
    candidates = AllocationCandidates(np.ones(size), np.ones(size), np.arange(size), np.arange(size), np.zeros(size, dtype=int))
    constraints = AllocationConstraints(float(size), {index: 1 for index in range(size)}, 1)
    result = check_feasibility(candidates, constraints, list(range(size)))
    assert result["feasible"]
    assert result["minimum_interval_violation_count"] == 0
