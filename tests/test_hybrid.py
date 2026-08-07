import pandas as pd
import pytest

from domopt.baselines import solve_greedy_baseline
from domopt.classical import solve_classical
from domopt.data import make_tiny_problem_data
from domopt.hybrid import (
    ExactLNSConfig,
    HybridConfig,
    _resource_pressure,
    solve_exact_lns,
    solve_hybrid,
)
from domopt.objective import evaluate_solution
from domopt.synthetic import make_synthetic_problem
from domopt.validation import validate_solution


def test_hybrid_improves_default_and_finds_tiny_optimum() -> None:
    problem = make_tiny_problem_data()
    solution = solve_hybrid(
        problem,
        config=HybridConfig(
            initial_method="default",
            sampler="exact",
            iterations=2,
            neighborhood_orders=2,
            max_qubo_variables=8,
            num_reads=32,
            sweeps=20,
            top_k_recourse=8,
        ),
    )

    assert validate_solution(problem, solution).is_feasible
    assert evaluate_solution(problem, solution).objective_value == pytest.approx(126.0)
    assert solution.metadata["improvement"] > 0
    assert solution.metadata["total_improvement"] == pytest.approx(
        solution.metadata["initial_polish_improvement"]
        + solution.metadata["improvement"]
    )
    assert solution.metadata["qpu_calls"] == 0
    assert solution.metadata["qpu_access_time_microseconds"] is None
    assert solution.metadata["hardware_wall_seconds"] is None
    assert solution.metadata["maximum_qubo_variables"] <= 8


def test_hybrid_does_not_degrade_greedy_incumbent() -> None:
    problem = make_tiny_problem_data()
    solution = solve_hybrid(
        problem,
        config=HybridConfig(
            initial_method="greedy",
            sampler="exact",
            iterations=1,
            neighborhood_orders=2,
            max_qubo_variables=8,
            top_k_recourse=4,
        ),
    )
    assert solution.metadata["final_objective"] >= solution.metadata["initial_objective"]


def test_resource_pressure_detects_higher_order_contention() -> None:
    key = ("capacity", "D1", pd.Timestamp("2026-07-14"), "throughput_cases")
    plans = pd.DataFrame(
        [
            {"order_id": order_id, "usage": {key: 4.0}}
            for order_id in ["O1", "O2", "O3"]
        ]
    )

    pressure = _resource_pressure(plans, {key: 10.0})

    assert pressure[key] == pytest.approx(1.0 / 6.0)


def test_candidate_reduction_keeps_qubo_within_cap() -> None:
    problem = make_synthetic_problem(
        order_count=1,
        dc_count=4,
        candidates_per_order=4,
        seed=3,
    )
    solution = solve_hybrid(
        problem,
        config=HybridConfig(
            iterations=1,
            neighborhood_orders=1,
            max_qubo_variables=3,
            max_candidates_per_order=2,
            sampler="exact",
            top_k_recourse=3,
        ),
    )

    assert validate_solution(problem, solution).is_feasible
    assert solution.metadata["maximum_qubo_variables"] == 3


def test_exact_lns_closes_verified_assignment_gap_without_degrading() -> None:
    problem = make_synthetic_problem(order_count=4, seed=6)
    solution = solve_exact_lns(
        problem,
        config=ExactLNSConfig(
            iterations=1,
            initial_neighborhood_groups=4,
            minimum_neighborhood_groups=2,
            maximum_neighborhood_groups=4,
            maximum_neighborhood_orders=4,
            local_time_limit_seconds=5,
            mip_relative_gap=0,
            polish_initial_incumbent=True,
            seed=6,
        ),
    )

    assert validate_solution(problem, solution).is_feasible
    assert evaluate_solution(problem, solution).objective_value == pytest.approx(-2548.4)
    assert solution.metadata["search_improvement"] == pytest.approx(197.2)
    assert solution.metadata["accepted_moves"] == 1
    assert solution.metadata["assignment_moves"] == 1
    assert solution.metadata["maximum_local_variables"] > 0


def test_exact_lns_matches_tiny_exact_optimum_without_degrading_greedy() -> None:
    problem = make_tiny_problem_data()
    greedy = solve_greedy_baseline(problem)
    exact = solve_classical(problem, time_limit_seconds=5, mip_relative_gap=0)
    lns = solve_exact_lns(
        problem,
        config=ExactLNSConfig(
            iterations=1,
            initial_neighborhood_groups=2,
            minimum_neighborhood_groups=1,
            maximum_neighborhood_groups=2,
            maximum_neighborhood_orders=2,
            maximum_local_fulfillment_variables=100,
            local_time_limit_seconds=5,
            mip_relative_gap=0,
            polish_initial_incumbent=False,
            seed=5,
        ),
    )

    greedy_value = evaluate_solution(problem, greedy).objective_value
    exact_value = evaluate_solution(problem, exact).objective_value
    lns_value = evaluate_solution(problem, lns).objective_value
    assert validate_solution(problem, lns).is_feasible
    assert lns_value >= greedy_value
    assert lns_value == pytest.approx(exact_value)
