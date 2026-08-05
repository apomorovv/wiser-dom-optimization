import pandas as pd
import pytest

from domopt.data import make_tiny_problem_data
from domopt.hybrid import HybridConfig, _resource_pressure, solve_hybrid
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
    assert solution.metadata["qpu_calls"] == 0
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
