from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from domopt.classical import solve_classical
from domopt.data import make_tiny_problem_data, normalize_problem_data
from domopt.objective import evaluate_solution
from domopt.penalties import order_penalty
from domopt.planner import _order_economics
from domopt.poc import select_penalty_subset
from domopt.schemas import ProblemData
from domopt.validation import validate_solution


def _problem() -> ProblemData:
    date = pd.Timestamp("2024-07-01")
    return normalize_problem_data(
        ProblemData(
            orders=pd.DataFrame(
                [
                    {
                        "order_id": "O1",
                        "default_dc": "D1",
                        "requested_delivery_date": date,
                        "penalty_threshold_fraction": 0.9,
                        "penalty_fixed": 10,
                        "penalty_per_cut_sku": 3,
                        "penalty_minimum": 100,
                        "penalty_maximum": 150,
                    }
                ]
            ),
            order_lines=pd.DataFrame(
                [
                    {
                        "order_id": "O1",
                        "sku_id": "S1",
                        "demand_cases": 100,
                        "unit_value": 1,
                        "penalty_per_unfilled_case": 2,
                    }
                ]
            ),
            inventory=pd.DataFrame(
                [
                    {
                        "dc_id": "D1",
                        "sku_id": "S1",
                        "date": date,
                        "cumulative_available_cases": 89,
                    }
                ]
            ),
            candidates=pd.DataFrame(
                [
                    {
                        "candidate_id": "C1",
                        "order_id": "O1",
                        "dc_id": "D1",
                        "pgi_date": date,
                        "shipping_cost": 0,
                        "is_default": True,
                        "eligible": True,
                    }
                ]
            ),
            capacities=pd.DataFrame(
                columns=["dc_id", "date", "resource", "capacity", "unit"]
            ),
            calendar=pd.DataFrame(
                [{"dc_id": "D1", "date": date, "is_open": True}]
            ),
            metadata={"penalty_mode": "thresholded_cut"},
        )
    )


def test_threshold_floor_and_cap_are_exact() -> None:
    problem = _problem()
    line = problem.order_lines.copy()

    line["unfulfilled_cases"] = 10
    assert order_penalty(problem, "O1", line) == 0

    line["unfulfilled_cases"] = 11
    assert order_penalty(problem, "O1", line) == 100

    line["unfulfilled_cases"] = 100
    assert order_penalty(problem, "O1", line) == 150


def test_milp_penalty_matches_independent_evaluator() -> None:
    problem = _problem()
    solution = solve_classical(problem, time_limit_seconds=10)

    assert validate_solution(problem, solution).is_feasible
    assert solution.fulfillment.iloc[0]["fulfilled_cases"] == 89
    evaluated = evaluate_solution(problem, solution)
    assert evaluated.penalty_cost == 100
    assert solution.raw_objective == pytest.approx(evaluated.objective_value)


def test_milp_active_cap_matches_independent_evaluator() -> None:
    problem = _problem()
    inventory = problem.inventory.copy()
    inventory["cumulative_available_cases"] = 0
    capped = replace(problem, inventory=inventory)

    solution = solve_classical(capped, time_limit_seconds=10, mip_relative_gap=0)
    evaluated = evaluate_solution(capped, solution)

    assert validate_solution(capped, solution).is_feasible
    assert evaluated.penalty_cost == pytest.approx(150.0)
    assert solution.raw_objective == pytest.approx(evaluated.objective_value)


def test_planner_uses_thresholded_floor_and_cap_penalty() -> None:
    problem = _problem()
    solution = solve_classical(problem, time_limit_seconds=10)

    economics = _order_economics(problem, solution)

    assert economics["O1"]["penalty_cost"] == pytest.approx(100.0)


def test_penalty_subset_excludes_orders_already_above_threshold() -> None:
    problem = make_tiny_problem_data()
    orders = problem.orders.copy()
    orders["penalty_threshold_fraction"] = 1.0
    orders["penalty_fixed"] = 0.0
    orders["penalty_per_cut_sku"] = 0.0
    orders["penalty_minimum"] = 0.0
    orders["penalty_maximum"] = 0.0
    thresholded = replace(
        problem,
        orders=orders,
        metadata={**problem.metadata, "penalty_mode": "thresholded_cut"},
    )

    selected = select_penalty_subset(thresholded, 1)

    assert selected.orders["order_id"].tolist() == ["O1"]
    assert selected.metadata["selection_basis"] == "active_penalty_exposure"


def test_randomized_multisku_milp_penalty_matches_evaluator() -> None:
    rng = np.random.default_rng(23)
    date = pd.Timestamp("2026-07-01")
    for case in range(12):
        demands = rng.integers(2, 20, size=2)
        available = [int(rng.integers(0, demand + 1)) for demand in demands]
        minimum = float(rng.integers(0, 25))
        maximum = float(minimum + rng.integers(1, 50)) if case % 2 else 0.0
        problem = normalize_problem_data(
            ProblemData(
                orders=pd.DataFrame(
                    [
                        {
                            "order_id": "O1",
                            "default_dc": "D1",
                            "requested_delivery_date": date,
                            "penalty_threshold_fraction": float(
                                rng.choice([0.0, 0.5, 0.8, 1.0])
                            ),
                            "penalty_fixed": float(rng.integers(0, 20)),
                            "penalty_per_cut_sku": float(rng.integers(0, 10)),
                            "penalty_minimum": minimum,
                            "penalty_maximum": maximum,
                        }
                    ]
                ),
                order_lines=pd.DataFrame(
                    [
                        {
                            "order_id": "O1",
                            "sku_id": f"S{index + 1}",
                            "demand_cases": int(demand),
                            "unit_value": float(index + 1),
                            "penalty_per_unfilled_case": float(
                                rng.integers(0, 8)
                            ),
                        }
                        for index, demand in enumerate(demands)
                    ]
                ),
                inventory=pd.DataFrame(
                    [
                        {
                            "dc_id": "D1",
                            "sku_id": f"S{index + 1}",
                            "date": date,
                            "cumulative_available_cases": quantity,
                        }
                        for index, quantity in enumerate(available)
                    ]
                ),
                candidates=pd.DataFrame(
                    [
                        {
                            "candidate_id": "C1",
                            "order_id": "O1",
                            "dc_id": "D1",
                            "pgi_date": date,
                            "shipping_cost": 0.0,
                            "is_default": True,
                            "eligible": True,
                        }
                    ]
                ),
                capacities=pd.DataFrame(
                    columns=["dc_id", "date", "resource", "capacity", "unit"]
                ),
                calendar=pd.DataFrame(
                    [{"dc_id": "D1", "date": date, "is_open": True}]
                ),
                metadata={"penalty_mode": "thresholded_cut"},
            )
        )

        solution = solve_classical(problem, time_limit_seconds=10, mip_relative_gap=0)
        evaluated = evaluate_solution(problem, solution)
        quantities = problem.order_lines.merge(
            solution.fulfillment[
                ["order_id", "sku_id", "fulfilled_cases", "unfulfilled_cases"]
            ],
            on=["order_id", "sku_id"],
            how="left",
            validate="one_to_one",
        )
        expected = order_penalty(problem, "O1", quantities)

        assert validate_solution(problem, solution).is_feasible
        assert evaluated.penalty_cost == pytest.approx(expected)
        assert solution.raw_objective == pytest.approx(evaluated.objective_value)
