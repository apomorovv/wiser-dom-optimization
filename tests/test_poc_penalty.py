import pandas as pd
import pytest

from domopt.classical import solve_classical
from domopt.data import normalize_problem_data
from domopt.objective import evaluate_solution
from domopt.penalties import order_penalty
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
