from dataclasses import replace

import pandas as pd
import pytest

from domopt.classical import solve_classical
from domopt.data import make_tiny_problem_data
from domopt.objective import evaluate_solution
from domopt.validation import validate_solution


def test_classical_solver_finds_documented_tiny_optimum() -> None:
    problem = make_tiny_problem_data()
    solution = solve_classical(problem, time_limit_seconds=30)

    validation = validate_solution(problem, solution)
    assert validation.is_feasible, validation.violations

    selected = solution.assignments.set_index("order_id")["selected_dc"].to_dict()
    assert selected == {"O1": "D2", "O2": "D1"}

    objective = evaluate_solution(problem, solution)
    assert objective.objective_value == pytest.approx(126.0)
    assert solution.metadata["optimality_gap"] == pytest.approx(0.0)


def test_classical_solver_cannot_use_inventory_after_last_checkpoint() -> None:
    problem = make_tiny_problem_data()
    inventory = problem.inventory.copy()
    mask = (inventory["dc_id"] == "D2") & (inventory["sku_id"] == "A")
    inventory.loc[mask, "date"] = pd.Timestamp("2026-07-13")
    instance = replace(problem, inventory=inventory)

    solution = solve_classical(instance, time_limit_seconds=30)

    validation = validate_solution(instance, solution)
    assert validation.is_feasible, validation.violations
    uncovered = solution.fulfillment.loc[
        (solution.fulfillment["selected_dc"] == "D2")
        & (solution.fulfillment["sku_id"] == "A")
        & (pd.to_datetime(solution.fulfillment["selected_pgi_date"]) > "2026-07-13")
    ]
    assert (uncovered["fulfilled_cases"] == 0).all()


