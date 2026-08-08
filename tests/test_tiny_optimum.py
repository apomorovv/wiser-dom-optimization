from dataclasses import replace

import pandas as pd
import pytest

from domopt.classical import (
    ClassicalSolverError,
    available_milp_backends,
    solve_classical,
)
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
    assert solution.metadata["milp_backend"] == "scipy-highs"


def test_milp_backend_discovery_keeps_portable_default_available() -> None:
    backends = available_milp_backends()

    assert backends["scipy-highs"] is True
    assert isinstance(backends["gurobi"], bool)


def test_unknown_milp_backend_has_actionable_error() -> None:
    with pytest.raises(ClassicalSolverError, match="scipy-highs"):
        solve_classical(make_tiny_problem_data(), backend="unknown")


@pytest.mark.skipif(
    not available_milp_backends()["gurobi"],
    reason="optional gurobipy package is not installed",
)
def test_optional_gurobi_backend_matches_highs_when_licensed() -> None:
    problem = make_tiny_problem_data()
    highs = solve_classical(problem, backend="scipy-highs")
    try:
        gurobi = solve_classical(problem, backend="gurobi")
    except ClassicalSolverError as error:
        pytest.skip(f"Gurobi is installed but no usable license is available: {error}")

    assert validate_solution(problem, gurobi).is_feasible
    assert evaluate_solution(problem, gurobi).objective_value == pytest.approx(
        evaluate_solution(problem, highs).objective_value
    )


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

