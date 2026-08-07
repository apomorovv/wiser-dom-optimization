import pandas as pd

from domopt.baselines import solve_default_baseline
from domopt.data import make_tiny_problem_data
from domopt.schemas import Solution
from domopt.validation import validate_solution


def test_validator_reports_unknown_candidate_with_capacities() -> None:
    problem = make_tiny_problem_data()
    capacities = pd.DataFrame(
        [
            {
                "dc_id": "D1",
                "date": pd.Timestamp("2026-07-14"),
                "resource": "dock",
                "capacity": 2,
                "unit": "orders",
            }
        ]
    )
    problem = type(problem)(
        orders=problem.orders,
        order_lines=problem.order_lines,
        inventory=problem.inventory,
        candidates=problem.candidates,
        capacities=capacities,
        calendar=problem.calendar,
        metadata=problem.metadata,
    )
    assignments = pd.DataFrame(
        [
            {
                "order_id": order_id,
                "candidate_id": "unknown" if order_id == "O1" else "O2_D1_T1",
                "selected_dc": "D1",
                "selected_pgi_date": pd.Timestamp("2026-07-14"),
                "is_unassigned": False,
                "is_divert": False,
                "method": "test",
            }
            for order_id in ["O1", "O2"]
        ]
    )
    fulfillment = pd.DataFrame(
        [
            {
                "order_id": row.order_id,
                "sku_id": row.sku_id,
                "fulfilled_cases": 0,
                "unfulfilled_cases": row.demand_cases,
                "selected_dc": "D1",
                "selected_pgi_date": pd.Timestamp("2026-07-14"),
            }
            for row in problem.order_lines.itertuples(index=False)
        ]
    )

    result = validate_solution(
        problem,
        Solution(method="test", assignments=assignments, fulfillment=fulfillment),
    )

    assert not result.is_feasible
    assert result.eligibility_violations
    assert result.schema_violations


def test_validator_parses_explicit_boolean_strings_without_truthiness_bug() -> None:
    problem = make_tiny_problem_data()
    solution = solve_default_baseline(problem)
    solution.assignments["is_unassigned"] = solution.assignments[
        "is_unassigned"
    ].map({True: "True", False: "False"})
    solution.assignments["is_divert"] = solution.assignments["is_divert"].map(
        {True: "True", False: "False"}
    )

    assert validate_solution(problem, solution).is_feasible


def test_validator_rejects_invalid_booleans_and_nonfinite_quantities() -> None:
    problem = make_tiny_problem_data()
    solution = solve_default_baseline(problem)
    solution.assignments["is_unassigned"] = solution.assignments[
        "is_unassigned"
    ].astype(object)
    solution.fulfillment["fulfilled_cases"] = solution.fulfillment[
        "fulfilled_cases"
    ].astype(float)
    solution.assignments.loc[0, "is_unassigned"] = "not-a-boolean"
    solution.fulfillment.loc[0, "fulfilled_cases"] = float("inf")

    result = validate_solution(problem, solution)

    assert not result.is_feasible
    assert any("expected a boolean" in item for item in result.schema_violations)
    assert any("nonfinite quantities" in item for item in result.demand_violations)
