import pandas as pd
import pytest

from domopt.classical import ClassicalSolverError, solve_classical
from domopt.data import make_tiny_problem_data, normalize_problem_data
from domopt.resources import solution_capacity_usage
from domopt.schemas import ProblemData


def test_projected_atp_may_decrease_over_protection_horizon() -> None:
    problem = make_tiny_problem_data()
    later = problem.inventory.iloc[[0]].copy()
    later["date"] = pd.Timestamp("2026-07-15")
    later["cumulative_available_cases"] = 2
    inventory = pd.concat([problem.inventory, later], ignore_index=True)

    normalized = normalize_problem_data(
        ProblemData(
            orders=problem.orders,
            order_lines=problem.order_lines,
            inventory=inventory,
            candidates=problem.candidates,
            capacities=problem.capacities,
            calendar=problem.calendar,
            metadata={**problem.metadata, "inventory_policy": "projected_atp"},
        )
    )
    assert len(normalized.inventory) == len(problem.inventory) + 1


def test_five_percent_rule_rejects_insufficient_alternate_fill() -> None:
    problem = make_tiny_problem_data()
    with pytest.raises(ClassicalSolverError):
        solve_classical(
            problem,
            fixed_assignments={"O1": "O1_D1_T1", "O2": "O2_D2_T1"},
        )


def test_exact_pallet_and_loose_case_capacity_accounting() -> None:
    problem = make_tiny_problem_data()
    lines = problem.order_lines.copy()
    lines["cases_per_pallet"] = 4
    capacities = pd.DataFrame(
        [
            {
                "dc_id": "D1",
                "date": "2026-07-14",
                "resource": "pallet_pick",
                "capacity": 1,
                "unit": "pallets",
            },
            {
                "dc_id": "D1",
                "date": "2026-07-14",
                "resource": "case_pick",
                "capacity": 3,
                "unit": "cases",
            },
            {
                "dc_id": "D2",
                "date": "2026-07-14",
                "resource": "pallet_pick",
                "capacity": 1,
                "unit": "pallets",
            },
            {
                "dc_id": "D2",
                "date": "2026-07-14",
                "resource": "case_pick",
                "capacity": 2,
                "unit": "cases",
            },
        ]
    )
    split_problem = normalize_problem_data(
        ProblemData(
            orders=problem.orders,
            order_lines=lines,
            inventory=problem.inventory,
            candidates=problem.candidates,
            capacities=capacities,
            calendar=problem.calendar,
            metadata={**problem.metadata, "pick_capacity_mode": "pallet_case"},
        )
    )
    solution = solve_classical(split_problem)
    usage = solution_capacity_usage(split_problem, solution)

    assert usage[("D1", pd.Timestamp("2026-07-14"), "pallet_pick")] == 1
    assert usage[("D1", pd.Timestamp("2026-07-14"), "case_pick")] == 3
    assert usage[("D2", pd.Timestamp("2026-07-14"), "pallet_pick")] == 1
    assert usage[("D2", pd.Timestamp("2026-07-14"), "case_pick")] == 2
