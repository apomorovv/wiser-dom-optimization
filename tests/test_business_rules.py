import pandas as pd
import pytest

from domopt.classical import ClassicalSolverError, solve_classical
from domopt.data import make_tiny_problem_data, normalize_problem_data
from domopt.resources import solution_capacity_usage
from domopt.rules import minimum_divert_fulfillment
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


def test_diversion_rule_uses_hundred_case_floor() -> None:
    problem = make_tiny_problem_data()
    orders = problem.orders.iloc[[0]].copy()
    orders["default_fillable_cases"] = 100
    lines = problem.order_lines.loc[problem.order_lines["order_id"] == "O1"].copy()
    lines.loc[lines.index[0], "demand_cases"] = 498
    lines.loc[lines.index[1], "demand_cases"] = 2
    large = ProblemData(
        orders=orders,
        order_lines=lines,
        inventory=problem.inventory,
        candidates=problem.candidates.loc[
            problem.candidates["order_id"] == "O1"
        ].copy(),
        capacities=problem.capacities,
        calendar=problem.calendar,
        metadata={
            **problem.metadata,
            "min_divert_improvement_cases": 100,
        },
    )

    assert minimum_divert_fulfillment(large, "O1") == 200


def test_exact_model_enforces_assignment_group_cohesion() -> None:
    problem = make_tiny_problem_data()
    orders = problem.orders.copy()
    orders["assignment_group"] = "G1"
    candidates = problem.candidates.copy()
    candidates["group_option_id"] = candidates["dc_id"] + "::T1"
    grouped = normalize_problem_data(
        ProblemData(
            orders=orders,
            order_lines=problem.order_lines,
            inventory=problem.inventory,
            candidates=candidates,
            capacities=problem.capacities,
            calendar=problem.calendar,
            metadata={
                **problem.metadata,
                "enforce_assignment_group": True,
                "enforce_min_divert_improvement": False,
            },
        )
    )

    with pytest.raises(ClassicalSolverError):
        solve_classical(
            grouped,
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
