from dataclasses import replace

import pandas as pd
import pytest

from domopt.data import (
    DataValidationError,
    make_tiny_problem_data,
    normalize_problem_data,
)
from domopt.schemas import Solution
from domopt.validation import validate_solution


def _solution(*, o1_a: int = 4, o1_a_unfilled: int = 0) -> Solution:
    assignments = pd.DataFrame(
        [
            {"order_id": "O1", "candidate_id": "O1_D2_T1", "selected_dc": "D2", "selected_pgi_date": "2026-07-14", "is_unassigned": False, "is_divert": True, "method": "test"},
            {"order_id": "O2", "candidate_id": "O2_D1_T1", "selected_dc": "D1", "selected_pgi_date": "2026-07-14", "is_unassigned": False, "is_divert": False, "method": "test"},
        ]
    )
    fulfillment = pd.DataFrame(
        [
            {"order_id": "O1", "sku_id": "A", "fulfilled_cases": o1_a, "unfulfilled_cases": o1_a_unfilled, "selected_dc": "D2", "selected_pgi_date": "2026-07-14"},
            {"order_id": "O1", "sku_id": "B", "fulfilled_cases": 2, "unfulfilled_cases": 0, "selected_dc": "D2", "selected_pgi_date": "2026-07-14"},
            {"order_id": "O2", "sku_id": "A", "fulfilled_cases": 3, "unfulfilled_cases": 0, "selected_dc": "D1", "selected_pgi_date": "2026-07-14"},
            {"order_id": "O2", "sku_id": "B", "fulfilled_cases": 4, "unfulfilled_cases": 0, "selected_dc": "D1", "selected_pgi_date": "2026-07-14"},
        ]
    )
    return Solution(method="test", assignments=assignments, fulfillment=fulfillment)


def test_tiny_optimal_solution_is_feasible() -> None:
    validation = validate_solution(make_tiny_problem_data(), _solution())
    assert validation.is_feasible
    assert validation.violations == []
    assert validation.diagnostics["validation_tolerance"] == pytest.approx(1e-8)
    assert validation.diagnostics["maximum_demand_balance_abs_error"] == 0.0
    assert validation.diagnostics["maximum_inventory_excess_cases"] == 0.0
    assert validation.diagnostics["enabled_capacity_resources"] == "none"
    assert validation.diagnostics["capacity_constraints_enabled"] is False


def test_validator_detects_demand_and_inventory_violation() -> None:
    validation = validate_solution(
        make_tiny_problem_data(),
        _solution(o1_a=5, o1_a_unfilled=0),
    )
    assert not validation.is_feasible
    assert validation.demand_violations
    assert validation.inventory_violations
    assert validation.diagnostics["maximum_demand_balance_abs_error"] == 1.0
    assert validation.diagnostics["maximum_inventory_excess_cases"] == 1.0


def test_canonical_gate_rejects_nonfinite_numbers_and_orders_without_lines() -> None:
    problem = make_tiny_problem_data()
    candidates = problem.candidates.copy()
    candidates.loc[0, "shipping_cost"] = float("inf")
    with pytest.raises(DataValidationError, match="Nonfinite"):
        normalize_problem_data(replace(problem, candidates=candidates))

    lines = problem.order_lines.loc[problem.order_lines["order_id"] != "O2"]
    with pytest.raises(DataValidationError, match="at least one line"):
        normalize_problem_data(replace(problem, order_lines=lines))
