import pandas as pd

from domopt.data import make_tiny_problem_data
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


def test_validator_detects_demand_and_inventory_violation() -> None:
    validation = validate_solution(
        make_tiny_problem_data(),
        _solution(o1_a=5, o1_a_unfilled=0),
    )
    assert not validation.is_feasible
    assert validation.demand_violations
    assert validation.inventory_violations


