import pandas as pd
import pytest

from domopt.data import make_tiny_problem_data
from domopt.objective import evaluate_solution
from domopt.schemas import Solution


def _optimal_solution() -> Solution:
    assignments = pd.DataFrame(
        [
            {
                "order_id": "O1",
                "candidate_id": "O1_D2_T1",
                "selected_dc": "D2",
                "selected_pgi_date": pd.Timestamp("2026-07-14"),
                "is_unassigned": False,
                "is_divert": True,
                "method": "test",
            },
            {
                "order_id": "O2",
                "candidate_id": "O2_D1_T1",
                "selected_dc": "D1",
                "selected_pgi_date": pd.Timestamp("2026-07-14"),
                "is_unassigned": False,
                "is_divert": False,
                "method": "test",
            },
        ]
    )
    fulfillment = pd.DataFrame(
        [
            {"order_id": "O1", "sku_id": "A", "fulfilled_cases": 4, "unfulfilled_cases": 0, "selected_dc": "D2", "selected_pgi_date": pd.Timestamp("2026-07-14")},
            {"order_id": "O1", "sku_id": "B", "fulfilled_cases": 2, "unfulfilled_cases": 0, "selected_dc": "D2", "selected_pgi_date": pd.Timestamp("2026-07-14")},
            {"order_id": "O2", "sku_id": "A", "fulfilled_cases": 3, "unfulfilled_cases": 0, "selected_dc": "D1", "selected_pgi_date": pd.Timestamp("2026-07-14")},
            {"order_id": "O2", "sku_id": "B", "fulfilled_cases": 4, "unfulfilled_cases": 0, "selected_dc": "D1", "selected_pgi_date": pd.Timestamp("2026-07-14")},
        ]
    )
    return Solution(method="test", assignments=assignments, fulfillment=fulfillment)


def test_tiny_objective_breakdown_is_exact() -> None:
    problem = make_tiny_problem_data()
    objective = evaluate_solution(problem, _optimal_solution())

    assert objective.fulfilled_value == pytest.approx(130.0)
    assert objective.penalty_cost == pytest.approx(0.0)
    assert objective.shipping_cost == pytest.approx(4.0)
    assert objective.objective_value == pytest.approx(126.0)

