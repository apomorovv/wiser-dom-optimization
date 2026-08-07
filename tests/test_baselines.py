import pytest

from domopt.baselines import (
    solve_default_baseline,
    solve_greedy_baseline,
    solve_polished_greedy,
)
from domopt.data import make_tiny_problem_data
from domopt.objective import evaluate_solution
from domopt.validation import validate_solution


def test_default_and_greedy_baselines_are_feasible() -> None:
    problem = make_tiny_problem_data()
    default = solve_default_baseline(problem)
    greedy = solve_greedy_baseline(problem)

    assert validate_solution(problem, default).is_feasible
    assert validate_solution(problem, greedy).is_feasible


def test_greedy_finds_tiny_split_assignment() -> None:
    problem = make_tiny_problem_data()
    greedy = solve_greedy_baseline(problem)
    selected = greedy.assignments.set_index("order_id")["selected_dc"].to_dict()

    assert selected == {"O1": "D2", "O2": "D1"}
    assert evaluate_solution(problem, greedy).objective_value == pytest.approx(126.0)


def test_greedy_is_better_than_default_on_tiny_instance() -> None:
    problem = make_tiny_problem_data()
    default_value = evaluate_solution(problem, solve_default_baseline(problem)).objective_value
    greedy_value = evaluate_solution(problem, solve_greedy_baseline(problem)).objective_value
    assert greedy_value > default_value


def test_polished_greedy_is_feasible_and_never_degrades_greedy() -> None:
    problem = make_tiny_problem_data()
    greedy = solve_greedy_baseline(problem)
    polished = solve_polished_greedy(problem, time_limit_seconds=5)

    assert validate_solution(problem, polished).is_feasible
    assert evaluate_solution(problem, polished).objective_value >= evaluate_solution(
        problem, greedy
    ).objective_value
    assert polished.metadata["execution_class"] == "classical-matheuristic"


