from domopt.classical import solve_classical
from domopt.data import make_tiny_problem_data
from domopt.planner import build_planner_table


def test_planner_view_explains_tiny_diversion() -> None:
    problem = make_tiny_problem_data()
    solution = solve_classical(problem)

    table = build_planner_table(problem, solution)

    diverted = table.loc[table["decision"] == "DIVERT"]
    assert len(table) == 2
    assert len(diverted) == 1
    assert diverted.iloc[0]["fill_uplift_cases"] > 0
