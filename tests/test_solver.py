import pytest

from domopt.data import make_tiny_problem_data
from domopt.hybrid import ExactLNSConfig, HybridConfig
from domopt.solver import SolverConfig, solve_dom
from domopt.validation import validate_solution


@pytest.mark.parametrize("mode", ["fast", "quality", "hybrid"])
def test_solver_modes_return_validated_feasible_incumbents(mode: str) -> None:
    problem = make_tiny_problem_data()
    config = SolverConfig(
        mode=mode,
        fast_time_limit_seconds=5.0,
        exact_lns=ExactLNSConfig(
            iterations=1,
            minimum_neighborhood_groups=1,
            initial_neighborhood_groups=2,
            maximum_neighborhood_groups=2,
            maximum_neighborhood_orders=2,
            local_time_limit_seconds=5.0,
        ),
        hybrid=HybridConfig(
            iterations=1,
            neighborhood_orders=2,
            max_qubo_variables=8,
            sampler="exact",
            top_k_recourse=4,
            recourse_time_limit_seconds=5.0,
        ),
    )

    solution = solve_dom(problem, config=config)

    assert validate_solution(problem, solution).is_feasible
    assert solution.metadata["solver_mode"] == mode
    assert solution.metadata["independently_validated"] is True
    assert solution.raw_objective == pytest.approx(solution.metadata["final_objective"])


def test_solver_config_rejects_unknown_mode() -> None:
    config = SolverConfig(mode="unknown")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="mode"):
        config.validate()
