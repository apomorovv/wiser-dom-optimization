import pandas as pd
import pytest

from domopt.copilot import answer_experiment_question, validate_copilot_data


def _results() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "experiment": "pareto_pruning_ablation",
                "level": "without_pruning",
                "method": "hybrid",
                "feasible": True,
                "objective_value": 10,
                "runtime_seconds": 2,
                "candidate_count": 8,
            },
            {
                "experiment": "pareto_pruning_ablation",
                "level": "with_pruning",
                "method": "hybrid",
                "feasible": True,
                "objective_value": 10,
                "runtime_seconds": 1,
                "candidate_count": 4,
            },
        ]
    )


def test_copilot_explains_pruning_from_aggregate_rows() -> None:
    answer = answer_experiment_question("Did Pareto pruning help?", _results())
    assert "4 candidates" in answer
    assert "without_pruning" in answer


def test_copilot_rejects_identifier_columns() -> None:
    unsafe = _results().assign(selected_order_id="private")
    with pytest.raises(ValueError, match="aggregate results only"):
        validate_copilot_data(unsafe)


def test_copilot_does_not_treat_false_string_as_feasible() -> None:
    results = _results().assign(feasible="False")
    answer = answer_experiment_question("Did Pareto pruning help?", results)
    assert answer.startswith("No feasible run")


def test_copilot_recommends_fastest_numerically_tied_method() -> None:
    rows = pd.DataFrame(
        [
            {
                "experiment": "solver_comparison",
                "level": "exact_lns",
                "method": "exact_lns",
                "feasible": True,
                "objective_value": 100.0,
                "runtime_seconds": 0.5,
                "optimality_gap": None,
            },
            {
                "experiment": "solver_comparison",
                "level": "classical",
                "method": "classical",
                "feasible": True,
                "objective_value": 100.0,
                "runtime_seconds": 2.0,
                "optimality_gap": 0.0,
            },
        ]
    )

    answer = answer_experiment_question("Which solver do you recommend?", rows)

    assert "exact_lns is fastest" in answer
    assert "zero-gap certificate" in answer
