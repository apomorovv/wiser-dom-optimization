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
    unsafe = _results().assign(order_id="private")
    with pytest.raises(ValueError, match="aggregate results only"):
        validate_copilot_data(unsafe)
