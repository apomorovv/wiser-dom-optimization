from __future__ import annotations

import json
from dataclasses import replace

import pandas as pd
import pytest

from domopt.data import make_tiny_problem_data
from domopt.experiments import experiment_profile, write_experiment_results
from domopt.hybrid import ExactLNSConfig
from domopt.pipeline import problem_fingerprint, run_methods


def test_common_pipeline_runs_polished_greedy_and_exact_lns(tmp_path) -> None:
    problem = make_tiny_problem_data()
    exact_lns = ExactLNSConfig(
        iterations=1,
        initial_neighborhood_groups=2,
        minimum_neighborhood_groups=1,
        maximum_neighborhood_groups=2,
        maximum_neighborhood_orders=2,
        maximum_local_fulfillment_variables=100,
        local_time_limit_seconds=5,
        mip_relative_gap=0,
        polish_initial_incumbent=False,
        seed=7,
    )

    summary = run_methods(
        problem,
        ["polished_greedy", "exact_lns"],
        tmp_path,
        experiment_id="pipeline-regression",
        seed=7,
        time_limit_seconds=5,
        exact_lns_config=exact_lns,
    )

    assert set(summary["method"]) == {"polished_greedy", "exact_lns"}
    assert summary["feasible"].all()
    assert (tmp_path / "comparison.csv").is_file()
    for method in ["polished_greedy", "exact_lns"]:
        run_dir = tmp_path / method
        assert (run_dir / "assignments.csv").is_file()
        assert (run_dir / "validation.json").is_file()
        configuration = json.loads((run_dir / "config.json").read_text())
        assert configuration["method"] == method


@pytest.mark.parametrize(
    "identifier_column",
    ["order_id", "best_candidate_id", "origin_zip3", "material_number", "trace_id"],
)
def test_experiment_aggregate_rejects_identifier_like_columns(
    tmp_path, identifier_column: str
) -> None:
    results = pd.DataFrame(
        {
            "experiment": ["privacy-regression"],
            "objective_value": [1.0],
            identifier_column: ["sensitive-value"],
        }
    )
    output = tmp_path / "aggregate.csv"

    with pytest.raises(ValueError, match="forbidden columns"):
        write_experiment_results(results, output)

    assert not output.exists()


def test_problem_fingerprint_is_invariant_to_table_row_order() -> None:
    problem = make_tiny_problem_data()

    def reverse_rows(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.iloc[::-1].reset_index(drop=True)

    reordered = replace(
        problem,
        orders=reverse_rows(problem.orders),
        order_lines=reverse_rows(problem.order_lines),
        inventory=reverse_rows(problem.inventory),
        candidates=reverse_rows(problem.candidates),
        capacities=reverse_rows(problem.capacities),
        calendar=reverse_rows(problem.calendar),
    )

    assert problem_fingerprint(reordered) == problem_fingerprint(problem)


def test_production_lns_profile_polishes_its_incumbent() -> None:
    assert experiment_profile("smoke").exact_lns.polish_initial_incumbent is True
    assert experiment_profile("full").exact_lns.polish_initial_incumbent is True


def test_scaling_visualization_aggregates_repetitions_and_new_methods(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    pytest.importorskip("matplotlib")
    from domopt.visualization import plot_challenge_results

    rows: list[dict[str, object]] = []
    methods = ["greedy", "polished_greedy", "exact_lns", "hybrid"]
    for groups in [4, 8]:
        for repetition in [1, 2]:
            for method_index, method in enumerate(methods):
                rows.append(
                    {
                        "experiment": "size_scaling",
                        "level": f"groups={groups}:{method}:rep={repetition}",
                        "method": method,
                        "feasible": True,
                        "actual_assignment_groups": groups,
                        "runtime_seconds": groups * (method_index + 1) + repetition,
                        "objective_capture_rate": 0.6 + 0.01 * method_index,
                        "candidate_count": 3 * groups,
                        "maximum_qubo_variables": (
                            groups + 2 if method == "hybrid" else float("nan")
                        ),
                        "maximum_local_variables": (
                            10 * groups if method == "exact_lns" else float("nan")
                        ),
                    }
                )

    generated = plot_challenge_results(pd.DataFrame(rows), tmp_path / "figures")

    assert set(generated) == {"size_scaling"}
    assert generated["size_scaling"].is_file()
    assert generated["size_scaling"].stat().st_size > 0
