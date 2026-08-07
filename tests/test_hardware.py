import pandas as pd
import pytest

from domopt.checkpoints import challenge_results_root
from domopt.experiments import rank_ibm_hardware_strategies
from domopt.hardware import benchmark_qubo_batch_scoring, hardware_capabilities


def test_cpu_qubo_scoring_benchmark_is_privacy_safe_and_executable() -> None:
    capabilities = hardware_capabilities()
    benchmark = benchmark_qubo_batch_scoring(
        variable_counts=(4,),
        sample_counts=(16,),
        repeats=1,
        include_gpu=False,
    )

    assert "logical_cpu_count" in capabilities
    assert benchmark.to_dict("records")[0]["backend"] == "numpy_cpu"
    assert benchmark.to_dict("records")[0]["samples_per_second"] > 0
    assert benchmark.to_dict("records")[0]["end_to_end_samples_per_second"] > 0
    assert benchmark.to_dict("records")[0]["end_to_end_seconds"] == benchmark.to_dict(
        "records"
    )[0]["compute_seconds"]


def test_cli_and_notebook_result_roots_cannot_overlap(tmp_path) -> None:
    cli = challenge_results_root(tmp_path, producer="cli")
    notebook = challenge_results_root(tmp_path, producer="notebook")

    assert cli == tmp_path / "results/challenge-study/cli"
    assert notebook == tmp_path / "results/challenge-study/notebook"
    assert cli != notebook


def _ibm_results() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    variants = [
        (1, "baseline", 0.10, 0.40, 80, 0.20, 12.0),
        (1, "dynamical_decoupling", 0.24, 0.62, 82, 0.24, 13.0),
        (2, "dd_measure_twirling", 0.18, 0.70, 150, 0.35, 18.0),
    ]
    for seed_offset in (0.0, 0.01, -0.01):
        for layers, mitigation, hit, one_hot, gates, quantum, runtime in variants:
            rows.append(
                {
                    "feasible": True,
                    "sampler_backend": "ibm-qpu",
                    "qaoa_layers": layers,
                    "hardware_mitigation_strategy": mitigation,
                    "hardware_optimal_hit_rate": hit + seed_offset,
                    "raw_one_hot_rate": one_hot + seed_offset,
                    "search_improvement": 197.2,
                    "hardware_two_qubit_gates": gates,
                    "hardware_quantum_seconds": quantum,
                    "hardware_queue_seconds": 2.0,
                    "hardware_execution_seconds": 1.0,
                    "runtime_seconds": runtime,
                }
            )
    return pd.DataFrame(rows)


def test_ibm_strategy_ranking_prioritizes_raw_exact_quality() -> None:
    ranking = rank_ibm_hardware_strategies(_ibm_results())

    best = ranking.iloc[0]
    assert best["qaoa_layers"] == 1
    assert best["hardware_mitigation_strategy"] == "dynamical_decoupling"
    assert bool(best["selected_best_observed"])
    assert best["successful_runs"] == 3
    assert best["hardware_optimal_hit_rate"] == pytest.approx(0.24)


def test_ibm_presentation_figures_render(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    pytest.importorskip("matplotlib")
    from domopt.visualization import (
        plot_ibm_backend_snapshot,
        plot_ibm_hardware_study,
    )

    queue = pd.DataFrame(
        [
            {
                "backend": "ibm_a",
                "pending_jobs": 4,
                "selected_least_busy": True,
                "selected_for_study": True,
            },
            {
                "backend": "ibm_b",
                "pending_jobs": 7,
                "selected_least_busy": False,
                "selected_for_study": False,
            },
        ]
    )
    queue_path = plot_ibm_backend_snapshot(queue, tmp_path / "queue.png")
    study_path = plot_ibm_hardware_study(_ibm_results(), tmp_path / "study.png")

    assert queue_path.is_file() and queue_path.stat().st_size > 0
    assert study_path.is_file() and study_path.stat().st_size > 0
