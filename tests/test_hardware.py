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
