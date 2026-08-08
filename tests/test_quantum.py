from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.linalg import expm

from domopt.quantum import (
    QuantumSolverError,
    _append_linear_w_state,
    _hardware_sample_statistics,
    _ibm_job_timing,
    _path_edges,
    _xy_edge_unitary,
    ibm_sampler_options,
    sample_qubo,
)
from domopt.qubo import build_candidate_qubo


def _model():
    plans = pd.DataFrame(
        [
            {"plan_id": "A0", "order_id": "A", "value": 0},
            {"plan_id": "A1", "order_id": "A", "value": 5},
            {"plan_id": "B0", "order_id": "B", "value": 0},
            {"plan_id": "B1", "order_id": "B", "value": 4},
        ]
    )
    return build_candidate_qubo(plans, one_hot_penalty=20)


def test_exact_qubo_sample_respects_one_hot_ground_state() -> None:
    samples = sample_qubo(_model(), method="exact")
    best = samples.iloc[0]
    assert int(best["A1"]) == 1
    assert int(best["B1"]) == 1
    assert int(best["A0"]) == 0
    assert int(best["B0"]) == 0


def test_feasible_exact_enumerates_only_one_hot_states() -> None:
    samples = sample_qubo(_model(), method="exact_feasible")

    assert len(samples) == 4
    assert (samples[["A0", "A1"]].sum(axis=1) == 1).all()
    assert (samples[["B0", "B1"]].sum(axis=1) == 1).all()
    assert samples.attrs["sampler_info"]["feasible_state_dimension"] == 4


def test_qaoa_statevector_is_reproducible_and_constraint_preserving() -> None:
    first = sample_qubo(
        _model(),
        method="qaoa_statevector",
        num_samples=32,
        seed=7,
        qaoa_restarts=2,
    )
    second = sample_qubo(
        _model(),
        method="qaoa_statevector",
        num_samples=32,
        seed=7,
        qaoa_restarts=2,
    )

    assert first[["bitstring", "energy"]].equals(second[["bitstring", "energy"]])
    assert (first[["A0", "A1"]].sum(axis=1) == 1).all()
    assert (first[["B0", "B1"]].sum(axis=1) == 1).all()
    assert first.attrs["sampler_info"]["algorithm"].startswith("QAOA")
    assert first.attrs["sampler_info"]["statevector_probability_sum"] == pytest.approx(
        1.0
    )
    assert first.attrs["sampler_info"]["initial_sample_used"] is False
    assert first.attrs["sampler_info"]["mixer_topology"] == "path"
    assert first.attrs["sampler_info"]["sample_raw_one_hot_rate"] == 1.0
    assert first.attrs["sampler_info"]["uniform_feasible_optimal_rate"] == 0.25


def test_optimized_qaoa_parameters_can_be_reused_without_reoptimization() -> None:
    first = sample_qubo(
        _model(),
        method="qaoa_statevector",
        num_samples=8,
        seed=5,
        qaoa_restarts=1,
    )
    parameters = tuple(first.attrs["sampler_info"]["optimized_parameters"])

    reused = sample_qubo(
        _model(),
        method="qaoa_statevector",
        num_samples=8,
        seed=5,
        qaoa_restarts=1,
        qaoa_parameters=parameters,
    )

    assert reused.attrs["sampler_info"]["parameters_reused"] is True
    assert reused.attrs["sampler_info"]["parameter_source"] == "provided and reused"


def test_path_mixer_is_connected_with_one_fewer_edge_than_ring() -> None:
    assert _path_edges(1) == ()
    assert _path_edges(4) == ((0, 1), (1, 2), (2, 3))


@pytest.mark.parametrize("qubits", [1, 2, 3, 4])
def test_linear_w_state_preparation_is_uniform_and_one_hot(qubits: int) -> None:
    qiskit = pytest.importorskip("qiskit")
    from qiskit.quantum_info import Statevector

    circuit = qiskit.QuantumCircuit(qubits)
    _append_linear_w_state(circuit, tuple(range(qubits)))
    probabilities = np.abs(Statevector.from_instruction(circuit).data) ** 2
    expected = np.zeros(2**qubits)
    for state in range(2**qubits):
        if state.bit_count() == 1:
            expected[state] = 1.0 / qubits

    assert np.allclose(probabilities, expected)


def test_restricted_xy_mixer_matches_dense_two_qubit_circuit() -> None:
    beta = 0.37
    x = np.asarray([[0, 1], [1, 0]], dtype=complex)
    y = np.asarray([[0, -1j], [1j, 0]], dtype=complex)
    dense = expm(-1j * beta * 0.5 * (np.kron(x, x) + np.kron(y, y)))
    one_excitation = dense[np.ix_([2, 1], [2, 1])]

    assert np.allclose(one_excitation, _xy_edge_unitary(2, (0, 1), beta))


def test_qaoa_readout_noise_is_reproducible_and_exposes_raw_violations() -> None:
    samples = sample_qubo(
        _model(),
        method="qaoa_statevector",
        num_samples=128,
        seed=13,
        qaoa_restarts=1,
        qaoa_readout_bitflip_probability=0.5,
    )
    one_hot = (samples[["A0", "A1"]].sum(axis=1) == 1) & (
        samples[["B0", "B1"]].sum(axis=1) == 1
    )

    assert not one_hot.all()
    assert samples.attrs["sampler_info"]["readout_bitflip_probability"] == 0.5
    assert "not gate" in samples.attrs["sampler_info"]["noise_scope"]


def test_simulated_annealing_is_reproducible() -> None:
    first = sample_qubo(
        _model(),
        method="simulated_annealing",
        num_samples=4,
        sweeps=20,
        seed=11,
        initial_sample=np.array([0, 1, 0, 1]),
    )
    second = sample_qubo(
        _model(),
        method="simulated_annealing",
        num_samples=4,
        sweeps=20,
        seed=11,
        initial_sample=np.array([0, 1, 0, 1]),
    )
    assert first[["bitstring", "energy"]].equals(second[["bitstring", "energy"]])


def test_ibm_qpu_requires_explicit_privacy_approval() -> None:
    with pytest.raises(QuantumSolverError):
        sample_qubo(_model(), method="ibm-qpu", allow_remote=False)


def test_unknown_backend_is_rejected() -> None:
    with pytest.raises(QuantumSolverError, match="Unknown"):
        sample_qubo(_model(), method="unsupported-qpu")  # type: ignore[arg-type]


def test_ibm_mitigation_options_are_explicit_and_cost_bounded() -> None:
    baseline = ibm_sampler_options("baseline", shots=256)
    mitigated = ibm_sampler_options(
        "dd_measure_twirling",
        shots=512,
        max_execution_time_seconds=20_000,
    )

    assert baseline["dynamical_decoupling"]["enable"] is False
    assert baseline["twirling"]["enable_measure"] is False
    assert mitigated["dynamical_decoupling"]["sequence_type"] == "XpXm"
    assert mitigated["twirling"]["enable_measure"] is True
    assert mitigated["max_execution_time"] == 10_800


def test_hardware_sample_statistics_measure_raw_feasibility_and_optimum() -> None:
    samples = [
        np.asarray([0, 1, 0, 1], dtype=np.int8),
        np.asarray([1, 1, 0, 0], dtype=np.int8),
    ]

    statistics = _hardware_sample_statistics(
        _model(), samples, max_feasible_states=16
    )

    assert statistics["hardware_raw_one_hot_rate"] == pytest.approx(0.5)
    assert statistics["hardware_feasible_shots"] == 1
    assert statistics["hardware_qubo_optimal_hit_rate"] == pytest.approx(0.5)
    assert statistics["hardware_best_feasible_normalized_gap"] == pytest.approx(0.0)


def test_ibm_job_timing_uses_runtime_metrics() -> None:
    class FakeJob:
        def __init__(self) -> None:
            self.usage_estimation = {"quantum_seconds": 0.125}

        @staticmethod
        def metrics():
            return {
                "timestamps": {
                    "created": "2026-08-07T10:00:00Z",
                    "running": "2026-08-07T10:00:03Z",
                    "finished": "2026-08-07T10:00:05Z",
                }
            }

    timing = _ibm_job_timing(FakeJob(), wall_seconds=5.5)

    assert timing["hardware_queue_seconds"] == pytest.approx(3.0)
    assert timing["hardware_execution_seconds"] == pytest.approx(2.0)
    assert timing["hardware_turnaround_seconds"] == pytest.approx(5.0)
    assert timing["hardware_quantum_seconds"] == pytest.approx(0.125)


def test_ibm_adapter_transpiles_and_collects_current_runtime_schema(monkeypatch) -> None:
    runtime = pytest.importorskip("qiskit_ibm_runtime")
    pytest.importorskip("qiskit")
    from qiskit.providers.fake_provider import GenericBackendV2
    from qiskit_ibm_runtime.options import SamplerOptions

    class FakeBackend(GenericBackendV2):
        def status(self):
            return SimpleNamespace(operational=True, pending_jobs=3)

    backend = FakeBackend(20, seed=7)

    class FakeService:
        @staticmethod
        def least_busy(**kwargs):
            assert kwargs["min_num_qubits"] == 4
            assert kwargs["operational"] is True
            return backend

        @staticmethod
        def backend(name, **kwargs):
            assert name == backend.name
            assert kwargs["use_fractional_gates"] is False
            return backend

    class FakeMeasurement:
        @staticmethod
        def get_counts():
            return {"1010": 16}

    class FakeJob:
        usage_estimation = {"quantum_seconds": 0.125}  # noqa: RUF012

        @staticmethod
        def result():
            publication = SimpleNamespace(
                data=SimpleNamespace(meas=FakeMeasurement())
            )
            return [publication]

        @staticmethod
        def metrics():
            return {
                "timestamps": {
                    "created": "2026-08-07T10:00:00Z",
                    "running": "2026-08-07T10:00:03Z",
                    "finished": "2026-08-07T10:00:05Z",
                }
            }

        @staticmethod
        def job_id():
            return "synthetic-test-job"

    class FakeSampler:
        def __init__(self, *, mode, options):
            assert mode is backend
            SamplerOptions(**options)

        @staticmethod
        def run(publications, *, shots):
            assert len(publications) == 1
            assert shots == 16
            return FakeJob()

    monkeypatch.setattr(runtime, "QiskitRuntimeService", FakeService)
    monkeypatch.setattr(runtime, "SamplerV2", FakeSampler)

    samples = sample_qubo(
        _model(),
        method="ibm-qpu",
        num_samples=16,
        seed=3,
        allow_remote=True,
        qaoa_restarts=1,
        ibm_backend_name=backend.name,
        ibm_mitigation_strategy="dd_measure_twirling",
        ibm_transpiler_trials=2,
    )
    info = samples.attrs["sampler_info"]

    assert len(samples) == 16
    assert info["backend_name"] == backend.name
    assert info["mitigation_strategy"] == "dd_measure_twirling"
    assert info["returned_samples"] == 16
    assert info["hardware_qubo_optimal_hit_rate"] == pytest.approx(1.0)
    assert info["transpiled_two_qubit_gates"] > 0
    assert info["hardware_queue_seconds"] == pytest.approx(3.0)
