import numpy as np
import pandas as pd
import pytest
from scipy.linalg import expm

from domopt.quantum import QuantumSolverError, _xy_edge_unitary, sample_qubo
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


@pytest.mark.parametrize("method", ["dwave-qpu", "ibm-qpu"])
def test_remote_qpu_requires_explicit_privacy_approval(method: str) -> None:
    with pytest.raises(QuantumSolverError):
        sample_qubo(_model(), method=method, allow_remote=False)
