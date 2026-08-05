import numpy as np
import pandas as pd
import pytest

from domopt.quantum import QuantumSolverError, sample_qubo
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


def test_remote_qpu_requires_explicit_privacy_approval() -> None:
    with pytest.raises(QuantumSolverError):
        sample_qubo(_model(), method="dwave-qpu", allow_remote=False)
