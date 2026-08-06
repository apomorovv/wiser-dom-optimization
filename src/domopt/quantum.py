"""Reproducible local samplers and an optional privacy-gated D-Wave adapter."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import product
from math import exp
from typing import Literal

import numpy as np
import pandas as pd

from .qubo import qubo_energy
from .schemas import QUBOModel


class QuantumSolverError(RuntimeError):
    pass


def _initial_vector(
    model: QUBOModel,
    initial_sample: Mapping[str, int] | np.ndarray | list[int] | None,
) -> np.ndarray | None:
    if initial_sample is None:
        return None
    if isinstance(initial_sample, Mapping):
        return np.asarray(
            [int(initial_sample.get(name, 0)) for name in model.variable_names],
            dtype=np.int8,
        )
    vector = np.asarray(initial_sample, dtype=np.int8)
    if vector.shape != (len(model.variable_names),):
        raise QuantumSolverError(
            f"Initial sample has shape {vector.shape}; "
            f"expected {(len(model.variable_names),)}"
        )
    if not np.isin(vector, [0, 1]).all():
        raise QuantumSolverError("Initial sample must be binary")
    return vector


def _simulated_annealing_samples(
    model: QUBOModel,
    *,
    num_samples: int,
    sweeps: int,
    seed: int,
    initial_sample: Mapping[str, int] | np.ndarray | list[int] | None,
) -> list[np.ndarray]:
    if num_samples <= 0 or sweeps <= 0:
        raise QuantumSolverError("num_samples and sweeps must be positive")
    n = len(model.variable_names)
    if n == 0:
        return [np.zeros(0, dtype=np.int8)]

    matrix = np.asarray(model.Q, dtype=float)
    symmetric = 0.5 * (matrix + matrix.T)
    diagonal = np.diag(symmetric)
    field_scale = float(
        np.max(np.abs(diagonal) + 2.0 * np.sum(np.abs(symmetric), axis=1))
    )
    start_temperature = max(1.0, field_scale)
    end_temperature = max(1e-6, start_temperature * 1e-3)
    temperatures = np.geomspace(start_temperature, end_temperature, sweeps)
    rng = np.random.default_rng(seed)
    warm = _initial_vector(model, initial_sample)
    samples: list[np.ndarray] = []

    for read in range(num_samples):
        if warm is not None and read == 0:
            vector = warm.copy()
        elif warm is not None:
            vector = warm.copy()
            flips = max(1, n // 10)
            chosen = rng.choice(n, size=min(flips, n), replace=False)
            vector[chosen] = 1 - vector[chosen]
        else:
            vector = rng.integers(0, 2, size=n, dtype=np.int8)

        best = vector.copy()
        current_energy = qubo_energy(model, vector)
        best_energy = current_energy
        fields = diagonal + 2.0 * (symmetric @ vector - diagonal * vector)
        for temperature in temperatures:
            for index in rng.permutation(n):
                step = 1 - 2 * int(vector[index])
                delta = step * float(fields[index])
                if delta <= 0 or rng.random() < exp(-delta / float(temperature)):
                    vector[index] = 1 - vector[index]
                    current_energy += delta
                    fields += 2.0 * symmetric[:, index] * step
                    fields[index] -= 2.0 * diagonal[index] * step
                    if current_energy < best_energy:
                        best = vector.copy()
                        best_energy = current_energy
        samples.append(best)
    return samples


def _sample_dwave(
    model: QUBOModel,
    *,
    method: str,
    num_samples: int,
    allow_remote: bool,
    time_limit_seconds: float | None,
) -> tuple[list[np.ndarray], dict[str, object]]:
    if not allow_remote:
        raise QuantumSolverError(
            "Remote QPU access is disabled. Set allow_remote=True only after the "
            "challenge data owner approves sending model coefficients off-platform."
        )
    try:
        import dimod
        from dwave.system import DWaveSampler, EmbeddingComposite, LeapHybridSampler
    except ImportError as error:
        raise QuantumSolverError(
            "D-Wave sampling requires the optional 'qpu' dependencies"
        ) from error

    matrix = np.asarray(model.Q, dtype=float)
    linear = {index: float(matrix[index, index]) for index in range(len(matrix))}
    quadratic = {
        (i, j): float(matrix[i, j] + matrix[j, i])
        for i in range(len(matrix))
        for j in range(i + 1, len(matrix))
        if abs(matrix[i, j] + matrix[j, i]) > 0
    }
    # Integer labels prevent order, SKU, and DC identifiers from becoming remote labels.
    bqm = dimod.BinaryQuadraticModel(
        linear,
        quadratic,
        float(model.constant),
        dimod.BINARY,
    )
    if method == "dwave-hybrid":
        sampler = LeapHybridSampler()
        kwargs = {}
        if time_limit_seconds is not None:
            kwargs["time_limit"] = float(time_limit_seconds)
        response = sampler.sample(bqm, **kwargs)
    else:
        sampler = EmbeddingComposite(DWaveSampler())
        response = sampler.sample(bqm, num_reads=int(num_samples))

    samples: list[np.ndarray] = []
    for datum in response.data(fields=["sample", "energy"]):
        samples.append(
            np.asarray(
                [int(datum.sample[index]) for index in range(len(model.variable_names))],
                dtype=np.int8,
            )
        )
        if len(samples) >= num_samples:
            break
    if not samples:
        raise QuantumSolverError("D-Wave returned no samples")
    response_info = dict(getattr(response, "info", {}) or {})
    timing = {
        str(key): float(value)
        for key, value in dict(response_info.get("timing", {}) or {}).items()
        if isinstance(value, (int, float, np.integer, np.floating))
    }
    record_names = set(getattr(response.record.dtype, "names", ()) or ())
    chain_break = None
    if "chain_break_fraction" in record_names:
        chain_break = float(np.mean(response.record.chain_break_fraction))
    child = getattr(sampler, "child", sampler)
    solver = getattr(child, "solver", None)
    metadata: dict[str, object] = {
        "backend": method,
        "problem_id": response_info.get("problem_id"),
        "solver_id": getattr(solver, "id", None),
        "timing": timing,
        "mean_chain_break_fraction": chain_break,
        "returned_samples": len(samples),
    }
    return samples, metadata


def sample_qubo(
    model: QUBOModel,
    *,
    method: Literal[
        "exact",
        "random",
        "simulated_annealing",
        "dwave-qpu",
        "dwave-hybrid",
    ] = "exact",
    num_samples: int = 1000,
    sweeps: int = 500,
    seed: int = 0,
    max_exact_variables: int = 24,
    initial_sample: Mapping[str, int] | np.ndarray | list[int] | None = None,
    allow_remote: bool = False,
    time_limit_seconds: float | None = None,
) -> pd.DataFrame:
    """Return sampled bitstrings and energies using one common schema.

    Exact enumeration and simulated annealing are reproducible validation
    backends. D-Wave methods are optional execution adapters for the same QUBO;
    they are not enabled implicitly and do not claim quantum advantage.
    """

    n = len(model.variable_names)
    sampler_info: dict[str, object] = {"backend": method, "remote": False}
    if method == "exact":
        if n > max_exact_variables:
            raise QuantumSolverError(
                f"Exact QUBO enumeration requested for {n} variables; "
                f"limit is {max_exact_variables}"
            )
        bitstrings = [np.asarray(bits, dtype=np.int8) for bits in product((0, 1), repeat=n)]
    elif method == "random":
        if num_samples <= 0:
            raise QuantumSolverError("num_samples must be positive")
        rng = np.random.default_rng(seed)
        bitstrings = [
            rng.integers(0, 2, size=n, dtype=np.int8) for _ in range(num_samples)
        ]
    elif method == "simulated_annealing":
        bitstrings = _simulated_annealing_samples(
            model,
            num_samples=num_samples,
            sweeps=sweeps,
            seed=seed,
            initial_sample=initial_sample,
        )
    elif method in {"dwave-qpu", "dwave-hybrid"}:
        bitstrings, sampler_info = _sample_dwave(
            model,
            method=method,
            num_samples=num_samples,
            allow_remote=allow_remote,
            time_limit_seconds=time_limit_seconds,
        )
        sampler_info["remote"] = True
    else:
        raise QuantumSolverError(f"Unknown QUBO sampling method {method!r}")

    rows: list[dict[str, object]] = []
    for vector in bitstrings:
        rows.append(
            {
                "bitstring": "".join(map(str, vector.tolist())),
                "energy": qubo_energy(model, vector),
                "backend": method,
                **{
                    name: int(value)
                    for name, value in zip(model.variable_names, vector)
                },
            }
        )
    frame = pd.DataFrame(rows)
    frame = frame.sort_values(["energy", "bitstring"], kind="mergesort").reset_index(
        drop=True
    )
    frame.attrs["sampler_info"] = sampler_info
    return frame
