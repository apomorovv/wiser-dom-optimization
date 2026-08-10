"""Reproducible local samplers and optional privacy-gated hardware adapters.

``qaoa_statevector`` is a genuine gate-model algorithm simulation. It uses a
weight-one Dicke (W) state for each assignment group and a connected XY mixer,
so every ideal simulated state satisfies the one-hot assignment constraints.
IBM hardware is exposed separately as an execution target for the same QAOA
proposal method.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from itertools import product
from math import exp
from time import perf_counter
from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .qubo import qubo_energy
from .schemas import QUBOModel


class QuantumSolverError(RuntimeError):
    pass


IBM_MITIGATION_STRATEGIES = (
    "baseline",
    "dynamical_decoupling",
    "dd_measure_twirling",
)

_QAOA_PARAMETER_CACHE: dict[tuple[object, ...], tuple[float, ...]] = {}


def _qaoa_parameter_cache_key(
    model: QUBOModel,
    *,
    seed: int,
    layers: int,
    restarts: int,
    mixer_topology: str,
) -> tuple[object, ...]:
    digest = hashlib.sha256()
    digest.update("\0".join(model.variable_names).encode("utf-8"))
    digest.update(np.asarray(model.Q, dtype=np.float64).tobytes(order="C"))
    digest.update(np.asarray([float(model.constant)], dtype=np.float64).tobytes())
    return (digest.hexdigest(), int(seed), int(layers), int(restarts), mixer_topology)


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def ibm_sampler_options(
    strategy: str,
    *,
    shots: int,
    max_execution_time_seconds: float | None = None,
) -> dict[str, object]:
    """Return explicit SamplerV2 options for a matched mitigation ablation."""

    normalized = str(strategy).strip().lower()
    if normalized not in IBM_MITIGATION_STRATEGIES:
        raise QuantumSolverError(
            f"IBM mitigation strategy must be one of {IBM_MITIGATION_STRATEGIES}"
        )
    if shots <= 0:
        raise QuantumSolverError("IBM shots must be positive")
    use_dd = normalized in {"dynamical_decoupling", "dd_measure_twirling"}
    use_measure_twirling = normalized == "dd_measure_twirling"
    options: dict[str, object] = {
        "default_shots": int(shots),
        "dynamical_decoupling": {
            "enable": use_dd,
            "sequence_type": "XpXm",
            "scheduling_method": "alap",
        },
        "twirling": {
            "enable_gates": False,
            "enable_measure": use_measure_twirling,
        },
        "environment": {
            "job_tags": ["wiser-dom", "synthetic-only", normalized],
        },
    }
    if max_execution_time_seconds is not None:
        options["max_execution_time"] = max(
            1, min(10_800, int(np.ceil(max_execution_time_seconds)))
        )
    return options


def _wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    """Return a 95% Wilson score interval for one observed proportion."""

    if trials <= 0:
        return 0.0, 1.0
    z = 1.959963984540054
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    radius = (
        z
        * np.sqrt(proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials**2))
        / denominator
    )
    return max(0.0, float(center - radius)), min(1.0, float(center + radius))


def _sample_quality_statistics(
    model: QUBOModel,
    samples: list[np.ndarray],
    *,
    max_feasible_states: int,
) -> dict[str, float | int | None]:
    """Compare raw samples with exact feasible and uniform-feasible references."""

    feasible, groups = _feasible_bitstrings(
        model,
        max_feasible_states=max_feasible_states,
    )
    optimum = float(min(qubo_energy(model, vector) for vector in feasible))
    feasible_energies = np.asarray(
        [qubo_energy(model, vector) for vector in feasible], dtype=float
    )
    span = max(1.0, float(np.ptp(feasible_energies)))
    raw_energies = np.asarray(
        [qubo_energy(model, vector) for vector in samples], dtype=float
    )
    one_hot = np.asarray(
        [all(int(vector[list(group)].sum()) == 1 for group in groups) for vector in samples],
        dtype=bool,
    )
    feasible_raw = raw_energies[one_hot]
    tolerance = 1e-9 * max(1.0, abs(optimum))
    best = float(np.min(feasible_raw)) if len(feasible_raw) else None
    feasible_gaps = (feasible_raw - optimum) / span
    raw_optimal = one_hot & (np.abs(raw_energies - optimum) <= tolerance)
    raw_near_optimal = one_hot & ((raw_energies - optimum) / span <= 0.01 + 1e-12)
    optimum_successes = int(raw_optimal.sum())
    feasible_successes = int(one_hot.sum())
    optimum_low, optimum_high = _wilson_interval(optimum_successes, len(samples))
    one_hot_low, one_hot_high = _wilson_interval(feasible_successes, len(samples))
    exact_gaps = (feasible_energies - optimum) / span
    return {
        "sample_raw_one_hot_rate": float(one_hot.mean()) if len(one_hot) else 0.0,
        "sample_raw_one_hot_rate_ci95_low": one_hot_low,
        "sample_raw_one_hot_rate_ci95_high": one_hot_high,
        "sample_feasible_shots": feasible_successes,
        "sample_qubo_optimal_hit_rate": float(np.mean(raw_optimal)) if len(samples) else 0.0,
        "sample_qubo_optimal_hit_rate_ci95_low": optimum_low,
        "sample_qubo_optimal_hit_rate_ci95_high": optimum_high,
        "sample_qubo_optimal_hit_rate_given_feasible": float(
            np.mean(np.abs(feasible_raw - optimum) <= tolerance)
        )
        if len(feasible_raw)
        else 0.0,
        "sample_qubo_near_optimal_1pct_rate": (
            float(np.mean(raw_near_optimal)) if len(samples) else 0.0
        ),
        "sample_best_feasible_energy": best,
        "sample_best_feasible_normalized_gap": (
            None if best is None else float((best - optimum) / span)
        ),
        "sample_mean_feasible_normalized_gap": (
            float(np.mean(feasible_gaps)) if len(feasible_gaps) else None
        ),
        "sample_mean_raw_energy": (
            float(np.mean(raw_energies)) if len(raw_energies) else None
        ),
        "sample_mean_feasible_energy": (
            float(np.mean(feasible_raw)) if len(feasible_raw) else None
        ),
        "exact_best_feasible_qubo_energy": optimum,
        "exact_feasible_state_count": len(feasible),
        "uniform_feasible_optimal_rate": float(np.mean(exact_gaps <= 1e-12)),
        "uniform_feasible_near_optimal_1pct_rate": float(
            np.mean(exact_gaps <= 0.01 + 1e-12)
        ),
        "uniform_feasible_mean_normalized_gap": float(np.mean(exact_gaps)),
    }


def _hardware_sample_statistics(
    model: QUBOModel,
    samples: list[np.ndarray],
    *,
    max_feasible_states: int,
) -> dict[str, float | int | None]:
    """Return common sample metrics plus legacy hardware-prefixed aliases."""

    common = _sample_quality_statistics(
        model,
        samples,
        max_feasible_states=max_feasible_states,
    )
    aliases = {
        f"hardware_{key.removeprefix('sample_')}": value
        for key, value in common.items()
        if key.startswith("sample_")
    }
    return {**common, **aliases}


def _duration_seconds(timestamps: dict[str, object], start: str, end: str) -> float | None:
    start_value = timestamps.get(start)
    end_value = timestamps.get(end)
    if start_value is None or end_value is None:
        return None
    try:
        return float((pd.Timestamp(end_value) - pd.Timestamp(start_value)).total_seconds())
    except (TypeError, ValueError):
        return None


def _ibm_job_timing(job: object, wall_seconds: float) -> dict[str, object]:
    """Normalize IBM Runtime wall, queue, execution, and quantum usage timing."""

    try:
        metrics = dict(job.metrics() or {})
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        metrics = {}
    timestamps = dict(metrics.get("timestamps", {}) or {})
    queue_seconds = _duration_seconds(timestamps, "created", "running")
    execution_seconds = _duration_seconds(timestamps, "running", "finished")
    turnaround_seconds = _duration_seconds(timestamps, "created", "finished")

    usage = dict(metrics.get("usage", {}) or {})
    estimation = getattr(job, "usage_estimation", {})
    if callable(estimation):
        try:
            estimation = estimation()
        except (OSError, RuntimeError, TypeError, ValueError):
            estimation = {}
    estimation = dict(estimation or {}) if isinstance(estimation, dict) else {}
    quantum_seconds = estimation.get("quantum_seconds")
    if quantum_seconds is None:
        quantum_seconds = usage.get("quantum_seconds", usage.get("seconds"))
    return {
        "hardware_wall_seconds": float(wall_seconds),
        "hardware_created_at": (
            None if timestamps.get("created") is None else str(timestamps["created"])
        ),
        "hardware_running_at": (
            None if timestamps.get("running") is None else str(timestamps["running"])
        ),
        "hardware_finished_at": (
            None if timestamps.get("finished") is None else str(timestamps["finished"])
        ),
        "hardware_queue_seconds": queue_seconds,
        "hardware_execution_seconds": execution_seconds,
        "hardware_turnaround_seconds": turnaround_seconds,
        "hardware_quantum_seconds": (
            None if quantum_seconds is None else float(quantum_seconds)
        ),
    }


def _one_hot_group_indices(model: QUBOModel) -> tuple[tuple[int, ...], ...]:
    raw_groups = model.metadata.get("one_hot_groups")
    if not raw_groups:
        raise QuantumSolverError(
            "This sampler requires QUBO metadata['one_hot_groups']"
        )
    name_to_index = {name: index for index, name in enumerate(model.variable_names)}
    try:
        groups = tuple(
            tuple(name_to_index[str(name)] for name in group) for group in raw_groups
        )
    except KeyError as error:
        raise QuantumSolverError(
            f"One-hot metadata references unknown variable {error.args[0]!r}"
        ) from error
    flattened = [index for group in groups for index in group]
    if any(not group for group in groups) or sorted(flattened) != list(
        range(len(model.variable_names))
    ):
        raise QuantumSolverError(
            "One-hot groups must partition every QUBO variable exactly once"
        )
    return groups


def _feasible_bitstrings(
    model: QUBOModel,
    *,
    max_feasible_states: int,
) -> tuple[list[np.ndarray], tuple[tuple[int, ...], ...]]:
    groups = _one_hot_group_indices(model)
    state_count = int(np.prod([len(group) for group in groups], dtype=object))
    if state_count > max_feasible_states:
        raise QuantumSolverError(
            f"Feasible-subspace simulation needs {state_count:,} states; "
            f"limit is {max_feasible_states:,}"
        )
    vectors: list[np.ndarray] = []
    for choices in product(*(range(len(group)) for group in groups)):
        vector = np.zeros(len(model.variable_names), dtype=np.int8)
        for group, choice in zip(groups, choices):
            vector[group[choice]] = 1
        vectors.append(vector)
    return vectors, groups


def _ring_edges(size: int) -> tuple[tuple[int, int], ...]:
    if size <= 1:
        return ()
    if size == 2:
        return ((0, 1),)
    return tuple((index, (index + 1) % size) for index in range(size))


def _path_edges(size: int) -> tuple[tuple[int, int], ...]:
    if size <= 1:
        return ()
    return tuple((index, index + 1) for index in range(size - 1))


def _mixer_edges(size: int, topology: str) -> tuple[tuple[int, int], ...]:
    normalized = str(topology).strip().lower()
    if normalized == "path":
        return _path_edges(size)
    if normalized == "ring":
        return _ring_edges(size)
    raise QuantumSolverError("QAOA mixer topology must be 'path' or 'ring'")


def _xy_edge_unitary(size: int, edge: tuple[int, int], beta: float) -> np.ndarray:
    """XY edge evolution restricted to the single-excitation subspace."""

    i, j = edge
    unitary = np.eye(size, dtype=complex)
    unitary[i, i] = unitary[j, j] = np.cos(beta)
    unitary[i, j] = unitary[j, i] = -1j * np.sin(beta)
    return unitary


def _apply_axis_unitary(
    state: np.ndarray,
    unitary: np.ndarray,
    *,
    axis: int,
    shape: tuple[int, ...],
) -> np.ndarray:
    tensor = state.reshape(shape)
    moved = np.moveaxis(tensor, axis, 0)
    updated = unitary @ moved.reshape(shape[axis], -1)
    restored = np.moveaxis(updated.reshape(moved.shape), 0, axis)
    return restored.reshape(-1)


def _qaoa_statevector_samples(
    model: QUBOModel,
    *,
    num_samples: int,
    seed: int,
    layers: int,
    restarts: int,
    max_feasible_states: int,
    mixer_topology: str,
    parameters: tuple[float, ...] | list[float] | np.ndarray | None = None,
) -> tuple[list[np.ndarray], dict[str, object]]:
    if num_samples <= 0 or layers <= 0 or restarts <= 0:
        raise QuantumSolverError("QAOA samples, layers, and restarts must be positive")
    feasible, groups = _feasible_bitstrings(
        model,
        max_feasible_states=max_feasible_states,
    )
    shape = tuple(len(group) for group in groups)
    energies = np.asarray([qubo_energy(model, vector) for vector in feasible])
    energy_span = float(np.ptp(energies))
    if energy_span > 0:
        phase_energies = (energies - float(np.mean(energies))) / energy_span
    else:
        phase_energies = np.zeros_like(energies)

    mixer_edges = tuple(_mixer_edges(size, mixer_topology) for size in shape)

    initial = np.full(len(feasible), 1.0 / np.sqrt(len(feasible)), dtype=complex)

    def statevector(parameters: np.ndarray) -> np.ndarray:
        state = initial.copy()
        gammas = parameters[:layers]
        betas = parameters[layers:]
        for gamma, beta in zip(gammas, betas):
            state *= np.exp(-1j * float(gamma) * phase_energies)
            # Use the same ordered first-order schedule as the hardware RXX/RYY
            # circuit, avoiding a simulator/hardware mixer mismatch.
            for axis, edges in enumerate(mixer_edges):
                for edge in edges:
                    state = _apply_axis_unitary(
                        state,
                        _xy_edge_unitary(shape[axis], edge, float(beta)),
                        axis=axis,
                        shape=shape,
                    )
        return state

    def expected_energy(parameters: np.ndarray) -> float:
        probabilities = np.abs(statevector(parameters)) ** 2
        return float(probabilities @ energies)

    rng = np.random.default_rng(seed)
    if parameters is None:
        best_result = None
        bounds = [(0.0, 2.0 * np.pi)] * layers + [(0.0, np.pi)] * layers
        starts = [np.full(2 * layers, 0.5)]
        starts.extend(
            np.concatenate(
                [
                    rng.uniform(0.0, 2.0 * np.pi, size=layers),
                    rng.uniform(0.0, np.pi, size=layers),
                ]
            )
            for _ in range(restarts - 1)
        )
        for initial_parameters in starts:
            result = minimize(
                expected_energy,
                initial_parameters,
                method="Powell",
                bounds=bounds,
                options={"maxiter": 80, "xtol": 1e-4, "ftol": 1e-7},
            )
            if best_result is None or float(result.fun) < float(best_result.fun):
                best_result = result
        assert best_result is not None
        optimized_parameters = np.asarray(best_result.x, dtype=float)
        optimizer_success: bool | None = bool(best_result.success)
        parameter_source = "locally optimized"
    else:
        optimized_parameters = np.asarray(parameters, dtype=float)
        if optimized_parameters.shape != (2 * layers,) or not np.isfinite(
            optimized_parameters
        ).all():
            raise QuantumSolverError(
                f"QAOA parameters must contain {2 * layers} finite values"
            )
        optimizer_success = None
        parameter_source = "provided and reused"
    optimized_expected_energy = expected_energy(optimized_parameters)
    probabilities = np.abs(statevector(optimized_parameters)) ** 2
    statevector_norm = float(probabilities.sum())
    if not np.isclose(statevector_norm, 1.0, atol=1e-10):
        raise QuantumSolverError(
            f"QAOA statevector normalization failed: norm={statevector_norm}"
        )
    probabilities = probabilities / statevector_norm
    sampled_indices = rng.choice(
        len(feasible),
        size=num_samples,
        replace=True,
        p=probabilities,
    )
    samples = [feasible[int(index)].copy() for index in sampled_indices]
    info: dict[str, object] = {
        "backend": "qaoa_statevector",
        "remote": False,
        "algorithm": (
            "QAOA with product W/Dicke(1) initial state and "
            f"XY {mixer_topology} mixers"
        ),
        "constraint_encoding": "feasible one-hot subspace",
        "mixer_topology": str(mixer_topology),
        "layers": int(layers),
        "optimizer_restarts": int(restarts),
        "feasible_state_dimension": len(feasible),
        "optimized_expected_energy": float(optimized_expected_energy),
        "uniform_expected_energy": float(np.mean(energies)),
        "best_feasible_energy": float(np.min(energies)),
        "statevector_probability_sum": float(probabilities.sum()),
        "optimized_parameters": [float(value) for value in optimized_parameters],
        "optimizer_success": optimizer_success,
        "parameter_source": parameter_source,
        "parameters_reused": parameters is not None,
        "initial_sample_used": False,
        "returned_samples": len(samples),
    }
    return samples, info


def optimize_qaoa_parameters(
    model: QUBOModel,
    *,
    seed: int = 0,
    layers: int = 1,
    restarts: int = 4,
    max_feasible_states: int = 65_536,
    mixer_topology: str = "path",
) -> tuple[tuple[float, ...], dict[str, object]]:
    """Optimize one reusable QAOA parameter vector for a fixed QUBO.

    Hardware ablations should call this once per depth and reuse the returned
    vector across mitigation strategies and repetitions. This prevents local
    classical angle search from being counted repeatedly as QPU runtime.
    """

    _, info = _qaoa_statevector_samples(
        model,
        num_samples=1,
        seed=seed,
        layers=layers,
        restarts=restarts,
        max_feasible_states=max_feasible_states,
        mixer_topology=mixer_topology,
    )
    return tuple(float(value) for value in info["optimized_parameters"]), info


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


def _append_linear_w_state(circuit: object, group: tuple[int, ...]) -> None:
    """Append a deterministic linear-size W-state preparation circuit.

    The construction uses one controlled rotation and one CNOT per added qubit,
    avoiding the exponentially described generic ``StatePreparation`` gate.
    """

    if not group:
        raise QuantumSolverError("A W-state group cannot be empty")
    circuit.x(group[-1])
    for local_index in range(len(group) - 1, 0, -1):
        angle = 2.0 * np.arccos(np.sqrt(1.0 / (local_index + 1.0)))
        circuit.cry(float(angle), group[local_index], group[local_index - 1])
        circuit.cx(group[local_index - 1], group[local_index])


def _sample_ibm_qpu(
    model: QUBOModel,
    *,
    num_samples: int,
    seed: int,
    allow_remote: bool,
    layers: int,
    restarts: int,
    max_feasible_states: int,
    time_limit_seconds: float | None,
    backend_name: str | None,
    mitigation_strategy: str,
    transpiler_optimization_level: int,
    transpiler_trials: int,
    transpiler_seed: int | None,
    mixer_topology: str,
    qaoa_parameters: tuple[float, ...] | list[float] | np.ndarray | None,
) -> tuple[list[np.ndarray], dict[str, object]]:
    """Run constraint-preserving QAOA on a configured IBM Quantum account.

    Angles are optimized locally with the matching feasible-subspace simulator.
    Only the compiled synthetic or approved QUBO circuit is sent remotely.
    """

    if not allow_remote:
        raise QuantumSolverError(
            "Remote IBM QPU access is disabled. Set allow_remote=True only after "
            "the data owner approves sending a circuit that encodes model coefficients."
    )
    try:
        from qiskit import QuantumCircuit
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    except ImportError as error:
        raise QuantumSolverError(
            "IBM QPU sampling requires the optional 'ibm' dependencies"
        ) from error

    cache_key = _qaoa_parameter_cache_key(
        model,
        seed=seed,
        layers=layers,
        restarts=restarts,
        mixer_topology=mixer_topology,
    )
    cached_parameters = qaoa_parameters
    cache_hit = False
    if cached_parameters is None and cache_key in _QAOA_PARAMETER_CACHE:
        cached_parameters = _QAOA_PARAMETER_CACHE[cache_key]
        cache_hit = True
    angle_start = perf_counter()
    _, local_info = _qaoa_statevector_samples(
        model,
        num_samples=1,
        seed=seed,
        layers=layers,
        restarts=restarts,
        max_feasible_states=max_feasible_states,
        mixer_topology=mixer_topology,
        parameters=cached_parameters,
    )
    angle_optimization_seconds = perf_counter() - angle_start
    if cached_parameters is None:
        _QAOA_PARAMETER_CACHE[cache_key] = tuple(
            float(value) for value in local_info["optimized_parameters"]
        )
    circuit_start = perf_counter()
    parameters = np.asarray(local_info["optimized_parameters"], dtype=float)
    gammas = parameters[:layers]
    betas = parameters[layers:]
    groups = _one_hot_group_indices(model)
    n = len(model.variable_names)
    circuit = QuantumCircuit(n)

    # Product of weight-one Dicke states: one excitation per assignment group.
    for group in groups:
        _append_linear_w_state(circuit, group)

    matrix = np.asarray(model.Q, dtype=float)
    feasible, _ = _feasible_bitstrings(
        model,
        max_feasible_states=max_feasible_states,
    )
    feasible_energies = np.asarray([qubo_energy(model, value) for value in feasible])
    energy_span = float(np.ptp(feasible_energies))
    if energy_span <= 0:
        energy_span = 1.0
    linear_z = -np.diag(matrix) / (2.0 * energy_span)
    pair_zz: dict[tuple[int, int], float] = {}
    for i in range(n):
        for j in range(i + 1, n):
            weight = float(matrix[i, j] + matrix[j, i]) / energy_span
            if abs(weight) <= 1e-15:
                continue
            linear_z[i] -= weight / 4.0
            linear_z[j] -= weight / 4.0
            pair_zz[(i, j)] = weight / 4.0

    for gamma, beta in zip(gammas, betas):
        for index, coefficient in enumerate(linear_z):
            if abs(coefficient) > 1e-15:
                circuit.rz(2.0 * float(gamma) * float(coefficient), index)
        for (i, j), coefficient in pair_zz.items():
            circuit.rzz(2.0 * float(gamma) * coefficient, i, j)
        for group in groups:
            for local_i, local_j in _mixer_edges(len(group), mixer_topology):
                i, j = group[local_i], group[local_j]
                circuit.rxx(float(beta), i, j)
                circuit.ryy(float(beta), i, j)
    circuit.measure_all()
    circuit_construction_seconds = perf_counter() - circuit_start

    if transpiler_optimization_level not in {0, 1, 2, 3}:
        raise QuantumSolverError("IBM transpiler optimization level must be 0, 1, 2, or 3")
    if transpiler_trials <= 0:
        raise QuantumSolverError("IBM transpiler trials must be positive")
    backend_selection_start = perf_counter()
    try:
        service = QiskitRuntimeService()
        if backend_name:
            backend = service.backend(str(backend_name), use_fractional_gates=False)
            status = backend.status()
            if not bool(getattr(status, "operational", False)):
                raise QuantumSolverError(
                    f"Requested IBM processor target (Qiskit backend) {backend_name!r} "
                    "is not operational"
                )
            if int(getattr(backend, "num_qubits", 0)) < n:
                raise QuantumSolverError(
                    f"Requested IBM processor target (Qiskit backend) {backend_name!r} "
                    f"has fewer than {n} qubits"
                )
        else:
            backend = service.least_busy(
                operational=True,
                simulator=False,
                min_num_qubits=n,
                use_fractional_gates=False,
            )
            status = backend.status()
    except QuantumSolverError:
        raise
    except Exception as error:
        raise QuantumSolverError(
            "IBM processor-target selection through Qiskit failed: "
            f"{type(error).__name__}: {error}"
        ) from error
    backend_selection_seconds = perf_counter() - backend_selection_start

    transpiled: list[tuple[tuple[int, int, int], int, object]] = []
    transpilation_errors: list[str] = []
    base_transpiler_seed = int(seed if transpiler_seed is None else transpiler_seed)
    transpilation_start = perf_counter()
    for trial in range(transpiler_trials):
        trial_seed = int(base_transpiler_seed + 104_729 * trial)
        try:
            pass_manager = generate_preset_pass_manager(
                backend=backend,
                optimization_level=transpiler_optimization_level,
                seed_transpiler=trial_seed,
            )
            candidate = pass_manager.run(circuit)
            two_qubit_gates = sum(
                1 for instruction in candidate.data if len(instruction.qubits) == 2
            )
            transpiled.append(
                (
                    (two_qubit_gates, int(candidate.depth()), int(candidate.size())),
                    trial_seed,
                    candidate,
                )
            )
        # Qiskit pass plugins can raise backend-specific exceptions that are not
        # part of a stable public hierarchy; one failed seed must not abort trials.
        except Exception as error:  # noqa: BLE001
            transpilation_errors.append(f"{type(error).__name__}: {error}")
    if not transpiled:
        detail = transpilation_errors[0] if transpilation_errors else "unknown error"
        raise QuantumSolverError(f"IBM transpilation failed for every trial: {detail}")
    transpilation_seconds = perf_counter() - transpilation_start
    score, selected_transpiler_seed, isa_circuit = min(
        transpiled, key=lambda item: item[0]
    )
    two_qubit_gates, transpiled_depth, transpiled_size = score
    try:
        two_qubit_depth = int(
            isa_circuit.depth(filter_function=lambda item: len(item.qubits) == 2)
        )
    except (AttributeError, TypeError, ValueError):
        two_qubit_depth = None
    try:
        layout = getattr(isa_circuit, "layout", None)
        final_index_layout = layout.final_index_layout
        try:
            mapped = final_index_layout(filter_ancillas=True)
        except TypeError:
            mapped = final_index_layout()
        physical_qubits = list(mapped)[:n]
        physical_qubit_mapping = ",".join(str(int(index)) for index in physical_qubits)
    except (AttributeError, TypeError, ValueError):
        physical_qubit_mapping = None
    try:
        properties_method = getattr(backend, "properties", None)
        properties = properties_method() if callable(properties_method) else None
        calibration_value = getattr(properties, "last_update_date", None)
        calibration_last_update_at = (
            None if calibration_value is None else str(calibration_value)
        )
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        calibration_last_update_at = None

    options = ibm_sampler_options(
        mitigation_strategy,
        shots=int(num_samples),
        max_execution_time_seconds=time_limit_seconds,
    )
    try:
        sampler = SamplerV2(mode=backend, options=options)
        submit_start = perf_counter()
        job = sampler.run([isa_circuit], shots=int(num_samples))
        primitive_submit_seconds = perf_counter() - submit_start
        wait_start = perf_counter()
        publication = job.result()[0]
        primitive_wait_seconds = perf_counter() - wait_start
        wall_seconds = primitive_submit_seconds + primitive_wait_seconds
    except Exception as error:
        raise QuantumSolverError(
            f"IBM Runtime execution failed: {type(error).__name__}: {error}"
        ) from error
    decode_start = perf_counter()
    counts = publication.data.meas.get_counts()
    samples: list[np.ndarray] = []
    for remote_bits, count in counts.items():
        # Qiskit displays classical bits from highest to lowest index.
        clean = str(remote_bits).replace(" ", "")
        vector = np.asarray([int(value) for value in clean[::-1]], dtype=np.int8)
        samples.extend(vector.copy() for _ in range(int(count)))
    if not samples:
        raise QuantumSolverError("IBM Runtime returned no samples")
    decode_seconds = perf_counter() - decode_start
    backend_name = getattr(backend, "name", "unknown")
    if callable(backend_name):
        backend_name = backend_name()
    sample_statistics = _hardware_sample_statistics(
        model,
        samples,
        max_feasible_states=max_feasible_states,
    )
    timing = _ibm_job_timing(job, wall_seconds)
    metadata: dict[str, object] = {
        **local_info,
        "backend": "ibm-qpu",
        "remote": True,
        "provider": "IBM Quantum",
        "backend_name": str(backend_name),
        "job_id": str(job.job_id()),
        "returned_samples": len(samples),
        "requested_time_limit_seconds": time_limit_seconds,
        "backend_pending_jobs_at_selection": int(getattr(status, "pending_jobs", 0)),
        "backend_num_qubits": int(getattr(backend, "num_qubits", 0)),
        "logical_qubits": n,
        "physical_qubit_mapping": physical_qubit_mapping,
        "calibration_last_update_at": calibration_last_update_at,
        "qiskit_version": _package_version("qiskit"),
        "qiskit_ibm_runtime_version": _package_version("qiskit-ibm-runtime"),
        "angle_seed": int(seed),
        "qaoa_parameter_source": local_info["parameter_source"],
        "qaoa_parameters_reused": local_info["parameters_reused"],
        "qaoa_parameter_cache_hit": cache_hit,
        "w_state_preparation": "linear controlled-rotation construction",
        "mixer_topology": mixer_topology,
        "logical_w_preparation_two_qubit_gates": int(
            2 * sum(max(0, len(group) - 1) for group in groups)
        ),
        "logical_mixer_two_qubit_gates": int(
            2
            * layers
            * sum(len(_mixer_edges(len(group), mixer_topology)) for group in groups)
        ),
        "mitigation_strategy": mitigation_strategy,
        "transpiler_optimization_level": transpiler_optimization_level,
        "transpiler_trials": transpiler_trials,
        "transpiler_base_seed": base_transpiler_seed,
        "transpiler_failed_trials": len(transpilation_errors),
        "selected_transpiler_seed": selected_transpiler_seed,
        "transpiled_depth": transpiled_depth,
        "transpiled_size": transpiled_size,
        "transpiled_two_qubit_gates": two_qubit_gates,
        "transpiled_two_qubit_depth": two_qubit_depth,
        "angle_optimization_seconds": angle_optimization_seconds,
        "circuit_construction_seconds": circuit_construction_seconds,
        "backend_selection_seconds": backend_selection_seconds,
        "transpilation_seconds": transpilation_seconds,
        "primitive_submit_seconds": primitive_submit_seconds,
        "primitive_wait_seconds": primitive_wait_seconds,
        "decode_seconds": decode_seconds,
        **timing,
        **sample_statistics,
    }
    return samples[:num_samples], metadata


def sample_qubo(
    model: QUBOModel,
    *,
    method: Literal[
        "exact",
        "exact_feasible",
        "random",
        "simulated_annealing",
        "qaoa_statevector",
        "ibm-qpu",
    ] = "exact",
    num_samples: int = 1000,
    sweeps: int = 500,
    seed: int = 0,
    max_exact_variables: int = 18,
    initial_sample: Mapping[str, int] | np.ndarray | list[int] | None = None,
    allow_remote: bool = False,
    time_limit_seconds: float | None = None,
    qaoa_layers: int = 1,
    qaoa_restarts: int = 4,
    qaoa_mixer_topology: Literal["path", "ring"] = "path",
    qaoa_parameters: tuple[float, ...] | list[float] | np.ndarray | None = None,
    qaoa_readout_bitflip_probability: float = 0.0,
    max_feasible_states: int = 65_536,
    ibm_backend_name: str | None = None,
    ibm_mitigation_strategy: str = "baseline",
    ibm_transpiler_optimization_level: int = 3,
    ibm_transpiler_trials: int = 4,
    ibm_transpiler_seed: int | None = None,
) -> pd.DataFrame:
    """Return sampled bitstrings and energies using one common schema.

    Exact enumeration and simulated annealing are reproducible classical
    proposal methods. ``qaoa_statevector`` locally simulates a constraint-
    preserving gate-model circuit. IBM hardware is an optional execution target;
    the algorithm, execution target, and downstream exact validator are reported
    separately.
    """

    n = len(model.variable_names)
    if not 0.0 <= qaoa_readout_bitflip_probability <= 1.0:
        raise QuantumSolverError(
            "qaoa_readout_bitflip_probability must be between zero and one"
        )
    _mixer_edges(2, qaoa_mixer_topology)
    sampler_info: dict[str, object] = {"backend": method, "remote": False}
    if method == "exact":
        if n > max_exact_variables:
            raise QuantumSolverError(
                f"Exact QUBO enumeration requested for {n} variables; "
                f"limit is {max_exact_variables}"
            )
        bitstrings = [np.asarray(bits, dtype=np.int8) for bits in product((0, 1), repeat=n)]
    elif method == "exact_feasible":
        bitstrings, groups = _feasible_bitstrings(
            model,
            max_feasible_states=max_feasible_states,
        )
        sampler_info.update(
            {
                "constraint_encoding": "feasible one-hot subspace",
                "one_hot_group_count": len(groups),
                "feasible_state_dimension": len(bitstrings),
            }
        )
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
    elif method == "qaoa_statevector":
        bitstrings, sampler_info = _qaoa_statevector_samples(
            model,
            num_samples=num_samples,
            seed=seed,
            layers=qaoa_layers,
            restarts=qaoa_restarts,
            max_feasible_states=max_feasible_states,
            mixer_topology=qaoa_mixer_topology,
            parameters=qaoa_parameters,
        )
        if qaoa_readout_bitflip_probability > 0:
            readout_rng = np.random.default_rng(seed + 1_000_003)
            bitstrings = [
                np.bitwise_xor(
                    vector,
                    readout_rng.random(n) < qaoa_readout_bitflip_probability,
                ).astype(np.int8)
                for vector in bitstrings
            ]
        sampler_info.update(
            {
                "readout_bitflip_probability": qaoa_readout_bitflip_probability,
                "noise_scope": (
                    "independent symmetric measurement bit flips after the ideal "
                    "statevector; not gate, decoherence, or hardware noise"
                ),
            }
        )
    elif method == "ibm-qpu":
        bitstrings, sampler_info = _sample_ibm_qpu(
            model,
            num_samples=num_samples,
            seed=seed,
            allow_remote=allow_remote,
            layers=qaoa_layers,
            restarts=qaoa_restarts,
            max_feasible_states=max_feasible_states,
            time_limit_seconds=time_limit_seconds,
            backend_name=ibm_backend_name,
            mitigation_strategy=ibm_mitigation_strategy,
            transpiler_optimization_level=ibm_transpiler_optimization_level,
            transpiler_trials=ibm_transpiler_trials,
            transpiler_seed=ibm_transpiler_seed,
            mixer_topology=qaoa_mixer_topology,
            qaoa_parameters=qaoa_parameters,
        )
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
    if model.metadata.get("one_hot_groups"):
        sampler_info.update(
            _sample_quality_statistics(
                model,
                bitstrings,
                max_feasible_states=max_feasible_states,
            )
        )
    frame.attrs["sampler_info"] = sampler_info
    return frame
