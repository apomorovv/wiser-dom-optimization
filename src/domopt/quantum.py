"""Reproducible local samplers and optional privacy-gated hardware adapters.

``qaoa_statevector`` is a genuine gate-model algorithm simulation.  It uses a
weight-one Dicke (W) state for each assignment group and an XY ring mixer, so
every simulated state satisfies the one-hot assignment constraints.  It is not
a QPU run and no claim of quantum advantage follows from it.
"""

from __future__ import annotations

from collections.abc import Mapping
from itertools import product
from math import exp
from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .qubo import qubo_energy
from .schemas import QUBOModel


class QuantumSolverError(RuntimeError):
    pass


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

    mixer_edges = tuple(_ring_edges(size) for size in shape)

    initial = np.full(len(feasible), 1.0 / np.sqrt(len(feasible)), dtype=complex)

    def statevector(parameters: np.ndarray) -> np.ndarray:
        state = initial.copy()
        gammas = parameters[:layers]
        betas = parameters[layers:]
        for gamma, beta in zip(gammas, betas):
            state *= np.exp(-1j * float(gamma) * phase_energies)
            # Use the same ordered first-order ring schedule as the hardware
            # RXX/RYY circuit, avoiding a simulator/hardware mixer mismatch.
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
    probabilities = np.abs(statevector(np.asarray(best_result.x, dtype=float))) ** 2
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
        "algorithm": "QAOA with product W/Dicke(1) initial state and XY ring mixers",
        "constraint_encoding": "feasible one-hot subspace",
        "layers": int(layers),
        "optimizer_restarts": int(restarts),
        "feasible_state_dimension": len(feasible),
        "optimized_expected_energy": float(best_result.fun),
        "uniform_expected_energy": float(np.mean(energies)),
        "best_feasible_energy": float(np.min(energies)),
        "statevector_probability_sum": float(probabilities.sum()),
        "optimized_parameters": [float(value) for value in best_result.x],
        "optimizer_success": bool(best_result.success),
        "initial_sample_used": False,
        "returned_samples": len(samples),
    }
    return samples, info


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
        from qiskit.circuit.library import StatePreparation
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    except ImportError as error:
        raise QuantumSolverError(
            "IBM QPU sampling requires the optional 'ibm' dependencies"
        ) from error

    _, local_info = _qaoa_statevector_samples(
        model,
        num_samples=1,
        seed=seed,
        layers=layers,
        restarts=restarts,
        max_feasible_states=max_feasible_states,
    )
    parameters = np.asarray(local_info["optimized_parameters"], dtype=float)
    gammas = parameters[:layers]
    betas = parameters[layers:]
    groups = _one_hot_group_indices(model)
    n = len(model.variable_names)
    if max(map(len, groups), default=0) > 12:
        raise QuantumSolverError(
            "IBM W-state preparation is limited to 12 variables per one-hot group; "
            "reduce max_candidates_per_order or provide a specialized preparation circuit"
        )
    circuit = QuantumCircuit(n)

    # Product of weight-one Dicke states: one excitation per assignment group.
    for group in groups:
        amplitudes = np.zeros(2 ** len(group), dtype=complex)
        for local_index in range(len(group)):
            amplitudes[1 << local_index] = 1.0 / np.sqrt(len(group))
        circuit.append(StatePreparation(amplitudes), list(group))

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
            for local_i, local_j in _ring_edges(len(group)):
                i, j = group[local_i], group[local_j]
                circuit.rxx(float(beta), i, j)
                circuit.ryy(float(beta), i, j)
    circuit.measure_all()

    service = QiskitRuntimeService()
    backend = service.least_busy(
        operational=True,
        simulator=False,
        min_num_qubits=n,
    )
    pass_manager = generate_preset_pass_manager(
        backend=backend,
        optimization_level=3,
    )
    isa_circuit = pass_manager.run(circuit)
    sampler = SamplerV2(mode=backend)
    job = sampler.run([isa_circuit], shots=int(num_samples))
    publication = job.result()[0]
    counts = publication.data.meas.get_counts()
    samples: list[np.ndarray] = []
    for remote_bits, count in counts.items():
        # Qiskit displays classical bits from highest to lowest index.
        clean = str(remote_bits).replace(" ", "")
        vector = np.asarray([int(value) for value in clean[::-1]], dtype=np.int8)
        samples.extend(vector.copy() for _ in range(int(count)))
    if not samples:
        raise QuantumSolverError("IBM Runtime returned no samples")
    backend_name = getattr(backend, "name", "unknown")
    if callable(backend_name):
        backend_name = backend_name()
    metadata: dict[str, object] = {
        **local_info,
        "backend": "ibm-qpu",
        "remote": True,
        "provider": "IBM Quantum",
        "backend_name": str(backend_name),
        "job_id": str(job.job_id()),
        "returned_samples": len(samples),
        "requested_time_limit_seconds": time_limit_seconds,
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
        "dwave-qpu",
        "dwave-hybrid",
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
    qaoa_readout_bitflip_probability: float = 0.0,
    max_feasible_states: int = 65_536,
) -> pd.DataFrame:
    """Return sampled bitstrings and energies using one common schema.

    Exact enumeration and simulated annealing are reproducible classical
    validation backends. ``qaoa_statevector`` locally simulates a constraint-
    preserving gate-model circuit. IBM and D-Wave methods are optional execution
    adapters; none of these backends imply quantum advantage.
    """

    n = len(model.variable_names)
    if not 0.0 <= qaoa_readout_bitflip_probability <= 1.0:
        raise QuantumSolverError(
            "qaoa_readout_bitflip_probability must be between zero and one"
        )
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
