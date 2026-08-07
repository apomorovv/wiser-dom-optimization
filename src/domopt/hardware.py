"""Hardware discovery and a privacy-safe CPU/GPU QUBO scoring benchmark."""

from __future__ import annotations

import importlib.util
import os
import platform
from time import perf_counter

import numpy as np
import pandas as pd


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError):
        return False


def hardware_capabilities() -> dict[str, object]:
    """Return capability flags without exposing credentials or host identifiers."""

    result: dict[str, object] = {
        "python": platform.python_version(),
        "logical_cpu_count": os.cpu_count(),
        "cupy_available": False,
        "cuda_device_count": 0,
        "cuda_device_name": None,
        "cuda_memory_gib": None,
        "cuopt_available": _module_available("cuopt"),
        "qiskit_available": _module_available("qiskit"),
        "qiskit_ibm_runtime_available": _module_available("qiskit_ibm_runtime"),
        "ibm_credentials_environment_configured": bool(
            os.environ.get("QISKIT_IBM_TOKEN")
            or os.environ.get("QISKIT_IBM_INSTANCE")
        ),
        "dwave_system_available": _module_available("dwave.system"),
        "dwave_credentials_configured": bool(
            os.environ.get("DWAVE_API_TOKEN") or os.environ.get("DWAVE_CONFIG_FILE")
        ),
    }
    try:
        import cupy as cp

        count = int(cp.cuda.runtime.getDeviceCount())
        result["cupy_available"] = count > 0
        result["cuda_device_count"] = count
        if count:
            properties = cp.cuda.runtime.getDeviceProperties(0)
            name = properties["name"]
            result["cuda_device_name"] = (
                name.decode() if isinstance(name, bytes) else str(name)
            )
            result["cuda_memory_gib"] = float(properties["totalGlobalMem"]) / 2**30
    except (ImportError, ModuleNotFoundError, OSError, RuntimeError) as error:
        # CUDA imports and device discovery can fail cleanly on CPU-only machines.
        result["cuda_error"] = f"{type(error).__name__}: {error}"
    return result


def _time_cpu(samples: np.ndarray, matrix: np.ndarray, repeats: int) -> float:
    np.einsum("bi,ij,bj->b", samples, matrix, samples, optimize=True)
    timings: list[float] = []
    for _ in range(repeats):
        start = perf_counter()
        np.einsum("bi,ij,bj->b", samples, matrix, samples, optimize=True)
        timings.append(perf_counter() - start)
    return float(np.median(timings))


def _time_gpu(samples: np.ndarray, matrix: np.ndarray, repeats: int) -> tuple[float, float]:
    import cupy as cp

    transfer_start = perf_counter()
    gpu_samples = cp.asarray(samples)
    gpu_matrix = cp.asarray(matrix)
    cp.cuda.Stream.null.synchronize()
    transfer_seconds = perf_counter() - transfer_start
    cp.einsum("bi,ij,bj->b", gpu_samples, gpu_matrix, gpu_samples, optimize=True)
    cp.cuda.Stream.null.synchronize()
    timings: list[float] = []
    for _ in range(repeats):
        start = perf_counter()
        cp.einsum("bi,ij,bj->b", gpu_samples, gpu_matrix, gpu_samples, optimize=True)
        cp.cuda.Stream.null.synchronize()
        timings.append(perf_counter() - start)
    return float(np.median(timings)), transfer_seconds


def benchmark_qubo_batch_scoring(
    *,
    variable_counts: tuple[int, ...] = (16, 40, 96),
    sample_counts: tuple[int, ...] = (256, 4096),
    repeats: int = 3,
    seed: int = 7,
    include_gpu: bool = True,
) -> pd.DataFrame:
    """Measure only batched QUBO energy scoring on synthetic coefficients.

    This is a crossover diagnostic, not an end-to-end solver benchmark. It shows
    whether GPU launch/transfer overhead is justified for the local QUBO sizes and
    read counts under consideration.
    """

    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if not variable_counts or not sample_counts:
        raise ValueError("variable_counts and sample_counts must be nonempty")
    if min(variable_counts) <= 0 or min(sample_counts) <= 0:
        raise ValueError("benchmark sizes must be positive")

    rng = np.random.default_rng(seed)
    gpu_available = bool(hardware_capabilities()["cupy_available"]) and include_gpu
    rows: list[dict[str, object]] = []
    for variables in variable_counts:
        raw = rng.normal(size=(variables, variables))
        matrix = np.asarray(0.5 * (raw + raw.T), dtype=np.float64)
        for samples_count in sample_counts:
            samples = rng.integers(
                0,
                2,
                size=(samples_count, variables),
                dtype=np.int8,
            ).astype(np.float64)
            cpu_seconds = _time_cpu(samples, matrix, repeats)
            rows.append(
                {
                    "backend": "numpy_cpu",
                    "variables": variables,
                    "samples": samples_count,
                    "compute_seconds": cpu_seconds,
                    "transfer_seconds": 0.0,
                    "samples_per_second": samples_count / cpu_seconds,
                    "end_to_end_seconds": cpu_seconds,
                    "end_to_end_samples_per_second": samples_count / cpu_seconds,
                }
            )
            if gpu_available:
                gpu_seconds, transfer_seconds = _time_gpu(samples, matrix, repeats)
                rows.append(
                    {
                        "backend": "cupy_gpu",
                        "variables": variables,
                        "samples": samples_count,
                        "compute_seconds": gpu_seconds,
                        "transfer_seconds": transfer_seconds,
                        "samples_per_second": samples_count / gpu_seconds,
                        "end_to_end_seconds": gpu_seconds + transfer_seconds,
                        "end_to_end_samples_per_second": samples_count
                        / (gpu_seconds + transfer_seconds),
                    }
                )
    return pd.DataFrame(rows)
