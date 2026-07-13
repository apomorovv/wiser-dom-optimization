"""Small-instance QUBO samplers and quantum-backend integration boundary."""

from __future__ import annotations

from itertools import product
from typing import Literal

import numpy as np
import pandas as pd

from .qubo import qubo_energy
from .schemas import QUBOModel


class QuantumSolverError(RuntimeError):
    pass


def sample_qubo(
    model: QUBOModel,
    *,
    method: Literal["exact", "random"] = "exact",
    num_samples: int = 1000,
    seed: int = 0,
    max_exact_variables: int = 24,
) -> pd.DataFrame:
    """Return sampled bitstrings and energies.

    The exact and random samplers provide a reproducible interface for tests and
    QUBO debugging. Hardware/QAOA adapters should return the same table schema.
    """

    n = len(model.variable_names)
    rows: list[dict[str, object]] = []

    if method == "exact":
        if n > max_exact_variables:
            raise QuantumSolverError(
                f"Exact QUBO enumeration requested for {n} variables; limit is {max_exact_variables}"
            )
        bitstrings = product((0, 1), repeat=n)
    elif method == "random":
        if num_samples <= 0:
            raise QuantumSolverError("num_samples must be positive")
        rng = np.random.default_rng(seed)
        bitstrings = (rng.integers(0, 2, size=n).tolist() for _ in range(num_samples))
    else:
        raise QuantumSolverError(f"Unknown QUBO sampling method {method!r}")

    for bits in bitstrings:
        vector = np.asarray(bits, dtype=int)
        rows.append(
            {
                "bitstring": "".join(map(str, vector.tolist())),
                "energy": qubo_energy(model, vector),
                **{name: int(value) for name, value in zip(model.variable_names, vector)},
            }
        )

    frame = pd.DataFrame(rows)
    return frame.sort_values(["energy", "bitstring"], kind="mergesort").reset_index(drop=True)

