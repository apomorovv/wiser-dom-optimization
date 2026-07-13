"""QUBO construction for the reduced candidate-column formulation."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from .schemas import QUBOModel


class QUBOConstructionError(ValueError):
    pass


def build_candidate_qubo(
    plans: pd.DataFrame,
    *,
    one_hot_penalty: float,
    conflicts: pd.DataFrame | None = None,
    conflict_penalty: float | None = None,
) -> QUBOModel:
    """Build a QUBO over fixed candidate plans.

    Parameters
    ----------
    plans:
        One row per plan with ``plan_id``, ``order_id``, and ``value``. ``value``
        is the complete business objective contribution of selecting that plan.
    one_hot_penalty:
        Penalty for violating exactly one plan per order.
    conflicts:
        Optional rows ``plan_id_a``, ``plan_id_b`` for pairwise-incompatible plans.
        Pairwise conflicts are exact only when the original resource infeasibility
        is truly pairwise.
    conflict_penalty:
        Energy added when both plans in one conflict are selected.
    """

    required = {"plan_id", "order_id", "value"}
    if missing := required - set(plans.columns):
        raise QUBOConstructionError(f"plans is missing {sorted(missing)}")
    if plans["plan_id"].duplicated().any():
        raise QUBOConstructionError("plan_id must be unique")
    if one_hot_penalty <= 0:
        raise QUBOConstructionError("one_hot_penalty must be positive")

    ordered = plans.sort_values(["order_id", "plan_id"], kind="mergesort").reset_index(drop=True)
    names = tuple(ordered["plan_id"].astype(str))
    index = {name: i for i, name in enumerate(names)}
    Q = np.zeros((len(names), len(names)), dtype=float)
    constant = 0.0

    for row in ordered.itertuples(index=False):
        Q[index[str(row.plan_id)], index[str(row.plan_id)]] -= float(row.value)

    # lambda * (1 - sum y)^2 = lambda - lambda*sum y + 2lambda*sum_{i<j} y_i y_j
    for _, group in ordered.groupby("order_id", sort=False):
        ids = [index[str(plan_id)] for plan_id in group["plan_id"]]
        constant += float(one_hot_penalty)
        for i in ids:
            Q[i, i] -= float(one_hot_penalty)
        for i, j in combinations(ids, 2):
            Q[i, j] += float(one_hot_penalty)
            Q[j, i] += float(one_hot_penalty)

    if conflicts is not None and not conflicts.empty:
        required_conflicts = {"plan_id_a", "plan_id_b"}
        if missing := required_conflicts - set(conflicts.columns):
            raise QUBOConstructionError(f"conflicts is missing {sorted(missing)}")
        penalty = float(conflict_penalty or one_hot_penalty)
        if penalty <= 0:
            raise QUBOConstructionError("conflict_penalty must be positive")
        for row in conflicts.itertuples(index=False):
            a, b = str(row.plan_id_a), str(row.plan_id_b)
            if a not in index or b not in index:
                raise QUBOConstructionError(f"Unknown conflict plan pair ({a}, {b})")
            i, j = index[a], index[b]
            # x^T Q x contains 2*Q_ij*x_i*x_j for symmetric Q.
            Q[i, j] += penalty / 2.0
            Q[j, i] += penalty / 2.0

    return QUBOModel(
        variable_names=names,
        Q=Q,
        constant=constant,
        metadata={
            "one_hot_penalty": float(one_hot_penalty),
            "conflict_penalty": None if conflict_penalty is None else float(conflict_penalty),
            "plan_count": len(names),
        },
    )


def qubo_energy(model: QUBOModel, sample: np.ndarray | list[int]) -> float:
    vector = np.asarray(sample, dtype=float)
    if vector.shape != (len(model.variable_names),):
        raise ValueError(
            f"Sample has shape {vector.shape}; expected {(len(model.variable_names),)}"
        )
    return float(model.constant + vector @ np.asarray(model.Q) @ vector)

