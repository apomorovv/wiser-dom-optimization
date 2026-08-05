"""Public-safe synthetic DOM instance generation for scaling experiments."""

from __future__ import annotations

from math import ceil

import numpy as np
import pandas as pd

from .data import normalize_problem_data
from .schemas import ASSUMPTION_VERSION, SCHEMA_VERSION, ProblemData


def make_synthetic_problem(
    *,
    order_count: int,
    dc_count: int = 4,
    sku_count: int = 12,
    candidates_per_order: int = 3,
    seed: int = 0,
) -> ProblemData:
    """Create a reproducible, independently generated benchmark instance."""

    if order_count <= 0 or dc_count <= 0 or sku_count <= 0:
        raise ValueError("order_count, dc_count, and sku_count must be positive")
    if candidates_per_order <= 0 or candidates_per_order > dc_count:
        raise ValueError("candidates_per_order must be in [1, dc_count]")

    rng = np.random.default_rng(seed)
    dcs = [f"D{index:02d}" for index in range(dc_count)]
    skus = [f"S{index:03d}" for index in range(sku_count)]
    pgi_date = pd.Timestamp("2026-07-14")
    delivery_date = pd.Timestamp("2026-07-16")

    order_rows: list[dict[str, object]] = []
    line_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    demand_by_sku: dict[str, int] = {sku: 0 for sku in skus}

    for order_index in range(order_count):
        order_id = f"O{order_index:06d}"
        default_dc = dcs[order_index % dc_count]
        order_rows.append(
            {
                "order_id": order_id,
                "default_dc": default_dc,
                "requested_delivery_date": delivery_date,
                "default_pgi_date": pgi_date,
                "load_id": f"L{order_index // 2:06d}",
            }
        )
        line_count = int(rng.integers(1, min(4, sku_count) + 1))
        selected_skus = rng.choice(skus, size=line_count, replace=False)
        for sku_id in selected_skus:
            demand = int(rng.integers(2, 18))
            value = float(rng.integers(8, 35))
            penalty = float(rng.integers(10, 45))
            demand_by_sku[str(sku_id)] += demand
            line_rows.append(
                {
                    "order_id": order_id,
                    "sku_id": str(sku_id),
                    "demand_cases": demand,
                    "unit_value": value,
                    "penalty_per_unfilled_case": penalty,
                    "unit_weight": float(rng.uniform(1.0, 8.0)),
                    "unit_volume": float(rng.uniform(0.1, 1.5)),
                    "cases_per_pallet": int(rng.integers(12, 49)),
                }
            )

        alternatives = [dc for dc in dcs if dc != default_dc]
        chosen_dcs = [default_dc]
        if candidates_per_order > 1:
            chosen_dcs.extend(
                rng.choice(
                    alternatives,
                    size=candidates_per_order - 1,
                    replace=False,
                ).tolist()
            )
        for dc_id in chosen_dcs:
            distance = float(rng.integers(50, 1200))
            candidate_rows.append(
                {
                    "candidate_id": f"{order_id}::{dc_id}::{pgi_date.date()}",
                    "order_id": order_id,
                    "dc_id": dc_id,
                    "pgi_date": pgi_date,
                    "shipping_cost": float(100 + 0.8 * distance),
                    "is_default": dc_id == default_dc,
                    "eligible": True,
                    "distance": distance,
                    "lead_time_days": int(rng.integers(1, 3)),
                    "arrival_date": delivery_date,
                    "dock_units": 1.0,
                }
            )

    inventory_rows: list[dict[str, object]] = []
    for dc_id in dcs:
        for sku_id in skus:
            network_share = demand_by_sku[sku_id] / dc_count
            available = max(1, int(round(network_share * rng.uniform(0.65, 1.10))))
            inventory_rows.append(
                {
                    "dc_id": dc_id,
                    "sku_id": sku_id,
                    "date": pgi_date,
                    "cumulative_available_cases": available,
                }
            )

    total_cases = sum(row["demand_cases"] for row in line_rows)
    per_dc_throughput = max(1, ceil(total_cases * 0.75 / dc_count))
    per_dc_dock = max(1, ceil(order_count * 0.80 / dc_count))
    capacity_rows: list[dict[str, object]] = []
    for dc_id in dcs:
        capacity_rows.extend(
            [
                {
                    "dc_id": dc_id,
                    "date": pgi_date,
                    "resource": "dock",
                    "capacity": per_dc_dock,
                    "unit": "orders",
                },
                {
                    "dc_id": dc_id,
                    "date": pgi_date,
                    "resource": "throughput_cases",
                    "capacity": per_dc_throughput,
                    "unit": "cases",
                },
            ]
        )

    calendar = pd.DataFrame(
        [{"dc_id": dc_id, "date": pgi_date, "is_open": True} for dc_id in dcs]
    )
    return normalize_problem_data(
        ProblemData(
            orders=pd.DataFrame(order_rows),
            order_lines=pd.DataFrame(line_rows),
            inventory=pd.DataFrame(inventory_rows),
            candidates=pd.DataFrame(candidate_rows),
            capacities=pd.DataFrame(capacity_rows),
            calendar=calendar,
            metadata={
                "dataset_id": f"synthetic-{order_count}",
                "schema_version": SCHEMA_VERSION,
                "assumption_version": ASSUMPTION_VERSION,
                "currency": "synthetic_units",
                "quantity_unit": "cases",
                "inventory_policy": "projected_atp",
                "pick_capacity_mode": "auto",
                "enforce_min_divert_improvement": False,
                "generator_seed": seed,
            },
        )
    )
