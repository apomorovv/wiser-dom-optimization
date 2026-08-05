"""Canonical typed structures used throughout the DOM optimization package."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

SCHEMA_VERSION = "0.2.0"
ASSUMPTION_VERSION = "v1"


@dataclass(frozen=True)
class ProblemData:
    """Canonical in-memory representation of one DOM instance."""

    orders: pd.DataFrame
    order_lines: pd.DataFrame
    inventory: pd.DataFrame
    candidates: pd.DataFrame
    capacities: pd.DataFrame
    calendar: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)
    source_dir: Path | None = None


@dataclass
class Solution:
    """Common solver output consumed by validation, objective, and metrics."""

    method: str
    assignments: pd.DataFrame
    fulfillment: pd.DataFrame
    runtime_seconds: float = 0.0
    raw_objective: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ObjectiveBreakdown:
    fulfilled_value: float
    penalty_cost: float
    shipping_cost: float

    @property
    def objective_value(self) -> float:
        return self.fulfilled_value - self.penalty_cost - self.shipping_cost

    def to_dict(self) -> dict[str, float]:
        return {
            "fulfilled_value": float(self.fulfilled_value),
            "penalty_cost": float(self.penalty_cost),
            "shipping_cost": float(self.shipping_cost),
            "objective_value": float(self.objective_value),
        }


@dataclass
class ValidationResult:
    is_feasible: bool
    assignment_violations: list[str] = field(default_factory=list)
    demand_violations: list[str] = field(default_factory=list)
    inventory_violations: list[str] = field(default_factory=list)
    eligibility_violations: list[str] = field(default_factory=list)
    capacity_violations: list[str] = field(default_factory=list)
    schema_violations: list[str] = field(default_factory=list)

    @property
    def violations(self) -> list[str]:
        return (
            self.assignment_violations
            + self.demand_violations
            + self.inventory_violations
            + self.eligibility_violations
            + self.capacity_violations
            + self.schema_violations
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_feasible": bool(self.is_feasible),
            "assignment_violations": list(self.assignment_violations),
            "demand_violations": list(self.demand_violations),
            "inventory_violations": list(self.inventory_violations),
            "eligibility_violations": list(self.eligibility_violations),
            "capacity_violations": list(self.capacity_violations),
            "schema_violations": list(self.schema_violations),
            "violations": list(self.violations),
        }


@dataclass(frozen=True)
class QUBOModel:
    """Dense QUBO representation E(x) = constant + x^T Q x."""

    variable_names: tuple[str, ...]
    Q: Any
    constant: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


ORDERS_COLUMNS = {
    "order_id",
    "default_dc",
    "requested_delivery_date",
}
ORDER_LINES_COLUMNS = {
    "order_id",
    "sku_id",
    "demand_cases",
    "unit_value",
    "penalty_per_unfilled_case",
}
INVENTORY_COLUMNS = {
    "dc_id",
    "sku_id",
    "date",
    "cumulative_available_cases",
}
CANDIDATES_COLUMNS = {
    "candidate_id",
    "order_id",
    "dc_id",
    "pgi_date",
    "shipping_cost",
    "is_default",
    "eligible",
}
CAPACITY_COLUMNS = {
    "dc_id",
    "date",
    "resource",
    "capacity",
    "unit",
}
CALENDAR_COLUMNS = {
    "dc_id",
    "date",
    "is_open",
}

ASSIGNMENT_COLUMNS = {
    "order_id",
    "candidate_id",
    "selected_dc",
    "selected_pgi_date",
    "is_unassigned",
    "is_divert",
    "method",
}
FULFILLMENT_COLUMNS = {
    "order_id",
    "sku_id",
    "fulfilled_cases",
    "unfulfilled_cases",
    "selected_dc",
    "selected_pgi_date",
}

