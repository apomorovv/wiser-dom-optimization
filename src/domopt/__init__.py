"""WISER–Nestlé Distributed Order Management optimization package."""

from .hybrid import HybridConfig, solve_hybrid
from .planner import build_planner_table, write_planner_artifacts
from .schemas import (
    ASSUMPTION_VERSION,
    SCHEMA_VERSION,
    ObjectiveBreakdown,
    ProblemData,
    Solution,
    ValidationResult,
)

__all__ = [
    "ASSUMPTION_VERSION",
    "SCHEMA_VERSION",
    "HybridConfig",
    "ObjectiveBreakdown",
    "ProblemData",
    "Solution",
    "ValidationResult",
    "build_planner_table",
    "solve_hybrid",
    "write_planner_artifacts",
]

__version__ = "0.3.0"
