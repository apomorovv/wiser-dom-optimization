"""WISER–Nestlé Distributed Order Management optimization package."""

from .schemas import (
    ASSUMPTION_VERSION,
    SCHEMA_VERSION,
    ObjectiveBreakdown,
    ProblemData,
    Solution,
    ValidationResult,
)
from .hybrid import HybridConfig, solve_hybrid
from .planner import build_planner_table, write_planner_artifacts

__all__ = [
    "ASSUMPTION_VERSION",
    "SCHEMA_VERSION",
    "ObjectiveBreakdown",
    "ProblemData",
    "Solution",
    "ValidationResult",
    "HybridConfig",
    "build_planner_table",
    "solve_hybrid",
    "write_planner_artifacts",
]

__version__ = "0.2.0"
