"""WISER–Nestlé Distributed Order Management optimization package."""

from .hybrid import ExactLNSConfig, HybridConfig, solve_exact_lns, solve_hybrid
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
    "ExactLNSConfig",
    "HybridConfig",
    "ObjectiveBreakdown",
    "ProblemData",
    "Solution",
    "ValidationResult",
    "build_planner_table",
    "solve_exact_lns",
    "solve_hybrid",
    "write_planner_artifacts",
]

__version__ = "0.3.0"
