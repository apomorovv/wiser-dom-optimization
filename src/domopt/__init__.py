"""WISER–Nestlé Distributed Order Management optimization package."""

__version__ = "0.5.0"

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
from .solver import SolverConfig, SolverMode, solve_dom

__all__ = [
    "ASSUMPTION_VERSION",
    "SCHEMA_VERSION",
    "ExactLNSConfig",
    "HybridConfig",
    "ObjectiveBreakdown",
    "ProblemData",
    "Solution",
    "SolverConfig",
    "SolverMode",
    "ValidationResult",
    "build_planner_table",
    "solve_dom",
    "solve_exact_lns",
    "solve_hybrid",
    "write_planner_artifacts",
]
