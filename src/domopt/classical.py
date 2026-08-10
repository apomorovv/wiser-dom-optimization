"""Exact deterministic MILP and fixed-assignment recourse solver."""

from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.util import find_spec
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .penalties import (
    THRESHOLDED_CUT,
    build_penalty_context,
    penalty_mode,
)
from .resources import candidate_fixed_consumption, uses_split_pick_accounting
from .rules import minimum_divert_fulfillment
from .schemas import ProblemData, Solution


class ClassicalSolverError(RuntimeError):
    """Raised when the MILP cannot produce a usable solution."""


@dataclass(frozen=True)
class _MILPResult:
    """Backend-independent result from one compiled linear mixed-integer model."""

    x: np.ndarray
    fun: float
    status: int
    message: str
    success: bool
    best_bound: float | None
    mip_gap: float | None
    mip_node_count: float | None
    solver: str
    backend: str


def available_milp_backends() -> dict[str, bool]:
    """Report installed MILP adapters without opening a commercial license.

    ``scipy-highs`` is always available because SciPy is a core dependency.
    Optional values report importable adapters only. A usable local or remote
    Gurobi license is still checked only when that backend is selected.
    """

    return {
        "scipy-highs": True,
        "highspy": find_spec("highspy") is not None,
        "scip": find_spec("pyscipopt") is not None,
        "gurobi": find_spec("gurobipy") is not None,
    }


def available_cpu_count() -> int:
    """Return the CPU budget visible to this process on every supported OS."""

    process_cpu_count = getattr(os, "process_cpu_count", None)
    if process_cpu_count is not None:
        detected = process_cpu_count()
        if detected:
            return max(1, int(detected))
    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is not None:
        try:
            return max(1, len(affinity(0)))
        except OSError:
            pass
    return max(1, int(os.cpu_count() or 1))


def _effective_thread_count(thread_count: int | None) -> int:
    if thread_count is not None and int(thread_count) <= 0:
        raise ClassicalSolverError("thread_count must be positive when provided")
    available = available_cpu_count()
    return available if thread_count is None else min(int(thread_count), available)


def _solve_compiled_milp(
    *,
    backend: str,
    objective: np.ndarray,
    integrality: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    matrix: Any,
    constraint_lower: np.ndarray,
    constraint_upper: np.ndarray,
    time_limit_seconds: float,
    mip_relative_gap: float,
    seed: int | None,
    thread_count: int | None,
) -> _MILPResult:
    normalized = str(backend).strip().lower().replace("_", "-")
    if normalized in {"scipy", "highs", "scipy-highs"}:
        options: dict[str, Any] = {
            "time_limit": float(time_limit_seconds),
            "mip_rel_gap": float(mip_relative_gap),
            "presolve": True,
        }
        result = milp(
            c=objective,
            integrality=integrality,
            bounds=Bounds(lower_bounds, upper_bounds),
            constraints=LinearConstraint(
                matrix,
                constraint_lower,
                constraint_upper,
            ),
            options=options,
        )
        if result.x is None:
            raise ClassicalSolverError(
                f"HiGHS returned no incumbent. status={result.status}, message={result.message}"
            )
        if not bool(result.success) and int(result.status) not in {1}:
            raise ClassicalSolverError(
                f"HiGHS failed. status={result.status}, message={result.message}"
            )
        return _MILPResult(
            x=np.asarray(result.x, dtype=float),
            fun=float(result.fun),
            status=int(result.status),
            message=str(result.message),
            success=bool(result.success),
            best_bound=(
                None
                if getattr(result, "mip_dual_bound", None) is None
                else float(result.mip_dual_bound)
            ),
            mip_gap=(None if getattr(result, "mip_gap", None) is None else float(result.mip_gap)),
            mip_node_count=getattr(result, "mip_node_count", None),
            solver="scipy.optimize.milp/HiGHS",
            backend="scipy-highs",
        )

    if normalized in {"highspy", "native-highs", "highs-native"}:
        try:
            import highspy
        except ImportError as error:
            raise ClassicalSolverError(
                "The optional native HiGHS backend requires "
                "`pip install -e '.[highs]'`. SciPy/HiGHS remains the default."
            ) from error

        try:
            compiled = matrix.tocsr()
            model = highspy.HighsLp()
            model.num_col_ = len(objective)
            model.num_row_ = int(compiled.shape[0])
            model.col_cost_ = np.asarray(objective, dtype=np.float64)
            model.col_lower_ = np.asarray(lower_bounds, dtype=np.float64)
            model.col_upper_ = np.asarray(upper_bounds, dtype=np.float64)
            model.row_lower_ = np.asarray(constraint_lower, dtype=np.float64)
            model.row_upper_ = np.asarray(constraint_upper, dtype=np.float64)
            model.integrality_ = [
                highspy.HighsVarType.kInteger if bool(value) else highspy.HighsVarType.kContinuous
                for value in integrality
            ]
            model.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
            model.a_matrix_.num_col_ = model.num_col_
            model.a_matrix_.num_row_ = model.num_row_
            model.a_matrix_.start_ = np.asarray(compiled.indptr, dtype=np.int32)
            model.a_matrix_.index_ = np.asarray(compiled.indices, dtype=np.int32)
            model.a_matrix_.value_ = np.asarray(compiled.data, dtype=np.float64)

            # HiGHS owns a process-global scheduler. Reset it before applying a
            # possibly different per-run cap so repeated cockpit/benchmark runs do
            # not inherit the first solve's thread count.
            highspy.Highs.resetGlobalScheduler(True)
            highs = highspy.Highs()
            highs.setOptionValue("output_flag", False)
            highs.setOptionValue("time_limit", float(time_limit_seconds))
            highs.setOptionValue("mip_rel_gap", float(mip_relative_gap))
            highs.setOptionValue("presolve", "on")
            highs.setOptionValue("parallel", "on")
            highs.setOptionValue("threads", _effective_thread_count(thread_count))
            if seed is not None:
                highs.setOptionValue("random_seed", int(seed))
            pass_status = highs.passModel(model)
            if pass_status != highspy.HighsStatus.kOk:
                raise ClassicalSolverError(
                    f"Native HiGHS rejected the compiled model: {pass_status}"
                )
            run_status = highs.run()
            status = highs.getModelStatus()
            solution = highs.getSolution()
            if run_status == highspy.HighsStatus.kError or not bool(solution.value_valid):
                raise ClassicalSolverError(
                    "Native HiGHS returned no incumbent. "
                    f"status={highs.modelStatusToString(status)}"
                )
            info = highs.getInfo()
            optimal = status == highspy.HighsModelStatus.kOptimal
            acceptable = {
                highspy.HighsModelStatus.kOptimal,
                highspy.HighsModelStatus.kTimeLimit,
                highspy.HighsModelStatus.kIterationLimit,
                highspy.HighsModelStatus.kSolutionLimit,
                highspy.HighsModelStatus.kInterrupt,
                highspy.HighsModelStatus.kHighsInterrupt,
            }
            if status not in acceptable:
                raise ClassicalSolverError(
                    f"Native HiGHS failed. status={highs.modelStatusToString(status)}"
                )
            return _MILPResult(
                x=np.asarray(solution.col_value, dtype=float),
                fun=float(info.objective_function_value),
                status=int(status),
                message=highs.modelStatusToString(status),
                success=optimal,
                best_bound=float(info.mip_dual_bound),
                mip_gap=float(info.mip_gap),
                mip_node_count=float(info.mip_node_count),
                solver=f"highspy/HiGHS {highs.version()}",
                backend="highspy",
            )
        except ClassicalSolverError:
            raise
        except Exception as error:
            raise ClassicalSolverError(
                f"Native HiGHS could not build or solve the model: {error}"
            ) from error

    if normalized in {"scip", "pyscipopt"}:
        try:
            import pyscipopt
            from pyscipopt import Model, quicksum
        except ImportError as error:
            raise ClassicalSolverError(
                "The optional SCIP backend requires `pip install -e '.[scip]'`. "
                "SciPy/HiGHS remains the default."
            ) from error

        try:
            scip = Model("wiser-dom")
            scip.hideOutput()
            scip.setRealParam("limits/time", float(time_limit_seconds))
            scip.setRealParam("limits/gap", float(mip_relative_gap))
            scip.setIntParam("parallel/maxnthreads", _effective_thread_count(thread_count))
            if seed is not None:
                scip.setIntParam("randomization/randomseedshift", int(seed))
            variables = [
                scip.addVar(
                    name=f"v_{index}",
                    vtype="I" if bool(integrality[index]) else "C",
                    lb=(
                        -scip.infinity()
                        if not np.isfinite(lower_bounds[index])
                        else float(lower_bounds[index])
                    ),
                    ub=(
                        None if not np.isfinite(upper_bounds[index]) else float(upper_bounds[index])
                    ),
                    obj=float(objective[index]),
                )
                for index in range(len(objective))
            ]
            compiled = matrix.tocsr()
            for row in range(compiled.shape[0]):
                start, end = int(compiled.indptr[row]), int(compiled.indptr[row + 1])
                expression = quicksum(
                    float(compiled.data[position]) * variables[int(compiled.indices[position])]
                    for position in range(start, end)
                )
                lower = float(constraint_lower[row])
                upper = float(constraint_upper[row])
                if (
                    np.isfinite(lower)
                    and np.isfinite(upper)
                    and np.isclose(lower, upper, rtol=0.0, atol=1e-12)
                ):
                    scip.addCons(expression == lower)
                else:
                    if np.isfinite(lower):
                        scip.addCons(expression >= lower)
                    if np.isfinite(upper):
                        scip.addCons(expression <= upper)
            scip.setMinimize()
            scip.optimize()
            if int(scip.getNSols()) <= 0:
                raise ClassicalSolverError(f"SCIP returned no incumbent. status={scip.getStatus()}")
            incumbent = scip.getBestSol()
            status = str(scip.getStatus())
            return _MILPResult(
                x=np.asarray(
                    [scip.getSolVal(incumbent, variable) for variable in variables],
                    dtype=float,
                ),
                fun=float(scip.getObjVal()),
                status=0 if status == "optimal" else 1,
                message=status,
                success=status == "optimal",
                best_bound=float(scip.getDualbound()),
                mip_gap=float(scip.getGap()),
                mip_node_count=float(scip.getNNodes()),
                solver=f"PySCIPOpt/SCIP {pyscipopt.__version__}",
                backend="scip",
            )
        except ClassicalSolverError:
            raise
        except Exception as error:
            raise ClassicalSolverError(
                f"SCIP could not build or solve the model: {error}"
            ) from error

    if normalized != "gurobi":
        raise ClassicalSolverError(
            "Unknown MILP backend. Choose 'scipy-highs' (default), 'highspy', 'scip', or 'gurobi'."
        )
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as error:
        raise ClassicalSolverError(
            "The optional Gurobi backend requires `pip install -e '.[gurobi]'` "
            "and a valid Gurobi license. SciPy/HiGHS remains the default."
        ) from error

    try:
        gurobi = gp.Model("wiser-dom")
        gurobi.Params.OutputFlag = 0
        gurobi.Params.TimeLimit = float(time_limit_seconds)
        gurobi.Params.MIPGap = float(mip_relative_gap)
        gurobi.Params.Threads = _effective_thread_count(thread_count)
        if seed is not None:
            gurobi.Params.Seed = int(seed)
        lower = np.where(np.isfinite(lower_bounds), lower_bounds, -GRB.INFINITY)
        upper = np.where(np.isfinite(upper_bounds), upper_bounds, GRB.INFINITY)
        variable_types = np.where(integrality.astype(bool), GRB.INTEGER, GRB.CONTINUOUS)
        variables = gurobi.addMVar(
            len(objective),
            lb=lower,
            ub=upper,
            obj=objective,
            vtype=variable_types.tolist(),
            name="v",
        )
        equality = (
            np.isfinite(constraint_lower)
            & np.isfinite(constraint_upper)
            & np.isclose(constraint_lower, constraint_upper, rtol=0.0, atol=1e-12)
        )
        upper_rows = np.isfinite(constraint_upper) & ~equality
        lower_rows = np.isfinite(constraint_lower) & ~equality
        if bool(equality.any()):
            gurobi.addMConstr(
                matrix[equality], variables, "=", constraint_lower[equality], name="eq"
            )
        if bool(upper_rows.any()):
            gurobi.addMConstr(
                matrix[upper_rows], variables, "<", constraint_upper[upper_rows], name="ub"
            )
        if bool(lower_rows.any()):
            gurobi.addMConstr(
                matrix[lower_rows], variables, ">", constraint_lower[lower_rows], name="lb"
            )
        gurobi.ModelSense = GRB.MINIMIZE
        gurobi.optimize()
        if int(gurobi.SolCount) <= 0:
            raise ClassicalSolverError(f"Gurobi returned no incumbent. status={int(gurobi.Status)}")
        status_names = {
            GRB.OPTIMAL: "optimal",
            GRB.TIME_LIMIT: "time limit with incumbent",
            GRB.SUBOPTIMAL: "suboptimal incumbent",
            GRB.INTERRUPTED: "interrupted with incumbent",
        }
        return _MILPResult(
            x=np.asarray(variables.X, dtype=float),
            fun=float(gurobi.ObjVal),
            status=int(gurobi.Status),
            message=status_names.get(gurobi.Status, f"Gurobi status {gurobi.Status}"),
            success=int(gurobi.Status) == int(GRB.OPTIMAL),
            best_bound=float(gurobi.ObjBound),
            mip_gap=float(gurobi.MIPGap),
            mip_node_count=float(gurobi.NodeCount),
            solver=f"gurobipy/Gurobi {gp.gurobi.version()}",
            backend="gurobi",
        )
    except ClassicalSolverError:
        raise
    except gp.GurobiError as error:
        raise ClassicalSolverError(
            "Gurobi could not start or solve the model. Confirm that a compatible "
            f"license is available. Original error: {error}"
        ) from error


@dataclass
class _RowBuilder:
    rows: list[int]
    cols: list[int]
    data: list[float]
    lower: list[float]
    upper: list[float]
    row_count: int = 0

    @classmethod
    def create(cls) -> _RowBuilder:
        return cls([], [], [], [], [])

    def add(self, coefficients: Mapping[int, float], lb: float, ub: float) -> None:
        for column, value in coefficients.items():
            if abs(value) > 0:
                self.rows.append(self.row_count)
                self.cols.append(int(column))
                self.data.append(float(value))
        self.lower.append(float(lb))
        self.upper.append(float(ub))
        self.row_count += 1


def solve_classical(
    problem: ProblemData,
    *,
    backend: str = "scipy-highs",
    time_limit_seconds: float = 60.0,
    mip_relative_gap: float = 0.0,
    seed: int | None = None,
    thread_count: int | None = None,
    fixed_assignments: Mapping[str, str | None] | None = None,
    minimum_objective: float | None = None,
) -> Solution:
    """Solve the detailed DOM MILP with a selected interchangeable backend.

    fixed_assignments turns the model into an exact fulfillment-recourse
    problem. A value is either a candidate ID or None for no assignment.
    The hybrid optimizer uses this bounded recourse mode after sampling a local
    assignment QUBO. ``minimum_objective`` adds a safe incumbent cutoff in the
    maximization convention used by :mod:`domopt.objective`; polish calls use it
    to prevent a loose MIP-gap stop from returning a policy worse than its input.

    ``scipy-highs`` is the portable, license-free default. Native ``highspy``,
    open-source ``scip``, and commercial ``gurobi`` adapters are opt-in controlled
    comparisons and are never selected implicitly. Native adapters use the CPU
    budget visible to the process unless ``thread_count`` supplies a lower cap.
    Seed is recorded for experiment consistency; SciPy's portable HiGHS interface
    does not currently expose random-seed or thread-count options.
    """

    recorded_seed = seed
    if thread_count is not None:
        _effective_thread_count(thread_count)
    if minimum_objective is not None and not np.isfinite(minimum_objective):
        raise ClassicalSolverError("minimum_objective must be finite when provided")
    start = perf_counter()
    fixed = {str(key): value for key, value in (fixed_assignments or {}).items()}

    all_candidates = (
        problem.candidates.loc[problem.candidates["eligible"].astype(bool)]
        .sort_values(["order_id", "pgi_date", "candidate_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    lines = problem.order_lines.sort_values(["order_id", "sku_id"], kind="mergesort").reset_index(
        drop=True
    )
    orders = problem.orders.sort_values("order_id", kind="mergesort").reset_index(drop=True)
    order_ids = set(orders["order_id"].astype(str))
    unknown_fixed = sorted(set(fixed) - order_ids)
    if unknown_fixed:
        raise ClassicalSolverError(f"Fixed assignments reference unknown orders: {unknown_fixed}")

    # Validate fixed candidate IDs before structural presolve removes unused rows.
    # The previous recourse path constructed variables for every eligible candidate
    # and then fixed one binary per order.  On the POC data that made a quantity-only
    # recourse solve almost as large as the unrestricted assignment MILP.  Filtering
    # fully fixed decisions here is mathematically equivalent and reduces variables,
    # constraints, and Python model-construction work before HiGHS is called.
    all_candidate_records = [row._asdict() for row in all_candidates.itertuples(index=False)]
    full_candidate_lookup = {
        str(candidate["candidate_id"]): candidate for candidate in all_candidate_records
    }
    for order_id, candidate_id in fixed.items():
        if candidate_id is None:
            continue
        candidate_key = str(candidate_id)
        if candidate_key not in full_candidate_lookup:
            raise ClassicalSolverError(
                f"Fixed candidate {candidate_key!r} is not eligible in this problem"
            )
        candidate = full_candidate_lookup[candidate_key]
        if str(candidate["order_id"]) != order_id:
            raise ClassicalSolverError(
                f"Fixed candidate {candidate_key!r} does not belong to order {order_id!r}"
            )

    structurally_fixed = set(fixed)
    if bool(problem.metadata.get("enforce_assignment_group", False)):
        if "assignment_group" not in orders.columns:
            raise ClassicalSolverError("Group cohesion requires orders.assignment_group")
        # A partially fixed assignment group must retain every option because the
        # unfixed members still need the original cohesion equations.  A fully fixed
        # group can be reduced only after checking that all members select the same
        # group option (or are all unassigned).
        structurally_fixed = set()
        if "group_option_id" not in all_candidates.columns:
            raise ClassicalSolverError("Group cohesion requires candidates.group_option_id")
        for group_id, group in orders.groupby("assignment_group", sort=False):
            members = set(group["order_id"].astype(str))
            fixed_members = members & set(fixed)
            if fixed_members != members:
                continue
            choices = {fixed[order_id] for order_id in members}
            if choices == {None}:
                structurally_fixed.update(members)
                continue
            if None in choices:
                raise ClassicalSolverError(f"Fixed assignments split assignment group {group_id!r}")
            option_ids = {
                str(full_candidate_lookup[str(candidate_id)]["group_option_id"])
                for candidate_id in choices
            }
            if len(option_ids) != 1:
                raise ClassicalSolverError(f"Fixed assignments split assignment group {group_id!r}")
            structurally_fixed.update(members)

    keep_candidate_ids = {
        str(candidate_id)
        for order_id, candidate_id in fixed.items()
        if order_id in structurally_fixed and candidate_id is not None
    }
    candidates = all_candidates.loc[
        ~all_candidates["order_id"].astype(str).isin(structurally_fixed)
        | all_candidates["candidate_id"].astype(str).isin(keep_candidate_ids)
    ].reset_index(drop=True)

    candidate_records = [row._asdict() for row in candidates.itertuples(index=False)]
    candidates_by_order: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for candidate in candidate_records:
        candidates_by_order[str(candidate["order_id"])].append(candidate)
    candidate_lookup = {
        str(candidate["candidate_id"]): candidate for candidate in candidate_records
    }
    line_records = [
        {"line_id": line_id, **row._asdict()}
        for line_id, row in enumerate(lines.itertuples(index=False))
    ]
    line_by_id = {int(line["line_id"]): line for line in line_records}
    lines_by_order: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    line_ids_by_sku: defaultdict[str, list[int]] = defaultdict(list)
    for line in line_records:
        line_id = int(line["line_id"])
        lines_by_order[str(line["order_id"])].append(line)
        line_ids_by_sku[str(line["sku_id"])].append(line_id)
    penalty_context = build_penalty_context(problem)

    resources = set(problem.capacities.get("resource", pd.Series(dtype=str)).astype(str))
    split_picks = uses_split_pick_accounting(problem) and bool(
        resources & {"case_pick", "pallet_pick"}
    )
    if split_picks:
        if "cases_per_pallet" not in lines.columns or lines["cases_per_pallet"].isna().any():
            raise ClassicalSolverError(
                "Pallet/case-pick accounting requires cases_per_pallet for every order line"
            )
        if (lines["cases_per_pallet"].astype(int) <= 0).any():
            raise ClassicalSolverError("cases_per_pallet must be positive")
    if "pallet_pick" in resources and not split_picks:
        raise ClassicalSolverError(
            "pallet_pick capacity requires metadata.pick_capacity_mode='pallet_case'"
        )

    # Variable order: assignment x, no-assignment z, fulfillment f, unmet u,
    # and optional full-pallet p / loose-case k variables.
    next_index = 0
    x_index: dict[str, int] = {}
    for candidate in candidate_records:
        x_index[str(candidate["candidate_id"])] = next_index
        next_index += 1

    z_index: dict[str, int] = {}
    for order_id in orders["order_id"].astype(str):
        z_index[order_id] = next_index
        next_index += 1

    f_index: dict[tuple[int, str], int] = {}
    for line in line_records:
        line_id = int(line["line_id"])
        for candidate in candidates_by_order.get(str(line["order_id"]), []):
            candidate_id = str(candidate["candidate_id"])
            f_index[(line_id, candidate_id)] = next_index
            next_index += 1

    u_index: dict[int, int] = {}
    for line_id in lines.index:
        u_index[int(line_id)] = next_index
        next_index += 1

    p_index: dict[tuple[int, str], int] = {}
    k_index: dict[tuple[int, str], int] = {}
    if split_picks:
        for key in f_index:
            p_index[key] = next_index
            next_index += 1
            k_index[key] = next_index
            next_index += 1

    thresholded_penalties = penalty_mode(problem) == THRESHOLDED_CUT
    penalty_active_index: dict[str, int] = {}
    penalty_raw_index: dict[str, int] = {}
    penalty_floor_value_index: dict[str, int] = {}
    penalty_value_index: dict[str, int] = {}
    penalty_floor_binary_index: dict[str, int] = {}
    penalty_cap_binary_index: dict[str, int] = {}
    penalty_unmet_product_index: dict[int, int] = {}
    penalty_cut_sku_index: dict[int, int] = {}
    if thresholded_penalties:
        for order_id in orders["order_id"].astype(str):
            penalty_active_index[order_id] = next_index
            next_index += 1
            penalty_raw_index[order_id] = next_index
            next_index += 1
            penalty_floor_value_index[order_id] = next_index
            next_index += 1
            penalty_value_index[order_id] = next_index
            next_index += 1
            parameters = penalty_context.parameters_by_order[order_id]
            if parameters["minimum"] > 0:
                penalty_floor_binary_index[order_id] = next_index
                next_index += 1
            if parameters["maximum"] > 0:
                penalty_cap_binary_index[order_id] = next_index
                next_index += 1
        for line_id in lines.index:
            penalty_unmet_product_index[int(line_id)] = next_index
            next_index += 1
            penalty_cut_sku_index[int(line_id)] = next_index
            next_index += 1

    n_variables = next_index
    objective = np.zeros(n_variables, dtype=float)
    lower_bounds = np.zeros(n_variables, dtype=float)
    upper_bounds = np.full(n_variables, np.inf, dtype=float)
    integrality = np.ones(n_variables, dtype=np.uint8)

    for candidate in candidate_records:
        index = x_index[str(candidate["candidate_id"])]
        objective[index] = float(candidate["shipping_cost"])
        upper_bounds[index] = 1.0
    for index in z_index.values():
        upper_bounds[index] = 1.0

    inventory_last_checkpoint = {
        (str(dc_id), str(sku_id)): pd.Timestamp(group["date"].max())
        for (dc_id, sku_id), group in problem.inventory.groupby(["dc_id", "sku_id"], sort=False)
    }
    for line in line_records:
        line_id = int(line["line_id"])
        demand = int(line["demand_cases"])
        for candidate in candidates_by_order.get(str(line["order_id"]), []):
            candidate_id = str(candidate["candidate_id"])
            key = (line_id, candidate_id)
            objective[f_index[key]] = -float(line["unit_value"])
            last_checkpoint = inventory_last_checkpoint.get(
                (str(candidate["dc_id"]), str(line["sku_id"]))
            )
            has_covering_checkpoint = (
                last_checkpoint is not None
                and pd.Timestamp(candidate["pgi_date"]) <= last_checkpoint
            )
            # Without a checkpoint at or after shipment, the cumulative-ATP
            # constraints contain no row that can cover this fulfillment. Such
            # a line has zero modeled availability (matching the validator and
            # greedy inventory state), rather than unbounded free inventory.
            upper_bounds[f_index[key]] = demand if has_covering_checkpoint else 0
            if split_picks:
                per_pallet = int(line["cases_per_pallet"])
                upper_bounds[p_index[key]] = demand // per_pallet if has_covering_checkpoint else 0
                upper_bounds[k_index[key]] = per_pallet - 1 if has_covering_checkpoint else 0
        if not thresholded_penalties:
            objective[u_index[line_id]] = float(line["penalty_per_unfilled_case"])
        upper_bounds[u_index[line_id]] = demand

    if thresholded_penalties:
        for order_id in orders["order_id"].astype(str):
            parameters = penalty_context.parameters_by_order[order_id]
            group = lines_by_order[order_id]
            raw_upper = (
                sum(
                    float(line["penalty_per_unfilled_case"]) * int(line["demand_cases"])
                    for line in group
                )
                + parameters["fixed"]
                + parameters["per_cut_sku"] * len(group)
            )
            floor_upper = max(raw_upper, parameters["minimum"])
            final_upper = (
                min(floor_upper, parameters["maximum"])
                if parameters["maximum"] > 0
                else floor_upper
            )
            active = penalty_active_index[order_id]
            raw = penalty_raw_index[order_id]
            floor_value = penalty_floor_value_index[order_id]
            value = penalty_value_index[order_id]
            upper_bounds[active] = 1.0
            upper_bounds[raw] = raw_upper
            upper_bounds[floor_value] = floor_upper
            upper_bounds[value] = final_upper
            integrality[raw] = 0
            integrality[floor_value] = 0
            integrality[value] = 0
            objective[value] = 1.0
            if order_id in penalty_floor_binary_index:
                upper_bounds[penalty_floor_binary_index[order_id]] = 1.0
            if order_id in penalty_cap_binary_index:
                upper_bounds[penalty_cap_binary_index[order_id]] = 1.0
        for line in line_records:
            line_id = int(line["line_id"])
            product = penalty_unmet_product_index[line_id]
            cut = penalty_cut_sku_index[line_id]
            upper_bounds[product] = int(line["demand_cases"])
            upper_bounds[cut] = 1.0
            integrality[product] = 0

    constraints = _RowBuilder.create()

    # Exactly one modeled outcome per order.
    for order_id in orders["order_id"].astype(str):
        coefficients = {z_index[order_id]: 1.0}
        for candidate in candidates_by_order.get(order_id, []):
            coefficients[x_index[str(candidate["candidate_id"])]] = 1.0
        constraints.add(coefficients, 1.0, 1.0)

    # Optional load/assignment-group cohesion. Every member must choose the same
    # DC/date option (or all be unassigned); cost and dock usage can be placed on
    # one deterministic group leader by the data adapter.
    if bool(problem.metadata.get("enforce_assignment_group", False)):
        required_group_columns = {"assignment_group"}
        required_candidate_columns = {"group_option_id"}
        if not required_group_columns <= set(orders.columns):
            raise ClassicalSolverError("Group cohesion requires orders.assignment_group")
        if not required_candidate_columns <= set(candidates.columns):
            raise ClassicalSolverError("Group cohesion requires candidates.group_option_id")
        for _, group_orders in orders.groupby("assignment_group", sort=False):
            members = group_orders["order_id"].astype(str).tolist()
            if len(members) <= 1:
                continue
            leader = members[0]
            leader_options = {
                str(row["group_option_id"]): str(row["candidate_id"])
                for row in candidates_by_order.get(leader, [])
            }
            for member in members[1:]:
                member_options = {
                    str(row["group_option_id"]): str(row["candidate_id"])
                    for row in candidates_by_order.get(member, [])
                }
                if set(member_options) != set(leader_options):
                    raise ClassicalSolverError(
                        "All orders in an assignment group must expose identical "
                        f"group_option_id values; mismatch in group {group_orders.iloc[0]['assignment_group']!r}"
                    )
                constraints.add({z_index[member]: 1.0, z_index[leader]: -1.0}, 0.0, 0.0)
                for option_id, leader_candidate in leader_options.items():
                    constraints.add(
                        {
                            x_index[member_options[option_id]]: 1.0,
                            x_index[leader_candidate]: -1.0,
                        },
                        0.0,
                        0.0,
                    )

    # Fix assignment decisions for classical recourse when requested.
    for order_id, candidate_id in fixed.items():
        if candidate_id is None:
            constraints.add({z_index[order_id]: 1.0}, 1.0, 1.0)
            continue
        candidate_key = str(candidate_id)
        if candidate_key not in x_index:
            raise ClassicalSolverError(
                f"Fixed candidate {candidate_key!r} is not eligible in this problem"
            )
        candidate = candidate_lookup[candidate_key]
        if str(candidate["order_id"]) != order_id:
            raise ClassicalSolverError(
                f"Fixed candidate {candidate_key!r} does not belong to order {order_id!r}"
            )
        constraints.add({x_index[candidate_key]: 1.0}, 1.0, 1.0)

    # Demand balance, assignment linking, and optional exact pick decomposition.
    for line in line_records:
        line_id = int(line["line_id"])
        order_id = str(line["order_id"])
        demand = int(line["demand_cases"])
        balance = {u_index[line_id]: 1.0}
        for candidate in candidates_by_order.get(order_id, []):
            candidate_id = str(candidate["candidate_id"])
            key = (line_id, candidate_id)
            index_f = f_index[key]
            balance[index_f] = 1.0
            constraints.add(
                {index_f: 1.0, x_index[candidate_id]: -float(demand)},
                -np.inf,
                0.0,
            )
            if split_picks:
                per_pallet = int(line["cases_per_pallet"])
                constraints.add(
                    {
                        index_f: 1.0,
                        p_index[key]: -float(per_pallet),
                        k_index[key]: -1.0,
                    },
                    0.0,
                    0.0,
                )
                constraints.add(
                    {
                        k_index[key]: 1.0,
                        x_index[candidate_id]: -float(per_pallet - 1),
                    },
                    -np.inf,
                    0.0,
                )
        constraints.add(balance, float(demand), float(demand))

    # POC penalty: activate below the order fill threshold, charge cut cases and
    # cut SKUs, then apply the supplied floor and optional cap exactly.
    if thresholded_penalties:
        for order_id in orders["order_id"].astype(str):
            group = lines_by_order[order_id]
            parameters = penalty_context.parameters_by_order[order_id]
            active = penalty_active_index[order_id]
            total_demand = sum(int(line["demand_cases"]) for line in group)
            required_fill = penalty_context.activation_fill_by_order[order_id]
            if required_fill <= 0:
                constraints.add({active: 1.0}, 0.0, 0.0)
            else:
                trigger_unmet = total_demand - required_fill + 1
                activation = {active: -float(total_demand)}
                for line in group:
                    activation[u_index[int(line["line_id"])]] = 1.0
                constraints.add(activation, -np.inf, float(trigger_unmet - 1))
                constraints.add(
                    activation,
                    float(trigger_unmet - total_demand),
                    np.inf,
                )

            raw_equation = {
                penalty_raw_index[order_id]: 1.0,
                active: -parameters["fixed"],
            }
            for line in group:
                line_id = int(line["line_id"])
                demand = int(line["demand_cases"])
                unmet = u_index[line_id]
                product = penalty_unmet_product_index[line_id]
                cut = penalty_cut_sku_index[line_id]
                constraints.add({product: 1.0, unmet: -1.0}, -np.inf, 0.0)
                constraints.add({product: 1.0, active: -float(demand)}, -np.inf, 0.0)
                constraints.add(
                    {product: 1.0, unmet: -1.0, active: -float(demand)},
                    -float(demand),
                    np.inf,
                )
                constraints.add({cut: 1.0, active: -1.0}, -np.inf, 0.0)
                constraints.add(
                    {unmet: 1.0, cut: -float(demand), active: float(demand)},
                    -np.inf,
                    float(demand),
                )
                raw_equation[product] = -float(line["penalty_per_unfilled_case"])
                raw_equation[cut] = -parameters["per_cut_sku"]
            constraints.add(raw_equation, 0.0, 0.0)

            raw = penalty_raw_index[order_id]
            floor_value = penalty_floor_value_index[order_id]
            value = penalty_value_index[order_id]
            raw_upper = float(upper_bounds[raw])
            big_m = (
                max(
                    1.0,
                    raw_upper,
                    parameters["minimum"],
                    parameters["maximum"],
                )
                + 1.0
            )

            if parameters["minimum"] > 0:
                floor_binary = penalty_floor_binary_index[order_id]
                constraints.add({floor_value: 1.0, raw: -1.0}, 0.0, np.inf)
                constraints.add(
                    {floor_value: 1.0, active: -parameters["minimum"]},
                    0.0,
                    np.inf,
                )
                constraints.add(
                    {floor_value: 1.0, raw: -1.0, floor_binary: -big_m},
                    -np.inf,
                    0.0,
                )
                constraints.add(
                    {
                        floor_value: 1.0,
                        active: -parameters["minimum"],
                        floor_binary: big_m,
                    },
                    -np.inf,
                    big_m,
                )
            else:
                constraints.add({floor_value: 1.0, raw: -1.0}, 0.0, 0.0)

            if parameters["maximum"] > 0:
                cap_binary = penalty_cap_binary_index[order_id]
                maximum = parameters["maximum"]
                constraints.add({cap_binary: 1.0, active: -1.0}, -np.inf, 0.0)
                constraints.add(
                    {floor_value: 1.0, cap_binary: -big_m},
                    -np.inf,
                    maximum,
                )
                constraints.add(
                    {floor_value: 1.0, cap_binary: -big_m},
                    maximum - big_m,
                    np.inf,
                )
                constraints.add(
                    {value: 1.0, floor_value: -1.0, cap_binary: -big_m},
                    -np.inf,
                    0.0,
                )
                constraints.add(
                    {value: 1.0, floor_value: -1.0, cap_binary: big_m},
                    0.0,
                    np.inf,
                )
                constraints.add(
                    {value: 1.0, active: -maximum, cap_binary: big_m},
                    -np.inf,
                    big_m,
                )
                constraints.add(
                    {value: 1.0, active: -maximum, cap_binary: -big_m},
                    -big_m,
                    np.inf,
                )
            else:
                constraints.add({value: 1.0, floor_value: -1.0}, 0.0, 0.0)

    # Projected-ATP protection. A fulfillment at tau consumes every checkpoint t >= tau.
    for inventory in problem.inventory.itertuples(index=False):
        checkpoint = pd.Timestamp(inventory.date)
        coefficients: dict[int, float] = {}
        for line_id in line_ids_by_sku.get(str(inventory.sku_id), []):
            order_id = str(line_by_id[line_id]["order_id"])
            for candidate in candidates_by_order.get(order_id, []):
                if (
                    str(candidate["dc_id"]) == str(inventory.dc_id)
                    and pd.Timestamp(candidate["pgi_date"]) <= checkpoint
                ):
                    index = f_index[(line_id, str(candidate["candidate_id"]))]
                    coefficients[index] = coefficients.get(index, 0.0) + 1.0
        if coefficients:
            constraints.add(
                coefficients,
                -np.inf,
                float(inventory.cumulative_available_cases),
            )

    # Optional exact-date resources.
    if not problem.capacities.empty:
        for capacity in problem.capacities.itertuples(index=False):
            dc_id = str(capacity.dc_id)
            date = pd.Timestamp(capacity.date)
            resource = str(capacity.resource)
            coefficients: dict[int, float] = {}

            if resource == "dock":
                for candidate in candidate_records:
                    if (
                        str(candidate["dc_id"]) == dc_id
                        and pd.Timestamp(candidate["pgi_date"]) == date
                    ):
                        candidate_id = str(candidate["candidate_id"])
                        coefficients[x_index[candidate_id]] = candidate_fixed_consumption(
                            candidate, "dock"
                        )
            elif resource in {
                "throughput_cases",
                "case_pick",
                "pallet_pick",
                "weight",
                "volume",
            }:
                attribute = {
                    "weight": "unit_weight",
                    "volume": "unit_volume",
                }.get(resource)
                if attribute is not None and attribute not in lines.columns:
                    raise ClassicalSolverError(
                        f"Capacity resource {resource!r} requires order_lines.{attribute}"
                    )

                for line in line_records:
                    line_id = int(line["line_id"])
                    for candidate in candidates_by_order.get(str(line["order_id"]), []):
                        if (
                            str(candidate["dc_id"]) != dc_id
                            or pd.Timestamp(candidate["pgi_date"]) != date
                        ):
                            continue
                        key = (line_id, str(candidate["candidate_id"]))
                        if resource == "pallet_pick":
                            index = p_index[key]
                            coefficient = 1.0
                        elif resource == "case_pick" and split_picks:
                            index = k_index[key]
                            coefficient = 1.0
                        else:
                            index = f_index[key]
                            coefficient = 1.0 if attribute is None else float(line[attribute])
                        coefficients[index] = coefficients.get(index, 0.0) + coefficient
            else:
                raise ClassicalSolverError(
                    f"Unsupported capacity resource {resource!r}. "
                    "Add its exact linear consumption rule before enabling it."
                )

            if coefficients:
                constraints.add(coefficients, -np.inf, float(capacity.capacity))

    # Nestle's minimum alternate-fill improvement rule.
    default_lookup = orders.set_index("order_id")["default_dc"].astype(str).to_dict()
    if bool(problem.metadata.get("enforce_min_divert_improvement", False)):
        divert_thresholds = {
            order_id: minimum_divert_fulfillment(problem, order_id)
            for order_id in orders["order_id"].astype(str)
        }
        for candidate in candidate_records:
            order_id = str(candidate["order_id"])
            if str(candidate["dc_id"]) == default_lookup[order_id]:
                continue
            threshold = divert_thresholds[order_id]
            if threshold is None:
                continue
            candidate_id = str(candidate["candidate_id"])
            coefficients = {x_index[candidate_id]: -float(threshold)}
            for line in lines_by_order[order_id]:
                line_id = int(line["line_id"])
                coefficients[f_index[(line_id, candidate_id)]] = 1.0
            constraints.add(coefficients, 0.0, np.inf)

    if minimum_objective is not None:
        # The compiled MILP minimizes the negative business objective. Permit a
        # tiny scale-aware tolerance so an exactly reproducible incumbent is not
        # excluded by floating-point coefficient accumulation.
        tolerance = 1e-9 * max(1.0, abs(float(minimum_objective)))
        constraints.add(
            {
                index: float(coefficient)
                for index, coefficient in enumerate(objective)
                if abs(float(coefficient)) > 0
            },
            -np.inf,
            -float(minimum_objective) + tolerance,
        )

    matrix = coo_matrix(
        (constraints.data, (constraints.rows, constraints.cols)),
        shape=(constraints.row_count, n_variables),
    ).tocsr()
    result = _solve_compiled_milp(
        backend=backend,
        objective=objective,
        integrality=integrality,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        matrix=matrix,
        constraint_lower=np.asarray(constraints.lower, dtype=float),
        constraint_upper=np.asarray(constraints.upper, dtype=float),
        time_limit_seconds=time_limit_seconds,
        mip_relative_gap=mip_relative_gap,
        seed=seed,
        thread_count=thread_count,
    )

    values = np.asarray(result.x, dtype=float)
    assignment_rows: list[dict[str, object]] = []
    fulfillment_rows: list[dict[str, object]] = []
    selected_by_order: dict[str, str | None] = {}

    for order_id in orders["order_id"].astype(str):
        selected = [
            str(candidate["candidate_id"])
            for candidate in candidates_by_order.get(order_id, [])
            if values[x_index[str(candidate["candidate_id"])]] > 0.5
        ]
        if len(selected) > 1:
            raise ClassicalSolverError(
                f"Numerical extraction found multiple candidates for order {order_id}: {selected}"
            )
        candidate_id = selected[0] if selected else None
        selected_by_order[order_id] = candidate_id
        if candidate_id is None:
            assignment_rows.append(
                {
                    "order_id": order_id,
                    "candidate_id": None,
                    "selected_dc": None,
                    "selected_pgi_date": None,
                    "is_unassigned": True,
                    "is_divert": False,
                    "method": "classical",
                }
            )
        else:
            candidate = candidate_lookup[candidate_id]
            dc_id = str(candidate["dc_id"])
            assignment_rows.append(
                {
                    "order_id": order_id,
                    "candidate_id": candidate_id,
                    "selected_dc": dc_id,
                    "selected_pgi_date": pd.Timestamp(candidate["pgi_date"]),
                    "is_unassigned": False,
                    "is_divert": dc_id != default_lookup[order_id],
                    "method": "classical",
                }
            )

    for line in line_records:
        line_id = int(line["line_id"])
        order_id = str(line["order_id"])
        candidate_id = selected_by_order[order_id]
        fulfilled = 0
        selected_dc: str | None = None
        selected_date: pd.Timestamp | None = None
        if candidate_id is not None:
            fulfilled = round(values[f_index[(line_id, candidate_id)]])
            candidate = candidate_lookup[candidate_id]
            selected_dc = str(candidate["dc_id"])
            selected_date = pd.Timestamp(candidate["pgi_date"])
        fulfillment_rows.append(
            {
                "order_id": order_id,
                "sku_id": str(line["sku_id"]),
                "fulfilled_cases": fulfilled,
                "unfulfilled_cases": int(line["demand_cases"]) - fulfilled,
                "selected_dc": selected_dc,
                "selected_pgi_date": selected_date,
            }
        )

    best_bound = None if result.best_bound is None else -float(result.best_bound)
    optimality_gap = result.mip_gap
    return Solution(
        method="classical",
        assignments=pd.DataFrame(assignment_rows),
        fulfillment=pd.DataFrame(fulfillment_rows),
        runtime_seconds=perf_counter() - start,
        raw_objective=-float(result.fun),
        metadata={
            "solver": result.solver,
            "milp_backend": result.backend,
            "status": int(result.status),
            "message": str(result.message),
            "best_bound": best_bound,
            "optimality_gap": optimality_gap,
            "mip_node_count": result.mip_node_count,
            "n_variables": n_variables,
            "n_binary_assignment_variables": len(x_index) + len(z_index),
            "n_fulfillment_variables": len(f_index),
            "n_pick_variables": len(p_index) + len(k_index),
            "n_constraints": constraints.row_count,
            "fixed_assignment_count": len(fixed),
            "eligible_candidates_before_fixed_presolve": len(all_candidates),
            "eligible_candidates_after_fixed_presolve": len(candidates),
            "fixed_candidate_columns_removed": len(all_candidates) - len(candidates),
            "seed": recorded_seed,
            "minimum_objective": minimum_objective,
            "thread_count": (
                None if result.backend == "scipy-highs" else _effective_thread_count(thread_count)
            ),
            "available_cpu_count": available_cpu_count(),
        },
    )
