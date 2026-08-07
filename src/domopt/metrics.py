"""Common business and computational metrics."""

from __future__ import annotations

from typing import Any

from .objective import evaluate_solution
from .schemas import ProblemData, Solution
from .validation import validate_solution


def compute_metrics(problem: ProblemData, solution: Solution) -> dict[str, Any]:
    validation = validate_solution(problem, solution)
    objective = evaluate_solution(problem, solution)

    fulfillment = solution.fulfillment.copy()
    lines = problem.order_lines.copy()
    merged = lines.merge(
        fulfillment[["order_id", "sku_id", "fulfilled_cases"]],
        on=["order_id", "sku_id"],
        how="left",
        validate="one_to_one",
    )

    total_cases = float(merged["demand_cases"].sum())
    fulfilled_cases = float(merged["fulfilled_cases"].sum())
    total_requested_value = float((merged["unit_value"] * merged["demand_cases"]).sum())

    case_fill_rate = fulfilled_cases / total_cases if total_cases > 0 else 1.0
    value_fill_rate = (
        objective.fulfilled_value / total_requested_value
        if total_requested_value > 0
        else 1.0
    )
    lost_value = total_requested_value - objective.fulfilled_value
    total_business_cost = lost_value + objective.penalty_cost + objective.shipping_cost
    objective_capture_rate = (
        objective.objective_value / total_requested_value
        if total_requested_value > 0
        else 1.0
    )
    assignment_groups = int(
        problem.orders.get("assignment_group", problem.orders["order_id"]).nunique()
    )

    assignments = solution.assignments
    reassigned = int(
        (
            (~assignments["is_unassigned"].astype(bool))
            & assignments["is_divert"].astype(bool)
        ).sum()
    )
    unassigned = int(assignments["is_unassigned"].astype(bool).sum())
    order_to_group = problem.orders.set_index("order_id").get("assignment_group")
    if order_to_group is None:
        reassigned_groups = reassigned
    else:
        diverted_orders = set(
            assignments.loc[
                (~assignments["is_unassigned"].astype(bool))
                & assignments["is_divert"].astype(bool),
                "order_id",
            ].astype(str)
        )
        reassigned_groups = int(
            order_to_group.loc[
                order_to_group.index.astype(str).isin(diverted_orders)
            ].nunique()
        )

    metrics: dict[str, Any] = {
        "method": solution.method,
        "dataset_id": problem.metadata.get("dataset_id", "unknown"),
        "feasible": bool(validation.is_feasible),
        **objective.to_dict(),
        "fulfilled_cases": fulfilled_cases,
        "requested_cases": total_cases,
        "requested_value": total_requested_value,
        "lost_value": float(lost_value),
        "total_business_cost": float(total_business_cost),
        "case_fill_rate": float(case_fill_rate),
        "value_fill_rate": float(value_fill_rate),
        "objective_capture_rate": float(objective_capture_rate),
        "objective_per_requested_case": (
            float(objective.objective_value / total_cases) if total_cases > 0 else 0.0
        ),
        "objective_per_assignment_group": (
            float(objective.objective_value / assignment_groups)
            if assignment_groups > 0
            else 0.0
        ),
        "reassigned_orders": reassigned,
        "reassigned_assignment_groups": reassigned_groups,
        "unassigned_orders": unassigned,
        "runtime_seconds": float(solution.runtime_seconds),
        "best_bound": solution.metadata.get("best_bound"),
        "optimality_gap": solution.metadata.get("optimality_gap"),
        "initial_objective": solution.metadata.get("initial_objective"),
        "hybrid_improvement": (
            solution.metadata.get("improvement")
            if solution.method == "hybrid"
            else None
        ),
        "search_improvement": solution.metadata.get(
            "search_improvement", solution.metadata.get("improvement")
        ),
        "lns_improvement": (
            solution.metadata.get("search_improvement")
            if solution.method == "exact_lns"
            else None
        ),
        "raw_initial_objective": solution.metadata.get("raw_initial_objective"),
        "polished_initial_objective": solution.metadata.get(
            "polished_initial_objective"
        ),
        "initial_polish_improvement": solution.metadata.get(
            "initial_polish_improvement"
        ),
        "total_hybrid_improvement": (
            solution.metadata.get("total_improvement")
            if solution.method == "hybrid"
            else None
        ),
        "total_search_improvement": solution.metadata.get("total_improvement"),
        "sampler_backend": solution.metadata.get("sampler"),
        "execution_class": solution.metadata.get("execution_class"),
        "sampler_calls": solution.metadata.get("sampler_calls"),
        "qpu_calls": solution.metadata.get("qpu_calls"),
        "quantum_simulator_calls": solution.metadata.get(
            "quantum_simulator_calls"
        ),
        "qpu_access_time_microseconds": solution.metadata.get(
            "qpu_access_time_microseconds"
        ),
        "hardware_backend": solution.metadata.get("hardware_backend"),
        "hardware_backend_pending_jobs": solution.metadata.get(
            "hardware_backend_pending_jobs"
        ),
        "hardware_mitigation_strategy": solution.metadata.get(
            "hardware_mitigation_strategy"
        ),
        "hardware_wall_seconds": solution.metadata.get("hardware_wall_seconds"),
        "hardware_queue_seconds": solution.metadata.get("hardware_queue_seconds"),
        "hardware_execution_seconds": solution.metadata.get(
            "hardware_execution_seconds"
        ),
        "hardware_turnaround_seconds": solution.metadata.get(
            "hardware_turnaround_seconds"
        ),
        "hardware_quantum_seconds": solution.metadata.get(
            "hardware_quantum_seconds"
        ),
        "hardware_returned_samples": solution.metadata.get(
            "hardware_returned_samples"
        ),
        "hardware_feasible_shots": solution.metadata.get(
            "hardware_feasible_shots"
        ),
        "hardware_backend_num_qubits": solution.metadata.get(
            "hardware_backend_num_qubits"
        ),
        "hardware_logical_qubits": solution.metadata.get(
            "hardware_logical_qubits"
        ),
        "hardware_transpiled_depth": solution.metadata.get(
            "hardware_transpiled_depth"
        ),
        "hardware_two_qubit_gates": solution.metadata.get(
            "hardware_two_qubit_gates"
        ),
        "hardware_two_qubit_depth": solution.metadata.get(
            "hardware_two_qubit_depth"
        ),
        "hardware_optimal_hit_rate": solution.metadata.get(
            "hardware_optimal_hit_rate"
        ),
        "hardware_optimal_hit_rate_given_feasible": solution.metadata.get(
            "hardware_optimal_hit_rate_given_feasible"
        ),
        "hardware_best_feasible_normalized_gap": solution.metadata.get(
            "hardware_best_feasible_normalized_gap"
        ),
        "accepted_moves": solution.metadata.get("accepted_moves"),
        "raw_one_hot_rate": solution.metadata.get("raw_one_hot_rate"),
        "hybrid_iterations": solution.metadata.get("iterations"),
        "maximum_qubo_variables": solution.metadata.get("maximum_qubo_variables"),
        "maximum_candidates_per_order": solution.metadata.get(
            "maximum_candidates_per_order"
        ),
        "recourse_solves": solution.metadata.get("recourse_solves"),
        "local_solves": solution.metadata.get("local_solves"),
        "assignment_moves": solution.metadata.get("assignment_moves"),
        "maximum_active_groups": solution.metadata.get("maximum_active_groups"),
        "maximum_active_orders": solution.metadata.get("maximum_active_orders"),
        "maximum_local_variables": solution.metadata.get("maximum_local_variables"),
        "maximum_local_constraints": solution.metadata.get(
            "maximum_local_constraints"
        ),
        "maximum_local_mip_nodes": solution.metadata.get(
            "maximum_local_mip_nodes"
        ),
        "model_variables": solution.metadata.get("n_variables"),
        "model_constraints": solution.metadata.get("n_constraints"),
        "mip_node_count": solution.metadata.get("mip_node_count"),
        "initialization_seconds": solution.metadata.get("initialization_seconds"),
        "baseline_initialization_seconds": solution.metadata.get(
            "baseline_initialization_seconds"
        ),
        "initial_polish_seconds": solution.metadata.get("initial_polish_seconds"),
        "qubo_build_seconds": solution.metadata.get("qubo_build_seconds"),
        "sampling_seconds": solution.metadata.get("sampling_seconds"),
        "recourse_seconds": solution.metadata.get("recourse_seconds"),
        "local_solve_seconds": solution.metadata.get("local_solve_seconds"),
        "residualization_seconds": solution.metadata.get(
            "residualization_seconds"
        ),
        "global_validation_seconds": solution.metadata.get(
            "global_validation_seconds"
        ),
        "other_seconds": solution.metadata.get("other_seconds"),
        "remote_enabled": solution.metadata.get("remote_enabled"),
        "qubo_noise_relative_sigma": solution.metadata.get(
            "qubo_noise_relative_sigma"
        ),
        "qaoa_readout_bitflip_probability": solution.metadata.get(
            "qaoa_readout_bitflip_probability"
        ),
        "violations": validation.to_dict(),
    }
    return metrics
