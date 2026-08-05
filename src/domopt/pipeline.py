"""End-to-end experiment orchestration and artifact writing."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from .baselines import solve_default_baseline, solve_greedy_baseline
from .classical import solve_classical
from .hybrid import HybridConfig, solve_hybrid
from .metrics import compute_metrics
from .schemas import ASSUMPTION_VERSION, SCHEMA_VERSION, ProblemData, Solution
from .validation import validate_solution


def _json_default(value: object) -> object:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _csv_ready(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in output.columns:
        if pd.api.types.is_datetime64_any_dtype(output[column]):
            output[column] = output[column].dt.strftime("%Y-%m-%d")
    return output


def problem_fingerprint(problem: ProblemData) -> str:
    """Hash canonical tables and metadata without exposing raw file contents."""

    digest = hashlib.sha256()
    for name, frame in [
        ("orders", problem.orders),
        ("order_lines", problem.order_lines),
        ("inventory", problem.inventory),
        ("candidates", problem.candidates),
        ("capacities", problem.capacities),
        ("calendar", problem.calendar),
    ]:
        digest.update(name.encode())
        canonical = _csv_ready(frame).sort_index(axis=1)
        digest.update(canonical.to_csv(index=False).encode())
    digest.update(json.dumps(problem.metadata, sort_keys=True, default=str).encode())
    return digest.hexdigest()


def current_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_solution_artifacts(
    problem: ProblemData,
    solution: Solution,
    output_dir: str | Path,
    *,
    experiment_id: str,
    seed: int,
    configuration: dict[str, object] | None = None,
) -> dict[str, object]:
    """Validate a solution and write the documented run-directory contract."""

    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    validation = validate_solution(problem, solution)
    metrics = compute_metrics(problem, solution)
    metrics.update(
        {
            "experiment_id": experiment_id,
            "seed": int(seed),
            "schema_version": problem.metadata.get("schema_version", SCHEMA_VERSION),
            "assumption_version": problem.metadata.get(
                "assumption_version", ASSUMPTION_VERSION
            ),
            "git_commit": current_git_commit(),
        }
    )

    _csv_ready(solution.assignments).to_csv(path / "assignments.csv", index=False)
    _csv_ready(solution.fulfillment).to_csv(path / "fulfillment.csv", index=False)
    _write_json(path / "validation.json", validation.to_dict())
    _write_json(path / "metrics.json", metrics)
    _write_json(path / "solver_metadata.json", solution.metadata)
    _write_json(
        path / "config.json",
        {
            "experiment_id": experiment_id,
            "method": solution.method,
            "seed": int(seed),
            **(configuration or {}),
        },
    )
    _write_json(
        path / "input_fingerprint.json",
        {
            "dataset_id": problem.metadata.get("dataset_id", "unknown"),
            "sha256": problem_fingerprint(problem),
            "schema_version": problem.metadata.get("schema_version", SCHEMA_VERSION),
            "assumption_version": problem.metadata.get(
                "assumption_version", ASSUMPTION_VERSION
            ),
        },
    )
    return metrics


def append_registry_row(
    registry_path: str | Path,
    metrics: dict[str, object],
    *,
    run_id: str,
    run_path: str | Path,
    config_path: str | Path,
    notes: str = "",
) -> None:
    path = Path(registry_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "experiment_id": metrics.get("experiment_id"),
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": metrics.get("dataset_id"),
        "schema_version": metrics.get("schema_version"),
        "assumption_version": metrics.get("assumption_version"),
        "method": metrics.get("method"),
        "seed": metrics.get("seed"),
        "git_commit": metrics.get("git_commit"),
        "config_path": str(config_path),
        "run_path": str(run_path),
        "feasible": metrics.get("feasible"),
        "objective_value": metrics.get("objective_value"),
        "case_fill_rate": metrics.get("case_fill_rate"),
        "value_fill_rate": metrics.get("value_fill_rate"),
        "shipping_cost": metrics.get("shipping_cost"),
        "penalty_cost": metrics.get("penalty_cost"),
        "reassigned_orders": metrics.get("reassigned_orders"),
        "runtime_seconds": metrics.get("runtime_seconds"),
        "optimality_gap": metrics.get("optimality_gap"),
        "notes": notes,
    }
    frame = pd.DataFrame([row])
    frame.to_csv(path, mode="a", header=not path.exists() or path.stat().st_size == 0, index=False)


def run_methods(
    problem: ProblemData,
    methods: Iterable[str],
    output_dir: str | Path,
    *,
    experiment_id: str,
    seed: int = 0,
    time_limit_seconds: float = 60.0,
    registry_path: str | Path | None = None,
    hybrid_config: HybridConfig | None = None,
) -> pd.DataFrame:
    """Run requested methods and return a comparable metrics table."""

    requested = [str(method).strip().lower() for method in methods]
    unknown = sorted(set(requested) - {"default", "greedy", "classical", "hybrid"})
    if unknown:
        raise ValueError(f"Unsupported methods in common pipeline: {unknown}")

    base = Path(output_dir)
    rows: list[dict[str, object]] = []
    for method in requested:
        settings: HybridConfig | None = None
        if method == "default":
            solution = solve_default_baseline(problem)
        elif method == "greedy":
            solution = solve_greedy_baseline(problem)
        elif method == "classical":
            solution = solve_classical(
                problem,
                time_limit_seconds=time_limit_seconds,
                seed=seed,
            )
        else:
            settings = hybrid_config or HybridConfig(
                seed=seed,
                recourse_time_limit_seconds=min(10.0, time_limit_seconds),
            )
            solution = solve_hybrid(problem, config=settings)

        run_path = base / method
        metrics = write_solution_artifacts(
            problem,
            solution,
            run_path,
            experiment_id=experiment_id,
            seed=seed,
            configuration={
                "time_limit_seconds": time_limit_seconds,
                "hybrid": None if settings is None else asdict(settings),
            },
        )
        rows.append(metrics)
        if registry_path is not None:
            append_registry_row(
                registry_path,
                metrics,
                run_id=f"{experiment_id}-{method}-{seed}",
                run_path=run_path,
                config_path=run_path / "config.json",
            )

    summary = pd.DataFrame(rows)
    base.mkdir(parents=True, exist_ok=True)
    summary.drop(columns=["violations"], errors="ignore").to_csv(
        base / "comparison.csv", index=False
    )
    return summary
