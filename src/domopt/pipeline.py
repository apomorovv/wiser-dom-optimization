"""End-to-end experiment orchestration and artifact writing."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Iterable
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .baselines import (
    solve_default_baseline,
    solve_greedy_baseline,
    solve_polished_greedy,
)
from .classical import solve_classical
from .hybrid import ExactLNSConfig, HybridConfig, solve_exact_lns, solve_hybrid
from .metrics import compute_metrics
from .provenance import runtime_environment, runtime_environment_json
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
        if not canonical.empty:
            canonical = canonical.sort_values(
                list(canonical.columns),
                kind="mergesort",
                na_position="first",
            ).reset_index(drop=True)
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


def current_source_state() -> dict[str, object]:
    """Return a fresh commit, dirty flag, and source-content hash.

    Notebook users commonly edit parameters or source and rerun cells in the
    same kernel.  Recomputing this small provenance record prevents a cached
    pre-edit value from validating a checkpoint produced by different code.
    """

    commit = current_git_commit()
    try:
        root_result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        )
        root = Path(root_result.stdout.strip())
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
        diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout.split(b"\0")
        digest = hashlib.sha256()
        digest.update((commit or "no-commit").encode())
        digest.update(diff)
        for raw_name in sorted(name for name in untracked if name):
            relative = raw_name.decode("utf-8", errors="surrogateescape")
            path = root / relative
            if not path.is_file():
                continue
            digest.update(raw_name)
            digest.update(path.read_bytes())
        return {
            "git_commit": commit,
            "git_dirty": bool(status.strip()),
            "source_state_sha256": digest.hexdigest(),
        }
    except (OSError, subprocess.CalledProcessError):
        return {
            "git_commit": commit,
            "git_dirty": None,
            "source_state_sha256": None,
        }


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
            "runtime_environment": runtime_environment(),
            **current_source_state(),
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
    exact_lns_config: ExactLNSConfig | None = None,
) -> pd.DataFrame:
    """Run requested methods and return a comparable metrics table."""

    requested = [str(method).strip().lower() for method in methods]
    supported = {
        "default",
        "greedy",
        "polished_greedy",
        "exact_lns",
        "classical",
        "hybrid",
    }
    unknown = sorted(set(requested) - supported)
    if unknown:
        raise ValueError(f"Unsupported methods in common pipeline: {unknown}")

    base = Path(output_dir)
    rows: list[dict[str, object]] = []
    for method in requested:
        hybrid_settings: HybridConfig | None = None
        lns_settings: ExactLNSConfig | None = None
        if method == "default":
            solution = solve_default_baseline(problem)
        elif method == "greedy":
            solution = solve_greedy_baseline(problem)
        elif method == "polished_greedy":
            solution = solve_polished_greedy(
                problem,
                time_limit_seconds=time_limit_seconds,
                seed=seed,
            )
        elif method == "exact_lns":
            lns_settings = exact_lns_config or ExactLNSConfig(
                local_time_limit_seconds=min(10.0, time_limit_seconds),
                seed=seed,
            )
            solution = solve_exact_lns(problem, config=lns_settings)
        elif method == "classical":
            solution = solve_classical(
                problem,
                time_limit_seconds=time_limit_seconds,
                seed=seed,
            )
        else:
            hybrid_settings = hybrid_config or HybridConfig(
                seed=seed,
                recourse_time_limit_seconds=min(10.0, time_limit_seconds),
            )
            solution = solve_hybrid(problem, config=hybrid_settings)

        run_path = base / method
        metrics = write_solution_artifacts(
            problem,
            solution,
            run_path,
            experiment_id=experiment_id,
            seed=seed,
            configuration={
                "time_limit_seconds": time_limit_seconds,
                "hybrid": (
                    None if hybrid_settings is None else asdict(hybrid_settings)
                ),
                "exact_lns": None if lns_settings is None else asdict(lns_settings),
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
    comparison = summary.drop(columns=["violations"], errors="ignore").copy()
    if "runtime_environment" in comparison.columns:
        comparison["runtime_environment"] = comparison["runtime_environment"].map(
            lambda value: (
                runtime_environment_json(value) if isinstance(value, dict) else value
            )
        )
    comparison.to_csv(base / "comparison.csv", index=False)
    return summary
