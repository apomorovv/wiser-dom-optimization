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
    """Return commit and normalized source identity for checkpoint safety.

    Jupyter writes execution counts and cell outputs while a study is running.
    Those presentation-only changes must not invalidate checkpoints or make the
    source hash drift between experiments. Notebook identity therefore includes
    code-cell source only. ``git_dirty`` means computation-relevant source is
    different from ``HEAD``; ``git_worktree_dirty`` separately records any Git
    change, including notebook outputs and documentation.
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
        tracked = subprocess.run(
            ["git", "ls-files", "--cached", "-z"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout.split(b"\0")
        present = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout.split(b"\0")

        def relevant(relative: str) -> bool:
            path = Path(relative)
            if relative in {"pyproject.toml", "environment.yml"}:
                return True
            return bool(path.parts) and path.parts[0] in {
                "src",
                "scripts",
                "configs",
                "notebooks",
            }

        def normalize_newlines(content: bytes) -> bytes:
            """Canonicalize text line endings so source identity is OS-independent."""

            return content.replace(b"\r\n", b"\n").replace(b"\r", b"\n")

        def normalize_notebook_source(source: object) -> object:
            if isinstance(source, str):
                return source.replace("\r\n", "\n").replace("\r", "\n")
            if isinstance(source, list):
                return [
                    item.replace("\r\n", "\n").replace("\r", "\n")
                    if isinstance(item, str)
                    else item
                    for item in source
                ]
            return source

        def normalized(relative: str, content: bytes | None) -> bytes:
            if content is None:
                return b"<deleted>"
            if Path(relative).suffix.lower() != ".ipynb":
                return normalize_newlines(content)
            try:
                notebook = json.loads(content.decode("utf-8"))
                code_cells = [
                    {
                        "cell_type": "code",
                        "source": normalize_notebook_source(cell.get("source", [])),
                    }
                    for cell in notebook.get("cells", [])
                    if cell.get("cell_type") == "code"
                ]
                return json.dumps(
                    code_cells,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ).encode("utf-8")
            except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
                return normalize_newlines(content)

        tracked_names = {
            name.decode("utf-8", errors="surrogateescape")
            for name in tracked
            if name
        }
        current_names = {
            name.decode("utf-8", errors="surrogateescape")
            for name in present
            if name
        }
        names = sorted(
            relative
            for relative in tracked_names | current_names
            if relevant(relative)
        )
        digest = hashlib.sha256()
        source_dirty = False
        for relative in names:
            path = root / relative
            current_content = path.read_bytes() if path.is_file() else None
            try:
                head_content = subprocess.run(
                    ["git", "show", f"HEAD:{relative}"],
                    cwd=root,
                    check=True,
                    capture_output=True,
                ).stdout
            except subprocess.CalledProcessError:
                head_content = None
            current_normalized = normalized(relative, current_content)
            head_normalized = normalized(relative, head_content)
            source_dirty = source_dirty or current_normalized != head_normalized
            digest.update(relative.encode("utf-8", errors="surrogateescape"))
            digest.update(b"\0")
            digest.update(current_normalized)
            digest.update(b"\0")
        return {
            "git_commit": commit,
            "git_dirty": bool(source_dirty),
            "git_worktree_dirty": bool(status.strip()),
            "source_state_sha256": digest.hexdigest(),
        }
    except (OSError, subprocess.CalledProcessError):
        return {
            "git_commit": commit,
            "git_dirty": None,
            "git_worktree_dirty": None,
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
