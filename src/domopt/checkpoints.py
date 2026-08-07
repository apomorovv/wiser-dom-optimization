"""Manifest-verified aggregate experiment checkpoints with stable paths."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .pipeline import current_source_state, problem_fingerprint
from .provenance import (
    CHECKPOINT_SCHEMA_VERSION,
    EXPERIMENT_SCHEMA_VERSION,
    runtime_environment,
    runtime_environment_json,
)
from .schemas import ASSUMPTION_VERSION, SCHEMA_VERSION, ProblemData


class StaleCheckpointError(RuntimeError):
    """Raised when a checkpoint does not match the requested experiment identity."""


def challenge_results_root(
    project_root: str | Path,
    *,
    producer: str,
) -> Path:
    """Return the non-overlapping result root for one execution surface.

    ``runs/`` remains reserved for row-level solver artifacts.  Aggregate study
    evidence belongs in ``results/`` and is separated by the process that created
    it so a CLI run cannot overwrite notebook checkpoints (or vice versa).
    """

    normalized = str(producer).strip().lower()
    if normalized not in {"cli", "notebook"}:
        raise ValueError("producer must be 'cli' or 'notebook'")
    return Path(project_root) / "results" / "challenge-study" / normalized


def checkpoint_identity(
    problem: ProblemData,
    *,
    profile: str,
    experiment: str,
    configuration: dict[str, Any],
) -> dict[str, Any]:
    """Return the complete identity promised by the experiment protocol."""

    return {
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "experiment_schema_version": EXPERIMENT_SCHEMA_VERSION,
        "experiment": str(experiment),
        "profile": str(profile),
        "bundle_sha256": problem.metadata.get("bundle_sha256"),
        "problem_sha256": problem_fingerprint(problem),
        "schema_version": problem.metadata.get("schema_version", SCHEMA_VERSION),
        "assumption_version": problem.metadata.get(
            "assumption_version", ASSUMPTION_VERSION
        ),
        "objective_version": "fulfilled-value-minus-thresholded-penalty-minus-shipping-v2",
        "configuration": configuration,
        "runtime_environment": runtime_environment(),
        **current_source_state(),
    }


def checkpoint_key(identity: dict[str, Any]) -> str:
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def checkpoint_run_directory(
    output_root: str | Path,
    identity: dict[str, Any],
) -> Path:
    """Return one stable, human-readable directory per execution profile.

    Configuration, problem, and source hashes remain in each table's manifest,
    where they protect checkpoint reuse without burying outputs below opaque
    directory names.  A stale manifest causes that table to be recomputed and
    replaced in place.
    """

    profile = str(identity["profile"]).strip()
    if not profile or Path(profile).name != profile or profile in {".", ".."}:
        raise ValueError("checkpoint profile must be one safe directory name")
    return Path(output_root) / profile


def write_checkpoint(
    frame: pd.DataFrame,
    csv_path: str | Path,
    identity: dict[str, Any],
) -> tuple[Path, Path]:
    """Atomically replace a checkpoint CSV and its integrity manifest."""

    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = path.with_suffix(".manifest.json")
    csv_temporary = path.with_name(f".{path.name}.tmp")
    manifest_temporary = manifest_path.with_name(f".{manifest_path.name}.tmp")
    frame.to_csv(csv_temporary, index=False)
    manifest = {
        "identity": identity,
        "identity_sha256": checkpoint_key(identity),
        "csv_sha256": hashlib.sha256(csv_temporary.read_bytes()).hexdigest(),
        "rows": len(frame),
        "columns": list(frame.columns),
    }
    manifest_temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    csv_temporary.replace(path)
    manifest_temporary.replace(manifest_path)
    return path, manifest_path


def load_checkpoint(
    csv_path: str | Path,
    identity: dict[str, Any],
    *,
    required_columns: set[str] | None = None,
) -> pd.DataFrame:
    """Load only when manifest, identity, row count, and schema all agree."""

    path = Path(csv_path)
    manifest_path = path.with_suffix(".manifest.json")
    if not path.is_file() or not manifest_path.is_file():
        raise StaleCheckpointError("checkpoint CSV or manifest is missing")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StaleCheckpointError(f"checkpoint manifest is unreadable: {error}") from error
    if manifest.get("identity_sha256") != checkpoint_key(identity):
        raise StaleCheckpointError("checkpoint identity does not match this run")
    if manifest.get("identity") != json.loads(
        json.dumps(identity, sort_keys=True, default=str)
    ):
        raise StaleCheckpointError("checkpoint manifest details do not match this run")
    if manifest.get("csv_sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
        raise StaleCheckpointError("checkpoint CSV content hash does not match its manifest")

    frame = pd.read_csv(path)
    if int(manifest.get("rows", -1)) != len(frame):
        raise StaleCheckpointError("checkpoint row count does not match its manifest")
    if list(manifest.get("columns", [])) != list(frame.columns):
        raise StaleCheckpointError("checkpoint columns do not match its manifest")
    required = required_columns or {
        "experiment",
        "level",
        "method",
        "feasible",
        "experiment_schema_version",
        "problem_sha256",
        "source_state_sha256",
        "runtime_environment",
    }
    missing = required - set(frame.columns)
    if missing:
        raise StaleCheckpointError(
            f"checkpoint is missing required columns {sorted(missing)}"
        )
    if set(frame["experiment"].dropna().astype(str)) != {str(identity["experiment"])}:
        raise StaleCheckpointError("checkpoint contains rows for another experiment")
    expected_schema = str(identity.get("experiment_schema_version"))
    if set(frame["experiment_schema_version"].dropna().astype(str)) != {
        expected_schema
    }:
        raise StaleCheckpointError("checkpoint experiment schema does not match this run")
    expected_environment = identity.get("runtime_environment")
    if not isinstance(expected_environment, dict):
        raise StaleCheckpointError("checkpoint identity has no runtime environment")
    if set(frame["runtime_environment"].dropna().astype(str)) != {
        runtime_environment_json(expected_environment)
    }:
        raise StaleCheckpointError("checkpoint runtime environment does not match this run")
    return frame
