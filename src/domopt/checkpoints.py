"""Content-addressed aggregate experiment checkpoints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .pipeline import current_source_state, problem_fingerprint
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
        "checkpoint_schema_version": 1,
        "experiment_schema_version": 3,
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
        **current_source_state(),
    }


def checkpoint_key(identity: dict[str, Any]) -> str:
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def checkpoint_run_directory(
    output_root: str | Path,
    identity: dict[str, Any],
) -> Path:
    """Scope artifacts by profile, problem, and exact source/config identity."""

    profile = str(identity["profile"])
    problem = str(identity["problem_sha256"])[:12]
    run = checkpoint_key(identity)[:12]
    return Path(output_root) / profile / problem / run


def write_checkpoint(
    frame: pd.DataFrame,
    csv_path: str | Path,
    identity: dict[str, Any],
) -> tuple[Path, Path]:
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = path.with_suffix(".manifest.json")
    frame.to_csv(path, index=False)
    manifest = {
        "identity": identity,
        "identity_sha256": checkpoint_key(identity),
        "csv_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rows": len(frame),
        "columns": list(frame.columns),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
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
    }
    missing = required - set(frame.columns)
    if missing:
        raise StaleCheckpointError(
            f"checkpoint is missing required columns {sorted(missing)}"
        )
    if set(frame["experiment"].dropna().astype(str)) != {str(identity["experiment"])}:
        raise StaleCheckpointError("checkpoint contains rows for another experiment")
    return frame
