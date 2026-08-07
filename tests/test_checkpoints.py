from pathlib import Path

import pandas as pd
import pytest

from domopt.checkpoints import (
    StaleCheckpointError,
    checkpoint_identity,
    checkpoint_run_directory,
    load_checkpoint,
    write_checkpoint,
)
from domopt.data import make_tiny_problem_data


def _frame(identity: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "experiment": identity["experiment"],
                "level": "tiny",
                "method": "greedy",
                "feasible": True,
                "experiment_schema_version": "3",
                "problem_sha256": identity["problem_sha256"],
                "source_state_sha256": identity["source_state_sha256"],
            }
        ]
    )


def test_checkpoint_round_trip_is_content_addressed(tmp_path: Path) -> None:
    problem = make_tiny_problem_data()
    identity = checkpoint_identity(
        problem,
        profile="smoke",
        experiment="solver_comparison",
        configuration={"sizes": [4, 8]},
    )
    run_dir = checkpoint_run_directory(tmp_path, identity)
    path = run_dir / "tables" / "solver_comparison.csv"
    write_checkpoint(_frame(identity), path, identity)

    loaded = load_checkpoint(path, identity)

    assert loaded["feasible"].all()
    assert "smoke" in run_dir.parts


def test_checkpoint_rejects_changed_profile_or_configuration(tmp_path: Path) -> None:
    problem = make_tiny_problem_data()
    original = checkpoint_identity(
        problem,
        profile="smoke",
        experiment="size_scaling",
        configuration={"sizes": [4, 8]},
    )
    path = checkpoint_run_directory(tmp_path, original) / "tables" / "size_scaling.csv"
    write_checkpoint(_frame(original), path, original)
    changed = checkpoint_identity(
        problem,
        profile="full",
        experiment="size_scaling",
        configuration={"sizes": [8, 20, 50]},
    )

    with pytest.raises(StaleCheckpointError, match="identity"):
        load_checkpoint(path, changed)


def test_checkpoint_rejects_manifest_schema_drift(tmp_path: Path) -> None:
    problem = make_tiny_problem_data()
    identity = checkpoint_identity(
        problem,
        profile="smoke",
        experiment="sampler_ablation",
        configuration={"reads": 4},
    )
    path = checkpoint_run_directory(tmp_path, identity) / "tables" / "sampler.csv"
    write_checkpoint(_frame(identity).drop(columns=["method"]), path, identity)

    with pytest.raises(StaleCheckpointError, match="missing required columns"):
        load_checkpoint(path, identity)


def test_checkpoint_rejects_value_tampering_with_same_schema(tmp_path: Path) -> None:
    problem = make_tiny_problem_data()
    identity = checkpoint_identity(
        problem,
        profile="smoke",
        experiment="solver_comparison",
        configuration={"sizes": [4, 8]},
    )
    path = checkpoint_run_directory(tmp_path, identity) / "tables" / "solver.csv"
    frame = _frame(identity)
    write_checkpoint(frame, path, identity)
    tampered = frame.assign(method="unverified")
    tampered.to_csv(path, index=False)

    with pytest.raises(StaleCheckpointError, match="content hash"):
        load_checkpoint(path, identity)
