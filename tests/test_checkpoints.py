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
from domopt.provenance import (
    CHECKPOINT_SCHEMA_VERSION,
    EXPERIMENT_SCHEMA_VERSION,
    runtime_environment_json,
)


def _frame(identity: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "experiment": identity["experiment"],
                "level": "tiny",
                "method": "greedy",
                "feasible": True,
                "experiment_schema_version": str(EXPERIMENT_SCHEMA_VERSION),
                "problem_sha256": identity["problem_sha256"],
                "source_state_sha256": identity["source_state_sha256"],
                "runtime_environment": runtime_environment_json(
                    identity["runtime_environment"]
                ),
            }
        ]
    )


def test_checkpoint_round_trip_uses_stable_profile_directory(tmp_path: Path) -> None:
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
    assert run_dir == tmp_path / "smoke"
    assert identity["checkpoint_schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert identity["experiment_schema_version"] == EXPERIMENT_SCHEMA_VERSION
    assert identity["runtime_environment"]["wiser_dom_version"] == "0.5.0"


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


def test_checkpoint_rejects_runtime_environment_row_drift(tmp_path: Path) -> None:
    problem = make_tiny_problem_data()
    identity = checkpoint_identity(
        problem,
        profile="smoke",
        experiment="solver_comparison",
        configuration={"sizes": [4, 8]},
    )
    path = checkpoint_run_directory(tmp_path, identity) / "tables" / "solver.csv"
    frame = _frame(identity)
    frame.loc[:, "runtime_environment"] = runtime_environment_json(
        {**identity["runtime_environment"], "numpy_version": "different"}
    )
    write_checkpoint(frame, path, identity)

    with pytest.raises(StaleCheckpointError, match="runtime environment"):
        load_checkpoint(path, identity)
