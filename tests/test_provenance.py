from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path

from domopt import __version__
from domopt.baselines import solve_greedy_baseline
from domopt.data import make_tiny_problem_data
from domopt.experiments import _attempt, _record
from domopt.pipeline import current_source_state
from domopt.provenance import (
    RUNTIME_ENVIRONMENT_SCHEMA_VERSION,
    runtime_environment,
    runtime_environment_json,
)


def test_runtime_environment_records_core_and_optional_versions() -> None:
    environment = runtime_environment()

    assert environment["runtime_environment_schema_version"] == (
        RUNTIME_ENVIRONMENT_SCHEMA_VERSION
    )
    assert environment["python_version"] == platform.python_version()
    assert environment["wiser_dom_version"] == __version__
    for key in (
        "numpy_version",
        "pandas_version",
        "scipy_version",
        "pyyaml_version",
    ):
        assert isinstance(environment[key], str) and environment[key]
    for key in ("qiskit_version", "qiskit_ibm_runtime_version", "gurobipy_version"):
        assert environment[key] is None or isinstance(environment[key], str)


def test_notebook_outputs_do_not_change_computation_source_identity(
    tmp_path: Path, monkeypatch
) -> None:
    notebook_path = tmp_path / "notebooks" / "study.ipynb"
    notebook_path.parent.mkdir()
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "source": ["value = 1\n"],
                "outputs": [],
                "execution_count": None,
                "metadata": {},
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook_path.write_text(json.dumps(notebook), encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='identity-test'\n")
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"], cwd=tmp_path, check=True
    )
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"], cwd=tmp_path, check=True, capture_output=True
    )
    monkeypatch.chdir(tmp_path)

    clean = current_source_state()
    notebook["cells"][0]["outputs"] = [{"output_type": "stream", "text": ["1\n"]}]
    notebook["cells"][0]["execution_count"] = 1
    notebook_path.write_text(json.dumps(notebook), encoding="utf-8")
    output_only = current_source_state()

    assert output_only["source_state_sha256"] == clean["source_state_sha256"]
    assert output_only["git_dirty"] is False
    assert output_only["git_worktree_dirty"] is True

    notebook["cells"][0]["source"] = ["value = 2\n"]
    notebook_path.write_text(json.dumps(notebook), encoding="utf-8")
    source_changed = current_source_state()

    assert source_changed["source_state_sha256"] != clean["source_state_sha256"]
    assert source_changed["git_dirty"] is True


def test_runtime_environment_json_is_canonical_and_round_trips() -> None:
    environment = runtime_environment()
    encoded = runtime_environment_json(environment)

    assert json.loads(encoded) == environment
    assert encoded == json.dumps(environment, sort_keys=True, separators=(",", ":"))


def test_successful_and_failed_experiment_rows_record_same_environment() -> None:
    problem = make_tiny_problem_data()
    successful = _record(
        problem,
        solve_greedy_baseline(problem),
        experiment="provenance",
        level="successful",
    )
    failed: list[dict[str, object]] = []

    def fail() -> None:
        raise ValueError("expected test failure")

    _attempt(
        failed,
        problem,
        fail,
        experiment="provenance",
        level="failed",
    )

    assert json.loads(successful["runtime_environment"]) == runtime_environment()
    assert failed[0]["method"] == "failed"
    assert json.loads(failed[0]["runtime_environment"]) == runtime_environment()
