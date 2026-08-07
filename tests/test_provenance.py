from __future__ import annotations

import json
import platform

from domopt import __version__
from domopt.baselines import solve_greedy_baseline
from domopt.data import make_tiny_problem_data
from domopt.experiments import _attempt, _record
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
    for key in ("qiskit_version", "qiskit_ibm_runtime_version"):
        assert environment[key] is None or isinstance(environment[key], str)


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
