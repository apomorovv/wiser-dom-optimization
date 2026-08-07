"""Canonical, privacy-safe runtime provenance for reproducible solver evidence."""

from __future__ import annotations

import json
import platform
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version

from . import __version__

RUNTIME_ENVIRONMENT_SCHEMA_VERSION = 1
EXPERIMENT_SCHEMA_VERSION = 5
CHECKPOINT_SCHEMA_VERSION = 3


def _distribution_version(distribution: str) -> str | None:
    """Return an installed distribution version without importing the package."""

    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def runtime_environment() -> dict[str, int | str | None]:
    """Return versions that can change solver behavior or serialized evidence.

    Optional IBM dependencies remain explicit ``None`` values when unavailable,
    which distinguishes a CPU-only environment from an incomplete provenance row.
    The project version comes from the imported source package, so a checkout run
    remains identifiable even when it has not been installed as a distribution.
    """

    return {
        "runtime_environment_schema_version": RUNTIME_ENVIRONMENT_SCHEMA_VERSION,
        "python_version": platform.python_version(),
        "wiser_dom_version": __version__,
        "numpy_version": _distribution_version("numpy"),
        "pandas_version": _distribution_version("pandas"),
        "scipy_version": _distribution_version("scipy"),
        "pyyaml_version": _distribution_version("PyYAML"),
        "qiskit_version": _distribution_version("qiskit"),
        "qiskit_ibm_runtime_version": _distribution_version(
            "qiskit-ibm-runtime"
        ),
    }


def runtime_environment_json(
    environment: Mapping[str, object] | None = None,
) -> str:
    """Return the canonical JSON representation used in aggregate CSV rows."""

    record = runtime_environment() if environment is None else dict(environment)
    return json.dumps(record, sort_keys=True, separators=(",", ":"))
