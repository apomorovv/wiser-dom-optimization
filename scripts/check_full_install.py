#!/usr/bin/env python3
"""Verify that the one-shot ``full`` extra exposed every supported capability."""

from __future__ import annotations

import importlib.util
import platform
from importlib.metadata import PackageNotFoundError, version

REQUIRED_MODULES = {
    "core": ("numpy", "pandas", "scipy", "yaml"),
    "app": ("streamlit",),
    "IBM Runtime": ("qiskit", "qiskit_ibm_runtime"),
    "notebook": ("IPython", "ipykernel", "jupyterlab", "matplotlib", "nbformat"),
    "open-source MILP": ("highspy", "pyscipopt"),
    "optional commercial MILP adapter": ("gurobipy",),
    "tests": ("pytest",),
}


def gpu_package_expected() -> bool:
    """Match the PEP 508 platform marker in ``pyproject.toml``."""

    system = platform.system()
    machine = platform.machine()
    return (system == "Linux" and machine == "x86_64") or (
        system == "Windows" and machine == "AMD64"
    )


def missing_full_components() -> list[str]:
    """Return human-readable missing components without importing GPU drivers."""

    missing: list[str] = []
    for capability, modules in REQUIRED_MODULES.items():
        for module in modules:
            if importlib.util.find_spec(module) is None:
                missing.append(f"{capability}: Python module {module}")
    try:
        version("ruff")
    except PackageNotFoundError:
        missing.append("tests: ruff distribution")
    if gpu_package_expected() and importlib.util.find_spec("cupy") is None:
        missing.append("GPU scoring: Python module cupy")
    return missing


def main() -> int:
    missing = missing_full_components()
    if missing:
        print("The full installation is incomplete:")
        for item in missing:
            print(f"  - {item}")
        return 1
    gpu_status = "included" if gpu_package_expected() else "not supported on this platform"
    print(f"Full installation verified; GPU package: {gpu_status}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
