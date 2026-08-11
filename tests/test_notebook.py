import json
from pathlib import Path

NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "notebooks"
    / "nestle_challenge_experiments.ipynb"
)


def _source(cell: dict[str, object]) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else str(source)


def test_notebook_has_top_cell_controls_and_no_external_switches() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    bootstrap = _source(code_cells[0])
    setup = _source(code_cells[1])
    all_code = "\n".join(_source(cell) for cell in code_cells)

    assert "pip" in bootstrap
    assert "sys.path.insert" in bootstrap

    for setting in [
        "BUNDLE_DIR",
        "PROFILE",
        "FORCE_RERUN",
        "POC_SETTINGS",
        "PROFILE_OVERRIDES",
        "HYBRID_OVERRIDES",
        "EXACT_LNS_OVERRIDES",
        "ENABLED_EXPERIMENTS",
        "ENABLE_GPU_BENCHMARK",
        "ENABLE_IBM_HARDWARE",
        "IBM_HARDWARE_PROFILE",
        "IBM_SHOTS",
        "IBM_BACKEND_NAME",
    ]:
        assert setting in setup

    # Compute-library thread caps may be defined in the notebook itself. What is
    # prohibited is requiring hidden terminal environment switches to select data,
    # experiments, or IBM execution.
    assert "DOMOPT_ENABLE" not in all_code
    assert "NESTLE_EXPERIMENT" not in all_code


def test_each_notebook_experiment_explains_purpose_and_importance() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    experiment_sections = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "markdown":
            continue
        source = _source(cell)
        if source.startswith("## ") and not source.startswith("## 22."):
            experiment_sections.append(source)

    assert len(experiment_sections) == 21
    for source in experiment_sections:
        assert "**Purpose.**" in source
        assert "**Why it matters.**" in source


def test_notebook_defaults_match_the_attached_final_evidence_profile() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    setup = _source(code_cells[1])

    for expected in [
        'PROFILE = "full"',
        "ENABLE_GPU_BENCHMARK = True",
        "ENABLE_IBM_HARDWARE = True",
        'IBM_HARDWARE_PROFILE = "presentation"',
        "IBM_SHOTS = 8_192",
        'IBM_BACKEND_NAME = "ibm_marrakesh"',
    ]:
        assert expected in setup

    assert "QPU execution backend, not a solver" in setup
