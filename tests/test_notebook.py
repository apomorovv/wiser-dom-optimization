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


def test_notebook_has_top_cell_controls_and_no_terminal_switches() -> None:
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
    assert "os.environ" not in all_code
    assert "DOMOPT_ENABLE" not in all_code
    assert "NESTLE_EXPERIMENT" not in all_code


def test_each_notebook_experiment_explains_purpose_and_importance() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    experiment_sections = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "markdown":
            continue
        source = _source(cell)
        if source.startswith("## ") and not source.startswith("## 21."):
            experiment_sections.append(source)

    assert len(experiment_sections) == 20
    for source in experiment_sections:
        assert "**Purpose.**" in source
        assert "**Why it matters.**" in source
