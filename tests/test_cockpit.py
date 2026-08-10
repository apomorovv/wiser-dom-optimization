from pathlib import Path

COCKPIT = Path(__file__).resolve().parents[1] / "apps" / "solver_cockpit.py"


def test_sidebar_inputs_keep_visible_text_on_white_controls() -> None:
    source = COCKPIT.read_text(encoding="utf-8")

    assert '[data-testid="stSidebar"] *' not in source
    assert '[data-baseweb="input"] input' in source
    assert "-webkit-text-fill-color: #11221c !important" in source
    assert "Entered values remain" in source
