from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_final_submission_artifacts_and_language_are_current() -> None:
    expected = [
        "reports/final_report.pdf",
        "reports/business_technical_summary.pdf",
        "reports/planner_view.pdf",
        "reports/final_presentation.pdf",
        "reports/final_presentation.pptx",
        "notebooks/nestle_challenge_experiments.ipynb",
    ]
    assert all((ROOT / relative).is_file() for relative in expected)
    intentionally_excluded = [
        "reports/challenge_submission_report.md",
        "reports/challenge_submission_report.pdf",
    ]
    assert all(not (ROOT / relative).exists() for relative in intentionally_excluded)

    text_paths = [
        ROOT / "README.md",
        ROOT / "reports/final_report.md",
        ROOT / "docs/challenge_checklist.md",
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in text_paths)
    for stale in ["247 / 247", "20-group subset", "512 shots", "full exact MILP"]:
        assert stale not in combined

    assert "In Qiskit, an IBM `backend` is the quantum" in combined
    assert "processor or execution target; it is not an additional solver" in combined


def test_report_builder_preserves_math_as_native_office_math() -> None:
    builder = (ROOT / "scripts/build_submission_documents.py").read_text(
        encoding="utf-8"
    )
    paper = (ROOT / "reports/final_report.md").read_text(encoding="utf-8")

    assert "def _latex_to_text" not in builder
    assert "def _latex_to_omml_xml" in builder
    assert 'qn("m:oMath")' in builder
    assert "Pandoc is required to render report equations" in builder
    assert r"\lvert W_{m_g}\rangle" in paper
    assert r"\lvert e_j\rangle" in paper
