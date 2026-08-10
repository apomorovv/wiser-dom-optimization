import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_challenge_report_has_a_portal_summary_and_required_baseline_costs() -> None:
    report = (ROOT / "reports/challenge_submission_report.md").read_text(
        encoding="utf-8"
    )
    portal_summary = report.split("## Portal summary", 1)[1].split("## 1.", 1)[0]

    assert len(re.findall(r"\b[\w–-]+\b", portal_summary)) <= 250
    assert "Penalty / requested value" in report
    assert "Shipping / requested value" in report
    assert "0.4817%" in report and "0.6754%" in report
    assert "0.4232%" in report and "0.7060%" in report


def test_final_submission_artifacts_and_language_are_current() -> None:
    expected = [
        "reports/final_report.pdf",
        "reports/challenge_submission_report.pdf",
        "reports/business_technical_summary.pdf",
        "reports/planner_view.pdf",
        "reports/final_presentation.pdf",
        "reports/final_presentation.pptx",
        "notebooks/nestle_challenge_experiments.ipynb",
    ]
    assert all((ROOT / relative).is_file() for relative in expected)

    text_paths = [
        ROOT / "README.md",
        ROOT / "reports/final_report.md",
        ROOT / "reports/challenge_submission_report.md",
        ROOT / "docs/challenge_checklist.md",
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in text_paths)
    for stale in ["247 / 247", "20-group subset", "512 shots", "full exact MILP"]:
        assert stale not in combined

    assert "IBM’s Qiskit `backend` term names the" in combined
    assert "processor or execution target; it is not a solver" in combined


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
