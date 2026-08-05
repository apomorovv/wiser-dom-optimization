from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_markdown_code_fences_are_balanced_and_openings_are_tagged() -> None:
    for path in sorted(ROOT.rglob("*.md")):
        if any(part.startswith(".") for part in path.relative_to(ROOT).parts):
            continue
        open_fence = False
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.startswith("```"):
                continue
            marker = line.strip()
            if open_fence:
                assert marker == "```", f"{path}:{line_number}: malformed closing fence"
                open_fence = False
            else:
                assert len(marker) > 3, f"{path}:{line_number}: opening fence needs a language"
                open_fence = True
        assert not open_fence, f"{path}: unclosed code fence"


def test_markdown_uses_github_math_delimiters() -> None:
    for path in sorted(ROOT.rglob("*.md")):
        if any(part.startswith(".") for part in path.relative_to(ROOT).parts):
            continue
        text = path.read_text(encoding="utf-8")
        assert "\\(" not in text and "\\)" not in text, (
            f"{path}: use $...$ for inline GitHub math"
        )
        assert "\\[" not in text and "\\]" not in text, (
            f"{path}: use $$...$$ for display GitHub math"
        )
