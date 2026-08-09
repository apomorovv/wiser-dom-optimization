from pathlib import Path

import pytest

from domopt.poc import POC_INPUT_FILENAMES, PocDataError
from scripts.run_challenge_study import main, resolve_bundle_dir


def _make_bundle(path: Path) -> Path:
    path.mkdir(parents=True)
    for filename in POC_INPUT_FILENAMES.values():
        (path / filename).write_text("placeholder\n", encoding="utf-8")
    return path


def test_default_bundle_is_repository_relative(tmp_path: Path) -> None:
    project_root = tmp_path / "repository"
    prepared = _make_bundle(project_root / "data" / "raw" / "nestle_challenge")

    resolved = resolve_bundle_dir(
        None,
        project_root=project_root,
        working_directory=tmp_path / "different-working-directory",
    )

    assert resolved == prepared.resolve()


def test_explicit_missing_bundle_is_not_silently_replaced(tmp_path: Path) -> None:
    project_root = tmp_path / "repository"
    prepared = _make_bundle(project_root / "data" / "raw" / "nestle_challenge")
    missing = tmp_path / "Wiser" / "input_data"

    with pytest.raises(PocDataError) as caught:
        resolve_bundle_dir(missing, project_root=project_root)

    message = str(caught.value)
    assert str(missing.resolve()) in message
    assert str(prepared.resolve()) in message
    assert "never replaced silently" in message


def test_missing_bundle_cli_has_no_traceback(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["--bundle-dir", str(tmp_path / "missing"), "--profile", "smoke"])

    assert caught.value.code == 2
    standard_error = capsys.readouterr().err
    assert "Challenge bundle directory does not exist" in standard_error
    assert "Traceback" not in standard_error
