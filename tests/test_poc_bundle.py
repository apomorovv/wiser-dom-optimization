from pathlib import Path

import pytest

from domopt.poc import (
    POC_INPUT_FILENAMES,
    POC_REFERENCE_FILENAMES,
    PocDataError,
    prepare_poc_bundle,
)


def _write_csv(path: Path, value: int = 1) -> None:
    path.write_text(f"value\n{value}\n", encoding="utf-8")


def test_prepare_bundle_normalizes_names_and_excludes_reference_documents(
    tmp_path: Path,
) -> None:
    source = tmp_path / "downloads"
    output = tmp_path / "clean"
    source.mkdir()
    upload_names = {
        "inventory": "input_capacity_planning(1).csv",
        "orders": "input_order data(2).csv",
        "shipping": "input_shipping_cost_data(1).csv",
        "dock": "input_dock_capacity(3).csv",
        "throughput": "input_throughput_capacity(1).csv",
        "order_output": "Output_order_level_data(4).csv",
        "sku_output": "output_order_sku_level_data(4).csv",
    }
    for filename in upload_names.values():
        _write_csv(source / filename)
    _write_csv(source / "_input_order data.csv", value=999)
    (source / "Nestle - WISER Quantum Challenge [SHARED](3).pdf").write_bytes(b"pdf")
    (source / "DOM Equations(1).docx").write_bytes(b"docx")
    (source / "Example(1).xlsx").write_bytes(b"xlsx")

    copied = prepare_poc_bundle(source, output)

    expected_names = set(POC_INPUT_FILENAMES.values()) | set(
        POC_REFERENCE_FILENAMES.values()
    )
    assert {path.name for path in copied.values()} == expected_names
    assert {path.name for path in output.iterdir()} == expected_names
    assert all("(" not in name and " " not in name for name in expected_names)


def test_prepare_bundle_rejects_conflicting_duplicate_inputs(tmp_path: Path) -> None:
    source = tmp_path / "downloads"
    output = tmp_path / "clean"
    source.mkdir()
    for filename in POC_INPUT_FILENAMES.values():
        _write_csv(source / filename)
    _write_csv(source / "input_order data(1).csv", value=2)

    with pytest.raises(PocDataError, match="Ambiguous uploads"):
        prepare_poc_bundle(source, output, include_reference_outputs=False)
