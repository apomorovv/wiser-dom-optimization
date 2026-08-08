from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import pytest

from domopt.challenge_outputs import summarize_challenge_outputs


def test_challenge_output_summary_is_aggregate_only() -> None:
    orders = pd.DataFrame(
        [
            {
                "SalesDocument/GroupingIndicator": "private-order",
                "LoadNumber": "private-load",
                "IsDivert": "Non-Default",
                "DefaultDC": "D1",
                "RecommendedDC": "D2",
                "OrderedQty_Cases": 10,
                "Default_Qty_Cases": 5,
                "Divert_Qty_Cases": 9,
            }
        ]
    )
    skus = pd.DataFrame(
        [
            {
                "SalesDocument/GroupingIndicator": "private-order",
                "MaterialNumber": "private-sku",
                "IsDivert": "Non-Default",
                "DefaultDC": "D1",
                "RecommendedDC": "D2",
                "OrderedQty_Cases": 10,
                "Default_Qty_Cases": 5,
                "Divert_Qty_Cases": 9,
            }
        ]
    )
    with TemporaryDirectory() as directory:
        order_path = Path(directory) / "orders.csv"
        sku_path = Path(directory) / "skus.csv"
        orders.to_csv(order_path, index=False)
        skus.to_csv(sku_path, index=False)
        summary = summarize_challenge_outputs(order_path, sku_path)

    assert summary["case_fill_rate"] == pytest.approx(0.9)
    assert summary["diverted_orders"] == 1
    assert summary["contains_raw_identifiers"] is False
    assert "private-order" not in str(summary)

