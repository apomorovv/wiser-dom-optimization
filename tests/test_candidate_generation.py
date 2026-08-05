import pandas as pd

from domopt.candidates import filter_feasible_candidates, generate_candidates


def test_candidate_generation_filters_closed_and_late_options() -> None:
    orders = pd.DataFrame(
        [
            {
                "order_id": "O1",
                "default_dc": "D1",
                "requested_delivery_date": "2026-07-15",
            }
        ]
    )
    lanes = pd.DataFrame(
        [
            {
                "order_id": "O1",
                "dc_id": "D1",
                "pgi_date": "2026-07-14",
                "arrival_date": "2026-07-15",
                "shipping_cost": 1,
            },
            {
                "order_id": "O1",
                "dc_id": "D2",
                "pgi_date": "2026-07-14",
                "arrival_date": "2026-07-16",
                "shipping_cost": 2,
            },
        ]
    )
    calendar = pd.DataFrame(
        [
            {"dc_id": "D1", "date": "2026-07-14", "is_open": True},
            {"dc_id": "D2", "date": "2026-07-14", "is_open": True},
        ]
    )
    result = generate_candidates(orders, lanes, calendar=calendar)

    assert list(result["dc_id"]) == ["D1"]
    assert bool(result.iloc[0]["is_default"])
    assert bool(result.iloc[0]["eligible"])


def test_filter_removes_explicitly_ineligible_rows() -> None:
    candidates = pd.DataFrame(
        [
            {
                "candidate_id": "C1",
                "order_id": "O1",
                "dc_id": "D1",
                "pgi_date": "2026-07-14",
                "shipping_cost": 0,
                "is_default": True,
                "eligible": True,
            },
            {
                "candidate_id": "C2",
                "order_id": "O1",
                "dc_id": "D2",
                "pgi_date": "2026-07-14",
                "shipping_cost": 0,
                "is_default": False,
                "eligible": False,
            },
        ]
    )
    result = filter_feasible_candidates(candidates)
    assert list(result["candidate_id"]) == ["C1"]
