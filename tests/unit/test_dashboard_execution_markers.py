from __future__ import annotations

import pandas as pd

from vecm_project.dashboard.app import _trade_markers_for_pair


def test_trade_markers_map_long_and_short_to_stock_buy_sell_events() -> None:
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    prices = pd.DataFrame(
        {
            "ANTM.JK": [10.0, 10.5, 10.2, 10.6],
            "TLKM.JK": [20.0, 19.8, 20.1, 19.9],
        },
        index=idx,
    )
    trades = pd.DataFrame(
        [
            {"side": "LONG", "open_date": "2024-01-02", "close_date": "2024-01-03"},
            {"side": "SHORT", "open_date": "2024-01-04", "close_date": "2024-01-05"},
        ]
    )

    markers = _trade_markers_for_pair(trades, prices, "ANTM.JK", "TLKM.JK")

    # LONG trade mapping:
    # open -> buy lhs, sell rhs ; close -> sell lhs, buy rhs
    assert {"x": "2024-01-02", "y": 10.0, "event": "OPEN_LONG"} in markers["ANTM.JK"]["buy"]
    assert {"x": "2024-01-03", "y": 10.5, "event": "CLOSE_LONG"} in markers["ANTM.JK"]["sell"]
    assert {"x": "2024-01-02", "y": 20.0, "event": "OPEN_LONG"} in markers["TLKM.JK"]["sell"]
    assert {"x": "2024-01-03", "y": 19.8, "event": "CLOSE_LONG"} in markers["TLKM.JK"]["buy"]

    # SHORT trade mapping:
    # open -> sell lhs, buy rhs ; close -> buy lhs, sell rhs
    assert {"x": "2024-01-04", "y": 10.2, "event": "OPEN_SHORT"} in markers["ANTM.JK"]["sell"]
    assert {"x": "2024-01-05", "y": 10.6, "event": "CLOSE_SHORT"} in markers["ANTM.JK"]["buy"]
    assert {"x": "2024-01-04", "y": 20.1, "event": "OPEN_SHORT"} in markers["TLKM.JK"]["buy"]
    assert {"x": "2024-01-05", "y": 19.9, "event": "CLOSE_SHORT"} in markers["TLKM.JK"]["sell"]
