import pandas as pd

from vecm_project.scripts.data_streaming import _history_to_tidy_frame


def test_history_to_tidy_frame_handles_flat_columns() -> None:
    index = pd.to_datetime(["2025-01-02", "2025-01-03"])
    history = pd.DataFrame(
        {
            "Adj Close": [101.0, 102.5],
            "Close": [101.1, 102.6],
        },
        index=index,
    )
    history.index.name = "Date"

    tidy = _history_to_tidy_frame(history, "ANTM.JK")

    assert list(tidy.columns) == ["Date", "Ticker", "AdjClose"]
    assert tidy["Ticker"].unique().tolist() == ["ANTM.JK"]
    assert tidy["AdjClose"].tolist() == [101.0, 102.5]


def test_history_to_tidy_frame_handles_multiindex_columns() -> None:
    index = pd.to_datetime(["2025-01-02", "2025-01-03"])
    columns = pd.MultiIndex.from_product(
        [["Adj Close", "Close"], ["^JKSE"]],
        names=["Price", "Ticker"],
    )
    history = pd.DataFrame(
        [
            [7163.205078, 7163.205078],
            [7164.429199, 7164.429199],
        ],
        index=index,
        columns=columns,
    )
    history.index.name = "Date"

    tidy = _history_to_tidy_frame(history, "^JKSE")

    assert list(tidy.columns) == ["Date", "Ticker", "AdjClose"]
    assert tidy["Ticker"].unique().tolist() == ["^JKSE"]
    assert tidy["AdjClose"].tolist() == [7163.205078, 7164.429199]
