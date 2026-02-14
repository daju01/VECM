import pandas as pd

from vecm_project.scripts.data_streaming import _history_to_tidy_frame


def test_history_to_tidy_frame_handles_flat_columns() -> None:
    index = pd.to_datetime(["2025-01-02", "2025-01-03"])
    history = pd.DataFrame(
        {
            "Adj Close": [101.0, 102.5],
            "Close": [101.1, 102.6],
            "Volume": [1_000_000, 1_200_000],
        },
        index=index,
    )
    history.index.name = "Date"

    tidy = _history_to_tidy_frame(history, "ANTM.JK")

    assert list(tidy.columns) == ["Date", "Ticker", "AdjClose", "Volume"]
    assert tidy["Ticker"].unique().tolist() == ["ANTM.JK"]
    assert tidy["AdjClose"].tolist() == [101.0, 102.5]
    assert tidy["Volume"].tolist() == [1_000_000, 1_200_000]


def test_history_to_tidy_frame_handles_multiindex_columns() -> None:
    index = pd.to_datetime(["2025-01-02", "2025-01-03"])
    columns = pd.MultiIndex.from_product(
        [["Adj Close", "Close", "Volume"], ["^JKSE"]],
        names=["Price", "Ticker"],
    )
    history = pd.DataFrame(
        [
            [7163.205078, 7163.205078, 5_100_000_000],
            [7164.429199, 7164.429199, 4_900_000_000],
        ],
        index=index,
        columns=columns,
    )
    history.index.name = "Date"

    tidy = _history_to_tidy_frame(history, "^JKSE")

    assert list(tidy.columns) == ["Date", "Ticker", "AdjClose", "Volume"]
    assert tidy["Ticker"].unique().tolist() == ["^JKSE"]
    assert tidy["AdjClose"].tolist() == [7163.205078, 7164.429199]
    assert tidy["Volume"].tolist() == [5_100_000_000, 4_900_000_000]
