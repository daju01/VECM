from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vecm_project.scripts.playbook_types import PlaybookConfig
from vecm_project.scripts.playbook_vecm import execute_trades


def _base_cfg(*, fee_buy: float = 0.0, fee_sell: float = 0.0) -> PlaybookConfig:
    return PlaybookConfig(
        input_file="dummy.csv",
        exit="zexit",
        z_exit=0.10,
        z_stop=10.0,
        max_hold=20,
        fee_buy=fee_buy,
        fee_sell=fee_sell,
    )


def test_execute_trades_accumulates_trade_pnl_across_holding_window() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    returns = np.array([0.0, 0.01, 0.01, -0.005, 0.0], dtype=float)
    p1 = 100.0 + np.cumsum(returns)
    p2 = np.full_like(p1, 200.0)

    lp_pair = pd.DataFrame({"A": p1, "B": p2}, index=idx)
    signals = pd.DataFrame({"long": [0.0, 1.0, 0.0, 0.0, 0.0], "short": 0.0}, index=idx)
    beta_series = pd.Series(0.0, index=idx)
    zect = pd.Series([1.0, 1.0, 1.0, 1.0, 0.0], index=idx)

    result = execute_trades(zect, signals, lp_pair, beta_series, _base_cfg())

    assert result.trades.shape[0] == 1
    trade = result.trades.iloc[0]
    assert trade["open_index"] == 1
    assert trade["close_index"] == 4
    assert trade["pnl"] == pytest.approx(0.015, abs=1e-12)
    assert trade["gross_pnl"] == pytest.approx(0.015, abs=1e-12)
    assert trade["total_cost"] == pytest.approx(0.0, abs=1e-12)
    assert trade["holding_days"] == 4


def test_execute_trades_applies_cost_only_on_entry_and_exit_once() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    p1 = np.full(5, 100.0)
    p2 = np.full(5, 200.0)

    lp_pair = pd.DataFrame({"A": p1, "B": p2}, index=idx)
    signals = pd.DataFrame({"long": [0.0, 1.0, 0.0, 0.0, 0.0], "short": 0.0}, index=idx)
    beta_series = pd.Series(0.0, index=idx)
    zect = pd.Series([1.0, 1.0, 1.0, 1.0, 0.0], index=idx)

    fee_buy = 0.001
    fee_sell = 0.002
    result = execute_trades(
        zect,
        signals,
        lp_pair,
        beta_series,
        _base_cfg(fee_buy=fee_buy, fee_sell=fee_sell),
    )

    fee_per_event = (fee_buy + fee_sell) * 1.0
    expected_total_cost = 2 * fee_per_event

    nonzero_cost_idx = np.flatnonzero(result.cost.to_numpy())
    assert nonzero_cost_idx.tolist() == [1, 4]
    assert float(result.cost.sum()) == pytest.approx(expected_total_cost, abs=1e-12)

    trade = result.trades.iloc[0]
    assert trade["gross_pnl"] == pytest.approx(0.0, abs=1e-12)
    assert trade["total_cost"] == pytest.approx(expected_total_cost, abs=1e-12)
    assert trade["pnl"] == pytest.approx(-expected_total_cost, abs=1e-12)
