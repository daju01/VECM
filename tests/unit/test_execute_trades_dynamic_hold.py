from __future__ import annotations

import pandas as pd

from vecm_project.scripts.playbook_types import PlaybookConfig
from vecm_project.scripts.playbook_vecm import execute_trades


def _lp_pair(index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "A": [100.0 + i * 0.1 for i in range(len(index))],
            "B": [200.0 for _ in range(len(index))],
        },
        index=index,
    )


def _signals(index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {"long": [1.0] + [0.0] * (len(index) - 1), "short": [0.0] * len(index)},
        index=index,
    )


def _cfg(**kwargs: object) -> PlaybookConfig:
    base = PlaybookConfig(
        input_file="dummy.csv",
        exit="zexit",
        z_entry=1.0,
        z_exit=0.55,
        z_stop=10.0,
        max_hold=2,
        min_hold=0,
        dynamic_hold=False,
        dynamic_hold_max_add=0,
        dynamic_hold_step=0.5,
        fee_buy=0.0,
        fee_sell=0.0,
    )
    cfg_dict = base.to_dict()
    cfg_dict.update(kwargs)
    return PlaybookConfig(**cfg_dict)


def test_min_hold_delays_exit_until_minimum_days() -> None:
    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    zect = pd.Series([1.0, 0.1, 0.1, 0.1, 0.1, 0.1], index=idx)
    beta = pd.Series(0.0, index=idx)

    fast_exit = execute_trades(zect, _signals(idx), _lp_pair(idx), beta, _cfg(min_hold=0, max_hold=10))
    delayed_exit = execute_trades(zect, _signals(idx), _lp_pair(idx), beta, _cfg(min_hold=2, max_hold=10))

    assert int(fast_exit.trades.iloc[0]["close_index"]) == 1
    assert int(delayed_exit.trades.iloc[0]["close_index"]) == 2
    assert int(delayed_exit.trades.iloc[0]["holding_days"]) == 3


def test_dynamic_hold_extends_time_exit_for_stronger_entry_signal() -> None:
    idx = pd.date_range("2024-01-01", periods=7, freq="D")
    zect = pd.Series([2.6] * len(idx), index=idx)
    beta = pd.Series(0.0, index=idx)

    static_hold = execute_trades(
        zect,
        _signals(idx),
        _lp_pair(idx),
        beta,
        _cfg(max_hold=2, dynamic_hold=False),
    )
    dynamic_hold = execute_trades(
        zect,
        _signals(idx),
        _lp_pair(idx),
        beta,
        _cfg(
            max_hold=2,
            dynamic_hold=True,
            dynamic_hold_max_add=3,
            dynamic_hold_step=0.5,
        ),
    )

    static_trade = static_hold.trades.iloc[0]
    dynamic_trade = dynamic_hold.trades.iloc[0]
    assert int(static_trade["max_hold_effective"]) == 2
    assert int(dynamic_trade["max_hold_effective"]) == 5
    assert int(static_trade["close_index"]) == 2
    assert int(dynamic_trade["close_index"]) == 5
