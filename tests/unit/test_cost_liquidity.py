from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vecm_project.scripts.cost_liquidity import (
    compute_liquidity_metrics,
    compute_nav_cagr_calmar,
)
from vecm_project.scripts.playbook_types import PlaybookConfig
from vecm_project.scripts.playbook_vecm import compute_metrics, execute_trades


def _cfg() -> PlaybookConfig:
    return PlaybookConfig(
        input_file="dummy.csv",
        exit="zexit",
        z_entry=1.0,
        z_exit=0.10,
        z_stop=10.0,
        max_hold=20,
        fee_buy=0.02,
        fee_sell=0.02,
        cost_model="simple",
        ann_days=252,
    )


def _toy_trade_inputs() -> tuple[pd.Series, pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    close1 = pd.Series([100.0, 101.0, 102.0, 101.5, 101.0], index=idx)
    close2 = pd.Series([200.0, 200.0, 200.5, 200.0, 199.5], index=idx)
    lp_pair = pd.DataFrame({"A": np.log(close1), "B": np.log(close2)}, index=idx)
    volume_pair = pd.DataFrame({"A": [1_000_000] * 5, "B": [2_000_000] * 5}, index=idx)
    signals = pd.DataFrame({"long": [0.0, 1.0, 0.0, 0.0, 0.0], "short": [0.0] * 5}, index=idx)
    beta_series = pd.Series(0.0, index=idx)
    zect = pd.Series([1.2, 1.2, 1.0, 0.9, 0.0], index=idx)
    return zect, signals, lp_pair, beta_series, volume_pair


def test_idx_mode_zero_cost_components_yield_ret_net_equal_ret_core() -> None:
    zect, signals, lp_pair, beta_series, volume_pair = _toy_trade_inputs()
    cfg = _cfg()
    cfg.cost_model = "idx_realistic"
    cfg.broker_buy_rate = 0.0
    cfg.broker_sell_rate = 0.0
    cfg.exchange_levy = 0.0
    cfg.sell_tax = 0.0
    cfg.spread_bps = 0.0
    cfg.impact_model = "none"
    cfg.impact_k = 0.0
    result = execute_trades(zect, signals, lp_pair, beta_series, cfg, volume_pair=volume_pair)

    assert float(result.cost.sum()) == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(result.ret.to_numpy(), result.ret_core.to_numpy(), atol=1e-12)
    metrics = compute_metrics(result, cfg, oos_start=lp_pair.index[0].date())
    gross = result.ret_core
    gross_sd = float(gross.std())
    expected_sharpe = float(gross.mean() / gross_sd * np.sqrt(252)) if gross_sd > 0 else 0.0
    assert metrics["sharpe_oos_net"] == pytest.approx(expected_sharpe, rel=1e-9, abs=1e-12)


def test_idx_mode_sell_tax_applied_once_on_sell_notional() -> None:
    zect, signals, lp_pair, beta_series, volume_pair = _toy_trade_inputs()
    cfg = _cfg()
    cfg.cost_model = "idx_realistic"
    cfg.broker_buy_rate = 0.0
    cfg.broker_sell_rate = 0.0
    cfg.exchange_levy = 0.0
    cfg.sell_tax = 0.001
    cfg.spread_bps = 0.0
    cfg.impact_model = "none"
    cfg.impact_k = 0.0
    result = execute_trades(zect, signals, lp_pair, beta_series, cfg, volume_pair=volume_pair)
    cb = result.cost_breakdown
    assert cb is not None
    sell_notional = float(
        cb.loc[cb["dexp1"] < 0, "notional_leg1"].sum()
        + cb.loc[cb["dexp2"] < 0, "notional_leg2"].sum()
    )
    expected_tax = 0.001 * sell_notional
    assert float(cb["cost_sell_tax"].sum()) == pytest.approx(expected_tax, rel=1e-9, abs=1e-12)
    assert float(result.cost.sum()) == pytest.approx(expected_tax, rel=1e-9, abs=1e-12)
    assert float(result.ret_core.sum() - result.ret.sum()) == pytest.approx(expected_tax, rel=1e-9, abs=1e-12)


def test_idx_mode_cost_breakdown_matches_total_and_no_legacy_double_count(
) -> None:
    zect, signals, lp_pair, beta_series, volume_pair = _toy_trade_inputs()
    cfg = _cfg()
    cfg.cost_model = "idx_realistic"
    cfg.broker_buy_rate = 0.0
    cfg.broker_sell_rate = 0.0
    cfg.exchange_levy = 0.0
    cfg.sell_tax = 0.0
    cfg.spread_bps = 0.0
    cfg.impact_model = "none"
    cfg.impact_k = 0.0
    cfg.fee_buy = 0.5
    cfg.fee_sell = 0.5
    result = execute_trades(zect, signals, lp_pair, beta_series, cfg, volume_pair=volume_pair)
    cb = result.cost_breakdown
    assert cb is not None
    summed = cb["cost_broker"] + cb["cost_levy"] + cb["cost_spread"] + cb["cost_impact"] + cb["cost_sell_tax"]
    assert np.allclose(cb["cost_total"].to_numpy(), summed.to_numpy(), atol=1e-12)
    assert float(result.cost.sum()) == pytest.approx(float(cb["cost_total"].sum()), abs=1e-12)
    assert float(result.cost.sum()) == pytest.approx(0.0, abs=1e-12)


def test_liquidity_metrics_and_illiq_cap_are_deterministic() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    close1 = pd.Series([10.0, 10.0, 10.0, 10.0, 10.0], index=idx)
    close2 = pd.Series([20.0, 20.0, 20.0, 20.0, 20.0], index=idx)
    volume1 = pd.Series([100.0, 100.0, 100.0, 100.0, 100.0], index=idx)
    volume2 = pd.Series([50.0, 50.0, 50.0, 50.0, 50.0], index=idx)
    ret1 = pd.Series([0.0, 0.01, 0.02, 0.0, -0.01], index=idx)
    ret2 = pd.Series([0.0, -0.005, 0.005, 0.0, 0.0], index=idx)
    notional1 = pd.Series([0.0, 10.0, 20.0, 0.0, 5.0], index=idx)
    notional2 = pd.Series([0.0, 8.0, 15.0, 0.0, 2.0], index=idx)
    oos_mask = pd.Series([False, False, False, True, True], index=idx)

    metrics = compute_liquidity_metrics(
        close1=close1,
        close2=close2,
        volume1=volume1,
        volume2=volume2,
        ret1=ret1,
        ret2=ret2,
        notional_leg1=notional1,
        notional_leg2=notional2,
        oos_mask=oos_mask,
        adtv_win=2,
        illiq_cap_mode="insample_p80",
        illiq_cap_value=None,
    )

    # Dollar volume is constant 1000 for both legs.
    expected_participation_max_oos = max(max(0.0, 0.0) / 1000.0, max(5.0, 2.0) / 1000.0)
    assert metrics["participation_max"] == pytest.approx(expected_participation_max_oos, abs=1e-12)

    illiq_pair = pd.Series(
        [
            0.0,
            max(abs(0.01) / 1000.0, abs(-0.005) / 1000.0),
            max(abs(0.02) / 1000.0, abs(0.005) / 1000.0),
            0.0,
            max(abs(-0.01) / 1000.0, 0.0),
        ],
        index=idx,
    )
    expected_oos_amihud = float(illiq_pair[oos_mask].mean())
    expected_cap = float(np.percentile(illiq_pair[~oos_mask].values, 80))
    assert metrics["amihud_illiq"] == pytest.approx(expected_oos_amihud, abs=1e-12)
    assert metrics["illiq_cap"] == pytest.approx(expected_cap, abs=1e-12)


def test_calmar_eps_floor_prevents_blow_up() -> None:
    ret = pd.Series([0.0, 0.01, -0.01, 0.0])
    stats = compute_nav_cagr_calmar(ret, ann_days=252, calmar_eps=0.01)
    assert np.isfinite(stats["calmar"])
