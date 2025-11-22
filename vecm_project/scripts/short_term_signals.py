from __future__ import annotations

import logging

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the price panel has a DatetimeIndex."""

    if "date" in df.columns:
        out = df.copy()
        out["date"] = pd.to_datetime(out["date"])
        out = out.set_index("date").sort_index()
        return out
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("price_panel must have a 'date' column or DatetimeIndex")
    return df.sort_index()


def _robust_zscore_cross_section(df: pd.DataFrame, cap: float = 3.0) -> pd.DataFrame:
    """Compute robust cross-sectional z-scores per date."""

    def _z(row: pd.Series) -> pd.Series:
        x = row.to_numpy(dtype=float)
        mask = np.isfinite(x)
        if mask.sum() < 3:
            return pd.Series(np.zeros_like(x), index=row.index)

        x_valid = x[mask]
        med = np.median(x_valid)
        mad = np.median(np.abs(x_valid - med))

        if not np.isfinite(mad) or mad == 0.0:
            mean = np.mean(x_valid)
            std = np.std(x_valid)
            if not np.isfinite(std) or std == 0.0:
                z = np.zeros_like(x)
            else:
                z = (x - mean) / std
        else:
            z = 0.6745 * (x - med) / (mad + 1e-8)

        z = np.clip(z, -cap, cap)
        out = np.zeros_like(x)
        out[mask] = z[mask]
        return pd.Series(out, index=row.index)

    return df.apply(_z, axis=1)


def build_short_term_signals(
    price_panel: pd.DataFrame,
    market_col: str,
    *,
    lookback_mom: int = 21,
    lookback_rev: int = 5,
    lookback_vol: int = 21,
) -> pd.DataFrame:
    """Construct short-term cross-sectional overlay signals.

    Returns a DataFrame indexed by date with per-ticker composite scores where a
    higher value indicates a riskier/over-extended state.
    """

    df = _ensure_datetime_index(price_panel)

    if market_col not in df.columns:
        raise KeyError(f"market_col '{market_col}' not found in price_panel")

    mkt = df[market_col].astype(float)
    eq_cols = [c for c in df.columns if c.endswith(".JK")]
    if not eq_cols:
        raise ValueError("No '.JK' equity columns found in price_panel")

    px = df[eq_cols].astype(float)

    ret = px.pct_change()
    ret_mkt = mkt.pct_change()

    mom_1m = px / px.shift(lookback_mom) - 1.0
    rev_5d = px / px.shift(lookback_rev) - 1.0

    cov = ret.rolling(lookback_vol, min_periods=10).cov(ret_mkt)
    var_mkt = ret_mkt.rolling(lookback_vol, min_periods=10).var()
    beta = cov.div(var_mkt, axis=0)
    resid = ret - beta.mul(ret_mkt, axis=0)
    idio_vol = resid.rolling(lookback_vol, min_periods=10).std()

    monthly_px = px.resample("M").last()
    monthly_ret = monthly_px.pct_change()
    if monthly_ret.shape[0] < 6:
        LOGGER.warning("Monthly history is short; seasonality may be unstable")

    seasonality_template = monthly_ret.groupby(monthly_ret.index.month).mean()
    month_index = px.index.month
    seasonality_daily = seasonality_template.reindex(month_index).copy()
    seasonality_daily.index = px.index
    seasonality_daily = seasonality_daily[eq_cols]

    z_mom = _robust_zscore_cross_section(mom_1m)
    z_rev = _robust_zscore_cross_section(rev_5d)
    z_idio = _robust_zscore_cross_section(idio_vol)
    z_season_raw = _robust_zscore_cross_section(seasonality_daily)
    z_season = -z_season_raw

    score_short = (z_mom + z_rev + z_idio + z_season) / 4.0
    score_short = score_short.reindex(px.index).sort_index()

    out = score_short.copy()
    out.columns = eq_cols
    return out.astype(float)
