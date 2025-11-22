from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """Pastikan index datetime; kalau ada kolom 'date' dipakai sebagai index."""
    if "date" in df.columns:
        out = df.copy()
        out["date"] = pd.to_datetime(out["date"])
        out = out.set_index("date").sort_index()
        return out
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("price_panel must have a 'date' column or DatetimeIndex")
    return df.sort_index()


def _robust_zscore_cross_section(df: pd.DataFrame, cap: float = 3.0) -> pd.DataFrame:
    """Hitung robust z-score cross-section per tanggal (row-wise)."""

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
    lookback_mom_1m: int = 21,
    lookback_rev_5d: int = 5,
    lookback_vol_1m: int = 21,
    lookback_mom_12m: int = 252,
    skip_1m: int = 21,
) -> pd.DataFrame:
    """
    Bangun sinyal jangka pendek + 12M momentum ala factor investing dari panel harga.

    Data yang dipakai hanya:
    - Historical prices (Adj Close) dari Yahoo Finance (sudah ada di adj_close_data.csv).

    Sinyal yang dihitung per saham:
    - 1M momentum        : return 21 hari terakhir (approx 1 bulan).
    - 5D reversal        : return 5 hari terakhir.
    - 12M momentum (12-1): return ~12 bulan dengan skip 1 bulan terakhir.
    - Idiosyncratic vol  : std dev residual CAPM rolling 1 bulan.
    - Seasonality bulanan: rata-rata return bulanan historis -> diproyeksi ke harian.

    Output:
    - DataFrame `score_short` dengan index tanggal, kolom ticker .JK.
      Nilai lebih tinggi = saham kelihatan "lebih jelek" (overextended/risky)
      secara jangka pendek + 12M momentum.

    Tambahan:
    - Out.attrs["z_mom12"] berisi panel z-score 12M momentum (pd.DataFrame)
      kalau kamu mau analisa terpisah.
    """
    df = _ensure_datetime_index(price_panel)

    # Ambil hanya saham ekuitas .JK (sesuaikan kalau universe kamu beda)
    eq_cols = [c for c in df.columns if c.endswith(".JK")]
    if not eq_cols:
        raise ValueError("Tidak ada kolom saham .JK yang ditemukan di price_panel")

    px = df[eq_cols].astype(float)

    if market_col in df.columns:
        mkt = df[market_col].astype(float)
    else:
        LOGGER.warning(
            "market_col '%s' tidak ditemukan; memakai rata-rata equal-weight %d saham sebagai proxy indeks",
            market_col,
            len(eq_cols),
        )
        mkt = px.mean(axis=1)

    # --- Daily returns ---
    ret = px.pct_change(fill_method=None)
    ret_mkt = mkt.pct_change(fill_method=None)

    # --- 1M momentum: return kumulatif ~21 hari ---
    mom_1m = px / px.shift(lookback_mom_1m) - 1.0

    # --- 5D reversal: return 5 hari terakhir ---
    rev_5d = px / px.shift(lookback_rev_5d) - 1.0

    # --- 12M momentum dengan 1M skip (approx 12-1) ---
    # Secara konsep: return dari t-12m sampai t-1m.
    # Di sini kita approx dengan hari trading:
    #   - lookback_mom_12m ≈ 252 hari,
    #   - skip_1m ≈ 21 hari (1 bulan) sebelum t.
    # Jadi:
    #   mom_12m(t) ≈ Px(t - skip_1m) / Px(t - skip_1m - lookback_mom_12m) - 1
    px_1m_before = px.shift(skip_1m)
    px_12m_start = px.shift(skip_1m + lookback_mom_12m)
    mom_12m = px_1m_before / px_12m_start - 1.0

    # --- Idiosyncratic volatility 1M vs indeks (CAPM rolling) ---
    cov = ret.rolling(lookback_vol_1m, min_periods=10).cov(ret_mkt)
    var_mkt = ret_mkt.rolling(lookback_vol_1m, min_periods=10).var()
    beta = cov.div(var_mkt, axis=0)
    resid = ret - beta.mul(ret_mkt, axis=0)
    idio_vol = resid.rolling(lookback_vol_1m, min_periods=10).std()

    # --- Seasonality bulanan: rata-rata return bulanan historis per saham ---
    monthly_px = px.resample("M").last()
    monthly_ret = monthly_px.pct_change(fill_method=None)

    if monthly_ret.shape[0] < 6:
        LOGGER.warning("Data bulanan terlalu pendek, seasonality mungkin kurang stabil")

    seasonality_template = monthly_ret.groupby(monthly_ret.index.month).mean()

    month_index = px.index.month
    seasonality_daily = seasonality_template.reindex(month_index).copy()
    seasonality_daily.index = px.index
    seasonality_daily = seasonality_daily[eq_cols]

    # --- Robust z-score cross-section per fitur ---
    z_mom1 = _robust_zscore_cross_section(mom_1m)
    z_rev = _robust_zscore_cross_section(rev_5d)
    z_idio = _robust_zscore_cross_section(idio_vol)

    # Seasonality: bulan dengan return historis rendah = 'jelek' → sign dibalik
    z_season_raw = _robust_zscore_cross_section(seasonality_daily)
    z_season = -z_season_raw

    # 12M momentum: high 12M momentum (baru naik banyak) = 'jelek' → langsung pakai
    z_mom12 = _robust_zscore_cross_section(mom_12m)

    # Composite 'badness' score: semakin besar = semakin over-extended/berisiko
    score_short = (z_mom12 + z_mom1 + z_rev + z_idio + z_season) / 5.0

    score_short = score_short.reindex(px.index).sort_index()
    out = score_short.copy()
    out.columns = eq_cols
    out = out.astype(float)

    # Simpan z-score 12M momentum di attrs, supaya bisa diambil kalau perlu
    out.attrs["z_mom12"] = z_mom12.astype(float)

    return out
