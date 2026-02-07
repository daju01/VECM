from __future__ import annotations

import datetime as dt
import json
import logging
import pathlib
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

LOGGER = logging.getLogger(__name__)
BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
L2_L3_CACHE_DIR = BASE_DIR / "cache" / "l2_l3"
SHORT_TERM_CACHE_DIR = L2_L3_CACHE_DIR / "short_term_overlays"


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


def _short_term_cache_paths(
    cache_dir: pathlib.Path, data_hash: str
) -> Tuple[pathlib.Path, pathlib.Path, pathlib.Path, pathlib.Path]:
    base = cache_dir / f"short_term_{data_hash}"
    return (
        base.with_suffix(".parquet"),
        base.with_suffix(".mom12.parquet"),
        base.with_suffix(".json"),
        base.with_suffix(".model.joblib"),
    )


def _load_short_term_cache(
    cache_dir: pathlib.Path,
    data_hash: str,
) -> Optional[pd.DataFrame]:
    cache_path, mom12_path, meta_path, _ = _short_term_cache_paths(cache_dir, data_hash)
    if not cache_path.exists():
        return None
    try:
        cached = pd.read_parquet(cache_path)
        if "date" in cached.columns:
            cached = cached.set_index("date")
        cached.index = pd.to_datetime(cached.index)
        cached = cached.sort_index()
        if cached.empty:
            raise ValueError("cached short-term panel is empty")
        if mom12_path.exists():
            mom12 = pd.read_parquet(mom12_path)
            if "date" in mom12.columns:
                mom12 = mom12.set_index("date")
            mom12.index = pd.to_datetime(mom12.index)
            cached.attrs["z_mom12"] = mom12.sort_index()
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta.get("data_hash") != data_hash:
                raise ValueError("cached short-term metadata hash mismatch")
        return cached
    except Exception as exc:
        LOGGER.warning("Failed to load short-term overlay cache at %s: %s", cache_path, exc)
        return None


def _save_short_term_cache(
    cache_dir: pathlib.Path,
    data_hash: str,
    panel: pd.DataFrame,
) -> None:
    cache_path, mom12_path, meta_path, _ = _short_term_cache_paths(cache_dir, data_hash)
    cache_dir.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(cache_path)
    z_mom12 = panel.attrs.get("z_mom12")
    if isinstance(z_mom12, pd.DataFrame):
        z_mom12.to_parquet(mom12_path)
    meta = {
        "data_hash": data_hash,
        "rows": int(panel.shape[0]),
        "cols": int(panel.shape[1]),
        "created_at": dt.datetime.utcnow().isoformat(),
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))


def _train_or_load_ml_model(
    *,
    cache_dir: pathlib.Path,
    data_hash: str,
    features: pd.DataFrame,
    target: pd.Series,
) -> Optional[RandomForestClassifier]:
    _, _, _, model_path = _short_term_cache_paths(cache_dir, data_hash)
    if model_path.exists():
        try:
            return joblib.load(model_path)
        except Exception as exc:
            LOGGER.warning("Failed to load cached short-term ML model: %s", exc)
    if features.empty or target.empty:
        return None
    aligned = features.join(target.rename("target"), how="inner").dropna()
    if len(aligned) < 500:
        LOGGER.info("Skipping ML overlay training; insufficient samples=%d", len(aligned))
        return None
    x = aligned[features.columns].to_numpy(dtype=float)
    y = aligned["target"].to_numpy(dtype=int)
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x, y)
    cache_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return model


def _apply_ml_overlay(
    *,
    cache_dir: pathlib.Path,
    data_hash: str,
    px: pd.DataFrame,
    z_mom12: pd.DataFrame,
    z_mom1: pd.DataFrame,
    z_rev: pd.DataFrame,
    z_idio: pd.DataFrame,
    z_season: pd.DataFrame,
    score_short: pd.DataFrame,
    lookahead_days: int = 5,
) -> pd.DataFrame:
    future_ret = px.shift(-lookahead_days) / px - 1.0
    rev_5d = px / px.shift(5) - 1.0
    target = (future_ret * rev_5d) < 0
    target = target.where(future_ret.notna() & rev_5d.notna())
    target = target.stack(dropna=False)

    feature_panel = pd.concat(
        {
            "z_mom12": z_mom12.stack(dropna=False),
            "z_mom1": z_mom1.stack(dropna=False),
            "z_rev": z_rev.stack(dropna=False),
            "z_idio": z_idio.stack(dropna=False),
            "z_season": z_season.stack(dropna=False),
        },
        axis=1,
    )
    if len(px.index) <= lookahead_days + 1:
        LOGGER.info("ML overlay skipped; insufficient history for lookahead=%d", lookahead_days)
        return score_short
    train_end_date = px.index[-lookahead_days - 1]
    train_mask = feature_panel.index.get_level_values(0) <= train_end_date
    model = _train_or_load_ml_model(
        cache_dir=cache_dir,
        data_hash=data_hash,
        features=feature_panel.loc[train_mask],
        target=target.loc[train_mask].astype(float),
    )
    if model is None:
        LOGGER.info("ML overlay unavailable; using base short-term score")
        return score_short

    cutoff_date = px.index[-1]
    score_mask = feature_panel.index.get_level_values(0) >= cutoff_date
    valid_features = feature_panel.loc[score_mask].dropna()
    if valid_features.empty:
        LOGGER.info("ML overlay skipped; no valid feature rows after cutoff=%s", cutoff_date.date())
        return score_short

    proba = model.predict_proba(valid_features.to_numpy(dtype=float))[:, 1]
    prob_series = pd.Series(proba, index=valid_features.index, name="ml_prob")
    ml_prob = prob_series.unstack().reindex(px.index)
    ml_prob = ml_prob.reindex(columns=score_short.columns)
    ml_score = _robust_zscore_cross_section(ml_prob)
    blended = score_short.copy()
    if cutoff_date in blended.index:
        blended.loc[cutoff_date] = (score_short.loc[cutoff_date] + ml_score.loc[cutoff_date]) / 2.0
    blended.attrs["ml_prob"] = ml_prob
    blended.attrs["ml_score"] = ml_score
    return blended


def build_short_term_signals(
    price_panel: pd.DataFrame,
    market_col: str,
    *,
    data_hash: Optional[str] = None,
    cache_dir: Optional[pathlib.Path] = None,
    lookback_mom_1m: int = 21,
    lookback_rev_5d: int = 5,
    lookback_vol_1m: int = 21,
    lookback_mom_12m: int = 252,
    skip_1m: int = 21,
    ml_lookahead_days: int = 5,
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

    if data_hash:
        cache_root = cache_dir or SHORT_TERM_CACHE_DIR
        out = _apply_ml_overlay(
            cache_dir=cache_root,
            data_hash=data_hash,
            px=px,
            z_mom12=z_mom12.astype(float),
            z_mom1=z_mom1.astype(float),
            z_rev=z_rev.astype(float),
            z_idio=z_idio.astype(float),
            z_season=z_season.astype(float),
            score_short=out,
            lookahead_days=ml_lookahead_days,
        )

    # Simpan z-score 12M momentum di attrs, supaya bisa diambil kalau perlu
    out.attrs["z_mom12"] = z_mom12.astype(float)

    return out


def build_short_term_overlay(
    price_panel: pd.DataFrame,
    market_col: str,
    *,
    data_hash: str,
    cache_dir: Optional[pathlib.Path] = None,
    lookback_mom_1m: int = 21,
    lookback_rev_5d: int = 5,
    lookback_vol_1m: int = 21,
    lookback_mom_12m: int = 252,
    skip_1m: int = 21,
) -> pd.DataFrame:
    """Build or reuse cached short-term overlay keyed by data hash."""
    cache_root = cache_dir or SHORT_TERM_CACHE_DIR
    cached = _load_short_term_cache(cache_root, data_hash)
    if cached is not None:
        LOGGER.info("Loaded cached short-term overlay (hash=%s)", data_hash[:12])
        return cached
    built = build_short_term_signals(
        price_panel,
        market_col,
        data_hash=data_hash,
        cache_dir=cache_root,
        lookback_mom_1m=lookback_mom_1m,
        lookback_rev_5d=lookback_rev_5d,
        lookback_vol_1m=lookback_vol_1m,
        lookback_mom_12m=lookback_mom_12m,
        skip_1m=skip_1m,
    )
    _save_short_term_cache(cache_root, data_hash, built)
    LOGGER.info("Saved short-term overlay cache (hash=%s)", data_hash[:12])
    return built
