"""High fidelity VECM/TVECM playbook inspired by the production R script.

The module mirrors the structure of the R reference implementation shared in the
prompt.  It covers the following steps:

* CLI/Config parsing with extensive defaults that can be overridden from the
  command line or by passing a config dictionary to :func:`run_playbook`.
* Data loading & validation with automatic streaming refresh via
  :func:`data_streaming.ensure_price_data`.
* Pre-processing including optional subset filtering, cleaning, rolling window
  truncation, and xts-like conversions implemented with pandas.
* Pair selection through Johansen/Engle-Granger style tests plus correlation
  scoring.
* Dynamic beta tracking using a light-weight Kalman recursion (with fallback
  rolling regressions) that approximates the behaviour of the R version which
  leverages ``KFAS``.
* Gate construction (half-life and correlation) and pseudo TVECM thresholding
  to mimic the requested regime filters.
* Signal generation with confirm streaks, cooldown logic, optional
  momentum add-on, and the enforced long-only patch seen in the R script.
* Trade execution with position book keeping, fee modelling, drawdown stats,
  Kelly-style sizing, and stress metrics.
* Artifact emission (positions/returns/trades/metrics CSV, manifest CSV/JSON)
  into ``out_ms`` along with DuckDB persistence for ``model_checks``.

The implementation keeps the surface area compatible with the rest of the
project: :func:`pipeline` is still exported for the optimiser and API layers
while offering a much richer result payload.
"""
from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass
import datetime as dt
import json
import logging
import math
import os
import pathlib
import sys
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from vecm_project.config.settings import settings
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import VECM

from . import storage
from .cache_keys import hash_config, hash_dataframe
from .data_streaming import ensure_price_data, load_cached_prices
from .ms_spread import compute_regime_prob, fit_ms_spread
from .playbook_types import (
    DecisionParams,
    FeatureBuildResult,
    FeatureBundle,
    FeatureConfig,
    PlaybookConfig,
)
from .short_term_signals import build_short_term_overlay

# ---------------------------------------------------------------------------
# Logging / paths -----------------------------------------------------------
# ---------------------------------------------------------------------------
BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
OUT_DIR = BASE_DIR / "out_ms"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = BASE_DIR / "cache"
PANEL_CACHE_DIR = CACHE_DIR / "panels"
FEATURE_CACHE_DIR = CACHE_DIR / "features"

RUN_ID_FMT = "%Y%m%d_%H%M%S"

LOGGER = storage.configure_logging("playbook_vecm")


_DATAFRAME_CACHE: Dict[str, Tuple[float, pd.DataFrame]] = {}


# ---------------------------------------------------------------------------
# Dataclasses & configuration ------------------------------------------------
# ---------------------------------------------------------------------------


def _default_input_path() -> str:
    """Return the default price cache path without forcing it to exist."""

    data_path = BASE_DIR / "data" / "adj_close_data.csv"
    return str(data_path)


def _safe_ensure_price_cache(
    *,
    tickers: Optional[Iterable[str]] = None,
    force_refresh: bool = False,
    cache_path: Optional[str] = None,
) -> pd.DataFrame:
    """Refresh the price cache and return a normalized DataFrame.

    This wrapper centralises error handling so callers do not depend on the
    details of ``data_streaming.ensure_price_data`` and always receive a
    DataFrame with a ``date`` column when available.
    """

    try:
        ensure_price_data(tickers=list(tickers) if tickers else None, force_refresh=force_refresh)
    except Exception as exc:  # pragma: no cover - download/runtime errors
        LOGGER.warning("Price streaming refresh failed: %s", exc)
    try:
        return load_cached_prices(cache_path)
    except Exception as exc:  # pragma: no cover - missing cache/runtime errors
        LOGGER.warning("Price cache unavailable after refresh: %s", exc)
        return pd.DataFrame()


def _ensure_default_input(path: str) -> str:
    """Ensure the default price cache exists before returning it."""

    if not os.path.exists(path):
        LOGGER.info("Default input %s missing; invoking streaming loader", path)
        _safe_ensure_price_cache(force_refresh=False)
    return path


def _env_default_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _env_default_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


def parse_args(argv: Optional[Iterable[str]] = None) -> PlaybookConfig:
    parser = argparse.ArgumentParser(description="VECM/TVECM Trading Playbook")
    parser.add_argument("input_file", nargs="?", default=None)
    parser.add_argument("--subset", default="")
    parser.add_argument("--method", default="TVECM")
    parser.add_argument("--roll_years", type=float, default=3.0)
    parser.add_argument("--oos_start", default="")
    parser.add_argument("--horizon", default="")
    parser.add_argument("--stage", type=int, default=0)
    parser.add_argument("--notes", default="")
    parser.add_argument("--exit", default="zexit")
    parser.add_argument("--z_entry", type=float, default=None)
    parser.add_argument("--z_exit", type=float, default=0.55)
    parser.add_argument("--z_stop", type=float, default=0.8)
    parser.add_argument("--max_hold", type=int, default=8)
    parser.add_argument("--min_hold", type=int, default=0)
    parser.add_argument("--dynamic_hold", type=int, default=0)
    parser.add_argument("--dynamic_hold_max_add", type=int, default=0)
    parser.add_argument("--dynamic_hold_step", type=float, default=0.5)
    parser.add_argument("--cooldown", type=int, default=1)
    parser.add_argument("--z_auto_method", default="mfpt")
    parser.add_argument("--z_auto", type=float, default=0.7)
    parser.add_argument("--z_entry_cap", type=float, default=0.85)
    parser.add_argument("--gate_require_corr", type=int, default=0)
    parser.add_argument("--gate_corr_min", type=float, default=0.60)
    parser.add_argument("--gate_corr_win", type=int, default=45)
    parser.add_argument("--gate_enforce", type=int, default=1)
    parser.add_argument("--short_filter", type=int, default=0)
    parser.add_argument(
        "--signal_mode",
        default="normal",
        choices=["normal", "long_from_short_only"],
        help="Signal execution mode: normal or contrarian long from short signal only",
    )
    parser.add_argument("--beta_weight", type=int, default=1)
    parser.add_argument("--cost_bps", type=float, default=5.0)
    parser.add_argument(
        "--cost_model",
        default=os.getenv("VECM_COST_MODEL", "simple"),
        choices=["simple", "idx_realistic"],
        help="Transaction cost model: simple (legacy) or idx_realistic",
    )
    parser.add_argument(
        "--broker_buy_rate",
        type=float,
        default=_env_default_float("IDX_BROKER_BUY_RATE", 0.0019),
    )
    parser.add_argument(
        "--broker_sell_rate",
        type=float,
        default=_env_default_float("IDX_BROKER_SELL_RATE", 0.0029),
    )
    parser.add_argument(
        "--exchange_levy",
        type=float,
        default=_env_default_float("IDX_EXCHANGE_LEVY", 0.0),
    )
    parser.add_argument(
        "--sell_tax",
        type=float,
        default=_env_default_float("IDX_SELL_TAX_RATE", 0.001),
    )
    parser.add_argument(
        "--spread_bps",
        type=float,
        default=_env_default_float("IDX_SPREAD_BPS", 20.0),
    )
    parser.add_argument(
        "--impact_model",
        default=os.getenv("IDX_IMPACT_MODEL", "sqrt"),
        choices=["sqrt", "linear", "none"],
    )
    parser.add_argument(
        "--impact_k",
        type=float,
        default=_env_default_float("IDX_IMPACT_K", 1.0),
    )
    parser.add_argument(
        "--adtv_win",
        type=int,
        default=_env_default_int("IDX_ADTV_WIN", 20),
    )
    parser.add_argument(
        "--sigma_win",
        type=int,
        default=_env_default_int("IDX_SIGMA_WIN", 20),
    )
    parser.add_argument(
        "--illiq_cap_mode",
        default=os.getenv("IDX_ILLIQ_CAP_MODE", "insample_p80"),
        choices=["insample_p80", "static"],
    )
    parser.add_argument(
        "--illiq_cap_value",
        type=float,
        default=_env_default_float("IDX_ILLIQ_CAP_VALUE", float("nan")),
    )
    parser.add_argument("--half_life_max", type=float, default=120.0)
    parser.add_argument("--dd_stop", type=float, default=0.25)
    parser.add_argument("--fee_buy", type=float, default=0.0019)
    parser.add_argument("--fee_sell", type=float, default=0.0029)
    parser.add_argument("--p_th", type=float, default=0.50)
    parser.add_argument("--regime_confirm", type=int, default=1)
    parser.add_argument("--long_only", action="store_true", default=True)
    parser.add_argument(
        "--allow_short",
        dest="long_only",
        action="store_false",
        help="Permit both legs instead of forcing long-only entries",
    )
    parser.add_argument("--kelly_frac", type=float, default=0.5)
    parser.add_argument("--vol_cap", type=float, default=0.20)
    parser.add_argument("--ann_days", type=int, default=252)
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--selftest", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--tag", default="")
    parser.add_argument("--mom_enable", type=int, default=0)
    parser.add_argument("--mom_z", type=float, default=0.60)
    parser.add_argument("--mom_k", type=int, default=2)
    parser.add_argument("--mom_gate_k", type=int, default=3)
    parser.add_argument("--mom_cooldown", type=int, default=2)
    parser.add_argument("--outlier_iqr_mult", type=float, default=3.0)
    parser.add_argument("--outlier_max_ratio", type=float, default=0.02)
    args = parser.parse_args(argv)
    input_path = str(args.input_file) if args.input_file else _ensure_default_input(_default_input_path())

    cfg = PlaybookConfig(
        input_file=input_path,
        subset=args.subset,
        method=args.method.upper(),
        roll_years=float(args.roll_years),
        oos_start=args.oos_start,
        horizon=args.horizon,
        stage=int(args.stage),
        notes=args.notes,
        exit=args.exit.lower(),
        z_entry=float(args.z_entry) if args.z_entry is not None else None,
        z_exit=float(args.z_exit),
        z_stop=float(args.z_stop),
        max_hold=int(args.max_hold),
        min_hold=max(0, int(args.min_hold)),
        dynamic_hold=bool(args.dynamic_hold),
        dynamic_hold_max_add=max(0, int(args.dynamic_hold_max_add)),
        dynamic_hold_step=max(float(args.dynamic_hold_step), 1e-6),
        cooldown=int(args.cooldown),
        z_auto_method=args.z_auto_method.lower(),
        z_auto_q=float(args.z_auto),
        z_entry_cap=float(args.z_entry_cap),
        gate_require_corr=int(args.gate_require_corr),
        gate_corr_min=float(args.gate_corr_min),
        gate_corr_win=int(args.gate_corr_win),
        gate_enforce=bool(args.gate_enforce),
        short_filter=bool(args.short_filter),
        signal_mode=str(args.signal_mode).lower(),
        beta_weight=bool(args.beta_weight),
        cost_bps=float(args.cost_bps),
        cost_model=str(args.cost_model).lower(),
        broker_buy_rate=float(args.broker_buy_rate),
        broker_sell_rate=float(args.broker_sell_rate),
        exchange_levy=float(args.exchange_levy),
        sell_tax=float(args.sell_tax),
        spread_bps=float(args.spread_bps),
        impact_model=str(args.impact_model).lower(),
        impact_k=float(args.impact_k),
        adtv_win=max(1, int(args.adtv_win)),
        sigma_win=max(2, int(args.sigma_win)),
        illiq_cap_mode=str(args.illiq_cap_mode).lower(),
        illiq_cap_value=float(args.illiq_cap_value) if args.illiq_cap_value is not None and np.isfinite(args.illiq_cap_value) else None,
        half_life_max=float(args.half_life_max),
        dd_stop=float(args.dd_stop),
        fee_buy=float(args.fee_buy),
        fee_sell=float(args.fee_sell),
        p_th=float(args.p_th),
        regime_confirm=int(args.regime_confirm),
        long_only=bool(args.long_only),
        kelly_frac=float(args.kelly_frac),
        vol_cap=float(args.vol_cap),
        ann_days=int(args.ann_days),
        debug=bool(args.debug),
        selftest=bool(args.selftest),
        seed=args.seed,
        tag=args.tag,
        mom_enable=bool(args.mom_enable),
        mom_z=float(args.mom_z),
        mom_k=int(args.mom_k),
        mom_gate_k=int(args.mom_gate_k),
        mom_cooldown=int(args.mom_cooldown),
        outlier_iqr_mult=float(args.outlier_iqr_mult),
        outlier_max_ratio=float(args.outlier_max_ratio),
    )

    _apply_settings_overrides(cfg)
    return cfg


def _apply_settings_overrides(cfg: PlaybookConfig) -> None:
    overrides = {
        "input_file": str(settings.playbook_input_file) if settings.playbook_input_file else None,
        "subset": settings.playbook_subset,
        "method": settings.playbook_method,
        "roll_years": settings.playbook_roll_years,
        "oos_start": settings.playbook_oos_start,
        "horizon": settings.playbook_horizon,
        "stage": settings.playbook_stage,
        "notes": settings.playbook_notes,
        "exit": settings.playbook_exit,
        "z_entry": settings.playbook_z_entry,
        "z_exit": settings.playbook_z_exit,
        "z_stop": settings.playbook_z_stop,
        "max_hold": settings.playbook_max_hold,
        "cooldown": settings.playbook_cooldown,
        "z_auto_method": settings.playbook_z_auto_method,
        "z_auto_q": settings.playbook_z_auto_q,
        "z_entry_cap": settings.playbook_z_entry_cap,
        "gate_require_corr": settings.playbook_gate_require_corr,
        "gate_corr_min": settings.playbook_gate_corr_min,
        "gate_corr_win": settings.playbook_gate_corr_win,
        "gate_enforce": settings.playbook_gate_enforce,
        "short_filter": settings.playbook_short_filter,
        "beta_weight": settings.playbook_beta_weight,
        "cost_bps": settings.playbook_cost_bps,
        "cost_model": getattr(settings, "playbook_cost_model", None),
        "broker_buy_rate": getattr(settings, "playbook_broker_buy_rate", None),
        "broker_sell_rate": getattr(settings, "playbook_broker_sell_rate", None),
        "exchange_levy": getattr(settings, "playbook_exchange_levy", None),
        "sell_tax": getattr(settings, "playbook_sell_tax", None),
        "spread_bps": getattr(settings, "playbook_spread_bps", None),
        "impact_model": getattr(settings, "playbook_impact_model", None),
        "impact_k": getattr(settings, "playbook_impact_k", None),
        "adtv_win": getattr(settings, "playbook_adtv_win", None),
        "sigma_win": getattr(settings, "playbook_sigma_win", None),
        "illiq_cap_mode": getattr(settings, "playbook_illiq_cap_mode", None),
        "illiq_cap_value": getattr(settings, "playbook_illiq_cap_value", None),
        "half_life_max": settings.playbook_half_life_max,
        "dd_stop": settings.playbook_dd_stop,
        "fee_buy": settings.playbook_fee_buy,
        "fee_sell": settings.playbook_fee_sell,
        "p_th": settings.playbook_p_th,
        "regime_confirm": settings.playbook_regime_confirm,
        "long_only": settings.playbook_long_only,
        "kelly_frac": settings.playbook_kelly_frac,
        "vol_cap": settings.playbook_vol_cap,
        "ann_days": settings.playbook_ann_days,
        "debug": settings.playbook_debug,
        "selftest": settings.playbook_selftest,
        "seed": settings.playbook_seed,
        "tag": settings.playbook_tag,
        "mom_enable": settings.playbook_mom_enable,
        "mom_z": settings.playbook_mom_z,
        "mom_k": settings.playbook_mom_k,
        "mom_gate_k": settings.playbook_mom_gate_k,
        "mom_cooldown": settings.playbook_mom_cooldown,
        "outlier_iqr_mult": settings.playbook_outlier_iqr_mult,
        "outlier_max_ratio": settings.playbook_outlier_max_ratio,
    }
    for key, value in overrides.items():
        if value is not None:
            setattr(cfg, key, value)


# ---------------------------------------------------------------------------
# Utility helpers -----------------------------------------------------------
# ---------------------------------------------------------------------------
def _log_path(run_id: str, suffix: str) -> pathlib.Path:
    return OUT_DIR / f"{suffix}_{run_id}.log"


def _now_utc() -> dt.datetime:
    return dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)


def _ensure_run_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def _write_text(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _resolve_universe(
    universe: Optional[Iterable[str] | str],
    fallback: str,
) -> str:
    if universe is None:
        return fallback
    if isinstance(universe, str):
        return universe
    tokens = [str(token).strip() for token in universe if str(token).strip()]
    return ",".join(tokens) if tokens else fallback


def _df_to_csv(df: pd.DataFrame, path: pathlib.Path) -> None:
    df_out = df.copy()
    if "date" not in df_out.columns:
        if isinstance(df_out.index, pd.DatetimeIndex):
            df_out.insert(0, "date", df_out.index.date)
        else:
            df_out.insert(0, "date", df_out.index)
    path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(path, index=False)


def _safe_cache_slug(value: str) -> str:
    return "".join(char if char.isalnum() or char in ("-", "_", ".") else "_" for char in value)


def _panel_universe_id(tickers: List[str], cfg: PlaybookConfig) -> str:
    payload = {
        "subset": cfg.subset or "all",
        "tickers": sorted(tickers),
        "roll_years": cfg.roll_years,
        "outlier_iqr_mult": cfg.outlier_iqr_mult,
        "outlier_max_ratio": cfg.outlier_max_ratio,
    }
    return hash_config(payload)


def _panel_cache_paths(universe_id: str, data_hash: str) -> tuple[pathlib.Path, pathlib.Path]:
    cache_dir = PANEL_CACHE_DIR / universe_id
    return cache_dir / f"{data_hash}.parquet", cache_dir / f"{data_hash}.json"


def _read_panel_cache(path: pathlib.Path) -> pd.DataFrame:
    panel = pd.read_parquet(path)
    if not isinstance(panel.index, pd.DatetimeIndex):
        if "date" in panel.columns:
            panel = panel.set_index("date")
        panel.index = pd.to_datetime(panel.index)
    return panel.sort_index()


def _write_panel_cache(path: pathlib.Path, panel: pd.DataFrame, meta_path: pathlib.Path, meta: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(path)
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True))


def _feature_cache_key(cfg: PlaybookConfig, data_hash: str, pair: str) -> str:
    payload = {"data_hash": data_hash, "pair": pair, "cfg": cfg.to_dict()}
    return hash_config(payload)


def _feature_cache_path(pair_id: str, feature_key: str) -> pathlib.Path:
    return FEATURE_CACHE_DIR / pair_id / f"{feature_key}.parquet"


def _half_life(series: pd.Series) -> float:
    vals = series.dropna().values
    if len(vals) < 60:
        return float("nan")
    y = vals[1:]
    x = vals[:-1]
    if len(y) < 10:
        return float("nan")
    beta = np.polyfit(x, y, 1)[0]
    if not np.isfinite(beta) or abs(beta) >= 1:
        return float("inf")
    return -math.log(2) / math.log(abs(beta))


def _convergence_stats(zect: pd.Series) -> Tuple[float, float]:
    """
    Estimasi kecepatan konvergensi spread pada full sample.

    Returns
    -------
    alpha_ec : float
        Koefisien error–correction (≈ phi - 1 dari AR(1)).
    half_life : float
        Half-life dalam hari (unit sama dengan step waktu index).
        Mengembalikan +inf kalau tidak bisa diestimasi.
    """

    vals = zect.dropna().values
    if len(vals) < 60:
        return float("nan"), float("inf")

    y = vals[1:]
    x = vals[:-1]
    if len(y) < 10:
        return float("nan"), float("inf")

    beta = np.polyfit(x, y, 1)[0]
    if not np.isfinite(beta) or abs(beta) >= 1:
        return float("nan"), float("inf")

    half_life = -math.log(2.0) / math.log(abs(beta))
    alpha_ec = beta - 1.0

    return float(alpha_ec), float(half_life)


def _run_cor(left: pd.Series, right: pd.Series, window: int) -> pd.Series:
    return left.rolling(window).corr(right)


def _confirm_streak(mask: pd.Series, streak: int) -> pd.Series:
    if streak <= 1:
        return mask.fillna(False)
    arr = mask.fillna(False).to_numpy(dtype=bool)
    out = np.zeros_like(arr, dtype=bool)
    counter = 0
    for idx, flag in enumerate(arr):
        counter = counter + 1 if flag else 0
        out[idx] = counter >= streak
    return pd.Series(out, index=mask.index)


def _resolve_oos_start_date(
    cfg: PlaybookConfig,
    base_index: pd.Index,
    *,
    allow_empty: bool = False,
) -> dt.date:
    if cfg.oos_start:
        return pd.to_datetime(cfg.oos_start).date()
    if not len(base_index):
        if allow_empty:
            return dt.datetime.utcnow().date()
        raise ValueError("Cannot determine OOS start date; no observations available")
    cutoff_idx = int(len(base_index) * 0.7)
    cutoff_idx = min(max(cutoff_idx, 0), len(base_index) - 1)
    return base_index[cutoff_idx].date()


def _build_horizon(index: pd.Index) -> Dict[str, object]:
    if not len(index):
        return {
            "train_obs": 0,
            "test_obs": 0,
            "train_start": "",
            "train_end": "",
            "test_start": "",
            "test_end": "",
        }
    cutoff_idx = int(len(index) * 0.7)
    cutoff_idx = min(max(cutoff_idx, 0), len(index) - 1)
    return {
        "train_obs": int(cutoff_idx),
        "test_obs": len(index) - int(cutoff_idx),
        "train_start": str(index.min().date()),
        "train_end": str(index[cutoff_idx].date()),
        "test_start": str(index[cutoff_idx].date()),
        "test_end": str(index.max().date()),
    }


def _cooldown(mask: pd.Series, cooldown: int) -> pd.Series:
    if cooldown <= 0:
        return mask
    arr = mask.to_numpy(dtype=bool)
    out = np.zeros_like(arr, dtype=bool)
    last_hit = -cooldown - 1
    for idx, flag in enumerate(arr):
        if flag and idx - last_hit > cooldown:
            out[idx] = True
            last_hit = idx
    return pd.Series(out, index=mask.index)


def _mfpt_threshold(z_series: pd.Series, z_grid: Iterable[float], z_exit: float,
                    fee_buy: float, fee_sell: float, ann_days: int) -> Tuple[float, pd.DataFrame]:
    # Simplified Monte-Carlo mean first passage time approximation for AR(1)
    vals = z_series.dropna().to_numpy()
    if len(vals) < 100:
        return 1.5, pd.DataFrame()
    diffs = np.diff(vals)
    if len(diffs) < 10:
        return 1.5, pd.DataFrame()
    phi = np.corrcoef(vals[1:], vals[:-1])[0, 1]
    if not np.isfinite(phi):
        phi = 0.8
    sigma = np.std(diffs)
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 0.1
    results: List[Tuple[float, float, float, float]] = []
    rng = np.random.default_rng(42)
    for thresh in z_grid:
        steps = []
        for _ in range(400):
            z = 0.0
            for step in range(1, 5000):
                z = phi * z + rng.normal(scale=sigma)
                if abs(z) >= thresh:
                    steps.append(step)
                    break
        if not steps:
            continue
        mfpt = float(np.mean(steps))
        trades_pa = ann_days / mfpt
        profit_per = (thresh - z_exit) - (fee_buy + fee_sell)
        results.append((thresh, mfpt, trades_pa, trades_pa * profit_per))
    if not results:
        return 1.5, pd.DataFrame()
    df = pd.DataFrame(results, columns=["U", "mfpt", "trades_pa", "profit_rate"])
    best = df.loc[df["profit_rate"].idxmax()]
    return float(best["U"]), df


# ---------------------------------------------------------------------------
# Data loading & preprocessing ----------------------------------------------
# ---------------------------------------------------------------------------
def load_and_validate_data(path: str) -> pd.DataFrame:
    cached = _DATAFRAME_CACHE.get(path)
    if not os.path.exists(path):
        LOGGER.info("Input file %s missing; invoking streaming loader", path)
        _safe_ensure_price_cache(force_refresh=False)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")
    mtime = os.path.getmtime(path)
    if cached is not None:
        cached_mtime, cached_df = cached
        if mtime == cached_mtime:
            return cached_df.copy(deep=True)
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Input CSV has no rows")
    date_col = None
    for candidate in ["date", "Date", "timestamp", "time", "datetime"]:
        if candidate in df.columns:
            date_col = candidate
            break
    if date_col is None:
        date_col = df.columns[0]
        LOGGER.warning("No explicit date column found; using '%s'", date_col)
    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    df = df.drop_duplicates(subset="date", keep="last")
    quick_flag = os.getenv("VECM_QUICK_TEST", "").lower() in {"1", "true", "yes", "on"}
    quick_days = os.getenv("VECM_QUICK_TEST_DAYS")
    if quick_flag and quick_days:
        try:
            day_count = int(quick_days)
        except ValueError:
            LOGGER.warning("Quick-test override ignored: VECM_QUICK_TEST_DAYS=%r (not an int)", quick_days)
        else:
            if day_count > 0:
                max_date = df["date"].max()
                cutoff = max_date - pd.Timedelta(days=day_count)
                before_rows = len(df)
                df = df.loc[df["date"] >= cutoff]
                LOGGER.info(
                    "Quick-test truncation enabled: VECM_QUICK_TEST_DAYS=%s rows=%d->%d",
                    day_count,
                    before_rows,
                    len(df),
                )
    elif quick_flag and not quick_days:
        LOGGER.info("Quick-test mode enabled but VECM_QUICK_TEST_DAYS is unset; skipping truncation.")
    price_cols = [c for c in df.columns if c != "date"]
    for col in price_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[["date", *price_cols]]
    df = df.dropna(how="all", subset=price_cols)
    df = _maybe_attach_market_index(df)
    price_cols = [c for c in df.columns if c != "date"]
    if len(price_cols) < 2:
        raise ValueError("Need at least two price columns")
    df = df.reset_index(drop=True)
    _DATAFRAME_CACHE[path] = (mtime, df.copy(deep=True))
    return df


def _maybe_attach_market_index(df: pd.DataFrame) -> pd.DataFrame:
    """Add ^JKSE from Yahoo Finance when missing to support factor overlay."""

    market_col = "^JKSE"
    if market_col in df.columns:
        return df

    market = _safe_ensure_price_cache(tickers=[market_col], force_refresh=False)
    if market.empty or market_col not in market.columns or "date" not in market.columns:
        LOGGER.warning("Unduhan %s kosong atau tidak memuat kolom tanggal/harga", market_col)
        return df

    market = market[["date", market_col]].copy()
    market[market_col] = pd.to_numeric(market[market_col], errors="coerce")
    merged = df.merge(market, on="date", how="left")
    filled = merged[market_col].notna().sum()
    if filled:
        LOGGER.info("Menambahkan kolom %s dari Yahoo Finance (%d baris terisi)", market_col, filled)
    else:
        LOGGER.warning("Kolom %s berhasil ditambahkan tetapi seluruh nilai kosong", market_col)
    return merged


def _load_macro_tickers() -> List[str]:
    ticker_path = BASE_DIR / "config" / "ticker_groups.json"
    if not ticker_path.exists():
        return []
    try:
        payload = json.loads(ticker_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        LOGGER.warning("Failed to read macro tickers: %s", exc)
        return []
    if not isinstance(payload, dict):
        return []
    macro = payload.get("macro_tickers", [])
    if not isinstance(macro, list):
        return []
    return [str(ticker) for ticker in macro if ticker]


def _build_macro_exog(price_panel: pd.DataFrame, target_index: pd.Index) -> Optional[pd.DataFrame]:
    macro_tickers = _load_macro_tickers()
    if not macro_tickers:
        return None

    df = price_panel.copy()
    if "date" not in df.columns:
        df = df.reset_index().rename(columns={df.index.name or "index": "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values("date")

    available_cols = [c for c in macro_tickers if c in df.columns]
    missing = [c for c in macro_tickers if c not in df.columns]
    if missing:
        LOGGER.info("Fetching missing macro tickers: %s", ", ".join(missing))
        fetched = _safe_ensure_price_cache(tickers=missing, force_refresh=False)
        if not fetched.empty and "date" in fetched.columns:
            fetched["date"] = pd.to_datetime(fetched["date"], errors="coerce")
            fetched = fetched.dropna(subset=["date"]).sort_values("date")
            df = df.merge(fetched[["date", *[c for c in missing if c in fetched.columns]]], on="date", how="left")
            available_cols = [c for c in macro_tickers if c in df.columns]

    if not available_cols:
        LOGGER.warning("Macro exogenous series unavailable; skipping exog integration")
        return None

    macro_df = df[["date", *available_cols]].copy()
    macro_df = macro_df.set_index("date").sort_index()
    macro_df = macro_df.reindex(target_index).ffill().bfill()
    macro_df = macro_df.replace([np.inf, -np.inf], np.nan)
    macro_df = macro_df.dropna(how="all")
    if macro_df.empty:
        LOGGER.warning("Macro exogenous series empty after alignment")
        return None
    return np.log(macro_df.astype(float))


def preprocess_data(
    df: pd.DataFrame, cfg: PlaybookConfig
) -> Tuple[pd.DataFrame, List[str], Dict[str, float]]:
    LOGGER.info("Pre-processing data (rows=%d, cols=%d)", len(df), len(df.columns))
    if cfg.subset:
        raw_tokens = [token.strip() for token in cfg.subset.split(",") if token.strip()]
        enriched = []
        for tok in raw_tokens:
            if tok.endswith(".JK"):
                enriched.append(tok)
            else:
                enriched.append(f"{tok}.JK")
        available = [c for c in df.columns if c != "date"]
        missing = sorted(set(enriched) - set(available))
        if missing:
            LOGGER.warning("Requested tickers missing: %s", ", ".join(missing))
        keep = [c for c in available if c in enriched]
        if len(keep) < 2:
            raise ValueError("Need at least two valid tickers after subset filter")
        df = df[["date", *keep]]
    df = df.set_index("date")
    df = df.sort_index()
    df = df.loc[:, df.columns.notnull()]
    df = df.dropna(how="all")
    # Clean zero/negative values and light outlier treatment
    outlier_ratios: Dict[str, float] = {}
    for col in df.columns:
        series = df[col]
        series = series.where(series > 0)
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        outlier_ratio = 0.0
        if np.isfinite(iqr) and iqr > 0:
            iqr_mult = float(cfg.outlier_iqr_mult)
            lower = q1 - iqr_mult * iqr
            upper = q3 + iqr_mult * iqr
            mask = (series < lower) | (series > upper)
            if mask.any():
                outlier_ratio = float(mask.sum() / max(series.notna().sum(), 1))
                series = series.clip(lower=lower, upper=upper)
        outlier_ratios[col] = outlier_ratio
        df[col] = series.interpolate(limit=5)
    # If some tickers stop updating (e.g. due to offline cache issues) the tail of
    # the dataset can be filled with NaNs for those symbols. Trim the history to
    # the last date where all *retained* tickers still have observations so the
    # downstream NA-ratio check does not discard the entire series.
    last_valid_dates = []
    for col in df.columns:
        last_valid = df[col].last_valid_index()
        if last_valid is not None:
            last_valid_dates.append(last_valid)
    if last_valid_dates:
        cutoff = min(last_valid_dates)
        if cutoff < df.index.max():
            df = df.loc[:cutoff]
    na_ratios = df.isna().mean()
    drop_cols = [c for c, ratio in na_ratios.items() if ratio > 0.20]
    if drop_cols:
        LOGGER.warning("Dropping >20%% NA series: %s", ", ".join(drop_cols))
        df = df.drop(columns=drop_cols)
    rolling_window = min(60, len(df))
    critical_window = min(30, len(df))
    rolling_threshold = 0.35
    critical_threshold = 0.25
    density_drop_cols: List[str] = []
    if rolling_window >= 10 or critical_window >= 5:
        for col in df.columns:
            series = df[col]
            rolling_ratio = np.nan
            if rolling_window >= 10:
                rolling_ratio = (
                    series.isna()
                    .rolling(window=rolling_window, min_periods=rolling_window)
                    .mean()
                    .max()
                )
            tail_ratio = np.nan
            if critical_window >= 5:
                tail_ratio = series.isna().tail(critical_window).mean()
            if (
                (np.isfinite(rolling_ratio) and rolling_ratio > rolling_threshold)
                or (np.isfinite(tail_ratio) and tail_ratio > critical_threshold)
            ):
                density_drop_cols.append(col)
                LOGGER.warning(
                    "Dropping %s due to poor density (rolling_na=%.2f over %d, tail_na=%.2f over %d)",
                    col,
                    float(rolling_ratio) if np.isfinite(rolling_ratio) else float("nan"),
                    rolling_window,
                    float(tail_ratio) if np.isfinite(tail_ratio) else float("nan"),
                    critical_window,
                )
    if density_drop_cols:
        df = df.drop(columns=density_drop_cols)
    if cfg.roll_years > 0 and not df.empty:
        window = int(cfg.roll_years * 365.25)
        cutoff = df.index.max() - pd.Timedelta(days=window)
        cutoff = max(cutoff, df.index.min() + pd.Timedelta(days=250))
        df = df[df.index >= cutoff]
    if len(df.columns) < 2:
        raise ValueError("Insufficient columns after cleaning")
    if len(df) < 100:
        raise ValueError("Insufficient history after preprocessing")
    return df, list(df.columns), outlier_ratios


# ---------------------------------------------------------------------------
# Pair selection -------------------------------------------------------------
# ---------------------------------------------------------------------------
def _align_exog(
    log_prices: pd.DataFrame,
    exog: Optional[pd.DataFrame],
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    if exog is None or exog.empty:
        return log_prices, None
    exog_aligned = exog.reindex(log_prices.index).ffill().bfill()
    combined = pd.concat([log_prices, exog_aligned], axis=1).replace([np.inf, -np.inf], np.nan)
    combined = combined.dropna()
    if combined.empty:
        LOGGER.warning("Exog alignment produced empty panel; proceeding without exogenous inputs")
        return log_prices, None
    log_aligned = combined[log_prices.columns]
    exog_aligned = combined[exog_aligned.columns]
    return log_aligned, exog_aligned


def _johansen_beta(log_prices: pd.DataFrame, exog: Optional[pd.DataFrame] = None) -> float:
    if len(log_prices) < 120:
        return float("nan")
    log_prices, exog = _align_exog(log_prices, exog)
    if len(log_prices) < 120:
        return float("nan")
    vecm_kwargs = {"k_ar_diff": 1, "deterministic": "ci"}
    try:
        vecm = VECM(log_prices, coint_rank=1, exog=exog, **vecm_kwargs)
    except TypeError:
        try:
            vecm = VECM(log_prices, rank=1, exog=exog, **vecm_kwargs)
        except TypeError:
            vecm = VECM(log_prices, rank=1, **vecm_kwargs)
    result = vecm.fit()
    beta = result.beta[:, 0]
    if len(beta) != 2 or not np.all(np.isfinite(beta)):
        return float("nan")
    return float(-beta[0] / beta[1])


def _eg_beta(log_prices: pd.DataFrame) -> float:
    arr = log_prices.dropna()
    if len(arr) < 60:
        return float("nan")
    y = arr.iloc[:, 0]
    x = arr.iloc[:, 1]
    beta = np.dot(x, y) / np.dot(x, x)
    if not np.isfinite(beta):
        return float("nan")
    return float(beta)


def _adf_pvalue(spread: pd.Series) -> float:
    spread = spread.dropna()
    if len(spread) < 60:
        return 1.0
    try:
        pvalue = adfuller(spread)[1]
    except Exception:
        pvalue = 1.0
    return float(pvalue)


class PairSelectionError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        pair: Optional[str] = None,
        observations: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.pair = pair
        self.observations = observations


def select_pair(
    df: pd.DataFrame,
    cfg: PlaybookConfig,
    outlier_ratios: Optional[Mapping[str, float]] = None,
    *,
    macro_exog: Optional[pd.DataFrame] = None,
) -> Tuple[str, str, float]:
    tickers = list(df.columns)
    max_ratio = float(cfg.outlier_max_ratio)
    outlier_ratios = outlier_ratios or {}
    if len(tickers) == 2:
        if any(outlier_ratios.get(t, 0.0) > max_ratio for t in tickers):
            raise RuntimeError(
                "Pair rejected due to extreme-value ratio exceeding threshold "
                f"({max_ratio:.2%})."
            )
        beta = _johansen_beta(np.log(df.iloc[:, :2]), exog=macro_exog)
        lp = np.log(df.iloc[:, :2].dropna())
        obs = len(lp)
        if obs < 120:
            LOGGER.warning(
                "Insufficient log-price observations for %s~%s (%d < 120); marking pair invalid",
                tickers[0],
                tickers[1],
                obs,
            )
            raise PairSelectionError(
                "Insufficient log-price observations for Johansen beta",
                pair=f"{tickers[0]}~{tickers[1]}",
                observations=obs,
            )
        beta = _johansen_beta(lp, exog=macro_exog)
        if not np.isfinite(beta):
            beta = 1.0
        return tickers[0], tickers[1], beta
    best_pair: Optional[Tuple[str, str, float]] = None
    best_score = -np.inf
    combos = list(zip(*np.triu_indices(len(tickers), k=1)))
    rng = np.random.default_rng(123)
    if len(combos) > 200:
        combos = rng.choice(combos, size=200, replace=False)
    skipped_obs = 0
    for idx0, idx1 in combos:
        t1, t2 = tickers[idx0], tickers[idx1]
        if outlier_ratios.get(t1, 0.0) > max_ratio or outlier_ratios.get(t2, 0.0) > max_ratio:
            LOGGER.info(
                "Skipping pair %s/%s due to extreme-value ratio (%.2f%%, %.2f%%)",
                t1,
                t2,
                outlier_ratios.get(t1, 0.0) * 100.0,
                outlier_ratios.get(t2, 0.0) * 100.0,
            )
            continue
        lp = np.log(df[[t1, t2]].dropna())
        if len(lp) < 120:
            LOGGER.info(
                "Skipping pair %s~%s due to insufficient log-price observations (%d < 120)",
                t1,
                t2,
                len(lp),
            )
            skipped_obs += 1
            continue
        beta = _johansen_beta(lp, exog=macro_exog)
        if not np.isfinite(beta):
            beta = _eg_beta(lp)
        if not np.isfinite(beta):
            continue
        spread = lp.iloc[:, 0] - beta * lp.iloc[:, 1]
        pvalue = _adf_pvalue(spread)
        corr = lp.diff().corr().iloc[0, 1]
        score = -math.log10(max(pvalue, 1e-6)) + max(0.0, corr or 0.0)
        if score > best_score:
            best_score = score
            best_pair = (t1, t2, beta)
    if best_pair is None:
        if skipped_obs:
            LOGGER.warning(
                "No valid pairs found; %d pairs skipped due to <120 observations",
                skipped_obs,
            )
            raise PairSelectionError(
                "No valid pairs available after minimum observation filter",
                observations=skipped_obs,
            )
        raise RuntimeError("Pair selection failed. Try specifying --subset")
    return best_pair


# ---------------------------------------------------------------------------
# Dynamic beta & gate construction ------------------------------------------
# ---------------------------------------------------------------------------
def kalman_dynamic_beta(y: pd.Series, x: pd.Series, beta0: float) -> Tuple[pd.Series, pd.Series]:
    # Simple scalar Kalman recursion akin to the R KFAS model
    q_var = 1e-5
    h_var = 1e-4
    beta = beta0
    p_var = 1.0
    betas = []
    idx = []
    for timestamp, (yi, xi) in zip(y.index, zip(y.values, x.values)):
        if not (np.isfinite(yi) and np.isfinite(xi)):
            betas.append(beta)
            idx.append(timestamp)
            continue
        p_var = p_var + q_var
        s = xi * p_var * xi + h_var
        if s <= 0:
            s = h_var
        k_gain = p_var * xi / s
        resid = yi - beta * xi
        beta = beta + k_gain * resid
        p_var = (1 - k_gain * xi) * p_var
        betas.append(beta)
        idx.append(timestamp)
    beta_series = pd.Series(betas, index=y.index)
    ect = y - beta_series * x
    return beta_series, ect


# ---------------------------------------------------------------------------
# Signal & execution --------------------------------------------------------
# ---------------------------------------------------------------------------
def build_signals(
    zect: pd.Series,
    cfg: PlaybookConfig,
    gates: pd.Series,
    *,
    delta_score: Optional[pd.Series] = None,
    p_regime: Optional[pd.Series] = None,
    delta_mom12: Optional[pd.Series] = None,
) -> Tuple[pd.DataFrame, float]:
    # Determine z threshold
    if cfg.z_auto_method == "mfpt":
        auto_th, _ = _mfpt_threshold(
            zect,
            np.arange(0.6, 2.55, 0.1),
            cfg.z_exit,
            cfg.fee_buy,
            cfg.fee_sell,
            cfg.ann_days,
        )
    else:
        vals = zect.loc[~pd.isna(zect)]
        if cfg.oos_start:
            vals = vals[vals.index < pd.to_datetime(cfg.oos_start)]
        if vals.empty:
            vals = zect.loc[~pd.isna(zect)]
        auto_th = float(np.quantile(np.abs(vals), cfg.z_auto_q)) if not vals.empty else 1.5
    z_th = auto_th
    manual_entry = cfg.z_entry
    source = cfg.z_auto_method
    if manual_entry is not None and np.isfinite(manual_entry):
        if manual_entry <= 0:
            LOGGER.warning("Ignoring non-positive z_entry override: %.3f", manual_entry)
        else:
            z_th = float(manual_entry)
            source = "z_entry"
            LOGGER.info(
                "Manual z_entry override applied (auto=%.3f -> z_entry=%.3f)",
                auto_th,
                z_th,
            )
    cap = getattr(cfg, "z_entry_cap", None)
    if source != "z_entry" and cap is not None and np.isfinite(cap) and cap > 0:
        if z_th > cap:
            LOGGER.info("Capping auto z_th from %.3f to %.3f (z_entry_cap)", z_th, cap)
            z_th = float(cap)
    LOGGER.info("Using entry threshold z_th=%.3f (source=%s)", z_th, source)
    abs_z = np.abs(zect)

    if p_regime is not None:
        prob_series = p_regime.reindex(zect.index).astype(float)
        if prob_series.isna().any():
            base_prob = np.where(abs_z >= np.nanquantile(abs_z, 0.75), 0.85, 0.55)
            prob_series = prob_series.fillna(pd.Series(base_prob, index=abs_z.index))
        LOGGER.info("Using MS-spread regime probabilities for entry gating")
    else:
        base_prob = np.where(abs_z >= np.nanquantile(abs_z, 0.75), 0.85, 0.55)
        prob_series = pd.Series(base_prob, index=abs_z.index)

    enter_long = (zect <= -z_th) & (prob_series >= cfg.p_th)
    enter_short = (zect >= z_th) & (prob_series >= cfg.p_th)

    if delta_score is not None:
        ds = delta_score.reindex(zect.index).astype(float)
        if cfg.short_filter:
            enter_long = enter_long & (ds < 0)
            enter_short = enter_short & (ds > 0)

    # Overlay 12M momentum spread (delta_mom12, faktor jangka panjang)
    # z_mom12 tinggi = "lebih jelek" (overextended 12M), sama interpretasi dengan score_short.
    # delta_mom12 = z_mom12_A - z_mom12_B
    # z < 0 (A lebih murah) → kita ingin A "lebih bagus" → delta_mom12 < 0
    # z > 0 (A lebih mahal) → kita ingin A "lebih jelek" → delta_mom12 > 0
    if delta_mom12 is not None:
        dm = delta_mom12.reindex(zect.index).astype(float)
        if cfg.short_filter:
            enter_long = enter_long & (dm < 0)
            enter_short = enter_short & (dm > 0)
    gates = gates.reindex(zect.index).fillna(False)
    if cfg.gate_enforce:
        enter_long &= gates
        enter_short &= gates
    signal_mode = str(getattr(cfg, "signal_mode", "normal") or "normal").lower()
    if signal_mode == "long_from_short_only":
        # Contrarian execution mode:
        # - legacy short signal becomes long entry
        # - legacy long signal is ignored
        enter_long = enter_short.copy()
        enter_short = pd.Series(False, index=enter_short.index)
    elif signal_mode != "normal":
        LOGGER.warning("Unknown signal_mode=%s, falling back to normal", signal_mode)
    long_conf = _confirm_streak(enter_long, cfg.regime_confirm)
    short_conf = _confirm_streak(enter_short, cfg.regime_confirm)
    long_cd = _cooldown(long_conf, cfg.cooldown)
    short_cd = _cooldown(short_conf, cfg.cooldown)
    signals = pd.DataFrame({"long": long_cd.astype(float), "short": short_cd.astype(float)}, index=zect.index, dtype=float)
    if delta_score is not None:
        signals["delta_score"] = delta_score.reindex(zect.index)
    if delta_mom12 is not None:
        signals["delta_mom12"] = delta_mom12.reindex(zect.index)
    # Enforce long-only patch when configured
    if cfg.long_only:
        signals["short"] = 0.0
    if cfg.mom_enable:
        LOGGER.info(
            "Momentum add-on enabled | z=%.2f, k=%d, gate_k=%d, cooldown=%d",
            cfg.mom_z,
            cfg.mom_k,
            cfg.mom_gate_k,
            cfg.mom_cooldown,
        )
        gate_bad = (~gates).astype(bool)
        gate_bad_streak = _confirm_streak(gate_bad, cfg.mom_gate_k)
        dz = zect.diff()
        mom_up = _confirm_streak((dz > 0), cfg.mom_k)
        mag_ok = (np.abs(zect) >= cfg.mom_z)
        mom_long = gate_bad_streak & mom_up & mag_ok
        mom_long = _cooldown(mom_long, cfg.mom_cooldown)
        before = int(signals["long"].sum())
        signals["long"] = np.where(mom_long, 1.0, signals["long"])
        after = int(signals["long"].sum())
        LOGGER.info("Momentum long entries added: +%d", max(0, after - before))
    return signals, float(z_th)


@dataclass
class ExecutionResult:
    pos: pd.Series
    ret: pd.Series
    ret_core: pd.Series
    cost: pd.Series
    trades: pd.DataFrame
    p_regime: Optional[pd.Series] = None
    delta_score: Optional[pd.Series] = None
    delta_mom12: Optional[pd.Series] = None


def execute_trades(
    zect: pd.Series,
    signals: pd.DataFrame,
    lp_pair: pd.DataFrame,
    beta_series: pd.Series,
    cfg: PlaybookConfig,
    p_regime: Optional[pd.Series] = None,
    delta_score: Optional[pd.Series] = None,
    delta_mom12: Optional[pd.Series] = None,
) -> ExecutionResult:
    idx = zect.index
    signals = signals.reindex(idx).fillna(0.0)
    lp_pair = lp_pair.reindex(idx).ffill().dropna()
    beta_series = beta_series.reindex(lp_pair.index).ffill()
    zect = zect.reindex(lp_pair.index)
    if p_regime is not None:
        p_regime = p_regime.reindex(lp_pair.index)
    r1 = lp_pair.iloc[:, 0].diff().fillna(0.0)
    r2 = lp_pair.iloc[:, 1].diff().fillna(0.0)
    beta_vals = beta_series.fillna(beta_series.mean()).to_numpy()
    weights = 1.0 + np.abs(beta_vals)
    w1 = 1.0 / weights
    w2 = np.abs(beta_vals) / weights
    pos = np.zeros(len(lp_pair))
    cost = np.zeros(len(lp_pair))
    ret_core_vals = np.zeros(len(lp_pair))
    trades: List[Dict[str, object]] = []
    open_idx: Optional[int] = None
    open_side = 0
    trade_pnl = 0.0
    trade_gross_pnl = 0.0
    trade_total_cost = 0.0
    trade_days = 0
    trade_max_hold = max(1, int(cfg.max_hold))

    def _resolve_trade_max_hold(entry_abs_z: float) -> int:
        base_hold = max(1, int(cfg.max_hold))
        if not cfg.dynamic_hold or cfg.dynamic_hold_max_add <= 0:
            return base_hold
        if not np.isfinite(entry_abs_z):
            return base_hold
        if cfg.z_entry is not None and np.isfinite(cfg.z_entry) and cfg.z_entry > 0:
            z_ref = float(cfg.z_entry)
        else:
            z_ref = max(float(cfg.z_exit), 1e-6)
        strength = max(0.0, float(entry_abs_z) - z_ref)
        extra_days = int(np.floor(strength / max(float(cfg.dynamic_hold_step), 1e-6)))
        extra_days = max(0, min(extra_days, int(cfg.dynamic_hold_max_add)))
        return int(base_hold + extra_days)

    for i in range(len(lp_pair)):
        prev = pos[i - 1] if i > 0 else 0
        enter_long = bool(signals.iloc[i]["long"] > 0)
        enter_short = bool(signals.iloc[i]["short"] > 0)
        if prev == 0:
            if enter_long:
                pos[i] = 1
                open_idx = i
                open_side = 1
                trade_pnl = 0.0
                trade_gross_pnl = 0.0
                trade_total_cost = 0.0
                trade_days = 0
                entry_z = float(zect.iloc[i]) if np.isfinite(zect.iloc[i]) else float("nan")
                trade_max_hold = _resolve_trade_max_hold(abs(entry_z))
            elif enter_short:
                pos[i] = -1
                open_idx = i
                open_side = -1
                trade_pnl = 0.0
                trade_gross_pnl = 0.0
                trade_total_cost = 0.0
                trade_days = 0
                entry_z = float(zect.iloc[i]) if np.isfinite(zect.iloc[i]) else float("nan")
                trade_max_hold = _resolve_trade_max_hold(abs(entry_z))
            else:
                pos[i] = 0
        else:
            pos[i] = prev
            trade_days += 1
            zval = float(zect.iloc[i]) if np.isfinite(zect.iloc[i]) else np.nan
            exit_flag = False
            if cfg.exit == "zexit" and np.isfinite(zval):
                exit_flag = abs(zval) <= cfg.z_exit
            elif cfg.exit == "zcross" and i > 0:
                prev_z = float(zect.iloc[i - 1])
                if np.isfinite(zval) and np.isfinite(prev_z):
                    exit_flag = np.sign(zval) != np.sign(prev_z)
            elif cfg.exit == "tplus1":
                exit_flag = trade_days >= 1
            if trade_days < max(0, int(cfg.min_hold)):
                exit_flag = False
            stop_flag = np.isfinite(zval) and abs(zval) >= cfg.z_stop
            time_flag = trade_days >= trade_max_hold
            if exit_flag or stop_flag or time_flag:
                pos[i] = 0
        if pos[i] != prev:
            # Apply fees
            fee = cfg.fee_buy + cfg.fee_sell
            if prev == 0 and pos[i] == 1:
                cost[i] = fee * (w1[i] + w2[i])
            elif prev == 0 and pos[i] == -1:
                cost[i] = fee * (w1[i] + w2[i])
            elif prev != 0 and pos[i] == 0:
                cost[i] = fee * (w1[i] + w2[i])
            else:
                cost[i] = 2 * fee * (w1[i] + w2[i])
        if pos[i] > 0:
            pnl_core = w1[i] * r1.iloc[i] - w2[i] * r2.iloc[i]
        elif pos[i] < 0:
            pnl_core = -w1[i] * r1.iloc[i] + w2[i] * r2.iloc[i]
        else:
            pnl_core = 0.0
        ret_core_vals[i] = pnl_core
        if open_idx is not None:
            trade_gross_pnl += pnl_core
            trade_total_cost += cost[i]
            trade_pnl = trade_gross_pnl - trade_total_cost
        if open_idx is not None and pos[i] == 0:
            holding_days = i - open_idx + 1
            if cfg.debug:
                LOGGER.debug(
                    "Trade closed | side=%s open_idx=%d close_idx=%d gross=%.6f cost=%.6f net=%.6f",
                    "LONG" if open_side > 0 else "SHORT",
                    open_idx,
                    i,
                    trade_gross_pnl,
                    trade_total_cost,
                    trade_pnl,
                )
            trades.append(
                {
                    "open_index": open_idx,
                    "close_index": i,
                    "side": "LONG" if open_side > 0 else "SHORT",
                    "days": trade_days,
                    "holding_days": holding_days,
                    "pnl": trade_pnl,
                    "gross_pnl": trade_gross_pnl,
                    "total_cost": trade_total_cost,
                    "max_hold_effective": trade_max_hold,
                    "open_date": lp_pair.index[open_idx],
                    "close_date": lp_pair.index[i],
                }
            )
            open_idx = None
            open_side = 0
            trade_pnl = 0.0
            trade_gross_pnl = 0.0
            trade_total_cost = 0.0
            trade_days = 0
            trade_max_hold = max(1, int(cfg.max_hold))
    pos_series = pd.Series(pos, index=lp_pair.index, name="pos")
    cost_series = pd.Series(cost, index=lp_pair.index, name="cost")
    ret_core = pd.Series(ret_core_vals, index=lp_pair.index, name="ret_core")
    ret_net = ret_core - cost_series
    ret_net.name = "ret"
    trades_df = pd.DataFrame(trades)
    return ExecutionResult(
        pos=pos_series,
        ret=ret_net,
        ret_core=ret_core,
        cost=cost_series,
        trades=trades_df,
        p_regime=p_regime,
        delta_score=delta_score,
        delta_mom12=delta_mom12,
    )


# ---------------------------------------------------------------------------
# Metrics -------------------------------------------------------------------
# ---------------------------------------------------------------------------
def compute_metrics(
    exec_res: ExecutionResult,
    cfg: PlaybookConfig,
    oos_start: dt.date,
) -> Dict[str, float]:
    """Compute per-run performance metrics.

    Tambahan:
    - n_trades
    - avg_hold_days
    - turnover_annualised (turnover per tahun, berdasarkan posisi OOS)
    """
    # --- trade-based stats dulu ---
    trades = getattr(exec_res, "trades", None)
    if trades is not None and not trades.empty:
        n_trades = int(trades.shape[0])
        if "days" in trades.columns:
            avg_hold_days = float(trades["days"].mean())
        else:
            avg_hold_days = 0.0
    else:
        n_trades = 0
        avg_hold_days = 0.0

    p_regime = getattr(exec_res, "p_regime", None)
    if isinstance(p_regime, pd.Series) and not p_regime.empty:
        p_mr_mean = float(np.nanmean(p_regime))
        pos_mask = exec_res.pos.reindex(p_regime.index).fillna(0.0) != 0
        if pos_mask.any():
            p_mr_inpos = float(np.nanmean(p_regime[pos_mask]))
        else:
            p_mr_inpos = float("nan")
    else:
        p_mr_mean = float("nan")
        p_mr_inpos = float("nan")

    ret = exec_res.ret

    # Kalau sama sekali tidak ada return, kembalikan metrik default
    if ret.empty:
        return {
            "sharpe_oos": 0.0,
            "maxdd": 0.0,
            "turnover": 0.0,
            "turnover_annualised": 0.0,
            "n_trades": n_trades,
            "avg_hold_days": avg_hold_days,
            "p_mr_mean": p_mr_mean,
            "p_mr_inpos_mean": p_mr_inpos,
            "cagr": 0.0,
            "nav_oos": 1.0,
        }

    # --- OOS mask & NAV ---
    mask_oos = ret.index.date >= oos_start
    ret_oos = ret[mask_oos]
    if ret_oos.empty:
        ret_oos = ret

    nav = (1 + ret_oos).cumprod()
    nav0 = nav.iloc[0] if not nav.empty else 1.0
    nav1 = nav.iloc[-1] if not nav.empty else 1.0

    ann_days = cfg.ann_days if cfg.ann_days else 252
    total_days = max(len(ret_oos), 1)
    cagr = (nav1 / nav0) ** (ann_days / total_days) - 1 if nav0 > 0 and nav1 > 0 else 0.0

    dd = nav / nav.cummax() - 1
    maxdd = float(dd.min()) if not dd.empty else 0.0

    mu = float(ret_oos.mean())
    sd = float(ret_oos.std())
    sharpe = (mu / sd) * math.sqrt(252) if sd > 0 else 0.0

    # --- Turnover: dihitung di horizon OOS ---
    pos = exec_res.pos
    pos_oos = pos[mask_oos]
    if pos_oos.empty:
        pos_oos = pos

    turnover_total = float(np.abs(pos_oos.diff().fillna(0)).sum())
    days_pos = max(len(pos_oos), 1)
    turnover_annualised = float(turnover_total * (ann_days / days_pos))

    return {
        "sharpe_oos": float(sharpe),
        "maxdd": float(maxdd),
        "turnover": float(turnover_total),
        "turnover_annualised": float(turnover_annualised),
        "n_trades": int(n_trades),
        "avg_hold_days": float(avg_hold_days),
        "p_mr_mean": float(p_mr_mean),
        "p_mr_inpos_mean": float(p_mr_inpos),
        "cagr": float(cagr),
        "nav_oos": float(nav1),
    }


def _build_invalid_result(
    *,
    run_id: str,
    cfg: PlaybookConfig,
    reason: str,
    pair: Optional[str] = None,
    observations: Optional[int] = None,
) -> Dict[str, object]:
    empty_index = pd.DatetimeIndex([])
    empty_series = pd.Series(dtype=float, index=empty_index)
    exec_res = ExecutionResult(
        pos=empty_series,
        ret=empty_series,
        ret_core=empty_series,
        cost=empty_series,
        trades=pd.DataFrame(),
    )
    metrics = {
        "status": reason,
        "z_th": float("nan"),
    }
    model_checks = {
        "pair": pair or "",
        "rank": 0,
        "deterministic": "ci",
        "threshold": float("nan"),
        "spec_ok": False,
        "status": reason,
    }
    if observations is not None:
        model_checks["observations"] = int(observations)
    return {
        "run_id": run_id,
        "params": cfg.to_dict(),
        "config": cfg.to_dict(),
        "status": reason,
        "metrics": metrics,
        "model_checks": model_checks,
        "signals": pd.DataFrame(),
        "execution": exec_res,
        "horizon": {
            "train_obs": 0,
            "test_obs": 0,
            "train_start": "",
            "train_end": "",
            "test_start": "",
            "test_end": "",
        },
    }


def _apply_decision_params(
    cfg: PlaybookConfig,
    decision_params: DecisionParams,
) -> PlaybookConfig:
    updated_cfg = dataclasses.replace(
        cfg,
        z_entry=decision_params.z_entry,
        z_exit=float(decision_params.z_exit),
        max_hold=int(decision_params.max_hold),
        cooldown=int(decision_params.cooldown),
        z_stop=float(max(decision_params.z_entry or 0.0, decision_params.z_exit)),
        p_th=decision_params.p_th if decision_params.p_th is not None else cfg.p_th,
    )
    if updated_cfg.z_entry is not None:
        if not np.isfinite(updated_cfg.z_entry) or updated_cfg.z_entry <= 0:
            LOGGER.warning(
                "Invalid z_entry %.3f supplied; disabling manual override",
                updated_cfg.z_entry,
            )
            updated_cfg = dataclasses.replace(updated_cfg, z_entry=None)
        elif updated_cfg.z_stop < updated_cfg.z_entry:
            LOGGER.info(
                "Adjusting z_stop from %.3f to %.3f to respect z_entry threshold",
                updated_cfg.z_stop,
                updated_cfg.z_entry,
            )
            updated_cfg = dataclasses.replace(updated_cfg, z_stop=float(updated_cfg.z_entry))
    return updated_cfg


def build_features(
    universe: Optional[Iterable[str] | str],
    feature_config: FeatureConfig,
) -> FeatureBuildResult:
    run_id = feature_config.run_id or dt.datetime.utcnow().strftime(RUN_ID_FMT)
    subset = _resolve_universe(universe, feature_config.base_config.subset)
    cfg = dataclasses.replace(
        feature_config.base_config,
        subset=subset,
        method=feature_config.method,
        horizon=feature_config.horizon,
    )
    if cfg.seed is not None:
        np.random.seed(cfg.seed)
    LOGGER.info(
        "Feature build start | run_id=%s pair=%s method=%s",
        run_id,
        subset,
        cfg.method,
    )

    if feature_config.data_frame is not None:
        df = feature_config.data_frame.copy(deep=True)
    else:
        df = load_and_validate_data(cfg.input_file)

    available_cols = [c for c in df.columns if c != "date"]
    subset_cols = available_cols
    if cfg.subset:
        raw_tokens = [token.strip() for token in cfg.subset.split(",") if token.strip()]
        enriched = [tok if tok.endswith(".JK") else f"{tok}.JK" for tok in raw_tokens]
        subset_cols = [c for c in available_cols if c in enriched] or available_cols
    panel_df = df[["date", *subset_cols]].copy()
    universe_id = _panel_universe_id(subset_cols, cfg)
    panel_hash = hash_dataframe(panel_df)
    panel_path, panel_meta_path = _panel_cache_paths(universe_id, panel_hash)

    short_panel: Optional[pd.DataFrame] = None
    try:
        short_panel = build_short_term_overlay(df, market_col="^JKSE", data_hash=panel_hash)
        LOGGER.info("Short-term signals panel ready with shape %s", short_panel.shape)
    except Exception as exc:
        LOGGER.warning("Short-term signals construction failed; disabling overlay: %s", exc)

    outlier_ratios: Dict[str, float] = {}
    df_clean: pd.DataFrame
    if panel_path.exists():
        df_clean = _read_panel_cache(panel_path)
        if panel_meta_path.exists():
            try:
                meta_payload = json.loads(panel_meta_path.read_text())
                outlier_ratios = {
                    str(key): float(value)
                    for key, value in meta_payload.get("outlier_ratios", {}).items()
                }
            except (json.JSONDecodeError, OSError, TypeError, ValueError):
                LOGGER.warning("Failed to read panel cache metadata at %s", panel_meta_path)
        LOGGER.info("Loaded cached panel %s with shape %s", panel_path, df_clean.shape)
    else:
        df_clean, _, outlier_ratios = preprocess_data(df, cfg)
        _write_panel_cache(panel_path, df_clean, panel_meta_path, {"outlier_ratios": outlier_ratios})
    macro_exog = feature_config.macro_exog
    if macro_exog is None:
        macro_exog = _build_macro_exog(df, df_clean.index)
    try:
        selected_l, selected_r, beta0 = select_pair(df_clean, cfg, outlier_ratios, macro_exog=macro_exog)
    except PairSelectionError as exc:
        LOGGER.warning("Pair selection failed: %s", exc)
        result = _build_invalid_result(
            run_id=run_id,
            cfg=cfg,
            reason="INVALID_PAIR_OBSERVATIONS",
            pair=exc.pair,
            observations=exc.observations,
        )
        return FeatureBuildResult(features=None, skip_result=result)
    LOGGER.info("Selected pair: %s ~ %s | beta0=%.4f", selected_l, selected_r, beta0)
    pair_prices = df_clean[[selected_l, selected_r]].dropna()
    lp = np.log(pair_prices)
    beta_series, ect = kalman_dynamic_beta(lp.iloc[:, 0], lp.iloc[:, 1], beta0)
    mect = ect.rolling(252, min_periods=20).mean()
    sect = ect.rolling(252, min_periods=20).std()
    sect = sect.replace(0.0, np.nan)
    zect = (ect - mect) / sect
    zect = zect.replace([np.inf, -np.inf], np.nan)
    zect_valid = zect.dropna()
    min_spread_std = 1e-6
    ect_std = float(ect.std()) if len(ect) else float("nan")
    if zect_valid.empty:
        fallback_std_val = ect_std
        if np.isfinite(fallback_std_val) and fallback_std_val >= min_spread_std:
            LOGGER.warning(
                "Z-score series empty after normalisation; falling back to global standard deviation"
            )
            fallback = (ect - ect.mean()) / fallback_std_val
            zect_valid = fallback.dropna()
    if zect_valid.empty:
        LOGGER.error(
            "Z-score series empty after normalisation; skipping pair %s~%s",
            selected_l,
            selected_r,
        )
        empty_idx = lp.index
        empty_signals = pd.DataFrame(
            {"long": 0.0, "short": 0.0}, index=empty_idx, dtype=float
        )
        empty_series = pd.Series(0.0, index=empty_idx)
        exec_res = ExecutionResult(
            pos=empty_series.rename("pos"),
            ret=empty_series.rename("ret"),
            ret_core=empty_series.rename("ret_core"),
            cost=empty_series.rename("cost"),
            trades=pd.DataFrame(),
        )
        oos_start_date = _resolve_oos_start_date(cfg, empty_idx, allow_empty=True)
        metrics = compute_metrics(exec_res, cfg, oos_start_date)
        metrics["skip_reason"] = "zscore_empty"
        result = {
            "run_id": run_id,
            "params": cfg.to_dict(),
            "config": cfg.to_dict(),
            "metrics": metrics,
            "model_checks": {
                "pair": f"{selected_l}~{selected_r}",
                "rank": 1,
                "deterministic": "ci",
                "threshold": float("nan"),
                "skipped": True,
            },
            "signals": empty_signals,
            "execution": exec_res,
            "horizon": _build_horizon(lp.index),
            "skipped_reason": "zscore_empty",
        }
        return FeatureBuildResult(features=None, skip_result=result)
    if not np.isfinite(ect_std) or ect_std < min_spread_std:
        LOGGER.error(
            "Spread variance too low (std=%.6g); skipping pair %s~%s",
            ect_std,
            selected_l,
            selected_r,
        )
        empty_idx = lp.index
        empty_signals = pd.DataFrame(
            {"long": 0.0, "short": 0.0}, index=empty_idx, dtype=float
        )
        empty_series = pd.Series(0.0, index=empty_idx)
        exec_res = ExecutionResult(
            pos=empty_series.rename("pos"),
            ret=empty_series.rename("ret"),
            ret_core=empty_series.rename("ret_core"),
            cost=empty_series.rename("cost"),
            trades=pd.DataFrame(),
        )
        oos_start_date = _resolve_oos_start_date(cfg, empty_idx, allow_empty=True)
        metrics = compute_metrics(exec_res, cfg, oos_start_date)
        metrics["skip_reason"] = "spread_variance_low"
        result = {
            "run_id": run_id,
            "params": cfg.to_dict(),
            "config": cfg.to_dict(),
            "metrics": metrics,
            "model_checks": {
                "pair": f"{selected_l}~{selected_r}",
                "rank": 1,
                "deterministic": "ci",
                "threshold": float("nan"),
                "skipped": True,
            },
            "signals": empty_signals,
            "execution": exec_res,
            "horizon": _build_horizon(lp.index),
            "skipped_reason": "spread_variance_low",
        }
        return FeatureBuildResult(features=None, skip_result=result)
    zect = zect_valid

    pair_label = f"{selected_l}~{selected_r}"
    pair_id = _safe_cache_slug(f"{selected_l}__{selected_r}")
    feature_key = _feature_cache_key(cfg, panel_hash, pair_label)
    feature_cache_path = _feature_cache_path(pair_id, feature_key)
    cached_feature_panel: Optional[pd.DataFrame] = None
    if feature_cache_path.exists():
        try:
            cached_feature_panel = pd.read_parquet(feature_cache_path)
            if "date" in cached_feature_panel.columns:
                cached_feature_panel = cached_feature_panel.set_index("date")
            cached_feature_panel.index = pd.to_datetime(cached_feature_panel.index)
            cached_feature_panel = cached_feature_panel.sort_index()
            LOGGER.info("Loaded cached pair features from %s", feature_cache_path)
        except Exception as exc:
            LOGGER.warning("Failed to read feature cache at %s: %s", feature_cache_path, exc)
            cached_feature_panel = None

    delta_score: Optional[pd.Series] = None
    if short_panel is not None:
        try:
            score_l = short_panel[selected_l].reindex(zect.index)
            score_r = short_panel[selected_r].reindex(zect.index)
            delta_score = (score_l - score_r).astype(float)
        except KeyError:
            LOGGER.warning(
                "Short-term signals missing for %s or %s; disabling delta_score overlay",
                selected_l,
                selected_r,
            )
    if delta_score is None and cached_feature_panel is not None and "short_term_delta" in cached_feature_panel.columns:
        delta_score = cached_feature_panel["short_term_delta"].reindex(zect.index)

    # --- Long-horizon factor: 12M momentum spread (delta_mom12) ---
    delta_mom12: Optional[pd.Series] = None
    if short_panel is not None:
        z_mom12_panel = short_panel.attrs.get("z_mom12")
        if isinstance(z_mom12_panel, pd.DataFrame):
            try:
                mom_l = z_mom12_panel[selected_l].reindex(zect.index)
                mom_r = z_mom12_panel[selected_r].reindex(zect.index)
                delta_mom12 = (mom_l - mom_r).astype(float)
            except KeyError:
                LOGGER.warning(
                    "z_mom12 not available for %s or %s; disabling delta_mom12 overlay",
                    selected_l,
                    selected_r,
                )
        else:
            LOGGER.info("No z_mom12 panel found in short_term_signals attrs; skip delta_mom12")

    # Regime switching model on the spread to infer mean-reverting regime probability
    ms_status = "ok"
    ms_error = ""
    if cached_feature_panel is not None and "p_regime" in cached_feature_panel.columns:
        p_mr_series = cached_feature_panel["p_regime"].reindex(zect.index)
    else:
        ms_model = fit_ms_spread(zect, pair_id=pair_label, data_hash=panel_hash)
        if not ms_model.get("success", False):
            ms_error = str(ms_model.get("error") or "unknown")
            fallback = ms_model.get("fallback") or ""
            if ms_model.get("skipped", False):
                ms_status = "skipped"
                LOGGER.warning(
                    "MS spread model skipped (%s); continuing with flat regime probability",
                    ms_error,
                )
                p_mr_series = pd.Series(0.7, index=zect.index)
            elif fallback:
                ms_status = "fallback"
                LOGGER.warning(
                    "MS spread model fallback=%s (%s); using proxy regime probability",
                    fallback,
                    ms_error,
                )
                p_mr_series = compute_regime_prob(ms_model, zect)
            else:
                LOGGER.error("MS spread model failed: %s", ms_error)
                raise RuntimeError(f"MS spread model failed: {ms_error}")
        else:
            p_mr_series = compute_regime_prob(ms_model, zect)

    # Estimasi parameter konvergensi spread pada full sample
    alpha_ec, half_life_full = _convergence_stats(zect)

    if cached_feature_panel is not None and "corr_gate" in cached_feature_panel.columns:
        corr_gate = cached_feature_panel["corr_gate"].reindex(zect.index).fillna(False)
    else:
        corr_vals = _run_cor(lp.iloc[:, 0].diff(), lp.iloc[:, 1].diff(), max(cfg.gate_corr_win, 10))
        if cfg.gate_require_corr:
            corr_gate = (corr_vals >= cfg.gate_corr_min).reindex(zect.index).fillna(False)
        else:
            corr_gate = pd.Series(True, index=zect.index)
    if cached_feature_panel is not None and "half_life" in cached_feature_panel.columns:
        hl_series = cached_feature_panel["half_life"].reindex(zect.index)
    else:
        hl_series = zect.rolling(252, min_periods=60).apply(_half_life, raw=False)
    hl_gate = (hl_series <= cfg.half_life_max).reindex(zect.index).fillna(False)
    if cfg.gate_enforce:
        combined_gate = (corr_gate & hl_gate).reindex(zect.index).fillna(False)
    else:
        combined_gate = pd.Series(True, index=zect.index)

    expected_feature_cols = ["spread", "zscore", "p_regime", "corr_gate", "half_life", "short_term_delta"]
    write_feature_cache = cached_feature_panel is None or any(
        col not in cached_feature_panel.columns for col in expected_feature_cols
    )
    if write_feature_cache:
        feature_cache_path.parent.mkdir(parents=True, exist_ok=True)
        feature_payload = pd.DataFrame(
            {
                "spread": ect.reindex(zect.index),
                "zscore": zect,
                "p_regime": p_mr_series,
                "corr_gate": corr_gate,
                "half_life": hl_series,
                "short_term_delta": delta_score,
            }
        )
        feature_payload.to_parquet(feature_cache_path)
        LOGGER.info("Wrote pair feature cache to %s", feature_cache_path)

    oos_start_date = _resolve_oos_start_date(cfg, zect.index, allow_empty=False)
    horizon = _build_horizon(lp.index)
    features = FeatureBundle(
        run_id=run_id,
        cfg=cfg,
        pair=f"{selected_l},{selected_r}",
        selected_l=selected_l,
        selected_r=selected_r,
        lp=lp,
        beta_series=beta_series,
        zect=zect,
        combined_gate=combined_gate,
        p_mr_series=p_mr_series,
        delta_score=delta_score,
        delta_mom12=delta_mom12,
        alpha_ec=alpha_ec,
        half_life_full=half_life_full,
        ms_status=ms_status,
        ms_error=ms_error,
        oos_start_date=oos_start_date,
        horizon=horizon,
    )
    LOGGER.info(
        "Feature build complete | run_id=%s pair=%s rows=%d",
        run_id,
        features.pair,
        len(zect),
    )
    return FeatureBuildResult(features=features)


def evaluate_rules(
    feature_result: FeatureBuildResult,
    decision_params: DecisionParams,
) -> Dict[str, object]:
    if feature_result.skip_result is not None:
        return feature_result.skip_result
    if feature_result.features is None:
        raise ValueError("Feature bundle missing; cannot evaluate rules")
    features = feature_result.features
    cfg = _apply_decision_params(features.cfg, decision_params)

    signals, z_th = build_signals(
        features.zect,
        cfg,
        features.combined_gate,
        delta_score=features.delta_score,
        p_regime=features.p_mr_series,
        delta_mom12=features.delta_mom12,
    )
    exec_res = execute_trades(
        features.zect,
        signals,
        features.lp,
        features.beta_series,
        cfg,
        p_regime=features.p_mr_series,
        delta_score=features.delta_score,
        delta_mom12=features.delta_mom12,
    )
    metrics = compute_metrics(exec_res, cfg, features.oos_start_date)
    metrics["alpha_ec"] = features.alpha_ec
    metrics["half_life_full"] = features.half_life_full
    metrics["z_th"] = z_th
    metrics["ms_regime_status"] = features.ms_status
    if features.ms_error:
        metrics["ms_regime_error"] = features.ms_error
    if cfg.z_entry is not None and np.isfinite(cfg.z_entry):
        metrics["z_entry"] = float(cfg.z_entry)
    run_id = decision_params.run_id or features.run_id
    return {
        "run_id": run_id,
        "params": cfg.to_dict(),
        "config": cfg.to_dict(),
        "metrics": metrics,
        "model_checks": {
            "pair": f"{features.selected_l}~{features.selected_r}",
            "rank": 1,
            "deterministic": "ci",
            "threshold": float(z_th),
            "spec_ok": True,
        },
        "signals": signals,
        "execution": exec_res,
        "horizon": features.horizon,
    }


# ---------------------------------------------------------------------------
# Main pipeline -------------------------------------------------------------
# ---------------------------------------------------------------------------
def run_playbook(
    config: Mapping[str, object],
    *,
    persist: bool = True,
    data_frame: Optional[pd.DataFrame] = None,
) -> Dict[str, object]:
    if isinstance(config, PlaybookConfig):
        cfg = config
    else:
        default_cfg = parse_args([])
        cfg_dict = default_cfg.to_dict()
        cfg_dict.update({k: v for k, v in dict(config).items() if v is not None})
        cfg = PlaybookConfig(**cfg_dict)
    run_id = dt.datetime.utcnow().strftime(RUN_ID_FMT)
    LOGGER.info("=== VECM Playbook start | run_id=%s ===", run_id)
    feature_result = build_features(
        cfg.subset,
        FeatureConfig(
            base_config=cfg,
            pair=cfg.subset,
            method=cfg.method,
            horizon=cfg.horizon,
            data_frame=data_frame,
            run_id=run_id,
        ),
    )
    if feature_result.skip_result is not None:
        result = feature_result.skip_result
        if persist:
            persist_artifacts(run_id, cfg, result)
        LOGGER.info("=== VECM Playbook complete (skipped) | run_id=%s ===", run_id)
        return result

    decision_params = DecisionParams(
        z_entry=cfg.z_entry,
        z_exit=cfg.z_exit,
        max_hold=cfg.max_hold,
        cooldown=cfg.cooldown,
        p_th=cfg.p_th,
        run_id=run_id,
    )
    result = evaluate_rules(feature_result, decision_params)
    if persist:
        persist_artifacts(run_id, cfg, result)
    LOGGER.info("=== VECM Playbook complete | run_id=%s ===", run_id)
    return result


def pipeline(
    params: Mapping[str, object],
    *,
    persist: bool = True,
    data_frame: Optional[pd.DataFrame] = None,
) -> Dict[str, object]:
    return run_playbook(params, persist=persist, data_frame=data_frame)


# ---------------------------------------------------------------------------
# Persistence ---------------------------------------------------------------
# ---------------------------------------------------------------------------
def persist_artifacts(run_id: str, cfg: PlaybookConfig, result: Dict[str, object]) -> None:
    exec_res: ExecutionResult = result["execution"]
    artifacts_dir = OUT_DIR / "artifacts" / run_id
    journal_dir = OUT_DIR / "trade_journal"
    pos_path = OUT_DIR / f"positions_{run_id}.csv"
    ret_path = OUT_DIR / f"returns_{run_id}.csv"
    trades_path = OUT_DIR / f"trades_{run_id}.csv"
    metrics_path = OUT_DIR / f"metrics_{run_id}.csv"
    artifacts_path = OUT_DIR / f"artifacts_{run_id}.json"
    manifest_path = OUT_DIR / "run_manifest.csv"
    manifest_lock = manifest_path.with_suffix(".lock")

    _df_to_csv(exec_res.pos.to_frame("pos"), pos_path)

    ret_df = pd.DataFrame({"ret": exec_res.ret, "cost": exec_res.cost})
    if getattr(exec_res, "p_regime", None) is not None:
        ret_df["p_regime"] = exec_res.p_regime
    if getattr(exec_res, "delta_score", None) is not None:
        ret_df["delta_score"] = exec_res.delta_score
    if getattr(exec_res, "delta_mom12", None) is not None:
        ret_df["delta_mom12"] = exec_res.delta_mom12
    _df_to_csv(ret_df, ret_path)
    if not exec_res.trades.empty:
        trades_df = exec_res.trades.copy()
        if "open_date" in trades_df.columns:
            trades_df["open_date"] = trades_df["open_date"].astype(str)
        if "close_date" in trades_df.columns:
            trades_df["close_date"] = trades_df["close_date"].astype(str)
        _df_to_csv(trades_df, trades_path)
    pd.DataFrame([result["metrics"]]).to_csv(metrics_path, index=False)
    with open(artifacts_path, "w", encoding="utf-8") as fh:
        json.dump({"params": cfg.to_dict(), "run_id": run_id}, fh, indent=2, default=str)

    status = result.get("status")
    if not status:
        status = "OK_HAS_TRADES" if exec_res.trades.shape[0] else "OK_NO_TRADES"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    run_manifest = {
        "run_id": run_id,
        "timestamp": _now_utc().isoformat(),
        "params": cfg.to_dict(),
        "metrics": result.get("metrics", {}),
        "status": status,
    }
    with open(artifacts_dir / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(run_manifest, fh, indent=2, ensure_ascii=False, default=str)
    with open(artifacts_dir / "status.json", "w", encoding="utf-8") as fh:
        json.dump(
            {
                "run_id": run_id,
                "status": status,
                "trades": int(exec_res.trades.shape[0]),
                "duration_sec": result.get("metrics", {}).get("elapsed_s"),
                "timestamp": _now_utc().isoformat(),
            },
            fh,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    journal_dir.mkdir(parents=True, exist_ok=True)
    trade_summary = {
        "run_id": run_id,
        "pair": cfg.subset,
        "expected_hold_days": result.get("metrics", {}).get("avg_hold_days"),
        "n_trades": int(exec_res.trades.shape[0]),
        "total_return": float(exec_res.ret.sum()),
        "core_return": float(exec_res.ret_core.sum()) if hasattr(exec_res, "ret_core") else None,
        "cost_total": float(exec_res.cost.sum()),
        "attribution": {
            "alpha_core": float(exec_res.ret_core.sum()) if hasattr(exec_res, "ret_core") else None,
            "transaction_costs": float(exec_res.cost.sum()),
        },
    }
    if not exec_res.trades.empty:
        if "days" in exec_res.trades.columns:
            trade_summary["actual_avg_hold_days"] = float(exec_res.trades["days"].mean())
        if "pnl" in exec_res.trades.columns:
            trade_summary["avg_trade_pnl"] = float(exec_res.trades["pnl"].mean())
    with open(journal_dir / f"trade_journal_{run_id}.json", "w", encoding="utf-8") as fh:
        json.dump(trade_summary, fh, indent=2, ensure_ascii=False, default=str)
    pd.DataFrame([trade_summary]).to_csv(journal_dir / f"trade_journal_{run_id}.csv", index=False)
    manifest_row = {
        "run_id": run_id,
        "ts_utc": _now_utc().isoformat(),
        "tag": cfg.tag,
        "subset": cfg.subset,
        "pair": result["model_checks"]["pair"],
        "z_source": cfg.z_auto_method,
        "z_th": result["metrics"].get("z_th"),
        "status": status,
        "trades": int(exec_res.trades.shape[0]),
        "Sharpe": float(result["metrics"].get("sharpe_oos", 0.0)),
        "CAGR": float(result["metrics"].get("cagr", 0.0)),
        "maxDD": float(result["metrics"].get("maxdd", 0.0)),
        "totalRet": float(exec_res.ret.sum()),
    }
    # File lock for manifest writes
    try:
        from filelock import FileLock

        lock = FileLock(manifest_lock)
        with lock:
            _append_manifest(manifest_path, manifest_row)
    except Exception:
        LOGGER.warning("filelock unavailable; appending manifest without lock")
        _append_manifest(manifest_path, manifest_row)

    # Persist model checks to DuckDB
    tickers = [token.strip() for token in cfg.subset.split(",") if token.strip()]
    with storage.managed_storage(cfg.tag) as conn:
        with storage.with_transaction(conn):
            storage.write_run_metrics(
                conn,
                run_id,
                metric_ts=_now_utc(),
                universe_id=_panel_universe_id(tickers, cfg),
                pairs_count=1,
                trades_count=int(exec_res.trades.shape[0]),
                sharpe_oos=float(result["metrics"].get("sharpe_oos", 0.0)),
                turnover_annualised=float(result["metrics"].get("turnover_annualised", 0.0)),
                score=float(result["metrics"].get("score", 0.0)),
                maxdd=float(result["metrics"].get("maxdd", 0.0)),
                fees_total=float(exec_res.cost.sum()),
            )
            storage.write_model_checks(
                conn,
                run_id,
                pair=result["model_checks"]["pair"],
                johansen_rank=1,
                det_term="ci",
                tvecm_thresholds={
                    "z_exit": cfg.z_exit,
                    "z_stop": cfg.z_stop,
                    "z_th": result["metrics"].get("z_th"),
                },
                spec_ok=bool(result["model_checks"].get("spec_ok", True)),
            )
            storage.write_regime_stats(
                conn,
                run_id,
                p_mr_mean=float(result["metrics"].get("p_mr_mean", math.nan)),
                p_mr_inpos_mean=float(result["metrics"].get("p_mr_inpos_mean", math.nan)),
            )


def _append_manifest(path: pathlib.Path, row: Mapping[str, object]) -> None:
    df_row = pd.DataFrame([row])
    if path.exists():
        df_row.to_csv(path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(path, mode="w", header=True, index=False)


# ---------------------------------------------------------------------------
# CLI Entrypoint ------------------------------------------------------------
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    try:
        cfg = parse_args(argv)
        run_playbook(cfg)
        return 0
    except Exception as exc:
        LOGGER.exception("Playbook failed: %s", exc)
        error_path = OUT_DIR / "last_error.txt"
        _write_text(error_path, f"{_now_utc().isoformat()}\n{exc}\n")
        return 1


if __name__ == "__main__":  # pragma: no cover - CLI smoke test
    sys.exit(main())
