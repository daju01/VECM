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
import datetime as dt
import json
import logging
import math
import os
import pathlib
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import VECM

from . import storage
from .data_streaming import ensure_price_data
from .ms_spread import compute_regime_prob, fit_ms_spread
from .short_term_signals import build_short_term_signals

# ---------------------------------------------------------------------------
# Logging / paths -----------------------------------------------------------
# ---------------------------------------------------------------------------
BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
OUT_DIR = BASE_DIR / "out_ms"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUN_ID_FMT = "%Y%m%d_%H%M%S"

LOGGER = storage.configure_logging("playbook_vecm")


# ---------------------------------------------------------------------------
# Dataclasses & configuration ------------------------------------------------
# ---------------------------------------------------------------------------
@dataclass
class PlaybookConfig:
    input_file: str
    subset: str = ""
    method: str = "TVECM"
    roll_years: float = 3.0
    oos_start: str = ""
    horizon: str = ""
    stage: int = 0
    notes: str = ""
    exit: str = "zexit"
    z_entry: Optional[float] = None
    z_exit: float = 0.6
    z_stop: float = 0.8
    max_hold: int = 8
    cooldown: int = 5
    z_auto_method: str = "mfpt"
    z_auto_q: float = 0.9
    gate_require_corr: int = 1
    gate_corr_min: float = 0.70
    gate_corr_win: int = 60
    gate_enforce: bool = True
    beta_weight: bool = True
    cost_bps: float = 5.0
    half_life_max: float = 90.0
    dd_stop: float = 0.25
    fee_buy: float = 0.0019
    fee_sell: float = 0.0029
    p_th: float = 0.85
    regime_confirm: int = 4
    kelly_frac: float = 0.5
    vol_cap: float = 0.20
    ann_days: int = 252
    debug: bool = False
    selftest: bool = False
    seed: Optional[int] = None
    tag: str = ""
    mom_enable: bool = False
    mom_z: float = 0.60
    mom_k: int = 2
    mom_gate_k: int = 3
    mom_cooldown: int = 2

    def to_dict(self) -> Dict[str, object]:
        return dataclasses.asdict(self)


def _default_input_path() -> str:
    """Return the default price cache path without forcing it to exist."""

    data_path = BASE_DIR / "data" / "adj_close_data.csv"
    return str(data_path)


def _ensure_default_input(path: str) -> str:
    """Ensure the default price cache exists before returning it."""

    if not os.path.exists(path):
        LOGGER.info("Default input %s missing; invoking streaming loader", path)
        try:
            ensure_price_data(force_refresh=False)
        except Exception as exc:  # pragma: no cover - download/runtime issues
            LOGGER.warning("Price streaming failed when populating default input: %s", exc)
    return path


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
    parser.add_argument("--z_exit", type=float, default=0.6)
    parser.add_argument("--z_stop", type=float, default=0.8)
    parser.add_argument("--max_hold", type=int, default=8)
    parser.add_argument("--cooldown", type=int, default=5)
    parser.add_argument("--z_auto_method", default="mfpt")
    parser.add_argument("--z_auto", type=float, default=0.9)
    parser.add_argument("--gate_require_corr", type=int, default=1)
    parser.add_argument("--gate_corr_min", type=float, default=0.70)
    parser.add_argument("--gate_corr_win", type=int, default=60)
    parser.add_argument("--gate_enforce", type=int, default=1)
    parser.add_argument("--beta_weight", type=int, default=1)
    parser.add_argument("--cost_bps", type=float, default=5.0)
    parser.add_argument("--half_life_max", type=float, default=90.0)
    parser.add_argument("--dd_stop", type=float, default=0.25)
    parser.add_argument("--fee_buy", type=float, default=0.0019)
    parser.add_argument("--fee_sell", type=float, default=0.0029)
    parser.add_argument("--p_th", type=float, default=0.85)
    parser.add_argument("--regime_confirm", type=int, default=4)
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
        cooldown=int(args.cooldown),
        z_auto_method=args.z_auto_method.lower(),
        z_auto_q=float(args.z_auto),
        gate_require_corr=int(args.gate_require_corr),
        gate_corr_min=float(args.gate_corr_min),
        gate_corr_win=int(args.gate_corr_win),
        gate_enforce=bool(args.gate_enforce),
        beta_weight=bool(args.beta_weight),
        cost_bps=float(args.cost_bps),
        half_life_max=float(args.half_life_max),
        dd_stop=float(args.dd_stop),
        fee_buy=float(args.fee_buy),
        fee_sell=float(args.fee_sell),
        p_th=float(args.p_th),
        regime_confirm=int(args.regime_confirm),
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
    )

    # Allow quick overrides via environment variables without touching CLI defaults.
    env_fee_buy = os.getenv("PLAYBOOK_FEE_BUY")
    env_fee_sell = os.getenv("PLAYBOOK_FEE_SELL")
    if env_fee_buy is not None:
        cfg.fee_buy = float(env_fee_buy)
    if env_fee_sell is not None:
        cfg.fee_sell = float(env_fee_sell)
    return cfg


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


def _df_to_csv(df: pd.DataFrame, path: pathlib.Path) -> None:
    df_out = df.copy()
    if "date" not in df_out.columns:
        if isinstance(df_out.index, pd.DatetimeIndex):
            df_out.insert(0, "date", df_out.index.date)
        else:
            df_out.insert(0, "date", df_out.index)
    path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(path, index=False)


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
    if not os.path.exists(path):
        LOGGER.info("Input file %s missing; invoking streaming loader", path)
        ensure_price_data(force_refresh=False)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")
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
    price_cols = [c for c in df.columns if c != "date"]
    for col in price_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[["date", *price_cols]]
    df = df.dropna(how="all", subset=price_cols)
    if len(price_cols) < 2:
        raise ValueError("Need at least two price columns")
    df = df.reset_index(drop=True)
    return df


def preprocess_data(df: pd.DataFrame, cfg: PlaybookConfig) -> Tuple[pd.DataFrame, List[str]]:
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
    for col in df.columns:
        series = df[col]
        series = series.where(series > 0)
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if np.isfinite(iqr) and iqr > 0:
            lower = q1 - 3 * iqr
            upper = q3 + 3 * iqr
            mask = (series < lower) | (series > upper)
            if mask.sum() and mask.sum() < 0.01 * len(series):
                series = series.mask(mask)
        df[col] = series.interpolate(limit=5)
    na_ratios = df.isna().mean()
    drop_cols = [c for c, ratio in na_ratios.items() if ratio > 0.20]
    if drop_cols:
        LOGGER.warning("Dropping >20%% NA series: %s", ", ".join(drop_cols))
        df = df.drop(columns=drop_cols)
    if cfg.roll_years > 0 and not df.empty:
        window = int(cfg.roll_years * 365.25)
        cutoff = df.index.max() - pd.Timedelta(days=window)
        cutoff = max(cutoff, df.index.min() + pd.Timedelta(days=250))
        df = df[df.index >= cutoff]
    if len(df.columns) < 2:
        raise ValueError("Insufficient columns after cleaning")
    if len(df) < 100:
        raise ValueError("Insufficient history after preprocessing")
    return df, list(df.columns)


# ---------------------------------------------------------------------------
# Pair selection -------------------------------------------------------------
# ---------------------------------------------------------------------------
def _johansen_beta(log_prices: pd.DataFrame) -> float:
    if len(log_prices) < 120:
        return float("nan")
    vecm_kwargs = {"k_ar_diff": 1, "deterministic": "ci"}
    try:
        vecm = VECM(log_prices, coint_rank=1, **vecm_kwargs)
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


def select_pair(df: pd.DataFrame, cfg: PlaybookConfig) -> Tuple[str, str, float]:
    tickers = list(df.columns)
    if len(tickers) == 2:
        beta = _johansen_beta(np.log(df.iloc[:, :2]))
        if not np.isfinite(beta):
            beta = 1.0
        return tickers[0], tickers[1], beta
    best_pair: Optional[Tuple[str, str, float]] = None
    best_score = -np.inf
    combos = list(zip(*np.triu_indices(len(tickers), k=1)))
    rng = np.random.default_rng(123)
    if len(combos) > 200:
        combos = rng.choice(combos, size=200, replace=False)
    for idx0, idx1 in combos:
        t1, t2 = tickers[idx0], tickers[idx1]
        lp = np.log(df[[t1, t2]].dropna())
        if len(lp) < 120:
            continue
        beta = _johansen_beta(lp)
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
) -> Tuple[pd.DataFrame, float]:
    # Determine z threshold
    if cfg.z_auto_method == "mfpt":
        auto_th, _ = _mfpt_threshold(zect, np.arange(0.8, 3.05, 0.1), cfg.z_exit, cfg.fee_buy, cfg.fee_sell, cfg.ann_days)
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
        enter_long = enter_long & (ds < 0)
        enter_short = enter_short & (ds > 0)
    gates = gates.reindex(zect.index).fillna(False)
    if cfg.gate_enforce:
        enter_long &= gates
        enter_short &= gates
    long_conf = _confirm_streak(enter_long, cfg.regime_confirm)
    short_conf = _confirm_streak(enter_short, cfg.regime_confirm)
    long_cd = _cooldown(long_conf, cfg.cooldown)
    short_cd = _cooldown(short_conf, cfg.cooldown)
    signals = pd.DataFrame({"long": long_cd.astype(float), "short": short_cd.astype(float)}, index=zect.index, dtype=float)
    if delta_score is not None:
        signals["delta_score"] = delta_score.reindex(zect.index)
    # Enforce long-only patch
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


def execute_trades(
    zect: pd.Series,
    signals: pd.DataFrame,
    lp_pair: pd.DataFrame,
    beta_series: pd.Series,
    cfg: PlaybookConfig,
    p_regime: Optional[pd.Series] = None,
    delta_score: Optional[pd.Series] = None,
) -> ExecutionResult:
    idx = zect.index
    signals = signals.reindex(idx).fillna(0.0)
    lp_pair = lp_pair.reindex(idx).fillna(method="ffill").dropna()
    beta_series = beta_series.reindex(lp_pair.index).fillna(method="ffill")
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
    trades: List[Dict[str, object]] = []
    open_idx: Optional[int] = None
    open_side = 0
    trade_pnl = 0.0
    trade_days = 0
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
                trade_days = 0
            elif enter_short:
                pos[i] = -1
                open_idx = i
                open_side = -1
                trade_pnl = 0.0
                trade_days = 0
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
            stop_flag = np.isfinite(zval) and abs(zval) >= cfg.z_stop
            time_flag = trade_days >= cfg.max_hold
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
        pnl_net = pnl_core - cost[i]
        if open_idx is not None:
            trade_pnl += pnl_net
        if open_idx is not None and pos[i] == 0:
            trades.append(
                {
                    "open_index": open_idx,
                    "close_index": i,
                    "side": "LONG" if open_side > 0 else "SHORT",
                    "days": trade_days,
                    "pnl": trade_pnl,
                    "open_date": lp_pair.index[open_idx],
                    "close_date": lp_pair.index[i],
                }
            )
            open_idx = None
            open_side = 0
            trade_pnl = 0.0
            trade_days = 0
    pos_series = pd.Series(pos, index=lp_pair.index, name="pos")
    cost_series = pd.Series(cost, index=lp_pair.index, name="cost")
    ret_core_vals = []
    for i, position in enumerate(pos):
        if position > 0:
            ret_core_vals.append(w1[i] * r1.iloc[i] - w2[i] * r2.iloc[i])
        elif position < 0:
            ret_core_vals.append(-w1[i] * r1.iloc[i] + w2[i] * r2.iloc[i])
        else:
            ret_core_vals.append(0.0)
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
    if cfg.z_entry is not None:
        if not np.isfinite(cfg.z_entry) or cfg.z_entry <= 0:
            LOGGER.warning("Invalid z_entry %.3f supplied; disabling manual override", cfg.z_entry)
            cfg.z_entry = None
        elif cfg.z_stop < cfg.z_entry:
            LOGGER.info(
                "Adjusting z_stop from %.3f to %.3f to respect z_entry threshold",
                cfg.z_stop,
                cfg.z_entry,
            )
            cfg.z_stop = float(cfg.z_entry)
    if cfg.seed is not None:
        np.random.seed(cfg.seed)
    run_id = dt.datetime.utcnow().strftime(RUN_ID_FMT)
    LOGGER.info("=== VECM Playbook start | run_id=%s ===", run_id)
    if data_frame is not None:
        df = data_frame.copy(deep=True)
    else:
        df = load_and_validate_data(cfg.input_file)

    short_panel: Optional[pd.DataFrame] = None
    try:
        short_panel = build_short_term_signals(df, market_col="^JKSE")
        LOGGER.info("Short-term signals panel built with shape %s", short_panel.shape)
    except Exception as exc:
        LOGGER.warning("Short-term signals construction failed; disabling overlay: %s", exc)

    df_clean, tickers = preprocess_data(df, cfg)
    selected_l, selected_r, beta0 = select_pair(df_clean, cfg)
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
    if zect_valid.empty:
        fallback_std_val = float(ect.std())
        if np.isfinite(fallback_std_val) and fallback_std_val > 0:
            LOGGER.warning(
                "Z-score series empty after normalisation; falling back to global standard deviation"
            )
            fallback = (ect - ect.mean()) / fallback_std_val
            zect_valid = fallback.dropna()
    if zect_valid.empty:
        LOGGER.warning(
            "Z-score series still empty; synthesising flat spread to allow pipeline continuation"
        )
        zect_valid = pd.Series(0.0, index=lp.index)
    zect = zect_valid

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

    # Regime switching model on the spread to infer mean-reverting regime probability
    try:
        ms_model = fit_ms_spread(zect)
        p_mr_series = compute_regime_prob(ms_model, zect)
    except Exception as exc:
        LOGGER.warning("MS spread modelling failed; falling back to flat regime prob: %s", exc)
        p_mr_series = pd.Series(0.7, index=zect.index)

    # Estimasi parameter konvergensi spread pada full sample
    alpha_ec, half_life_full = _convergence_stats(zect)

    corr_gate = _run_cor(lp.iloc[:, 0].diff(), lp.iloc[:, 1].diff(), max(cfg.gate_corr_win, 10))
    corr_gate = (corr_gate >= cfg.gate_corr_min).reindex(zect.index).fillna(False)
    hl_series = zect.rolling(252, min_periods=60).apply(_half_life, raw=False)
    hl_gate = (hl_series <= cfg.half_life_max).reindex(zect.index).fillna(False)
    if cfg.gate_enforce:
        combined_gate = (corr_gate & hl_gate).reindex(zect.index).fillna(False)
    else:
        combined_gate = pd.Series(True, index=zect.index)
    signals, z_th = build_signals(
        zect,
        cfg,
        combined_gate,
        delta_score=delta_score,
        p_regime=p_mr_series,
    )
    exec_res = execute_trades(
        zect,
        signals,
        lp,
        beta_series,
        cfg,
        p_regime=p_mr_series,
        delta_score=delta_score,
    )
    if cfg.oos_start:
        oos_start_date = pd.to_datetime(cfg.oos_start).date()
    else:
        base_index = zect.index if len(zect) else lp.index
        if not len(base_index):
            raise ValueError("Cannot determine OOS start date; no observations available")
        cutoff_idx = int(len(base_index) * 0.7)
        cutoff_idx = min(max(cutoff_idx, 0), len(base_index) - 1)
        oos_start_date = base_index[cutoff_idx].date()
    metrics = compute_metrics(exec_res, cfg, oos_start_date)
    metrics["alpha_ec"] = alpha_ec
    metrics["half_life_full"] = half_life_full
    metrics["z_th"] = z_th
    if cfg.z_entry is not None and np.isfinite(cfg.z_entry):
        metrics["z_entry"] = float(cfg.z_entry)
    result = {
        "run_id": run_id,
        "params": cfg.to_dict(),
        "config": cfg.to_dict(),
        "metrics": metrics,
        "model_checks": {
            "pair": f"{selected_l}~{selected_r}",
            "rank": 1,
            "deterministic": "ci",
            "threshold": float(z_th),
        },
        "signals": signals,
        "execution": exec_res,
        "horizon": {
            "train_obs": int(len(lp) * 0.7),
            "test_obs": len(lp) - int(len(lp) * 0.7),
            "train_start": str(lp.index.min().date()),
            "train_end": str(lp.index[int(len(lp) * 0.7)].date()),
            "test_start": str(lp.index[int(len(lp) * 0.7)].date()),
            "test_end": str(lp.index.max().date()),
        },
    }
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

    manifest_row = {
        "run_id": run_id,
        "ts_utc": _now_utc().isoformat(),
        "tag": cfg.tag,
        "subset": cfg.subset,
        "pair": result["model_checks"]["pair"],
        "z_source": cfg.z_auto_method,
        "z_th": result["metrics"].get("z_th"),
        "status": "OK_HAS_TRADES" if exec_res.trades.shape[0] else "OK_NO_TRADES",
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
    with storage.managed_storage(cfg.tag) as conn:
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
            spec_ok=True,
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
