"""Shared IDX-aware cost and liquidity helpers for playbook/stage2 scoring."""
from __future__ import annotations

import math
import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

TINY = 1e-12


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


def _coalesce_float(value: Optional[float], env_name: str, default: float) -> float:
    if value is not None and np.isfinite(float(value)):
        return float(value)
    return _env_float(env_name, default)


def _coalesce_int(value: Optional[int], env_name: str, default: int) -> int:
    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            pass
    return _env_int(env_name, default)


def resolve_idx_cost_config(
    *,
    fee_buy: float,
    fee_sell: float,
    broker_buy_rate: Optional[float] = None,
    broker_sell_rate: Optional[float] = None,
    exchange_levy: Optional[float] = None,
    sell_tax: Optional[float] = None,
    spread_bps: Optional[float] = None,
    impact_model: Optional[str] = None,
    impact_k: Optional[float] = None,
    adtv_win: Optional[int] = None,
    sigma_win: Optional[int] = None,
    illiq_cap_mode: Optional[str] = None,
    illiq_cap_value: Optional[float] = None,
    calmar_eps: Optional[float] = None,
) -> Dict[str, Any]:
    """Resolve IDX cost model parameters from config with env fallback."""

    raw_impact_model = (impact_model or os.getenv("IDX_IMPACT_MODEL", "sqrt")).strip().lower()
    if raw_impact_model not in {"sqrt", "linear", "none"}:
        raw_impact_model = "sqrt"
    raw_illiq_mode = (illiq_cap_mode or os.getenv("IDX_ILLIQ_CAP_MODE", "insample_p80")).strip().lower()
    if raw_illiq_mode not in {"insample_p80", "static"}:
        raw_illiq_mode = "insample_p80"

    return {
        "broker_buy_rate": _coalesce_float(broker_buy_rate, "IDX_BROKER_BUY_RATE", fee_buy),
        "broker_sell_rate": _coalesce_float(broker_sell_rate, "IDX_BROKER_SELL_RATE", fee_sell),
        "levy_rate": _coalesce_float(exchange_levy, "IDX_LEVY_RATE", 0.0),
        "sell_tax_rate": _coalesce_float(sell_tax, "IDX_SELL_TAX_RATE", 0.001),
        "spread_bps": _coalesce_float(spread_bps, "IDX_SPREAD_BPS", 20.0),
        "impact_model": raw_impact_model,
        "impact_k": _coalesce_float(impact_k, "IDX_IMPACT_K", 1.0),
        "adtv_win": max(1, _coalesce_int(adtv_win, "IDX_ADTV_WIN", 20)),
        "sigma_win": max(2, _coalesce_int(sigma_win, "IDX_SIGMA_WIN", 20)),
        "illiq_cap_mode": raw_illiq_mode,
        "illiq_cap_value": _coalesce_float(illiq_cap_value, "IDX_ILLIQ_CAP_VALUE", float("nan")),
        "calmar_eps": _coalesce_float(calmar_eps, "IDX_CALMAR_EPS", 0.01),
    }


def _safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    den_clean = den.copy()
    den_clean = den_clean.where(np.isfinite(den_clean) & (den_clean > 0), np.nan)
    out = num / den_clean
    return out.replace([np.inf, -np.inf], np.nan)


def compute_cost_events(
    *,
    dexp1: pd.Series,
    dexp2: pd.Series,
    nav_prev: pd.Series,
    close1: pd.Series,
    close2: pd.Series,
    ret1: pd.Series,
    ret2: pd.Series,
    volume1: Optional[pd.Series] = None,
    volume2: Optional[pd.Series] = None,
    broker_buy_rate: float = 0.0019,
    broker_sell_rate: float = 0.0029,
    levy_rate: float = 0.0,
    sell_tax_rate: float = 0.001,
    spread_bps: float = 20.0,
    impact_model: str = "sqrt",
    impact_k: float = 1.0,
    adtv_win: int = 20,
    sigma_win: int = 20,
) -> pd.DataFrame:
    """Compute per-bar cost components from executed delta exposures."""

    idx = dexp1.index
    dexp1 = dexp1.reindex(idx).fillna(0.0).astype(float)
    dexp2 = dexp2.reindex(idx).fillna(0.0).astype(float)
    nav_prev = nav_prev.reindex(idx).fillna(1.0).astype(float)
    close1 = close1.reindex(idx).astype(float)
    close2 = close2.reindex(idx).astype(float)
    ret1 = ret1.reindex(idx).fillna(0.0).astype(float)
    ret2 = ret2.reindex(idx).fillna(0.0).astype(float)
    if volume1 is None:
        volume1 = pd.Series(np.nan, index=idx, dtype=float)
    else:
        volume1 = volume1.reindex(idx).astype(float)
    if volume2 is None:
        volume2 = pd.Series(np.nan, index=idx, dtype=float)
    else:
        volume2 = volume2.reindex(idx).astype(float)

    notional_leg1 = dexp1.abs() * nav_prev
    notional_leg2 = dexp2.abs() * nav_prev
    buy_notional_leg1 = notional_leg1.where(dexp1 > 0, 0.0)
    buy_notional_leg2 = notional_leg2.where(dexp2 > 0, 0.0)
    sell_notional_leg1 = notional_leg1.where(dexp1 < 0, 0.0)
    sell_notional_leg2 = notional_leg2.where(dexp2 < 0, 0.0)

    dollar_leg1 = close1 * volume1
    dollar_leg2 = close2 * volume2
    adtv_leg1 = dollar_leg1.rolling(max(1, int(adtv_win)), min_periods=1).mean()
    adtv_leg2 = dollar_leg2.rolling(max(1, int(adtv_win)), min_periods=1).mean()

    spread_rate = max(float(spread_bps), 0.0) / 10_000.0
    sigma_leg1 = ret1.rolling(max(2, int(sigma_win)), min_periods=2).std().fillna(0.0)
    sigma_leg2 = ret2.rolling(max(2, int(sigma_win)), min_periods=2).std().fillna(0.0)

    impact_mode = str(impact_model or "sqrt").strip().lower()
    impact_k_safe = max(float(impact_k), 0.0)
    ratio1 = _safe_ratio(notional_leg1, adtv_leg1).fillna(0.0).clip(lower=0.0)
    ratio2 = _safe_ratio(notional_leg2, adtv_leg2).fillna(0.0).clip(lower=0.0)
    if impact_mode == "none" or impact_k_safe <= 0:
        impact_leg1 = pd.Series(0.0, index=idx, dtype=float)
        impact_leg2 = pd.Series(0.0, index=idx, dtype=float)
    elif impact_mode == "linear":
        impact_leg1 = impact_k_safe * ratio1 * notional_leg1
        impact_leg2 = impact_k_safe * ratio2 * notional_leg2
    else:
        impact_leg1 = impact_k_safe * sigma_leg1 * np.sqrt(ratio1) * notional_leg1
        impact_leg2 = impact_k_safe * sigma_leg2 * np.sqrt(ratio2) * notional_leg2

    broker = (
        buy_notional_leg1 * max(float(broker_buy_rate), 0.0)
        + buy_notional_leg2 * max(float(broker_buy_rate), 0.0)
        + sell_notional_leg1 * max(float(broker_sell_rate), 0.0)
        + sell_notional_leg2 * max(float(broker_sell_rate), 0.0)
    )
    levy = (notional_leg1 + notional_leg2) * max(float(levy_rate), 0.0)
    spread = 0.5 * spread_rate * (notional_leg1 + notional_leg2)
    impact = impact_leg1 + impact_leg2
    sell_tax = (sell_notional_leg1 + sell_notional_leg2) * max(float(sell_tax_rate), 0.0)
    total = broker + levy + spread + impact + sell_tax

    min_adtv = pd.concat([adtv_leg1, adtv_leg2], axis=1).min(axis=1, skipna=False)
    max_notional = pd.concat([notional_leg1, notional_leg2], axis=1).max(axis=1)
    participation = _safe_ratio(max_notional, min_adtv)
    turnover_daily = _safe_ratio(notional_leg1 + notional_leg2, nav_prev).fillna(0.0)

    return pd.DataFrame(
        {
            "notional_leg1": notional_leg1,
            "notional_leg2": notional_leg2,
            "buy_notional_leg1": buy_notional_leg1,
            "buy_notional_leg2": buy_notional_leg2,
            "sell_notional_leg1": sell_notional_leg1,
            "sell_notional_leg2": sell_notional_leg2,
            "adtv_leg1": adtv_leg1,
            "adtv_leg2": adtv_leg2,
            "sigma_leg1": sigma_leg1,
            "sigma_leg2": sigma_leg2,
            "participation": participation,
            "turnover_daily": turnover_daily,
            "cost_broker": broker,
            "cost_levy": levy,
            "cost_spread": spread,
            "cost_impact": impact,
            "cost_sell_tax": sell_tax,
            "cost_total": total,
        },
        index=idx,
    )


def compute_liquidity_metrics(
    *,
    close1: pd.Series,
    close2: pd.Series,
    volume1: Optional[pd.Series],
    volume2: Optional[pd.Series],
    ret1: pd.Series,
    ret2: pd.Series,
    notional_leg1: pd.Series,
    notional_leg2: pd.Series,
    oos_mask: pd.Series,
    adtv_win: int = 20,
    illiq_cap_mode: str = "insample_p80",
    illiq_cap_value: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute ADTV, participation, and Amihud ILLIQ metrics."""

    idx = close1.index
    close1 = close1.reindex(idx).astype(float)
    close2 = close2.reindex(idx).astype(float)
    ret1 = ret1.reindex(idx).fillna(0.0).astype(float)
    ret2 = ret2.reindex(idx).fillna(0.0).astype(float)
    notional_leg1 = notional_leg1.reindex(idx).fillna(0.0).astype(float)
    notional_leg2 = notional_leg2.reindex(idx).fillna(0.0).astype(float)
    if volume1 is None:
        volume1 = pd.Series(np.nan, index=idx, dtype=float)
    else:
        volume1 = volume1.reindex(idx).astype(float)
    if volume2 is None:
        volume2 = pd.Series(np.nan, index=idx, dtype=float)
    else:
        volume2 = volume2.reindex(idx).astype(float)

    adtv_win = max(1, int(adtv_win))
    dollar1 = close1 * volume1
    dollar2 = close2 * volume2
    adtv_leg1 = dollar1.rolling(adtv_win, min_periods=1).mean()
    adtv_leg2 = dollar2.rolling(adtv_win, min_periods=1).mean()

    illiq_leg1 = _safe_ratio(ret1.abs(), dollar1)
    illiq_leg2 = _safe_ratio(ret2.abs(), dollar2)
    illiq_pair = pd.concat([illiq_leg1, illiq_leg2], axis=1).max(axis=1, skipna=False)

    min_adtv = pd.concat([adtv_leg1, adtv_leg2], axis=1).min(axis=1, skipna=False)
    max_notional = pd.concat([notional_leg1, notional_leg2], axis=1).max(axis=1)
    participation = _safe_ratio(max_notional, min_adtv)

    oos_mask = oos_mask.reindex(idx).fillna(False).astype(bool)
    in_sample_mask = ~oos_mask

    illiq_mode = (illiq_cap_mode or "insample_p80").strip().lower()
    cap_value = float("nan")
    if illiq_mode == "static":
        if illiq_cap_value is not None and np.isfinite(float(illiq_cap_value)):
            cap_value = float(illiq_cap_value)
    else:
        baseline = illiq_pair[in_sample_mask].dropna()
        if baseline.empty:
            baseline = illiq_pair.dropna()
        if not baseline.empty:
            cap_value = float(np.percentile(baseline.values, 80))
        illiq_mode = "insample_p80"

    illiq_oos = illiq_pair[oos_mask].dropna()
    if illiq_oos.empty:
        illiq_oos = illiq_pair.dropna()
    amihud_illiq = float(illiq_oos.mean()) if not illiq_oos.empty else float("nan")

    part_oos = participation[oos_mask].dropna()
    if part_oos.empty:
        part_oos = participation.dropna()
    participation_mean = float(part_oos.mean()) if not part_oos.empty else 0.0
    participation_max = float(part_oos.max()) if not part_oos.empty else 0.0

    return {
        "adtv_leg1": adtv_leg1,
        "adtv_leg2": adtv_leg2,
        "participation": participation,
        "illiq_pair": illiq_pair,
        "amihud_illiq": amihud_illiq,
        "participation_mean": participation_mean,
        "participation_max": participation_max,
        "illiq_cap": cap_value,
        "illiq_cap_mode": illiq_mode,
    }


def compute_nav_cagr_calmar(
    ret_net: pd.Series,
    *,
    ann_days: int = 252,
    calmar_eps: float = 0.01,
) -> Dict[str, Any]:
    """Build NAV path and compute CAGR/Calmar from net daily returns."""

    if ret_net is None or ret_net.empty:
        empty_nav = pd.Series(dtype=float)
        return {
            "nav": empty_nav,
            "nav_end": 1.0,
            "cagr": 0.0,
            "maxdd_raw": 0.0,
            "maxdd": 0.0,
            "calmar": 0.0,
        }

    ret_clean = ret_net.fillna(0.0).astype(float)
    nav = (1.0 + ret_clean).cumprod()
    nav_start = float(nav.iloc[0]) if not nav.empty else 1.0
    nav_end = float(nav.iloc[-1]) if not nav.empty else 1.0
    obs = max(len(ret_clean), 1)
    if nav_start > 0 and nav_end > 0:
        cagr = float((nav_end / nav_start) ** (float(ann_days) / float(obs)) - 1.0)
    else:
        cagr = 0.0
    dd = nav / nav.cummax() - 1.0
    maxdd_raw = float(dd.min()) if not dd.empty and np.isfinite(dd.min()) else 0.0
    maxdd = float(abs(min(maxdd_raw, 0.0)))
    calmar_den = max(maxdd, max(float(calmar_eps), TINY))
    calmar = float(cagr / calmar_den) if np.isfinite(cagr) else 0.0

    return {
        "nav": nav,
        "nav_end": nav_end,
        "cagr": cagr,
        "maxdd_raw": maxdd_raw,
        "maxdd": maxdd,
        "calmar": calmar,
    }
