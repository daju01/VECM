"""Markov-switching spread regime model utilities.

This module provides a light-weight 2-state Markov-switching regression on the
spread/z-score series and surfaces the filtered probability that the spread is
currently in a mean-reverting regime.  When fitting fails or the sample is too
short, it falls back to a flat probability to avoid breaking the pipeline.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

try:  # statsmodels is optional at runtime
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
except Exception:  # pragma: no cover - defensive import guard
    MarkovRegression = None  # type: ignore

LOGGER = logging.getLogger(__name__)
_MS_SPREAD_CACHE: Dict[Tuple[object, ...], Dict[str, Any]] = {}


def _build_cache_key(
    z_series: pd.Series,
    cache_key: Optional[object],
    k_regimes: int,
    min_len: int,
    max_iter: int,
) -> Tuple[object, ...]:
    if cache_key is not None:
        return (cache_key, k_regimes, min_len, max_iter)

    if z_series.empty:
        return ("empty", k_regimes, min_len, max_iter)

    series_hash = int(pd.util.hash_pandas_object(z_series, index=True).sum())
    start = z_series.index[0]
    end = z_series.index[-1]
    return (series_hash, len(z_series), start, end, k_regimes, min_len, max_iter)


def fit_ms_spread(
    z_series: pd.Series,
    *,
    k_regimes: int = 2,
    min_len: int = 80,
    max_iter: int = 200,
    cache_key: Optional[object] = None,
) -> Dict[str, Any]:
    """Fit a Markov-switching model on the spread/z-score series.

    Parameters
    ----------
    z_series:
        Spread atau z-score (index: tanggal).
    k_regimes:
        Number of regimes (default 2).
    min_len:
        Minimum length required to attempt fitting.
    max_iter:
        Maximum ML iterations.

    Returns
    -------
    dict
        At least contains:
        - "p_mr": pd.Series probability of MR regime per date,
        - "regime_mr": regime index assumed to be MR,
        - "success": bool,
        - "result": statsmodels result (or None on failure).
    """

    cache_key_tuple = _build_cache_key(z_series, cache_key, k_regimes, min_len, max_iter)
    cached = _MS_SPREAD_CACHE.get(cache_key_tuple)
    if cached is not None:
        return cached

    z = z_series.dropna().astype(float)
    if len(z) < min_len or MarkovRegression is None:
        LOGGER.info(
            "MS spread fit skipped (len=%d, has_markov=%s)",
            len(z),
            MarkovRegression is not None,
        )
        p = pd.Series(0.7, index=z_series.index)
        model = {"p_mr": p, "regime_mr": None, "success": False, "result": None}
        _MS_SPREAD_CACHE[cache_key_tuple] = model
        return model

    try:
        mod = MarkovRegression(
            z.values,
            k_regimes=k_regimes,
            trend="c",
            order=0,
            switching_variance=True,
        )
        res = mod.fit(maxiter=max_iter, disp=False)
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("MS spread fit failed: %s", exc)
        p = pd.Series(0.7, index=z_series.index)
        model = {"p_mr": p, "regime_mr": None, "success": False, "result": None}
        _MS_SPREAD_CACHE[cache_key_tuple] = model
        return model

    params = res.params
    sigma2_vals = []
    for i in range(k_regimes):
        key = f"sigma2[{i}]"
        if key in params.index:
            sigma2_vals.append(float(params[key]))
        else:
            sigma2_vals.append(np.nan)

    regime_mr = int(np.nanargmin(sigma2_vals)) if np.isfinite(np.nanmin(sigma2_vals)) else 0

    try:
        probs = res.filtered_marginal_probabilities[regime_mr]
        p = pd.Series(probs, index=z.index)
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Failed to extract filtered probabilities: %s", exc)
        p = pd.Series(0.7, index=z.index)

    p_full = p.reindex(z_series.index)
    if p_full.isna().any():
        p_full = p_full.ffill().bfill().fillna(0.7)

    model = {
        "p_mr": p_full.astype(float),
        "regime_mr": regime_mr,
        "success": True,
        "result": res,
    }
    _MS_SPREAD_CACHE[cache_key_tuple] = model
    return model


def compute_regime_prob(
    model: Mapping[str, Any],
    series: Optional[pd.Series] = None,
) -> pd.Series:
    """Extract MR regime probabilities from the fitted model.

    When a ``series`` is provided, the probabilities are reindexed to its
    index, with basic forward/backward fill for any gaps.
    """

    p = model.get("p_mr")
    if not isinstance(p, pd.Series):
        raise ValueError("Model does not contain a 'p_mr' Series")

    if series is None:
        return p

    aligned = p.reindex(series.index)
    if aligned.isna().any():
        aligned = aligned.ffill().bfill().fillna(0.7)
    return aligned.astype(float)
