"""Markov-switching spread regime model utilities.

This module provides a light-weight 2-state Markov-switching regression on the
spread/z-score series and surfaces the filtered probability that the spread is
currently in a mean-reverting regime.  When fitting fails or the sample is too
short, it falls back to a flat probability to avoid breaking the pipeline.
"""
from __future__ import annotations

import json
import logging
import os
import pathlib
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd

from .cache_keys import hash_config, hash_dataframe

try:  # statsmodels is optional at runtime
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
except Exception:  # pragma: no cover - defensive import guard
    MarkovRegression = None  # type: ignore

LOGGER = logging.getLogger(__name__)
_MS_SPREAD_CACHE: Dict[str, Dict[str, Any]] = {}
_MS_WARM_START_CACHE: Dict[str, np.ndarray] = {}
_MAX_ITER_ENV = "VECM_MS_MAX_ITER"
_TOL_ENV = "VECM_MS_TOL"
_CACHE_DIR = pathlib.Path(__file__).resolve().parents[1] / "cache" / "ms_spread"


def _build_cache_key(
    z_series: pd.Series,
    cache_key: Optional[object],
    pair_id: Optional[str],
    data_hash: Optional[str],
    ms_config_hash: str,
) -> str:
    if cache_key is not None:
        return str(cache_key)

    resolved_data_hash = data_hash
    if resolved_data_hash is None:
        resolved_data_hash = hash_dataframe(z_series.to_frame("zscore"))

    payload = {
        "pair_id": pair_id or "unknown",
        "data_hash": resolved_data_hash,
        "ms_config_hash": ms_config_hash,
    }
    return hash_config(payload)


def _resolve_max_iter(max_iter: int) -> int:
    env_value = os.getenv(_MAX_ITER_ENV)
    if env_value:
        try:
            return max(int(env_value), 1)
        except ValueError:
            LOGGER.warning("Invalid %s=%s; using max_iter=%d", _MAX_ITER_ENV, env_value, max_iter)
    return max_iter


def _resolve_tol(tol: Optional[float]) -> Optional[float]:
    env_value = os.getenv(_TOL_ENV)
    if env_value:
        try:
            return float(env_value)
        except ValueError:
            LOGGER.warning("Invalid %s=%s; ignoring", _TOL_ENV, env_value)
    return tol


def _warm_start_path(cache_key: str) -> pathlib.Path:
    return _CACHE_DIR / f"{cache_key}.json"


def _load_warm_start(cache_key: str) -> Optional[np.ndarray]:
    cached = _MS_WARM_START_CACHE.get(cache_key)
    if cached is not None:
        return cached

    path = _warm_start_path(cache_key)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
        params = payload.get("params")
        if isinstance(params, list):
            arr = np.asarray(params, dtype=float)
            _MS_WARM_START_CACHE[cache_key] = arr
            return arr
    except Exception as exc:  # pragma: no cover - cache corruption
        LOGGER.warning("Failed to read warm-start params at %s: %s", path, exc)
    return None


def _save_warm_start(cache_key: str, params: np.ndarray) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"params": [float(value) for value in np.asarray(params).ravel().tolist()]}
    path = _warm_start_path(cache_key)
    try:
        path.write_text(json.dumps(payload))
    except Exception as exc:  # pragma: no cover - cache write failure
        LOGGER.warning("Failed to persist warm-start params at %s: %s", path, exc)
    else:
        _MS_WARM_START_CACHE[cache_key] = np.asarray(params, dtype=float)


def _extract_converged(result: Any) -> bool:
    if result is None:
        return False
    retvals = getattr(result, "mle_retvals", None)
    if isinstance(retvals, Mapping):
        return bool(retvals.get("converged", False))
    return bool(getattr(result, "converged", False))


def fit_ms_spread(
    z_series: pd.Series,
    *,
    k_regimes: int = 2,
    min_len: int = 80,
    max_iter: int = 200,
    tol: Optional[float] = None,
    cache_key: Optional[object] = None,
    pair_id: Optional[str] = None,
    data_hash: Optional[str] = None,
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
    tol:
        Optional convergence tolerance override.
    pair_id:
        Identifier for the pair/regression target to help stabilize caching.
    data_hash:
        Stable hash for input data; if absent a hash of the series is used.

    Returns
    -------
    dict
        At least contains:
        - "p_mr": pd.Series probability of MR regime per date,
        - "regime_mr": regime index assumed to be MR,
        - "success": bool,
        - "result": statsmodels result (or None on failure).
    """

    resolved_max_iter = _resolve_max_iter(max_iter)
    resolved_tol = _resolve_tol(tol)
    ms_config_hash = hash_config(
        {
            "k_regimes": k_regimes,
            "min_len": min_len,
            "max_iter": resolved_max_iter,
            "tol": resolved_tol,
        }
    )
    cache_key_value = _build_cache_key(z_series, cache_key, pair_id, data_hash, ms_config_hash)
    cached = _MS_SPREAD_CACHE.get(cache_key_value)
    if cached is not None:
        return cached

    z = z_series.dropna().astype(float)
    if len(z) < min_len or MarkovRegression is None:
        reason = "insufficient_data" if len(z) < min_len else "statsmodels_unavailable"
        LOGGER.info(
            "MS spread fit skipped (len=%d, has_markov=%s)",
            len(z),
            MarkovRegression is not None,
        )
        p = pd.Series(0.7, index=z_series.index)
        model = {
            "p_mr": p,
            "regime_mr": None,
            "success": False,
            "result": None,
            "error": reason,
            "skipped": True,
            "converged": False,
            "fallback": "flat",
            "cache_key": cache_key_value,
        }
        _MS_SPREAD_CACHE[cache_key_value] = model
        return model

    mod = MarkovRegression(
        z.values,
        k_regimes=k_regimes,
        trend="c",
        order=0,
        switching_variance=True,
    )
    start_params = _load_warm_start(cache_key_value)
    if start_params is not None and start_params.size != mod.k_params:
        start_params = None

    res = None
    fit_error: Optional[str] = None
    for attempt, iter_limit in enumerate([resolved_max_iter, max(resolved_max_iter * 2, resolved_max_iter + 50)]):
        try:
            res = mod.fit(
                maxiter=iter_limit,
                disp=False,
                tol=resolved_tol,
                start_params=start_params,
            )
            if _extract_converged(res):
                break
            fit_error = "non_converged"
            start_params = res.params
        except Exception as exc:  # pragma: no cover
            fit_error = str(exc)
            LOGGER.warning("MS spread fit failed (attempt %d): %s", attempt + 1, exc)
            start_params = None

    if res is None or not _extract_converged(res):
        p = pd.Series(0.7, index=z_series.index)
        model = {
            "p_mr": p,
            "regime_mr": None,
            "success": False,
            "result": res,
            "error": fit_error or "fit_failed",
            "skipped": False,
            "converged": False,
            "fallback": "proxy_flat",
            "cache_key": cache_key_value,
        }
        _MS_SPREAD_CACHE[cache_key_value] = model
        return model

    params = res.params
    sigma2_vals = []
    param_index = getattr(params, "index", None)
    param_names = getattr(res, "param_names", None)
    for i in range(k_regimes):
        key = f"sigma2[{i}]"
        if param_index is not None and key in param_index:
            sigma2_vals.append(float(params[key]))
        elif param_names and key in param_names:
            sigma2_vals.append(float(params[param_names.index(key)]))
        else:
            sigma2_vals.append(np.nan)

    regime_mr = int(np.nanargmin(sigma2_vals)) if np.isfinite(np.nanmin(sigma2_vals)) else 0

    try:
        probs = res.filtered_marginal_probabilities
        if isinstance(probs, pd.DataFrame):
            if regime_mr in probs.columns:
                series = probs[regime_mr]
            elif probs.shape[1] >= k_regimes:
                series = probs.iloc[:, regime_mr]
            else:
                series = probs.iloc[regime_mr]
            p = pd.Series(series, index=z.index)
        else:
            arr = np.asarray(probs)
            if arr.ndim == 2:
                if arr.shape[0] == k_regimes and arr.shape[1] != k_regimes:
                    values = arr[regime_mr]
                elif arr.shape[1] >= k_regimes:
                    values = arr[:, regime_mr]
                else:
                    values = arr[regime_mr]
            else:
                values = arr
            p = pd.Series(values, index=z.index)
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Failed to extract filtered probabilities: %s", exc)
        p = pd.Series(0.7, index=z.index)
        model = {
            "p_mr": p,
            "regime_mr": None,
            "success": False,
            "result": res,
            "error": str(exc),
            "skipped": True,
            "converged": _extract_converged(res),
            "fallback": "proxy_flat",
            "cache_key": cache_key_value,
        }
        _MS_SPREAD_CACHE[cache_key_value] = model
        return model

    p_full = p.reindex(z_series.index)
    if p_full.isna().any():
        p_full = p_full.ffill().bfill().fillna(0.7)

    model = {
        "p_mr": p_full.astype(float),
        "regime_mr": regime_mr,
        "success": True,
        "result": res,
        "error": "",
        "skipped": False,
        "converged": _extract_converged(res),
        "fallback": "",
        "cache_key": cache_key_value,
    }
    if model["converged"]:
        _save_warm_start(cache_key_value, np.asarray(res.params))
    _MS_SPREAD_CACHE[cache_key_value] = model
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
