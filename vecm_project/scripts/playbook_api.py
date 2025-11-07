"""Thin adapter around the Python VECM playbook matching the R workflow."""
from __future__ import annotations

import json
import re
import sys
import time
from functools import lru_cache
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional

import pandas as pd

from .data_streaming import DATA_PATH, load_cached_prices
from .playbook_vecm import pipeline

# ---------------------------------------------------------------------------
# Helpers mirroring the R adapter -------------------------------------------------
# ---------------------------------------------------------------------------

BOOLEAN_TRUE = {"true", "t", "1", "yes", "y"}
BOOLEAN_FALSE = {"false", "f", "0", "no", "n"}


def _coerce_value(value: Any) -> Any:
    """Best-effort conversion from string values to Python primitives."""

    if isinstance(value, str):
        text = value.strip()
        lower = text.lower()
        if lower in BOOLEAN_TRUE:
            return True
        if lower in BOOLEAN_FALSE:
            return False
        try:
            if re.fullmatch(r"[+-]?\d+", text):
                return int(text)
            if re.fullmatch(r"[+-]?(?:\d*\.\d+|\d+\.\d*)(?:[eE][+-]?\d+)?", text):
                return float(text)
        except ValueError:
            return text
        return text
    if isinstance(value, MutableMapping):
        return {k: _coerce_value(v) for k, v in value.items()}
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        return type(value)(_coerce_value(v) for v in value)  # type: ignore[arg-type]
    return value


def _normalise_params(params: Mapping[str, Any]) -> Dict[str, Any]:
    """Align user supplied parameters with the Python playbook schema."""

    normalised = {k: _coerce_value(v) for k, v in dict(params).items()}
    if "pair" in normalised and not normalised.get("subset"):
        normalised["subset"] = normalised["pair"]
    if "z_auto" in normalised and "z_auto_q" not in normalised:
        normalised["z_auto_q"] = normalised["z_auto"]
    if "gate_enforce" in normalised:
        normalised["gate_enforce"] = bool(_coerce_value(normalised["gate_enforce"]))
    if "beta_weight" in normalised:
        normalised["beta_weight"] = bool(_coerce_value(normalised["beta_weight"]))
    if "mom_enable" in normalised:
        normalised["mom_enable"] = bool(_coerce_value(normalised["mom_enable"]))
    return normalised


# ---------------------------------------------------------------------------
# Public API ----------------------------------------------------------------
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def _cached_prices(path: str) -> pd.DataFrame:
    return load_cached_prices(path)


def playbook_score_once(params: Mapping[str, Any]) -> Dict[str, float]:
    """Execute the Python playbook once and surface the core metrics."""

    normalised = _normalise_params(params)
    input_path = str(normalised.get("input_file") or DATA_PATH)
    frame: Optional[pd.DataFrame]
    try:
        frame = _cached_prices(str(input_path))
    except FileNotFoundError:
        frame = None
    start_perf = time.perf_counter()
    result = pipeline(normalised, persist=False, data_frame=frame)
    _ = time.perf_counter() - start_perf
    metrics = result.get("metrics", {})
    score = {
        "sharpe_oos": float(metrics.get("sharpe_oos", 0.0)),
        "maxdd": float(metrics.get("maxdd", 0.0)),
        "turnover": float(metrics.get("turnover", 0.0)),
    }

    return score


# ---------------------------------------------------------------------------
# CLI entry point -----------------------------------------------------------
# ---------------------------------------------------------------------------

def _parse_cli(argv: Iterable[str]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for arg in argv:
        if not arg.startswith("--"):
            continue
        if "=" in arg:
            key, value = arg[2:].split("=", 1)
            params[key.replace("-", "_")] = _coerce_value(value)
        else:
            params[arg[2:].replace("-", "_")] = True
    return params


if __name__ == "__main__":
    args = _parse_cli(sys.argv[1:])
    if not args:
        print("{}", file=sys.stdout)
        sys.exit(0)
    result = playbook_score_once(args)
    print(json.dumps(result, indent=2, default=float))
