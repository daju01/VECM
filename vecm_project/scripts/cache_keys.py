"""Stable cache key helpers for VECM pipeline artifacts."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Mapping

import pandas as pd


def _normalise_config(config: Any) -> Any:
    if dataclasses.is_dataclass(config):
        return dataclasses.asdict(config)
    if isinstance(config, Mapping):
        return {str(key): _normalise_config(value) for key, value in config.items()}
    if isinstance(config, (list, tuple)):
        return [_normalise_config(value) for value in config]
    return config


def hash_config(config: Any) -> str:
    """Hash a configuration object using sorted JSON."""
    payload = _normalise_config(config)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def hash_dataframe(df: pd.DataFrame) -> str:
    """Hash a DataFrame using deterministic row/column hashing."""
    if df is None:
        return hashlib.sha256(b"").hexdigest()
    frame = df.copy()
    frame.columns = [str(col) for col in frame.columns]
    data_hash = pd.util.hash_pandas_object(frame, index=True).to_numpy()
    col_hash = pd.util.hash_pandas_object(pd.Index(frame.columns), index=False).to_numpy()
    hasher = hashlib.sha256()
    hasher.update(data_hash.tobytes())
    hasher.update(col_hash.tobytes())
    return hasher.hexdigest()
