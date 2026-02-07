"""Shared helpers for parallel run configuration parsing."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional


def env_path(name: str, default: Path) -> Path:
    """Read an environment variable as a Path with a fallback."""
    raw = os.getenv(name)
    if raw:
        return Path(raw).expanduser()
    return default


def env_float(name: str, default: float) -> float:
    """Read an environment variable as float, returning default on failure."""
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def env_int(name: str, default: int) -> int:
    """Read an environment variable as int, returning default on failure."""
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def parse_csv_values(raw: Optional[str], cast) -> Optional[List[Any]]:
    """Parse a CSV string into a list, casting each element."""
    if not raw:
        return None
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        return None
    return [cast(item) for item in values]


def coerce_list(name: str, payload: Dict[str, Any], cast) -> Optional[List[Any]]:
    """Coerce a JSON payload entry into a list of casted values."""
    raw = payload.get(name)
    if raw is None:
        return None
    if isinstance(raw, list):
        return [cast(item) for item in raw]
    if isinstance(raw, str):
        return parse_csv_values(raw, cast)
    return None
