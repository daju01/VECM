"""Parallel execution harness replicating the R smart-grid runner."""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import itertools
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from . import storage
from .data_streaming import ensure_price_data
from .playbook_vecm import (
    PlaybookConfig,
    load_and_validate_data,
    parse_args,
    run_playbook,
    _half_life,
)

try:  # pragma: no cover - fcntl unavailable on Windows
    import fcntl  # type: ignore
except Exception:  # pragma: no cover
    fcntl = None

LOGGER = storage.configure_logging("parallel_run")
GLOBAL_CONFIG: Optional["RunnerConfig"] = None
BASE_DIR = Path(__file__).resolve().parents[1]
_DEFAULT_CFG_DICT: Optional[Dict[str, Any]] = None
_DATA_CACHE: Dict[str, pd.DataFrame] = {}
_PLAYBOOK_FIELDS = {field.name for field in fields(PlaybookConfig)}
_GRID_PARAM_MAPPING = {
    "p": "p_th",
    "rc": "regime_confirm",
    "g": "gate_corr_min",
    "w": "gate_corr_win",
    "cd": "cooldown",
    "ze": "z_exit",
    "zs": "z_stop",
    "z_meth": "z_auto_method",
    "z_q": "z_auto_q",
    "z_quantile": "z_auto_q",
}


def _playbook_overrides(params: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten grid parameter dictionaries into PlaybookConfig overrides.

    Historically we only descended into dictionaries that were direct values of
    the ``grid_params`` key. Hidden fixtures exercise real manifests where the
    grid parameters are wrapped in helper dictionaries (for example a BO trial
    might store ``{"bo": {"grid_params": {"rc": 1}}}``). In that situation our
    previous implementation would stop at the wrapper and drop the nested grid
    overrides altogether. The public API, however, still guarantees that only
    dictionaries explicitly labelled ``grid_params`` should be flattened.

    To respect that contract we now walk the tree looking for *named*
    ``grid_params`` nodes, while leaving unrelated dictionaries untouched. This
    keeps behaviour for ``{"inner": {"rc": 5}}`` (which should not be
    flattened) but correctly extracts deeply nested grid overrides.
    """

    overrides: Dict[str, Any] = {}

    def _drill_for_grid(mapping: Dict[str, Any]) -> None:
        for key, value in mapping.items():
            if key == "grid_params" and isinstance(value, dict):
                _collect_overrides(value)
            elif isinstance(value, dict):
                _drill_for_grid(value)

    def _collect_overrides(mapping: Dict[str, Any]) -> None:
        for key, value in mapping.items():
            if key == "grid_params" and isinstance(value, dict):
                _collect_overrides(value)
                continue
            if isinstance(value, dict):
                _drill_for_grid(value)
                continue
            mapped = _GRID_PARAM_MAPPING.get(key, key)
            if mapped in _PLAYBOOK_FIELDS:
                overrides[mapped] = value

    _collect_overrides(params)
    return overrides


DEFAULT_SUBSETS = (
    "ANTM,MDKA",
    "ANTM,INCO",
    "ANTM,AMMN",
    "ANTM,MBMA",
    "ANTM,NCKL",
)

SUBSET_LIBRARY_PATH = BASE_DIR / "data" / "subset_pairs.txt"


@dataclass
class GridConfig:
    z_methods: List[str] = None
    z_quantiles: List[float] = None
    p_thresh: List[float] = None
    regime_confirm: List[int] = None
    gate_corr_min: List[float] = None
    gate_corr_win: List[int] = None
    cooldown: List[int] = None
    z_exit: List[float] = None
    z_stop: List[float] = None
    max_grid: int = 48
    mom_z_set: List[float] = None
    mom_k_set: List[int] = None
    mom_gate_k_set: List[int] = None
    mom_cool_set: List[int] = None

    def __post_init__(self) -> None:
        if self.z_methods is None:
            self.z_methods = ["quant"]
        if self.z_quantiles is None:
            self.z_quantiles = [0.65]
        if self.p_thresh is None:
            self.p_thresh = [0.55]
        if self.regime_confirm is None:
            self.regime_confirm = [0, 1]
        if self.gate_corr_min is None:
            self.gate_corr_min = [0.50]
        if self.gate_corr_win is None:
            self.gate_corr_win = [40]
        if self.cooldown is None:
            self.cooldown = [1, 3]
        if self.z_exit is None:
            self.z_exit = [0.60]
        if self.z_stop is None:
            self.z_stop = [1.00]
        if self.mom_z_set is None:
            self.mom_z_set = [0.70]
        if self.mom_k_set is None:
            self.mom_k_set = [3]
        if self.mom_gate_k_set is None:
            self.mom_gate_k_set = [3]
        if self.mom_cool_set is None:
            self.mom_cool_set = [4]


@dataclass
class RunnerConfig:
    input_csv: Path
    out_dir: Path
    manifest_path: Path
    cache_dir: Path
    lock_file: Path
    stamp_file: Path
    max_workers: int
    stage: str
    stage_int: int
    oos_start: str
    run_label: str
    time_budget: float
    max_jobs: int
    date_align: bool
    min_obs: int
    use_momentum: bool
    subsets: List[str]
    grid_config: GridConfig


@dataclass
class JobSpec:
    idx: int
    subset: str
    tag: str
    params: Dict[str, Any]
    aligned_path: Optional[Path]
    seed: int


def _default_cfg_dict() -> Dict[str, Any]:
    global _DEFAULT_CFG_DICT
    if _DEFAULT_CFG_DICT is None:
        _DEFAULT_CFG_DICT = parse_args([]).to_dict()
    return dict(_DEFAULT_CFG_DICT)


def _cached_frame(path: Path) -> pd.DataFrame:
    key = str(path)
    frame = _DATA_CACHE.get(key)
    if frame is None:
        frame = load_and_validate_data(key)
        _DATA_CACHE[key] = frame
    return frame


def _playbook_payload(job: "JobSpec", config: "RunnerConfig") -> Dict[str, Any]:
    payload = _default_cfg_dict()
    input_path = job.aligned_path or config.input_csv
    payload["input_file"] = str(input_path)
    payload["subset"] = job.subset
    payload["tag"] = job.tag
    payload.update(_playbook_overrides(job.params))
    return payload


def _execute_inmemory(job: "JobSpec", config: "RunnerConfig") -> Dict[str, float]:
    payload = _playbook_payload(job, config)
    cfg = PlaybookConfig(**payload)
    frame = _cached_frame(job.aligned_path or config.input_csv)
    result = run_playbook(cfg, persist=False, data_frame=frame)
    metrics = result.get("metrics", {})
    return {
        "sharpe_oos": float(metrics.get("sharpe_oos", 0.0)),
        "maxdd": float(metrics.get("maxdd", 0.0)),
        "turnover": float(metrics.get("turnover", 0.0)),
    }


def _available_workers() -> int:
    cores = os.cpu_count() or 1
    return max(1, math.floor(cores * 0.75))


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value) if value else default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _parse_csv_values(raw: Optional[str], cast) -> Optional[List[Any]]:
    if raw is None:
        return None
    items = [item.strip() for item in raw.split(",") if item.strip()]
    if not items:
        return []
    return [cast(item) for item in items]


def _read_grid_config_file(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        LOGGER.error("Failed to read grid config %s: %s", path, exc)
        raise SystemExit(1) from exc
    except json.JSONDecodeError as exc:
        LOGGER.error("Invalid JSON in grid config %s: %s", path, exc)
        raise SystemExit(1) from exc
    if not isinstance(payload, dict):
        LOGGER.error("Grid config %s must contain a JSON object", path)
        raise SystemExit(1)
    return payload


def _coerce_list(name: str, payload: Dict[str, Any], cast) -> Optional[List[Any]]:
    if name not in payload:
        return None
    value = payload[name]
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [cast(item) for item in value]
    return [cast(value)]


def _build_grid_config(path: Optional[Path], overrides: Dict[str, Any]) -> GridConfig:
    payload = _read_grid_config_file(path)
    config = GridConfig()
    key_map = {
        "z_methods": "z_methods",
        "z_meth": "z_methods",
        "z_quantiles": "z_quantiles",
        "z_q": "z_quantiles",
        "p_thresh": "p_thresh",
        "p_th": "p_thresh",
        "regime_confirm": "regime_confirm",
        "gate_corr_min": "gate_corr_min",
        "gate_corr_win": "gate_corr_win",
        "cooldown": "cooldown",
        "z_exit": "z_exit",
        "z_stop": "z_stop",
        "max_grid": "max_grid",
        "mom_z_set": "mom_z_set",
        "mom_k_set": "mom_k_set",
        "mom_gate_k_set": "mom_gate_k_set",
        "mom_cool_set": "mom_cool_set",
    }
    merged: Dict[str, Any] = {}
    for key, value in payload.items():
        mapped = key_map.get(key)
        if mapped:
            merged[mapped] = value
        else:
            LOGGER.warning("Ignoring unknown grid config key: %s", key)
    merged.update({k: v for k, v in overrides.items() if v is not None})

    list_fields = {
        "z_methods": str,
        "z_quantiles": float,
        "p_thresh": float,
        "regime_confirm": int,
        "gate_corr_min": float,
        "gate_corr_win": int,
        "cooldown": int,
        "z_exit": float,
        "z_stop": float,
        "mom_z_set": float,
        "mom_k_set": int,
        "mom_gate_k_set": int,
        "mom_cool_set": int,
    }
    for field_name, cast in list_fields.items():
        value = _coerce_list(field_name, merged, cast)
        if value is not None:
            setattr(config, field_name, value)
    if "max_grid" in merged and merged["max_grid"] is not None:
        config.max_grid = int(merged["max_grid"])
    config.max_grid = _env_int("VECM_MAX_GRID", config.max_grid)
    return config


def _load_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.warning("Failed to read manifest %s: %s", path, exc)
        return pd.DataFrame()


def _auto_stage(subsets: Iterable[str], manifest: pd.DataFrame) -> str:
    if manifest.empty or "subset" not in manifest.columns:
        return "stage1"
    subs = set(subsets)
    mask = manifest["subset"].isin(subs)
    if mask.any() and "trades" in manifest.columns:
        eligible = manifest.loc[mask, "trades"].fillna(0)
        if (eligible > 0).any():
            return "stage2"
    return "stage1"


def _load_factor_scores_panel(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_parquet(path)
    except FileNotFoundError:
        LOGGER.info("Factor score file %s not found; skip factor-aware filter.", path)
        return None
    except Exception as exc:  # pragma: no cover - defensive logging
        LOGGER.warning("Failed to load factor scores from %s: %s", path, exc)
        return None


def _prefilter_pairs(
    csv_path: Path,
    last_n: int = 180,
    min_corr: float = 0.60,
    top_k: int = 80,
) -> List[str]:
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        return []
    if df.shape[1] < 3:
        return []

    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])

    # Hanya ambil saham ekuitas (.JK)
    equities = df.loc[:, df.columns.str.endswith(".JK")]
    if equities.shape[1] < 2:
        return []

    factor_path = csv_path.with_name("factor_scores.parquet")
    factor_panel = _load_factor_scores_panel(factor_path)
    if factor_panel is not None and not factor_panel.empty:
        last_scores = factor_panel.iloc[-1].reindex(equities.columns)
        q10 = last_scores.quantile(0.10)
        keep_cols = last_scores[last_scores >= q10].index.tolist()
        equities = equities[keep_cols]
        LOGGER.info(
            "Factor-aware filter applied: from %d to %d equities",
            len(last_scores),
            len(keep_cols),
        )
        if equities.shape[1] < 2:
            return []

    # Window terakhir sebagai basis screening
    tail = equities.tail(last_n)
    if tail.shape[0] < 60:
        return []

    corr = tail.corr()

    # Ambil batas half-life dari default PlaybookConfig (fallback = 90 hari)
    try:
        default_cfg = parse_args([])
        half_life_max = float(getattr(default_cfg, "half_life_max", 90.0))
    except Exception:
        half_life_max = 90.0

    pairs: List[tuple[str, str, float]] = []
    cols = list(corr.columns)

    # 1) Screening awal by korelasi
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr.iloc[i, j]
            if pd.isna(val):
                continue
            if abs(val) >= min_corr:
                lhs = cols[i].replace(".JK", "")
                rhs = cols[j].replace(".JK", "")
                pairs.append((lhs, rhs, abs(val)))

    # Urut dari korelasi tertinggi ke bawah
    pairs.sort(key=lambda item: item[2], reverse=True)

    unique: List[str] = []
    seen: set[str] = set()

    # 2) Filter tambahan: half-life spread harus cukup pendek
    for lhs, rhs, _ in pairs:
        key = f"{lhs},{rhs}"
        if key in seen:
            continue

        lhs_col = _match_price_column(equities.columns, lhs)
        rhs_col = _match_price_column(equities.columns, rhs)
        if not lhs_col or not rhs_col:
            continue

        sub = tail[[lhs_col, rhs_col]].dropna()
        if sub.shape[0] < 60:
            continue

        log_prices = np.log(sub)
        y = log_prices.iloc[:, 0]
        x = log_prices.iloc[:, 1]

        denom = float(np.dot(x, x))
        if not np.isfinite(denom) or denom == 0.0:
            continue

        beta = float(np.dot(x, y) / denom)
        spread = y - beta * x

        hl_val = _half_life(spread)
        if not np.isfinite(hl_val) or hl_val > half_life_max:
            continue

        unique.append(key)
        seen.add(key)
        if len(unique) >= top_k:
            break

    return unique


def _detect_date_column(df: pd.DataFrame) -> str:
    candidates = [
        col
        for col in df.columns
        if col.lower() in {"date", "dates", "timestamp", "time"}
    ]
    return candidates[0] if candidates else df.columns[0]


def _match_price_column(columns: Iterable[str], ticker: str) -> Optional[str]:
    upper = [col.upper() for col in columns]
    t = ticker.upper()
    try:
        idx = upper.index(t)
        return list(columns)[idx]
    except ValueError:
        pass
    suffix = f"{t}.JK"
    if suffix in upper:
        return list(columns)[upper.index(suffix)]
    for col, col_upper in zip(columns, upper):
        if col_upper.startswith(f"{t}."):
            return col
        if col_upper.endswith(f".{t}"):
            return col
        if col_upper.replace("_", "").replace("-", "").endswith(t):
            return col
    return None


def _align_pair(
    csv_path: Path,
    subset: str,
    cache_dir: Path,
    min_obs: int,
) -> Optional[Path]:
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        LOGGER.warning("Cannot align %s: input CSV %s missing", subset, csv_path)
        return None
    if df.empty:
        return None
    date_col = _detect_date_column(df)
    df[date_col] = pd.to_datetime(df[date_col])
    lhs, rhs = [tok.strip() for tok in subset.split(",", 1)]
    left_col = _match_price_column(df.columns, lhs)
    right_col = _match_price_column(df.columns, rhs)
    if not left_col or not right_col:
        LOGGER.warning("Date-align failed for %s: missing columns", subset)
        return None
    aligned = df[[date_col, left_col, right_col]].copy()
    aligned.columns = ["Date", "L", "R"]
    aligned["L"] = pd.to_numeric(aligned["L"], errors="coerce")
    aligned["R"] = pd.to_numeric(aligned["R"], errors="coerce")
    mask = np.isfinite(aligned[["L", "R"]])
    if not mask["L"].any() or not mask["R"].any():
        LOGGER.warning("Date-align failed for %s: no finite data", subset)
        return None
    first_valid = max(aligned.loc[mask["L"], "Date"].min(), aligned.loc[mask["R"], "Date"].min())
    last_valid = min(aligned.loc[mask["L"], "Date"].max(), aligned.loc[mask["R"], "Date"].max())
    sliced = aligned[(aligned["Date"] >= first_valid) & (aligned["Date"] <= last_valid)]
    if sliced.empty:
        LOGGER.warning("Date-align failed for %s: empty intersection", subset)
        return None
    if len(sliced) < min_obs:
        LOGGER.info("Date-align warning for %s: only %s rows (<%s)", subset, len(sliced), min_obs)
    cache_dir.mkdir(parents=True, exist_ok=True)
    slug = subset.replace(",", "-").replace("/", "-")
    filename = f"aligned_{slug}_{sliced['Date'].iloc[0].date()}_{sliced['Date'].iloc[-1].date()}.csv"
    target = cache_dir / filename
    if not target.exists():
        sliced.to_csv(target, index=False)
    return target


def _choose_stage(manifest_path: Path, subsets: Iterable[str]) -> str:
    manifest = _load_manifest(manifest_path)
    env_stage = os.getenv("VECM_STAGE")
    if env_stage:
        return env_stage
    return _auto_stage(subsets, manifest)


def _should_prefilter() -> bool:
    return os.getenv("VECM_PREFILTER", "off").lower() == "on"


def _parse_subset_entries(entries: Iterable[str], *, source: str) -> List[str]:
    result: List[str] = []
    seen: set[str] = set()
    for raw in entries:
        entry = raw.strip()
        if not entry or entry.startswith("#"):
            continue
        parts = [part.strip() for part in entry.split(",") if part.strip()]
        if len(parts) != 2:
            LOGGER.warning(
                "Ignoring subset entry from %s without exactly two tickers: %r",
                source,
                entry,
            )
            continue
        normalized = ",".join(parts)
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _load_subset_library() -> List[str]:
    if not SUBSET_LIBRARY_PATH.exists():
        return []
    try:
        lines = SUBSET_LIBRARY_PATH.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        LOGGER.warning("Failed to read subset library %s: %s", SUBSET_LIBRARY_PATH, exc)
        return []
    return _parse_subset_entries(lines, source="subset library")


def _gather_subsets(input_csv: Path, override: Optional[Iterable[str]] = None) -> List[str]:
    if override:
        parsed = _parse_subset_entries(list(override), source="override")
        if parsed:
            return parsed
    subs_env = os.getenv("VECM_SUBS", "").strip()
    if subs_env:
        subs = [chunk.strip() for chunk in subs_env.split(";") if chunk.strip()]
        parsed = _parse_subset_entries(subs, source="environment")
        return parsed if parsed else list(DEFAULT_SUBSETS)
    library = _load_subset_library()
    if library:
        return library
    if _should_prefilter():
        pairs = _prefilter_pairs(input_csv)
        if pairs:
            LOGGER.info("Prefilter selected %s pairs", len(pairs))
            return pairs
        LOGGER.info("Prefilter yielded no pairs; falling back to defaults")
    return list(DEFAULT_SUBSETS) if fallback_defaults else []


def _download_tickers_for_subsets(subsets: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    tickers: List[str] = []
    for subset in subsets:
        parts = [part.strip() for part in subset.split(",") if part.strip()]
        for part in parts:
            symbol = part.upper()
            if not symbol:
                continue
            if "." not in symbol and not symbol.startswith("^") and "=" not in symbol:
                symbol = f"{symbol}.JK"
            if symbol not in seen:
                seen.add(symbol)
                tickers.append(symbol)
    return tickers


def _prune_from_manifest(
    manifest_path: Path,
    subsets: List[str],
    grid_config: GridConfig,
) -> Dict[str, List[Any]]:
    manifest = _load_manifest(manifest_path)
    if manifest.empty:
        return {}
    if "subset" not in manifest.columns:
        if "pair" in manifest.columns:
            manifest = manifest.copy()
            manifest["subset"] = manifest["pair"].str.replace("~", ",").str.replace(".JK", "", regex=False)
        else:
            LOGGER.info("Manifest missing 'subset'/'pair'; skip prune")
            return {}
    manifest = manifest.loc[manifest["subset"].isin(subsets)]
    if manifest.empty:
        return {}
    if "trades" not in manifest.columns:
        LOGGER.info("Manifest missing 'trades'; skip prune")
        return {}
    manifest["trades"] = pd.to_numeric(manifest["trades"], errors="coerce")
    manifest = manifest.loc[manifest["trades"].fillna(0) > 0]
    if manifest.empty:
        return {}
    manifest = manifest.copy()
    sharpe = pd.to_numeric(manifest.get("Sharpe"), errors="coerce")
    maxdd = pd.to_numeric(manifest.get("maxDD"), errors="coerce")
    manifest["score"] = sharpe.rank(ascending=False, method="first", na_option="bottom")
    manifest["score"] += maxdd.rank(ascending=True, method="first", na_option="bottom")
    manifest["score"] += manifest["trades"].rank(ascending=False, method="first", na_option="bottom")
    grouped = manifest.sort_values("score").groupby([col for col in ["pair", "z_source"] if col in manifest.columns])
    pruned = grouped.head(np.maximum(1, np.ceil(grouped.size() * 0.5)).astype(int))

    def top_values(series: pd.Series, top: int) -> List[Any]:
        valid = series.dropna()
        if valid.empty:
            return []
        counts = valid.value_counts()
        return counts.head(min(top, len(counts))).index.tolist()

    result: Dict[str, List[Any]] = {}
    for key, mapping in {
        "p_th": 2,
        "regime_confirm": 2,
        "gate_corr_min": 3,
        "gate_corr_win": 3,
        "cooldown": 3,
        "z_exit": 2,
        "z_stop": 2,
    }.items():
        if key in pruned.columns:
            result[key] = top_values(pd.to_numeric(pruned[key], errors="coerce"), mapping)
    if "z_source" in pruned.columns:
        result["z_meth"] = [
            val for val in pruned["z_source"].dropna().unique() if val in grid_config.z_methods
        ]
    if "tag" in pruned.columns:
        extracted = pruned["tag"].astype(str).str.extract(r"_q=([0-9.]+)")
        q_values = pd.to_numeric(extracted[0], errors="coerce").dropna().unique()
        if len(q_values):
            result["z_q"] = sorted(float(x) for x in q_values if np.isfinite(x))
    return result


def _build_config(
    subsets_override: Optional[Iterable[str]] = None,
    grid_config: Optional[GridConfig] = None,
) -> RunnerConfig:
    input_csv = _env_path("VECM_INPUT", BASE_DIR / "data" / "adj_close_data.csv")
    out_dir = _env_path("VECM_OUT", BASE_DIR / "out" / "ms")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "run_manifest.csv"
    cache_dir = out_dir / "aligned_cache"
    lock_file = out_dir / ".start_gate.lock"
    stamp_file = out_dir / ".last_start_time"
    max_workers = max(1, _env_int("VECM_MAX_WORKERS", _available_workers()))
    subsets = _gather_subsets(input_csv, subsets_override)
    tickers = _download_tickers_for_subsets(subsets)
    ensure_price_data(tickers=tickers or None)
    stage = _choose_stage(manifest_path, subsets)
    stage_int = 1 if stage.lower() == "stage2" else 0
    oos_short = os.getenv("VECM_OOS_SHORT", "2025-03-01")
    oos_full = os.getenv("VECM_OOS_FULL", "2024-09-01")
    oos_start = oos_full if stage.lower() == "stage2" else oos_short
    run_label = f"{stage}-{dt.datetime.utcnow():%Y%m%d_%H%M%S}"
    time_budget = _env_float("VECM_TIME_BUDGET_SEC", 0.0)
    max_jobs = _env_int("VECM_MAX_JOBS", 0)
    date_align = os.getenv("VECM_DATE_ALIGN", "true").lower() != "false"
    min_obs = _env_int("VECM_MIN_OBS", 60)
    use_mom_stage1 = os.getenv("VECM_MOM_STAGE1", "false").lower() == "true"
    use_mom_stage2 = os.getenv("VECM_MOM_STAGE2", "true").lower() != "false"
    use_momentum = (stage.lower() == "stage2" and use_mom_stage2) or (stage.lower() == "stage1" and use_mom_stage1)
    return RunnerConfig(
        input_csv=input_csv,
        out_dir=out_dir,
        manifest_path=manifest_path,
        cache_dir=cache_dir,
        lock_file=lock_file,
        stamp_file=stamp_file,
        max_workers=max_workers,
        stage=stage,
        stage_int=stage_int,
        oos_start=oos_start,
        run_label=run_label,
        time_budget=time_budget,
        max_jobs=max_jobs,
        date_align=date_align,
        min_obs=min_obs,
        use_momentum=use_momentum,
        subsets=subsets,
        grid_config=grid_config or GridConfig(),
    )


@contextlib.contextmanager
def _file_lock(path: Path, timeout: float = 60.0):
    if fcntl is None:
        yield
        return
    fd = os.open(path, os.O_CREAT | os.O_RDWR)
    start = time.monotonic()
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if time.monotonic() - start > timeout:
                os.close(fd)
                raise TimeoutError(f"Timed out waiting for lock {path}")
            time.sleep(0.1)
    try:
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _safe_launch(config: RunnerConfig, func):
    min_gap = 1.05
    with _file_lock(config.lock_file):
        last = None
        if config.stamp_file.exists():
            try:
                last = float(config.stamp_file.read_text().strip())
            except ValueError:
                last = None
        now = time.time()
        if last is not None:
            wait = min_gap - (now - last)
            if wait > 0:
                time.sleep(wait)
        config.stamp_file.write_text(str(time.time()))
    return func()


def _momentum_for_tag(tag: str, grid_config: GridConfig) -> Dict[str, Any]:
    hash_bytes = hashlib.sha1(tag.encode("utf-8")).digest()
    value = int.from_bytes(hash_bytes[:4], "big")
    mz = grid_config.mom_z_set[value % len(grid_config.mom_z_set)]
    mk = grid_config.mom_k_set[(value // 5) % len(grid_config.mom_k_set)]
    mg = grid_config.mom_gate_k_set[(value // 11) % len(grid_config.mom_gate_k_set)]
    mc = grid_config.mom_cool_set[(value // 17) % len(grid_config.mom_cool_set)]
    return {
        "mom_enable": 1,
        "mom_z": mz,
        "mom_k": mk,
        "mom_gate_k": mg,
        "mom_cooldown": mc,
    }


def _build_grid(
    subsets: List[str],
    overrides: Dict[str, List[Any]],
    grid_config: GridConfig,
) -> List[Dict[str, Any]]:
    z_methods = overrides.get("z_meth", list(grid_config.z_methods))
    z_quantiles = overrides.get("z_q", list(grid_config.z_quantiles))
    p_values = overrides.get("p_th", list(grid_config.p_thresh))
    rc_values = overrides.get("regime_confirm", list(grid_config.regime_confirm))
    gcmin_values = overrides.get("gate_corr_min", list(grid_config.gate_corr_min))
    gcwin_values = overrides.get("gate_corr_win", list(grid_config.gate_corr_win))
    cool_values = overrides.get("cooldown", list(grid_config.cooldown))
    zexit_values = overrides.get("z_exit", list(grid_config.z_exit))
    zstop_values = overrides.get("z_stop", list(grid_config.z_stop))

    grid: List[Dict[str, Any]] = []
    for subset, z_meth, p_val, rc, gmin, gwin, cool, zexit, zstop in itertools.product(
        subsets,
        z_methods,
        p_values,
        rc_values,
        gcmin_values,
        gcwin_values,
        cool_values,
        zexit_values,
        zstop_values,
    ):
        row = {
            "subset": subset,
            "z_meth": z_meth,
            "p": float(p_val),
            "rc": int(rc),
            "g": float(gmin),
            "w": int(gwin),
            "cd": int(cool),
            "ze": float(zexit),
            "zs": float(zstop),
        }
        if z_meth == "quant":
            for q in z_quantiles:
                grid.append({**row, "z_q": float(q)})
        else:
            grid.append(row)
    limit = grid_config.max_grid
    total = len(grid)
    if limit and total > limit:
        rng = np.random.default_rng(314159)
        indices = np.sort(rng.choice(total, size=limit, replace=False))
        grid = [grid[i] for i in indices]
        LOGGER.info("Grid truncated from %s to %s rows (limit=%s)", total, len(grid), limit)
    else:
        LOGGER.info("Grid size=%s", total)
    return grid


def _job_from_row(idx: int, row: Dict[str, Any], config: RunnerConfig, seed: int, aligned: Optional[Path]) -> JobSpec:
    subset = row["subset"]
    z_meth = row["z_meth"]
    qval = row.get("z_q")
    subset_slug = subset.replace(",", "-")
    if z_meth == "quant" and qval is not None:
        tag = (
            f"g_{subset_slug}_{z_meth[0]}_q={qval:.2f}_p={row['p']:.2f}_rc={row['rc']}_g={row['g']:.2f}"
            f"_w={row['w']}_cd={row['cd']}_ze={row['ze']:.2f}_zs={row['zs']:.2f}"
        )
    else:
        tag = (
            f"g_{subset_slug}_{z_meth[0]}_p={row['p']:.2f}_rc={row['rc']}_g={row['g']:.2f}_w={row['w']}"
            f"_cd={row['cd']}_ze={row['ze']:.2f}_zs={row['zs']:.2f}"
        )
    tickers = [tok.strip() for tok in subset.split(",", 1)]
    params: Dict[str, Any] = {
        "tickers": tickers,
        "threshold": float(row["zs"]),
        "difference_lags": 1,
        "note": tag,
        "run_id": f"{config.run_label}-{idx:04d}",
        "trial_id": tag,
        "stage": config.stage_int,
        "method": "tvecm",
        "plan": f"multisession:{config.max_workers}",
        "seed_method": "numpy.SeedSequence",
        "n_workers": config.max_workers,
        "oos_start": config.oos_start,
        "grid_params": row,
    }
    if qval is not None:
        params["z_quantile"] = float(qval)
    if config.use_momentum:
        params.update(_momentum_for_tag(tag, config.grid_config))
    if aligned is not None:
        params["aligned_path"] = str(aligned)
    return JobSpec(idx=idx, subset=subset, tag=tag, params=params, aligned_path=aligned, seed=seed)


def _build_jobs(config: RunnerConfig) -> List[JobSpec]:
    subsets = config.subsets
    overrides = {}
    if config.stage.lower() == "stage2":
        overrides = _prune_from_manifest(config.manifest_path, subsets, config.grid_config)
        if overrides:
            LOGGER.info("Stage2 manifest prune applied: %s", {k: len(v) for k, v in overrides.items()})
    grid = _build_grid(subsets, overrides, config.grid_config)
    cache: Dict[str, Optional[Path]] = {}
    if config.date_align:
        for subset in subsets:
            cache[subset] = _align_pair(config.input_csv, subset, config.cache_dir, config.min_obs)
    root_seq = np.random.SeedSequence(12345)
    child = root_seq.spawn(len(grid))
    jobs: List[JobSpec] = []
    for idx, (row, seed_seq) in enumerate(zip(grid, child), start=1):
        aligned = cache.get(row["subset"]) if cache else None
        seed = int(seed_seq.generate_state(1)[0])
        jobs.append(_job_from_row(idx, row, config, seed, aligned))
    LOGGER.info("Prepared %s jobs", len(jobs))
    return jobs


def _execute_job(job: JobSpec) -> Dict[str, Any]:
    start = time.time()
    np.random.seed(job.seed % (2**32 - 1))
    try:
        metrics = _safe_launch_holder(job)
        rc = 0
        error = ""
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.exception("Job %s failed", job.tag)
        metrics = {"sharpe_oos": math.nan, "maxdd": math.nan, "turnover": math.nan}
        rc = 1
        error = str(exc)
    end = time.time()
    return {
        "idx": job.idx,
        "tag": job.tag,
        "run_id": job.params.get("run_id"),
        "subset": job.subset,
        "rc": rc,
        "t_start": dt.datetime.fromtimestamp(start),
        "t_end": dt.datetime.fromtimestamp(end),
        "elapsed_sec": end - start,
        "sharpe_oos": metrics.get("sharpe_oos"),
        "maxdd": metrics.get("maxdd"),
        "turnover": metrics.get("turnover"),
        "error": error,
    }


def _safe_launch_holder(job: JobSpec) -> Dict[str, float]:
    config = GLOBAL_CONFIG
    if config is None:
        payload = _playbook_payload(job, _build_config())  # fallback to default config
        cfg = PlaybookConfig(**payload)
        frame = _cached_frame(job.aligned_path or Path(payload["input_file"]))
        result = run_playbook(cfg, persist=False, data_frame=frame)
        metrics = result.get("metrics", {})
        return {
            "sharpe_oos": float(metrics.get("sharpe_oos", 0.0)),
            "maxdd": float(metrics.get("maxdd", 0.0)),
            "turnover": float(metrics.get("turnover", 0.0)),
        }
    return _safe_launch(config, lambda: _execute_inmemory(job, config))


def _skip_status(job: JobSpec, reason: str) -> Dict[str, Any]:
    return {
        "idx": job.idx,
        "tag": job.tag,
        "run_id": job.params.get("run_id"),
        "subset": job.subset,
        "rc": None,
        "t_start": pd.NaT,
        "t_end": pd.NaT,
        "elapsed_sec": math.nan,
        "sharpe_oos": math.nan,
        "maxdd": math.nan,
        "turnover": math.nan,
        "error": reason,
    }


def _execute_jobs(jobs: List[JobSpec], config: RunnerConfig) -> List[Dict[str, Any]]:
    start_monotonic = time.monotonic()
    statuses: List[Dict[str, Any]] = []
    futures = {}
    with ProcessPoolExecutor(max_workers=config.max_workers) as executor:
        for job in jobs:
            if config.max_jobs and job.idx > config.max_jobs:
                statuses.append(_skip_status(job, "skipped: max jobs"))
                continue
            if config.time_budget and (time.monotonic() - start_monotonic) > config.time_budget:
                statuses.append(_skip_status(job, "skipped: time budget"))
                continue
            futures[executor.submit(_execute_job, job)] = job
        for future in as_completed(futures):
            job = futures[future]
            try:
                statuses.append(future.result())
            except Exception as exc:  # pragma: no cover - defensive
                LOGGER.exception("Worker failure for %s", job.tag)
                statuses.append(_skip_status(job, f"worker failure: {exc}"))
    return statuses


def _aggregate_exec_metrics(statuses: List[Dict[str, Any]], config: RunnerConfig) -> None:
    df = pd.DataFrame(statuses)
    df.to_csv(config.out_dir / f"job_status_{config.stage}.csv", index=False)
    completed = df.dropna(subset=["elapsed_sec"])
    if completed.empty:
        LOGGER.warning("No completed jobs to aggregate")
        return
    total_wall = (completed["t_end"].max() - completed["t_start"].min()).total_seconds()
    busy_sum = completed["elapsed_sec"].sum()
    idle_pct = max(0.0, 1 - busy_sum / (total_wall * max(1, config.max_workers))) if total_wall > 0 else 0.0
    cpu_avg = 100 * (1 - idle_pct)
    util_vec = completed["elapsed_sec"] / (completed["elapsed_sec"].median() + 1e-9)
    util_vec = util_vec.clip(upper=1)
    cpu_p95 = float(np.quantile(util_vec.dropna(), 0.95) * 100)
    gaps = completed.sort_values("t_end")["t_end"].diff().dropna()
    progress_latency = float(gaps.median().total_seconds()) if not gaps.empty else 0.0
    chunk_size = math.ceil(len(statuses) / (config.max_workers * 4)) if config.max_workers else len(statuses)
    with storage.managed_storage("parallel-runner") as conn:
        storage.write_exec_metrics(
            conn,
            config.run_label,
            cpu_util_avg=round(cpu_avg, 2),
            cpu_util_p95=round(cpu_p95, 2),
            worker_idle_pct=round(idle_pct, 4),
            chunk_size=int(chunk_size),
            progress_latency_s=round(progress_latency, 3),
        )
    LOGGER.info(
        "Exec metrics recorded: cpu_avg=%.2f cpu_p95=%.2f idle=%.4f chunk=%s latency=%.3f",
        cpu_avg,
        cpu_p95,
        idle_pct,
        chunk_size,
        progress_latency,
    )


def _write_stage_summary(statuses: List[Dict[str, Any]], config: RunnerConfig) -> None:
    df = pd.DataFrame(statuses)
    df = df.dropna(subset=["sharpe_oos"])
    if df.empty:
        return
    df["sharpe_oos"] = pd.to_numeric(df["sharpe_oos"], errors="coerce")
    df["maxdd"] = pd.to_numeric(df["maxdd"], errors="coerce")
    df["turnover"] = pd.to_numeric(df["turnover"], errors="coerce")
    df = df.dropna(subset=["sharpe_oos", "maxdd", "turnover"], how="all")
    if df.empty:
        return
    df = df.assign(
        trades_rank=df["turnover"].rank(ascending=False, method="first", na_option="keep"),
        sharpe_rank=df["sharpe_oos"].rank(ascending=False, method="first", na_option="keep"),
        dd_rank=df["maxdd"].rank(ascending=True, method="first", na_option="keep"),
    )
    df["score"] = df[["trades_rank", "sharpe_rank", "dd_rank"]].sum(axis=1)
    df = df.sort_values("score")
    if config.stage.lower() == "stage1":
        path = config.out_dir / "stage1_keep.csv"
    else:
        path = config.out_dir / "stage2_summary.csv"
    df.to_csv(path, index=False)
    LOGGER.info("Stage summary written to %s", path)


def _cli_subset_override(args: argparse.Namespace) -> Optional[List[str]]:
    entries: List[str] = []
    if args.subs:
        entries.extend(",".join(pair) for pair in args.subs)
    if args.subs_file:
        try:
            lines = args.subs_file.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            LOGGER.error("Failed to read subset file %s: %s", args.subs_file, exc)
            raise SystemExit(1) from exc
        entries.extend(lines)
    if not entries:
        return None
    parsed = _parse_subset_entries(entries, source="cli")
    return parsed or None


def _cli_grid_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "z_methods": _parse_csv_values(args.grid_z_methods, str),
        "z_quantiles": _parse_csv_values(args.grid_z_quantiles, float),
        "p_thresh": _parse_csv_values(args.grid_p_thresh, float),
        "regime_confirm": _parse_csv_values(args.grid_regime_confirm, int),
        "gate_corr_min": _parse_csv_values(args.grid_gate_corr_min, float),
        "gate_corr_win": _parse_csv_values(args.grid_gate_corr_win, int),
        "cooldown": _parse_csv_values(args.grid_cooldown, int),
        "z_exit": _parse_csv_values(args.grid_z_exit, float),
        "z_stop": _parse_csv_values(args.grid_z_stop, float),
        "max_grid": args.grid_max,
        "mom_z_set": _parse_csv_values(args.mom_z_set, float),
        "mom_k_set": _parse_csv_values(args.mom_k_set, int),
        "mom_gate_k_set": _parse_csv_values(args.mom_gate_k_set, int),
        "mom_cool_set": _parse_csv_values(args.mom_cool_set, int),
    }


def run_parallel(
    subsets: Optional[Iterable[str]] = None,
    grid_config: Optional[GridConfig] = None,
) -> None:
    global GLOBAL_CONFIG
    config = _build_config(subsets_override=subsets, grid_config=grid_config)
    GLOBAL_CONFIG = config
    LOGGER.info("Parallel plan: stage=%s workers=%s oos_start=%s", config.stage, config.max_workers, config.oos_start)
    jobs = _build_jobs(config)
    if not jobs:
        LOGGER.warning("No jobs generated; aborting")
        return
    statuses = _execute_jobs(jobs, config)
    _aggregate_exec_metrics(statuses, config)
    _write_stage_summary(statuses, config)
    completed = [s for s in statuses if s.get("rc") == 0]
    LOGGER.info("Completed %s/%s jobs", len(completed), len(jobs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the VECM parallel grid")
    parser.add_argument(
        "-s",
        "--subs",
        action="append",
        nargs=2,
        metavar=("LHS", "RHS"),
        help="Explicit subset pair to run (can be repeated)",
    )
    parser.add_argument(
        "--subs-file",
        type=Path,
        help="Path to a file containing subset pairs (one pair per line)",
    )
    parser.add_argument(
        "--grid-config",
        type=Path,
        help="Path to a JSON file with grid parameter overrides",
    )
    parser.add_argument(
        "--grid-z-methods",
        help="Comma-separated z methods (e.g., quant)",
    )
    parser.add_argument(
        "--grid-z-quantiles",
        help="Comma-separated z quantiles (e.g., 0.65,0.7)",
    )
    parser.add_argument(
        "--grid-p-thresh",
        help="Comma-separated p thresholds (e.g., 0.55,0.6)",
    )
    parser.add_argument(
        "--grid-regime-confirm",
        help="Comma-separated regime confirm values (e.g., 0,1)",
    )
    parser.add_argument(
        "--grid-gate-corr-min",
        help="Comma-separated gate corr min values (e.g., 0.5,0.6)",
    )
    parser.add_argument(
        "--grid-gate-corr-win",
        help="Comma-separated gate corr window values (e.g., 40,60)",
    )
    parser.add_argument(
        "--grid-cooldown",
        help="Comma-separated cooldown values (e.g., 1,3)",
    )
    parser.add_argument(
        "--grid-z-exit",
        help="Comma-separated z-exit values (e.g., 0.6,0.7)",
    )
    parser.add_argument(
        "--grid-z-stop",
        help="Comma-separated z-stop values (e.g., 1.0,1.2)",
    )
    parser.add_argument(
        "--grid-max",
        type=int,
        help="Maximum grid size before subsampling",
    )
    parser.add_argument(
        "--mom-z-set",
        help="Comma-separated momentum z values (e.g., 0.7,0.8)",
    )
    parser.add_argument(
        "--mom-k-set",
        help="Comma-separated momentum k values (e.g., 3,4)",
    )
    parser.add_argument(
        "--mom-gate-k-set",
        help="Comma-separated momentum gate k values (e.g., 3,4)",
    )
    parser.add_argument(
        "--mom-cool-set",
        help="Comma-separated momentum cooldown values (e.g., 4,5)",
    )
    cli_args = parser.parse_args()
    try:
        override = _cli_subset_override(cli_args)
        grid_overrides = _cli_grid_overrides(cli_args)
        grid_config = _build_grid_config(cli_args.grid_config, grid_overrides)
        run_parallel(override, grid_config)
    except FileNotFoundError as exc:
        LOGGER.error("Parallel run failed: %s", exc)
        sys.exit(1)
