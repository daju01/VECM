"""Automated monitoring + retraining loop for VECM Stage-2 optimisation."""
from __future__ import annotations

import json
import os
import pathlib
from typing import Any, Dict, List, Mapping, Optional, Tuple

import duckdb
import pandas as pd

from . import storage
from .playbook_vecm import OUT_DIR, parse_args
from .stage2_bo import run_bo

LOGGER = storage.configure_logging("auto_retrain")
BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = BASE_DIR / "config" / "daily_pairs.json"
RUN_MANIFEST = OUT_DIR / "run_manifest.csv"


def _load_latest_run() -> Tuple[Optional[str], Optional[pd.Series]]:
    if RUN_MANIFEST.exists():
        manifest = pd.read_csv(RUN_MANIFEST)
        if not manifest.empty and "ts_utc" in manifest.columns:
            manifest["ts_utc"] = pd.to_datetime(manifest["ts_utc"], errors="coerce")
            manifest = manifest.sort_values("ts_utc", ascending=False)
            row = manifest.iloc[0]
            return str(row.get("run_id")), row
    candidates = list(OUT_DIR.glob("metrics_*.csv"))
    if not candidates:
        return None, None
    latest_file = max(candidates, key=lambda path: path.stat().st_mtime)
    run_id = latest_file.stem.replace("metrics_", "")
    return run_id, None


def _load_manifest_params(run_id: str) -> Dict[str, Any]:
    manifest_path = OUT_DIR / "artifacts" / run_id / "manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        LOGGER.warning("Failed to read manifest params: %s", exc)
        return {}
    params = payload.get("params", {})
    return params if isinstance(params, dict) else {}


def _latest_maxdd_from_duckdb() -> Optional[float]:
    try:
        with storage.storage_open(read_only=True) as conn:
            query = """
                SELECT maxdd
                FROM run_metrics
                ORDER BY metric_ts DESC
                LIMIT 1
            """
            df = conn.execute(query).df()
        if df.empty:
            return None
        return float(df.iloc[0]["maxdd"])
    except duckdb.Error as exc:
        LOGGER.warning("DuckDB query failed: %s", exc)
        return None


def _latest_maxdd_from_csv() -> Optional[float]:
    candidates = list(OUT_DIR.glob("metrics_*.csv"))
    if not candidates:
        return None
    latest_file = max(candidates, key=lambda path: path.stat().st_mtime)
    df = pd.read_csv(latest_file)
    if df.empty or "maxdd" not in df.columns:
        return None
    return float(df.iloc[0]["maxdd"])


def _resolve_latest_maxdd() -> Optional[float]:
    maxdd = _latest_maxdd_from_duckdb()
    if maxdd is not None:
        return maxdd
    return _latest_maxdd_from_csv()


def _load_config_payload(path: pathlib.Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        default_params = dict(payload.get("default_params", {}))
        raw_pairs = payload.get("pairs", [])
    else:
        default_params = {}
        raw_pairs = payload
    if not isinstance(raw_pairs, list):
        raise ValueError("Config pairs must be a list")
    return raw_pairs, default_params


def _pair_key(entry: Mapping[str, Any]) -> Optional[str]:
    pair = entry.get("pair") or entry.get("subset")
    if pair:
        return str(pair)
    ticker_a = entry.get("tickerA")
    ticker_b = entry.get("tickerB")
    if ticker_a and ticker_b:
        return f"{ticker_a},{ticker_b}"
    return None


def _deploy_params(
    config_path: pathlib.Path,
    pair: str,
    params: Mapping[str, Any],
) -> None:
    raw_pairs, default_params = _load_config_payload(config_path)
    updated = False

    for idx, entry in enumerate(raw_pairs):
        if isinstance(entry, str):
            entry_dict: Dict[str, Any] = {"pair": entry, "params": {}}
        elif isinstance(entry, Mapping):
            entry_dict = dict(entry)
        else:
            continue
        entry_pair = _pair_key(entry_dict)
        if entry_pair != pair:
            raw_pairs[idx] = entry_dict
            continue
        entry_params = dict(entry_dict.get("params", {}))
        entry_params.update(params)
        entry_dict["params"] = entry_params
        raw_pairs[idx] = entry_dict
        updated = True

    if not updated:
        raw_pairs.append({"pair": pair, "params": dict(params)})

    payload: Dict[str, Any]
    if default_params:
        payload = {"default_params": default_params, "pairs": raw_pairs}
    else:
        payload = {"pairs": raw_pairs}
    config_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def _run_stage2_optimizer(pair: str, params: Mapping[str, Any]) -> Dict[str, Any]:
    method = params.get("method", "TVECM")
    horizon = params.get("horizon", "oos_full")
    mode = os.getenv("VECM_AUTO_RETRAIN_MODE", "bo").lower()
    LOGGER.info("Auto-retraining trigger for %s (method=%s horizon=%s)", pair, method, horizon)
    if mode == "sh":
        try:
            from .stage2_sh import run_successive_halving

            study = run_successive_halving(pair=pair, method=method, cfg=params)
            best_params = study.best_trial.user_attrs.get("params", {})
        except Exception as exc:
            LOGGER.warning("Successive halving failed; falling back to BO: %s", exc)
            study = run_bo(pair=pair, method=method, horizon=horizon, cfg=params)
            best_params = study.best_trial.params
    else:
        study = run_bo(pair=pair, method=method, horizon=horizon, cfg=params)
        best_params = study.best_trial.params
    return {
        "z_entry": float(best_params.get("z_entry")),
        "z_exit": float(best_params.get("z_exit")),
        "max_hold": int(best_params.get("max_hold")),
        "cooldown": int(best_params.get("cooldown")),
        "p_th": float(best_params.get("p_th")),
    }


def run_auto_retraining(
    *,
    config_path: pathlib.Path = DEFAULT_CONFIG_PATH,
    dd_stop: Optional[float] = None,
) -> Dict[str, Any]:
    run_id, manifest_row = _load_latest_run()
    if run_id is None:
        return {"status": "no_runs"}

    manifest_params = _load_manifest_params(run_id)
    pair = manifest_params.get("subset") or manifest_params.get("pair")
    if not pair and manifest_row is not None:
        pair = manifest_row.get("subset") or manifest_row.get("pair")
    if not pair:
        return {"status": "missing_pair"}

    maxdd = _resolve_latest_maxdd()
    if maxdd is None:
        return {"status": "missing_metrics"}

    if dd_stop is None:
        dd_stop = float(manifest_params.get("dd_stop") or parse_args([]).dd_stop)

    if abs(float(maxdd)) < float(dd_stop):
        LOGGER.info("MaxDD %.4f below dd_stop %.4f; skipping retrain", maxdd, dd_stop)
        return {"status": "healthy", "maxdd": float(maxdd), "dd_stop": float(dd_stop)}

    retrain_params = _run_stage2_optimizer(str(pair), manifest_params)
    _deploy_params(config_path, str(pair), retrain_params)
    LOGGER.info("Auto-retraining complete; deployed params for %s", pair)
    return {
        "status": "retrained",
        "pair": str(pair),
        "maxdd": float(maxdd),
        "dd_stop": float(dd_stop),
        "params": retrain_params,
    }


if __name__ == "__main__":
    result = run_auto_retraining()
    print(json.dumps(result, indent=2))
