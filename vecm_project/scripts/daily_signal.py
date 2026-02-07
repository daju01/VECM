from __future__ import annotations

import datetime as dt
import json
import logging
import pathlib
import re
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import pandas as pd

from . import playbook_vecm

LOGGER = logging.getLogger(__name__)
BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
CONFIG_PATH = BASE_DIR / "config" / "daily_pairs.json"
OUTPUT_DIR = BASE_DIR / "outputs" / "daily"

BOOLEAN_TRUE = {"true", "t", "1", "yes", "y"}
BOOLEAN_FALSE = {"false", "f", "0", "no", "n"}


def _coerce_value(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        lower = text.lower()
        if lower in BOOLEAN_TRUE:
            return True
        if lower in BOOLEAN_FALSE:
            return False
        if re.fullmatch(r"[+-]?\d+", text):
            return int(text)
        if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", text):
            return float(text)
        return text
    if isinstance(value, MutableMapping):
        return {k: _coerce_value(v) for k, v in value.items()}
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        return type(value)(_coerce_value(v) for v in value)
    return value


def _load_config(path: pathlib.Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any], bool]:
    if not path.exists():
        raise FileNotFoundError(f"daily pairs config not found at {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    default_params: Dict[str, Any] = {}
    persist_artifacts = True
    if isinstance(payload, dict):
        default_params = dict(payload.get("default_params", {}))
        persist_artifacts = bool(payload.get("persist_artifacts", True))
        raw_pairs = payload.get("pairs", [])
    else:
        raw_pairs = payload

    if not isinstance(raw_pairs, Sequence):
        raise ValueError("daily_pairs.json must contain a list of pairs or a dict with a 'pairs' list")

    pairs: List[Dict[str, Any]] = []
    for entry in raw_pairs:
        if isinstance(entry, str):
            pairs.append({"pair": entry, "params": {}})
            continue
        if isinstance(entry, Mapping):
            entry_dict = dict(entry)
            pair = entry_dict.pop("pair", entry_dict.pop("subset", None))
            if not pair:
                raise ValueError(f"Pair entry missing 'pair'/'subset': {entry}")
            params = dict(entry_dict.pop("params", {}))
            params.update(entry_dict)
            pairs.append({"pair": pair, "params": params})
            continue
        raise ValueError(f"Unsupported pair entry type: {type(entry)}")

    return pairs, default_params, persist_artifacts


def _build_config(pair: str, params: Mapping[str, Any]) -> playbook_vecm.PlaybookConfig:
    base_cfg = playbook_vecm.parse_args([])
    cfg_dict = base_cfg.to_dict()
    cfg_dict.update({k: _coerce_value(v) for k, v in params.items() if v is not None})
    cfg_dict["subset"] = pair
    if "pair" in cfg_dict:
        cfg_dict.pop("pair")
    return playbook_vecm.PlaybookConfig(**cfg_dict)


def _infer_trade_direction(signals: pd.DataFrame) -> str:
    if signals.empty:
        return "FLAT"
    last = signals.iloc[-1]
    if last.get("long", 0.0) > 0:
        return "LONG"
    if last.get("short", 0.0) > 0:
        return "SHORT"
    return "FLAT"


def _infer_action(direction: str) -> str:
    normalized = direction.upper()
    if normalized == "LONG":
        return "BUY_SPREAD"
    if normalized == "SHORT":
        return "SELL_SPREAD"
    if normalized == "CLOSE":
        return "CLOSE"
    return "HOLD"


def _extract_entry_prices(signals: pd.DataFrame) -> Dict[str, Optional[float]]:
    if signals.empty:
        return {"entry_price_l": None, "entry_price_r": None}
    last = signals.iloc[-1].to_dict()
    for key in ("entry_price_l", "entry_price_r"):
        if key not in last:
            last[key] = None
    return {
        "entry_price_l": _coerce_float(last.get("entry_price_l")),
        "entry_price_r": _coerce_float(last.get("entry_price_r")),
    }


def _extract_target_ratio(signals: pd.DataFrame) -> Optional[float]:
    if signals.empty:
        return None
    last = signals.iloc[-1].to_dict()
    for key in ("target_ratio", "hedge_ratio", "beta"):
        if key in last:
            return _coerce_float(last.get(key))
    return None


def _compute_confidence(z_score: Optional[float], z_th: Optional[float]) -> Optional[float]:
    if z_score is None or z_th in (None, 0):
        return None
    if not (pd.notna(z_score) and pd.notna(z_th)):
        return None
    return float(min(1.0, abs(z_score) / float(z_th)))


def _summarize_overlay(execution: playbook_vecm.ExecutionResult) -> Dict[str, Optional[float]]:
    overlay: Dict[str, Optional[float]] = {}
    if execution.delta_score is not None and not execution.delta_score.empty:
        overlay["delta_score"] = float(execution.delta_score.iloc[-1])
    else:
        overlay["delta_score"] = None
    if execution.delta_mom12 is not None and not execution.delta_mom12.empty:
        overlay["delta_mom12"] = float(execution.delta_mom12.iloc[-1])
    else:
        overlay["delta_mom12"] = None
    return overlay


def run_daily_signals(config_path: pathlib.Path = CONFIG_PATH) -> List[Dict[str, Any]]:
    pairs, default_params, persist_artifacts = _load_config(config_path)
    timestamp = dt.datetime.utcnow().isoformat()
    results: List[Dict[str, Any]] = []

    base_run_id = dt.datetime.utcnow().strftime(playbook_vecm.RUN_ID_FMT)

    for idx, entry in enumerate(pairs, start=1):
        pair = entry["pair"]
        params = dict(default_params)
        params.update(entry.get("params", {}))
        cfg = _build_config(pair, params)
        run_id = f"{base_run_id}_{idx:02d}"
        feature_result = playbook_vecm.build_features(
            cfg.subset,
            playbook_vecm.FeatureConfig(
                base_config=cfg,
                pair=cfg.subset,
                method=cfg.method,
                horizon=cfg.horizon,
                data_frame=None,
                run_id=run_id,
            ),
        )
        if feature_result.skip_result is not None:
            result = feature_result.skip_result
        else:
            decision_params = playbook_vecm.DecisionParams(
                z_entry=cfg.z_entry,
                z_exit=cfg.z_exit,
                max_hold=cfg.max_hold,
                cooldown=cfg.cooldown,
                run_id=run_id,
            )
            result = playbook_vecm.evaluate_rules(feature_result, decision_params)

        if persist_artifacts:
            playbook_vecm.persist_artifacts(run_id, cfg, result)

        execution: playbook_vecm.ExecutionResult = result["execution"]
        signals = result.get("signals")
        if isinstance(signals, pd.DataFrame):
            signals_df = signals
        else:
            signals_df = pd.DataFrame()
        z_score = None
        if feature_result.features is not None and not feature_result.features.zect.empty:
            z_score = float(feature_result.features.zect.iloc[-1])
        regime = None
        if execution.p_regime is not None and not execution.p_regime.empty:
            regime = float(execution.p_regime.iloc[-1])
        overlay = _summarize_overlay(execution)
        metrics = result.get("metrics", {})
        z_th = metrics.get("z_th") if isinstance(metrics, Mapping) else None
        confidence = _compute_confidence(z_score, z_th)
        direction = _infer_trade_direction(signals_df)
        action = _infer_action(direction)
        entry_prices = _extract_entry_prices(signals_df)
        target_ratio = _extract_target_ratio(signals_df)
        risk_metrics = {
            "potential_drawdown": metrics.get("maxdd") if isinstance(metrics, Mapping) else None,
            "capital_at_risk": metrics.get("capital_at_risk") if isinstance(metrics, Mapping) else None,
        }
        allocation = metrics.get("allocation") if isinstance(metrics, Mapping) else None

        summary = {
            "pair": pair,
            "direction": direction,
            "action": action,
            "confidence": confidence,
            "expected_holding_period": metrics.get("avg_hold_days") if isinstance(metrics, Mapping) else None,
            "entry_price_l": entry_prices["entry_price_l"],
            "entry_price_r": entry_prices["entry_price_r"],
            "target_ratio": target_ratio,
            "allocation": allocation,
            "risk_metrics": risk_metrics,
            "timestamp": timestamp,
            "metrics": {
                "z_score": z_score,
                "regime": regime,
                "overlay": overlay,
            },
            "run_id": run_id,
        }
        results.append(summary)

    return results


def _write_outputs(results: List[Dict[str, Any]], output_dir: pathlib.Path = OUTPUT_DIR) -> Tuple[pathlib.Path, pathlib.Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    date_tag = dt.datetime.utcnow().strftime("%Y%m%d")
    json_path = output_dir / f"daily_signal_{date_tag}.json"
    csv_path = output_dir / f"daily_signal_{date_tag}.csv"

    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False, default=str)

    rows = []
    for item in results:
        metrics = item.get("metrics", {})
        overlay = metrics.get("overlay", {}) if isinstance(metrics, Mapping) else {}
        risk = item.get("risk_metrics", {}) if isinstance(item.get("risk_metrics"), Mapping) else {}
        rows.append(
            {
                "pair": item.get("pair"),
                "direction": item.get("direction"),
                "action": item.get("action"),
                "confidence": item.get("confidence"),
                "expected_holding_period": item.get("expected_holding_period"),
                "entry_price_l": item.get("entry_price_l"),
                "entry_price_r": item.get("entry_price_r"),
                "target_ratio": item.get("target_ratio"),
                "timestamp": item.get("timestamp"),
                "z_score": metrics.get("z_score") if isinstance(metrics, Mapping) else None,
                "regime": metrics.get("regime") if isinstance(metrics, Mapping) else None,
                "potential_drawdown": risk.get("potential_drawdown"),
                "capital_at_risk": risk.get("capital_at_risk"),
                "delta_score": overlay.get("delta_score"),
                "delta_mom12": overlay.get("delta_mom12"),
                "run_id": item.get("run_id"),
            }
        )
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    return json_path, csv_path


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    results = run_daily_signals(CONFIG_PATH)
    json_path, csv_path = _write_outputs(results, OUTPUT_DIR)
    LOGGER.info("Daily signals written to %s and %s", json_path, csv_path)


if __name__ == "__main__":
    main()
