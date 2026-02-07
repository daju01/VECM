"""Simple read-only dashboard for VECM outputs."""
from __future__ import annotations

import dataclasses
import datetime as dt
import importlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, render_template, request

from vecm_project.scripts import playbook_vecm

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "out_ms"
RUN_MANIFEST = OUT_DIR / "run_manifest.csv"


def _read_env_credentials() -> Tuple[Optional[str], Optional[str]]:
    return os.getenv("DASHBOARD_USER"), os.getenv("DASHBOARD_PASS")


def _require_basic_auth() -> Optional[Response]:
    user, password = _read_env_credentials()
    if not user or not password:
        return Response(
            "Dashboard auth not configured. Set DASHBOARD_USER and DASHBOARD_PASS.",
            status=500,
        )
    auth = request.authorization
    if auth is None or auth.username != user or auth.password != password:
        return Response(
            "Unauthorized",
            401,
            {"WWW-Authenticate": 'Basic realm="VECM Dashboard"'},
        )
    return None


app = Flask(__name__)
_PROM_METRICS = None


def _ensure_prom_metrics() -> Optional[Dict[str, Any]]:
    global _PROM_METRICS
    if importlib.util.find_spec("prometheus_client") is None:
        return None
    prometheus_client = importlib.import_module("prometheus_client")
    gauge = getattr(prometheus_client, "Gauge", None)
    if gauge is None:
        return None
    if _PROM_METRICS is None:
        _PROM_METRICS = {
            "price_age": gauge("vecm_price_data_age_hours", "Age of price data in hours"),
            "signal_age": gauge("vecm_daily_signal_age_hours", "Age of daily signal in hours"),
        }
    return _PROM_METRICS


@dataclass
class LatestSignal:
    date: str
    position: Optional[float]
    ret: Optional[float]
    cost: Optional[float]
    p_regime: Optional[float]
    delta_score: Optional[float]
    delta_mom12: Optional[float]


@dataclass
class MetricSummary:
    sharpe_oos: Optional[float]
    maxdd: Optional[float]
    turnover: Optional[float]
    turnover_annualised: Optional[float]
    n_trades: Optional[int]
    cagr: Optional[float]


@dataclass
class PairChart:
    pair: str
    dates: List[str]
    spread: List[float]
    zscore: List[float]


@dataclass
class DashboardData:
    run_id: Optional[str]
    latest_signal: Optional[LatestSignal]
    metrics: Optional[MetricSummary]
    pair_charts: List[PairChart]


@dataclass
class BacktestDefaults:
    z_entry: Optional[float]
    z_exit: Optional[float]
    max_hold: Optional[int]
    cooldown: Optional[int]
    p_th: Optional[float]


@app.before_request
def _basic_auth_guard() -> Optional[Response]:
    if request.endpoint in {"healthz"}:
        return None
    return _require_basic_auth()


def _file_age_hours(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    mtime = dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.timezone.utc)
    return (dt.datetime.now(dt.timezone.utc) - mtime).total_seconds() / 3600


def _latest_daily_signal_age() -> Optional[float]:
    output_dir = Path(__file__).resolve().parents[1] / "outputs" / "daily"
    candidates = sorted(output_dir.glob("daily_signal_*.json"))
    if not candidates:
        return None
    return _file_age_hours(candidates[-1])


def _load_strategy_profiles() -> Dict[str, Any]:
    profile_path = BASE_DIR / "config" / "strategy_profiles.json"
    if not profile_path.exists():
        return {}
    return json.loads(profile_path.read_text(encoding="utf-8"))


def _load_ticker_groups() -> Dict[str, List[str]]:
    path = BASE_DIR / "config" / "ticker_groups.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    return {key: value for key, value in payload.items() if isinstance(value, list)}


@app.route("/healthz")
def healthz() -> Dict[str, object]:
    base_dir = Path(__file__).resolve().parents[1]
    data_path = base_dir / "data" / "adj_close_data.csv"
    price_age_hours = _file_age_hours(data_path)
    signal_age_hours = _latest_daily_signal_age()
    status = "ok"
    if signal_age_hours is None or signal_age_hours > 48:
        status = "stale"
    return {
        "status": status,
        "price_data_age_hours": price_age_hours,
        "daily_signal_age_hours": signal_age_hours,
    }


@app.route("/metrics")
def metrics() -> Response:
    prom = _ensure_prom_metrics()
    if prom is None:
        return Response("prometheus_client not installed", status=501)
    prometheus_client = importlib.import_module("prometheus_client")
    generate_latest = getattr(prometheus_client, "generate_latest", None)
    content_type = getattr(prometheus_client, "CONTENT_TYPE_LATEST", None)
    if generate_latest is None or content_type is None:
        return Response("prometheus_client not installed", status=501)
    base_dir = Path(__file__).resolve().parents[1]
    data_path = base_dir / "data" / "adj_close_data.csv"
    price_age_hours = _file_age_hours(data_path)
    signal_age_hours = _latest_daily_signal_age()
    if price_age_hours is not None:
        prom["price_age"].set(price_age_hours)
    if signal_age_hours is not None:
        prom["signal_age"].set(signal_age_hours)
    return Response(generate_latest(), mimetype=content_type)


@app.route("/config", methods=["GET", "POST"])
def config_wizard() -> str:
    profiles = _load_strategy_profiles()
    ticker_groups = _load_ticker_groups()
    selected_profile = request.form.get("profile", "beginner")
    selected_pair = request.form.get("pair", "")
    preview: Optional[Dict[str, Any]] = None
    if request.method == "POST":
        profile_data = profiles.get(selected_profile, {})
        preview = {
            "profile": selected_profile,
            "pair": selected_pair,
            "params": profile_data.get("params", {}),
        }
    return render_template(
        "config_wizard.html",
        profiles=profiles,
        ticker_groups=ticker_groups,
        selected_profile=selected_profile,
        selected_pair=selected_pair,
        preview=preview,
    )


@app.route("/")
def index() -> str:
    data = build_dashboard()
    defaults = _load_backtest_defaults()
    return render_template(
        "index.html",
        dashboard=data,
        backtest_defaults=defaults,
        backtest_defaults_json=json.dumps(dataclasses.asdict(defaults)),
        charts_json=json.dumps(
            [
                {
                    "pair": chart.pair,
                    "dates": chart.dates,
                    "spread": chart.spread,
                    "zscore": chart.zscore,
                }
                for chart in data.pair_charts
            ]
        ),
    )


@app.route("/api/backtest", methods=["POST"])
def api_backtest() -> Response:
    run_id, _ = _latest_run()
    if not run_id:
        return jsonify({"error": "no_run"}), 404
    params = _load_manifest_params(run_id)
    if not params:
        params = playbook_vecm.parse_args([]).to_dict()
    payload = request.get_json(silent=True) or {}

    cfg_dict = dict(params)
    for key in ["z_entry", "z_exit", "max_hold", "cooldown", "p_th"]:
        if key in payload and payload[key] is not None:
            if key in {"max_hold", "cooldown"}:
                coerced = _as_int(payload[key])
            else:
                coerced = _as_float(payload[key])
            if coerced is not None:
                cfg_dict[key] = coerced
    cfg = playbook_vecm.PlaybookConfig(**cfg_dict)

    data_frame = playbook_vecm.load_and_validate_data(cfg.input_file)
    feature_result = playbook_vecm.build_features(
        cfg.subset,
        playbook_vecm.FeatureConfig(
            base_config=cfg,
            pair=cfg.subset,
            method=cfg.method,
            horizon=cfg.horizon,
            data_frame=data_frame,
            run_id=run_id,
        ),
    )
    if feature_result.skip_result is not None:
        return jsonify({"error": "feature_skip", "details": feature_result.skip_result}), 400
    decision_params = playbook_vecm.DecisionParams(
        z_entry=cfg.z_entry,
        z_exit=cfg.z_exit,
        max_hold=cfg.max_hold,
        cooldown=cfg.cooldown,
        p_th=cfg.p_th,
        run_id=run_id,
    )
    result = playbook_vecm.evaluate_rules(feature_result, decision_params)
    metrics = result.get("metrics", {})
    summary = {
        "sharpe_oos": metrics.get("sharpe_oos"),
        "maxdd": metrics.get("maxdd"),
        "turnover": metrics.get("turnover"),
        "turnover_annualised": metrics.get("turnover_annualised"),
        "n_trades": metrics.get("n_trades"),
        "cagr": metrics.get("cagr"),
    }
    return jsonify(summary)


def build_dashboard() -> DashboardData:
    run_id, manifest_row = _latest_run()
    latest_signal = _load_latest_signal(run_id) if run_id else None
    metrics = _load_metrics(run_id) if run_id else None
    pair_charts = _build_pair_charts(manifest_row)
    return DashboardData(
        run_id=run_id,
        latest_signal=latest_signal,
        metrics=metrics,
        pair_charts=pair_charts,
    )


def _latest_run() -> Tuple[Optional[str], Optional[pd.Series]]:
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
    except (json.JSONDecodeError, OSError):
        return {}
    params = payload.get("params", {})
    return params if isinstance(params, dict) else {}


def _load_backtest_defaults() -> BacktestDefaults:
    run_id, _ = _latest_run()
    params = _load_manifest_params(run_id) if run_id else {}
    if not params:
        params = playbook_vecm.parse_args([]).to_dict()
    return BacktestDefaults(
        z_entry=_as_float(params.get("z_entry")),
        z_exit=_as_float(params.get("z_exit")),
        max_hold=_as_int(params.get("max_hold")),
        cooldown=_as_int(params.get("cooldown")),
        p_th=_as_float(params.get("p_th")),
    )


def _load_metrics(run_id: str) -> Optional[MetricSummary]:
    path = OUT_DIR / f"metrics_{run_id}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    row = df.iloc[0]
    return MetricSummary(
        sharpe_oos=_as_float(row.get("sharpe_oos")),
        maxdd=_as_float(row.get("maxdd")),
        turnover=_as_float(row.get("turnover")),
        turnover_annualised=_as_float(row.get("turnover_annualised")),
        n_trades=_as_int(row.get("n_trades")),
        cagr=_as_float(row.get("cagr")),
    )


def _load_latest_signal(run_id: str) -> Optional[LatestSignal]:
    ret_path = OUT_DIR / f"returns_{run_id}.csv"
    pos_path = OUT_DIR / f"positions_{run_id}.csv"
    if not ret_path.exists() or not pos_path.exists():
        return None
    ret_df = pd.read_csv(ret_path)
    pos_df = pd.read_csv(pos_path)
    if ret_df.empty or pos_df.empty:
        return None

    ret_df["date"] = pd.to_datetime(ret_df.iloc[:, 0], errors="coerce")
    pos_df["date"] = pd.to_datetime(pos_df.iloc[:, 0], errors="coerce")
    latest_ret = ret_df.sort_values("date").iloc[-1]
    latest_pos = pos_df.sort_values("date").iloc[-1]
    return LatestSignal(
        date=str(latest_ret["date"].date()),
        position=_as_float(latest_pos.get("pos")),
        ret=_as_float(latest_ret.get("ret")),
        cost=_as_float(latest_ret.get("cost")),
        p_regime=_as_float(latest_ret.get("p_regime")),
        delta_score=_as_float(latest_ret.get("delta_score")),
        delta_mom12=_as_float(latest_ret.get("delta_mom12")),
    )


def _build_pair_charts(manifest_row: Optional[pd.Series]) -> List[PairChart]:
    pairs: List[Tuple[str, str]] = []
    if manifest_row is not None and "pair" in manifest_row:
        pair_value = str(manifest_row.get("pair"))
        if "~" in pair_value:
            lhs, rhs = pair_value.split("~", 1)
            pairs.append((lhs.strip(), rhs.strip()))

    if RUN_MANIFEST.exists() and len(pairs) < 3:
        manifest = pd.read_csv(RUN_MANIFEST)
        if not manifest.empty and "pair" in manifest.columns:
            if "Sharpe" in manifest.columns:
                manifest = manifest.sort_values("Sharpe", ascending=False)
            for value in manifest["pair"].dropna().tolist():
                if len(pairs) >= 3:
                    break
                if "~" in str(value):
                    lhs, rhs = str(value).split("~", 1)
                    candidate = (lhs.strip(), rhs.strip())
                    if candidate not in pairs:
                        pairs.append(candidate)

    if not pairs:
        return []

    panel = _load_price_panel()
    if panel is None:
        return []

    charts = []
    for lhs, rhs in pairs:
        chart = _compute_spread_zscore(panel, lhs, rhs)
        if chart is not None:
            charts.append(chart)
    return charts


def _load_price_panel() -> Optional[pd.DataFrame]:
    path = DATA_DIR / "adj_close_data.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    date_col = df.columns[0]
    df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.set_index("date")


def _compute_spread_zscore(
    panel: pd.DataFrame, lhs: str, rhs: str, window: int = 160
) -> Optional[PairChart]:
    if lhs not in panel.columns or rhs not in panel.columns:
        return None
    subset = panel[[lhs, rhs]].dropna().tail(window)
    if subset.empty:
        return None

    log_prices = np.log(subset)
    x = log_prices[lhs].values
    y = log_prices[rhs].values
    if len(x) < 5:
        return None
    beta, intercept = np.polyfit(x, y, 1)
    spread = y - (beta * x + intercept)
    spread_std = np.std(spread)
    if spread_std == 0:
        return None
    zscore = (spread - np.mean(spread)) / spread_std

    return PairChart(
        pair=f"{lhs} ~ {rhs}",
        dates=[d.strftime("%Y-%m-%d") for d in subset.index],
        spread=[float(v) for v in spread],
        zscore=[float(v) for v in zscore],
    )


def _as_float(value: Any) -> Optional[float]:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> Optional[int]:
    try:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
