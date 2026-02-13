"""Simple read-only dashboard for VECM outputs."""
from __future__ import annotations

import dataclasses
import datetime as dt
import importlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, render_template, request

from vecm_project.scripts import playbook_vecm

try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
except ImportError:  # pragma: no cover - optional dependency fallback
    Limiter = None
    get_remote_address = None

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
app.config.setdefault(
    "RATELIMIT_STORAGE_URI",
    os.getenv("RATELIMIT_STORAGE_URI", "memory://"),
)
_PROM_METRICS = None
_WHATIF_FEATURE_CACHE: Dict[str, playbook_vecm.FeatureBuildResult] = {}
if Limiter is not None and get_remote_address is not None:
    _LIMITER = Limiter(
        key_func=get_remote_address,
        default_limits=["200 per day", "50 per hour"],
    )
    _LIMITER.init_app(app)
else:
    _LIMITER = None


def _with_rate_limit(limit: str) -> Callable[[Callable[..., Response]], Callable[..., Response]]:
    def _decorator(fn: Callable[..., Response]) -> Callable[..., Response]:
        if _LIMITER is None:
            return fn
        return _LIMITER.limit(limit)(fn)

    return _decorator


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
class ExecutionChart:
    pair: str
    ticker: str
    dates: List[str]
    prices: List[float]
    buy_points: List[Dict[str, object]]
    sell_points: List[Dict[str, object]]


@dataclass
class DashboardData:
    run_id: Optional[str]
    latest_signal: Optional[LatestSignal]
    metrics: Optional[MetricSummary]
    pair_charts: List[PairChart]
    execution_charts: List[ExecutionChart]


@dataclass
class BacktestDefaults:
    z_entry: Optional[float]
    z_exit: Optional[float]
    max_hold: Optional[int]
    cooldown: Optional[int]
    p_th: Optional[float]
    pair: Optional[str]


@dataclass
class BeginnerMetricExplain:
    name: str
    value: str
    meaning: str
    status: str
    implication: str


@dataclass
class BeginnerExplain:
    headline: str
    signal_summary: str
    one_year_projection: str
    caveat: str
    metric_items: List[BeginnerMetricExplain]


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


def _normalise_pair(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        tokens = [str(token).strip() for token in value if str(token).strip()]
    else:
        raw = str(value).strip()
        if not raw:
            return None
        compact = raw.replace(" ", "")
        if "~" in compact:
            tokens = [token for token in compact.split("~") if token]
        else:
            tokens = [token for token in compact.split(",") if token]
    if len(tokens) != 2:
        return None
    return f"{tokens[0]},{tokens[1]}"


def _pair_label(pair_value: str) -> str:
    lhs, rhs = pair_value.split(",", 1)
    return f"{lhs} ~ {rhs}"


def _load_whatif_pair_options(default_pair: Optional[str]) -> List[Dict[str, str]]:
    options: List[Dict[str, str]] = []
    seen: set[str] = set()

    def _add(raw_value: Any) -> None:
        pair_value = _normalise_pair(raw_value)
        if pair_value is None or pair_value in seen:
            return
        seen.add(pair_value)
        options.append({"value": pair_value, "label": _pair_label(pair_value)})

    _add(default_pair)
    if RUN_MANIFEST.exists():
        manifest = pd.read_csv(RUN_MANIFEST)
        for column in ("subset", "pair"):
            if column not in manifest.columns:
                continue
            for raw_value in manifest[column].dropna().tolist():
                _add(raw_value)
                if len(options) >= 24:
                    return options
    return options


def _latest_backtest_context() -> Tuple[Optional[str], Optional[pd.Series], Dict[str, Any]]:
    run_id, manifest_row = _latest_run()
    if not run_id:
        return None, manifest_row, {}
    params = _load_manifest_params(run_id)
    if not params:
        params = playbook_vecm.parse_args([]).to_dict()
    return run_id, manifest_row, dict(params)


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
    beginner_explain = _build_beginner_explain(data.metrics, data.latest_signal)
    whatif_pairs = _load_whatif_pair_options(defaults.pair)
    return render_template(
        "index.html",
        dashboard=data,
        backtest_defaults=defaults,
        beginner_explain=beginner_explain,
        backtest_defaults_json=json.dumps(dataclasses.asdict(defaults)),
        whatif_pairs=whatif_pairs,
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
        execution_charts_json=json.dumps(
            [
                {
                    "pair": chart.pair,
                    "ticker": chart.ticker,
                    "dates": chart.dates,
                    "prices": chart.prices,
                    "buy_points": chart.buy_points,
                    "sell_points": chart.sell_points,
                }
                for chart in data.execution_charts
            ]
        ),
    )


def _json_error(error: str, status: int, **extra: Any) -> Response:
    payload: Dict[str, Any] = {"error": error}
    payload.update(extra)
    response = jsonify(payload)
    response.status_code = status
    return response


def _coerce_bounded_float(name: str, value: Any, low: float, high: float) -> float:
    parsed = _as_float(value)
    if parsed is None:
        raise ValueError(f"{name} must be numeric")
    if parsed < low or parsed > high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return float(parsed)


def _coerce_bounded_int(name: str, value: Any, low: int, high: int) -> int:
    parsed = _as_int(value)
    if parsed is None:
        raise ValueError(f"{name} must be integer")
    if parsed < low or parsed > high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return int(parsed)


def _build_whatif_decision_params(
    payload: Dict[str, Any],
    base_cfg: playbook_vecm.PlaybookConfig,
) -> playbook_vecm.DecisionParams:
    z_entry_default = base_cfg.z_entry
    if z_entry_default is None or not np.isfinite(z_entry_default) or z_entry_default <= 0:
        cap = _as_float(getattr(base_cfg, "z_entry_cap", None))
        if cap is not None and np.isfinite(cap) and cap > 0:
            z_entry_default = cap
        else:
            z_entry_default = 1.5
    z_entry_default = float(min(2.5, max(0.5, z_entry_default)))
    z_exit_default = base_cfg.z_exit
    max_hold_default = base_cfg.max_hold
    cooldown_default = base_cfg.cooldown
    p_th_default = base_cfg.p_th
    return playbook_vecm.DecisionParams(
        z_entry=_coerce_bounded_float(
            "z_entry",
            payload.get("z_entry", z_entry_default),
            low=0.50,
            high=2.50,
        ),
        z_exit=_coerce_bounded_float(
            "z_exit",
            payload.get("z_exit", z_exit_default),
            low=0.20,
            high=1.50,
        ),
        max_hold=_coerce_bounded_int(
            "max_hold",
            payload.get("max_hold", max_hold_default),
            low=1,
            high=60,
        ),
        cooldown=_coerce_bounded_int(
            "cooldown",
            payload.get("cooldown", cooldown_default),
            low=0,
            high=30,
        ),
        p_th=_coerce_bounded_float(
            "p_th",
            payload.get("p_th", p_th_default),
            low=0.50,
            high=0.95,
        ),
        run_id=f"whatif_{int(time.time())}",
    )


def _whatif_feature_cache_key(run_id: str, cfg: playbook_vecm.PlaybookConfig) -> str:
    cfg_payload = cfg.to_dict()
    for key in ("z_entry", "z_exit", "z_stop", "max_hold", "cooldown", "p_th"):
        cfg_payload.pop(key, None)
    cfg_payload["run_id"] = run_id
    return json.dumps(cfg_payload, sort_keys=True, default=str)


def _load_whatif_features(
    run_id: str,
    cfg: playbook_vecm.PlaybookConfig,
) -> playbook_vecm.FeatureBuildResult:
    cache_key = _whatif_feature_cache_key(run_id, cfg)
    cached = _WHATIF_FEATURE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    data_frame = playbook_vecm.load_and_validate_data(cfg.input_file)
    feature_result = playbook_vecm.build_features(
        cfg.subset,
        playbook_vecm.FeatureConfig(
            base_config=cfg,
            pair=cfg.subset,
            method=cfg.method,
            horizon=cfg.horizon,
            data_frame=data_frame,
            run_id=f"{run_id}_whatif",
        ),
    )
    _WHATIF_FEATURE_CACHE[cache_key] = feature_result
    return feature_result


def _run_whatif(payload: Dict[str, Any]) -> Response:
    run_id, manifest_row, params = _latest_backtest_context()
    if not run_id:
        return _json_error("no_run", 404)

    default_pair = _normalise_pair(params.get("subset"))
    if default_pair is None and manifest_row is not None:
        default_pair = _normalise_pair(manifest_row.get("pair"))
    requested_pair = _normalise_pair(
        payload.get("pair") or payload.get("subset") or payload.get("universe")
    )
    pair_value = requested_pair or default_pair
    if pair_value is None:
        return _json_error(
            "invalid_pair",
            400,
            details="Pair wajib dalam format 'AAA,BBB' atau 'AAA~BBB'.",
        )

    cfg_dict = dict(params)
    cfg_dict["subset"] = pair_value
    cfg = playbook_vecm.PlaybookConfig(**cfg_dict)
    try:
        decision_params = _build_whatif_decision_params(payload, cfg)
    except ValueError as exc:
        return _json_error("invalid_payload", 400, details=str(exc))

    feature_result = _load_whatif_features(run_id, cfg)
    if feature_result.skip_result is not None:
        skip_result = feature_result.skip_result
        details = skip_result.get("status") or skip_result.get("metrics", {}).get("skip_reason")
        return _json_error("feature_skip", 400, pair=pair_value, details=details)

    result = playbook_vecm.evaluate_rules(feature_result, decision_params)
    metrics = result.get("metrics", {})
    execution = result.get("execution")
    trades_count = _as_int(metrics.get("n_trades")) or 0
    if execution is not None and hasattr(execution, "trades"):
        trades_frame = getattr(execution, "trades")
        if isinstance(trades_frame, pd.DataFrame):
            trades_count = int(trades_frame.shape[0])
    summary = {
        "pair": pair_value,
        "run_id": run_id,
        "sharpe_oos": _as_float(metrics.get("sharpe_oos")),
        "maxdd": _as_float(metrics.get("maxdd")),
        "turnover": _as_float(metrics.get("turnover")),
        "turnover_annualised": _as_float(metrics.get("turnover_annualised")),
        "cagr": _as_float(metrics.get("cagr")),
        "n_trades": int(trades_count),
        "trades": int(trades_count),
    }
    return jsonify(summary)


@app.route("/api/whatif", methods=["POST"])
@_with_rate_limit("10 per minute")
def api_whatif() -> Response:
    payload = request.get_json(silent=True) or {}
    return _run_whatif(payload)


@app.route("/api/backtest", methods=["POST"])
@_with_rate_limit("10 per minute")
def api_backtest() -> Response:
    payload = request.get_json(silent=True) or {}
    return _run_whatif(payload)


def build_dashboard() -> DashboardData:
    run_id, manifest_row = _latest_run()
    latest_signal = _load_latest_signal(run_id) if run_id else None
    metrics = _load_metrics(run_id) if run_id else None
    pair_charts = _build_pair_charts(manifest_row)
    execution_charts = _build_execution_charts(run_id, manifest_row)
    return DashboardData(
        run_id=run_id,
        latest_signal=latest_signal,
        metrics=metrics,
        pair_charts=pair_charts,
        execution_charts=execution_charts,
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
    _, manifest_row, params = _latest_backtest_context()
    if not params:
        params = playbook_vecm.parse_args([]).to_dict()
    pair_value = _normalise_pair(params.get("subset"))
    if pair_value is None and manifest_row is not None:
        pair_value = _normalise_pair(manifest_row.get("pair"))
    z_entry_default = _as_float(params.get("z_entry"))
    if z_entry_default is None and manifest_row is not None:
        # If manual z_entry is absent, use the last run's effective threshold.
        z_entry_default = _as_float(manifest_row.get("z_th"))
    if z_entry_default is None:
        z_entry_default = _as_float(params.get("z_entry_cap"))
    if z_entry_default is not None and np.isfinite(z_entry_default):
        z_entry_default = float(min(2.5, max(0.5, z_entry_default)))
    return BacktestDefaults(
        z_entry=z_entry_default,
        z_exit=_as_float(params.get("z_exit")),
        max_hold=_as_int(params.get("max_hold")),
        cooldown=_as_int(params.get("cooldown")),
        p_th=_as_float(params.get("p_th")),
        pair=pair_value,
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
    try:
        ret_df = pd.read_csv(ret_path)
        pos_df = pd.read_csv(pos_path)
    except (OSError, ValueError, pd.errors.ParserError):
        return None
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


def _trade_markers_for_pair(
    trades_df: pd.DataFrame,
    price_df: pd.DataFrame,
    lhs: str,
    rhs: str,
) -> Dict[str, Dict[str, List[Dict[str, object]]]]:
    markers: Dict[str, Dict[str, List[Dict[str, object]]]] = {
        lhs: {"buy": [], "sell": []},
        rhs: {"buy": [], "sell": []},
    }
    if trades_df is None or trades_df.empty:
        return markers
    if lhs not in price_df.columns or rhs not in price_df.columns:
        return markers

    def _add_marker(ticker: str, action: str, date_value: object, event: str) -> None:
        ts = pd.to_datetime(date_value, errors="coerce")
        if pd.isna(ts):
            return
        ts = pd.Timestamp(ts).normalize()
        if ts not in price_df.index:
            return
        price = _as_float(price_df.at[ts, ticker])
        if price is None:
            return
        markers[ticker][action].append(
            {"x": ts.strftime("%Y-%m-%d"), "y": float(price), "event": event}
        )

    for _, trade in trades_df.iterrows():
        side = str(trade.get("side", "")).upper()
        open_date = trade.get("open_date")
        close_date = trade.get("close_date")
        if side == "LONG":
            _add_marker(lhs, "buy", open_date, "OPEN_LONG")
            _add_marker(rhs, "sell", open_date, "OPEN_LONG")
            _add_marker(lhs, "sell", close_date, "CLOSE_LONG")
            _add_marker(rhs, "buy", close_date, "CLOSE_LONG")
        elif side == "SHORT":
            _add_marker(lhs, "sell", open_date, "OPEN_SHORT")
            _add_marker(rhs, "buy", open_date, "OPEN_SHORT")
            _add_marker(lhs, "buy", close_date, "CLOSE_SHORT")
            _add_marker(rhs, "sell", close_date, "CLOSE_SHORT")
    return markers


def _build_execution_charts(run_id: Optional[str], manifest_row: Optional[pd.Series]) -> List[ExecutionChart]:
    if not run_id or manifest_row is None:
        return []
    pair_value = _normalise_pair(manifest_row.get("pair"))
    if pair_value is None:
        return []
    lhs, rhs = pair_value.split(",", 1)

    panel = _load_price_panel()
    if panel is None or lhs not in panel.columns or rhs not in panel.columns:
        return []

    pos_path = OUT_DIR / f"positions_{run_id}.csv"
    if not pos_path.exists():
        return []
    try:
        pos_df = pd.read_csv(pos_path)
    except (OSError, ValueError, pd.errors.ParserError):
        return []
    if pos_df.empty:
        return []
    date_col = pos_df.columns[0]
    pos_df["date"] = pd.to_datetime(pos_df[date_col], errors="coerce")
    pos_df = pos_df.dropna(subset=["date"]).sort_values("date")
    if pos_df.empty:
        return []

    idx = pd.DatetimeIndex(pos_df["date"]).normalize().unique()
    subset = panel[[lhs, rhs]].copy()
    subset.index = pd.DatetimeIndex(subset.index).normalize()
    subset = subset.loc[subset.index.intersection(idx)].sort_index()
    if subset.empty:
        return []

    trades_path = OUT_DIR / f"trades_{run_id}.csv"
    if trades_path.exists():
        try:
            trades_df = pd.read_csv(trades_path)
        except (OSError, ValueError, pd.errors.ParserError):
            trades_df = pd.DataFrame()
    else:
        trades_df = pd.DataFrame()
    markers = _trade_markers_for_pair(trades_df, subset, lhs, rhs)

    charts: List[ExecutionChart] = []
    for ticker in (lhs, rhs):
        charts.append(
            ExecutionChart(
                pair=f"{lhs} ~ {rhs}",
                ticker=ticker,
                dates=[d.strftime("%Y-%m-%d") for d in subset.index],
                prices=[float(v) for v in subset[ticker].tolist()],
                buy_points=markers[ticker]["buy"],
                sell_points=markers[ticker]["sell"],
            )
        )
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


def _fmt_float(value: Optional[float], digits: int = 4) -> str:
    if value is None or not np.isfinite(value):
        return "N/A"
    return f"{value:.{digits}f}"


def _fmt_percent(value: Optional[float], digits: int = 2) -> str:
    if value is None or not np.isfinite(value):
        return "N/A"
    return f"{value * 100:.{digits}f}%"


def _describe_signal(position: Optional[float]) -> str:
    if position is None or not np.isfinite(position):
        return "Posisi belum tersedia."
    if position > 0:
        return "Sinyal saat ini: LONG spread (long saham kiri, short saham kanan)."
    if position < 0:
        return "Sinyal saat ini: SHORT spread (short saham kiri, long saham kanan)."
    return "Sinyal saat ini: FLAT/HOLD (tidak ada posisi baru)."


def _metric_item_sharpe(sharpe: Optional[float]) -> BeginnerMetricExplain:
    value = _fmt_float(sharpe, 3)
    meaning = "Seberapa besar imbal hasil dibanding risiko volatilitas (lebih tinggi biasanya lebih baik)."
    if sharpe is None or not np.isfinite(sharpe):
        status = "Belum ada data"
        implication = "Belum bisa menilai konsistensi return strategi."
    elif sharpe < 0:
        status = "Kurang baik"
        implication = "Historisnya return belum sepadan dengan risiko. Perlu hati-hati."
    elif sharpe < 1:
        status = "Cukup"
        implication = "Ada potensi, tapi margin keunggulannya masih tipis."
    elif sharpe < 2:
        status = "Baik"
        implication = "Risk-adjusted return relatif sehat pada data historis."
    else:
        status = "Sangat baik"
        implication = "Risk-adjusted return sangat kuat pada data historis."
    return BeginnerMetricExplain(
        name="Sharpe Ratio (OOS)",
        value=value,
        meaning=meaning,
        status=status,
        implication=implication,
    )


def _metric_item_drawdown(maxdd: Optional[float]) -> BeginnerMetricExplain:
    drawdown_pct = abs(maxdd) if maxdd is not None and np.isfinite(maxdd) else None
    value = _fmt_percent(drawdown_pct, 2)
    meaning = "Penurunan terbesar dari puncak ke lembah nilai portofolio selama periode uji."
    if drawdown_pct is None:
        status = "Belum ada data"
        implication = "Belum bisa memperkirakan seberapa dalam kerugian sementara."
    elif drawdown_pct <= 0.05:
        status = "Rendah"
        implication = "Fluktuasi turun historis relatif kecil."
    elif drawdown_pct <= 0.15:
        status = "Menengah"
        implication = "Masih wajar untuk strategi aktif, tetapi tetap perlu batas risiko."
    else:
        status = "Tinggi"
        implication = "Potensi penurunan historis cukup dalam."
    return BeginnerMetricExplain(
        name="Max Drawdown",
        value=value,
        meaning=meaning,
        status=status,
        implication=implication,
    )


def _metric_item_trades(n_trades: Optional[int]) -> BeginnerMetricExplain:
    value = "N/A" if n_trades is None else str(n_trades)
    meaning = "Jumlah transaksi pada periode uji (semakin banyak, strategi makin aktif)."
    if n_trades is None:
        status = "Belum ada data"
        implication = "Belum terlihat gaya trading strateginya."
    elif n_trades < 5:
        status = "Sangat sedikit"
        implication = "Sinyal jarang muncul; hasil bisa kurang stabil."
    elif n_trades < 15:
        status = "Sedang"
        implication = "Frekuensi transaksi cukup untuk evaluasi awal."
    else:
        status = "Aktif"
        implication = "Transaksi sering; perlu perhatikan biaya dan disiplin eksekusi."
    return BeginnerMetricExplain(
        name="Jumlah Trade",
        value=value,
        meaning=meaning,
        status=status,
        implication=implication,
    )


def _metric_item_turnover(turnover_ann: Optional[float]) -> BeginnerMetricExplain:
    value = _fmt_float(turnover_ann, 2)
    meaning = "Perkiraan frekuensi pergantian posisi dalam setahun."
    if turnover_ann is None or not np.isfinite(turnover_ann):
        status = "Belum ada data"
        implication = "Belum bisa mengukur intensitas aktivitas strategi."
    elif turnover_ann < 5:
        status = "Rendah"
        implication = "Aktivitas relatif santai, biaya transaksi cenderung lebih ringan."
    elif turnover_ann < 12:
        status = "Menengah"
        implication = "Aktif secukupnya, biaya transaksi tetap perlu dipantau."
    else:
        status = "Tinggi"
        implication = "Sangat aktif; biaya/slippage bisa lebih berpengaruh."
    return BeginnerMetricExplain(
        name="Turnover Annualised",
        value=value,
        meaning=meaning,
        status=status,
        implication=implication,
    )


def _metric_item_cagr(cagr: Optional[float]) -> BeginnerMetricExplain:
    value = _fmt_percent(cagr, 2)
    meaning = "Perkiraan pertumbuhan tahunan majemuk dari hasil backtest."
    if cagr is None or not np.isfinite(cagr):
        status = "Belum ada data"
        implication = "Belum bisa memperkirakan laju pertumbuhan tahunan."
    elif cagr < 0:
        status = "Negatif"
        implication = "Backtest menunjukkan modal cenderung turun."
    elif cagr < 0.1:
        status = "Rendah"
        implication = "Ada pertumbuhan, tetapi belum agresif."
    else:
        status = "Positif"
        implication = "Backtest menunjukkan pertumbuhan tahunan yang sehat."
    return BeginnerMetricExplain(
        name="CAGR",
        value=value,
        meaning=meaning,
        status=status,
        implication=implication,
    )


def _build_beginner_explain(
    metrics: Optional[MetricSummary],
    latest_signal: Optional[LatestSignal],
) -> BeginnerExplain:
    sharpe = metrics.sharpe_oos if metrics else None
    maxdd = metrics.maxdd if metrics else None
    n_trades = metrics.n_trades if metrics else None
    turnover_ann = metrics.turnover_annualised if metrics else None
    cagr = metrics.cagr if metrics else None

    signal_summary = _describe_signal(latest_signal.position if latest_signal else None)

    if sharpe is None or not np.isfinite(sharpe):
        headline = "Data belum cukup untuk menilai kualitas strategi."
    elif sharpe < 0:
        headline = "Secara historis, strategi ini masih cenderung merugi relatif terhadap risikonya."
    elif sharpe < 1:
        headline = "Strategi berada di zona netral: ada potensi, tetapi belum kuat."
    else:
        headline = "Secara historis, strategi menunjukkan kualitas risk-return yang cukup baik."

    modal_awal = 10_000_000.0
    if cagr is None or not np.isfinite(cagr):
        projection = "Simulasi 1 tahun belum tersedia karena nilai CAGR tidak ada."
    else:
        modal_akhir = modal_awal * (1.0 + cagr)
        modal_akhir = max(modal_akhir, 0.0)
        projection = (
            f"Simulasi sederhana: jika modal awal Rp10.000.000 dan CAGR historis "
            f"{_fmt_percent(cagr, 2)}, estimasi menjadi sekitar Rp{modal_akhir:,.0f} "
            f"dalam 1 tahun."
        )

    caveat = (
        "Catatan penting: ini hasil backtest historis, bukan jaminan hasil masa depan "
        "dan bukan nasihat keuangan."
    )

    metric_items = [
        _metric_item_sharpe(sharpe),
        _metric_item_drawdown(maxdd),
        _metric_item_trades(n_trades),
        _metric_item_turnover(turnover_ann),
        _metric_item_cagr(cagr),
    ]

    return BeginnerExplain(
        headline=headline,
        signal_summary=signal_summary,
        one_year_projection=projection,
        caveat=caveat,
        metric_items=metric_items,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=False)
