"""Simple read-only dashboard for VECM outputs."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from flask import Flask, Response, render_template, request

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


@app.before_request
def _basic_auth_guard() -> Optional[Response]:
    if request.endpoint in {"healthz"}:
        return None
    return _require_basic_auth()


@app.route("/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@app.route("/")
def index() -> str:
    data = build_dashboard()
    return render_template(
        "index.html",
        dashboard=data,
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
