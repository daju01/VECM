"""Reimplementation of the `vecm_master_exports.R` workflow in Python.

The exporter stitches together the daily position/return series emitted by the
Python playbook, enriches them with price context, infers the effective beta per
trade, and produces the ``master_*`` CSV artefacts expected by downstream R
utilities.  The output mirrors the structure and calculations of the reference
R script so that hooks and dashboards can remain unchanged.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("vecm_master_exports")
EPS = 1e-12
DEFAULT_RUN_DIR = Path("out_ms")
DEFAULT_MANIFEST = DEFAULT_RUN_DIR / "run_manifest.csv"
DEFAULT_PRICES = Path("out_pre/daily_strategic_resource_plus.csv")
DEFAULT_CAPITAL = 100_000_000.0
DEFAULT_LOT_SIZE = 100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def configure_logging() -> None:
    if logging.getLogger().handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def sanitize(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    return value.replace("\r", "").replace("\n", "").strip()


def safe_read_csv(path: Path, what: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {what}: '{path}'")
    try:
        return pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - defensive branch
        raise RuntimeError(f"Failed to read {what} '{path}': {exc}") from exc


def path_in_run_dir(run_dir: Path, stem: str) -> Path:
    candidates = [run_dir / f"{stem}.csv", run_dir / stem]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    pattern = f"{stem}".replace("*", "")
    for path in run_dir.rglob("*.csv"):
        if path.name.startswith(pattern):
            return path
    raise FileNotFoundError(f"{stem} not found in {run_dir}")


def detect_date_column(df: pd.DataFrame) -> str:
    lower_cols = {col.lower(): col for col in df.columns}
    for candidate in ("date", "dates", "timestamp", "time"):
        if candidate in lower_cols:
            return lower_cols[candidate]
    return df.columns[0]


def match_price_col(columns: Iterable[str], ticker: str) -> Optional[str]:
    ticker = ticker.upper().strip()
    for col in columns:
        upper = col.upper()
        if upper == ticker:
            return col
        if upper == f"{ticker}.JK":
            return col
    for col in columns:
        upper = col.upper()
        if upper.startswith(f"{ticker}."):
            return col
    for col in columns:
        upper = col.upper()
        if ticker in upper.replace("_", "."):
            return col
    return None


def coerce_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.date


def load_manifest(manifest_path: Path, run_id: str) -> pd.Series:
    manifest = safe_read_csv(manifest_path, "manifest").rename(columns=str.lower)
    row = manifest.loc[manifest["run_id"] == run_id]
    if row.empty:
        raise ValueError(f"run_id '{run_id}' not found in manifest '{manifest_path}'")
    return row.iloc[0]


def resolve_prices_path(manifest_row: pd.Series, explicit: Optional[Path]) -> Path:
    if explicit is not None:
        return explicit
    for key in ("input", "input_file"):
        if key in manifest_row and isinstance(manifest_row[key], str) and manifest_row[key]:
            candidate = Path(manifest_row[key])
            if candidate.exists():
                return candidate
    return DEFAULT_PRICES


def parse_pair(manifest_row: pd.Series) -> Tuple[str, str]:
    for key in ("pair", "subset"):
        value = manifest_row.get(key)
        if isinstance(value, str) and value:
            tokens = [token.strip() for token in value.replace("~", ",").split(",") if token.strip()]
            if len(tokens) >= 2:
                return tokens[0], tokens[1]
    raise ValueError("Could not parse two tickers from manifest row")


def infer_beta(daily: pd.DataFrame) -> pd.Series:
    num = daily["r_lhs"] - daily["ret_core"]
    den = daily["ret_core"] + daily["r_rhs"]
    mask = (daily["pos"] == 1) & np.isfinite(num) & np.isfinite(den) & (np.abs(den) > EPS)
    beta = np.full(len(daily), np.nan, dtype=float)
    beta[mask] = num[mask] / den[mask]
    beta[beta < 0] = np.nan
    return pd.Series(beta, index=daily.index, name="beta_inferred")


def weights_from_beta(beta: float, beta_static: float) -> Dict[str, float]:
    if not np.isfinite(beta) or beta <= 0:
        beta = beta_static
    if not np.isfinite(beta) or beta <= 0:
        return {"beta": float("nan"), "w_lhs": 0.0, "w_rhs": 0.0}
    gross = 1.0 + abs(beta)
    return {"beta": float(beta), "w_lhs": 1.0 / gross, "w_rhs": abs(beta) / gross}


def write_csv(df: pd.DataFrame, path: Path, label: str) -> None:
    LOGGER.info("Writing %s -> %s", label, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    if not path.exists():  # pragma: no cover - defensive
        raise RuntimeError(f"Failed to write {label} to '{path}'")


@dataclass
class ExportConfig:
    run_id: str
    run_dir: Path
    out_dir: Path
    manifest: Path
    prices: Optional[Path]
    capital: float
    lot_size: int


# ---------------------------------------------------------------------------
# Core workflow
# ---------------------------------------------------------------------------

def build_daily_panel(config: ExportConfig, manifest_row: pd.Series, lhs: str, rhs: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    returns_path = path_in_run_dir(config.run_dir, f"returns_{config.run_id}")
    positions_path = path_in_run_dir(config.run_dir, f"positions_{config.run_id}")
    try:
        trades_path = path_in_run_dir(config.run_dir, f"trades_{config.run_id}")
        trades_df = safe_read_csv(trades_path, "trades").rename(columns=str.lower)
    except FileNotFoundError:
        trades_df = pd.DataFrame()

    returns_df = safe_read_csv(returns_path, "returns").rename(columns=str.lower)
    positions_df = safe_read_csv(positions_path, "positions").rename(columns=str.lower)

    returns_df["date"] = coerce_date(returns_df.iloc[:, 0])
    positions_df["date"] = coerce_date(positions_df.iloc[:, 0])
    returns_df = returns_df.rename(columns={"ret": "ret_net"})
    if "cost" not in returns_df:
        returns_df["cost"] = 0.0
    positions_df = positions_df.rename(columns={"pos": "pos"})

    prices_df = safe_read_csv(config.prices, "prices raw")
    date_col = detect_date_column(prices_df)
    prices_df = prices_df.rename(columns={date_col: "date"})
    prices_df["date"] = coerce_date(prices_df["date"])
    lhs_col = match_price_col(prices_df.columns, lhs)
    rhs_col = match_price_col(prices_df.columns, rhs)
    if lhs_col is None or rhs_col is None:
        sample = ", ".join(list(prices_df.columns[:10]))
        raise ValueError(f"Price columns not found for {lhs}/{rhs}. Sample columns: {sample}")

    px = prices_df[["date", lhs_col, rhs_col]].copy()
    px.columns = ["date", "price_lhs", "price_rhs"]
    px = px.sort_values("date")
    px["r_lhs"] = np.log(px["price_lhs"]).diff()
    px["r_rhs"] = np.log(px["price_rhs"]).diff()

    ret_cols = ["date", "ret_net", "cost"]
    if "p_regime" in returns_df.columns:
        ret_cols.append("p_regime")
    if "delta_score" in returns_df.columns:
        ret_cols.append("delta_score")
    if "delta_mom12" in returns_df.columns:
        ret_cols.append("delta_mom12")

    daily = (
        pd.merge(returns_df[ret_cols], positions_df[["date", "pos"]], on="date", how="outer")
        .merge(px, on="date", how="left")
        .sort_values("date")
        .reset_index(drop=True)
    )
    daily["pos"] = daily["pos"].fillna(0.0)
    daily["cost"] = daily["cost"].fillna(0.0)
    daily["ret_net"] = daily["ret_net"].fillna(0.0)
    daily["r_lhs"] = daily["r_lhs"].fillna(0.0)
    daily["r_rhs"] = daily["r_rhs"].fillna(0.0)
    daily["ret_core"] = daily["ret_net"] + daily["cost"]

    daily["beta_inferred"] = infer_beta(daily)
    na_beta = int(((daily["pos"] == 1) & daily["beta_inferred"].isna()).sum())
    if na_beta:
        LOGGER.warning("beta_inferred NA on %d in-position days", na_beta)

    beta = daily["beta_inferred"]
    weights = 1.0 + beta
    with np.errstate(divide="ignore", invalid="ignore"):
        w_lhs = 1.0 / weights
        w_rhs = beta / weights
    daily["contrib_lhs"] = w_lhs * daily["r_lhs"]
    daily["contrib_rhs"] = -w_rhs * daily["r_rhs"]
    daily["contrib_sum"] = daily["contrib_lhs"] + daily["contrib_rhs"]
    daily["diff_check"] = daily["ret_core"] - daily["contrib_sum"]
    daily["event"] = np.select(
        [daily["cost"].abs() > EPS, daily["pos"] == 1],
        ["ENTRY/EXIT", "HOLD"],
        default="FLAT",
    )

    if not trades_df.empty:
        trades_df["trade_id"] = np.arange(1, len(trades_df) + 1)
        trades_df["open_date"] = pd.to_datetime(trades_df.get("open_date"), errors="coerce").dt.date
        trades_df["close_date"] = pd.to_datetime(trades_df.get("close_date"), errors="coerce").dt.date
        daily["trade_id"] = np.nan
        for _, trade in trades_df.iterrows():
            mask = (daily["date"] >= trade["open_date"]) & (daily["date"] <= trade["close_date"])
            daily.loc[mask, "trade_id"] = trade["trade_id"]
        LOGGER.info(
            "Trades detected: %d | days-in-position=%d | entry/exit events=%d",
            len(trades_df),
            int((daily["pos"] == 1).sum()),
            int((daily["cost"].abs() > EPS).sum()),
        )
    else:
        daily["trade_id"] = np.nan
        LOGGER.warning("No trades rows; trade_id left NA (OK if strategy flat)")

    return daily, px, trades_df


def build_orders(trades: pd.DataFrame, px: pd.DataFrame, daily: pd.DataFrame, config: ExportConfig) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()

    px = px.set_index("date")
    daily_idx = daily.set_index("date")
    log_rhs = np.log(px["price_rhs"].replace(0, np.nan)).dropna()
    log_lhs = np.log(px["price_lhs"].replace(0, np.nan)).dropna()
    if not log_rhs.empty and not log_lhs.empty:
        beta_static = float((log_rhs * log_lhs).sum() / (log_rhs ** 2).sum())
    else:
        beta_static = 1.0
    if not np.isfinite(beta_static) or beta_static <= 0:
        beta_static = 1.0

    orders: List[Dict[str, object]] = []
    for _, trade in trades.iterrows():
        trade_id = int(trade.get("trade_id", 0)) or int(trade.name) + 1
        open_date = trade.get("open_date")
        close_date = trade.get("close_date")
        side = str(trade.get("side", "LONG")).upper()
        beta_open = float(
            daily_idx.loc[open_date, "beta_inferred"]
            if open_date in daily_idx.index
            else float("nan")
        )
        weights = weights_from_beta(beta_open, beta_static)
        price_open_l = px.loc[open_date, "price_lhs"] if open_date in px.index else np.nan
        price_open_r = px.loc[open_date, "price_rhs"] if open_date in px.index else np.nan
        price_close_l = px.loc[close_date, "price_lhs"] if close_date in px.index else np.nan
        price_close_r = px.loc[close_date, "price_rhs"] if close_date in px.index else np.nan

        notional_l = config.capital * weights["w_lhs"]
        notional_r = config.capital * weights["w_rhs"]

        def calc_qty(price: float, notional: float) -> int:
            if not np.isfinite(price) or price <= 0:
                return 0
            lots = math.floor((notional / price) / config.lot_size)
            return int(lots * config.lot_size)

        qty_l = calc_qty(price_open_l, notional_l)
        qty_r = calc_qty(price_open_r, notional_r)

        if side == "SHORT":
            orders.extend(
                [
                    {
                        "trade_id": trade_id,
                        "date": open_date,
                        "ticker": "lhs",
                        "action": "SELL",
                        "price": price_open_l,
                        "qty_shares": qty_l,
                        "notional_est": qty_l * price_open_l,
                        "beta_used": weights["beta"],
                        "w_leg": weights["w_lhs"],
                    },
                    {
                        "trade_id": trade_id,
                        "date": open_date,
                        "ticker": "rhs",
                        "action": "BUY",
                        "price": price_open_r,
                        "qty_shares": qty_r,
                        "notional_est": qty_r * price_open_r,
                        "beta_used": weights["beta"],
                        "w_leg": weights["w_rhs"],
                    },
                    {
                        "trade_id": trade_id,
                        "date": close_date,
                        "ticker": "lhs",
                        "action": "BUY",
                        "price": price_close_l,
                        "qty_shares": qty_l,
                        "notional_est": qty_l * price_close_l,
                        "beta_used": weights["beta"],
                        "w_leg": weights["w_lhs"],
                    },
                    {
                        "trade_id": trade_id,
                        "date": close_date,
                        "ticker": "rhs",
                        "action": "SELL",
                        "price": price_close_r,
                        "qty_shares": qty_r,
                        "notional_est": qty_r * price_close_r,
                        "beta_used": weights["beta"],
                        "w_leg": weights["w_rhs"],
                    },
                ]
            )
        else:
            orders.extend(
                [
                    {
                        "trade_id": trade_id,
                        "date": open_date,
                        "ticker": "lhs",
                        "action": "BUY",
                        "price": price_open_l,
                        "qty_shares": qty_l,
                        "notional_est": qty_l * price_open_l,
                        "beta_used": weights["beta"],
                        "w_leg": weights["w_lhs"],
                    },
                    {
                        "trade_id": trade_id,
                        "date": open_date,
                        "ticker": "rhs",
                        "action": "SELL",
                        "price": price_open_r,
                        "qty_shares": qty_r,
                        "notional_est": qty_r * price_open_r,
                        "beta_used": weights["beta"],
                        "w_leg": weights["w_rhs"],
                    },
                    {
                        "trade_id": trade_id,
                        "date": close_date,
                        "ticker": "lhs",
                        "action": "SELL",
                        "price": price_close_l,
                        "qty_shares": qty_l,
                        "notional_est": qty_l * price_close_l,
                        "beta_used": weights["beta"],
                        "w_leg": weights["w_lhs"],
                    },
                    {
                        "trade_id": trade_id,
                        "date": close_date,
                        "ticker": "rhs",
                        "action": "BUY",
                        "price": price_close_r,
                        "qty_shares": qty_r,
                        "notional_est": qty_r * price_close_r,
                        "beta_used": weights["beta"],
                        "w_leg": weights["w_rhs"],
                    },
                ]
            )

    if not orders:
        return pd.DataFrame()

    orders_df = pd.DataFrame(orders)
    orders_df["capital_assumed"] = config.capital
    orders_df["lot"] = orders_df["qty_shares"] / config.lot_size
    return orders_df


def build_trade_summary(trades: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    daily_idx = daily.set_index("date")
    rows = []
    for _, trade in trades.iterrows():
        trade_id = int(trade.get("trade_id", 0)) or int(trade.name) + 1
        subset = daily.loc[daily["trade_id"] == trade_id]
        rows.append(
            {
                "trade_id": trade_id,
                "side": str(trade.get("side", "LONG")),
                "open_date": trade.get("open_date"),
                "close_date": trade.get("close_date"),
                "days": int(len(subset)),
                "pnl_trade_file": float(trade.get("pnl", np.nan)),
                "sum_contrib_lhs": float(subset["contrib_lhs"].sum()),
                "sum_contrib_rhs": float(subset["contrib_rhs"].sum()),
                "sum_ret_core": float(subset["ret_core"].sum()),
                "sum_cost": float(subset["cost"].sum()),
                "net_ret_from_daily": float((subset["ret_core"] - subset["cost"]).sum()),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> ExportConfig:
    parser = argparse.ArgumentParser(description="Export master artefacts for a run")
    parser.add_argument("run_id", nargs="?", help="Run identifier, e.g. 20250928_140708")
    parser.add_argument("--run_id", dest="run_id_flag")
    parser.add_argument("--run_dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--out_dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--prices", default=None)
    parser.add_argument("--capital", type=float, default=DEFAULT_CAPITAL)
    parser.add_argument("--lot_size", type=int, default=DEFAULT_LOT_SIZE)
    args = parser.parse_args(argv)

    run_dir = Path(sanitize(args.run_dir) or DEFAULT_RUN_DIR)
    out_dir = Path(sanitize(args.out_dir) or DEFAULT_RUN_DIR)
    manifest = Path(sanitize(args.manifest) or DEFAULT_MANIFEST)
    prices = Path(sanitize(args.prices)) if args.prices else None
    run_id = sanitize(args.run_id) or sanitize(args.run_id_flag) or ""
    return ExportConfig(
        run_id=run_id,
        run_dir=run_dir,
        out_dir=out_dir,
        manifest=manifest,
        prices=prices,
        capital=float(args.capital),
        lot_size=int(args.lot_size),
    )


def main(argv: Optional[List[str]] = None) -> Dict[str, object]:
    configure_logging()
    config = parse_args(argv)
    if not config.run_id:
        raise ValueError("run_id must be provided")
    config.run_dir = config.run_dir.resolve()
    config.out_dir = config.out_dir.resolve()
    config.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = config.manifest.resolve()
    manifest_row = load_manifest(manifest_path, config.run_id)
    lhs, rhs = parse_pair(manifest_row)
    LOGGER.info("Config | run_id=%s | run_dir=%s | out_dir=%s", config.run_id, config.run_dir, config.out_dir)
    prices_path = resolve_prices_path(manifest_row, config.prices)
    config.prices = prices_path.expanduser().resolve()
    LOGGER.info("Prices file = %s", config.prices)
    LOGGER.info("Detected pair from manifest: %s ~ %s", lhs, rhs)

    daily, px, trades = build_daily_panel(config, manifest_row, lhs, rhs)
    orders = build_orders(trades, px, daily, config)
    per_trade = build_trade_summary(trades, daily)

    daily_out = config.out_dir / f"master_daily_{config.run_id}.csv"
    orders_out = config.out_dir / f"master_orders_{config.run_id}.csv"
    trades_out = config.out_dir / f"master_trades_{config.run_id}.csv"

    daily_bps = daily.copy()
    for col in [
        "ret_net",
        "cost",
        "ret_core",
        "r_lhs",
        "r_rhs",
        "contrib_lhs",
        "contrib_rhs",
        "contrib_sum",
        "diff_check",
    ]:
        if col in daily_bps:
            daily_bps[f"{col}_bps"] = daily_bps[col] * 10_000
    write_csv(daily_bps, daily_out, "daily panel")
    if not orders.empty:
        write_csv(orders, orders_out, "orders blotter")
    else:
        LOGGER.warning("No orders to write (no trades)")
    if not per_trade.empty:
        per_trade_bps = per_trade.copy()
        for col in [
            "pnl_trade_file",
            "sum_contrib_lhs",
            "sum_contrib_rhs",
            "sum_ret_core",
            "sum_cost",
            "net_ret_from_daily",
        ]:
            per_trade_bps[f"{col}_bps"] = per_trade_bps[col] * 10_000
        write_csv(per_trade_bps, trades_out, "per-trade summary")
    else:
        LOGGER.warning("No per-trade summary")

    result = {
        "daily": str(daily_out),
        "orders": str(orders_out),
        "trades": str(trades_out),
    }
    LOGGER.info("DONE. master files are in: %s", config.out_dir)
    return result


if __name__ == "__main__":  # pragma: no cover - CLI entry
    try:
        output = main()
        json.dump(output, sys.stdout, indent=2)
        sys.stdout.write("\n")
    except Exception as exc:  # pragma: no cover - CLI diagnostics
        LOGGER.error("Export failed: %s", exc)
        raise
