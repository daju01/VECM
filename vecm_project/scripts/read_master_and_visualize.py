"""Generate diagnostic plots from master exports produced by the playbook.

This module mirrors the behaviour of the reference R script
``read_master_and_visualize.R``.  It loads the ``master_daily_*``,
``master_orders_*`` and ``master_trades_*`` artefacts for a given ``run_id`` and
produces a dashboard-style figure summarising strategy performance.  The
component plots include

* cumulative out-of-sample returns,
* per-leg contribution bars during active positions,
* inferred hedge ratios (``beta``),
* per-trade profit and loss bars, and
* an order timeline showing BUY/SELL markers per ticker.

The script can either display the figure interactively (``--show``) or save it
as an image (``--save_path``).  When no explicit output path is provided, a
PNG file is written alongside the master artefacts.
"""

from __future__ import annotations

import argparse
import logging
import os
import pathlib
from dataclasses import dataclass
from typing import Iterable, Optional

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Use a non-interactive backend automatically in headless environments.
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")

LOGGER = logging.getLogger("vecm.read_master_and_visualize")


@dataclass
class MasterData:
    daily: pd.DataFrame
    orders: pd.DataFrame
    trades: pd.DataFrame


def _configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="[%(asctime)s] %(levelname)s: %(message)s")


def _load_csv(path: pathlib.Path, label: str, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"{label} file not found: {path}")
        LOGGER.warning("%s not found: %s", label, path)
        return pd.DataFrame()
    LOGGER.debug("Loading %s from %s", label, path)
    return pd.read_csv(path)


def _prepare_master_frames(run_id: str, out_dir: pathlib.Path) -> MasterData:
    daily_path = out_dir / f"master_daily_{run_id}.csv"
    orders_path = out_dir / f"master_orders_{run_id}.csv"
    trades_path = out_dir / f"master_trades_{run_id}.csv"

    daily = _load_csv(daily_path, "master_daily")
    orders = _load_csv(orders_path, "master_orders", required=False)
    trades = _load_csv(trades_path, "master_trades", required=False)

    if "date" not in daily.columns:
        raise ValueError("master_daily file must contain a 'date' column")

    daily["date"] = pd.to_datetime(daily["date"])
    daily.sort_values("date", inplace=True)

    if not orders.empty and "date" in orders.columns:
        orders["date"] = pd.to_datetime(orders["date"])
        orders.sort_values("date", inplace=True)

    if not trades.empty:
        for col in ("open_date", "close_date"):
            if col in trades.columns:
                trades[col] = pd.to_datetime(trades[col])
        if "trade_id" in trades.columns:
            trades["trade_id"] = trades["trade_id"].astype(str)

    return MasterData(daily=daily, orders=orders, trades=trades)


def _as_percent(series: pd.Series, scale_from_bps: bool = False) -> pd.Series:
    if series.empty:
        return series
    values = series.astype(float)
    if scale_from_bps:
        values = values / 100.0
    else:
        values = values * 100.0
    return values


def _prepare_daily_features(daily: pd.DataFrame) -> pd.DataFrame:
    daily = daily.copy()

    if "ret_net_bps" in daily:
        daily["ret_net_pct"] = _as_percent(daily["ret_net_bps"], scale_from_bps=True)
    elif "ret_net" in daily:
        daily["ret_net_pct"] = _as_percent(daily["ret_net"], scale_from_bps=False)
    else:
        daily["ret_net_pct"] = 0.0
        LOGGER.warning("ret_net column not found; assuming zero net returns")

    daily["cum_ret_pct"] = daily["ret_net_pct"].fillna(0.0).cumsum()
    daily["in_pos"] = daily.get("pos", 0).fillna(0).astype(int)

    if {"contrib_lhs_bps", "contrib_rhs_bps"}.issubset(daily.columns):
        daily["contrib_lhs_pct"] = _as_percent(daily["contrib_lhs_bps"], scale_from_bps=True)
        daily["contrib_rhs_pct"] = _as_percent(daily["contrib_rhs_bps"], scale_from_bps=True)
    elif {"contrib_lhs", "contrib_rhs"}.issubset(daily.columns):
        daily["contrib_lhs_pct"] = _as_percent(daily["contrib_lhs"], scale_from_bps=False)
        daily["contrib_rhs_pct"] = _as_percent(daily["contrib_rhs"], scale_from_bps=False)
    else:
        daily["contrib_lhs_pct"] = np.nan
        daily["contrib_rhs_pct"] = np.nan

    return daily


def _labels_from_orders(orders: pd.DataFrame) -> tuple[str, str]:
    if orders.empty or "ticker" not in orders:
        return ("LHS", "RHS")
    tickers = orders["ticker"].dropna().unique()
    if len(tickers) >= 2:
        return (str(tickers[0]), str(tickers[1]))
    if len(tickers) == 1:
        return (str(tickers[0]), "RHS")
    return ("LHS", "RHS")


def _plot_cumulative(ax: plt.Axes, daily: pd.DataFrame) -> None:
    ax.plot(daily["date"], daily["cum_ret_pct"], linewidth=1.2, color="#1f77b4")
    ax.set_title("Cumulative Net Return (OOS)")
    ax.set_ylabel("Cumulative (%)")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.grid(True, alpha=0.3)


def _plot_contributions(ax: plt.Axes, daily: pd.DataFrame, labels: Iterable[str]) -> None:
    mask = (daily["in_pos"] == 1) & daily["contrib_lhs_pct"].notna() & daily["contrib_rhs_pct"].notna()
    contrib = daily.loc[mask, ["date", "contrib_lhs_pct", "contrib_rhs_pct"]]
    if contrib.empty:
        ax.text(0.5, 0.5, "No contribution data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Per-leg Contribution (position on)")
        ax.axis("off")
        return

    contrib_long = contrib.melt(id_vars="date", var_name="leg", value_name="pct")
    mapping = {
        "contrib_lhs_pct": f"Leg A ({labels[0]})",
        "contrib_rhs_pct": f"Leg B ({labels[1]})",
    }
    contrib_long["leg"] = contrib_long["leg"].map(mapping)
    pivot = contrib_long.pivot_table(index="date", columns="leg", values="pct", aggfunc="sum").fillna(0.0)

    bottoms = np.zeros(len(pivot))
    for idx, (col, color) in enumerate(zip(pivot.columns, ("#2ca02c", "#d62728"))):
        ax.bar(pivot.index, pivot[col], bottom=bottoms, width=0.8, label=col, color=color)
        bottoms += pivot[col].to_numpy()

    ax.set_title("Per-leg Contribution (position on)")
    ax.set_ylabel("Contribution (%)")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))


def _plot_beta(ax: plt.Axes, daily: pd.DataFrame) -> None:
    if "beta_inferred" not in daily:
        ax.text(0.5, 0.5, "β not available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Daily β (inferred)")
        ax.axis("off")
        return

    series = daily["beta_inferred"].astype(float)
    if series.notna().sum() == 0:
        ax.text(0.5, 0.5, "β not available", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Daily β (inferred)")
        ax.axis("off")
        return

    ax.plot(daily["date"], series, linewidth=1.0, color="#ff7f0e")
    ax.set_title("Daily β (inferred)")
    ax.set_ylabel("β")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))


def _plot_trade_pnl(ax: plt.Axes, trades: pd.DataFrame) -> None:
    if trades.empty:
        ax.text(0.5, 0.5, "No trades", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("PnL per Trade")
        ax.axis("off")
        return

    for col in ("net_ret_from_daily_bps", "pnl_trade_file_bps"):
        if col in trades:
            values = trades[col].astype(float) / 100.0
            break
    else:
        ax.text(0.5, 0.5, "PnL column missing", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("PnL per Trade")
        ax.axis("off")
        return

    trade_ids = trades.get("trade_id")
    if trade_ids is None:
        trade_ids = pd.Index(range(1, len(values) + 1)).astype(str)
    ax.bar(trade_ids.astype(str), values, color="#9467bd")
    ax.set_title("PnL per Trade")
    ax.set_ylabel("PnL (%)")
    ax.set_xlabel("Trade ID")
    ax.axhline(0, color="black", linewidth=0.8)


def _plot_order_timeline(ax: plt.Axes, orders: pd.DataFrame) -> None:
    if orders.empty or "date" not in orders or "ticker" not in orders:
        ax.text(0.5, 0.5, "No orders", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Order Timeline")
        ax.axis("off")
        return

    data = orders.copy()
    data["date"] = pd.to_datetime(data["date"])
    data.sort_values("date", inplace=True)
    actions = data.get("action", "ACTION").astype(str)

    markers = {"BUY": "^", "SELL": "v"}
    ax.set_title("Order Timeline")
    for action, marker in markers.items():
        mask = actions.str.upper() == action
        subset = data.loc[mask]
        if subset.empty:
            continue
        ax.scatter(subset["date"], subset["ticker"], marker=marker, s=60, label=action)

    ax.set_ylabel("Ticker")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))


def _build_figure(master: MasterData, run_id: str, save_path: pathlib.Path, show: bool, dpi: int) -> pathlib.Path:
    daily = _prepare_daily_features(master.daily)
    leg_labels = _labels_from_orders(master.orders)

    fig = plt.figure(figsize=(14, 10))
    grid = fig.add_gridspec(3, 2, height_ratios=[1, 0.9, 1.0])

    ax_cum = fig.add_subplot(grid[0, 0])
    ax_beta = fig.add_subplot(grid[0, 1])
    ax_contrib = fig.add_subplot(grid[1, :])
    ax_trade = fig.add_subplot(grid[2, 0])
    ax_orders = fig.add_subplot(grid[2, 1])

    _plot_cumulative(ax_cum, daily)
    _plot_beta(ax_beta, daily)
    _plot_contributions(ax_contrib, daily, leg_labels)
    _plot_trade_pnl(ax_trade, master.trades)
    _plot_order_timeline(ax_orders, master.orders)

    fig.suptitle(f"VECM Diagnostics — Run {run_id}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi)
    LOGGER.info("Dashboard saved to %s", save_path)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return save_path


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualise VECM master exports")
    parser.add_argument("--run_id", required=True, help="Run identifier (suffix of master_* files)")
    parser.add_argument("--out_dir", default="out_ms", help="Directory containing master artefacts")
    parser.add_argument("--save_path", default=None, help="Optional output image path")
    parser.add_argument("--dpi", type=int, default=200, help="Figure DPI when saving")
    parser.add_argument("--show", action="store_true", help="Display the figure after saving")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> pathlib.Path:
    args = parse_args(argv)
    _configure_logging(args.verbose)

    run_id = args.run_id
    out_dir = pathlib.Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    save_path = pathlib.Path(args.save_path) if args.save_path else out_dir / f"visual_{run_id}.png"
    save_path = save_path.expanduser().resolve()

    LOGGER.info("Generating dashboard for run_id=%s (out_dir=%s)", run_id, out_dir)
    master = _prepare_master_frames(run_id, out_dir)
    return _build_figure(master, run_id, save_path=save_path, show=args.show, dpi=args.dpi)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    try:
        main()
    except Exception as exc:  # pragma: no cover - surface errors to caller
        LOGGER.error("visualisation failed: %s", exc)
        raise
