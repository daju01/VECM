"""Demo entrypoint for the VECM project components."""
from __future__ import annotations

import argparse
import logging
import uuid
from typing import List

from scripts import storage
from scripts.dashboard_aggregator import dashboard_aggregate
from scripts.data_streaming import DATA_PATH, ensure_price_data, load_cached_prices
from scripts.pareto import write_pareto_front
from scripts.playbook_api import playbook_score_once
from scripts.playbook_vecm import pipeline
from scripts.stage2_bo import run_bo
from scripts.stage2_sh import run_successive_halving
from scripts.vecm_hooks import vh_run_export, vh_run_visualize

LOGGER = storage.configure_logging("run_demo")


def _normalise_single_ticker(raw: str) -> str:
    ticker = raw.strip().upper()
    if not ticker:
        return ""
    if ticker.startswith("^") or "=" in ticker or ticker.endswith(".JK"):
        return ticker
    if "." in ticker:
        return ticker
    return f"{ticker}.JK"


def parse_ticker_prompt(prompt: str) -> List[str]:
    """Parse a flexible ticker prompt into normalised Yahoo Finance symbols."""

    tokens = []
    seen = set()
    for raw in prompt.split(","):
        ticker = _normalise_single_ticker(raw)
        if not ticker or ticker in seen:
            continue
        tokens.append(ticker)
        seen.add(ticker)
    if len(tokens) < 2:
        raise ValueError("Ticker prompt must include at least two tickers")
    return tokens


def choose_tickers_from_prompt(prompt: str) -> list[str]:
    tickers = parse_ticker_prompt(prompt)
    ensure_price_data(tickers=tickers)
    prices = load_cached_prices()
    available = {col for col in prices.columns if col != "Date"}
    missing = [ticker for ticker in tickers if ticker not in available]
    if missing:
        raise ValueError(
            f"Ticker prompt requested unavailable tickers: {', '.join(missing)}"
        )
    return tickers[:2]


def choose_default_tickers() -> list[str]:
    if not DATA_PATH.exists():
        ensure_price_data()
    prices = load_cached_prices()
    columns = [col for col in prices.columns if col != "Date"]
    if len(columns) < 2:
        raise ValueError("Need at least two tickers available from the streaming loader")
    return columns[:2]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the VECM demo workflow")
    parser.add_argument(
        "--prompt",
        "--tickers",
        dest="ticker_prompt",
        help="Comma separated ticker prompt (e.g. 'BBRI,BBNI')",
    )
    return parser


def main(*, ticker_prompt: str | None = None) -> None:
    LOGGER.info("Initialising storage")
    with storage.managed_storage("demo-bootstrap") as conn:
        storage.storage_init(conn)

    if ticker_prompt:
        tickers = choose_tickers_from_prompt(ticker_prompt)
    else:
        ensure_price_data()
        tickers = choose_default_tickers()
    subset = ",".join(tickers)
    LOGGER.info("Using tickers: %s", tickers)

    base_params = {"input_file": str(DATA_PATH), "subset": subset, "method": "TVECM"}

    metrics = playbook_score_once(base_params)
    LOGGER.info("playbook_score_once metrics: %s", metrics)

    detailed = pipeline(base_params, persist=True)
    run_id = detailed["run_id"]
    LOGGER.info("Pipeline completed with run_id=%s", run_id)

    try:
        vh_run_export(run_id)
        vh_run_visualize(run_id)
    except Exception as exc:  # pragma: no cover - demo convenience
        LOGGER.warning("Hook execution failed: %s", exc)

    bo_run_id = f"demo_bo_{uuid.uuid4().hex[:8]}"
    run_bo(pair=subset, cfg=base_params, run_id=bo_run_id, n_init=2, iters=3, n_jobs=1)

    sh_run_id = f"demo_sh_{uuid.uuid4().hex[:8]}"
    run_successive_halving(
        pair=subset,
        cfg=base_params,
        run_id=sh_run_id,
        horizons=("short", "long"),
        n_trials=3,
        n_jobs=1,
    )

    with storage.managed_storage("demo-summary") as conn:
        frontier = write_pareto_front(conn, bo_run_id)
        LOGGER.info("Stored %d Pareto rows for %s", len(frontier), bo_run_id)
        dashboard = dashboard_aggregate(conn, bo_run_id)
        LOGGER.info("Dashboard summary for %s: %s", bo_run_id, dashboard)

    LOGGER.info("Demo complete. Primary run_id=%s", run_id)


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    main(ticker_prompt=args.ticker_prompt)
