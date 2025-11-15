"""Demo entrypoint for the VECM project components."""
from __future__ import annotations

import argparse
import logging
import sys
import uuid
from typing import Sequence

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


def parse_ticker_list(raw: str) -> list[str]:
    """Parse a comma separated ticker string into a unique list."""

    tickers = [item.strip() for item in raw.split(",") if item.strip()]
    deduped = list(dict.fromkeys(ticker.upper() for ticker in tickers))
    if len(deduped) < 2:
        raise ValueError("Need at least two tickers in the provided list")
    return deduped


def _normalise_provided_tickers(tickers: Sequence[str]) -> list[str]:
    """Normalise raw ticker aliases to canonical symbols for downloads."""

    normalised: list[str] = []
    for ticker in tickers:
        upper = ticker.upper().strip()
        if not upper:
            continue
        if "." not in upper:
            upper = f"{upper}.JK"
        if upper not in normalised:
            normalised.append(upper)
    return normalised


def prompt_for_tickers(prompt: str) -> list[str]:
    """Request tickers from stdin using *prompt*.

    If the user provides no input or stdin is unavailable, an empty list is
    returned so the caller can fall back to an automatic selection.
    """

    try:
        raw = input(prompt)
    except EOFError:  # pragma: no cover - non-interactive execution
        return []
    if not raw.strip():
        return []
    return parse_ticker_list(raw)


def _alias_map(columns: Sequence[str]) -> dict[str, str]:
    """Build a case-insensitive alias map for the cached ticker columns."""

    mapping: dict[str, str] = {}
    for column in columns:
        upper = column.upper()
        mapping.setdefault(upper, column)
        if "." in upper:
            prefix = upper.split(".", 1)[0]
            mapping.setdefault(prefix, column)
        if upper.endswith(".JK"):
            mapping.setdefault(upper[:-3], column)
    return mapping


def _resolve_tickers(candidates: Sequence[str], columns: Sequence[str]) -> list[str]:
    mapping = _alias_map(columns)
    resolved: list[str] = []
    missing: list[str] = []

    for ticker in candidates:
        key = ticker.upper().strip()
        column = mapping.get(key)
        if column is None:
            missing.append(ticker)
            continue
        if column not in resolved:
            resolved.append(column)

    if missing:
        raise ValueError(
            "Tickers not found in cached data: %s" % ", ".join(missing)
        )
    if len(resolved) < 2:
        raise ValueError("Need at least two unique tickers after resolving aliases")
    return resolved


def choose_tickers(*, provided: Sequence[str] | None = None, ticker_prompt: str | None = None) -> list[str]:
    """Determine which tickers to use for the demo run.

    *provided* may contain a user-specified list (already parsed). If not
    supplied, *ticker_prompt* is used to ask the user for input. Both pathways
    are validated against the cached price data. When neither path yields a
    selection, the first two cached tickers are returned for backwards
    compatibility with the original behaviour.
    """

    if not DATA_PATH.exists():
        ensure_price_data()
    prices = load_cached_prices()
    columns = [col for col in prices.columns if col.lower() != "date"]
    if len(columns) < 2:
        raise ValueError("Need at least two tickers available from the streaming loader")

    if provided:
        return _resolve_tickers(provided, columns)

    if ticker_prompt is not None and sys.stdin is not None and sys.stdin.isatty():
        prompted = prompt_for_tickers(ticker_prompt)
        if prompted:
            return _resolve_tickers(prompted, columns)

    return columns[:2]


_PROMPT_UNSET = object()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the VECM demo pipeline")
    parser.add_argument(
        "--tickers",
        help=(
            "Comma separated list of tickers to use (e.g. 'BBRI,BBNI'). "
            "Accepts raw tickers or their .JK variants."
        ),
    )
    parser.add_argument(
        "--prompt",
        nargs="?",
        const=(
            "Masukkan daftar ticker dipisahkan koma (kosongkan untuk default): "
        ),
        default=_PROMPT_UNSET,
        help=(
            "Aktifkan mode interaktif untuk memasukkan daftar ticker melalui stdin. "
            "Tanpa argumen tambahan menggunakan pesan default."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    LOGGER.info("Initialising storage")
    with storage.managed_storage("demo-bootstrap") as conn:
        storage.storage_init(conn)

    provided = parse_ticker_list(args.tickers) if args.tickers else None
    if provided:
        ensure_price_data(tickers=_normalise_provided_tickers(provided))
    ticker_prompt = None if args.prompt is _PROMPT_UNSET else args.prompt
    tickers = choose_tickers(provided=provided, ticker_prompt=ticker_prompt)
    if not provided:
        ensure_price_data(tickers=tickers)
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
    main()
