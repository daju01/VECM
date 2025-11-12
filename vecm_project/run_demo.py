"""Demo entrypoint for the VECM project components."""
from __future__ import annotations

import logging
import uuid

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
DEMO_TICKERS = ["BBCA.JK", "BMRI.JK", "BBRI.JK", "TLKM.JK"]


def choose_tickers() -> list[str]:
    if not DATA_PATH.exists():
        ensure_price_data(tickers=DEMO_TICKERS)
    prices = load_cached_prices()
    columns = [col for col in prices.columns if col.lower() != "date"]
    if len(columns) < 2:
        raise ValueError("Need at least two tickers available from the streaming loader")
    return columns[:2]


def main() -> None:
    LOGGER.info("Initialising storage")
    with storage.managed_storage("demo-bootstrap") as conn:
        storage.storage_init(conn)

    ensure_price_data(tickers=DEMO_TICKERS)
    tickers = choose_tickers()
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
    run_bo(pair=subset, cfg=base_params, run_id=bo_run_id, n_init=1, iters=1, n_jobs=1)

    sh_run_id = f"demo_sh_{uuid.uuid4().hex[:8]}"
    run_successive_halving(
        pair=subset,
        cfg=base_params,
        run_id=sh_run_id,
        horizons=("short", "long"),
        n_trials=1,
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
