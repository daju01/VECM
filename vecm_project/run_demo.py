"""Demo entrypoint for the VECM project components.

Demo ini menjalankan satu pass playbook (regime-aware + short-term overlay)
dan kemudian Stage-2 Bayesian optimisation untuk satu pair pilihan Anda.
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import sys
import uuid
from typing import Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from vecm_project.scripts import storage
from vecm_project.scripts.dashboard_aggregator import dashboard_aggregate
from vecm_project.scripts.data_streaming import DATA_PATH, ensure_price_data, load_cached_prices
from vecm_project.scripts.pareto import write_pareto_front
from vecm_project.scripts.playbook_api import playbook_score_once
from vecm_project.scripts.stage2_bo import run_bo

LOGGER = logging.getLogger(__name__)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VECM demo: regime-aware pairs trading + short-term overlay"
    )
    parser.add_argument(
        "--pair",
        default="ANTM,INCO",
        help="Pair ticker untuk Stage-2 BO, format 'LHS,RHS' (default: ANTM,INCO)",
    )
    parser.add_argument(
        "--method",
        default="TVECM",
        help="Metode utama (default: TVECM)",
    )
    parser.add_argument(
        "--horizon",
        default="oos_full",
        help="Nama horizon OOS (default: oos_full)",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force refresh Yahoo price data ke adj_close_data.csv sebelum demo.",
    )
    parser.add_argument(
        "--n-init",
        type=int,
        default=4,
        help="Jumlah initial points untuk Stage-2 BO (default: 4)",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=12,
        help="Jumlah iterasi TPE tambahan untuk Stage-2 BO (default: 12)",
    )

    return parser.parse_args(list(argv) if argv is not None else sys.argv[1:])


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    args = _parse_args(argv)

    if args.refresh:
        LOGGER.info("Refreshing Yahoo price data into %s", DATA_PATH)
        ensure_price_data(DATA_PATH)

    # Load panel harga (adj_close_data.csv) – dipakai oleh playbook, short-term
    # overlay, dan gating regime spread.
    prices = load_cached_prices(DATA_PATH)
    LOGGER.info("Loaded price panel from %s with shape %s", DATA_PATH, prices.shape)

    # ------------------------------------------------------------------
    # 1) Single pass playbook (regime-aware + short-term overlay aktif)
    # ------------------------------------------------------------------
    run_id = f"demo-{uuid.uuid4().hex[:8]}"
    base_params = {
        "input_file": str(DATA_PATH),
        "method": args.method,
        "horizon": args.horizon,
        "stage": 2,
        "tag": run_id,
    }

    LOGGER.info(
        "Running single playbook pass (regime-aware + short-term overlay ON) ..."
    )
    score = playbook_score_once(base_params)
    LOGGER.info(
        "Single-run metrics | sharpe_oos=%.4f maxdd=%.4f turnover=%.4f",
        score.get("sharpe_oos", 0.0),
        score.get("maxdd", 0.0),
        score.get("turnover", 0.0),
    )

    # ------------------------------------------------------------------
    # 2) Stage-2 Bayesian optimisation untuk pair tertentu
    # ------------------------------------------------------------------
    bo_run_id = f"demo_stage2_bo_{args.pair.replace(',', '-')}_{uuid.uuid4().hex[:6]}"
    LOGGER.info(
        "Launching Stage-2 BO | pair=%s method=%s horizon=%s run_id=%s ...",
        args.pair,
        args.method,
        args.horizon,
        bo_run_id,
    )

    study = run_bo(
        pair=args.pair,
        method=args.method,
        horizon=args.horizon,
        cfg={"input_file": str(DATA_PATH)},
        run_id=bo_run_id,
        n_init=args.n_init,
        iters=args.iters,
        n_jobs=1,
    )

    best = study.best_trial
    best_rec = best.user_attrs.get("record", {})
    best_diag = best_rec.get("diagnostics", {})

    LOGGER.info(
        "Best Stage-2 trial | number=%d Score=%.4f sharpe_oos=%.4f maxdd=%.4f "
        "turnover=%.4f t_ann=%.4f",
        best.number,
        best.value,
        best_diag.get("sharpe_oos", 0.0),
        best_diag.get("maxdd", 0.0),
        best_diag.get("turnover", 0.0),
        best_diag.get("turnover_annualised", 0.0),
    )

    # ------------------------------------------------------------------
    # 3) Tuliskan Pareto frontier & ringkasan dashboard untuk run BO ini
    # ------------------------------------------------------------------
    with storage.managed_storage("demo-summary") as conn:
        frontier = write_pareto_front(conn, bo_run_id)
        LOGGER.info("Stored %d Pareto rows for %s", len(frontier), bo_run_id)
        dashboard = dashboard_aggregate(conn, bo_run_id)
        LOGGER.info("Dashboard summary for %s: %s", bo_run_id, dashboard)

    LOGGER.info("Demo complete. Primary run_id=%s", run_id)


if __name__ == "__main__":
    main()
