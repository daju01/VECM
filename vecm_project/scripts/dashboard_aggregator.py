"""Dashboard aggregation helpers mirroring the R workflow."""
from __future__ import annotations

import datetime as dt
import logging
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd

from . import storage

LOGGER = storage.configure_logging("dashboard_aggregator")
OUT_DASHBOARD_DIR = storage.BASE_DIR / "out" / "dashboard"

try:  # Optional dependency used by ``benchmark_storage``.
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover - optional dependency
    pq = None


def _fetch_scalar(
    conn: Any, query: str, params: Iterable[Any] | None = None
) -> Optional[Any]:
    row = conn.execute(query, params or []).fetchone()
    if row is None:
        return None
    if isinstance(row, tuple):
        return row[0]
    return row


def _safe_percentile(value: Optional[Any]) -> Optional[float]:
    return float(value) if value is not None else None


def dashboard_aggregate(conn, run_id: str) -> Dict[str, Any]:
    """Compute and persist daily dashboard metrics for ``run_id``."""

    if not run_id:
        raise ValueError("run_id must be provided for dashboard aggregation")

    q1 = conn.execute(
        """
        SELECT COUNT(*) AS n_trials, SUM(eval_time_s) AS sum_eval
        FROM trials WHERE run_id = ? AND stage = 2
        """,
        [run_id],
    ).fetchone()
    n_trials = int(q1[0]) if q1 and q1[0] is not None else 0
    sum_eval = float(q1[1]) if q1 and q1[1] is not None else None
    trials_per_hour = None
    if sum_eval and sum_eval > 0:
        trials_per_hour = n_trials / (sum_eval / 3600.0)

    idle_p95 = _safe_percentile(
        _fetch_scalar(
            conn,
            """
            SELECT PERCENTILE_CONT(worker_idle_pct, 0.95) AS idle_p95
            FROM exec_metrics WHERE run_id = ?
            """,
            [run_id],
        )
    )

    pruned_pct = _fetch_scalar(
        conn,
        """
        SELECT 100.0 * AVG(CASE WHEN pruned THEN 1 ELSE 0 END)
        FROM trials WHERE run_id = ? AND stage = 2
        """,
        [run_id],
    )
    pruned_pct = float(pruned_pct) if pruned_pct is not None else None

    n_pareto = _fetch_scalar(
        conn,
        """
        SELECT COUNT(*) AS n_pareto
        FROM pareto_front WHERE run_id = ? AND is_nondominated
        """,
        [run_id],
    )
    n_pareto = int(n_pareto) if n_pareto is not None else None

    dd_p95 = _safe_percentile(
        _fetch_scalar(
            conn,
            """
            SELECT PERCENTILE_CONT(maxdd, 0.95) AS dd_p95
            FROM trials WHERE run_id = ? AND stage = 2
            """,
            [run_id],
        )
    )

    rowid_result = conn.execute(
        """
        SELECT duckdb_query_p95_s, parquet_read_p95_s
        FROM storage_metrics WHERE run_id = ?
        ORDER BY rowid DESC LIMIT 1
        """,
        [run_id],
    ).fetchone()
    duckdb_q = float(rowid_result[0]) if rowid_result and rowid_result[0] is not None else None
    parquet_q = float(rowid_result[1]) if rowid_result and rowid_result[1] is not None else None

    today = dt.date.today()
    dashboard_row = {
        "date": today,
        "run_id": run_id,
        "trials_per_hour": trials_per_hour,
        "idle_p95": idle_p95,
        "pruned_pct": pruned_pct,
        "n_pareto": n_pareto,
        "dd_p95": dd_p95,
        "ttr_days": None,
        "duckdb_q_p95_s": duckdb_q,
        "parquet_p95_s": parquet_q,
    }

    df = pd.DataFrame([dashboard_row])
    storage.storage_write_df(conn, "dashboard_daily", df)

    OUT_DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DASHBOARD_DIR / f"dashboard_{today.isoformat()}.csv"
    df.to_csv(csv_path, index=False)
    storage.storage_analyze(conn)

    LOGGER.info("Dashboard row written for run_id=%s", run_id)
    return dashboard_row


def benchmark_storage(
    conn, parquet_path: Path | str, run_id: Optional[str] = None
) -> Dict[str, Optional[float]]:
    """Benchmark DuckDB and Parquet access latencies."""

    parquet_path = Path(parquet_path)
    metrics: Dict[str, Optional[float]] = {"duckdb_query_p95_s": None, "parquet_read_p95_s": None}

    start = time.perf_counter()
    conn.execute("SELECT COUNT(*) FROM trials").fetchone()
    metrics["duckdb_query_p95_s"] = time.perf_counter() - start

    if parquet_path.exists() and pq is not None:
        start = time.perf_counter()
        pq.read_table(parquet_path)
        metrics["parquet_read_p95_s"] = time.perf_counter() - start
    elif not parquet_path.exists():
        LOGGER.warning("Parquet path %s does not exist; skipping benchmark", parquet_path)
    else:  # pq is None
        LOGGER.warning("pyarrow is unavailable; cannot benchmark parquet read")

    metrics_row = {
        "run_id": run_id or "benchmark",
        "manifest_writes": None,
        "write_conflicts": None,
        "duckdb_query_p95_s": metrics["duckdb_query_p95_s"],
        "parquet_read_p95_s": metrics["parquet_read_p95_s"],
    }
    storage.storage_write_df(conn, "storage_metrics", pd.DataFrame([metrics_row]))
    return metrics


if __name__ == "__main__":
    with storage.managed_storage("dashboard-smoke") as conn:
        try:
            runs = conn.execute("SELECT DISTINCT run_id FROM trials ORDER BY run_id DESC LIMIT 1").fetchone()
            if not runs:
                raise RuntimeError("No trials recorded; cannot run dashboard aggregation smoke test")
            run_id = runs[0]
            row = dashboard_aggregate(conn, run_id)
            print(f"Dashboard aggregation completed for {run_id}: {row}")
        except Exception as exc:  # pragma: no cover - smoke test guard
            print(f"Dashboard aggregation failed: {exc}")
