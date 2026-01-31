"""DuckDB storage helpers mirroring the R stack layout."""
from __future__ import annotations

import contextlib
import datetime as dt
import json
import logging
import pathlib
import tempfile
from typing import Any, Iterable, Iterator, Mapping, Sequence

import duckdb
import pandas as pd

try:  # Optional dependency for Parquet export helpers.
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover - optional dependency
    pa = None
    pq = None

DuckDBConnection = duckdb.DuckDBPyConnection

# Project level constants ----------------------------------------------------
BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
DB_DIR = BASE_DIR / "out" / "db"
DB_PATH = DB_DIR / "vecm.duckdb"
LOG_DIR = BASE_DIR / "out" / "logs"

LOGGER = logging.getLogger(__name__)

DIRTY_META_TABLE = "storage_meta_dirty"
ANALYZE_MIN_INTERVAL = dt.timedelta(hours=6)
TRACKED_TABLES: Sequence[str] = (
    "runs",
    "trials",
    "exec_metrics",
    "trade_stats",
    "storage_metrics",
    "model_checks",
    "pareto_front",
    "sd_loop",
    "dashboard_daily",
)


def _ensure_dirs() -> None:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def _escape(name: str) -> str:
    try:
        return duckdb.escape_identifier(name)
    except AttributeError:  # pragma: no cover - compatibility fallback
        escaped = name.replace('"', '""')
        return f'"{escaped}"'


def configure_logging(name: str = __name__) -> logging.Logger:
    _ensure_dirs()
    log_path = LOG_DIR / f"{name}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(name)


def storage_open(read_only: bool = False) -> DuckDBConnection:
    """Open a DuckDB connection, ensuring directories exist first."""

    _ensure_dirs()
    conn = duckdb.connect(str(DB_PATH), read_only=read_only)
    if not read_only:
        try:
            conn.execute("PRAGMA busy_timeout=60000")
        except duckdb.Error:  # pragma: no cover - compatibility fallback
            logging.getLogger(__name__).debug(
                "DuckDB busy_timeout pragma unsupported; continuing without it"
            )
        conn.execute("PRAGMA threads=4")
    return conn


def storage_init(conn: DuckDBConnection) -> None:
    """Create tables mirroring the reference R implementation."""

    statements = [
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            started_at TIMESTAMP,
            finished_at TIMESTAMP,
            n_workers INT,
            plan TEXT,
            seed_method TEXT,
            notes TEXT
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS trials (
            run_id TEXT,
            trial_id TEXT,
            stage INT,
            pair TEXT,
            method TEXT,
            params TEXT,
            horizon TEXT,
            eval_time_s DOUBLE,
            sharpe_oos DOUBLE,
            maxdd DOUBLE,
            turnover DOUBLE,
            alpha_ec DOUBLE,
            half_life_full DOUBLE,
            pruned BOOLEAN
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS exec_metrics (
            run_id TEXT,
            metric_ts TIMESTAMP,
            cpu_util_avg DOUBLE,
            cpu_util_p95 DOUBLE,
            worker_idle_pct DOUBLE,
            chunk_size INT,
            progress_latency_s DOUBLE
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS trade_stats (
            run_id TEXT PRIMARY KEY,
            n_trades INT,
            avg_hold_days DOUBLE,
            turnover_annualised DOUBLE
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS regime_stats (
            run_id TEXT PRIMARY KEY,
            p_mr_mean DOUBLE,
            p_mr_inpos_mean DOUBLE
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS storage_metrics (
            run_id TEXT,
            manifest_writes INT,
            write_conflicts INT,
            duckdb_query_p95_s DOUBLE,
            parquet_read_p95_s DOUBLE
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS model_checks (
            run_id TEXT,
            pair TEXT,
            johansen_rank INT,
            det_term TEXT,
            tvecm_thresholds TEXT,
            spec_ok BOOLEAN
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS pareto_front (
            run_id TEXT,
            trial_id TEXT,
            is_nondominated BOOLEAN
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS sd_loop (
            run_id TEXT,
            date DATE,
            vol_regime TEXT,
            corr_level DOUBLE,
            liquidity_score DOUBLE,
            gating DOUBLE,
            kelly_eff DOUBLE,
            z_th_eff DOUBLE
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS dashboard_daily (
            date DATE,
            run_id TEXT,
            trials_per_hour DOUBLE,
            idle_p95 DOUBLE,
            pruned_pct DOUBLE,
            n_pareto INT,
            dd_p95 DOUBLE,
            ttr_days DOUBLE,
            duckdb_q_p95_s DOUBLE,
            parquet_p95_s DOUBLE,
            n_trades INT,
            turnover_annualised DOUBLE,
            avg_p_regime DOUBLE,
            avg_abs_delta_score_pos DOUBLE,
            avg_abs_delta_mom12_pos DOUBLE,
            avg_delta_value_entry DOUBLE,
            avg_delta_quality_entry DOUBLE
        );
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {DIRTY_META_TABLE} (
            table_name TEXT PRIMARY KEY,
            dirty BOOLEAN,
            last_marked TIMESTAMP,
            last_analyze TIMESTAMP
        );
        """,
    ]
    with with_transaction(conn):
        for sql in statements:
            conn.execute(sql)

        index_specs = {
            "runs": ("run_id",),
            "trials": ("run_id", "trial_id"),
            "exec_metrics": ("run_id",),
            "trade_stats": ("run_id",),
            "regime_stats": ("run_id",),
            "storage_metrics": ("run_id",),
            "model_checks": ("run_id", "pair"),
            "pareto_front": ("run_id",),
            "sd_loop": ("run_id", "date"),
            "dashboard_daily": ("run_id", "date"),
        }
        for table, columns in index_specs.items():
            storage_create_index(conn, table, columns)
        _bootstrap_dirty_meta(conn)


def storage_create_index(
    conn: DuckDBConnection, table: str, columns: Sequence[str]
) -> None:
    for column in columns:
        index_name = f"idx_{table}_{column}"
        conn.execute(
            "CREATE INDEX IF NOT EXISTS {} ON {}({})".format(
                _escape(index_name),
                _escape(table),
                _escape(column),
            )
        )


def _bootstrap_dirty_meta(conn: DuckDBConnection) -> None:
    now = dt.datetime.utcnow()
    for table in TRACKED_TABLES:
        conn.execute(
            f"""
            INSERT INTO {DIRTY_META_TABLE} (table_name, dirty, last_marked, last_analyze)
            VALUES (?, FALSE, ?, NULL)
            ON CONFLICT(table_name) DO NOTHING
            """,
            [table, now],
        )


def _mark_table_dirty(conn: DuckDBConnection, table: str) -> None:
    conn.execute(
        f"""
        INSERT INTO {DIRTY_META_TABLE} (table_name, dirty, last_marked)
        VALUES (?, TRUE, ?)
        ON CONFLICT(table_name) DO UPDATE SET
            dirty = TRUE,
            last_marked = excluded.last_marked
        """,
        [table, dt.datetime.utcnow()],
    )


def storage_schedule_analyze(conn: DuckDBConnection, table: str) -> None:
    """Mark a table as needing statistics refresh."""

    try:
        _mark_table_dirty(conn, table)
    except duckdb.Error as exc:  # pragma: no cover - defensive
        LOGGER.warning("Failed to mark table %%s dirty: %%s", table, exc)


def _eligible_dirty_tables(conn: DuckDBConnection, *, force: bool) -> Iterable[str]:
    threshold = dt.datetime.utcnow() - ANALYZE_MIN_INTERVAL
    if force:
        query = f"SELECT table_name FROM {DIRTY_META_TABLE} WHERE dirty"
        rows = conn.execute(query).fetchall()
        return [row[0] for row in rows]
    query = f"""
        SELECT table_name
        FROM {DIRTY_META_TABLE}
        WHERE dirty AND (last_analyze IS NULL OR last_analyze <= ?)
    """
    rows = conn.execute(query, [threshold]).fetchall()
    return [row[0] for row in rows]


def storage_run_maintenance(
    conn: DuckDBConnection, *, force: bool = False
) -> None:
    """Run pending ANALYZE statements in batch if tables are marked dirty."""

    tables = list(_eligible_dirty_tables(conn, force=force))
    if not tables:
        return
    now = dt.datetime.utcnow()
    for table in tables:
        try:
            conn.execute(f"ANALYZE {_escape(table)}")
            conn.execute(
                f"""
                UPDATE {DIRTY_META_TABLE}
                SET dirty = FALSE,
                    last_analyze = ?,
                    last_marked = COALESCE(last_marked, ?)
                WHERE table_name = ?
                """,
                [now, now, table],
            )
        except duckdb.Error as exc:  # pragma: no cover - defensive
            LOGGER.warning("ANALYZE failed for %%s: %%s", table, exc)


def storage_analyze(conn: DuckDBConnection) -> None:
    """Force an ANALYZE run for all dirty tables regardless of interval."""

    storage_run_maintenance(conn, force=True)


def write_run(
    conn: DuckDBConnection,
    run_id: str,
    *,
    started_at: dt.datetime | None = None,
    finished_at: dt.datetime | None = None,
    n_workers: int | None = None,
    plan: str | None = None,
    seed_method: str | None = None,
    notes: str = "",
) -> None:
    started = started_at or dt.datetime.utcnow()
    conn.execute(
        """
        INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (run_id) DO UPDATE SET
            started_at = excluded.started_at,
            finished_at = excluded.finished_at,
            n_workers = excluded.n_workers,
            plan = excluded.plan,
            seed_method = excluded.seed_method,
            notes = excluded.notes;
        """,
        [run_id, started, finished_at, n_workers, plan, seed_method, notes],
    )
    storage_schedule_analyze(conn, "runs")


def mark_run_finished(
    conn: DuckDBConnection,
    run_id: str,
    finished_at: dt.datetime | None = None,
) -> None:
    conn.execute(
        "UPDATE runs SET finished_at = ? WHERE run_id = ?",
        [finished_at or dt.datetime.utcnow(), run_id],
    )
    storage_schedule_analyze(conn, "runs")


def write_trial(
    conn: DuckDBConnection,
    *,
    run_id: str,
    trial_id: str,
    stage: int,
    pair: str,
    method: str,
    params: Mapping[str, Any],
    horizon: Mapping[str, Any] | None,
    eval_time_s: float,
    sharpe_oos: float,
    maxdd: float,
    turnover: float,
    alpha_ec: float | None = None,
    half_life_full: float | None = None,
    pruned: bool = False,
) -> None:
    conn.execute(
        """
        INSERT INTO trials VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            run_id,
            trial_id,
            int(stage),
            pair,
            method,
            json.dumps(params, default=str),
            json.dumps(horizon or {}, default=str),
            float(eval_time_s),
            float(sharpe_oos),
            float(maxdd),
            float(turnover),
            alpha_ec,
            half_life_full,
            bool(pruned),
        ],
    )
    storage_schedule_analyze(conn, "trials")


def write_exec_metrics(
    conn: DuckDBConnection,
    run_id: str,
    *,
    metric_ts: dt.datetime | None = None,
    cpu_util_avg: float | None = None,
    cpu_util_p95: float | None = None,
    worker_idle_pct: float | None = None,
    chunk_size: int | None = None,
    progress_latency_s: float | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO exec_metrics VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            run_id,
            metric_ts or dt.datetime.utcnow(),
            cpu_util_avg,
            cpu_util_p95,
            worker_idle_pct,
            chunk_size,
            progress_latency_s,
        ],
    )
    storage_schedule_analyze(conn, "exec_metrics")


def write_storage_metrics(
    conn: DuckDBConnection,
    run_id: str,
    *,
    manifest_writes: int | None = None,
    write_conflicts: int | None = None,
    duckdb_query_p95_s: float | None = None,
    parquet_read_p95_s: float | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO storage_metrics VALUES (?, ?, ?, ?, ?)
        """,
        [
            run_id,
            manifest_writes,
            write_conflicts,
            duckdb_query_p95_s,
            parquet_read_p95_s,
        ],
    )
    storage_schedule_analyze(conn, "storage_metrics")


def write_trade_stats(
    conn: DuckDBConnection,
    run_id: str,
    *,
    n_trades: int,
    avg_hold_days: float,
    turnover_annualised: float,
) -> None:
    """Upsert statistik trading per run ke tabel trade_stats."""
    conn.execute(
        """
        INSERT INTO trade_stats (run_id, n_trades, avg_hold_days, turnover_annualised)
        VALUES (?, ?, ?, ?)
        ON CONFLICT (run_id) DO UPDATE SET
            n_trades = excluded.n_trades,
            avg_hold_days = excluded.avg_hold_days,
            turnover_annualised = excluded.turnover_annualised
        """,
        [
            run_id,
            int(n_trades),
            float(avg_hold_days),
            float(turnover_annualised),
        ],
    )
    storage_schedule_analyze(conn, "trade_stats")


def write_regime_stats(
    conn: DuckDBConnection,
    run_id: str,
    *,
    p_mr_mean: float,
    p_mr_inpos_mean: float,
) -> None:
    """Upsert ringkasan probabilitas regime MR per run."""

    conn.execute(
        """
        INSERT INTO regime_stats (run_id, p_mr_mean, p_mr_inpos_mean)
        VALUES (?, ?, ?)
        ON CONFLICT (run_id) DO UPDATE SET
            p_mr_mean = excluded.p_mr_mean,
            p_mr_inpos_mean = excluded.p_mr_inpos_mean
        """,
        [run_id, float(p_mr_mean), float(p_mr_inpos_mean)],
    )
    storage_schedule_analyze(conn, "regime_stats")


def write_model_checks(
    conn: DuckDBConnection,
    run_id: str,
    *,
    pair: str,
    johansen_rank: int,
    det_term: str,
    tvecm_thresholds: Mapping[str, Any] | None,
    spec_ok: bool,
) -> None:
    conn.execute(
        """
        INSERT INTO model_checks VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            run_id,
            pair,
            int(johansen_rank),
            det_term,
            json.dumps(tvecm_thresholds or {}, default=str),
            bool(spec_ok),
        ],
    )
    storage_schedule_analyze(conn, "model_checks")


def write_pareto_front(
    conn: DuckDBConnection,
    rows: Iterable[Mapping[str, Any]],
) -> None:
    if not rows:
        return
    values = [
        (row["run_id"], row["trial_id"], bool(row.get("is_nondominated", False)))
        for row in rows
    ]
    conn.executemany("INSERT INTO pareto_front VALUES (?, ?, ?)", values)
    storage_schedule_analyze(conn, "pareto_front")


def write_sd_loop(
    conn: DuckDBConnection,
    rows: Iterable[Mapping[str, Any]],
) -> None:
    if not rows:
        return
    values = [
        (
            row["run_id"],
            row["date"],
            row.get("vol_regime"),
            row.get("corr_level"),
            row.get("liquidity_score"),
            row.get("gating"),
            row.get("kelly_eff"),
            row.get("z_th_eff"),
        )
        for row in rows
    ]
    conn.executemany(
        "INSERT INTO sd_loop VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        values,
    )
    storage_schedule_analyze(conn, "sd_loop")


def write_dashboard_daily(
    conn: DuckDBConnection,
    rows: Iterable[Mapping[str, Any]],
) -> None:
    if not rows:
        return
    values = [
        (
            row["date"],
            row["run_id"],
            row.get("trials_per_hour"),
            row.get("idle_p95"),
            row.get("pruned_pct"),
            row.get("n_pareto"),
            row.get("dd_p95"),
            row.get("ttr_days"),
            row.get("duckdb_q_p95_s"),
            row.get("parquet_p95_s"),
            row.get("n_trades"),
            row.get("turnover_annualised"),
            row.get("avg_p_regime"),
            row.get("avg_abs_delta_score_pos"),
            row.get("avg_abs_delta_mom12_pos"),
            row.get("avg_delta_value_entry"),
            row.get("avg_delta_quality_entry"),
        )
        for row in rows
    ]
    conn.executemany(
        """
        INSERT INTO dashboard_daily VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?
        )
        """,
        values,
    )
    storage_schedule_analyze(conn, "dashboard_daily")


def storage_write_df(conn: DuckDBConnection, table: str, df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("DataFrame must contain at least one row")
    tmp_name = f"tmp_{table}_df"
    conn.register(tmp_name, df)
    try:
        conn.execute(
            f"INSERT INTO {_escape(table)} SELECT * FROM {_escape(tmp_name)}"
        )
    finally:
        conn.unregister(tmp_name)
    storage_schedule_analyze(conn, table)


def write_parquet_safely(
    df: pd.DataFrame,
    path: pathlib.Path,
    *,
    row_group_size: int = 100_000,
) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if pq is None or pa is None:
        df.to_parquet(path, index=False)
        return
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path, row_group_size=row_group_size)


def duckdb_copy_to(src_path: pathlib.Path, dst_path: pathlib.Path) -> None:
    src_path = pathlib.Path(src_path)
    dst_path = pathlib.Path(dst_path)
    with tempfile.TemporaryDirectory(prefix="duckdb_export_") as tmp:
        tmp_path = pathlib.Path(tmp)
        src = duckdb.connect(str(src_path), read_only=True)
        src.execute(f"EXPORT DATABASE '{tmp_path}' (FORMAT PARQUET)")
        src.close()
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if dst_path.exists():
            dst_path.unlink(missing_ok=True)
        dst = duckdb.connect(str(dst_path))
        dst.execute(f"IMPORT DATABASE '{tmp_path}'")
        dst.close()


@contextlib.contextmanager
def with_transaction(conn: DuckDBConnection) -> Iterator[DuckDBConnection]:
    conn.execute("BEGIN TRANSACTION")
    try:
        yield conn
    except Exception:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")


@contextlib.contextmanager
def managed_storage(_note: str = "") -> Iterator[DuckDBConnection]:
    conn = storage_open()
    try:
        storage_init(conn)
        yield conn
    finally:
        try:
            storage_run_maintenance(conn)
        except Exception as exc:  # pragma: no cover - defensive logging
            LOGGER.warning("Storage maintenance failed on close: %s", exc)
        conn.close()


if __name__ == "__main__":
    logger = configure_logging("storage_smoke")
    with managed_storage("smoke-test") as conn:
        write_run(conn, "smoke-run", notes="Storage smoke test")
        write_trial(
            conn,
            run_id="smoke-run",
            trial_id="baseline",
            stage=0,
            pair="AAA-BBB",
            method="vecm",
            params={"difference_lags": 1},
            horizon={"train": 100, "test": 50},
            eval_time_s=0.0,
            sharpe_oos=0.0,
            maxdd=0.0,
            turnover=0.0,
        )
        write_model_checks(
            conn,
            "smoke-run",
            pair="AAA-BBB",
            johansen_rank=1,
            det_term="co",
            tvecm_thresholds={"threshold": 0.5},
            spec_ok=True,
        )
        logger.info("Storage smoke test completed.")
