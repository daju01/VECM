"""Glue utilities for exporting playbook artefacts and registering outputs.

This module mirrors the behaviour of the reference R `vecm_hooks.R` script by
providing lightweight helpers to

* spawn external exporter / visualiser scripts (Python implementations by
  default, with optional custom overrides),
* record generated files inside DuckDB, and
* locate the ``master_*`` artefacts that downstream tooling expects.

The helpers purposely keep the R naming scheme (``vh_*``) so the Python code
remains drop-in compatible with documentation and notebooks that were written
for the original stack.
"""

from __future__ import annotations

import contextlib
import pathlib
import subprocess
import sys
from typing import Dict, Iterable, List, Mapping, Optional

import duckdb

from . import storage

LOGGER = storage.configure_logging("vecm_hooks")
BASE_DIR = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = BASE_DIR / "out_ms"
DEFAULT_EXPORTER = BASE_DIR / "vecm_master_exports.py"
DEFAULT_VISUALISER = BASE_DIR / "read_master_and_visualize.py"


def _normalise(path: pathlib.Path | str) -> str:
    return str(pathlib.Path(path).expanduser().resolve())


def vh_open_db(db_path: Optional[pathlib.Path | str] = None) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB connection.

    The default path mirrors ``storage.storage_open`` (``out/db/vecm.duckdb``),
    but callers can provide a custom ``db_path`` to keep parity with the R
    utilities that store exports under ``out/meta/results.duckdb``.
    """

    if db_path is None:
        return storage.storage_open()

    db_path = pathlib.Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.debug("Opening DuckDB at %s", db_path)
    conn = duckdb.connect(str(db_path), read_only=False)
    conn.execute("PRAGMA busy_timeout=60000")
    return conn


def vh_init_db(conn: duckdb.DuckDBPyConnection) -> None:
    """Ensure the ``exports`` table exists alongside the core schema."""

    storage.storage_init(conn)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS exports (
            run_id TEXT,
            out_dir TEXT,
            daily TEXT,
            orders TEXT,
            trades TEXT,
            exported_at TIMESTAMP
        );
        """
    )
    storage.storage_create_index(conn, "exports", ("run_id",))


def vh_register_export(
    conn: duckdb.DuckDBPyConnection,
    run_id: str,
    out_dir: pathlib.Path | str,
    daily_path: pathlib.Path | str,
    orders_path: pathlib.Path | str,
    trades_path: pathlib.Path | str,
) -> Mapping[str, str]:
    """Record exported artefacts in DuckDB and return the registered row."""

    if not run_id:
        raise ValueError("run_id must be a non-empty string")

    row = {
        "run_id": run_id,
        "out_dir": _normalise(out_dir),
        "daily": _normalise(daily_path),
        "orders": _normalise(orders_path),
        "trades": _normalise(trades_path),
    }
    LOGGER.info("Registering export for run %s", run_id)
    conn.execute(
        """
        INSERT INTO exports (run_id, out_dir, daily, orders, trades, exported_at)
        VALUES (?, ?, ?, ?, ?, NOW());
        """,
        [
            row["run_id"],
            row["out_dir"],
            row["daily"],
            row["orders"],
            row["trades"],
        ],
    )
    storage.storage_analyze(conn)
    return row


def vh_find_master_files(run_id: str, out_dir: pathlib.Path | str = DEFAULT_OUT_DIR) -> Dict[str, pathlib.Path]:
    """Return expected master file paths for ``run_id`` within ``out_dir``."""

    if not run_id:
        raise ValueError("run_id must be provided")

    base = pathlib.Path(out_dir)
    files = {
        "daily": base / f"master_daily_{run_id}.csv",
        "orders": base / f"master_orders_{run_id}.csv",
        "trades": base / f"master_trades_{run_id}.csv",
    }
    return files


def _build_cli_args(pairs: Iterable[str]) -> List[str]:
    return [str(arg) for arg in pairs if arg is not None]


def vh_run_export(
    run_id: str,
    run_dir: pathlib.Path | str = DEFAULT_OUT_DIR,
    out_dir: pathlib.Path | str = DEFAULT_OUT_DIR,
    prices: Optional[pathlib.Path | str] = None,
    capital: Optional[float] = 100_000_000.0,
    lot_size: Optional[int] = 100,
    rscript: str = sys.executable,
    exporter_path: pathlib.Path | str = DEFAULT_EXPORTER,
    db_path: Optional[pathlib.Path | str] = None,
) -> Mapping[str, object]:
    """Run the external exporter and register outputs.

    ``exporter_path`` defaults to the bundled Python exporter so existing
    documentation for the R stack continues to map one-to-one onto this
    implementation.  Callers can override the executable path if required.
    """

    exporter_path = pathlib.Path(exporter_path)
    if not exporter_path.exists():
        raise FileNotFoundError(f"Exporter not found: {exporter_path}")
    if not run_id:
        raise ValueError("run_id must be provided")

    run_dir = pathlib.Path(run_dir)
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    args: List[str] = [
        f"--run_id={run_id}",
        f"--run_dir={_normalise(run_dir)}",
        f"--out_dir={_normalise(out_dir)}",
    ]
    if prices is not None:
        args.append(f"--prices={_normalise(prices)}")
    if capital is not None:
        args.append(f"--capital={capital:.0f}")
    if lot_size is not None:
        args.append(f"--lot_size={int(lot_size)}")

    command = _build_cli_args([rscript, str(exporter_path), *args])
    LOGGER.info("Running exporter: %s", " ".join(command))
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        LOGGER.error("Exporter failed: %s", result.stderr)
        raise RuntimeError(f"Exporter failed with code {result.returncode}")

    files = vh_find_master_files(run_id, out_dir)
    missing = [name for name, path in files.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing exported files: " + ", ".join(f"{name} ({files[name]})" for name in missing)
        )

    if db_path is not None:
        with contextlib.closing(vh_open_db(db_path)) as conn:
            vh_init_db(conn)
            vh_register_export(conn, run_id, out_dir, files["daily"], files["orders"], files["trades"])

    return {"status": result.returncode, "files": files}


def vh_run_visualize(
    run_id: str,
    out_dir: pathlib.Path | str = DEFAULT_OUT_DIR,
    python_executable: str = sys.executable,
    viz_path: pathlib.Path | str = DEFAULT_VISUALISER,
    save_path: Optional[pathlib.Path | str] = None,
) -> Mapping[str, object]:
    """Invoke the external visualiser for a ``run_id``."""

    viz_path = pathlib.Path(viz_path)
    if not viz_path.exists():
        raise FileNotFoundError(f"Visualizer not found: {viz_path}")

    files = vh_find_master_files(run_id, out_dir)
    missing = [name for name, path in files.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing master files for visualisation: "
            + ", ".join(f"{name} ({files[name]})" for name in missing)
        )

    if save_path is None:
        save_path = pathlib.Path(out_dir) / f"visual_{run_id}.png"
    else:
        save_path = pathlib.Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    args: List[str] = [
        f"--run_id={run_id}",
        f"--out_dir={_normalise(out_dir)}",
        f"--save_path={_normalise(save_path)}",
    ]

    command = _build_cli_args([python_executable, str(viz_path), *args])
    LOGGER.info("Running visualiser: %s", " ".join(command))
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        LOGGER.error("Visualizer failed: %s", result.stderr)
        raise RuntimeError(f"Visualizer failed with code {result.returncode}")

    return {"status": result.returncode, "plotted": True}


def vh_export_many(
    winners: Iterable[str],
    run_dir: pathlib.Path | str = DEFAULT_OUT_DIR,
    out_dir: pathlib.Path | str = DEFAULT_OUT_DIR,
    prices: Optional[pathlib.Path | str] = None,
    capital: Optional[float] = 100_000_000.0,
    lot_size: Optional[int] = 100,
    db_path: Optional[pathlib.Path | str] = None,
    python_executable: str = sys.executable,
    exporter_path: pathlib.Path | str = DEFAULT_EXPORTER,
) -> List[Mapping[str, object]]:
    """Execute exports for multiple run identifiers."""

    winners = list(winners)
    if not winners:
        raise ValueError("winners must contain at least one run_id")

    results = []
    for run_id in winners:
        results.append(
            vh_run_export(
                run_id=run_id,
                run_dir=run_dir,
                out_dir=out_dir,
                prices=prices,
                capital=capital,
                lot_size=lot_size,
                db_path=db_path,
                rscript=python_executable,
                exporter_path=exporter_path,
            )
        )
    return results


if __name__ == "__main__":  # pragma: no cover - smoke test
    try:
        with storage.managed_storage("vecm_hooks_smoke") as conn:
            vh_init_db(conn)
            LOGGER.info("Existing exports rows: %s", conn.execute("SELECT COUNT(*) FROM exports").fetchone()[0])
    except Exception as exc:  # pragma: no cover - diagnostic logging only
        LOGGER.error("vecm_hooks smoke test failed: %s", exc)
