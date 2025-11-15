"""Pareto utilities mirroring the reference R implementation."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from . import storage

LOGGER = storage.configure_logging("pareto")


def is_nondominated(matrix: np.ndarray, maximize: Sequence[bool]) -> np.ndarray:
    """Return a boolean mask identifying non-dominated rows.

    The implementation mirrors the R helper: every objective is treated as a
    minimisation by flipping the sign of maximisation targets before performing
    the pairwise dominance check.
    """

    if matrix.size == 0:
        return np.zeros((0,), dtype=bool)

    values = np.asarray(matrix, dtype=float).copy()
    for j, is_max in enumerate(maximize):
        if is_max:
            values[:, j] = -values[:, j]

    n_obs = values.shape[0]
    keep = np.ones(n_obs, dtype=bool)
    for i in range(n_obs):
        if not keep[i]:
            continue
        dominated = np.all(values <= values[i], axis=1) & np.any(
            values < values[i], axis=1
        )
        dominated[i] = False
        keep[dominated] = False
    return keep


def compute_pareto_nds(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the non-dominated subset using Sharpe/MAXDD/turnover targets."""

    if df.empty:
        return df

    required_cols = {"sharpe_oos", "maxdd", "turnover"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise KeyError(f"Missing columns for Pareto evaluation: {sorted(missing)}")

    clean = df.dropna(subset=required_cols)
    if clean.empty:
        LOGGER.warning("No finite objective rows available for Pareto frontier")
        return clean

    objectives = clean.loc[:, ["sharpe_oos", "maxdd", "turnover"]].to_numpy()
    mask = is_nondominated(objectives, maximize=(True, False, False))
    return clean.loc[mask].copy()


def write_pareto_front(conn, run_id: str) -> pd.DataFrame:
    """Persist the Pareto frontier rows into the ``pareto_front`` table."""

    trials = conn.execute(
        """
        SELECT run_id, trial_id, sharpe_oos, maxdd, turnover
        FROM trials
        WHERE run_id = ?
        """,
        [run_id],
    ).fetch_df()

    frontier = compute_pareto_nds(trials)
    if frontier.empty:
        LOGGER.info("No Pareto rows computed for run %s", run_id)
        return frontier

    payload = pd.DataFrame(
        {
            "run_id": [run_id] * len(frontier),
            "trial_id": frontier["trial_id"].to_list(),
            "is_nondominated": [True] * len(frontier),
        }
    )

    with storage.with_transaction(conn):
        conn.execute("DELETE FROM pareto_front WHERE run_id = ?", [run_id])
        rows = list(payload.itertuples(index=False, name=None))
        if rows:
            conn.executemany(
                "INSERT INTO pareto_front VALUES (?, ?, ?)",
                rows,
            )
    storage.storage_schedule_analyze(conn, "pareto_front")
    LOGGER.info("Stored %d Pareto rows for run %s", len(payload), run_id)
    return frontier


def run_pareto_update(run_id: str) -> None:
    with storage.managed_storage("pareto") as conn:
        write_pareto_front(conn, run_id)


if __name__ == "__main__":
    try:
        run_pareto_update("demo-run")
    except Exception as exc:
        LOGGER.error("Pareto computation failed: %s", exc)
