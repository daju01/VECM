"""System-dynamics style gating loop mirroring the reference R template."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from . import storage


LOGGER = storage.configure_logging("sd_loop")


@dataclass(frozen=True)
class SDParams:
    """Parameters for the simple ODE system."""

    a1: float = 0.3
    b1: float = 0.2
    c1: float = 0.25
    a2: float = 0.4
    a3: float = 0.3
    targetR: float = 0.7

    def as_dict(self) -> Mapping[str, float]:
        return {
            "a1": self.a1,
            "b1": self.b1,
            "c1": self.c1,
            "a2": self.a2,
            "a3": self.a3,
            "targetR": self.targetR,
        }


DEFAULT_INIT = np.array([0.6, 0.5, 0.5], dtype=float)


def _sd_model(
    state: np.ndarray,
    params: Mapping[str, float],
    *,
    signal_quality: float,
    vol_spike: float,
    liquidity_obs: float,
    perf_recent: float,
) -> np.ndarray:
    """Compute derivatives for the Risk/Liquidity/Confidence system."""

    r, l, c = state
    a1 = params["a1"]
    b1 = params["b1"]
    c1 = params["c1"]
    a2 = params["a2"]
    a3 = params["a3"]
    target_r = params["targetR"]

    dr = a1 * (target_r - r) + b1 * signal_quality - c1 * vol_spike
    dl = a2 * (liquidity_obs - l)
    dc = a3 * (perf_recent - c)
    return np.array([dr, dl, dc], dtype=float)


def _rk4_step(
    state: np.ndarray,
    params: Mapping[str, float],
    *,
    signal_quality: float,
    vol_spike: float,
    liquidity_obs: float,
    perf_recent: float,
    dt: float = 1.0,
) -> np.ndarray:
    """Perform a single Runge–Kutta 4 integration step."""

    k1 = _sd_model(
        state,
        params,
        signal_quality=signal_quality,
        vol_spike=vol_spike,
        liquidity_obs=liquidity_obs,
        perf_recent=perf_recent,
    )
    k2 = _sd_model(
        state + 0.5 * dt * k1,
        params,
        signal_quality=signal_quality,
        vol_spike=vol_spike,
        liquidity_obs=liquidity_obs,
        perf_recent=perf_recent,
    )
    k3 = _sd_model(
        state + 0.5 * dt * k2,
        params,
        signal_quality=signal_quality,
        vol_spike=vol_spike,
        liquidity_obs=liquidity_obs,
        perf_recent=perf_recent,
    )
    k4 = _sd_model(
        state + dt * k3,
        params,
        signal_quality=signal_quality,
        vol_spike=vol_spike,
        liquidity_obs=liquidity_obs,
        perf_recent=perf_recent,
    )
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def run_sd_simulation(
    dates: Sequence[dt.date],
    inputs: pd.DataFrame,
    *,
    params: Optional[Mapping[str, float]] = None,
    init_state: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    """Simulate the simple system dynamics loop.

    Parameters mirror the shared R reference: the input frame is expected to
    contain the columns ``signal_quality``, ``vol_spike``, ``liquidity_obs`` and
    ``perf_recent``. Additional contextual columns (``vol_regime`` etc.) are
    preserved and returned alongside the simulated states.
    """

    if len(dates) != len(inputs):
        raise ValueError("dates and inputs length mismatch")

    params = dict(SDParams().as_dict(), **(params or {}))
    state = np.array(init_state if init_state is not None else DEFAULT_INIT, dtype=float)
    if state.shape != (3,):
        raise ValueError("init_state must provide three elements (R, L, C)")

    input_frame = inputs.reset_index(drop=True).copy()
    for col in ("signal_quality", "vol_spike", "liquidity_obs", "perf_recent"):
        if col not in input_frame:
            input_frame[col] = 0.0
        input_frame[col] = input_frame[col].fillna(0.0).astype(float)

    records: list[np.ndarray] = []
    for idx, row in input_frame.iterrows():
        state = _rk4_step(
            state,
            params,
            signal_quality=float(row.signal_quality),
            vol_spike=float(row.vol_spike),
            liquidity_obs=float(row.liquidity_obs),
            perf_recent=float(row.perf_recent),
        )
        records.append(state.copy())
        LOGGER.debug("step=%s state=%s", idx, state)

    state_frame = pd.DataFrame(records, columns=["R", "L", "C"])
    state_frame.insert(0, "date", pd.to_datetime(dates))
    return pd.concat([state_frame, input_frame], axis=1)


def policy_map(states: pd.DataFrame) -> pd.DataFrame:
    """Map Risk/Liquidity/Confidence states to gating controls."""

    lin = 0.5 * states["R"] + 0.3 * states["L"] + 0.2 * states["C"]
    gating = 1.0 / (1.0 + np.exp(-4.0 * (lin - 0.5)))
    gating = gating.clip(0.0, 1.0)
    kelly_eff = gating.clip(0.1, 1.0)
    z_th_eff = (1.2 - 0.5 * gating).clip(0.3, 1.5)
    return pd.DataFrame(
        {
            "gating": gating,
            "kelly_eff": kelly_eff,
            "z_th_eff": z_th_eff,
        }
    )


def build_policy(
    dates: Sequence[dt.date],
    inputs: pd.DataFrame,
    *,
    params: Optional[Mapping[str, float]] = None,
    init_state: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    """Convenience wrapper that returns states and mapped policy controls."""

    sim = run_sd_simulation(dates, inputs, params=params, init_state=init_state)
    mapped = policy_map(sim[["R", "L", "C"]])
    return pd.concat([sim, mapped], axis=1)


def persist_policy(
    run_id: str,
    policy_df: pd.DataFrame,
    *,
    conn: Optional[storage.duckdb.DuckDBPyConnection] = None,
) -> None:
    """Write the policy observations to DuckDB."""

    if policy_df.empty:
        raise ValueError("policy_df must contain at least one row")

    close_conn = False
    if conn is None:
        conn = storage.storage_open()
        close_conn = True

    try:
        storage.storage_init(conn)
        n = len(policy_df)
        payload = pd.DataFrame(
            {
                "run_id": run_id,
                "date": pd.to_datetime(policy_df["date"]).dt.date,
                "vol_regime": policy_df.get(
                    "vol_regime", pd.Series(["neutral"] * n)
                ),
                "corr_level": policy_df.get(
                    "corr_level", pd.Series([np.nan] * n, dtype=float)
                ),
                "liquidity_score": policy_df.get(
                    "liquidity_obs", pd.Series([np.nan] * n, dtype=float)
                ),
                "gating": policy_df["gating"],
                "kelly_eff": policy_df["kelly_eff"],
                "z_th_eff": policy_df["z_th_eff"],
            }
        )
        storage.storage_write_df(conn, "sd_loop", payload)
        storage.storage_schedule_analyze(conn, "sd_loop")
    finally:
        if close_conn:
            conn.close()


def run_sd_loop(
    run_id: str,
    dates: Sequence[dt.date],
    inputs: pd.DataFrame,
    *,
    params: Optional[Mapping[str, float]] = None,
    init_state: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    """High-level helper: build policy then persist it."""

    policy = build_policy(dates, inputs, params=params, init_state=init_state)
    persist_policy(run_id, policy)
    return policy


def _example_inputs(days: int = 10) -> tuple[list[dt.date], pd.DataFrame]:
    base = dt.date.today() - dt.timedelta(days=days)
    dates = [base + dt.timedelta(days=i + 1) for i in range(days)]
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(
        {
            "signal_quality": rng.uniform(0.0, 1.0, size=days),
            "vol_spike": rng.uniform(0.0, 1.0, size=days),
            "liquidity_obs": rng.uniform(0.3, 0.9, size=days),
            "perf_recent": rng.uniform(0.0, 1.0, size=days),
            "vol_regime": rng.choice(["low", "medium", "high"], size=days),
            "corr_level": rng.normal(0.5, 0.1, size=days),
        }
    )
    return dates, frame


if __name__ == "__main__":
    dates, features = _example_inputs()
    policy = run_sd_loop("demo_sd_loop", dates, features)
    LOGGER.info("Persisted %s policy rows", len(policy))
