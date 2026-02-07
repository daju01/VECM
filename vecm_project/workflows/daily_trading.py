from __future__ import annotations

"""Prefect-based daily trading workflow.

This module is optional and only runs if Prefect is installed.
"""

import importlib
import datetime as dt
from typing import Any, Dict, List

from vecm_project.scripts import data_streaming, daily_signal, notify

if importlib.util.find_spec("prefect") is None:
    def flow(*_args, **_kwargs):  # type: ignore[misc]
        def decorator(func):
            def wrapper(*_a, **_k):
                raise RuntimeError(
                    "Prefect is required to run this workflow. Install it via requirements-optional.txt"
                )

            return wrapper

        return decorator

    def task(*_args, **_kwargs):  # type: ignore[misc]
        def decorator(func):
            return func

        return decorator
else:
    prefect = importlib.import_module("prefect")
    flow = getattr(prefect, "flow")
    task = getattr(prefect, "task")


@task
def data_refresh() -> None:
    data_streaming.ensure_price_data(force_refresh=False)


@task
def signal_generation() -> List[Dict[str, Any]]:
    return daily_signal.run_daily_signals()


@task
def risk_check(signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return signals


@task
def notification() -> None:
    notify.main()


@flow(name="daily_trading")
def daily_trading_flow() -> None:
    data_refresh()
    signals = signal_generation()
    risk_check(signals)
    notification()


if __name__ == "__main__":  # pragma: no cover
    daily_trading_flow()
