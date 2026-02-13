from __future__ import annotations

import pandas as pd

from vecm_project.dashboard import app as dashboard_app
from vecm_project.scripts.playbook_types import PlaybookConfig


def test_load_backtest_defaults_uses_manifest_z_th_when_z_entry_missing(monkeypatch) -> None:
    manifest_row = pd.Series({"pair": "ANTM.JK~TLKM.JK", "z_th": 0.85})
    params = {
        "input_file": "vecm_project/data/adj_close_data.csv",
        "subset": "ANTM.JK,TLKM.JK",
        "z_entry": None,
        "z_entry_cap": 0.85,
        "z_exit": 0.55,
        "max_hold": 8,
        "cooldown": 1,
        "p_th": 0.5,
    }
    monkeypatch.setattr(
        dashboard_app,
        "_latest_backtest_context",
        lambda: ("run-1", manifest_row, params),
    )

    defaults = dashboard_app._load_backtest_defaults()

    assert defaults.pair == "ANTM.JK,TLKM.JK"
    assert defaults.z_entry == 0.85


def test_build_whatif_decision_params_falls_back_to_z_entry_cap() -> None:
    cfg = PlaybookConfig(
        input_file="vecm_project/data/adj_close_data.csv",
        subset="ANTM.JK,TLKM.JK",
        z_entry=None,
        z_entry_cap=0.85,
        z_exit=0.55,
        max_hold=8,
        cooldown=1,
        p_th=0.5,
    )

    decision = dashboard_app._build_whatif_decision_params({}, cfg)

    assert decision.z_entry == 0.85
