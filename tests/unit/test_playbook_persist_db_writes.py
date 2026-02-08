from __future__ import annotations

import contextlib
import tempfile
from pathlib import Path

import pandas as pd

from vecm_project.scripts.playbook_types import PlaybookConfig
from vecm_project.scripts.playbook_vecm import ExecutionResult, persist_artifacts


def test_persist_artifacts_writes_model_and_regime_stats_once(monkeypatch) -> None:
    calls = {"model_checks": 0, "regime_stats": 0}

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "out_ms"

        monkeypatch.setattr("vecm_project.scripts.playbook_vecm.OUT_DIR", out_dir)
        monkeypatch.setattr(
            "vecm_project.scripts.playbook_vecm.storage.managed_storage",
            lambda *_args, **_kwargs: contextlib.nullcontext(object()),
        )
        monkeypatch.setattr(
            "vecm_project.scripts.playbook_vecm.storage.with_transaction",
            lambda conn: contextlib.nullcontext(conn),
        )
        monkeypatch.setattr(
            "vecm_project.scripts.playbook_vecm.storage.write_run_metrics",
            lambda *_args, **_kwargs: None,
        )

        def _write_model_checks(*_args, **_kwargs):
            calls["model_checks"] += 1

        def _write_regime_stats(*_args, **_kwargs):
            calls["regime_stats"] += 1

        monkeypatch.setattr(
            "vecm_project.scripts.playbook_vecm.storage.write_model_checks",
            _write_model_checks,
        )
        monkeypatch.setattr(
            "vecm_project.scripts.playbook_vecm.storage.write_regime_stats",
            _write_regime_stats,
        )

        idx = pd.date_range("2024-01-01", periods=3, freq="D")
        zeros = pd.Series(0.0, index=idx)
        exec_res = ExecutionResult(
            pos=zeros.rename("pos"),
            ret=zeros.rename("ret"),
            ret_core=zeros.rename("ret_core"),
            cost=zeros.rename("cost"),
            trades=pd.DataFrame(),
        )
        cfg = PlaybookConfig(input_file="dummy.csv", subset="AAA,BBB", tag="unit-test")
        result = {
            "metrics": {"z_th": 0.8, "sharpe_oos": 0.0, "maxdd": 0.0, "cagr": 0.0},
            "model_checks": {"pair": "AAA~BBB", "spec_ok": False},
            "execution": exec_res,
        }

        persist_artifacts("run-test", cfg, result)

    assert calls["model_checks"] == 1
    assert calls["regime_stats"] == 1
