from __future__ import annotations

import contextlib

import pandas as pd

import vecm_project.run_demo as run_demo


class DummyTrial:
    number = 0
    value = 0.0
    user_attrs = {"record": {"diagnostics": {}}}


class DummyStudy:
    best_trial = DummyTrial()


def _patch_run_demo_dependencies(monkeypatch) -> None:
    monkeypatch.setattr(run_demo, "ensure_price_data", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        run_demo,
        "load_cached_prices",
        lambda *_args, **_kwargs: pd.DataFrame({"Date": [0], "AdjClose": [1.0]}),
    )
    monkeypatch.setattr(run_demo, "playbook_score_once", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(run_demo, "run_bo", lambda *_args, **_kwargs: DummyStudy())
    monkeypatch.setattr(run_demo, "dashboard_aggregate", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(run_demo, "write_pareto_front", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        run_demo,
        "storage",
        type("StorageStub", (), {"managed_storage": lambda *_args, **_kwargs: contextlib.nullcontext()}),
    )


def test_run_demo_smoke(monkeypatch) -> None:
    _patch_run_demo_dependencies(monkeypatch)
    run_demo.main(["--pair", "AAA,BBB", "--iters", "1", "--n-init", "1"])


def test_run_demo_refresh_forces_price_refresh(monkeypatch) -> None:
    calls = {"kwargs": None}

    def _capture_refresh(*_args, **kwargs):
        calls["kwargs"] = kwargs

    _patch_run_demo_dependencies(monkeypatch)
    monkeypatch.setattr(run_demo, "ensure_price_data", _capture_refresh)

    run_demo.main(["--pair", "AAA,BBB", "--iters", "1", "--n-init", "1", "--refresh"])

    assert calls["kwargs"] is not None
    assert calls["kwargs"].get("force_refresh") is True
