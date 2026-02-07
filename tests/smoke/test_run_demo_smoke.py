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


def test_run_demo_smoke(monkeypatch) -> None:
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

    run_demo.main(["--pair", "AAA,BBB", "--iters", "1", "--n-init", "1"])
