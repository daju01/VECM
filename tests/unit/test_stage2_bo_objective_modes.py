from __future__ import annotations

import datetime as dt
from types import SimpleNamespace

import pandas as pd
import pytest

from vecm_project.scripts.playbook_types import DecisionParams
from vecm_project.scripts import stage2_bo


def _feature_result(*, long_only: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        features=SimpleNamespace(
            cfg=SimpleNamespace(long_only=long_only),
            oos_start_date=dt.date(2024, 1, 1),
        )
    )


def _decision_params() -> DecisionParams:
    return DecisionParams(
        z_entry=1.0,
        z_exit=0.5,
        max_hold=5,
        cooldown=1,
        p_th=0.5,
        run_id="unit",
    )


def test_idx_v1_dd_hinge_penalty_applies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STAGE2_OBJ_MODE", "idx_v1")
    monkeypatch.setenv("IDX_DD_CAP", "0.15")
    monkeypatch.setenv("IDX_LAMBDA_DD", "4.0")
    monkeypatch.setenv("IDX_LAMBDA_CAP", "0")
    monkeypatch.setenv("IDX_LAMBDA_ILLIQ", "0")
    monkeypatch.setenv("IDX_LAMBDA_TO", "0")
    monkeypatch.setenv("STAGE2_MIN_TRADES", "1")

    def _fake_eval(*_args, **_kwargs):
        return {
            "metrics": {
                "sharpe_oos": 1.2,
                "sharpe_oos_net": 1.2,
                "maxdd_oos_net": 0.25,
                "turnover_annualised_notional": 0.0,
                "participation_mean": 0.0,
                "amihud_illiq": 0.0,
                "illiq_cap": 1.0,
                "n_trades": 10,
            },
            "execution": SimpleNamespace(pos=pd.Series([0.0, 1.0])),
        }

    monkeypatch.setattr(stage2_bo, "evaluate_rules", _fake_eval)
    diagnostics = stage2_bo.score_rules(feature_result=_feature_result(), decision_params=_decision_params())
    expected = 1.2 - 4.0 * (0.25 - 0.15) ** 2
    assert diagnostics["Score"] == pytest.approx(expected, abs=1e-12)


def test_idx_v2_uses_cagr_when_negative(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STAGE2_OBJ_MODE", "idx_v2_calmar")
    monkeypatch.setenv("IDX_LAMBDA_DD", "0")
    monkeypatch.setenv("IDX_LAMBDA_CAP", "0")
    monkeypatch.setenv("IDX_LAMBDA_ILLIQ", "0")
    monkeypatch.setenv("IDX_LAMBDA_TO", "0")
    monkeypatch.setenv("STAGE2_MIN_TRADES", "1")

    def _fake_eval(*_args, **_kwargs):
        return {
            "metrics": {
                "sharpe_oos": -1.0,
                "cagr_oos_net": -0.05,
                "calmar_oos_net": 9.99,
                "maxdd_oos_net": 0.01,
                "turnover_annualised_notional": 0.0,
                "participation_mean": 0.0,
                "amihud_illiq": 0.0,
                "illiq_cap": 1.0,
                "n_trades": 10,
            },
            "execution": SimpleNamespace(pos=pd.Series([0.0, 1.0])),
        }

    monkeypatch.setattr(stage2_bo, "evaluate_rules", _fake_eval)
    diagnostics = stage2_bo.score_rules(feature_result=_feature_result(), decision_params=_decision_params())
    assert diagnostics["Score"] == pytest.approx(-0.05, abs=1e-12)


def test_min_trades_penalty_applies_in_idx_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STAGE2_OBJ_MODE", "idx_v1")
    monkeypatch.setenv("STAGE2_MIN_TRADES", "5")
    monkeypatch.setenv("STAGE2_HUGE_PENALTY", "1000")
    monkeypatch.setenv("IDX_LAMBDA_DD", "0")
    monkeypatch.setenv("IDX_LAMBDA_CAP", "0")
    monkeypatch.setenv("IDX_LAMBDA_ILLIQ", "0")
    monkeypatch.setenv("IDX_LAMBDA_TO", "0")

    def _fake_eval(*_args, **_kwargs):
        return {
            "metrics": {
                "sharpe_oos_net": 1.0,
                "maxdd_oos_net": 0.05,
                "turnover_annualised_notional": 0.0,
                "participation_mean": 0.0,
                "amihud_illiq": 0.0,
                "illiq_cap": 1.0,
                "n_trades": 2,
            },
            "execution": SimpleNamespace(pos=pd.Series([0.0, 1.0])),
        }

    monkeypatch.setattr(stage2_bo, "evaluate_rules", _fake_eval)
    diagnostics = stage2_bo.score_rules(feature_result=_feature_result(), decision_params=_decision_params())
    assert diagnostics["min_trades_penalty"] == pytest.approx(1003.0, abs=1e-12)
    assert diagnostics["Score"] == pytest.approx(1.0 - 1003.0, abs=1e-12)


def test_short_feasibility_penalty_applies_when_short_not_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STAGE2_OBJ_MODE", "idx_v1")
    monkeypatch.setenv("STAGE2_HUGE_PENALTY", "500")
    monkeypatch.setenv("STAGE2_MIN_TRADES", "1")
    monkeypatch.setenv("IDX_LAMBDA_DD", "0")
    monkeypatch.setenv("IDX_LAMBDA_CAP", "0")
    monkeypatch.setenv("IDX_LAMBDA_ILLIQ", "0")
    monkeypatch.setenv("IDX_LAMBDA_TO", "0")

    def _fake_eval(*_args, **_kwargs):
        idx = pd.to_datetime(["2024-01-01", "2024-01-02"])
        return {
            "metrics": {
                "sharpe_oos_net": 1.0,
                "maxdd_oos_net": 0.01,
                "turnover_annualised_notional": 0.0,
                "participation_mean": 0.0,
                "amihud_illiq": 0.0,
                "illiq_cap": 1.0,
                "n_trades": 10,
            },
            "execution": SimpleNamespace(pos=pd.Series([0.0, -1.0], index=idx)),
        }

    monkeypatch.setattr(stage2_bo, "evaluate_rules", _fake_eval)
    diagnostics = stage2_bo.score_rules(feature_result=_feature_result(long_only=True), decision_params=_decision_params())
    assert diagnostics["penalty_short"] == pytest.approx(500.0, abs=1e-12)
    assert diagnostics["Score"] == pytest.approx(1.0 - 500.0, abs=1e-12)
