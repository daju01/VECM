from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from vecm_project.scripts.ops import extreme_sharpe_scan as scan


def test_compute_sharpe_full_matches_expected_formula() -> None:
    ret = pd.Series([0.01, 0.01, -0.005], dtype=float)
    expected = float((ret.mean() / ret.std()) * math.sqrt(252))
    assert scan.compute_sharpe_full(ret, ann_days=252) == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_compute_sharpe_full_returns_zero_when_std_is_zero() -> None:
    ret = pd.Series([0.0, 0.0, 0.0], dtype=float)
    assert scan.compute_sharpe_full(ret, ann_days=252) == 0.0


def test_rank_extremes_filters_invalid_and_returns_top_sides() -> None:
    df = pd.DataFrame(
        [
            {"run_id": "a", "sharpe_full": 1.2, "n_trades": 12, "maxdd": -0.10},
            {"run_id": "b", "sharpe_full": -2.3, "n_trades": 15, "maxdd": -0.20},
            {"run_id": "c", "sharpe_full": 0.4, "n_trades": 7, "maxdd": -0.05},
            {"run_id": "d", "sharpe_full": 0.7, "n_trades": 13, "maxdd": float("nan")},
            {"run_id": "e", "sharpe_full": -1.1, "n_trades": 11, "maxdd": -0.12},
        ]
    )
    valid = scan.filter_valid_candidates(df, min_trades=10)
    top_pos, top_neg = scan.rank_extremes(valid, top_k=2)

    assert set(valid["run_id"]) == {"a", "b", "e"}
    assert top_pos["run_id"].tolist() == ["a", "e"]
    assert top_neg["run_id"].tolist() == ["b", "e"]


def test_run_extreme_scan_smoke_writes_outputs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def _fake_prepare_feature_cache(pair, base_cfg, combos, scan_id):  # noqa: ANN001
        return {combo: object() for combo in combos}

    def _fake_evaluate_candidate(pair, candidate, feature_result, ann_days, scan_id, index):  # noqa: ANN001
        sharpe_full = 1.0 - 0.1 * index
        return {
            "run_id": f"{scan_id}_{index:04d}",
            "pair": pair,
            "focus": candidate.focus,
            "sharpe_full": sharpe_full,
            "sharpe_oos": sharpe_full - 0.05,
            "n_trades": 12 + (index % 3),
            "cagr": sharpe_full / 10.0,
            "maxdd": -0.1,
            "turnover_annualised": 6.0,
            "long_only": candidate.long_only,
            "short_filter": candidate.short_filter,
            "gate_enforce": candidate.gate_enforce,
            "z_entry": candidate.z_entry,
            "z_exit": candidate.z_exit,
            "max_hold": candidate.max_hold,
            "cooldown": candidate.cooldown,
            "p_th": candidate.p_th,
            "error": "",
        }

    monkeypatch.setattr(scan, "_prepare_feature_cache", _fake_prepare_feature_cache)
    monkeypatch.setattr(scan, "_evaluate_candidate", _fake_evaluate_candidate)

    artifacts = scan.run_extreme_scan(
        pair="ANTM.JK,TLKM.JK",
        n_runs=20,
        mode=scan.MODE_FULL_SAMPLE,
        signal_mode="normal",
        min_trades=10,
        top_k=10,
        seed=42,
        out_prefix=str(tmp_path / "extreme_sharpe"),
        input_file=None,
    )

    assert artifacts.total_runs == 20
    assert artifacts.candidates_path.exists()
    assert artifacts.summary_path.exists()

    candidates_df = pd.read_csv(artifacts.candidates_path)
    assert candidates_df.shape[0] == 20

    summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
    assert summary["n_runs"] == 20
    assert len(summary["top_positive"]) == 10
    assert len(summary["top_negative"]) == 10
