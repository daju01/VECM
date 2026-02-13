from __future__ import annotations

import pandas as pd

from vecm_project.scripts.playbook_types import PlaybookConfig
from vecm_project.scripts.playbook_vecm import build_signals


def _cfg(*, signal_mode: str) -> PlaybookConfig:
    return PlaybookConfig(
        input_file="dummy.csv",
        signal_mode=signal_mode,
        long_only=False,
        gate_enforce=True,
        short_filter=False,
        regime_confirm=1,
        cooldown=0,
        z_auto_method="quantile",
        z_auto_q=0.5,
        z_entry=1.0,
        z_exit=0.55,
        p_th=0.5,
    )


def test_long_from_short_only_uses_legacy_short_as_long_and_drops_short() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    zect = pd.Series([-2.0, -0.2, 1.8, 2.1], index=idx)
    gates = pd.Series(True, index=idx)
    p_regime = pd.Series(1.0, index=idx)

    signals, _ = build_signals(zect, _cfg(signal_mode="long_from_short_only"), gates, p_regime=p_regime)

    # Legacy short would fire on positive z tail (last two rows), now mapped to long.
    assert signals["long"].tolist() == [0.0, 0.0, 1.0, 1.0]
    assert signals["short"].tolist() == [0.0, 0.0, 0.0, 0.0]


def test_normal_mode_keeps_original_direction() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    zect = pd.Series([-2.0, -0.2, 1.8, 2.1], index=idx)
    gates = pd.Series(True, index=idx)
    p_regime = pd.Series(1.0, index=idx)

    signals, _ = build_signals(zect, _cfg(signal_mode="normal"), gates, p_regime=p_regime)

    assert signals["long"].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert signals["short"].tolist() == [0.0, 0.0, 1.0, 1.0]
