from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from vecm_project.scripts import short_term_signals as sts


def test_short_term_cache_ignores_non_serializable_attrs() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir) / "short_term_cache"
        idx = pd.date_range("2024-01-01", periods=3, freq="D")
        panel = pd.DataFrame({"AAA.JK": [0.1, 0.2, 0.3]}, index=idx)
        z_mom12 = pd.DataFrame({"AAA.JK": [1.0, 1.1, 1.2]}, index=idx)

        # This DataFrame attr is not parquet-serializable and used to break cache writes.
        panel.attrs["z_mom12"] = z_mom12
        panel.attrs["ml_prob"] = z_mom12.copy()

        sts._save_short_term_cache(cache_dir, "demo_hash", panel)
        loaded = sts._load_short_term_cache(cache_dir, "demo_hash")

        assert loaded is not None
        assert "AAA.JK" in loaded.columns
        assert "z_mom12" in loaded.attrs
        assert isinstance(loaded.attrs["z_mom12"], pd.DataFrame)
