import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from tests.fixtures.synthetic_data import make_synthetic_price_frame, write_synthetic_csv
from vecm_project.scripts import playbook_vecm


def _dummy_fit_ms_spread(z_series: pd.Series, **kwargs: object) -> dict:
    return {"p_mr": pd.Series(1.0, index=z_series.index), "success": True}


def _dummy_short_term_overlay(price_panel: pd.DataFrame, *_args: object, **_kwargs: object) -> pd.DataFrame:
    eq_cols = [c for c in price_panel.columns if c.endswith(".JK")]
    panel = pd.DataFrame(0.0, index=pd.to_datetime(price_panel["date"]), columns=eq_cols)
    panel.attrs["z_mom12"] = panel.copy()
    return panel


def _build_config(tmpdir: str, subset: str, **overrides: object) -> dict:
    base = {
        "input_file": str(Path(tmpdir) / "synthetic.csv"),
        "subset": subset,
        "roll_years": 0.0,
        "seed": 17,
        "gate_require_corr": 0,
        "gate_enforce": True,
        "z_entry": 0.3,
        "p_th": 0.0,
        "half_life_max": 120.0,
    }
    base.update(overrides)
    return base


class PipelineIntegrationTest(unittest.TestCase):
    def test_quick_test_mode_truncates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_root = Path(tmpdir) / "cache"
            panel_dir = cache_root / "panels"
            feature_dir = cache_root / "features"
            csv_path = Path(tmpdir) / "synthetic.csv"
            write_synthetic_csv(csv_path, days=260, seed=21)
            config = _build_config(tmpdir, "COIN_A,COIN_B")
            with mock.patch.object(playbook_vecm, "PANEL_CACHE_DIR", panel_dir), mock.patch.object(
                playbook_vecm,
                "FEATURE_CACHE_DIR",
                feature_dir,
            ), mock.patch.dict(os.environ, {"VECM_QUICK_TEST": "1", "VECM_QUICK_TEST_DAYS": "180"}):
                playbook_vecm._DATAFRAME_CACHE.clear()
                playbook_vecm.run_playbook(config, persist=False)
                panel_files = list(panel_dir.rglob("*.parquet"))
                self.assertTrue(panel_files)
                panel = pd.read_parquet(panel_files[0])
                if "date" in panel.columns:
                    panel = panel.set_index("date")
                self.assertLess(len(panel), 260)

    def test_trend_pairs_gated_out_cointegrated_signal(self) -> None:
        frame = make_synthetic_price_frame(days=260)

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_root = Path(tmpdir) / "cache"
            panel_dir = cache_root / "panels"
            feature_dir = cache_root / "features"
            with mock.patch.object(playbook_vecm, "PANEL_CACHE_DIR", panel_dir), mock.patch.object(
                playbook_vecm,
                "FEATURE_CACHE_DIR",
                feature_dir,
            ), mock.patch.object(playbook_vecm, "fit_ms_spread", side_effect=_dummy_fit_ms_spread), mock.patch.object(
                playbook_vecm,
                "build_short_term_overlay",
                side_effect=_dummy_short_term_overlay,
            ):
                playbook_vecm._DATAFRAME_CACHE.clear()
                coin_config = _build_config(tmpdir, "COIN_A,COIN_B")
                coin_result = playbook_vecm.run_playbook(coin_config, persist=False, data_frame=frame)
                coin_signals = int(coin_result["signals"]["long"].sum())
                self.assertGreater(coin_signals, 0)

                trend_config = _build_config(
                    tmpdir,
                    "TREND_A,TREND_B",
                    half_life_max=0.05,
                )
                trend_result = playbook_vecm.run_playbook(trend_config, persist=False, data_frame=frame)
                trend_signals = int(trend_result["signals"]["long"].sum())
                self.assertEqual(trend_signals, 0)


if __name__ == "__main__":
    unittest.main()
