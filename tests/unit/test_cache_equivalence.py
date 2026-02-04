import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from tests.fixtures.synthetic_data import make_synthetic_price_frame
from vecm_project.scripts import playbook_vecm


def _dummy_fit_ms_spread(z_series: pd.Series, **kwargs: object) -> dict:
    return {"p_mr": pd.Series(0.75, index=z_series.index), "success": True}


def _dummy_short_term_overlay(price_panel: pd.DataFrame, *_args: object, **_kwargs: object) -> pd.DataFrame:
    eq_cols = [c for c in price_panel.columns if c.endswith(".JK")]
    panel = pd.DataFrame(0.0, index=pd.to_datetime(price_panel["date"]), columns=eq_cols)
    panel.attrs["z_mom12"] = panel.copy()
    return panel


class CacheEquivalenceTest(unittest.TestCase):
    def test_cached_spread_zscore_regime_equivalence(self) -> None:
        frame = make_synthetic_price_frame()
        cfg = playbook_vecm.PlaybookConfig(
            input_file="synthetic.csv",
            subset="COIN_A,COIN_B",
            roll_years=0.0,
            seed=42,
        )
        feature_config = playbook_vecm.FeatureConfig(
            base_config=cfg,
            pair=cfg.subset,
            method=cfg.method,
            horizon=cfg.horizon,
            data_frame=frame,
            run_id="cache-test",
        )

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
                feature_result = playbook_vecm.build_features(cfg.subset, feature_config)

                self.assertIsNotNone(feature_result.features)
                features = feature_result.features
                self.assertIsNotNone(features)
                pair_label = f"{features.selected_l}~{features.selected_r}"
                pair_id = playbook_vecm._safe_cache_slug(f"{features.selected_l}__{features.selected_r}")
                panel_df = frame[["date", f"{features.selected_l}", f"{features.selected_r}"]].copy()
                panel_hash = playbook_vecm.hash_dataframe(panel_df)
                feature_key = playbook_vecm._feature_cache_key(cfg, panel_hash, pair_label)
                feature_cache_path = feature_dir / pair_id / f"{feature_key}.parquet"

                cached_panel = pd.read_parquet(feature_cache_path)
                if "date" in cached_panel.columns:
                    cached_panel = cached_panel.set_index("date")
                cached_panel.index = pd.to_datetime(cached_panel.index)
                cached_panel = cached_panel.sort_index()

                spread_fresh = (features.lp.iloc[:, 0] - features.beta_series * features.lp.iloc[:, 1]).reindex(
                    features.zect.index
                )
                zscore_fresh = features.zect
                p_regime_fresh = features.p_mr_series.reindex(features.zect.index)

                spread_cached = cached_panel["spread"].reindex(spread_fresh.index)
                zscore_cached = cached_panel["zscore"].reindex(zscore_fresh.index)
                p_regime_cached = cached_panel["p_regime"].reindex(p_regime_fresh.index)

                spread_compare = pd.concat([spread_fresh, spread_cached], axis=1).dropna()
                zscore_compare = pd.concat([zscore_fresh, zscore_cached], axis=1).dropna()
                regime_compare = pd.concat([p_regime_fresh, p_regime_cached], axis=1).dropna()

                np.testing.assert_allclose(
                    spread_compare.iloc[:, 0], spread_compare.iloc[:, 1], rtol=1e-6, atol=1e-6
                )
                np.testing.assert_allclose(
                    zscore_compare.iloc[:, 0], zscore_compare.iloc[:, 1], rtol=1e-6, atol=1e-6
                )
                np.testing.assert_allclose(
                    regime_compare.iloc[:, 0], regime_compare.iloc[:, 1], rtol=1e-6, atol=1e-6
                )


if __name__ == "__main__":
    unittest.main()
