import unittest
from unittest import mock

import numpy as np
import pandas as pd

from vecm_project.scripts import ms_spread


class MsSpreadTest(unittest.TestCase):
    def test_fit_ms_spread_skips_short_series(self) -> None:
        idx = pd.date_range("2024-01-01", periods=10, freq="D")
        series = pd.Series(np.linspace(-1.0, 1.0, 10), index=idx)

        model = ms_spread.fit_ms_spread(series, min_len=50, cache_key="short-series")

        self.assertFalse(model["success"])
        self.assertTrue(model.get("skipped", False))
        self.assertEqual(len(model["p_mr"]), len(series))

    def test_fit_ms_spread_success_path(self) -> None:
        idx = pd.date_range("2024-01-01", periods=120, freq="D")
        series = pd.Series(np.sin(np.linspace(0, 6, 120)), index=idx)

        class FakeResult:
            def __init__(self, n: int) -> None:
                self.params = pd.Series({"sigma2[0]": 0.3, "sigma2[1]": 1.1})
                self.filtered_marginal_probabilities = pd.DataFrame(
                    {0: np.full(n, 0.8), 1: np.full(n, 0.2)}
                )

        class FakeMarkovRegression:
            def __init__(self, *_args, **_kwargs) -> None:
                pass

            def fit(self, maxiter: int = 200, disp: bool = False) -> FakeResult:
                return FakeResult(len(series))

        with mock.patch.object(ms_spread, "MarkovRegression", FakeMarkovRegression):
            model = ms_spread.fit_ms_spread(series, min_len=80, cache_key="fake-success")

        self.assertTrue(model["success"])
        self.assertEqual(model["regime_mr"], 0)
        self.assertEqual(len(model["p_mr"]), len(series))
        self.assertTrue((model["p_mr"] <= 1.0).all())
        self.assertTrue((model["p_mr"] >= 0.0).all())


if __name__ == "__main__":
    unittest.main()
