import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from vecm_project.scripts import playbook_vecm


class PlaybookIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        fixture_path = Path(__file__).parent / "fixtures" / "price_sample.csv"
        self.price_frame = pd.read_csv(fixture_path, parse_dates=["date"])

    def test_run_playbook_generates_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "out_ms"
            out_dir.mkdir(parents=True, exist_ok=True)
            config = {
                "input_file": "fixture.csv",
                "subset": "AAA,BBB",
                "roll_years": 0.0,
                "seed": 123,
            }

            with mock.patch.object(playbook_vecm, "OUT_DIR", out_dir):
                result = playbook_vecm.run_playbook(
                    config,
                    persist=True,
                    data_frame=self.price_frame,
                )

            self.assertIn("run_id", result)
            self.assertIn("metrics", result)
            self.assertIn("execution", result)
            self.assertIn("model_checks", result)
            metrics = result["metrics"]
            self.assertIn("sharpe_oos", metrics)
            self.assertIn("turnover_annualised", metrics)
            self.assertIn("maxdd", metrics)

            run_id = result["run_id"]
            self.assertTrue((out_dir / f"positions_{run_id}.csv").exists())
            self.assertTrue((out_dir / f"returns_{run_id}.csv").exists())
            self.assertTrue((out_dir / f"metrics_{run_id}.csv").exists())
            self.assertTrue((out_dir / f"artifacts_{run_id}.json").exists())
            self.assertTrue((out_dir / "run_manifest.csv").exists())


if __name__ == "__main__":
    unittest.main()
