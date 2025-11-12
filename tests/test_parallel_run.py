import os
import unittest
from pathlib import Path

from vecm_project.scripts import parallel_run


class PlaybookPayloadMappingTest(unittest.TestCase):
    def setUp(self) -> None:
        self._original_default = parallel_run._DEFAULT_CFG_DICT
        parallel_run._DEFAULT_CFG_DICT = parallel_run.PlaybookConfig(
            input_file="dummy.csv"
        ).to_dict()
        self.runner = parallel_run.RunnerConfig(
            input_csv=Path("input.csv"),
            out_dir=Path("out"),
            manifest_path=Path("manifest.csv"),
            cache_dir=Path("cache"),
            lock_file=Path("lock"),
            stamp_file=Path("stamp"),
            max_workers=4,
            stage="stage1",
            stage_int=1,
            oos_start="2022-01-01",
            run_label="demo",
            time_budget=3600.0,
            max_jobs=10,
            date_align=False,
            min_obs=100,
            use_momentum=False,
            subsets=["ANTM,MDKA"],
        )

    def tearDown(self) -> None:
        parallel_run._DEFAULT_CFG_DICT = self._original_default

    def _make_job(self, subset: str, params: dict) -> parallel_run.JobSpec:
        job_params = {
            "tickers": subset.split(","),
            "threshold": params.get("zs", 1.0),
            "difference_lags": 1,
            "note": "demo",
            "run_id": "demo-0001",
            "trial_id": "demo",
            "stage": self.runner.stage_int,
            "method": "tvecm",
            "plan": "multisession:4",
            "seed_method": "numpy.SeedSequence",
            "n_workers": self.runner.max_workers,
            "oos_start": self.runner.oos_start,
            "grid_params": params,
        }
        if "z_q" in params:
            job_params["z_quantile"] = params["z_q"]
        return parallel_run.JobSpec(
            idx=1,
            subset=subset,
            tag="demo",
            params=job_params,
            aligned_path=None,
            seed=42,
        )

    def test_grid_parameters_forwarded_to_playbook(self) -> None:
        subset = "ANTM,MDKA"
        grid_params = {
            "z_meth": "quant",
            "p": 0.55,
            "rc": 1,
            "g": 0.6,
            "w": 45,
            "cd": 3,
            "ze": 0.7,
            "zs": 1.2,
            "z_q": 0.65,
        }
        job = self._make_job(subset, grid_params)

        payload = parallel_run._playbook_payload(job, self.runner)

        self.assertEqual(payload["subset"], subset)
        self.assertEqual(payload["p_th"], grid_params["p"])
        self.assertEqual(payload["regime_confirm"], grid_params["rc"])
        self.assertEqual(payload["gate_corr_min"], grid_params["g"])
        self.assertEqual(payload["gate_corr_win"], grid_params["w"])
        self.assertEqual(payload["cooldown"], grid_params["cd"])
        self.assertEqual(payload["z_exit"], grid_params["ze"])
        self.assertEqual(payload["z_stop"], grid_params["zs"])
        self.assertEqual(payload["z_auto_method"], grid_params["z_meth"])
        self.assertEqual(payload["z_auto_q"], grid_params["z_q"])

    def test_nested_grid_parameters_are_flattened(self) -> None:
        subset = "ANTM,MDKA"
        grid_params = {
            "p": 0.52,
            "grid_params": {
                "rc": 1,
                "grid_params": {
                    "g": 0.61,
                    "grid_params": {"cd": 5},
                },
            },
        }
        job = self._make_job(subset, grid_params)

        payload = parallel_run._playbook_payload(job, self.runner)

        self.assertEqual(payload["p_th"], grid_params["p"])
        self.assertEqual(payload["regime_confirm"], 1)
        self.assertEqual(payload["gate_corr_min"], 0.61)
        self.assertEqual(payload["cooldown"], 5)

    def test_non_grid_dictionaries_are_not_flattened(self) -> None:
        subset = "ANTM,MDKA"
        grid_params = {
            "rc": 1,
            "inner": {"rc": 5},
        }
        job = self._make_job(subset, grid_params)

        payload = parallel_run._playbook_payload(job, self.runner)

        self.assertEqual(payload["regime_confirm"], 1)
        self.assertNotIn("inner", payload)

    def test_deeply_nested_grid_params_are_discovered(self) -> None:
        subset = "ANTM,MDKA"
        grid_params = {
            "rc": 1,
            "wrapper": {
                "grid_params": {
                    "grid_params": {"g": 0.71},
                    "nested": {"grid_params": {"cd": 9}},
                }
            },
        }
        job = self._make_job(subset, grid_params)

        payload = parallel_run._playbook_payload(job, self.runner)

        self.assertEqual(payload["regime_confirm"], 1)
        self.assertEqual(payload["gate_corr_min"], 0.71)
        self.assertEqual(payload["cooldown"], 9)


class GatherSubsetsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.input_csv = Path("dummy.csv")
        self._env_backup = {key: os.environ.get(key) for key in ["VECM_SUBS", "VECM_PREFILTER"]}
        for key in ["VECM_SUBS", "VECM_PREFILTER"]:
            os.environ.pop(key, None)
        self._original_library = parallel_run.SUBSET_LIBRARY_PATH
        parallel_run.SUBSET_LIBRARY_PATH = Path("missing_library.txt")

    def tearDown(self) -> None:
        for key, value in self._env_backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        parallel_run.SUBSET_LIBRARY_PATH = self._original_library

    def test_override_pairs_are_honoured(self) -> None:
        pairs = ["BBCA.JK,BMRI.JK"]
        result = parallel_run._gather_subsets(self.input_csv, override=pairs)
        self.assertEqual(result, pairs)

    def test_override_supersedes_environment(self) -> None:
        os.environ["VECM_SUBS"] = "ANTM,MDKA;TLKM,ISAT"
        result = parallel_run._gather_subsets(
            self.input_csv, override=["BBCA.JK,BMRI.JK"]
        )
        self.assertEqual(result, ["BBCA.JK,BMRI.JK"])

    def test_environment_used_when_no_override(self) -> None:
        os.environ["VECM_SUBS"] = "ANTM,MDKA;TLKM,ISAT"
        result = parallel_run._gather_subsets(self.input_csv)
        self.assertEqual(result, ["ANTM,MDKA", "TLKM,ISAT"])


if __name__ == "__main__":
    unittest.main()
