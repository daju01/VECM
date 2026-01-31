import unittest
from datetime import datetime
from unittest import mock

import duckdb

from vecm_project.scripts import storage


class StorageTest(unittest.TestCase):
    def test_storage_init_and_basic_ops_in_memory(self) -> None:
        conn = duckdb.connect(":memory:")
        try:
            storage.storage_init(conn)
            storage.write_run(conn, "run-1", started_at=datetime(2024, 1, 1), notes="note")
            storage.write_trial(
                conn,
                run_id="run-1",
                trial_id="trial-1",
                stage=0,
                pair="AAA~BBB",
                method="vecm",
                params={"alpha": 0.1},
                horizon={"train": 80, "test": 40},
                eval_time_s=0.5,
                sharpe_oos=1.2,
                maxdd=-0.1,
                turnover=0.3,
            )
            storage.write_model_checks(
                conn,
                "run-1",
                pair="AAA~BBB",
                johansen_rank=1,
                det_term="ci",
                tvecm_thresholds={"threshold": 0.5},
                spec_ok=True,
            )
            rows = conn.execute("SELECT COUNT(*) FROM runs").fetchone()
            self.assertEqual(rows[0], 1)
            trial_rows = conn.execute("SELECT COUNT(*) FROM trials").fetchone()
            self.assertEqual(trial_rows[0], 1)
        finally:
            conn.close()

    def test_managed_storage_uses_context(self) -> None:
        conn = duckdb.connect(":memory:")

        def _open(_read_only: bool = False) -> duckdb.DuckDBPyConnection:
            return conn

        with mock.patch.object(storage, "storage_open", _open):
            with storage.managed_storage("test") as active_conn:
                storage.write_run(active_conn, "run-ctx")
                rows = active_conn.execute("SELECT COUNT(*) FROM runs").fetchone()
                self.assertEqual(rows[0], 1)


if __name__ == "__main__":
    unittest.main()
