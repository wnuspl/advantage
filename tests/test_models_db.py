"""models.db: idempotent schema, stable keys into a fixture tennis.db, run bookkeeping."""
from __future__ import annotations

import json
import sqlite3
import sys
import unittest
from contextlib import closing
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

from tennis import models_db          # noqa: E402
from test_build import Workspace      # noqa: E402


class ModelsDbTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ws = Workspace()
        cls.ws.build()

    @classmethod
    def tearDownClass(cls):
        cls.ws.cleanup()

    def models(self, name="models.db") -> sqlite3.Connection:
        con = models_db.connect(self.ws.tmp / name)
        models_db.attach_data(con, self.ws.db)
        return con

    def test_data_schema_has_no_model_tables(self):
        with closing(self.ws.connect()) as con:
            names = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertNotIn("prediction", names)
        self.assertNotIn("rating_snapshot", names)

    def test_connect_is_idempotent(self):
        path = self.ws.tmp / "idem.db"
        with closing(models_db.connect(path)) as con:
            con.execute("INSERT INTO model_run (model_id, params_json, started_at) VALUES ('m','{}','t')")
            con.commit()
        with closing(models_db.connect(path)) as con:     # re-applying the schema keeps data
            self.assertEqual(con.execute("SELECT COUNT(*) FROM model_run").fetchone()[0], 1)
            self.assertEqual(con.execute("SELECT value FROM models_meta").fetchall(), [("1",)])

    def test_attached_data_is_read_only(self):
        with closing(self.models("ro.db")) as con:
            with self.assertRaises(sqlite3.OperationalError):
                con.execute("DELETE FROM data.player")

    def test_stable_keys(self):
        with closing(self.models("keys.db")) as con:
            mk, pk = models_db.match_keys(con), models_db.player_keys(con)
            n_matches = con.execute("SELECT COUNT(*) FROM data.matches").fetchone()[0]
            self.assertEqual(len(mk), n_matches)                # every match has a key
            self.assertEqual(len(set(mk.values())), n_matches)  # and keys are unique
            fed = con.execute("SELECT player_id FROM data.player WHERE canonical_name='Roger Federer'").fetchone()[0]
            self.assertEqual(pk[fed], "sackmann_atp:100001")
            final = con.execute("""SELECT match_id FROM data.matches WHERE round='F'
                                   AND score='6-4 6-4 6-4'""").fetchone()[0]
            self.assertEqual(mk[final], "sackmann_atp:2020-0580:100:F:100003:100002")

    def test_run_bookkeeping_and_cascade(self):
        with closing(self.models("runs.db")) as con:
            run = models_db.start_run(con, "elo_test", {"k": 32, "blend": 0.5}, sweep_id="s1")
            row = con.execute("SELECT params_json, data_built_at, data_sources FROM model_run WHERE run_id=?",
                              (run,)).fetchone()
            self.assertEqual(json.loads(row[0]), {"blend": 0.5, "k": 32})
            self.assertIsNotNone(row[1])
            self.assertEqual(json.loads(row[2])["mcp"], "bbbb2222")
            con.execute("""INSERT INTO prediction VALUES (?, 'sackmann_atp:x', '2020-01-20',
                           'sackmann_atp:100002', 'sackmann_atp:100003', 0.4, 0)""", (run,))
            con.execute("""INSERT INTO run_metric (run_id, block, scope, date_from, date_to, n, log_loss)
                           VALUES (?, 'val', 'all', '2020-01-01', '2020-12-31', 1, 0.51)""", (run,))
            with self.assertRaises(sqlite3.IntegrityError):    # canonical order, never winner-first
                con.execute("""INSERT INTO prediction VALUES (?, 'sackmann_atp:y', '2020-01-20',
                               'sackmann_atp:100003', 'sackmann_atp:100002', 0.6, 1)""", (run,))
            models_db.finish_run(con, run)
            models_db.delete_run(con, run)
            for t in ("prediction", "run_metric", "model_run"):
                self.assertEqual(con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0], 0, t)


if __name__ == "__main__":
    unittest.main()
