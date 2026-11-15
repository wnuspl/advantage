from __future__ import annotations

import io
import sys
import unittest
from contextlib import closing, redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent
# archived test: archive src first (merges with the live tennis package), then live src,
# live tests (test_build.Workspace), archived scripts, this folder
_A = Path(__file__).resolve().parents[1]
for _p in (_A.parents[1] / "tests", _A.parents[1] / "src", _A / "scripts", _A / "tests", _A / "src"):
    sys.path.insert(0, str(_p))

from tennis import elo, models_db, standings   # noqa: E402
from test_build import Workspace                # noqa: E402
import elo_ratings                              # noqa: E402

PARAMS = elo.EloParams(k=32, shape=0.0, blend=0.5)


class StandingsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ws = Workspace()
        cls.ws.build()
        cls.models = cls.ws.tmp / "models.db"
        cls.con = models_db.connect(cls.models)
        models_db.attach_data(cls.con, cls.ws.db)
        cls.m = elo.load_matches(cls.con)

    @classmethod
    def tearDownClass(cls):
        cls.con.close()
        cls.ws.cleanup()

    def names(self, rows):
        return [r.name for r in rows]

    def test_until_is_strictly_before(self):
        self.assertEqual(len(elo.until(self.m, "2020-01-06")), 0)
        self.assertEqual(len(elo.until(self.m, "2020-01-07")), 3)   # Brisbane W/O + two ITF matches
        self.assertEqual(len(elo.until(self.m, "2099-01-01")), len(self.m))

    def test_final_state_includes_last_match(self):
        res = elo.run(self.m, PARAMS)
        last = len(self.m) - 1                                    # AO final: Djokovic d. Nadal
        self.assertAlmostEqual(res.state.rating[self.m.winner[last]],
                               res.r_winner[last] + 32 * (1 - elo._expected(res.r_winner[last] - res.r_loser[last])))

    def test_before_australian_open(self):
        rows, last = standings.compute(self.con, self.m, PARAMS, as_of="2020-01-20")
        self.assertEqual(last, "2020-01-06")
        # Brisbane final was a walkover (no update); John Smith won both ITF matches
        self.assertEqual(self.names(rows), ["John Smith", "Aljaž Bedene", "Sam Groth"])   # tie: by name
        self.assertEqual(rows[0].matches, 2)
        self.assertEqual(rows[0].surface["clay"], rows[0].overall)  # all his matches on clay
        self.assertIsNone(rows[0].surface["hard"])

    def test_end_of_data_and_filters(self):
        rows, _ = standings.compute(self.con, self.m, PARAMS, as_of=None)
        self.assertEqual(rows[0].name, "Novak Djokovic")            # won SF and F
        self.assertEqual([r.rank for r in rows], list(range(1, len(rows) + 1)))
        self.assertTrue(all(a.score >= b.score for a, b in zip(rows, rows[1:])))
        busy, _ = standings.compute(self.con, self.m, PARAMS, as_of=None, min_matches=3)
        self.assertEqual(self.names(busy), ["John Smith"])         # only player with 3 rated matches
        self.assertEqual(standings.compute(self.con, self.m, PARAMS, as_of=None, tour="wta")[0], [])

    def test_surface_ranking_uses_blend(self):
        rows, _ = standings.compute(self.con, self.m, PARAMS, as_of=None, surface="clay")
        smith = next(r for r in rows if r.name == "John Smith")
        self.assertAlmostEqual(smith.score, 0.5 * smith.overall + 0.5 * smith.surface["clay"])
        fed = next(r for r in rows if r.name == "Roger Federer")
        self.assertAlmostEqual(fed.score, 0.5 * fed.overall + 0.5 * 1500.0)   # never played on clay

    def test_script_runs(self):
        run_id = models_db.start_run(self.con, "elo_v1", PARAMS.as_dict())
        out = io.StringIO()
        csv_path = self.ws.tmp / "ratings.csv"
        with redirect_stdout(out):
            elo_ratings.main(["--data", str(self.ws.db), "--models", str(self.models), "--run", str(run_id),
                              "--tour", "atp", "--all", "--csv", str(csv_path)])
        text = out.getvalue()
        self.assertIn("Novak Djokovic", text)
        self.assertLess(text.index("Novak Djokovic"), text.index("Rafael Nadal"))
        self.assertEqual(len(csv_path.read_text(encoding="utf-8").splitlines()), 1 + 6)


if __name__ == "__main__":
    unittest.main()
