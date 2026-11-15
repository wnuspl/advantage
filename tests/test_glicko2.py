from __future__ import annotations

import io
import math
import sys
import unittest
from contextlib import closing, redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE.parent / "archive" / "v1" / "scripts"))   # archived scripts under test
sys.path.insert(0, str(HERE))

from tennis import elo, glicko2 as g2, models_db, standings, tuning   # noqa: E402
from test_build import Workspace                                      # noqa: E402
import ratings                                                        # noqa: E402

S = g2.SCALE


def matches(rows):
    """rows: (winner, loser, start, status[, surface])"""
    m = elo.empty_matches()
    for i, r in enumerate(rows):
        m.match_id.append(i + 1); m.winner.append(r[0]); m.loser.append(r[1])
        m.start.append(r[2]); m.status.append(r[3]); m.tour.append("atp")
        m.surface.append(r[4] if len(r) > 4 else "hard")
        m.tour_level.append(1)
    return m


class Glicko2MathTest(unittest.TestCase):
    def test_volatility_matches_glickman_example(self):
        # Glickman, "Example of the Glicko-2 system" (2013): sigma' = 0.05999 (printed truncated;
        # the exact value from the rounded v and Delta given there is 0.059996)
        self.assertAlmostEqual(g2._volatility(200 / S, 0.06, -0.4834, 1.7785, 0.5), 0.059996, places=6)

    def test_tau_zero_keeps_volatility(self):
        self.assertEqual(g2._volatility(1.0, 0.06, 2.0, 1.5, 0.0), 0.06)

    def test_single_update_by_hand(self):
        p = g2.Glicko2Params(tau=0.0)
        res = g2.run(matches([(1, 2, "2020-01-06", "completed"), (1, 2, "2020-01-06", "completed")]), p)
        phi = 350 / S
        g = 1 / math.sqrt(1 + 3 * phi ** 2 / math.pi ** 2)
        v = 1 / (g * g * 0.25)
        phi2 = 1 / math.sqrt(1 / phi ** 2 + 1 / v)
        self.assertAlmostEqual(res.r_winner[1], 1500 + S * phi2 ** 2 * g * 0.5, places=6)
        self.assertAlmostEqual(res.r_loser[1], 1500 - S * phi2 ** 2 * g * 0.5, places=6)
        self.assertAlmostEqual(res.d_winner[1], S * phi2, places=6)     # same week: no inflation
        self.assertAlmostEqual(res.p_winner[0], 0.5)

    def test_rd_grows_with_inactivity_and_is_capped(self):
        p = g2.Glicko2Params(tau=0.0, period_days=30.0, rd_max=300.0)
        rows = [(1, 2, "2020-01-06", "completed"), (1, 2, "2020-01-06", "completed"),
                (1, 3, "2020-07-06", "completed"), (1, 3, "2100-01-04", "completed")]
        res = g2.run(matches(rows), p)
        after_two = res.d_winner[1]                    # RD before the 2nd match (same week)
        self.assertLess(res.d_winner[1], 300)
        # six months later: RD grew by sigma^2 * 182/30 in phi^2 terms
        res_last = g2.run(matches(rows[:2]), p).state
        expect = math.sqrt(res_last.rd[1] ** 2 + (S * 0.06) ** 2 * (182 / 30))
        self.assertAlmostEqual(res.d_winner[2], expect, places=6)
        self.assertGreater(res.d_winner[2], res_last.rd[1])
        self.assertEqual(res.d_winner[3], 300.0)       # eighty years away: capped at rd_max
        self.assertLess(after_two, 350)

    def test_walkover_and_retirement(self):
        rows = [(1, 2, "2020-01-06", "walkover"), (1, 2, "2020-01-13", "retired"), (1, 2, "2020-01-20", "completed")]
        res = g2.run(matches(rows), g2.Glicko2Params(retired_weight=0.0))
        self.assertEqual((res.n_winner[2], res.r_winner[2], res.d_winner[2]), (0, 1500.0, 350.0))
        res = g2.run(matches(rows), g2.Glicko2Params(retired_weight=1.0))
        self.assertEqual(res.n_winner[2], 1)

    def test_blend_is_prediction_only(self):
        rows = [(1, 2, "2020-01-06", "completed", "clay"), (2, 1, "2020-02-03", "completed", "grass"),
                (1, 2, "2020-03-02", "completed", None), (1, 2, "2020-04-06", "completed", "clay")]
        a = g2.run(matches(rows), g2.Glicko2Params(blend=0.0))
        b = g2.run(matches(rows), g2.Glicko2Params(blend=0.6))
        self.assertEqual(list(a.r_winner), list(b.r_winner))
        self.assertEqual(list(a.d_winner), list(b.d_winner))
        self.assertEqual(list(b.p_winner), list(g2.predict(a, 0.6)))
        self.assertEqual(list(a.p_winner), list(g2.predict(a, 0.0)))
        self.assertTrue(b.s_winner[2] != b.s_winner[2])        # no surface: NaN, overall only
        self.assertEqual(b.p_winner[2], a.p_winner[2])
        self.assertEqual(b.s_winner[1], 1500.0)                # first grass match: fresh surface state

    def test_predictions_never_see_later_matches(self):
        rows = [(1, 2, "2020-01-06", "completed"), (2, 3, "2020-02-03", "completed"),
                (3, 1, "2020-03-02", "completed"), (1, 3, "2020-04-06", "completed")]
        p = g2.Glicko2Params(blend=0.3)
        full = g2.run(matches(rows), p)
        for cut in range(1, len(rows)):
            self.assertEqual(list(g2.run(matches(rows[:cut]), p).p_winner), list(full.p_winner[:cut]))

    def test_state_rd_at_later_date(self):
        st = g2.run(matches([(1, 2, "2020-01-06", "completed")]), g2.Glicko2Params()).state
        self.assertEqual(st.rd_at(1, "2020-01-06"), st.rd[1])
        self.assertGreater(st.rd_at(1, "2021-01-06"), st.rd[1])
        self.assertIsNone(st.srd_at(1, "clay", "2021-01-06"))


class Glicko2EndToEndTest(unittest.TestCase):
    def test_sweep_finalize_standings_and_script(self):
        ws = Workspace()
        try:
            ws.build()
            models = ws.tmp / "models.db"
            with closing(models_db.connect(models)) as con:
                models_db.attach_data(con, ws.db)
                prep = tuning.prepare(con)
                grid = tuning.expand_grid({"tau": [0.0], "blend": [0.0, 0.5]}, g2.Glicko2Params)
                out = tuning.sweep(con, prep, grid, sweep_id="g", val=("2020-01-01", "2020-01-19"),
                                   progress=lambda *_: None, model="glicko2_v1")
                self.assertEqual(len(out), 2)
                self.assertEqual({r[8] for r in tuning.leaderboard(con, sweep_id="g", scope="all")}, {"glicko2_v1"})
                run_id = out[0][0]
                spec, params = tuning.model_of(con, run_id)
                self.assertEqual((spec.model_id, type(params)), ("glicko2_v1", g2.Glicko2Params))
                with self.assertRaises(TypeError):     # Elo params cannot run as Glicko
                    tuning.sweep(con, prep, tuning.expand_grid({"k": [32]}), sweep_id="x", model="glicko2_v1")
                tuning.finalize(con, prep, run_id, test=("2020-01-20", "2020-12-31"), save_predictions=True)
                self.assertEqual(con.execute("""SELECT COUNT(*) FROM pre_match_rating WHERE run_id=?
                                                AND deviation IS NOT NULL""", (run_id,)).fetchone()[0], 14)
                m = elo.load_matches(con)
                rows, _ = standings.compute(con, m, params, as_of=None, run=spec.run)
                self.assertEqual(rows[0].name, "Novak Djokovic")
                self.assertTrue(all(r.deviation is not None and 0 < r.deviation <= 350 for r in rows))
                later, _ = standings.compute(con, m, params, as_of="2021-06-01", run=spec.run)
                dj = next(r for r in later if r.name == "Novak Djokovic")
                self.assertGreater(dj.deviation, rows[0].deviation)   # a year idle: wider RD
            buf = io.StringIO()
            with redirect_stdout(buf):
                ratings.main(["--data", str(ws.db), "--models", str(models), "--model", "glicko2_v1",
                              "--tour", "atp", "--all"])
            text = buf.getvalue()
            self.assertIn("(glicko2_v1)", text)
            self.assertIn(" RD ", text)
            self.assertIn("Novak Djokovic", text)
        finally:
            ws.cleanup()


if __name__ == "__main__":
    unittest.main()
