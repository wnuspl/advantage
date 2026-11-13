from __future__ import annotations

import math
import sys
import unittest
from array import array
from contextlib import closing
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

from tennis import elo, models_db, tuning      # noqa: E402
from tennis.evaluate import score               # noqa: E402
from test_build import Workspace                # noqa: E402


def matches(rows):
    """rows: (winner, loser, start, status[, tour_level])"""
    m = elo.Matches(array("q"), array("q"), array("q"), [], [], bytearray(), [])
    for i, r in enumerate(rows):
        m.match_id.append(i + 1); m.winner.append(r[0]); m.loser.append(r[1])
        m.start.append(r[2]); m.status.append(r[3]); m.tour.append("atp")
        m.tour_level.append(r[4] if len(r) > 4 else 1)
    return m


class EloTest(unittest.TestCase):
    def test_expected_score(self):
        res = elo.run(matches([(1, 2, "2020-01-01", "completed")]), elo.EloParams())
        self.assertAlmostEqual(res.p_winner[0], 0.5)

    def test_constant_k_update_is_zero_sum(self):
        p = elo.EloParams(k=32, shape=0)
        res = elo.run(matches([(1, 2, "2020-01-01", "completed"), (1, 2, "2020-01-08", "completed")]), p)
        self.assertAlmostEqual(res.r_winner[1], 1516.0)
        self.assertAlmostEqual(res.r_loser[1], 1484.0)
        self.assertAlmostEqual(res.p_winner[1], 1 / (1 + 10 ** (-32 / 400)))

    def test_decaying_k(self):
        p = elo.EloParams(k=250, offset=5, shape=0.4)
        res = elo.run(matches([(1, 2, "2020-01-01", "completed"), (3, 1, "2020-01-08", "completed")]), p)
        self.assertAlmostEqual(res.r_loser[1], 1500 + 250 / 5 ** 0.4 * 0.5)
        self.assertEqual((res.n_winner[1], res.n_loser[1]), (0, 1))

    def test_walkover_and_retirement_handling(self):
        rows = [(1, 2, "2020-01-01", "walkover"), (1, 2, "2020-01-02", "retired"), (1, 2, "2020-01-03", "completed")]
        res = elo.run(matches(rows), elo.EloParams(retired_weight=0.0))
        self.assertEqual(res.n_winner[2], 0)              # neither walkover nor retirement counted
        res = elo.run(matches(rows), elo.EloParams(retired_weight=1.0))
        self.assertEqual(res.n_winner[2], 1)              # retirement counted, walkover not
        half = elo.run(matches(rows), elo.EloParams(k=32, shape=0, retired_weight=0.5))
        self.assertAlmostEqual(half.r_winner[2], 1508.0)

    def test_predictions_never_see_later_matches(self):
        rows = [(1, 2, "2020-01-01", "completed"), (2, 3, "2020-01-08", "completed"),
                (3, 1, "2020-01-15", "completed"), (1, 3, "2020-01-22", "completed")]
        full = elo.run(matches(rows), elo.EloParams())
        for cut in range(1, len(rows)):
            part = elo.run(matches(rows[:cut]), elo.EloParams())
            self.assertEqual(list(part.p_winner), list(full.p_winner[:cut]))


class ScoreTest(unittest.TestCase):
    def test_coin_flip(self):
        s = score([0.5, 0.5], bytearray([1, 0]), [0, 1])
        self.assertAlmostEqual(s.log_loss, math.log(2))
        self.assertAlmostEqual(s.brier, 0.25)
        self.assertAlmostEqual(s.accuracy, 0.5)

    def test_values_and_calibration(self):
        s = score([0.8, 0.3], bytearray([1, 0]), [0, 1])
        self.assertAlmostEqual(s.log_loss, -(math.log(0.8) + math.log(0.3)) / 2)
        self.assertAlmostEqual(s.brier, (0.04 + 0.49) / 2)
        self.assertEqual(s.accuracy, 0.5)
        self.assertEqual(sum(b[2] for b in s.calibration), 2)
        # match 2: winner is p2 with P=0.3, so P(p1 wins)=0.7 lands in the 0.7-0.8 bin, p1 lost
        self.assertEqual(s.calibration[7][2:], [1, 0.7, 0.0])

    def test_grid(self):
        g = tuning.expand_grid([{"k": [100, 200], "shape": [0.4]}, {"k": [100], "shape": [0.4]}])
        self.assertEqual(len(g), 2)
        with self.assertRaises(ValueError):
            tuning.expand_grid({"kk": [1]})


class SweepEndToEndTest(unittest.TestCase):
    def test_sweep_and_finalize_on_fixture(self):
        ws = Workspace()
        try:
            ws.build()
            with closing(models_db.connect(ws.tmp / "models.db")) as con:
                models_db.attach_data(con, ws.db)
                prep = tuning.prepare(con)
                self.assertEqual(len(prep.m), 7)
                val = ("2020-01-01", "2020-01-19")             # Brisbane + ITF weeks
                out = tuning.sweep(con, prep, tuning.expand_grid({"k": [32, 64], "shape": [0.0]}),
                                   sweep_id="t", val=val, progress=lambda *_: None)
                self.assertEqual(len(out), 2)
                board = tuning.leaderboard(con, sweep_id="t", scope="all")
                self.assertEqual(len(board), 2)
                self.assertLessEqual(board[0][4], board[1][4])  # sorted by val log loss
                # Brisbane final is a walkover: not scored. Tour scope = Brisbane only -> 0 rows.
                n_all = con.execute("""SELECT n FROM run_metric WHERE block='val' AND scope='all'
                                       AND run_id=?""", (out[0][0],)).fetchone()[0]
                self.assertEqual(n_all, 2)                     # the two ITF matches
                run_id = board[0][0]
                scores = tuning.finalize(con, prep, run_id, test=("2020-01-20", "2020-12-31"),
                                         save_predictions=True)
                self.assertEqual(scores["tour"].n, 4)         # AO 2020: 2 R128, SF, F
                self.assertEqual(con.execute("SELECT COUNT(*) FROM prediction").fetchone()[0], 7)
                self.assertEqual(con.execute("SELECT COUNT(*) FROM pre_match_rating").fetchone()[0], 14)
                self.assertEqual(con.execute(
                    "SELECT COUNT(*) FROM prediction WHERE NOT p1_key < p2_key").fetchone()[0], 0)
                # same parameters, same answer
                again = tuning.finalize(con, prep, run_id, test=("2020-01-20", "2020-12-31"))
                self.assertEqual(again["tour"].log_loss, scores["tour"].log_loss)
        finally:
            ws.cleanup()


if __name__ == "__main__":
    unittest.main()
