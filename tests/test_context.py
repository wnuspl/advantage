from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

from tennis import context, points_rating as pr   # noqa: E402
from test_points import make                       # noqa: E402

F = {f: j for j, f in enumerate(context.FEATURES)}


def with_ctx(rows, editions, dob=None):
    m = make(rows)
    m.extra["context"] = context.features(m, editions, dob or {})
    return m


class FeatureTest(unittest.TestCase):
    def test_layoff_rust_and_gaps_are_point_in_time(self):
        # player 1: event A (2 matches), then 20 weeks off, event B (3 matches)
        rows = [(1, 2, "2020-01-06", "completed", "hard", -1, -1, -1, -1),
                (1, 3, "2020-01-06", "completed", "hard", -1, -1, -1, -1),
                (1, 4, "2020-05-25", "completed", "clay", -1, -1, -1, -1),
                (5, 1, "2020-05-25", "completed", "clay", -1, -1, -1, -1),
                (1, 6, "2020-06-01", "completed", "clay", -1, -1, -1, -1)]
        m = with_ctx(rows, [10, 10, 11, 11, 12])
        cw, cl = m.extra["context"]
        self.assertEqual(cw[0].sum(), 0)                    # first match ever: nothing known
        self.assertEqual(cw[1, F["back1"]], 0)              # same event, no layoff
        self.assertEqual(cw[2, F["back1"]], 1)              # first match back
        self.assertEqual(cl[3, F["back2_5"]], 1)            # second match back (player 1 lost)
        self.assertEqual(cw[4, F["back2_5"]], 1)
        self.assertEqual(cw[4, F["gap1"]], 1)               # event B started 1 week before
        self.assertAlmostEqual(cw[4, F["load28"]], 2 / 5)   # 2 matches in event B, last 4 weeks
        self.assertEqual(cw[2, F["load28"]], 0)
        # a later match never changes earlier features
        m2 = with_ctx(rows[:3], [10, 10, 11])
        np.testing.assert_array_equal(m2.extra["context"][0], cw[:3])

    def test_gap_5_12_weeks(self):
        rows = [(1, 2, "2020-01-06", "completed", "hard", -1, -1, -1, -1),
                (1, 2, "2020-03-02", "completed", "hard", -1, -1, -1, -1)]
        cw, _ = with_ctx(rows, [1, 2]).extra["context"]
        self.assertEqual(cw[1, F["gap5_12"]], 1)
        self.assertEqual(cw[1, F["back1"]], 0)

    def test_age_spline_is_continuous_and_missing_is_neutral(self):
        rows = [(1, 2, "2020-01-06", "completed", "hard", -1, -1, -1, -1)]
        cw, cl = with_ctx(rows, [1], {1: "1985-01-06"}).extra["context"]
        self.assertAlmostEqual(cw[0, F["age_gt28"]], 7.0, delta=0.02)
        self.assertAlmostEqual(cw[0, F["age_gt32"]], 3.0, delta=0.02)
        self.assertEqual(cw[0, F["age_lt26"]], 0)
        self.assertEqual(cl[0].sum(), 0)                    # player 2 has no dob


class ContextRunTest(unittest.TestCase):
    rows = [(1, 2, "2020-01-06", "completed", "hard", 60, 40, 60, 38),
            (1, 3, "2020-06-01", "completed", "hard", 60, 40, 60, 38),
            (2, 3, "2020-06-01", "completed", "hard", -1, -1, -1, -1)]
    eds = [1, 2, 2]

    def test_zero_context_reproduces_the_plain_run(self):
        m = with_ctx(self.rows, self.eds)
        a = pr.run(m, pr.PointsParams())
        b = pr.run(make(self.rows), pr.PointsParams())
        self.assertEqual(list(a.p_winner), list(b.p_winner))

    def test_rust_lowers_prediction_and_is_not_stored(self):
        p = pr.PointsParams(cs_back1=-0.2, cr_back1=-0.2)
        m = with_ctx(self.rows, self.eds)
        plain = pr.run(m, pr.PointsParams())
        rust = pr.run(m, p)
        self.assertLess(rust.p_winner[1], plain.p_winner[1])     # player 1 is back from 21 weeks off
        self.assertEqual(rust.p_winner[0], plain.p_winner[0])
        # the update compares to the rust-adjusted expectation: same points won, so the
        # rusty player's serve rating ends HIGHER than without context
        self.assertGreater(rust.state.serve[1], plain.state.serve[1])

    def test_symmetric_in_player_order(self):
        p = pr.PointsParams(cs_back1=-0.2, cr_back1=-0.1, cs_age_gt28=-0.01)
        rows = self.rows[:1] + [(3, 1, "2020-06-01", "completed", "hard", 60, 38, 60, 40)]
        a = pr.run(with_ctx(self.rows[:2], [1, 2], {1: "1985-01-01"}), p)
        b = pr.run(with_ctx(rows, [1, 2], {1: "1985-01-01"}), p)
        self.assertAlmostEqual(a.p_winner[1], 1 - b.p_winner[1], places=12)

    def test_per_tour_coefficients(self):
        m = with_ctx(self.rows, self.eds)
        m.tour[1] = "wta"                                        # the rusty match is a WTA one
        plain = pr.run(m, pr.PointsParams()).p_winner[1]
        atp_only = pr.PointsParams(cs_back1=-0.2, ctx_per_tour=1.0)
        self.assertEqual(pr.run(m, atp_only).p_winner[1], plain)  # ATP set ignored for WTA
        wta = pr.PointsParams(cs_back1_w=-0.2, ctx_per_tour=1.0)
        self.assertLess(pr.run(m, wta).p_winner[1], plain)
        shared = pr.PointsParams(cs_back1=-0.2)                  # ctx_per_tour 0: shared set
        self.assertAlmostEqual(pr.run(m, shared).p_winner[1], pr.run(m, wta).p_winner[1], places=12)
        p = pr.PointsParams().with_context(np.ones(context.K), np.zeros(context.K), tour="wta")
        self.assertEqual(p.cs_back1_w, 1.0)
        self.assertEqual(p.cs_back1, 0.0)

    def test_rust_weight_damps_the_update(self):
        m = with_ctx(self.rows, self.eds)
        full = pr.run(m, pr.PointsParams()).state
        damp = pr.run(m, pr.PointsParams(rust_weight=0.3)).state
        self.assertLess(abs(damp.serve[3]), abs(full.serve[3]))   # match 2 has rusty player 1


if __name__ == "__main__":
    unittest.main()
