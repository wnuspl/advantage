"""Milestone A4: joint form update and the in-play replay harness (train/inplay.py)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from tennis import points_rating as pr                       # noqa: E402
from tennis.train import inplay                              # noqa: E402

A, B = 1, 2


def game(server, winner, n=4, lose=0, set_idx=1):
    """A game the server/winner takes n-lose... as point list (set_idx, server, winner)."""
    other = B if winner == A else A
    return [(set_idx, server, other)] * lose + [(set_idx, server, winner)] * n


def six_love(winner, set_idx, first_server=A):
    pts, srv = [], first_server
    for _ in range(6):
        pts += game(srv, winner, set_idx=set_idx)
        srv = B if srv == A else A
    return pts, srv


class FormTest(unittest.TestCase):
    def test_zero_correlation_equals_one_d(self):
        a = pr.in_match_form(0.6, 0.01, 40, 60, 0.02)
        b = pr.in_match_form(0.4, 0.012, 30, 55, 0.02)
        got = pr.in_match_form2(0.6, 0.4, 0.01, 0.012, 40, 60, 30, 55, 0.02, 0.0)
        np.testing.assert_allclose(got, (a[0], b[0], a[1], b[1]), rtol=0, atol=1e-12)
        self.assertEqual(pr.in_match_form2(0.6, 0.4, 0.01, 0.012, 0, 0, 0, 0), (0.6, 0.4, 0.01, 0.012))

    def test_correlation_moves_the_other_stream(self):
        # A serves far above expectation; B has no points yet
        base = pr.in_match_form2(0.6, 0.4, 0.01, 0.01, 50, 60, 0, 0, 0.02, 0.0)
        neg = pr.in_match_form2(0.6, 0.4, 0.01, 0.01, 50, 60, 0, 0, 0.02, -0.5)
        pos = pr.in_match_form2(0.6, 0.4, 0.01, 0.01, 50, 60, 0, 0, 0.02, 0.5)
        self.assertEqual(base[1], 0.4)
        self.assertLess(neg[1], 0.4)          # day form: A also returns well, B serves worse
        self.assertGreater(pos[1], 0.4)       # conditions: both serve better
        self.assertGreater(neg[0], 0.6)


class ReplayTest(unittest.TestCase):
    def test_straight_sets_and_checkpoints(self):
        s1, srv = six_love(A, 1)
        s2, _ = six_love(A, 2, srv)
        got = inplay.replay(s1 + s2, A, B, 3, "tb", 7)
        self.assertIsNotNone(got)
        cps, y = got
        self.assertEqual(y, 1)
        self.assertEqual(len(cps), 11)                       # 12 games, the last ends the match
        self.assertTrue(cps[5].set_end)
        self.assertEqual((cps[5].sets1, cps[5].games1), (1, 0))
        self.assertEqual((cps[0].k1, cps[0].n1, cps[0].n2), (4, 4, 0))
        self.assertFalse(cps[0].p1_serving)                  # B serves game 2

    def test_rejects_inconsistent_points(self):
        s1, srv = six_love(A, 1)
        s2, _ = six_love(A, 2, srv)
        self.assertIsNone(inplay.replay(s1 + s2[:-1], A, B, 3, "tb", 7))      # unfinished
        self.assertIsNone(inplay.replay(s1 + s2 + s2[:4], A, B, 3, "tb", 7))   # points after match point
        wrong = [(1 if k < len(s1) - 1 else 2, s, w) for k, (_, s, w) in enumerate(s1)] + s2
        self.assertIsNone(inplay.replay(wrong, A, B, 3, "tb", 7))             # set_idx disagrees
        self.assertIsNone(inplay.replay(s1 + s2, A, 3, 3, "tb", 7))           # a stranger

    def test_tiebreak_and_match_tiebreak(self):
        pts, srv = [], A
        for k in range(12):                                  # 6-6 by holds
            pts += game(srv, srv, set_idx=1)
            srv = B if srv == A else A
        pts += [(1, srv, A)] * 7                             # tiebreak 7-0 to A
        s2, srv2 = six_love(B, 2, B if srv == A else A)
        mtb = [(3, srv2, A)] * 10
        got = inplay.replay(pts + s2 + mtb, A, B, 3, "mtb", 7)
        self.assertIsNotNone(got)
        self.assertEqual(got[1], 1)
        self.assertIsNone(inplay.replay(pts + s2 + mtb, A, B, 3, "tb", 7))    # a 10-point decider is not a set

    def test_aces_are_counted_and_used(self):
        s1, srv = six_love(A, 1)
        s2, _ = six_love(A, 2, srv)
        pts = [(s, sv, w, 1 if (k < 4 and sv == w) else 0) for k, (s, sv, w) in enumerate(s1 + s2)]
        cps, _ = inplay.replay(pts, A, B, 3, "tb", 7)
        self.assertEqual((cps[0].a1, cps[0].a2), (4, 0))
        md = inplay.MatchData(1, "2015", "atp", 3, "tb", 7, 0.6, 0.5, 0.01, 0.01, 1, cps[:2], 0.05, 0.05)
        params = pr.PointsParams(unc=3.0)
        plain = inplay.match_probs(md, inplay.InPlay(), params)
        bonus = inplay.match_probs(md, inplay.InPlay(ace_bonus=1.0), params)
        self.assertTrue(np.all(bonus >= plain))          # aces above the expected rate raise p1's serve logit
        self.assertEqual(inplay.InPlay(ace_bonus=0.5).label(), "f0.02 r0 ace0.5")

    def test_stage_alphas_identity(self):
        md = inplay.MatchData(1, "2015", "atp", 3, "tb", 7, 0.6, 0.5, 0.01, 0.01, 1,
                              [inplay.Checkpoint(0, 0, 1, 0, False, 4, 4, 0, 0, False)])
        out = inplay.apply_stage_alphas([md], [np.array([0.7])], {"all": 1.0})
        self.assertAlmostEqual(float(out[0][0]), 0.7)


if __name__ == "__main__":
    unittest.main()
