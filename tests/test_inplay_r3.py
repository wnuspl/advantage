"""Reboot item R3: per-direction form variances and per-player form volatility (train/inplay_r3.py)."""
from __future__ import annotations

import math
import random
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from tennis import matches, points_rating as pr              # noqa: E402
from tennis.train import inplay, inplay_r3                   # noqa: E402


def fake_prep(rows):
    """rows: (start, winner, loser, w_sw, w_sp, l_sw, l_sp)."""
    m = matches.empty_matches()
    d = SimpleNamespace(w_sp=[], w_sw=[], l_sp=[], l_sw=[])
    for k, (start, w, l, wk, wn, lk, ln) in enumerate(rows):
        m.match_id.append(100 + k); m.winner.append(w); m.loser.append(l)
        m.start.append(start); m.status.append("completed"); m.tour_level.append(1)
        m.tour.append("atp"); m.surface.append("hard")
        d.w_sw.append(wk); d.w_sp.append(wn); d.l_sw.append(lk); d.l_sp.append(ln)
    m.extra["points"] = d
    n = len(rows)
    pm = {"xw": np.full(n, 0.5), "xl": np.full(n, 0.5), "vw": np.zeros(n), "vl": np.zeros(n)}
    return SimpleNamespace(m=m, best_of=[3] * n), pm


class MomentTest(unittest.TestCase):
    def test_u_is_unbiased_for_the_form_variance(self):
        rng = random.Random(7)
        F, x, n = 0.05, 0.6, 80
        us = []
        for _ in range(20000):
            p = 1 / (1 + math.exp(-(x + rng.gauss(0, math.sqrt(F)))))
            k = sum(rng.random() < p for _ in range(n))
            us.append(inplay_r3._u(x, 0.0, k, n))
        self.assertAlmostEqual(float(np.mean(us)), F, delta=0.01)

    def test_walk_uses_only_earlier_editions(self):
        rows = [("2015-01-05", 1, 2, 60, 80, 30, 70),
                ("2015-01-05", 1, 3, 40, 80, 45, 70),      # same edition as the first
                ("2015-01-12", 1, 2, 50, 80, 40, 70)]
        prep, pm = fake_prep(rows)
        uw, ul = inplay_r3.match_u(prep, pm)
        vol = inplay_r3.walk(prep, uw, ul)
        self.assertEqual(vol[100][1], (0.0, 0, 0.0, 0))            # nothing before the first edition
        self.assertEqual(vol[101][1], (0.0, 0, 0.0, 0))            # same edition: still nothing
        s_s, n_s, s_r, n_r = vol[102][1]
        self.assertEqual((n_s, n_r), (2, 2))                       # player 1 served and returned twice
        F = np.nanmean(np.r_[uw[:2], ul[:2]])
        self.assertAlmostEqual(vol[102]["F"], F)
        self.assertAlmostEqual(s_s, uw[0] + uw[1] - 2 * F)
        self.assertAlmostEqual(s_r, ul[0] + ul[1] - 2 * F)
        self.assertAlmostEqual(vol[102][2][0], ul[0] - F)          # player 2: one serve so far

    def test_no_stats_no_u(self):
        prep, pm = fake_prep([("2015-01-05", 1, 2, 10, 20, -1, -1)])
        uw, ul = inplay_r3.match_u(prep, pm)
        self.assertTrue(np.isnan(uw[0]) and np.isnan(ul[0]))       # under MIN_SP / missing


class SplitFormTest(unittest.TestCase):
    def md(self):
        cps = [inplay.Checkpoint(0, 0, 1, 0, False, 4, 4, 0, 0, False),
               inplay.Checkpoint(0, 0, 3, 2, True, 12, 15, 5, 10, False)]
        md = inplay.MatchData(1, "2015", "atp", 3, "tb", 7, 0.6, 0.5, 0.01, 0.01, 1, cps)
        md.surface = "clay"
        md.r3 = (0.04, (0.8, 100, 0.0, 50), (-0.2, 100, 0.4, 100))   # F, p1 sums, p2 sums
        return md

    def test_base_equals_live_in_play(self):
        params = pr.PointsParams(unc=3.0)
        md = self.md()
        a = inplay_r3.SplitForm().match_probs(md, params)
        b = inplay.match_probs(md, inplay.InPlay(), params)
        np.testing.assert_allclose(a, b, atol=1e-12)

    def test_group_and_direction_vars(self):
        md = self.md()
        sf = inplay_r3.SplitForm(by_group=(("atp", 0.03), ("atp/clay", 0.01)))
        self.assertEqual(sf.direction_vars(md), (0.01, 0.01))       # most specific key wins
        sf = inplay_r3.SplitForm(kappa_s=100)
        f1, f2 = sf.direction_vars(md)
        self.assertAlmostEqual(f1, 0.02 * (1 + 0.8 / 200 / 0.04))   # p1 serving: p1's serve side
        self.assertAlmostEqual(f2, 0.02 * (1 - 0.2 / 200 / 0.04))
        sf = inplay_r3.SplitForm(kappa_r=100)
        f1, f2 = sf.direction_vars(md)
        self.assertAlmostEqual(f1, 0.02 * (1 + 0.4 / 200 / 0.04))   # p1 serving: p2's return side
        self.assertAlmostEqual(f2, 0.02)
        md.r3 = (0.04, (-100.0, 1, 0, 0), (0, 0, 0, 0))
        self.assertAlmostEqual(inplay_r3.SplitForm(kappa_s=1).direction_vars(md)[0], 0.02 * 0.25)   # floor
        md.r3 = None
        self.assertEqual(inplay_r3.SplitForm(kappa_s=1).direction_vars(md), (0.02, 0.02))

    def test_higher_volatility_moves_more(self):
        params = pr.PointsParams(unc=3.0)
        md = self.md()
        lo = inplay_r3.SplitForm(form_var=0.01).match_probs(md, params)
        hi = inplay_r3.SplitForm(form_var=0.05).match_probs(md, params)
        pre = inplay_r3.SplitForm(form_var=1e-9).match_probs(md, params)
        # p1 held serve 12/15 (above x1 = 0.6), p2 5/10: more form variance, p1 more favoured
        self.assertTrue(pre[1] < lo[1] < hi[1])


if __name__ == "__main__":
    unittest.main()
