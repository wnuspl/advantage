"""calibrate.py: walk-forward alphas, past data only (ported from archive/v1/tests/test_calibrate.py)."""
from __future__ import annotations

import sys
import unittest
from array import array
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from tennis import calibrate, matches   # noqa: E402


def quarterly(n_per_year=100, years=(2018, 2019, 2020, 2021), tour="atp"):
    m = matches.empty_matches()
    starts = sorted(f"{y}-{1 + k * 4 // n_per_year * 3:02d}-01" for y in years for k in range(n_per_year))
    for i, start in enumerate(starts):
        m.match_id.append(i + 1); m.winner.append(2 * i + 1); m.loser.append(2 * i + 2)
        m.start.append(start); m.status.append("completed"); m.surface.append("hard")
        m.tour.append(tour); m.tour_level.append(1)
    return m


class CalibrateTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.m = quarterly()
        self.pw = array("d", (0.5 + 0.3 * rng.random() for _ in range(len(self.m))))

    def test_alpha_uses_only_earlier_matches(self):
        p, idx, alphas = calibrate.calibrated(self.m, self.pw, "2021-01-01", "2021-12-31")
        self.assertEqual(len(idx), 100)
        pw2 = array("d", self.pw)
        for i in idx:                                    # change predictions inside the test year
            pw2[i] = 0.99
        _, _, alphas2 = calibrate.calibrated(self.m, pw2, "2021-01-01", "2021-01-31")
        self.assertEqual(alphas[("atp", "2021-01-01")], alphas2[("atp", "2021-01-01")])
        # later quarters of 2021 do see the changed 2021 rows
        _, _, alphas3 = calibrate.calibrated(self.m, pw2, "2021-04-01", "2021-04-30")
        self.assertNotEqual(alphas[("atp", "2021-04-01")], alphas3[("atp", "2021-04-01")])

    def test_per_best_of(self):
        p1, _, _ = calibrate.calibrated(self.m, self.pw, "2021-01-01", "2021-12-31")
        bo3 = [3] * len(self.m)                          # all best-of-3 = the per-tour calibration
        p3, _, _ = calibrate.calibrated(self.m, self.pw, "2021-01-01", "2021-12-31", best_of=bo3, min_rows=10)
        np.testing.assert_array_equal(p3, p1)
        mixed = [5 if i % 2 else 3 for i in range(len(self.m))]
        _, _, few = calibrate.calibrated(self.m, self.pw, "2021-01-01", "2021-03-31", best_of=mixed, min_rows=1000)
        self.assertEqual(few[("atp", "2021-01-01", 5)], few[("atp", "2021-01-01")])
        _, _, own = calibrate.calibrated(self.m, self.pw, "2021-01-01", "2021-03-31", best_of=mixed, min_rows=10)
        self.assertNotEqual(own[("atp", "2021-01-01", 5)], own[("atp", "2021-01-01")])

    def test_alphas_at_is_what_calibrated_applies(self):
        bo = [5 if i % 3 == 0 else 3 for i in range(len(self.m))]
        al = calibrate.alphas_at(self.m, self.pw, "2021-04-01", best_of=bo, min_rows=10)
        p, idx, alphas = calibrate.calibrated(self.m, self.pw, "2021-04-01", "2021-06-30", best_of=bo, min_rows=10)
        for k, i in enumerate(idx):
            a = calibrate.alpha_for(al, "atp", bo[i])
            self.assertAlmostEqual(p[k], calibrate.apply(self.pw[i], a), places=14)
        self.assertEqual(calibrate.alpha_for(al, "wta", 3), al["wta"])      # no WTA history: alpha 1
        self.assertEqual(al["wta"], 1.0)
        self.assertEqual(calibrate.alpha_for({}, "atp", 3), 1.0)

    def test_raw_tours_are_left_alone(self):
        p, idx, alphas = calibrate.calibrated(self.m, self.pw, "2021-01-01", "2021-12-31", raw_tours=("atp",))
        np.testing.assert_allclose(p, [self.pw[i] for i in idx], rtol=0, atol=1e-12)
        self.assertTrue(all(v == 1.0 for v in alphas.values()))
        p2, _, _ = calibrate.calibrated(self.m, self.pw, "2021-01-01", "2021-12-31", raw_tours=("wta",))
        p3, _, _ = calibrate.calibrated(self.m, self.pw, "2021-01-01", "2021-12-31")
        np.testing.assert_array_equal(p2, p3)                      # no WTA rows: unchanged

    def test_orientation_does_not_change_alpha(self):
        rng = np.random.default_rng(1)
        pw = 0.3 + 0.6 * rng.random(500)
        flip = rng.random(500) < 0.5
        p1 = np.where(flip, 1 - pw, pw)
        y = np.where(flip, 0.0, 1.0)
        a_win = calibrate.fit_alpha(calibrate.logit(pw), np.ones(500))
        a_can = calibrate.fit_alpha(calibrate.logit(p1), y)
        self.assertAlmostEqual(a_win, a_can, places=10)

    def test_fit_alpha_recovers_a_known_slope(self):
        rng = np.random.default_rng(2)
        lg = rng.normal(0, 1.5, 20000)
        y = (rng.random(20000) < 1 / (1 + np.exp(-0.8 * lg))).astype(float)
        self.assertAlmostEqual(calibrate.fit_alpha(lg, y), 0.8, delta=0.04)

    def test_blocks(self):
        self.assertEqual(calibrate.block_start("2026-11-15"), "2026-10-01")
        self.assertEqual(calibrate.block_start("2026-03-31"), "2026-01-01")
        self.assertEqual(calibrate.add_years("2024-02-29", -10), "2014-02-28")


class ScopeTest(unittest.TestCase):
    def test_scope_indices(self):
        m = quarterly(n_per_year=8, years=(2020,))
        m.status[0] = "walkover"
        m.tour_level[1] = 0
        m.tour[2] = "wta"
        self.assertEqual(matches.scope_indices(m, "2020-01-01", "2020-12-31", "all"), list(range(1, 8)))
        self.assertEqual(matches.scope_indices(m, "2020-01-01", "2020-12-31", "tour"), [2, 3, 4, 5, 6, 7])
        self.assertEqual(matches.scope_indices(m, "2020-01-01", "2020-12-31", "tour_wta"), [2])
        self.assertEqual(matches.scope_indices(m, "2020-04-01", "2020-04-01", "tour_atp"), [3])
        self.assertEqual(matches.scope_indices(m, "2020-04-01", "2020-07-01", "tour_atp"), [3, 4, 5])
        with self.assertRaises(ValueError):
            matches.scope_indices(m, "2020-01-01", "2020-12-31", "tour_main")


if __name__ == "__main__":
    unittest.main()
