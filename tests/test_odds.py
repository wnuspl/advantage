"""Track B5: closing-odds reading, linking, de-vig, blend and backtest arithmetic, on a tiny
hand-made Valuebetennis file (tests/fixtures/odds)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from tennis import odds                                               # noqa: E402

FIX = HERE / "fixtures/odds/valuebetennis-matchs-2025.csv"
NAMES = {1: ("Auger-Aliassime", "Félix", "Félix Auger-Aliassime"), 2: ("Rune", "Holger", "Holger Rune"),
         3: ("Dimitrov", "Grigor", "Grigor Dimitrov"), 4: ("Sinner", "Jannik", "Jannik Sinner")}
OURS = [odds.OurMatch(100, "atp", "2024-12-29", "2025-01-06", 1, 2, "completed"),
        odds.OurMatch(101, "atp", "2024-12-29", "2025-01-07", 1, 3, "completed"),
        odds.OurMatch(102, "atp", "2024-12-29", "2025-01-08", 4, 2, "retired")]


class ReadTest(unittest.TestCase):
    def test_read_counts_and_orientation(self):
        rows, c = odds.read_vb_file(FIX)
        self.assertEqual(c, {"read": 5, "not_tour_category": 1, "no_winner": 1, "kept": 3})
        r = {x.source_key: x for x in rows}
        self.assertEqual((r["vb:1"].winner, r["vb:1"].loser, r["vb:1"].tour), ("Félix Auger-Aliassime", "Holger Rune", "atp"))
        self.assertEqual(r["vb:1"].odds, {"PS_close": (1.70, 2.20), "PS_open": (1.80, 2.05)})
        self.assertEqual(r["vb:2"].winner, "Felix Auger Aliassime")             # second player won: flipped
        self.assertEqual(r["vb:2"].odds, {"PS_open": (1.55, 2.50)})             # no closing line
        self.assertEqual(r["vb:1"].date, "2025-01-06")

    def test_missing_column_raises(self):
        bad = HERE / "fixtures/odds/_bad.csv"
        bad.write_text("match_id;date\n1;2025-01-01\n", encoding="utf-8")
        try:
            with self.assertRaises(ValueError):
                odds.read_vb_file(bad)
        finally:
            bad.unlink()


class LinkTest(unittest.TestCase):
    def test_names(self):
        self.assertEqual(odds.name_score("Felix Auger Aliassime", *NAMES[1]), 4)
        self.assertEqual(odds.name_score("Holger Vitus Rune", *NAMES[2]), 3)
        self.assertEqual(odds.name_score("Holger Rooney", *NAMES[2]), 0)

    def test_every_row_has_a_reason(self):
        rows, _ = odds.read_vb_file(FIX)
        got = {lk.row.source_key: (lk.match_id, lk.reason) for lk in odds.link(rows, OURS, NAMES)}
        self.assertEqual(got, {"vb:1": (100, "linked"), "vb:2": (101, "linked"),
                               "vb:5": (None, "winner_disagrees")})       # the sources disagree: retirement

    def test_date_window_and_duplicates(self):
        rows, _ = odds.read_vb_file(FIX)
        r1 = rows[0]
        late = odds.OddsRow("x", "atp", "2025-03-01", "", "", r1.winner, r1.loser, "", "", r1.odds)
        dup = odds.OddsRow("y", "atp", "2025-01-06", "", "", "Felix Auger Aliassime", "Rune Holger", "", "", {})
        got = {lk.row.source_key: lk.reason for lk in odds.link([r1, late, dup], OURS, NAMES)}
        self.assertEqual(got["x"], "no_candidate")                          # outside the edition window
        self.assertEqual(sorted([got["vb:1"], got["y"]]), ["duplicate_claim", "linked"])


class MathTest(unittest.TestCase):
    def test_devig(self):
        o1, o2 = np.array([1.70, 1.25, 2.0]), np.array([2.20, 4.2, 1.9])
        for meth in ("power", "multiplicative", "shin"):
            p1 = odds.devig(o1, o2, meth)
            p2 = odds.devig(o2, o1, meth)
            np.testing.assert_allclose(p1 + p2, 1, atol=1e-9)
            self.assertTrue(np.all(p1 < 1 / o1))                          # margin removed
        np.testing.assert_allclose(odds.devig(2.0, 2.0), 0.5)
        self.assertAlmostEqual(float(odds.overround(1.9, 1.9)), 2 / 1.9 - 1)

    def test_blend_recovers_weights(self):
        rng = np.random.default_rng(0)
        m, x = rng.normal(0, 1, 40_000), rng.normal(0, 1, 40_000)
        y = (rng.random(40_000) < odds.sigmoid(0.8 * m + 0.3 * x)).astype(float)
        (a, b), se = odds.blend_fit(y, np.column_stack([m, x]))
        self.assertAlmostEqual(a, 0.8, delta=4 * se[0])
        self.assertAlmostEqual(b, 0.3, delta=4 * se[1])
        self.assertAlmostEqual(odds.model_weight(0.7, 0.3), 0.3)

    def test_backtest(self):
        # side 1 at 2.2 with p 0.6 (EV +0.32): bet; won once, lost once
        bt = odds.backtest([0.6, 0.6, 0.5], [1, 0, 1], [2.2, 2.2, 1.9], [1.8, 1.8, 1.9])
        self.assertEqual(bt.n_bets, 2)
        self.assertAlmostEqual(bt.profit, 1.2 - 1.0)
        self.assertAlmostEqual(bt.roi, 0.1)
        self.assertAlmostEqual(bt.hit_rate, 0.5)
        self.assertEqual(odds.backtest([0.6], [1], [2.2], [1.8], threshold=0.5).n_bets, 0)


if __name__ == "__main__":
    unittest.main()
