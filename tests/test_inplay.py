"""In-play win probability (markov.match_from) and the in-match form update.

The main check is an independent point-by-point state machine (`step`, written from the
rules, not from markov.py): from every reachable score,
    P(now) = P(A wins point) * P(after A wins it) + P(B wins point) * P(after B wins it)
exactly, plus a Monte Carlo run of whole matches from a few mid-match scores.
"""
from __future__ import annotations

import math
import random
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from tennis import markov as mk, points_rating as pr   # noqa: E402

FORMATS = [(3, "tb", 7), (5, "tb", 10), (5, "adv", 7), (3, "mtb", 7), (5, "tb12", 7), (1, "tb", 7)]


def step(s, a_won, bo, final, dtb):
    """Score after one point. s = (sets_a, sets_b, games_a, games_b, pts_a, pts_b,
    A serves next point). Returns a new state, or 'A' / 'B' when the match ends."""
    sa, sb, ga, gb, pa, pb, srv = s
    need = bo // 2 + 1
    deciding = sa == sb == need - 1
    if deciding and final == "mtb":
        in_tb, n = True, 10
    else:
        at = None if deciding and final == "adv" else 12 if deciding and final == "tb12" else 6
        in_tb, n = at is not None and ga == gb == at, (dtb if deciding else 7)
    if in_tb:
        k = pa + pb                                   # index of the point just played
        first = srv if ((k + 1) // 2) % 2 == 0 else not srv
        pa, pb = pa + a_won, pb + (not a_won)
        nxt = (not srv) if k % 2 == 0 else srv        # server changes after points 0, 2, 4, ...
        if max(pa, pb) >= n and abs(pa - pb) >= 2:
            a_set = pa > pb
            sa, sb = sa + a_set, sb + (not a_set)
            if max(sa, sb) == need:
                return "A" if sa == need else "B"
            return (sa, sb, 0, 0, 0, 0, not first)    # a tiebreak counts as one game
        return (sa, sb, ga, gb, pa, pb, nxt)
    pa, pb = pa + a_won, pb + (not a_won)
    if pa == pb >= 4:
        pa = pb = 3
    if not (max(pa, pb) >= 4 and abs(pa - pb) >= 2):
        return (sa, sb, ga, gb, pa, pb, srv)
    ga, gb = ga + (pa > pb), gb + (pb > pa)
    srv = not srv
    kind = final if deciding else "tb"
    hi, lo = max(ga, gb), min(ga, gb)
    over = (hi >= 6 and hi - lo >= 2) or (kind == "tb" and hi == 7) or (kind == "tb12" and hi == 13)
    if over:
        a_set = ga > gb
        sa, sb = sa + a_set, sb + (not a_set)
        if max(sa, sb) == need:
            return "A" if sa == need else "B"
        return (sa, sb, 0, 0, 0, 0, srv)
    return (sa, sb, ga, gb, 0, 0, srv)


def mf(pa, pb, s, bo, final, dtb):
    if s in ("A", "B"):
        return 1.0 if s == "A" else 0.0
    sa, sb, ga, gb, x, y, srv = s
    return mk.match_from(pa, pb, bo, final, sa, sb, ga, gb, x, y, srv, dtb)


class MatchFromTest(unittest.TestCase):
    def test_start_equals_prematch_roll_up(self):
        for pa, pb in ((0.66, 0.62), (0.55, 0.71), (0.8, 0.8)):
            for bo, final, _ in FORMATS:
                for srv in (True, False):
                    self.assertAlmostEqual(mk.match_from(pa, pb, bo, final, 0, 0, a_serving=srv),
                                           float(mk.match(pa, pb, bo, final)), places=12)

    def test_one_point_consistency_on_reached_scores(self):
        rng = random.Random(7)
        for bo, final, dtb in FORMATS:
            for pa, pb in ((0.64, 0.61), (0.58, 0.70)):
                seen = set()
                for _ in range(40):
                    s = (0, 0, 0, 0, 0, 0, rng.random() < 0.5)
                    while s not in ("A", "B"):
                        seen.add(s)
                        s = step(s, rng.random() < 0.5, bo, final, dtb)
                self.assertGreater(len(seen), 300)
                in_tb = [s for s in seen if s[2] == s[3] >= 6]
                sample = rng.sample(sorted(seen), min(400, len(seen))) + in_tb[:100]   # always some tiebreaks
                for s in sample:
                    pw = pa if s[6] else 1 - pb
                    lhs = mf(pa, pb, s, bo, final, dtb)
                    rhs = pw * mf(pa, pb, step(s, True, bo, final, dtb), bo, final, dtb) + \
                        (1 - pw) * mf(pa, pb, step(s, False, bo, final, dtb), bo, final, dtb)
                    self.assertAlmostEqual(lhs, rhs, places=12, msg=f"{bo} {final} {s}")

    def test_monte_carlo_from_mid_match(self):
        rng = random.Random(11)
        pa, pb = 0.65, 0.62
        cases = [((1, 1, 4, 5, 2, 3, True), 3, "tb", 7),        # deciding set, A serving 30-40
                 ((2, 2, 6, 6, 4, 6, False), 5, "tb", 10),      # 10-point deciding tiebreak
                 ((0, 1, 6, 6, 3, 2, True), 3, "tb", 7)]        # tiebreak mid-way
        for s0, bo, final, dtb in cases:
            n, wins = 20000, 0
            for _ in range(n):
                s = s0
                while s not in ("A", "B"):
                    s = step(s, rng.random() < (pa if s[6] else 1 - pb), bo, final, dtb)
                wins += s == "A"
            p = mf(pa, pb, s0, bo, final, dtb)
            se = math.sqrt(p * (1 - p) / n)
            self.assertLess(abs(wins / n - p), 4 * se, msg=f"{s0}: sim {wins / n:.4f} vs {p:.4f}")

    def test_game_from_matches_game(self):
        for p in (0.5, 0.63, 0.8):
            self.assertAlmostEqual(mk.game_from(p, 0, 0), mk.game(p), places=12)
            self.assertAlmostEqual(mk.game_from(p, 5, 5), mk.game_from(p, 3, 3), places=12)

    def test_tiebreak_from_start(self):
        for n in (7, 10):
            self.assertAlmostEqual(mk.tiebreak_from(0.66, 0.6, 0, 0, True, n), mk.tiebreak(0.66, 0.6, n), places=12)


class ParseLiveTest(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(mk.parse_live("6-4 3-2", "30-15"),
                         dict(sets_a=1, sets_b=0, games_a=3, games_b=2, pts_a=2, pts_b=1))
        self.assertEqual(mk.parse_live("6-4 7-6", "", 5),
                         dict(sets_a=2, sets_b=0, games_a=0, games_b=0, pts_a=0, pts_b=0))
        self.assertEqual(mk.parse_live("4-6 6-6", "15-14"),         # tiebreak: plain numbers
                         dict(sets_a=0, sets_b=1, games_a=6, games_b=6, pts_a=15, pts_b=14))
        self.assertEqual(mk.parse_live("6-4 4-6", "7-8", 3, "mtb")["pts_b"], 8)
        self.assertEqual(mk.parse_live("6-4 6-6", "AD-40", 3, "adv") if False else
                         mk.parse_live("6-4 4-6 6-6", "AD-40", 3, "adv")["pts_a"], 4)  # adv: no tiebreak
        self.assertEqual(mk.parse_live("6-4 4-6 9-8", "", 3, "adv")["games_a"], 9)

    def test_rejects_bad_scores(self):
        with self.assertRaises(ValueError):
            mk.parse_live("6-4 6-3")                 # match over
        with self.assertRaises(ValueError):
            mk.parse_live("3-2 6-4")                 # unfinished set before a finished one


class FormTest(unittest.TestCase):
    def test_in_match_form(self):
        x, v = 0.5, 0.01
        self.assertEqual(pr.in_match_form(x, v, 0, 0, 0.02), (x, v))
        x2, v2 = pr.in_match_form(x, v, 45, 60, 0.02)            # 75% vs expected 62%
        self.assertGreater(x2, x)
        self.assertLess(v2, v)
        x3, _ = pr.in_match_form(x, v, 4500, 6000, 0.02)        # lots of evidence: near observed
        self.assertAlmostEqual(1 / (1 + math.exp(-x3)), 0.75, delta=0.005)   # prior still pulls a little
        # posterior mode: the score of the log posterior is zero there
        p = 1 / (1 + math.exp(-x2))
        self.assertAlmostEqual((45 - 60 * p) - (x2 - x) / (v + 0.02), 0.0, places=8)

    def test_in_play_at_start_equals_prematch(self):
        params = pr.PointsParams(unc=1.0)
        st = pr.PointsState(params, *({} for _ in range(13)))
        xa, xb, va, vb = 0.62, 0.45, 0.02, 0.03
        start = dict(sets_a=0, sets_b=0, games_a=0, games_b=0, pts_a=0, pts_b=0)
        for bo in (3, 5):
            pre = pr.win_probability(st, xa, xb, va, vb, bo, "tb")
            live = pr.in_play_probability(st, xa, xb, va, vb, bo, "tb", start, True)
            self.assertAlmostEqual(pre, live, delta=2e-4)       # MatchGrid interpolation error


if __name__ == "__main__":
    unittest.main()
