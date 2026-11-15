from __future__ import annotations

import math
import sys
import unittest
from datetime import date
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

from tennis import context, kalshi                 # noqa: E402
from test_points import make                       # noqa: E402

# shape of a real orderbook_fp response (bids only, dollars as strings)
BOOK_A = {"yes_dollars": [["0.40", "100"], ["0.42", "50"]],
          "no_dollars": [["0.50", "30"], ["0.55", "20"]]}       # YES asks: 0.45 x20, 0.50 x30
BOOK_B = {"yes_dollars": [["0.50", "10"], ["0.52", "40"]],      # NO asks on B: 0.48 x40, 0.50 x10
          "no_dollars": [["0.40", "25"]]}


class BookTest(unittest.TestCase):
    def test_asks_mirror_opposite_bids(self):
        self.assertEqual(kalshi.asks(BOOK_A, "yes"), [(0.45, 20.0), (0.5, 30.0)])
        self.assertEqual(kalshi.asks(BOOK_B, "no"), [(0.48, 40.0), (0.5, 10.0)])
        self.assertEqual(kalshi.best_bid(BOOK_A, "yes"), 0.42)

    def test_ladder_merges_both_routes(self):
        lad = kalshi.ladder(BOOK_A, "A", BOOK_B, "B")
        self.assertEqual([(lv.price, lv.size, lv.route) for lv in lad],
                         [(0.45, 20, "YES A"), (0.48, 40, "NO B"), (0.5, 10, "NO B"), (0.5, 30, "YES A")])

    def test_fair_prob_removes_overround(self):
        # A mid (0.42+0.45)/2 = 0.435; B mid (0.52+0.60)/2 = 0.56 -> 1-0.56 = 0.44
        self.assertAlmostEqual(kalshi.fair_prob(BOOK_A, BOOK_B), (0.435 + 0.44) / 2)
        self.assertIsNone(kalshi.fair_prob({}, None))


class KellyTest(unittest.TestCase):
    def test_single_level_matches_textbook_kelly(self):
        p, c, W = 0.6, 0.5, 1000.0
        lv = [kalshi.Level(c, 1e9, "YES A")]
        s = kalshi.kelly_walk(p, lv, W, fee_rate=0.0)
        self.assertAlmostEqual(s.cost, W * (p - c) / (1 - c), delta=c)     # 200 dollars, +-1 contract
        self.assertAlmostEqual(s.ev, s.contracts * p - s.cost)
        half = kalshi.kelly_walk(p, lv, W, fraction=0.5, fee_rate=0.0)
        self.assertAlmostEqual(half.cost, s.cost / 2, delta=c)

    def test_no_bet_without_edge_after_fees(self):
        lv = [kalshi.Level(0.50, 1e9, "YES A")]
        self.assertEqual(kalshi.kelly_walk(0.50, lv, 1000).contracts, 0)
        self.assertEqual(kalshi.kelly_walk(0.51, lv, 1000).contracts, 0)    # fee 0.0175 eats the edge
        self.assertGreater(kalshi.kelly_walk(0.53, lv, 1000).contracts, 0)

    def test_walks_the_book_and_is_log_optimal(self):
        p, W = 0.7, 1000.0
        lad = [kalshi.Level(0.50, 50, "YES A"), kalshi.Level(0.55, 100, "NO B"), kalshi.Level(0.60, 1e9, "YES A")]
        s = kalshi.kelly_walk(p, lad, W, fee_rate=0.0)
        self.assertEqual([f.contracts for f in s.fills][:2], [50, 100])     # thin cheap levels taken fully

        def growth(n_extra):                       # log growth with n_extra contracts at the last level
            n = 150 + n_extra
            cost = 50 * 0.5 + 100 * 0.55 + n_extra * 0.6
            return p * math.log((W - cost + n) / W) + (1 - p) * math.log((W - cost) / W)
        k = s.fills[2].contracts
        self.assertGreaterEqual(growth(k), growth(k - 5))
        self.assertGreaterEqual(growth(k), growth(k + 5))
        self.assertAlmostEqual(s.log_growth, growth(k))

    def test_max_cost_caps(self):
        s = kalshi.kelly_walk(0.9, [kalshi.Level(0.5, 1e9, "YES A")], 1000, fee_rate=0.0, max_cost=50)
        self.assertLessEqual(s.cost, 50)
        self.assertEqual(s.contracts, 100)

    def test_scale_sizing(self):
        s = kalshi.kelly_walk(0.7, [kalshi.Level(0.5, 1e9, "YES A")], 1000, fee_rate=0.0)
        h = kalshi.scale_sizing(s, 0.7, 0.5, 1000, fee_rate=0.0)
        self.assertEqual(h.contracts, s.contracts // 2)
        self.assertAlmostEqual(h.ev, 0.7 * h.contracts - h.cost)
        self.assertEqual(kalshi.scale_sizing(s, 0.7, 0.0, 1000).contracts, 0)

    def test_blend(self):
        self.assertEqual(kalshi.blend(0.7, 0.5, 1.0), 0.7)
        self.assertAlmostEqual(kalshi.blend(0.7, 0.5, 0.0), 0.5)
        self.assertAlmostEqual(kalshi.blend(0.7, 0.3, 0.5), 0.5)
        self.assertEqual(kalshi.blend(0.7, None, 0.5), 0.7)


class FeaturesAtTest(unittest.TestCase):
    def test_matches_features_for_the_next_match(self):
        """features_at on the matches before i == features() at match i, for every i/side."""
        rows = [(1, 2, "2020-01-06", "completed", "hard", -1, -1, -1, -1),
                (1, 3, "2020-01-06", "completed", "hard", -1, -1, -1, -1),
                (4, 1, "2020-01-13", "completed", "hard", -1, -1, -1, -1),
                (1, 4, "2020-05-25", "completed", "clay", -1, -1, -1, -1),
                (5, 1, "2020-05-25", "retired", "clay", -1, -1, -1, -1),
                (1, 6, "2020-06-01", "completed", "clay", -1, -1, -1, -1),
                (2, 1, "2020-06-01", "completed", "clay", -1, -1, -1, -1)]
        eds = [10, 10, 11, 12, 12, 13, 13]
        dob = {1: "1990-03-01", 2: "2003-07-15"}
        m = make(rows)
        C = context.features(m, eds, dob)
        for i in range(len(rows)):
            sub = make(rows[:i])
            day = date.fromisoformat(rows[i][2]).toordinal()
            for side, pid in enumerate(rows[i][:2]):
                got = context.features_at(sub, eds[:i], dob, {pid: (eds[i], day)})[pid]
                np.testing.assert_allclose(got, C[side][i], err_msg=f"match {i} player {pid}")

    def test_unknown_edition_is_a_new_one(self):
        rows = [(1, 2, "2020-01-06", "completed", "hard", -1, -1, -1, -1)]
        m = make(rows)
        day = date.fromisoformat("2020-01-13").toordinal()
        got = context.features_at(m, [10], {}, {1: (None, day)})[1]
        self.assertEqual(got[context.FEATURES.index("gap1")], 1)
        self.assertAlmostEqual(got[context.FEATURES.index("load28")], 1 / 5)


def live(sets1, sets2, cur1, cur2, server="c1", adv="", status="live"):
    """A live_data details payload in Kalshi's shape (as captured 2026-11-15)."""
    def rounds(xs, ys, mine):
        out = []
        for k, (x, y) in enumerate(zip(xs, ys)):
            last = k == len(xs) - 1
            outcome = "ongoing" if last else ("winner" if (x > y) == mine else "loser")
            out.append({"score": x if mine else y, "outcome": outcome})
        return out
    return {"status": status, "competitor1_id": "c1", "competitor2_id": "c2", "server": server, "advantage": adv,
            "competitor1_round_scores": rounds(sets1, sets2, True), "competitor2_round_scores": rounds(sets1, sets2, False),
            "competitor1_current_round_score": cur1, "competitor2_current_round_score": cur2,
            "competitor1_statistics": {"service_points_won": 49, "service_points_lost": 17},
            "competitor2_statistics": {"service_points_won": 58, "service_points_lost": 34}}


class LiveStateTest(unittest.TestCase):
    def test_maps_by_competitor_id_not_order(self):
        d = live([6, 3, 5], [2, 6, 3], 30, 40, server="c1")
        s = kalshi.live_state(d, "c1", "c2")
        self.assertEqual((s.sets_a, s.sets_b, s.games_a, s.games_b, s.pts_a, s.pts_b, s.a_serving),
                         (1, 1, 5, 3, 2, 3, True))
        self.assertEqual((s.serve_a, s.serve_b), ((49, 66), (58, 92)))
        self.assertEqual(s.problem, "")
        r = kalshi.live_state(d, "c2", "c1")                 # market A is competitor 2
        self.assertEqual((r.sets_a, r.sets_b, r.games_a, r.games_b, r.pts_a, r.pts_b, r.a_serving),
                         (1, 1, 3, 5, 3, 2, False))
        self.assertEqual(r.serve_a, (58, 92))

    def test_advantage_and_tiebreak(self):
        s = kalshi.live_state(live([6, 4], [4, 4], 40, 40, adv="c2"), "c1", "c2")
        self.assertEqual((s.pts_a, s.pts_b), (3, 4))
        s = kalshi.live_state(live([6, 4], [4, 4], 40, 40, adv="c1"), "c1", "c2")
        self.assertEqual((s.pts_a, s.pts_b), (4, 3))
        s = kalshi.live_state(live([6, 4], [4, 4], 40, 50), "c1", "c2")    # Kalshi's AD = 50
        self.assertEqual((s.pts_a, s.pts_b, s.problem), (3, 4, ""))
        s = kalshi.live_state(live([6, 4], [4, 4], 50, 40), "c2", "c1")    # seen from the other side
        self.assertEqual((s.pts_a, s.pts_b), (3, 4))
        self.assertIn("unexpected", kalshi.live_state(live([6, 4], [4, 4], 50, 30), "c1", "c2").problem)
        s = kalshi.live_state(live([6, 6], [4, 6], 5, 3), "c1", "c2")      # tiebreak: raw points
        self.assertEqual((s.games_a, s.games_b, s.pts_a, s.pts_b), (6, 6, 5, 3))
        s = kalshi.live_state(live([6, 4], [4, 4], 40, 40, adv="somebody"), "c1", "c2")
        self.assertIn("advantage", s.problem)

    def test_refuses_what_it_cannot_read(self):
        self.assertIn("unexpected", kalshi.live_state(live([6, 2], [4, 1], 5, 0), "c1", "c2").problem)
        self.assertIn("ids", kalshi.live_state(live([6, 2], [4, 1], 0, 0), "c1", "zz").problem)
        self.assertIn("server", kalshi.live_state(live([6, 2], [4, 1], 0, 0, server=""), "c1", "c2").problem)
        s = kalshi.live_state(live([6], [4], 0, 0, status="not_started"), "c1", "c2")
        self.assertEqual((s.status, s.problem), ("not_started", ""))

    def test_between_sets(self):
        d = live([6, 0], [3, 0], 0, 0)                    # new set just started, 0-0
        s = kalshi.live_state(d, "c1", "c2")
        self.assertEqual((s.sets_a, s.sets_b, s.games_a, s.games_b), (1, 0, 0, 0))


if __name__ == "__main__":
    unittest.main()
