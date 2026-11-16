"""Track B5: live scorecard on a hand-built live.db (in memory)."""
from __future__ import annotations

import math
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from tennis.engine.decide import Policy                               # noqa: E402
from tennis.engine.recorder import Recorder                           # noqa: E402
from tennis.engine.scorecard import feed_lag, format_cards, scorecard  # noqa: E402

T0 = datetime(2026, 11, 16, 12, 0, tzinfo=timezone.utc)


def ll(p, y):
    return -math.log(p if y else 1 - p)


class Db:
    def __init__(self):
        self.rec = Recorder(":memory:")
        self.con = self.rec.con
        self.polls = {}

    def watch(self, predictor):
        return self.rec.start_watch(predictor, Policy(), ["X"], 60)

    def event(self, ev):
        self.con.execute(
            "INSERT INTO event (event, series, tour, title, a_ticker, b_ticker, a_name, b_name, a_comp, b_comp, "
            "first_seen, last_seen) VALUES (?, 'X', 'atp', ?, ?, ?, 'A', 'B', 'ca', 'cb', '', '')",
            (ev, ev, ev + "-A", ev + "-B"))

    def snap(self, wid, minute, ev, status, pm, pk, skip=""):
        key = (wid, minute)
        if key not in self.polls:
            t = (T0 + timedelta(minutes=minute)).isoformat(timespec="seconds")
            self.polls[key] = self.con.execute("INSERT INTO poll (watch_id, as_of, done_at) VALUES (?, ?, ?)",
                                               (wid, t, t)).lastrowid
        self.con.execute("INSERT INTO snapshot (poll_id, event, status, skip, p_model, p_market) VALUES (?,?,?,?,?,?)",
                         (self.polls[key], ev, status, skip, pm, pk))

    def settle(self, ev, winner):
        self.con.execute("INSERT INTO settlement (event, seen_at, winner) VALUES (?, '', ?)", (ev, winner))


class ScorecardTest(unittest.TestCase):
    def test_pre_match_in_play_per_predictor(self):
        db = Db()
        w1, w2 = db.watch("m1"), db.watch("m2")
        for ev in ("E1", "E2", "E3"):
            db.event(ev)
        # E1, A wins: two pre-match snapshots (the last counts), then live
        db.snap(w1, 0, "E1", "not_started", 0.50, 0.60)
        db.snap(w1, 1, "E1", "not_started", 0.55, 0.65)
        db.snap(w1, 2, "E1", "live", 0.70, 0.80)
        db.snap(w1, 3, "E1", "live", 0.90, 0.95)
        db.snap(w1, 4, "E1", "not_started", 0.10, 0.10)      # a feed glitch after going live: ignored pre-match
        db.snap(w1, 5, "E1", "closed", 0.99, 0.99)           # closed: neither pre-match nor in play
        # E2, B wins; one unpriced snapshot
        db.snap(w1, 0, "E2", "not_started", 0.40, 0.30)
        db.snap(w1, 1, "E2", "not_started", 0.40, None, skip="no order book")
        # E3 not settled: ignored
        db.snap(w1, 0, "E3", "not_started", 0.5, 0.5)
        # a second predictor saw E2 only
        db.snap(w2, 0, "E2", "not_started", 0.20, 0.30)
        db.settle("E1", "a")
        db.settle("E2", "b")
        cards = {c.predictor: c for c in scorecard(db.con)}
        self.assertEqual(sorted(cards), ["m1", "m2"])
        c = cards["m1"]
        self.assertEqual((c.settled, c.pre_match.n, c.in_play.n), (2, 2, 1))
        self.assertAlmostEqual(c.pre_match.model, (ll(0.55, 1) + ll(0.40, 0)) / 2)
        self.assertAlmostEqual(c.pre_match.market, (ll(0.65, 1) + ll(0.30, 0)) / 2)
        self.assertAlmostEqual(c.in_play.model, (ll(0.70, 1) + ll(0.90, 1)) / 2)
        self.assertAlmostEqual(c.in_play.d, c.in_play.model - c.in_play.market)
        self.assertTrue(math.isnan(c.in_play.se))                         # one match: no SE
        self.assertEqual((cards["m2"].settled, cards["m2"].pre_match.n), (1, 1))
        self.assertIn("## m1: 2 settled", format_cards(list(cards.values())))
        self.assertEqual(format_cards([]), "no settled, priced matches in live.db yet")

    def test_feed_lag_detects_a_leading_market(self):
        # the market moves one poll before the model (the feed shows the score a poll late)
        import random
        rnd = random.Random(1)
        moves = [rnd.choice([-0.05, 0.0, 0.05]) for _ in range(200)]
        pm, pk = [0.5], [0.5]
        for i in range(1, 200):
            pk.append(pk[-1] + moves[i])
            pm.append(pm[-1] + moves[i - 1])
        t = lambda i: (T0 + timedelta(seconds=60 * i)).isoformat()      # noqa: E731
        lag = feed_lag({"E": [(t(i), pm[i], pk[i]) for i in range(200)]})
        self.assertEqual((lag.n_steps, lag.n_matches, lag.median_gap_s), (199, 1, 60.0))
        self.assertGreater(lag.corr[-1], 0.9)
        self.assertLess(abs(lag.corr[0]), 0.3)
        # a long gap (laptop asleep) breaks the series: the step across it is dropped
        rows = [(t(0), 0.5, 0.5), (t(1), 0.6, 0.6), (t(30), 0.7, 0.9)]
        self.assertEqual(feed_lag({"E": rows}).n_steps, 1)


if __name__ == "__main__":
    unittest.main()
