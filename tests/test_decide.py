"""Track B3a: decisions on one snapshot (engine/decide.py), on the saved Kalshi fixture."""
from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

from tennis import kalshi                                         # noqa: E402
from tennis.engine import FixedPredictor, Prediction             # noqa: E402
from tennis.engine.decide import Policy, decide, decide_all      # noqa: E402
from tennis.engine.live import load_pull, snapshots              # noqa: E402
from test_live import FIX, NoNetwork, by_event, resolver         # noqa: E402

ALC = "KXATPMATCH-26NOV16ALCTIA"            # Alcaraz v Tiafoe, market ~0.885 on Alcaraz, not started
LIVE = "KXATPCHALLENGERMATCH-26NOV15GOGGUR"


class Const:
    """A non-placeholder predictor: fixed P(A) per event (by player pair), else `default`."""
    name = "const"

    def __init__(self, default=0.5, by_pair=None, stale=False):
        self.default, self.by_pair, self.stale = default, by_pair or {}, stale

    def predict(self, q):
        p = self.by_pair.get((q.player_a, q.player_b), self.default)
        return Prediction(p_a=p, model=self.name, stale=self.stale, in_play=q.live is not None)


class DecideTest(NoNetwork):
    def setUp(self):
        super().setUp()
        self.pull = load_pull(FIX)
        self.base = by_event(snapshots(self.pull, FixedPredictor(), resolver()))

    def rows(self, predictor):
        return snapshots(self.pull, predictor, resolver())

    def pair(self, ev):
        e = self.base[ev].event
        return (e.a.player_id, e.b.player_id)

    def test_placeholder_never_bets(self):
        ds = decide_all(self.rows(FixedPredictor(0.5)), self.pull.books, Policy(in_play=True, allow_stale=True))
        self.assertEqual(len(ds), 7)
        self.assertFalse(any(d.bet for d in ds))
        reasons = sorted(d.reason for d in ds)
        self.assertEqual(reasons.count("placeholder predictor"), 6)
        self.assertEqual(reasons[0], "not priced: unresolved: Nicholas Mekhael")

    def test_echoing_the_market_bets_nothing(self):
        echo = Const(by_pair={self.pair(ev): r.p_market for ev, r in self.base.items() if not r.skip})
        ds = decide_all(self.rows(echo), self.pull.books, Policy(in_play=True, allow_stale=True, kelly=1.0))
        self.assertFalse(any(d.bet for d in ds))
        self.assertEqual(sum(d.reason == "no edge after spread and fees" for d in ds), 6)

    def test_edge_below_one_contract_says_so(self):
        """A real edge that Kelly sizes under one contract on a tiny bankroll is not 'no edge'."""
        r = by_event(self.rows(Const(0.97)))[ALC]
        self.assertTrue(decide(r, self.pull.books, Policy(bankroll=1000)).bet)
        d = decide(r, self.pull.books, Policy(bankroll=0.5))
        self.assertFalse(d.bet)
        self.assertTrue(d.reason.startswith("under one contract: Carlos Alcaraz +"), d.reason)

    def test_market_only_weight_bets_nothing(self):
        ds = decide_all(self.rows(Const(0.99)), self.pull.books, Policy(model_weight=0.0, in_play=True, kelly=1.0))
        self.assertFalse(any(d.bet for d in ds))

    def test_confident_model_bets_the_right_side(self):
        pol = Policy()
        r = by_event(self.rows(Const(0.97)))[ALC]
        d = decide(r, self.pull.books, pol)
        self.assertTrue(d.bet)
        self.assertEqual((d.side, d.name, d.reason), ("a", "Carlos Alcaraz", ""))
        e = r.event
        lad = kalshi.ladder(self.pull.books[e.a.ticker], e.a.ticker, self.pull.books[e.b.ticker], e.b.ticker)
        want = kalshi.kelly_walk(0.97, lad, pol.bankroll, pol.kelly, pol.fee_rate)
        self.assertEqual((d.sizing.contracts, d.sizing.cost), (want.contracts, want.cost))
        self.assertEqual(d.ask, lad[0].price)
        d = decide(by_event(self.rows(Const(0.5)))[ALC], self.pull.books, pol)   # Tiafoe at ~0.12 ask
        self.assertEqual((d.side, d.name), ("b", "Frances Tiafoe"))
        self.assertAlmostEqual(d.p_side, 0.5)

    def test_refusals(self):
        r = by_event(self.rows(Const(0.97, stale=True)))
        self.assertEqual(decide(r[ALC], self.pull.books, Policy()).reason, "stale rating")
        self.assertTrue(decide(r[ALC], self.pull.books, Policy(allow_stale=True)).bet)
        r = by_event(self.rows(Const(0.99)))
        self.assertEqual(decide(r[LIVE], self.pull.books, Policy()).reason, "in play (policy off)")
        self.assertNotEqual(decide(r[LIVE], self.pull.books, Policy(in_play=True)).reason, "in play (policy off)")
        books = copy.deepcopy(self.pull.books)
        del books[r[ALC].event.b.ticker]
        self.assertEqual(decide(r[ALC], books, Policy()).reason, "no order book")

    def test_caps(self):
        r = by_event(self.rows(Const(0.97)))[ALC]
        d = decide(r, self.pull.books, Policy(max_stake=25.0))
        self.assertTrue(0 < d.sizing.cost <= 25.0)
        rows = self.rows(Const(0.97))
        free = decide_all(rows, self.pull.books, Policy(kelly=1.0, max_total=10.0, in_play=True))
        capped = decide_all(rows, self.pull.books, Policy(kelly=1.0, max_total=0.1, in_play=True))
        total_free = sum(d.sizing.cost for d in free)
        total = sum(d.sizing.cost for d in capped)
        self.assertGreater(total_free, 100.0)
        self.assertLessEqual(total, 100.0 + 1e-9)
        self.assertGreater(total, 90.0)                             # pro rata, floored per fill
        for d in capped:
            if not d.bet:
                self.assertIsNone(d.side)

    def test_json(self):
        d = decide(by_event(self.rows(Const(0.97)))[ALC], self.pull.books, Policy())
        js = d.to_json()
        self.assertEqual((js["side"], js["contracts"]), ("a", d.sizing.contracts))
        self.assertEqual(sum(f["contracts"] for f in js["fills"]), d.sizing.contracts)


if __name__ == "__main__":
    unittest.main()
