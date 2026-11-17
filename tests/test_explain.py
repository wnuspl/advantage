"""The match panel's "Why" tab: the model's price built step by step (engine/explain.py via
Predictor.predict's detail), the recorder's latest-explain rule, and livefeed's reasons to
doubt it, policy text and why_view."""
from __future__ import annotations

import json
import shutil
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

from tennis.engine import livefeed                                       # noqa: E402
from tennis.engine.predict import PointsPredictor, live_from_score        # noqa: E402
from tennis.engine.predictor import MatchQuery                           # noqa: E402
from tennis.engine.recorder import Recorder                             # noqa: E402

from test_predict import ROWS, predictor                                 # noqa: E402

EV = "KXATPCHALLENGERMATCH-26NOV17AAACCC"
NAMES = {"a": "Ann Aaa", "b": "Cy Van Ccc"}


class ExplainStepsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pp, _ = predictor(ROWS, len(ROWS))
        cls.q = MatchQuery(tour="atp", player_a=1, player_b=3, on="2020-03-05", surface="hard", edition_id=6)

    def test_prematch_steps_add_up_to_the_price(self):
        pr = self.pp.predict(self.q)
        ex = pr.detail["explain"]
        steps = ex["steps"]
        self.assertEqual(steps[0]["key"], "start")
        self.assertAlmostEqual(steps[0]["p"], 0.5, places=6)              # two tour-average players
        keys = [s["key"] for s in steps]
        self.assertEqual(keys[:4], ["start", "serve", "return", "surface"])
        self.assertEqual(keys[-2:], ["doubt", "calib"])
        self.assertTrue(all(s["key"] == "ctx" for s in steps[4:-2]))
        self.assertGreater(len(steps), 6)                                  # this fixture has context offsets
        for prev, s in zip(steps, steps[1:]):
            self.assertAlmostEqual(s["d"], s["p"] - prev["p"])
        self.assertAlmostEqual(steps[0]["p"] + sum(s["d"] for s in steps[1:]), pr.p_a, places=12)
        self.assertEqual(next(s for s in steps if s["key"] == "doubt")["p"], pr.p_raw)   # context fully applied
        self.assertEqual(ex["p"], pr.p_a)
        self.assertFalse(ex["in_play"])
        for s in steps:
            if s["key"] == "ctx":
                self.assertIn(s["group"], ("rust", "load", "age"))
                self.assertIn(s["side"], ("a", "b"))

    def test_in_play_adds_score_and_each_stream(self):
        ls = live_from_score("6-4 2-3", "30-40", False, serve_a=(31, 45), serve_b=(25, 42))
        pr = self.pp.predict(MatchQuery(**{**self.q.__dict__, "live": ls}))
        steps = pr.detail["explain"]["steps"]
        self.assertEqual([s["key"] for s in steps[-3:]], ["score", "form_a", "form_b"])
        self.assertEqual(steps[-1]["p"], pr.p_a)
        self.assertAlmostEqual(steps[-2]["form"], pr.detail["x_a_live"] - pr.detail["x_a"])
        self.assertEqual(steps[-1]["pts"], [25, 42])
        self.assertTrue(pr.detail["explain"]["in_play"])

    def test_switched_off(self):
        pp = PointsPredictor(self.pp.model, explain=False)
        self.assertNotIn("explain", pp.predict(self.q).detail)


def step(key, p, d, **kw):
    return {"key": key, "p": p, "d": d, **kw}


EXPLAIN = {"steps": [step("start", 0.5, None), step("serve", 0.56, 0.06), step("return", 0.58, 0.02),
                     step("surface", 0.58, 0.0), step("ctx", 0.62, 0.04, side="b", group="rust", feat={"back1": 1.0}),
                     step("doubt", 0.61, -0.01), step("calib", 0.6, -0.01)], "p": 0.6, "in_play": False}


class RecorderExplainTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.rec = Recorder(self.tmp / "live.db")
        c = self.rec.con
        c.execute("INSERT INTO watch VALUES (1, '2026-11-17T09:00:00+00:00', 'm', '{}', 'KX', 20, NULL, NULL)")
        for i in range(1, 5):
            c.execute("INSERT INTO poll (poll_id, watch_id, as_of, done_at) VALUES (?, 1, ?, ?)",
                      (i, f"2026-11-17T10:0{i}:00+00:00", f"2026-11-17T10:0{i}:00+00:00"))
        c.execute("INSERT INTO event VALUES (?, 'KXATPCHALLENGERMATCH', 'atp', 't', NULL, NULL, 'A', 'C', ?, ?, "
                  "'ca', 'cb', NULL, 'x', 'x')", (EV, NAMES["a"], NAMES["b"]))

    def tearDown(self):
        self.rec.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def snap(self, status, in_play, tag):
        return NS(status=status, event=NS(event=EV),
                  prediction=NS(model="m", in_play=in_play, detail={"explain": {**EXPLAIN, "tag": tag, "in_play": in_play}}))

    def row(self):
        r = self.rec.con.execute("SELECT poll_id, status, in_play, detail FROM explain WHERE event = ?", (EV,)).fetchone()
        return None if r is None else (r[0], r[1], r[2], json.loads(r[3])["tag"])

    def test_latest_wins_but_not_over_an_in_play_one(self):
        self.rec._explain(1, self.snap("not_started", False, "pre1"))
        self.assertEqual(self.row(), (1, "not_started", 0, "pre1"))
        self.rec._explain(2, self.snap("live", True, "live2"))
        self.assertEqual(self.row(), (2, "live", 1, "live2"))
        self.rec._explain(3, self.snap("live", False, "live-no-score"))      # priced without the live score
        self.rec._explain(4, self.snap("closed", False, "closed"))
        self.assertEqual(self.row(), (2, "live", 1, "live2"))
        self.rec._explain(4, NS(status="live", event=NS(event=EV), prediction=None))
        self.assertEqual(self.row()[3], "live2")


class DoubtsAndPolicyTest(unittest.TestCase):
    SNAP = {"p_model": 0.6, "p_market": 0.5, "p_sd": 0.08, "status": "not_started",
            "a_stale_days": 10, "b_stale_days": 175}

    def kinds(self, **kw):
        a = {"snap": self.SNAP, "explain": EXPLAIN, "ratings": None, "series": "KXATPMATCH", "side": "a", **kw}
        return [d["kind"] for d in livefeed.doubts(NAMES, a["snap"], a["explain"], a["ratings"], a["series"], a["side"])]

    def test_doubts(self):
        # B stale (175 > 120); gap 10 pts inside 2 SD (16); rust on B worth 4 of 10 pts (>= a third)
        self.assertEqual(self.kinds(), ["stale", "sd", "rust"])
        ds = livefeed.doubts(NAMES, self.SNAP, EXPLAIN, None, "KXATPMATCH", "a")
        self.assertIn("175 days", ds[0]["text"])
        self.assertIn("inside 2 SD", ds[1]["text"])
        self.assertIn("Van Ccc", ds[2]["text"])                           # the page's lastName rule
        self.assertEqual(self.kinds(snap={**self.SNAP, "b_stale_days": 90}), ["old", "sd", "rust"])
        new = {"rating_a": {"seen": False}, "rating_b": {"seen": True}}
        self.assertEqual(self.kinds(ratings=new)[0], "new")
        self.assertIn("data", self.kinds(series="KXATPCHALLENGERMATCH"))
        self.assertEqual(self.kinds(snap={**self.SNAP, "status": "live"})[-1], "lag")
        # a wide gap (30 pts) is outside 2 SD, and 4 pts of rust is under a third of it
        wide = {**self.SNAP, "p_model": 0.8}
        sd_text = livefeed.doubts(NAMES, wide, EXPLAIN, None, "KXATPMATCH", "a")[1]["text"]
        self.assertNotIn("inside 2 SD", sd_text)
        self.assertNotIn("rust", self.kinds(snap=wide))
        # rust that favours the other side is no reason to doubt this one
        self.assertNotIn("rust", self.kinds(side="b"))

    def test_policy_text(self):
        pol = {"model_weight": 0.5, "kelly": 0.25, "bankroll": 100}
        bet = {"side": "b", "p_use": 0.255, "p_side": 0.745, "ask": 0.7, "contracts": 8, "cost": 5.6, "fees": 0.12,
               "ev": 0.48, "reason": ""}
        t = livefeed.policy_text(NAMES, bet, pol)
        self.assertIn("weight 0.5 on the model", t)
        self.assertIn("25.5% for Aaa, so 74.5% for Van Ccc", t)
        self.assertIn("buys 8 for $5.60 plus $0.12 fees, expected profit $0.48", t)
        self.assertIn("0.25 Kelly on a $100 bankroll", t)
        self.assertIn("model alone", livefeed.policy_text(NAMES, bet, {**pol, "model_weight": 1}))
        on_a = livefeed.policy_text(NAMES, {**bet, "side": "a", "p_use": 0.7, "p_side": 0.7}, pol)
        self.assertIn("70.0% for Aaa. The cheapest Aaa contract", on_a)            # no repeat for side A
        nob = {**bet, "contracts": 0, "reason": "no edge after spread and fees"}
        self.assertTrue(livefeed.policy_text(NAMES, nob, pol).endswith("No bet: no edge after spread and fees."))
        self.assertEqual(livefeed.policy_text(NAMES, {"p_use": None, "reason": "stale rating"}, pol), "No bet: stale rating.")
        self.assertIsNone(livefeed.policy_text(NAMES, None, pol))


class WhyViewTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.db = self.tmp / "live.db"
        rec = Recorder(self.db)
        c = rec.con
        c.execute("INSERT INTO watch VALUES (1, '2026-11-17T09:00:00+00:00', 'm', ?, 'KX', 20, NULL, NULL)",
                  (json.dumps({"bankroll": 100, "kelly": 0.25, "model_weight": 0.5, "in_play": True}),))
        c.execute("INSERT INTO poll (poll_id, watch_id, as_of, done_at) VALUES (1, 1, ?, ?)",
                  ("2026-11-17T10:00:00+00:00",) * 2)
        c.execute("INSERT INTO event VALUES (?, 'KXATPMATCH', 'atp', 't', NULL, NULL, 'A', 'C', ?, ?, "
                  "'ca', 'cb', NULL, 'x', 'x')", (EV, NAMES["a"], NAMES["b"]))
        c.execute("""INSERT INTO snapshot (poll_id, event, status, skip, p_model, p_market, p_sd, a_stale_days, b_stale_days)
                     VALUES (1, ?, 'not_started', '', 0.6, 0.5, 0.08, 10, 30)""", (EV,))
        c.execute("""INSERT INTO decision VALUES (1, ?, 'a', 0.55, 0.55, 0.51, 3, 1.53, 0.05, 0.12, '[]', '')""", (EV,))
        c.execute("INSERT INTO explain VALUES (?, 1, 'm', 'not_started', 0, ?)", (EV, json.dumps(EXPLAIN)))
        c.commit()
        rec.close()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_view(self):
        con = livefeed.connect(self.db)
        try:
            v = livefeed.match_view(con, EV)["why"]
        finally:
            con.close()
        self.assertEqual((v["side"], v["p_model"], v["p_market"], v["in_play"]), ("a", 0.6, 0.5, False))
        self.assertEqual(len(v["explain"]["steps"]), 7)
        self.assertEqual(v["decision"]["contracts"], 3)
        self.assertIn("buys 3 for $1.53", v["policy_text"])
        # B's rust raises P(A) by 4 of the 10 pts the model sees over the market: a doubt
        self.assertEqual([d["kind"] for d in v["doubts"]], ["sd", "rust"])
        self.assertEqual(v["policy"]["model_weight"], 0.5)

    def test_missing(self):
        con = sqlite3.connect(self.db)
        con.execute("DELETE FROM explain")
        con.commit()
        con.close()
        con = livefeed.connect(self.db)
        try:
            self.assertIsNone(livefeed.match_view(con, EV)["why"])
        finally:
            con.close()
        con = sqlite3.connect(self.db)
        con.execute("DROP TABLE explain")
        con.commit()
        con.close()
        con = livefeed.connect(self.db)
        try:
            self.assertIsNone(livefeed.match_view(con, EV)["why"])
        finally:
            con.close()


if __name__ == "__main__":
    unittest.main()
