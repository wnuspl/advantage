"""Track B2: live.db recorder (engine/recorder.py) and one watch poll (scripts/watch.py),
on the saved Kalshi fixture, in memory, no network."""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
import unittest
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

from tennis.engine import FixedPredictor, Prediction                    # noqa: E402
from tennis.engine.decide import Policy, decide_all                     # noqa: E402
from tennis.engine.live import load_pull, snapshots                     # noqa: E402
from tennis.engine.recorder import Recorder, settlement_from            # noqa: E402
from test_live import FIX, NoNetwork, by_event, resolver                # noqa: E402

GG = "KXATPCHALLENGERMATCH-26NOV15GOGGUR"
ALC = "KXATPMATCH-26NOV16ALCTIA"
C1, C2 = "f23e2325-d880-40b6-8eca-9185fce2543f", "9a22428f-791d-4e55-8bb9-dd44155e0141"


class Const:
    name = "const"

    def predict(self, q):
        return Prediction(p_a=0.97, p_raw=0.96, model=self.name, in_play=q.live is not None,
                          detail={"last_a": "2026-11-01", "last_b": "2026-05-20"})


def finished_event(result_a="yes", result_b="no", status="finalized"):
    """GET /events/<GOGGUR> once settled (shape as fetched 2026-11-15)."""
    return {"event": {"event_ticker": GG}, "markets": [
        {"ticker": f"{GG}-GOG", "status": status, "result": result_a, "settlement_value_dollars": "1.0000",
         "expiration_value": "Vignesh Gogineni"},
        {"ticker": f"{GG}-GUR", "status": status, "result": result_b, "settlement_value_dollars": "0.0000"}]}


FINAL_LIVE = {"status": "closed", "match_status": "ended", "winner": C1, "competitor1_id": C1, "competitor2_id": C2,
              "competitor1_round_scores": [{"outcome": "winner", "score": 7}, {"outcome": "winner", "score": 6}],
              "competitor2_round_scores": [{"outcome": "loser", "score": 6}, {"outcome": "loser", "score": 4}]}


class RecorderTest(NoNetwork):
    def setUp(self):
        super().setUp()
        self.pull = load_pull(FIX)
        self.rec = Recorder(":memory:")
        self.wid = self.rec.start_watch("const", Policy(), ["KXATPMATCH"], 60)

    def tearDown(self):
        self.rec.close()
        super().tearDown()

    def record(self, pull, predictor=None, pol=Policy(in_play=True)):
        rows = snapshots(pull, predictor or Const(), resolver())
        ds = decide_all(rows, pull.books, pol)
        return self.rec.record_poll(self.wid, pull, rows, ds), rows, ds

    def test_round_trip_and_replay(self):
        pid, rows, _ = self.record(self.pull)
        back = self.rec.load_poll(pid)
        self.assertEqual((back.as_of, back.events, back.live, back.books),
                         (self.pull.as_of, self.pull.events, self.pull.live, self.pull.books))
        again = snapshots(back, Const(), resolver())
        self.assertEqual([r.to_json() for r in again], [r.to_json() for r in rows])

    def test_blobs_deduplicate(self):
        self.record(self.pull)
        later = replace(copy.deepcopy(self.pull), as_of="2026-11-15T22:32:32+00:00")
        self.record(later)
        c = self.rec.counts()
        self.assertEqual((c["polls"], c["snapshots"], c["events"]), (2, 14, 7))
        self.assertEqual(self.rec.con.execute("SELECT COUNT(*) FROM blob").fetchone()[0], 4)

    def test_snapshot_tags_and_decisions(self):
        pid, rows, ds = self.record(self.pull)
        r = dict(self.rec.con.execute("SELECT * FROM snapshot WHERE event = ?", (ALC,)).fetchone())
        self.assertEqual((r["p_model"], r["p_raw"], r["a_resolved"], r["b_resolved"]), (0.97, 0.96, "exact", "exact"))
        self.assertEqual((r["a_last_rated"], r["a_stale_days"], r["b_stale_days"]), ("2026-11-01", 15, 180))
        self.assertEqual((r["day"], r["surface_from"], r["best_of"], r["status"], r["skip"]),
                         ("2026-11-16", "unknown", 3, "not_started", ""))
        g = dict(self.rec.con.execute("SELECT * FROM snapshot WHERE event = ?", (GG,)).fetchone())
        self.assertEqual((g["status"], g["sets_a"], g["games_a"], g["games_b"], g["a_serving"], g["in_play"]),
                         ("live", 1, 3, 2, 1, 1))
        m = dict(self.rec.con.execute("SELECT * FROM snapshot WHERE event LIKE '%MEKCHO'").fetchone())
        self.assertEqual((m["skip"], m["p_model"], m["day"]), ("unresolved: Nicholas Mekhael", None, None))
        dec = {x["event"]: dict(x) for x in self.rec.con.execute("SELECT * FROM decision")}
        self.assertEqual(len(dec), 7)
        bets = [d for d in ds if d.bet]
        self.assertTrue(bets)
        for d in bets:
            self.assertEqual((dec[d.event]["contracts"], dec[d.event]["side"]), (d.sizing.contracts, d.side))
            self.assertEqual(sum(f["contracts"] for f in json.loads(dec[d.event]["fills"])), d.sizing.contracts)
        e = dict(self.rec.con.execute("SELECT * FROM event WHERE event = ?", (GG,)).fetchone())
        self.assertEqual((e["a_name"], e["a_comp"], e["tournament"]), ("Vignesh Gogineni", C1, "ATP Challenger Drummondville"))

    def test_at_time_t(self):
        self.record(self.pull)
        later = copy.deepcopy(self.pull)
        later.as_of = "2026-11-15T23:00:00+00:00"
        for book in later.books.values():                       # move the market
            book["yes_dollars"] = [["0.0100", "5"]]
        self.record(later)
        self.assertIsNone(self.rec.at(ALC, "2026-11-15T22:00:00+00:00"))
        r1 = self.rec.at(ALC, "2026-11-15T22:45:00+00:00")
        r2 = self.rec.at(ALC, "2026-11-15T23:30:00+00:00")
        self.assertEqual((r1["as_of"], r2["as_of"]), (self.pull.as_of, later.as_of))
        self.assertNotEqual(r1["p_market"], r2["p_market"])
        self.assertEqual((r1["p_model"], r1["a_name"], r1["settlement"]), (0.97, "Carlos Alcaraz", None))
        self.assertIn("contracts", r1)

    def test_settlement(self):
        self.record(self.pull)
        self.assertEqual(self.rec.to_settle({e for e in self.all_events()}, now=self.pull.as_of), [])
        todo = self.rec.to_settle(self.all_events() - {GG}, now=self.pull.as_of)
        self.assertEqual([t["event"] for t in todo], [GG])
        ev = todo[0]
        self.assertIsNone(self.rec.settle(ev, finished_event(status="active", result_a="", result_b=""), None))
        st = self.rec.settle(ev, finished_event(), FINAL_LIVE, seen_at="2026-11-15T22:53:00+00:00")
        self.assertEqual((st["winner"], st["match_status"], st["final_score"], st["value_a"]), ("a", "ended", "7-6 6-4", 1.0))
        self.assertEqual(self.rec.to_settle(self.all_events() - {GG}, now=self.pull.as_of), [])
        self.assertEqual(self.rec.at(GG, "2026-11-16T00:00:00+00:00")["settlement"]["winner"], "a")
        self.assertEqual(self.rec.to_settle(set(), now="2026-12-15T00:00:00+00:00"), [])     # gave up after 15 days

    def test_settlement_reads(self):
        ev = {"a_ticker": f"{GG}-GOG", "b_ticker": f"{GG}-GUR", "a_comp": C2, "b_comp": C1}   # A is competitor 2 here
        st = settlement_from(ev, finished_event(), FINAL_LIVE)
        self.assertEqual(st["final_score"], "6-7 4-6")                       # A first, by competitor id
        self.assertIsNone(settlement_from({**ev, "a_comp": "x"}, finished_event(), FINAL_LIVE)["match_status"])
        void = settlement_from(ev, finished_event("no", "no"), None)
        self.assertEqual((void["winner"], void["final_score"]), (None, None))

    def test_errors_and_milestones(self):
        self.rec.record_error(self.wid, "2026-11-15T22:00:00+00:00", "URLError: down")
        self.assertEqual(self.rec.milestones(), {})
        pull = copy.deepcopy(self.pull)
        pull.milestones = {ALC: {"id": "m1"}, GG: None}
        self.record(pull)
        self.assertEqual(self.rec.milestones(), {ALC: {"id": "m1"}})
        with self.assertRaises(KeyError):
            self.rec.load_poll(1)                                  # the failed poll
        self.assertEqual(self.rec.counts()["failed_polls"], 1)

    def all_events(self):
        return {r[0] for r in self.rec.con.execute("SELECT event FROM event")}


def load_watch():
    spec = importlib.util.spec_from_file_location("watch_script", HERE.parent / "scripts" / "watch.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class PollOnceTest(NoNetwork):
    def test_poll_records_and_settles(self):
        watch = load_watch()
        first = load_pull(FIX)
        second = copy.deepcopy(first)                  # GOGGUR has left the open list
        second.as_of = "2026-11-15T23:00:00+00:00"
        second.events["KXATPCHALLENGERMATCH"] = [e for e in second.events["KXATPCHALLENGERMATCH"]
                                                 if e["event_ticker"] != GG]
        pulls = iter([first, second])
        seen_known = []

        def fake_fetch(series, known_milestones=None):
            seen_known.append(dict(known_milestones or {}))
            return next(pulls)
        watch.fetch_pull = fake_fetch
        watch.kalshi.get = lambda path, *a, **k: finished_event() if path == f"/events/{GG}" else {}
        watch.kalshi.live_batch = lambda ids: {}

        class Retr:
            def __call__(self, pull):
                return snapshots(pull, FixedPredictor(), resolver())
        rec = Recorder(":memory:")
        wid = rec.start_watch("fixed(0.5)", Policy(), ["x"], 60)
        cache = {"KXATPMATCH-26NOV16ALCTIA": {"id": "cached"}}
        line1 = watch.poll_once(rec, wid, Retr(), Policy(), ("x",), cache)
        line2 = watch.poll_once(rec, wid, Retr(), Policy(), ("x",), cache)
        self.assertEqual(seen_known[0], {"KXATPMATCH-26NOV16ALCTIA": {"id": "cached"}})
        self.assertIn("7 open", line1)
        self.assertIn("0 bets", line1)                 # placeholder: never bets
        self.assertIn("settled", line2)
        self.assertIn("Vignesh Gogineni", line2)
        c = rec.counts()
        self.assertEqual((c["polls"], c["settled"], c["bets"]), (2, 1, 0))
        rec.close()


if __name__ == "__main__":
    unittest.main()
