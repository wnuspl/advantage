"""/scorecard and /pnl (track C): engine.scorecard.scorecard_view and engine.paper.pnl_view
match scorecard() and the replays on the same fixture, the server routes serve them through
SlowCache, the cache refreshes only when due, and bad query parameters get a 400."""
from __future__ import annotations

import contextlib
import io
import json
import math
import shutil
import sqlite3
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE.parent / "scripts"))
sys.path.insert(0, str(HERE))

from tennis.engine import livefeed, paper                               # noqa: E402
from tennis.engine.decide import Policy, decide_all                     # noqa: E402
from tennis.engine.live import load_pull, snapshots                     # noqa: E402
from tennis.engine.recorder import Recorder                             # noqa: E402
from tennis.engine.scorecard import (BENCHMARK_MODEL_WEIGHT, running, scorecard,  # noqa: E402
                                     scorecard_view, _by_predictor)
from test_live import FIX, NoNetwork, resolver                          # noqa: E402
from test_livefeed import Spread                                        # noqa: E402
from test_paper_replay import a_wins                                    # noqa: E402
from test_scorecard import Db                                           # noqa: E402

import paper_replay                                                     # noqa: E402
import web_server                                                       # noqa: E402

POL = Policy(bankroll=100, model_weight=0.6, in_play=True)


def build_live_db(path: Path) -> None:
    """Two watches (different policies), two identical polls each of the saved Kalshi pull,
    every match settled with side A winning."""
    pull = load_pull(FIX)
    rec = Recorder(path)
    rows = snapshots(pull, Spread(), resolver())
    for pol in (POL, Policy(bankroll=50, model_weight=0.3, kelly=0.5, in_play=True)):
        wid = rec.start_watch("spread", pol, ["KXATPMATCH"], 20)
        rec.record_poll(wid, pull, rows, decide_all(rows, pull.books, pol))
    for ev in rec.con.execute("SELECT * FROM event").fetchall():
        rec.settle(dict(ev), a_wins(dict(ev)), None, seen_at="2026-11-16T00:00:00+00:00")
    rec.close()


class Fixture(NoNetwork):
    def setUp(self):
        super().setUp()
        self.tmp = Path(tempfile.mkdtemp())
        self.db = self.tmp / "live.db"
        build_live_db(self.db)
        self.con = livefeed.connect(self.db)

    def tearDown(self):
        self.con.close()
        shutil.rmtree(self.tmp, ignore_errors=True)
        super().tearDown()


class PnlViewTest(Fixture):
    def test_matches_the_replays(self):
        v = paper.pnl_view(self.con, bankroll=1000)
        vs = paper.variants(self.con, {"bankroll": 1000})
        self.assertEqual([x["name"] for x in v["variants"]], list(vs))
        for out, pf in zip(v["variants"], vs.values()):
            bets = paper.first_bets(self.con, pf)
            self.assertTrue(bets)
            self.assertEqual({b["event"] for b in out["bets"]}, set(bets))
            for b in out["bets"]:
                for k in ("n", "cost", "pnl", "won", "side", "ask", "ev"):
                    self.assertEqual(b[k], bets[b["event"]][k], k)
            tot = paper.totals(bets.values())["all"]
            st = out["stats"]["all"]
            self.assertEqual((st["n"], st["won"]), (tot["n"], tot["won"]))
            self.assertAlmostEqual(st["pnl"], tot["pnl"])
            self.assertAlmostEqual(st["cost"], tot["cost"])
            self.assertAlmostEqual(st["roi"], tot["pnl"] / tot["cost"])
            self.assertAlmostEqual(out["curve"][-1]["bankroll"], 1000 + tot["pnl"], places=3)
            self.assertEqual(out["curve"][0], {"t": None, "bankroll": 1000})
        pos, state = paper.sell_replay(self.con, paper.newest_policy(self.con, {"bankroll": 1000}))
        self.assertEqual(v["sell"]["entries"], len(pos))
        for r in v["sell"]["rules"]:
            self.assertAlmostEqual(r["pnl"], sum(s["pnl"] for s in state[r["rule"]].values()))
        self.assertEqual(v["unsettled_flagged"], 0)
        json.dumps(v, allow_nan=False)                       # browser-safe JSON

    def test_kelly_override_and_watch_policies_differ(self):
        full = paper.pnl_view(self.con, bankroll=1000)
        small = paper.pnl_view(self.con, bankroll=1000, kelly=0.05)
        n = lambda v, i: sum(b["n"] for b in v["variants"][i]["bets"])     # noqa: E731
        self.assertLess(n(small, 1), n(full, 1))
        self.assertEqual(small["policies"]["newest"]["kelly"], 0.05)
        # the two variants run different policies (model weight 0.6 / Kelly 0.25 vs 0.3 / 0.5)
        self.assertNotEqual(n(full, 1), n(full, 0))

    def test_cli_prints_the_same_totals(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            paper_replay.print_pnl(self.con, 1000)
        v = paper.pnl_view(self.con, bankroll=1000)
        for var in v["variants"]:
            a = var["stats"]["all"]
            self.assertIn(f"{a['n']} bets, {a['won']} won, staked ${a['cost']:.2f}", out.getvalue())

    def test_stats(self):
        bets = [{"cost": 10, "pnl": 5, "won": True, "ev": 1, "n": 15, "p_side": 0.7},
                {"cost": 20, "pnl": -20, "won": False, "ev": 2, "n": 30, "p_side": 0.6},
                {"cost": 10, "pnl": 10, "won": True, "ev": 1, "n": 20, "p_side": 0.55}]
        s = paper.stats(bets)
        r = -5 / 40
        se = math.sqrt(((5 - r * 10) ** 2 + (-20 - r * 20) ** 2 + (10 - r * 10) ** 2) * 3 / 2) / 40
        self.assertAlmostEqual(s["roi"], r)
        self.assertAlmostEqual(s["roi_se"], se)
        self.assertAlmostEqual(s["hit_se"], math.sqrt(2 / 3 * 1 / 3 / 3))
        self.assertAlmostEqual(s["pnl_sd_model"], math.sqrt(225 * .21 + 900 * .24 + 400 * .2475))
        one = paper.stats(bets[:1])
        self.assertIsNone(one["roi_se"])
        self.assertIsNone(paper.stats([])["roi"])


class ScorecardViewTest(unittest.TestCase):
    def db(self):
        db = Db()
        w = db.watch("m1")
        evs = {"KXATPMATCH-1": ("a", 0.6, 0.5), "KXWTAMATCH-2": ("b", 0.3, 0.4), "KXATPMATCH-3": ("a", 0.7, 0.8),
               "KXATPCHALLENGERMATCH-4": ("b", 0.45, 0.55)}
        for i, (ev, (win, pm, pk)) in enumerate(evs.items()):
            db.event(ev)
            db.snap(w, 0, ev, "not_started", pm, pk)
            db.snap(w, 1, ev, "live", pm + 0.05, pk)
            db.snap(w, 2, ev, "live", pm + 0.1, pk - 0.05)
            db.con.execute("INSERT INTO settlement (event, seen_at, winner) VALUES (?, ?, ?)",
                           (ev, f"2026-11-16T0{4 - i}:00:00+00:00", win))
        return db

    def test_view_equals_scorecard(self):
        db = self.db()
        v = scorecard_view(db.con)
        c = scorecard(db.con)[0]
        card = v["cards"][0]
        self.assertEqual(card["settled"], c.settled)
        for ph in ("pre_match", "in_play"):
            self.assertAlmostEqual(card[ph]["d"], getattr(c, ph).d)
            self.assertAlmostEqual(card[ph]["se"], getattr(c, ph).se)
            last = card["running"][ph][-1]
            self.assertEqual(last["n"], getattr(c, ph).n)
            self.assertAlmostEqual(last["d"], getattr(c, ph).d)
            self.assertAlmostEqual(last["se"], getattr(c, ph).se)
            self.assertIsNone(card["running"][ph][0]["se"])            # NaN -> null
        # settlement order (seen_at), not event order
        self.assertEqual([p["event"] for p in card["running"]["pre_match"]],
                         ["KXATPCHALLENGERMATCH-4", "KXATPMATCH-3", "KXWTAMATCH-2", "KXATPMATCH-1"])
        self.assertEqual(v["tours"], ["ATP", "ATP Challenger", "WTA"])
        self.assertEqual(v["benchmark_model_weight"], list(BENCHMARK_MODEL_WEIGHT))
        self.assertEqual(v["watch"]["model_weight"], Policy().model_weight)
        self.assertEqual([k for k, _ in card["lag"]["corr"]], [-2, -1, 0, 1, 2])
        json.dumps(v, allow_nan=False)

    def test_tour_filter(self):
        db = self.db()
        atp = scorecard(db.con, {"ATP"})[0]
        self.assertEqual(atp.settled, 2)
        v = scorecard_view(db.con, {"ATP", "WTA"})
        self.assertEqual(v["cards"][0]["settled"], 3)
        self.assertEqual(v["filter"], ["ATP", "WTA"])
        self.assertEqual(v["tours"], ["ATP", "ATP Challenger", "WTA"])     # every tour, for the buttons
        self.assertEqual(scorecard_view(db.con, {"WTA 125"})["cards"], [])

    def test_running_by_hand(self):
        db = self.db()
        pairs = _by_predictor(db.con)["m1"][0]
        r = running(pairs, "pre_match")
        ds = [p["this"] for p in r]
        self.assertAlmostEqual(r[1]["d"], (ds[0] + ds[1]) / 2)
        self.assertAlmostEqual(r[1]["se"], abs(ds[0] - ds[1]) / 2)


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class SlowCacheTest(Fixture):
    def test_refresh_only_when_due(self):
        clock, calls = FakeClock(), []
        cache = web_server.SlowCache(self.db, lambda con, key: calls.append(key) or {"n": len(calls)},
                                     min_gap=60, max_age=600, clock=clock)
        e = cache.get("k", wait=5)
        self.assertEqual(json.loads(e["body"]), {"n": 1})
        self.assertFalse(e["refreshing"])
        clock.t += 30                                       # inside min_gap: never
        cache.get("k", wait=5)
        clock.t += 100                                      # past min_gap, nothing settled: no
        cache.get("k", wait=5)
        self.assertEqual(len(calls), 1)
        w = sqlite3.connect(self.db)                        # a new settlement: refresh
        ev = w.execute("SELECT event FROM event LIMIT 1").fetchone()[0]
        w.execute("UPDATE settlement SET seen_at = '2026-11-17T00:00:00+00:00' WHERE event = ?", (ev,))
        w.commit()
        w.close()
        cache.get("k")
        self._drain(cache)
        self.assertEqual(len(calls), 2)
        self.assertEqual(json.loads(cache.get("k")["body"]), {"n": 2})
        clock.t += 601                                      # past max_age: refresh anyway
        cache.get("k")
        self._drain(cache)
        self.assertEqual(len(calls), 3)

    def _drain(self, cache):
        w = cache.worker
        if w is not None:
            w.join(5)

    def test_first_request_does_not_block_and_errors_keep_the_old_body(self):
        gate, clock = threading.Event(), FakeClock()
        state = {"fail": False}

        def compute(con, key):
            gate.wait(5)
            if state["fail"]:
                raise RuntimeError("boom")
            return {"ok": True}
        cache = web_server.SlowCache(self.db, compute, min_gap=0, max_age=0, clock=clock)
        self.assertIsNone(cache.get("k"))                  # computing: the route answers 202
        gate.set()
        self._drain(cache)
        self.assertEqual(json.loads(cache.get("k")["body"]), {"ok": True})
        self._drain(cache)
        state["fail"] = True
        clock.t += 1
        cache.get("k")
        self._drain(cache)
        e = cache.get("k")
        self.assertEqual(json.loads(e["body"]), {"ok": True})
        self.assertIn("boom", e["error"])

    def test_keeps_at_most_keep_keys(self):
        cache = web_server.SlowCache(self.db, lambda con, key: {"k": key}, keep=2)
        for k in ("a", "b", "c"):
            cache.get(k, wait=5)
        self.assertEqual(len(cache.cache), 2)
        self.assertNotIn("a", cache.cache)


class QueryTest(unittest.TestCase):
    def test_pnl_overrides(self):
        self.assertEqual(web_server.pnl_overrides(""), (1000.0, None))
        self.assertEqual(web_server.pnl_overrides("bankroll=500&kelly=0.1&mw=0.9"), (500.0, 0.1))
        self.assertEqual(web_server.pnl_overrides("bankroll=1e9&kelly=5"), (1e6, 1.0))     # clamped
        self.assertEqual(web_server.pnl_overrides("kelly="), (1000.0, None))              # blank = default
        for bad in ("bankroll=abc", "kelly=nan", "bankroll=inf"):
            with self.assertRaises(ValueError, msg=bad):
                web_server.pnl_overrides(bad)

    def test_tour_filter(self):
        self.assertEqual(web_server.tour_filter(""), ())
        self.assertEqual(web_server.tour_filter("tour=WTA&tour=ATP"), ("ATP", "WTA"))
        self.assertEqual(web_server.tour_filter("tour=ATP%20Challenger,WTA%20125"), ("ATP Challenger", "WTA 125"))
        with self.assertRaises(ValueError):
            web_server.tour_filter("tour=ITF")


class RoutesTest(Fixture):
    def setUp(self):
        super().setUp()
        self.srv = web_server.ThreadingHTTPServer(
            ("127.0.0.1", 0), web_server.make_handler(self.tmp / "nope", quiet=True, live_db=self.db, wait=10))
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()
        super().tearDown()

    def get(self, path):
        return urllib.request.urlopen(self.url + path, timeout=15)

    def status(self, path):
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self.get(path)
        return cm.exception.code

    def test_pages_and_nav(self):
        for path, title in (("/scorecard", "<title>Scorecard"), ("/pnl", "<title>Paper P&amp;L")):
            r = self.get(path)
            self.assertEqual(r.status, 200)
            self.assertIn("text/html", r.headers["Content-Type"])
            self.assertIn(title, r.read().decode("utf-8"))
        nav = self.get("/nav.js").read().decode("utf-8")
        self.assertIn('href: "/scorecard"', nav)
        self.assertIn('href: "/pnl"', nav)

    def test_api_scorecard(self):
        j = json.loads(self.get("/api/scorecard").read())
        self.assertEqual(j["data"], json.loads(json.dumps(scorecard_view(self.con))))
        self.assertIsNone(j["error"])
        self.assertIn("computed_at", j)
        j = json.loads(self.get("/api/scorecard?tour=WTA").read())
        self.assertEqual(j["data"]["filter"], ["WTA"])
        self.assertEqual(self.status("/api/scorecard?tour=nope"), 400)

    def test_api_pnl(self):
        j = json.loads(self.get("/api/pnl?bankroll=500&kelly=0.2").read())
        self.assertEqual(j["data"], json.loads(json.dumps(paper.pnl_view(self.con, bankroll=500, kelly=0.2))))
        j = json.loads(self.get("/api/pnl").read())
        self.assertEqual(j["data"]["bankroll"], 1000)
        self.assertIsNone(j["data"]["kelly"])
        for bad in ("bankroll=abc", "kelly=nan", "bankroll=inf"):
            self.assertEqual(self.status("/api/pnl?" + bad), 400, bad)

    def test_computing_answers_202(self):
        gate = threading.Event()
        slow = {"pnl": web_server.SlowCache(self.db, lambda con, key: gate.wait(5) and {"x": 1}),
                "scorecard": web_server.SlowCache(self.db, lambda con, key: {})}
        srv = web_server.ThreadingHTTPServer(
            ("127.0.0.1", 0), web_server.make_handler(self.tmp, quiet=True, live_db=self.db, slow=slow, wait=0))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            r = urllib.request.urlopen(f"http://127.0.0.1:{srv.server_address[1]}/api/pnl", timeout=5)
            self.assertEqual(r.status, 202)
            self.assertTrue(json.loads(r.read())["computing"])
            # the live board does not wait for it
            self.assertEqual(urllib.request.urlopen(f"http://127.0.0.1:{srv.server_address[1]}/api/live",
                                                    timeout=5).status, 200)
        finally:
            gate.set()
            srv.shutdown()
            srv.server_close()

    def test_missing_db(self):
        srv = web_server.ThreadingHTTPServer(
            ("127.0.0.1", 0), web_server.make_handler(self.tmp, quiet=True, live_db=self.tmp / "missing.db"))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            for path in ("/api/scorecard", "/api/pnl"):
                with self.assertRaises(urllib.error.HTTPError) as cm:
                    urllib.request.urlopen(f"http://127.0.0.1:{srv.server_address[1]}{path}", timeout=5)
                self.assertEqual(cm.exception.code, 503)
        finally:
            srv.shutdown()
            srv.server_close()


if __name__ == "__main__":
    unittest.main()
