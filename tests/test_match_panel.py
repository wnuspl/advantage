"""The live page's match panel: break / set-end detection from recorded scores
(engine/livefeed.score_steps, match_marks), one match's view (livefeed.match_view) and the
web server's /api/match route, on a small hand-made live.db."""
from __future__ import annotations

import json
import shutil
import sqlite3
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE.parent / "scripts"))

from tennis.engine import livefeed                                       # noqa: E402
from tennis.engine.recorder import Recorder                             # noqa: E402

import web_server                                                       # noqa: E402

EV = "KXATPMATCH-26NOV16AAABBB"
EV2 = "KXWTAMATCH-26NOV16CCCDDD"


def sc(sets_a, sets_b, ga, gb, a_srv, done="", extra="0-0", t=0, status="live"):
    """One recorded poll: sets, games, server, and the score string the recorder stores."""
    s = f"{done} {ga}-{gb} {extra} *{'A' if a_srv else 'B'} serving".strip()
    return {"as_of": f"2026-11-16T12:{t:02d}:00+00:00", "status": status, "sets_a": sets_a, "sets_b": sets_b,
            "games_a": ga, "games_b": gb, "a_serving": int(a_srv), "score": s}


# A first set with a hold, a break each way, a feed glitch, a hold to close the set, then a
# second set the polls skip through (4 games in one step).
SET1 = [sc(0, 0, 0, 0, True, t=0),
        sc(0, 0, 1, 0, False, t=1),            # A held
        sc(0, 0, 2, 0, True, t=2),             # A broke B
        sc(0, 0, 2, 1, False, t=3),            # B broke A
        sc(0, 0, 2, 0, True, t=4),             # glitch: games went back -> ignored
        sc(0, 0, 3, 1, True, t=5),             # from 2-1 B serving: A broke B
        sc(0, 0, 5, 3, True, t=6),             # 2 games missed: A held, B held (no forced break)
        sc(1, 0, 0, 0, False, "6-3", t=7),     # A served 5-3 and won: set to A, no break
        sc(1, 0, 3, 1, False, "6-3", t=9)]     # 4 games, B served first: A won 3 of them -> >= 1 break


class ScoreStepsTest(unittest.TestCase):
    def kinds(self, polls, final=None):
        return [(m["i"], m["kind"], m["side"], m.get("exact", m.get("games")))
                for m in livefeed.match_marks(polls, final)]

    def test_breaks_and_set_ends(self):
        self.assertEqual(self.kinds(SET1), [
            (2, "break", "a", True), (3, "break", "b", True), (5, "break", "a", True),
            (7, "set", "a", [6, 3]), (8, "break", "a", False)])
        approx = livefeed.match_marks(SET1)[-1]
        self.assertEqual((approx["n"], approx["as_of"]), (1, SET1[8]["as_of"]))

    def test_hold_is_not_a_break(self):
        self.assertEqual(livefeed.score_steps(sc(0, 0, 0, 0, True), sc(0, 0, 1, 0, False)), [])
        self.assertEqual(livefeed.score_steps(sc(0, 0, 0, 0, False), sc(0, 0, 0, 1, True)), [])
        self.assertEqual(livefeed.score_steps(sc(0, 0, 0, 0, False), sc(0, 0, 1, 0, True)),
                         [{"kind": "break", "side": "a", "n": 1, "exact": True}])

    def test_tiebreak_is_not_a_break(self):
        tb = sc(1, 0, 6, 6, True, "6-3", "4-5 (tiebreak)")
        after = sc(1, 1, 0, 0, False, "6-3 6-7")
        self.assertEqual(livefeed.score_steps(tb, after), [{"kind": "set", "side": "b", "set": 2, "games": [6, 7]}])
        # a 7-6 set inside a stretch of missed games: the set end only, no break guessed
        self.assertEqual([m["kind"] for m in livefeed.score_steps(sc(0, 0, 5, 5, True), sc(1, 0, 1, 0, False, "7-6"))],
                         ["set"])

    def test_unreadable_steps_mark_nothing(self):
        # serve order that does not add up after one game
        self.assertEqual(livefeed.score_steps(sc(0, 0, 0, 0, False), sc(0, 0, 1, 0, False)), [])
        # a whole set skipped
        self.assertEqual(livefeed.score_steps(sc(0, 0, 2, 2, True), sc(2, 0, 0, 0, True, "6-2 6-2")), [])
        # server unknown
        nos = {**sc(0, 0, 0, 0, True), "a_serving": None}
        self.assertEqual(livefeed.score_steps(nos, sc(0, 0, 1, 0, False)), [])

    def test_final_set_from_the_settlement(self):
        polls = [sc(1, 1, 4, 4, True, "6-3 6-7", t=0)]
        got = livefeed.match_marks(polls, "6-3 6-7 6-4")
        self.assertEqual([(m["kind"], m["side"]) for m in got], [("set", "a"), ("break", "a")])
        self.assertFalse(got[1]["exact"])                       # A won 2 games, served 1: >= 1 break
        self.assertEqual(livefeed.match_marks(polls, "6-3 6-7"), [])     # no set left to add

    def test_set_scores(self):
        self.assertEqual(livefeed.set_scores("6-7 6-2 0-0 0-0 *A serving", 2), [(6, 7), (6, 2)])
        self.assertEqual(livefeed.set_scores("6-6 1-5 (tiebreak) *A serving", 0), [])
        self.assertEqual(livefeed.set_scores("6-7 6-2 5-7"), [(6, 7), (6, 2), (5, 7)])

    def test_valid_event(self):
        for ok in (EV, "KXATPCHALLENGERMATCH-26NOV15MATSUN"):
            self.assertTrue(livefeed.valid_event(ok))
        for bad in ("", None, "kxatp-lower", "KX ATP", "KX/../x", "A" * 81, "K'X", "KX%27"):
            self.assertFalse(livefeed.valid_event(bad), bad)


def make_db(path: Path) -> None:
    """A watch with two events: EV played live and settled (with a prematch row), EV2 still
    upcoming (no prematch row)."""
    Recorder(path).close()
    con = sqlite3.connect(path)
    con.execute("INSERT INTO watch VALUES (1, '2026-11-16T11:00:00+00:00', 'points_v1/run790', '{}', "
                "'KXATPMATCH,KXWTAMATCH', 20, NULL, NULL)")
    for ev, series, a, b in ((EV, "KXATPMATCH", "Ann Aaa", "Bea Bbb"), (EV2, "KXWTAMATCH", "Cy Ccc", "Di Ddd")):
        con.execute("INSERT INTO event VALUES (?,?,?,?,?,?,?,?,?,?,?,?,NULL,?,?)",
                    (ev, series, series[2:5].lower(), f"{a} vs {b}", "Test Open", "2026-11-16T12:00:00+00:00",
                     ev + "-A", ev + "-B", a, b, "ca", "cb", "2026-11-16T11:50:00+00:00", "2026-11-16T12:09:00+00:00"))
    pre = [{"as_of": "2026-11-16T11:50:00+00:00", "status": "not_started"},
           {"as_of": "2026-11-16T11:55:00+00:00", "status": "not_started"}]
    cols = ("status", "skip", "p_market", "score", "sets_a", "sets_b", "games_a", "games_b", "a_serving",
            "serve_a_won", "serve_a_played", "serve_b_won", "serve_b_played", "p_model", "p_sd")
    for pid, p in enumerate(pre + SET1, start=1):
        con.execute("INSERT INTO poll (poll_id, watch_id, as_of, done_at, n_events) VALUES (?,1,?,?,2)",
                    (pid, p["as_of"], p["as_of"]))
        live = p["status"] == "live"
        vals = {"status": p["status"], "skip": "", "p_market": 0.55 if not live else 0.6, "p_model": 0.58 + pid / 100,
                "p_sd": 0.05, "serve_a_won": 10 + pid if live else None, "serve_a_played": 15 + pid if live else None,
                "serve_b_won": 8 if live else None, "serve_b_played": 14 if live else None,
                **{k: p.get(k) for k in ("score", "sets_a", "sets_b", "games_a", "games_b", "a_serving")}}
        con.execute(f"INSERT INTO snapshot (poll_id, event, {', '.join(cols)}) VALUES (?,?,{', '.join('?' * len(cols))})",
                    (pid, EV, *(vals[c] for c in cols)))
        con.execute("INSERT INTO snapshot (poll_id, event, status, skip, p_market, p_model) VALUES (?,?,?,?,?,?)",
                    (pid, EV2, "not_started", "", 0.4, 0.45))
    det = {"rating_a": {"serve": 0.12, "ret": 0.03, "serve_sd": 0.1, "ret_sd": 0.11, "seen": True},
           "rating_b": {"serve": -0.04, "ret": -0.02, "serve_sd": 0.12, "ret_sd": 0.13, "seen": True},
           "base": 0.45, "x_a": 0.59, "x_b": 0.38, "ctx_a": 0.0, "ctx_b": -0.01, "p_prematch": 0.6,
           "p_sd_pre": 0.051234567, "last_a": "2026-11-01", "last_b": "2026-10-20"}
    con.execute("INSERT INTO prematch VALUES (?, 2, 'points_v1/run790', 'not_started', ?)", (EV, json.dumps(det)))
    con.execute("INSERT INTO settlement (event, seen_at, winner, final_score, match_status) VALUES (?,?,?,?,?)",
                (EV, "2026-11-16T13:30:00+00:00", "a", "6-3 6-4", "ended"))
    con.commit()
    con.close()


class MatchViewTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.db = cls.tmp / "live.db"
        make_db(cls.db)
        cls.srv = web_server.ThreadingHTTPServer(
            ("127.0.0.1", 0), web_server.make_handler(cls.tmp / "nope", quiet=True, live_db=cls.db))
        cls.url = f"http://127.0.0.1:{cls.srv.server_address[1]}"
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def view(self, ev):
        con = livefeed.connect(self.db)
        try:
            return livefeed.match_view(con, ev, datetime(2026, 11, 16, 14, tzinfo=timezone.utc))
        finally:
            con.close()

    def test_settled_match(self):
        v = self.view(EV)
        self.assertEqual((v["status"], v["a"]["name"], v["tour_label"]), ("settled", "Ann Aaa", "ATP"))
        self.assertEqual(len(v["polls"]), 2 + len(SET1))                # every poll, not thinned
        self.assertEqual(v["started_at"], SET1[0]["as_of"])
        # marks index the full poll list (2 pre-match polls first); the settlement adds set 2
        self.assertEqual([(m["i"], m["kind"], m["side"]) for m in v["marks"]],
                         [(4, "break", "a"), (5, "break", "b"), (7, "break", "a"), (9, "set", "a"),
                          (10, "break", "a"), (10, "set", "a")])
        pre = v["pre"]
        self.assertEqual((pre["p_model"], pre["p_market"], pre["as_of"]), (0.6, 0.55, "2026-11-16T11:55:00+00:00"))
        self.assertEqual(pre["ratings"]["rating_a"]["serve"], 0.12)
        self.assertEqual(pre["ratings"]["p_sd_pre"], 0.0512)            # rounded
        self.assertEqual(v["settlement"]["final_score"], "6-3 6-4")
        self.assertEqual(v["polls"][-1]["serve_a_won"], 10 + 2 + len(SET1))
        json.dumps(v)

    def test_upcoming_and_unknown(self):
        v = self.view(EV2)
        self.assertEqual((v["status"], v["started_at"], v["marks"], v["pre"]["ratings"]), ("not_started", None, [], None))
        self.assertEqual(v["pre"]["p_model"], 0.45)
        self.assertIsNone(self.view("KXATPMATCH-NOPE"))

    def test_old_live_db_without_prematch_table(self):
        tmp = self.tmp / "old.db"
        shutil.copy(self.db, tmp)
        con = sqlite3.connect(tmp)
        con.execute("DROP TABLE prematch")
        con.commit()
        con.close()
        con = livefeed.connect(tmp)
        try:
            self.assertIsNone(livefeed.match_view(con, EV)["pre"]["ratings"])
        finally:
            con.close()

    def get(self, path):
        return urllib.request.urlopen(self.url + path, timeout=5)

    def test_api_match(self):
        with self.get(f"/api/match?event={EV}") as r:
            self.assertEqual(r.headers["Cache-Control"], "no-store")
            v = json.loads(r.read())
        self.assertEqual((v["event"], len(v["marks"])), (EV, 6))
        for bad, code in (("", 400), ("?event=", 400), ("?event=bad%20one", 400), ("?event=KX'%3B--", 400),
                          ("?event=KXATPMATCH-26NOV16ZZZ", 404)):
            with self.assertRaises(urllib.error.HTTPError, msg=bad) as cm:
                self.get("/api/match" + bad)
            self.assertEqual(cm.exception.code, code, bad)
            self.assertIn("error", json.loads(cm.exception.read()))

    def test_cache(self):
        c = web_server.MatchCache(self.db, ttl=60, keep=1)
        a = c.get(EV)
        self.assertIs(c.get(EV), a)                                      # within ttl: same bytes
        self.assertIsNone(c.get("KXATPMATCH-NOPE"))
        self.assertEqual(list(c.cache), ["KXATPMATCH-NOPE"])             # keep=1: oldest dropped
        with self.assertRaises(ValueError):
            c.get("../etc")

    def test_api_match_without_db(self):
        srv = web_server.ThreadingHTTPServer(
            ("127.0.0.1", 0), web_server.make_handler(self.tmp / "nope", quiet=True, live_db=self.tmp / "missing.db"))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            with self.assertRaises(urllib.error.HTTPError) as cm:
                urllib.request.urlopen(f"http://127.0.0.1:{srv.server_address[1]}/api/match?event={EV}", timeout=5)
            self.assertEqual(cm.exception.code, 503)
        finally:
            srv.shutdown()
            srv.server_close()


if __name__ == "__main__":
    unittest.main()
