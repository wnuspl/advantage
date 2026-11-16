"""The live page's data (engine/livefeed.py), the live.db migration (recorder.migrate) and
the web server's live routes (scripts/web_server.py), on the saved Kalshi fixture."""
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
sys.path.insert(0, str(HERE))

from tennis.engine import Prediction, livefeed                           # noqa: E402
from tennis.engine.decide import Policy, decide_all                     # noqa: E402
from tennis.engine.live import load_pull, snapshots                     # noqa: E402
from tennis.engine.recorder import LIVE_SCHEMA, Recorder, migrate       # noqa: E402
from test_live import FIX, NoNetwork, resolver                          # noqa: E402
from test_recorder import FINAL_LIVE, GG, finished_event                # noqa: E402

import web_server                                                       # noqa: E402

NOW = datetime(2026, 11, 15, 22, 40, tzinfo=timezone.utc)


class Spread:
    name = "spread"

    def predict(self, q):
        return Prediction(p_a=0.6, p_raw=0.6, model=self.name, in_play=q.live is not None,
                          detail={"last_a": "2026-11-01", "last_b": "2026-11-02", "p_sd": 0.07})


class LiveFeedTest(NoNetwork):
    def setUp(self):
        super().setUp()
        self.tmp = Path(tempfile.mkdtemp())
        self.db = self.tmp / "live.db"
        self.pull = load_pull(FIX)
        rec = Recorder(self.db)
        wid = rec.start_watch("spread", Policy(model_weight=0.15), ["KXATPMATCH"], 60)
        rows = snapshots(self.pull, Spread(), resolver())
        rec.record_poll(wid, self.pull, rows, decide_all(rows, self.pull.books, Policy()))
        ev = rec.to_settle(self.all_events(rows) - {GG}, now=self.pull.as_of)[0]
        rec.settle(ev, finished_event(), FINAL_LIVE, seen_at="2026-11-15T22:35:00+00:00")
        rec.record_error(wid, "2026-11-15T22:39:00+00:00", "URLError: down")
        rec.close()
        self.n = len(rows)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        super().tearDown()

    @staticmethod
    def all_events(rows):
        return {r.event.event for r in rows}

    def view(self, **kw):
        con = livefeed.connect(self.db)
        try:
            return livefeed.live_view(con, NOW, **kw)
        finally:
            con.close()

    def test_view(self):
        v = self.view()
        self.assertEqual(v["watch"]["policy"]["model_weight"], 0.15)
        self.assertEqual(v["poll"]["as_of"], self.pull.as_of)
        self.assertEqual(v["error"]["error"], "URLError: down")          # newest poll failed: shown, not used
        self.assertEqual(len(v["matches"]), self.n)
        priced = [m for m in v["matches"] if not m["skip"]]
        self.assertTrue(priced)
        for m in priced:
            self.assertEqual((m["p_model"], m["p_sd"]), (0.6, 0.07))
            self.assertIn(m["tour_label"], livefeed.TOUR_LABEL.values())
            self.assertIsNotNone(m["decision"])
            self.assertEqual(len(m["hist"]), 1)
        st = v["settled"]
        self.assertEqual([s["event"] for s in st], [GG])
        self.assertEqual((st[0]["winner"], st[0]["final_score"]), ("a", "7-6 6-4"))
        gg = next(m for m in v["matches"] if m["event"] == GG)
        if gg["status"] == "not_started":                                # pre = the last poll before the start
            self.assertEqual(st[0]["pre"]["p_model"], 0.6)
        else:                                                            # first seen in play: no pre-match numbers
            self.assertIsNone(st[0]["pre"])
        json.dumps(v)                                                    # JSON-ready

    def test_windows(self):
        self.assertEqual(self.view(settled_hours=0.01)["settled"], [])
        self.assertTrue(all(m["hist"] == [] for m in self.view(history_hours=0.01)["matches"]))

    def test_thin_keeps_ends(self):
        pts = list(range(100))
        t = livefeed._thin(pts, 10)
        self.assertEqual((len(t), t[0], t[-1]), (10, 0, 99))
        self.assertEqual(livefeed._thin(pts[:5], 10), pts[:5])

    def test_read_only(self):
        con = livefeed.connect(self.db)
        with self.assertRaises(sqlite3.OperationalError):
            con.execute("DELETE FROM poll")
        con.close()


class MigrateTest(unittest.TestCase):
    def test_old_live_db_gains_p_sd(self):
        con = sqlite3.connect(":memory:")
        old = LIVE_SCHEMA.read_text(encoding="utf-8").replace(
            "  p_sd          REAL,", "").replace("p_sd", "removed_p_sd")
        con.executescript(old)
        self.assertNotIn("p_sd", {r[1] for r in con.execute("PRAGMA table_info(snapshot)")})
        migrate(con)
        migrate(con)                                                     # idempotent
        self.assertIn("p_sd", {r[1] for r in con.execute("PRAGMA table_info(snapshot)")})
        v = livefeed.live_view(_row_factory(con), NOW)                   # empty db: no poll yet
        self.assertEqual((v["poll"], v["matches"]), (None, []))


def _row_factory(con):
    con.row_factory = sqlite3.Row
    return con


class ServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.db = cls.tmp / "live.db"
        Recorder(cls.db).close()                                         # schema only, no polls
        cls.srv = web_server.ThreadingHTTPServer(
            ("127.0.0.1", 0), web_server.make_handler(cls.tmp / "nope", quiet=True, live_db=cls.db))
        cls.url = f"http://127.0.0.1:{cls.srv.server_address[1]}"
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.none = web_server.ThreadingHTTPServer(
            ("127.0.0.1", 0), web_server.make_handler(cls.tmp / "nope", quiet=True, live_db=cls.tmp / "missing.db"))
        cls.url_none = f"http://127.0.0.1:{cls.none.server_address[1]}"
        threading.Thread(target=cls.none.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        for s in (cls.srv, cls.none):
            s.shutdown()
            s.server_close()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def get(self, path, base=None):
        return urllib.request.urlopen((base or self.url) + path, timeout=5)

    def test_pages_and_nav(self):
        for path, marker in (("/", b"Live markets"), ("/live", b"Live markets"), ("/ratings", b"<table"),
                             ("/nav.js", b"topnav")):
            with self.get(path) as r:
                self.assertIn(marker, r.read(), path)
        for page in ("/", "/ratings"):
            with self.get(page) as r:
                self.assertIn(b'<script src="/nav.js"></script>', r.read())

    def test_api_live(self):
        with self.get("/api/live") as r:
            self.assertEqual(r.headers["Cache-Control"], "no-store")
            v = json.loads(r.read())
        self.assertEqual((v["poll"], v["matches"]), (None, []))

    def test_api_live_without_db(self):
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self.get("/api/live", self.url_none)
        self.assertEqual(cm.exception.code, 503)
        self.assertIn("no live.db", json.loads(cm.exception.read())["error"])


if __name__ == "__main__":
    unittest.main()
