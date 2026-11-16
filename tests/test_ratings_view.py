"""Ratings viewer: weekly standings are point-in-time, the export and the server."""
from __future__ import annotations

import gzip
import json
import sqlite3
import sys
import threading
import unittest
import urllib.request
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE.parent / "scripts"))
sys.path.insert(0, str(HERE))

from tennis import matches, models_db, points_rating as pr                 # noqa: E402
from tennis.engine import standings                                        # noqa: E402
from test_predict import P790                                              # noqa: E402

import ratings_export                                                      # noqa: E402
import ratings_server                                                      # noqa: E402

PARAMS = pr.PointsParams(**{**P790.as_dict(), "score_weight": 1.5})
CUTOFFS = standings.mondays("2020-01-01", "2020-02-01")   # fixture editions start 2020-01-06 and 2020-01-20


def players_of(con) -> dict:
    return {pid: standings.Player(t, date.fromisoformat(d).toordinal() if d else None)
            for pid, t, d in con.execute("SELECT player_id, tour, dob FROM data.player")}


class StandingsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_build import Workspace
        cls.ws = Workspace()
        cls.ws.build()
        cls.con = sqlite3.connect(cls.ws.db)
        cls.con.execute("ATTACH DATABASE ? AS data", (str(cls.ws.db),))
        cls.players = players_of(cls.con)
        m = matches.load_matches(cls.con)
        pr.attach(cls.con, m)
        cls.m = m
        cls.weeks = []
        standings.weekly(m, PARAMS, CUTOFFS, cls.players, min_matches=0, active_days=10**6, top=None,
                         on_week=cls.weeks.append)

    @classmethod
    def tearDownClass(cls):
        cls.con.close()
        cls.ws.cleanup()

    def test_mondays(self):
        self.assertEqual(CUTOFFS, ["2020-01-06", "2020-01-13", "2020-01-20", "2020-01-27", "2020-02-03"])
        self.assertEqual(standings.mondays("2026-11-16", "2026-11-16"), ["2026-11-16"])
        self.assertEqual(standings.mondays("2026-11-15", "2026-11-17"), ["2026-11-16", "2026-11-23"])

    def test_one_week_per_cutoff(self):
        self.assertEqual([w.cutoff for w in self.weeks], CUTOFFS)
        self.assertEqual(self.weeks[0].rows, {})       # nothing starts before the first edition
        self.assertTrue(self.weeks[-1].rows)

    def test_week_equals_replay_of_only_the_past(self):
        """Each week = the final state of a replay of only the editions before its cutoff."""
        for w in self.weeks:
            m = matches.load_matches(self.con, before=w.cutoff)
            pr.attach(self.con, m, before=w.cutoff)
            n_stat: dict = {}
            d = m.extra["points"]
            for i in range(len(m)):
                if d.w_sp[i] > 0 and d.l_sp[i] > 0 and m.status[i] not in ("walkover", "unfinished"):
                    for p in (m.winner[i], m.loser[i]):
                        n_stat[p] = n_stat.get(p, 0) + 1
            self.assertTrue(all(s < w.cutoff for s in m.start))
            if not len(m):
                self.assertEqual(w.rows, {})
                continue
            st = pr.run(m, PARAMS).state
            ref = standings.rows_at(st, w.cutoff, self.players, n_stat, min_matches=0, active_days=10**6, top=None)
            self.assertEqual(w.rows, ref, w.cutoff)

    def test_surface_and_sorting(self):
        rows = [r for rs in self.weeks[-1].rows.values() for r in rs]
        st = pr.run(self.m, PARAMS).state
        for r in rows:
            for x in standings.SURFACES:
                k = (r["player_id"], x)
                if k in st.sserve:
                    self.assertEqual(r["surf"][x], (st.sserve[k], st.sret[k], st.scount[k]))
                else:
                    self.assertIsNone(r["surf"][x])
        for rs in self.weeks[-1].rows.values():
            tot = [r["s"] + r["r"] + r["fs"] + r["fr"] for r in rs]
            self.assertEqual(tot, sorted(tot, reverse=True))

    def test_age_offsets(self):
        p = PARAMS
        if not p.has_context():
            self.skipTest("run without context")
        self.assertEqual(standings.age_offsets(p, "atp", None), (0.0, 0.0))
        self.assertEqual(standings.age_offsets(p, "atp", 27.0), (0.0, 0.0))   # flat 26-28
        a, b = standings.age_offsets(p, "atp", 20.0)
        self.assertTrue(a != 0.0 or b != 0.0)

    def test_filters(self):
        st = pr.run(self.m, PARAMS).state
        rows = standings.rows_at(st, CUTOFFS[-1], self.players, {}, min_matches=2, active_days=10**6, top=3)
        for rs in rows.values():
            self.assertLessEqual(len(rs), 3)
            self.assertTrue(all(r["n"] >= 2 for r in rs))
        none = standings.rows_at(st, "2030-01-07", self.players, {}, min_matches=0, active_days=30, top=None)
        self.assertEqual(none, {})


class ExportServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_build import Workspace
        cls.ws = Workspace()
        cls.ws.build()
        mdb = cls.ws.tmp / "models.db"
        con = models_db.connect(mdb)
        cls.run_id = models_db.start_run(con, "points_v1", PARAMS.as_dict())
        con.close()
        cls.out = cls.ws.tmp / "ratings"
        con = sqlite3.connect(mdb)
        con.execute("ATTACH DATABASE ? AS data", (str(cls.ws.db),))
        ratings_export.export(con, cls.run_id, cls.out, since="2020-01-01", today="2020-02-01", top=500,
                              min_matches=0, active_days=10**6, log=lambda s: None)
        ratings_export.export(con, cls.run_id, cls.out, since="2020-01-01", today="2020-02-01", top=500,
                              min_matches=0, active_days=10**6, log=lambda s: None)   # replaces the first
        con.close()
        cls.dir = cls.out / f"run{cls.run_id}"
        cls.srv = ratings_server.ThreadingHTTPServer(("127.0.0.1", 0), ratings_server.make_handler(cls.dir, quiet=True))
        cls.url = f"http://127.0.0.1:{cls.srv.server_address[1]}"
        cls.thread = threading.Thread(target=cls.srv.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()
        cls.ws.cleanup()

    def get(self, path: str):
        return urllib.request.urlopen(self.url + path, timeout=5)

    def test_export_layout(self):
        self.assertEqual(sorted(p.name for p in self.out.iterdir()), [f"run{self.run_id}"])
        meta = json.loads((self.dir / "index.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["weeks"], CUTOFFS)
        self.assertEqual(meta["fields"], ratings_export.FIELDS)
        players = json.loads((self.dir / "players.json").read_text(encoding="utf-8"))
        with gzip.open(self.dir / "weeks" / f"{CUTOFFS[-1]}.json.gz", "rt", encoding="utf-8") as f:
            wk = json.load(f)
        rows = [r for rs in wk["rows"].values() for r in rs]
        self.assertTrue(rows)
        for r in rows:
            self.assertEqual(len(r), len(meta["fields"]))
            self.assertLess(r[0], len(players))
        self.assertTrue(all(":" in p[0] for p in players))   # stable keys
        self.assertEqual(set(wk["base"]), {"atp", "wta"})

    def test_server_routes(self):
        with self.get("/") as r:
            self.assertIn(b"<table", r.read())
        with self.get("/data/index.json") as r:
            self.assertEqual(json.loads(r.read())["run_id"], self.run_id)
        with self.get(f"/data/weeks/{CUTOFFS[-1]}.json.gz") as r:
            self.assertEqual(r.headers["Content-Encoding"], "gzip")
            self.assertIn("rows", json.loads(gzip.decompress(r.read())))
        for bad in ("/data/../index.json", "/data/%2e%2e/%2e%2e/CLAUDE.md", "/etc/passwd", "/data/weeks/nope.json.gz"):
            with self.assertRaises(urllib.error.HTTPError, msg=bad) as cm:
                self.get(bad)
            self.assertEqual(cm.exception.code, 404)

    def test_resolve_never_leaves_data_dir(self):
        self.assertIsNone(ratings_server.resolve(self.dir, "/data/../../../CLAUDE.md"))
        self.assertIsNone(ratings_server.resolve(self.dir, "/data/%2e%2e/x.json"))
        self.assertIsNotNone(ratings_server.resolve(self.dir, "/data/index.json"))


if __name__ == "__main__":
    unittest.main()
