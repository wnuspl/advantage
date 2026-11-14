"""AcesAPI: parsing, pull planning, and an end-to-end build with two fixture pulls.

Fixture pulls (tests/fixtures/raw/acesapi-*, manifest tests/fixtures/build_manifest_acesapi.json):
  20261110T120000Z  Brisbane 2020 (before the frontier: must link), AO 2020 (the backbone's
                    last week: past the frontier) + US Open 2026, live
  20261114T120000Z  US Open 2026 again, final, one score corrected
Run: python -m unittest discover -s tests
"""
from __future__ import annotations

import json
import shutil
import sqlite3
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from tennis import acesapi as a                     # noqa: E402
from tennis import acesapi_fetch as af              # noqa: E402
from tennis.build import run_build                  # noqa: E402

FIXTURES = HERE / "fixtures"
MANIFEST = FIXTURES / "build_manifest_acesapi.json"


class ParsingTest(unittest.TestCase):
    def rows(self, labels):
        return [({"round": l, "draw": "qualifying" if "Qual" in l else "main"}, {}) for l in labels]

    def test_round_labels_seen_live(self):
        # Labels and counts from the real Roland Garros 2026 draw (2026-11-14).
        rows = self.rows(["Qualifying 1st Round"] * 64 + ["Qualifying 2nd Round"] * 32 +
                         ["Qualifying Final"] * 16 + ["Round 1"] * 64 + ["Round 2"] * 32 +
                         ["Round 3"] * 16 + ["Round 4"] * 8 + ["Quarterfinal"] * 4 +
                         ["Semifinal"] * 2 + ["Final"])
        mr, qr = a.n_main_rounds(rows), a.n_qual_rounds(rows)
        self.assertEqual((mr, qr), (7, 3))
        got = {l: a.map_round(l, "Qual" in l, mr, qr) for l in
               ["Qualifying 1st Round", "Qualifying 2nd Round", "Qualifying Final", "Round 1",
                "Round 2", "Round 4", "Quarterfinal", "Semifinal", "Final"]}
        self.assertEqual(got, {"Qualifying 1st Round": "Q1", "Qualifying 2nd Round": "Q2",
                               "Qualifying Final": "Q3", "Round 1": "R128", "Round 2": "R64",
                               "Round 4": "R16", "Quarterfinal": "QF", "Semifinal": "SF", "Final": "F"})

    def test_small_draw_and_unknown_labels(self):
        rows = self.rows(["Round 1"] * 12 + ["Round 2"] * 8 + ["Quarterfinal"] * 4 +
                         ["Semifinal"] * 2 + ["Final"])          # 28-player draw with byes
        self.assertEqual(a.map_round("Round 1", False, a.n_main_rounds(rows)), "R32")
        self.assertEqual(a.map_round("Round of 16", False, None), "R16")
        self.assertEqual(a.map_round("Round Robin", False, None), "RR")
        self.assertIsNone(a.map_round("Bronze-ish playoff", False, 5))
        self.assertIsNone(a.map_round("Qualifying Final", True, 5, None))

    def test_status_and_envelope(self):
        self.assertEqual(a.match_status("walkover", "w/o"), "walkover")
        self.assertEqual(a.match_status("retired", "6-3 4-1 ret."), "retired")
        self.assertEqual(a.match_status(None, "6-4 2-1 RET"), "retired")
        self.assertIsNone(a.match_status("cancelled", None))
        self.assertEqual(a.unwrap({"data": {"x": 1}}), {"x": 1})
        self.assertEqual(a.unwrap({"tour": "ATP", "data": {"x": 1}, "a": 1, "b": 2}),
                         {"tour": "ATP", "data": {"x": 1}, "a": 1, "b": 2})

    def test_edition_start_is_first_main_draw_day(self):
        rows = [({"draw": "qualifying"}, {"match_date": "2025-06-23"}),
                ({"draw": "main"}, {"match_date": "2025-07-01"}),
                ({"draw": "main"}, {"starts_at": "2025-06-30T10:00:00Z"})]
        self.assertEqual(a.edition_start({"start_date": "2025-06-23"}, rows), "2025-06-30")
        self.assertEqual(a.edition_start({"start_date": "2025-06-23"}, rows[:1]), "2025-06-23")


class PlanningTest(unittest.TestCase):
    T = {"tournament_id": "1-2026", "status": "final", "start_date": "2026-08-17", "end_date": "2026-09-02"}

    def test_wanted(self):
        since, today = date(2026, 5, 18), date(2026, 11, 14)
        self.assertEqual(af.wanted(self.T, None, since, today, 10), "new")
        self.assertIsNone(af.wanted({**self.T, "status": "scheduled"}, None, since, today, 10))
        self.assertIsNone(af.wanted({**self.T, "end_date": "2026-05-01"}, None, since, today, 10))
        live = af.Seen("20260901T000000Z", "live", "2026-09-02")
        self.assertEqual(af.wanted(self.T, live, since, today, 10), "was_live")
        fresh = af.Seen("20260905T000000Z", "final", "2026-09-02")       # inside correction window
        self.assertEqual(af.wanted(self.T, fresh, since, today, 10), "correction_window")
        settled = af.Seen("20260920T000000Z", "final", "2026-09-02")
        self.assertIsNone(af.wanted(self.T, settled, since, today, 10))
        # AcesAPI leaves some events "live" for months: past the window it is settled anyway.
        stale_live = af.Seen("20261114T183408Z", "live", "2026-07-27")
        self.assertIsNone(af.wanted({**self.T, "status": "live", "end_date": "2026-07-27"},
                                    stale_live, since, today, 10))

    def test_previous_pulls_takes_latest(self):
        prev = af.previous_pulls(FIXTURES / "raw")
        self.assertEqual(prev[("atp", "560-2026")].pull_id, "20261114T120000Z")
        self.assertEqual(prev[("atp", "560-2026")].status, "final")
        self.assertEqual(prev[("atp", "580-2020")].pull_id, "20261110T120000Z")

    def test_pin_keeps_layout_and_refuses_twice(self):
        tmp = Path(tempfile.mkdtemp(prefix="tennis-pin-"))
        try:
            m = tmp / "manifest.json"
            shutil.copy(MANIFEST, m)
            before = m.read_text(encoding="utf-8")
            entry = af.manifest_entry("20261115T000000Z", FIXTURES / "raw" / "acesapi-20261114T120000Z")
            entry["folder"] = "acesapi-20261115T000000Z"
            self.assertTrue(af.pin(m, entry))
            after = m.read_text(encoding="utf-8")
            self.assertTrue(after.startswith(before.rstrip()[:-6]))          # earlier text untouched
            self.assertEqual(json.loads(after)["snapshots"][-1], entry)
            self.assertFalse(af.pin(m, entry))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_inspect_reports_values(self):
        got = af.inspect(FIXTURES / "raw" / "acesapi-20261110T120000Z")
        self.assertEqual(got["envelope"], {"data": 3})
        self.assertIn("qualifying|Qualifying 1st Round|1", got["round"])


class AcesBuildTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="tennis-aces-"))
        shutil.copytree(FIXTURES / "raw", cls.tmp / "raw")
        cls.report = run_build(project_root=cls.tmp, manifest_path=MANIFEST, raw_root=cls.tmp / "raw",
                               db_path=cls.tmp / "tennis.db", lock_path=cls.tmp / "lock.json",
                               overrides_path=cls.tmp / "overrides.csv", report_path=None, quiet=True)
        cls.con = sqlite3.connect(cls.tmp / "tennis.db")

    @classmethod
    def tearDownClass(cls):
        cls.con.close()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def one(self, sql, *args):
        return self.con.execute(sql, args).fetchone()

    def run_of(self, folder, rel):
        return self.one("""SELECT r.rows_read, r.rows_inserted, r.rows_linked, r.rows_skipped, r.rows_queued
                           FROM load_run r JOIN ingest_snapshot s USING (snapshot_id)
                           WHERE s.raw_path LIKE ? AND s.file_relpath=?""", f"%{folder}%", rel)

    def test_invariants_hold(self):
        self.assertTrue(all(v == "ok" for v in self.report["invariants"].values()), self.report["invariants"])
        self.assertEqual(self.con.execute("SELECT * FROM v_inv_load_counts").fetchall(), [])

    def test_json_files_registered_with_row_counts(self):
        rows = dict(self.con.execute(
            "SELECT source_ref || '/' || file_relpath, row_count FROM ingest_snapshot WHERE source='acesapi'"))
        self.assertEqual(rows, {"20261110T120000Z/draws/atp/339-2020.json": 2,
                                "20261110T120000Z/draws/atp/580-2020.json": 7,
                                "20261110T120000Z/draws/atp/560-2026.json": 3,
                                "20261114T120000Z/draws/atp/560-2026.json": 8})
        self.assertEqual(self.one("SELECT fetched_at FROM ingest_snapshot WHERE source_ref='20261114T120000Z'"),
                         ("2026-11-14T12:00:00Z",))

    def test_older_pull_is_superseded_but_accounted(self):
        self.assertEqual(self.run_of("acesapi-20261110T120000Z", "draws/atp/560-2026.json"), (3, 0, 0, 3, 0))

    def test_before_frontier_links_or_queues_never_creates(self):
        # Fixture frontier: last Sackmann edition start (2020-01-20) - 7 days.
        self.assertEqual(self.report["acesapi"]["frontier"], {"atp": "2020-01-13"})
        self.assertEqual(self.run_of("acesapi-20261110T120000Z", "draws/atp/339-2020.json"), (2, 0, 1, 0, 1))
        self.assertEqual(self.one("""SELECT m.round, m.score FROM match_xref x JOIN matches m USING (match_id)
                                     WHERE x.source='acesapi' AND x.source_key='atp:3002'"""), ("F", "W/O"))
        self.assertEqual(self.one("""SELECT reason FROM review_queue WHERE source='acesapi'
                                     AND source_key='atp:3001'"""), ("player_unresolved",))
        self.assertEqual(self.one("""SELECT reason, status FROM review_queue WHERE entity_type='player'
                                     AND source='acesapi' AND source_key='atp:98'"""), ("no_candidate", "open"))

    def test_overlap_links_to_backbone_and_fills_missing_rounds(self):
        # AO 2020: three rows link to Sackmann matches; Q1 and R16 rounds are absent from the
        # backbone edition and get filled. Zed Nobody (no candidate, past the frontier) is new.
        self.assertEqual(self.run_of("acesapi-20261110T120000Z", "draws/atp/580-2020.json"), (7, 3, 3, 0, 1))
        # Sackmann has Bedene beating Groth in R128; this row says Groth won: never linked.
        self.assertEqual(self.one("""SELECT reason FROM review_queue WHERE source='acesapi'
                                     AND source_key='atp:1007'"""), ("result_conflict",))
        linked = {r[0]: r[1:] for r in self.con.execute(
            """SELECT x.source_key, m.round, x.method, x.confidence FROM match_xref x JOIN matches m USING (match_id)
               WHERE x.source='acesapi' AND x.method<>'seed'""")}
        self.assertEqual(linked, {"atp:3002": ("F", "pair_tournament_date", 1.0),
                                  "atp:1002": ("R128", "pair_tournament_date", 1.0),
                                  "atp:1003": ("R128", "pair_tournament_date", 1.0),
                                  "atp:1006": ("F", "pair_tournament_date", 1.0)})
        filled = self.con.execute(
            """SELECT m.round, m.status, m.match_date, m.date_precision, t.canonical_name, e.level
               FROM matches m JOIN match_xref x USING (match_id) JOIN tournament_edition e USING (edition_id)
               JOIN tournament t USING (tournament_id)
               WHERE x.source='acesapi' AND x.source_key IN ('atp:1001','atp:1005') ORDER BY m.round""").fetchall()
        self.assertEqual(filled, [("Q1", "walkover", "2020-01-15", "day", "Australian Open", "G"),
                                  ("R16", "completed", "2020-01-27", "day", "Australian Open", "G")])
        self.assertEqual(self.one("SELECT COUNT(*) FROM tournament_edition e JOIN tournament t USING (tournament_id) "
                                  "WHERE t.canonical_name='Australian Open'")[0], 1)
        # 'John Smith' is two players: the one in this event is chosen.
        self.assertEqual(self.one("""SELECT p.dob FROM player_xref x JOIN player p USING (player_id)
                                     WHERE x.source='acesapi' AND x.source_key='atp:15'"""), ("1995-01-01",))

    def test_past_frontier_extends_backbone(self):
        self.assertEqual(self.run_of("acesapi-20261114T120000Z", "draws/atp/560-2026.json"), (8, 6, 0, 1, 1))
        # Lucky loser: the same pair in qualifying and in the main draw are two matches.
        self.assertEqual(self.con.execute(
            """SELECT x.source_key, m.round, x.method FROM match_xref x JOIN matches m USING (match_id)
               WHERE x.source='acesapi' AND x.source_key IN ('atp:2101','atp:2102') ORDER BY 1""").fetchall(),
            [("atp:2101", "Q1", "seed"), ("atp:2102", "R16", "seed")])
        self.assertEqual(self.one("""SELECT t.canonical_name, e.start_date, e.end_date, e.level, e.best_of, e.draw_size
                                     FROM edition_xref x JOIN tournament_edition e USING (edition_id)
                                     JOIN tournament t USING (tournament_id)
                                     WHERE x.source='acesapi' AND x.source_key='atp:560-2026'"""),
                         ("US Open", "2026-08-24", "2026-09-02", "G", 5, 16))
        rounds = [r[0] for r in self.con.execute(
            """SELECT m.round FROM matches m JOIN match_xref x USING (match_id)
               WHERE x.source='acesapi' AND x.source_key LIKE 'atp:200%' ORDER BY m.round_order, m.match_id""")]
        self.assertEqual(rounds, ["R16", "R16", "QF", "F"])
        self.assertEqual(self.one("""SELECT k.reason FROM load_skip k JOIN load_run r USING (load_run_id)
                                     JOIN ingest_snapshot s USING (snapshot_id)
                                     WHERE s.source_ref='20261114T120000Z'"""), ("not_final",))

    def test_new_players_only_when_clearly_new(self):
        made = dict(self.con.execute(
            """SELECT p.canonical_name, p.country FROM player_xref x JOIN player p USING (player_id)
               WHERE x.source='acesapi' AND x.method='seed'"""))
        # No candidate at all; and a fuzzy candidate (Sam Groth, AUS) with a different country.
        self.assertEqual(made, {"Juan Newplayer": "ARG", "Sam Grothe": "USA", "Zed Nobody": "NOR"})
        # Johnny Smith (GBR) has a same-country fuzzy candidate (John Smith, GBR): queued.
        self.assertEqual(self.one("""SELECT reason, status FROM review_queue WHERE entity_type='player'
                                     AND source='acesapi' AND source_key='atp:52'"""), ("low_confidence", "open"))
        self.assertEqual(self.one("""SELECT status FROM review_queue WHERE entity_type='player'
                                     AND source='acesapi' AND source_key='atp:53'"""), ("resolved",))
        self.assertEqual(self.report["acesapi"]["new_players"], 3)

    def test_match_seen_timestamps_and_revisions(self):
        seen = {r[0]: r[1:] for r in self.con.execute(
            "SELECT source_key, first_seen_at, last_seen_at, n_pulls, revised FROM match_seen")}
        self.assertEqual(seen["atp:2001"], ("2026-11-10T12:00:00Z", "2026-11-14T12:00:00Z", 2, 1))  # score corrected
        self.assertEqual(seen["atp:2002"], ("2026-11-10T12:00:00Z", "2026-11-14T12:00:00Z", 2, 0))
        self.assertEqual(seen["atp:2006"], ("2026-11-14T12:00:00Z", "2026-11-14T12:00:00Z", 1, 0))  # was live
        self.assertNotIn("atp:2003", seen)        # queued, so no match row
        self.assertEqual(self.one("SELECT score FROM matches m JOIN match_xref x USING (match_id) "
                                  "WHERE x.source='acesapi' AND x.source_key='atp:2001'"), ("6-1 6-1 6-1",))

    def test_mcp_chart_after_sackmann_cutoff_now_resolves(self):
        self.assertEqual(self.one("""SELECT COUNT(*) FROM match_xref x JOIN matches m USING (match_id)
                                     JOIN match_xref y ON y.match_id=m.match_id
                                     WHERE x.source='mcp' AND x.source_key LIKE '20260901-%'
                                       AND y.source='acesapi' AND y.source_key='atp:2006'""")[0], 1)
        self.assertNotIn("after_backbone_cutoff",
                         dict(self.con.execute("""SELECT k.reason, k.n FROM load_skip k
                                                  JOIN load_run r USING (load_run_id) WHERE r.loader='mcp_matches'""")))

    def test_same_day_invariant_allows_doubleheaders_only(self):
        """Two rounds of one event on one day are real; another event or the same round is not."""
        con = self.con
        sid = self.one("SELECT snapshot_id FROM ingest_snapshot WHERE source='acesapi' LIMIT 1")[0]
        us = self.one("SELECT edition_id FROM edition_xref WHERE source='acesapi' AND source_key='atp:560-2026'")[0]
        p = [r[0] for r in con.execute("SELECT player_id FROM player WHERE is_placeholder=0 ORDER BY player_id LIMIT 4")]
        ins = """INSERT INTO matches (edition_id, round, round_order, match_date, date_precision, winner_id,
                     loser_id, status, snapshot_id) VALUES (?,?,?, '2026-08-26','day',?,?,'completed',?)"""
        other = con.execute("""INSERT INTO tournament_edition (tournament_id, year, start_date)
                               SELECT tournament_id, 2026, '2026-08-20' FROM tournament_edition
                               WHERE edition_id=?""", (us,)).lastrowid
        try:
            con.execute(ins, (us, "R32", 10, p[0], p[1], sid))
            con.execute(ins, (us, "R16", 11, p[0], p[2], sid))          # doubleheader: allowed
            self.assertEqual(con.execute("SELECT COUNT(*) FROM v_inv_same_day_conflicts").fetchone()[0], 0)
            con.execute(ins, (other, "R16", 11, p[0], p[3], sid))       # another event: flagged
            self.assertEqual(con.execute("SELECT COUNT(*) FROM v_inv_same_day_conflicts").fetchone()[0], 2)
        finally:
            con.rollback()

    def test_identical_refetch_is_its_own_snapshot(self):
        """A third pull re-fetches the US Open with byte-identical content (the common case)."""
        tmp = Path(tempfile.mkdtemp(prefix="tennis-aces3-"))
        try:
            raw = tmp / "raw"
            shutil.copytree(self.tmp / "raw", raw)
            src = raw / "acesapi-20261114T120000Z"
            dst = raw / "acesapi-20261115T080000Z"
            (dst / "draws" / "atp").mkdir(parents=True)
            shutil.copy(src / "draws" / "atp" / "560-2026.json", dst / "draws" / "atp" / "560-2026.json")
            shutil.copy(src / "pull.json", dst / "pull.json")
            man = json.loads(MANIFEST.read_text(encoding="utf-8"))
            man["snapshots"].append(af.manifest_entry("20261115T080000Z", dst))
            mpath = tmp / "manifest.json"
            mpath.write_text(json.dumps(man), encoding="utf-8")
            report = run_build(project_root=tmp, manifest_path=mpath, raw_root=raw, db_path=tmp / "t.db",
                               lock_path=tmp / "lock.json", overrides_path=tmp / "overrides.csv",
                               report_path=None, quiet=True)
            self.assertTrue(all(v == "ok" for v in report["invariants"].values()), report["invariants"])
            with sqlite3.connect(tmp / "t.db") as c2:
                runs = dict(c2.execute(
                    """SELECT s.source_ref, r.rows_skipped FROM ingest_snapshot s JOIN load_run r USING (snapshot_id)
                       WHERE s.file_relpath='draws/atp/560-2026.json'"""))
                self.assertEqual(runs, {"20261110T120000Z": 3, "20261114T120000Z": 8, "20261115T080000Z": 1})
                self.assertEqual(c2.execute("""SELECT COUNT(DISTINCT sha256) FROM ingest_snapshot
                                               WHERE source_ref IN ('20261114T120000Z','20261115T080000Z')""").fetchone()[0], 1)
                self.assertEqual(c2.execute("""SELECT first_seen_at, last_seen_at, n_pulls FROM match_seen
                                               WHERE source_key='atp:2002'""").fetchone(),
                                 ("2026-11-10T12:00:00Z", "2026-11-15T08:00:00Z", 3))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_rebuild_is_deterministic(self):
        tmp = Path(tempfile.mkdtemp(prefix="tennis-aces2-"))
        try:
            run_build(project_root=self.tmp, manifest_path=MANIFEST, raw_root=self.tmp / "raw",
                      db_path=tmp / "t.db", lock_path=self.tmp / "lock.json",
                      overrides_path=self.tmp / "overrides.csv", report_path=None, quiet=True)
            q = """SELECT x.source_key, m.round, m.score, m.status FROM match_xref x JOIN matches m USING (match_id)
                   WHERE x.source='acesapi' ORDER BY 1"""
            with sqlite3.connect(tmp / "t.db") as c2:
                self.assertEqual(c2.execute(q).fetchall(), self.con.execute(q).fetchall())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
