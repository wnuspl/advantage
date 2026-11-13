"""End-to-end build on tiny fixtures (tests/fixtures/raw). Run: python -m unittest discover -s tests"""
from __future__ import annotations

import shutil
from contextlib import closing
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from tennis.build import run_build          # noqa: E402
from tennis.snapshots import BuildError     # noqa: E402

FIXTURES = HERE / "fixtures"


class Workspace:
    """A throwaway copy of the fixture raw tree plus its own lock, overrides and db."""

    def __init__(self, overrides: str | None = None):
        self.tmp = Path(tempfile.mkdtemp(prefix="tennis-test-"))
        self.raw = self.tmp / "raw"
        shutil.copytree(FIXTURES / "raw", self.raw)
        self.db = self.tmp / "tennis.db"
        self.lock = self.tmp / "snapshot_lock.json"
        self.overrides = self.tmp / "overrides.csv"
        if overrides is not None:
            self.overrides.write_text(
                "entity_type,source,source_key,action,target_source,target_key,note\n" + overrides,
                encoding="utf-8")

    def build(self) -> dict:
        return run_build(project_root=self.tmp, manifest_path=FIXTURES / "build_manifest.json",
                         raw_root=self.raw, db_path=self.db, lock_path=self.lock,
                         overrides_path=self.overrides, report_path=self.tmp / "report.json", quiet=True)

    def connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db)
        con.execute("PRAGMA foreign_keys = ON")
        return con

    def cleanup(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


class BuildFixtureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ws = Workspace()
        cls.report = cls.ws.build()
        cls.con = cls.ws.connect()

    @classmethod
    def tearDownClass(cls):
        cls.con.close()
        cls.ws.cleanup()

    def one(self, sql, *args):
        return self.con.execute(sql, args).fetchone()

    def loads(self, loader):
        return self.one("""SELECT SUM(rows_read), SUM(rows_inserted), SUM(rows_linked), SUM(rows_skipped),
                                  SUM(rows_queued) FROM load_run WHERE loader=?""", loader)

    def skips(self, loader):
        return dict(self.con.execute(
            """SELECT k.reason, SUM(k.n) FROM load_skip k JOIN load_run r USING (load_run_id)
               WHERE r.loader=? GROUP BY k.reason""", (loader,)))

    # ---- invariants -------------------------------------------------------------------
    def test_invariants_all_empty(self):
        for view in ("v_inv_load_counts", "v_inv_same_day_conflicts", "v_inv_point_players"):
            self.assertEqual(self.con.execute(f"SELECT * FROM {view}").fetchall(), [], view)
        self.assertEqual(self.con.execute("PRAGMA foreign_key_check").fetchall(), [])
        self.assertTrue(all(v == "ok" for v in self.report["invariants"].values()), self.report["invariants"])

    def test_every_registered_file_has_one_finished_run(self):
        self.assertEqual(self.one("SELECT COUNT(*) FROM ingest_snapshot")[0], 7)
        self.assertEqual(self.one("""SELECT COUNT(*) FROM ingest_snapshot s JOIN load_run r
                                     ON r.snapshot_id=s.snapshot_id AND r.loader=s.loader
                                     WHERE r.finished_at IS NOT NULL AND r.rows_read=s.row_count""")[0], 7)
        self.assertEqual(self.report["ignored_files"], 4)

    # ---- backbone ---------------------------------------------------------------------
    def test_players(self):
        self.assertEqual(self.loads("sackmann_players"), (8, 8, 0, 0, 0))
        self.assertEqual(self.one("SELECT dob, hand FROM player WHERE canonical_name='Roger Federer'"),
                         ("1981-08-08", "R"))
        self.assertEqual(self.one("SELECT is_placeholder FROM player WHERE name_last='Unknown'"), (1,))
        self.assertEqual(self.one("""SELECT alias_normalized FROM player_alias a JOIN player p
                                     USING (player_id) WHERE p.name_last='Bedene'"""), ("aljaz bedene",))

    def test_match_outcomes(self):
        self.assertEqual(self.loads("sackmann_matches"), (11, 7, 0, 3, 1))
        self.assertEqual(self.skips("sackmann_matches"),
                         {"exact_duplicate_row": 1, "placeholder_player": 1, "bye": 1})
        self.assertEqual(self.one("""SELECT reason FROM review_queue WHERE source='sackmann_atp'"""),
                         ("duplicate_key_conflict",))

    def test_status_and_stats(self):
        self.assertEqual(self.one("SELECT status FROM matches WHERE score='6-4 6-4 RET'"), ("retired",))
        self.assertEqual(self.one("SELECT status FROM matches WHERE score='W/O'"), ("walkover",))
        self.assertEqual(self.one("SELECT COUNT(*) FROM match_player_stats")[0], 14)
        self.assertEqual(self.one("""SELECT aces, serve_points FROM match_player_stats s JOIN matches m
                                     USING (match_id) WHERE m.round='F' AND s.is_winner=1
                                     AND m.score='6-4 6-4 6-4'"""), (10, 80))
        self.assertEqual(self.one("SELECT COUNT(*) FROM matches WHERE date_precision='week' AND match_date IS NULL")[0], 7)

    def test_same_name_same_week_draws_get_separate_tournaments(self):
        names = [r[0] for r in self.con.execute(
            "SELECT canonical_name FROM tournament WHERE canonical_name LIKE 'M15 Monastir%' ORDER BY 1")]
        self.assertEqual(len(names), 2)
        self.assertEqual(self.one("SELECT COUNT(*) FROM tournament_edition")[0], 4)
        self.assertEqual(self.one("SELECT surface FROM tournament_edition WHERE level='15' LIMIT 1"), ("clay",))

    def test_rankings(self):
        self.assertEqual(self.loads("sackmann_rankings"), (4, 2, 0, 2, 0))
        self.assertEqual(self.skips("sackmann_rankings"), {"duplicate_ranking_row": 1, "unknown_player_id": 1})

    # ---- MCP --------------------------------------------------------------------------
    def test_mcp_match_resolution(self):
        self.assertEqual(self.loads("mcp_matches"), (7, 0, 4, 3, 0))
        self.assertEqual(self.skips("mcp_matches"),
                         {"after_backbone_cutoff": 1, "malformed_row": 1, "duplicate_match_id": 1})
        # 'John Smith' is two players; only one was active then.
        self.assertEqual(self.one("""SELECT x.method, p.dob FROM player_xref x JOIN player p USING (player_id)
                                     WHERE x.source='mcp' AND x.source_key='m:John Smith'"""),
                         ("context", "1995-01-01"))
        # 'Samuel Groth' is not an exact name, so it is queued with fuzzy candidates. It then
        # resolves through its opponent: Bedene's only R128 opponent at that event is Sam Groth.
        self.assertEqual(self.one("""SELECT reason, status FROM review_queue WHERE entity_type='player'
                                     AND source='mcp' AND source_key='m:Samuel Groth'"""),
                         ("low_confidence", "resolved"))
        self.assertEqual(self.one("""SELECT x.method, x.confidence, p.canonical_name FROM player_xref x
                                     JOIN player p USING (player_id)
                                     WHERE x.source='mcp' AND x.source_key='m:Samuel Groth'"""),
                         ("context", 0.9, "Sam Groth"))

    def test_mcp_points(self):
        self.assertEqual(self.loads("mcp_points"), (10, 6, 0, 4, 0))
        self.assertEqual(self.skips("mcp_points"),
                         {"malformed_row": 1, "duplicate_point": 1, "duplicate_chart": 1,
                          "match_row_skipped_or_missing": 1})
        pts = {r[0]: r[1:] for r in self.con.execute(
            """SELECT p.point_idx, p.serve_no, p.is_ace, p.is_double_fault, p.rally_len, p.end_type,
                      p.is_tiebreak, s.canonical_name, w.canonical_name
               FROM point p JOIN player s ON s.player_id=p.server_id JOIN player w ON w.player_id=p.winner_id
               JOIN matches m ON m.match_id=p.match_id
               WHERE p.data_source='mcp' AND m.round='F'""")}
        self.assertEqual(pts[1], (1, 1, 0, 1, "ace", 0, "Novak Djokovic", "Novak Djokovic"))
        self.assertEqual(pts[2], (2, 0, 1, 0, "double_fault", 0, "Novak Djokovic", "Rafael Nadal"))
        self.assertEqual(pts[3], (1, 0, 0, 4, "winner", 0, "Novak Djokovic", "Rafael Nadal"))
        self.assertEqual(pts[4], (1, 0, 0, 2, "unforced_error", 1, "Rafael Nadal", "Novak Djokovic"))

    # ---- slam point-by-point ----------------------------------------------------------
    def test_slam_resolution(self):
        self.assertEqual(self.loads("slam_pbp_matches"), (5, 0, 4, 1, 0))
        methods = dict(self.con.execute(
            "SELECT source_key, method FROM player_xref WHERE source='sackmann_slam_pbp'"))
        self.assertEqual(methods["atp:2020:ausopen:John Smith"], "context")   # edition participants
        self.assertEqual(methods["atp:2020:ausopen:A Bedene"], "context")     # initial + surname
        self.assertEqual(methods["atp:2020:ausopen:Roger Federer"], "exact_name")

    def test_slam_points(self):
        self.assertEqual(self.loads("slam_pbp_points"), (6, 3, 0, 3, 0))
        self.assertEqual(self.skips("slam_pbp_points"),
                         {"placeholder_point": 1, "duplicate_point": 1, "match_row_skipped_or_missing": 1})
        # AO from 2018 has placeholder rally lengths: stored as NULL.
        self.assertEqual(self.one("SELECT COUNT(*) FROM point WHERE data_source='sackmann_slam_pbp' AND rally_len IS NOT NULL")[0], 0)
        self.assertEqual(self.one("""SELECT is_ace, end_type FROM point WHERE data_source='sackmann_slam_pbp'
                                     AND point_idx=1 AND serve_no=1 AND is_ace=1"""), (1, "ace"))

    # ---- schema-level invariants --------------------------------------------------------
    def test_xref_source_key_maps_to_one_entity(self):
        pid = self.one("SELECT player_id FROM player LIMIT 1")[0]
        with self.assertRaises(sqlite3.IntegrityError):
            self.con.execute("""INSERT INTO player_xref (source, source_key, player_id, method, confidence)
                                VALUES ('sackmann_atp', '100001', ?, 'seed', 1.0)""", (pid,))

    def test_winner_must_differ_from_loser(self):
        ed, pid, sid = self.one("SELECT edition_id, winner_id, snapshot_id FROM matches LIMIT 1")
        with self.assertRaises(sqlite3.IntegrityError):
            self.con.execute("""INSERT INTO matches (edition_id, round, round_order, date_precision,
                                winner_id, loser_id, snapshot_id) VALUES (?, 'F', 16, 'week', ?, ?, ?)""",
                             (ed, pid, pid, sid))


class OverrideTest(unittest.TestCase):
    def test_override_link_wins(self):
        ws = Workspace("player,mcp,m:Samuel Groth,link,sackmann_atp,100006,Sam Groth\n")
        try:
            ws.build()
            with closing(ws.connect()) as con:
                self.assertEqual(con.execute(
                    """SELECT x.method, x.verified, p.canonical_name FROM player_xref x JOIN player p
                       USING (player_id) WHERE x.source='mcp' AND x.source_key='m:Samuel Groth'""").fetchone(),
                    ("override", 1, "Sam Groth"))
                self.assertEqual(con.execute(
                    "SELECT rows_linked, rows_queued FROM load_run WHERE loader='mcp_matches'").fetchone(), (4, 0))
        finally:
            ws.cleanup()

    def test_unresolvable_override_target_stops_build(self):
        ws = Workspace("player,mcp,m:Samuel Groth,link,sackmann_atp,does-not-exist,\n")
        try:
            with self.assertRaisesRegex(BuildError, "does not resolve"):
                ws.build()
        finally:
            ws.cleanup()


class SnapshotGuardTest(unittest.TestCase):
    def test_changed_raw_bytes_stop_the_build(self):
        ws = Workspace()
        try:
            ws.build()
            f = ws.raw / "archive-aaaa1111" / "atp" / "atp_rankings_20s.csv"
            f.write_text(f.read_text(encoding="utf-8") + "20200113,1,100002,10000\n", encoding="utf-8")
            with self.assertRaisesRegex(BuildError, "changed under a pinned commit"):
                ws.build()
            self.assertTrue(ws.db.exists())    # the previous good build is untouched
        finally:
            ws.cleanup()

    def test_unclassified_file_stops_the_build(self):
        ws = Workspace()
        try:
            (ws.raw / "mcp-bbbb2222" / "surprise.csv").write_text("a\n1\n", encoding="utf-8")
            with self.assertRaisesRegex(BuildError, "match no group"):
                ws.build()
        finally:
            ws.cleanup()


if __name__ == "__main__":
    unittest.main()
