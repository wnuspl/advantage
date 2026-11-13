"""The build protocol: raw snapshot folders + manifest -> a fresh, checked tennis.db.

Steps (each must finish before the next starts; any failure leaves the old tennis.db in
place and the half-built file at <db>.building for inspection):

  0. preflight     read data/build_manifest.json; classify every file in each pinned folder
                   (group / ignore / stop on anything unclassified); check SQLite version.
  1. schema        create <db>.building from schema.sql.
  2. register      sources from the manifest; hash + count every planned file; compare with
                   data/snapshot_lock.json (stop if any known file changed); one
                   ingest_snapshot row per file.
  3. backbone      Sackmann players -> tournaments/editions + matches + match_player_stats
                   -> rankings (ATP then WTA).
  4. overrides     data/overrides.csv -> override_link, targets resolved via the xrefs.
  5. mcp           charted matches -> player + match resolution cascade -> points.
  6. slam_pbp      slam matches -> resolution within the slam edition -> points.
  7. verify        invariants (v_inv_load_counts, v_inv_same_day_conflicts,
                   v_inv_point_players, foreign keys, overrides honoured). Any failure stops
                   the build here.
  8. publish       ANALYZE, write the build report, swap <db>.building -> <db>.
"""
from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

from . import db, load_mcp, load_sackmann, load_slam_pbp, overrides
from .resolve import PlayerResolver
from .snapshots import BuildError, load_manifest, plan_files, register_files, register_sources

LOADER_ORDER = [
    "sackmann_players", "sackmann_matches", "sackmann_rankings",
    "mcp_matches", "mcp_points", "slam_pbp_matches", "slam_pbp_points",
]


class Log:
    def __init__(self, quiet: bool = False):
        self.t0 = time.monotonic()
        self.quiet = quiet

    def __call__(self, msg: str) -> None:
        if not self.quiet:
            print(f"[{time.monotonic() - self.t0:7.1f}s] {msg}", flush=True)


def _files(by_loader, loader):
    return sorted(by_loader.get(loader, []), key=lambda x: (x[1].source, x[1].relpath))


def _run_loader(con, log, by_loader, loader, fn):
    for sid, rf in _files(by_loader, loader):
        run = fn(sid, rf)
        con.commit()
        log(f"  {loader:18s} {rf.relpath:55s} read={run.read:>8} ins={run.n_inserted:>8} "
            f"link={run.n_linked:>6} skip={run.n_skipped:>6} queue={run.n_queued:>5}")


def run_build(*, project_root: Path, manifest_path: Path, raw_root: Path, db_path: Path,
              lock_path: Path, overrides_path: Path, report_path: Path | None,
              quiet: bool = False) -> dict:
    log = Log(quiet)
    # 0. preflight
    manifest = load_manifest(manifest_path)
    files, ignored = plan_files(manifest, raw_root)
    unknown = sorted({rf.loader for rf in files} - set(LOADER_ORDER))
    if unknown:
        raise BuildError(f"manifest names unknown loader(s): {unknown}")
    log(f"preflight: {len(files)} files to load, {len(ignored)} ignored")

    # 1. schema
    tmp = Path(str(db_path) + ".building")
    con = db.create(tmp, bulk=True)
    try:
        # 2. register
        register_sources(con, manifest)
        by_loader = register_files(con, files, project_root, lock_path, progress=log)
        con.commit()
        log("registered snapshots")

        # 3. backbone
        bb = load_sackmann.Backbone()
        _run_loader(con, log, by_loader, "sackmann_players",
                    lambda sid, rf: load_sackmann.load_players(con, sid, rf, bb))
        _run_loader(con, log, by_loader, "sackmann_matches",
                    lambda sid, rf: load_sackmann.load_matches(con, sid, rf, bb))
        _run_loader(con, log, by_loader, "sackmann_rankings",
                    lambda sid, rf: load_sackmann.load_rankings(con, sid, rf, bb))

        # 4. overrides
        ov = overrides.load(con, overrides_path)
        con.commit()
        log(f"overrides: {len(ov.link)} link, {sum(len(v) for v in ov.forbid.values())} forbid")

        # 5-6. point sources, resolved onto the backbone
        resolver = PlayerResolver(con, ov)
        mcp = load_mcp.McpState()
        _run_loader(con, log, by_loader, "mcp_matches",
                    lambda sid, rf: load_mcp.load_matches(con, sid, rf, resolver, ov, mcp))
        _run_loader(con, log, by_loader, "mcp_points",
                    lambda sid, rf: load_mcp.load_points(con, sid, rf, mcp))
        slam = load_slam_pbp.SlamState()
        _run_loader(con, log, by_loader, "slam_pbp_matches",
                    lambda sid, rf: load_slam_pbp.load_matches(con, sid, rf, resolver, ov, slam))
        _run_loader(con, log, by_loader, "slam_pbp_points",
                    lambda sid, rf: load_slam_pbp.load_points(con, sid, rf, slam))

        # 7. verify
        checks = verify(con)
        report = build_report(con, checks, ignored, resolver, mcp)
        con.execute("INSERT INTO schema_meta VALUES ('built_at', ?)",
                    (datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),))
        con.commit()
        failed = {k: v for k, v in checks.items() if v}
        if failed:
            if report_path:
                _write_report(report_path, report)
            raise BuildError("invariant checks failed (database left at "
                             f"{tmp}):\n{json.dumps(failed, indent=1, default=str)[:4000]}")
        log("invariants: all passed")

        # 8. publish
        con.execute("ANALYZE")
        con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        con.commit()
    finally:
        con.close()
    try:
        # Old -wal/-shm must go first: SQLite would replay a stale WAL into the new file.
        db.remove_db(db_path)
        tmp.replace(db_path)
    except PermissionError as e:
        raise BuildError(f"could not replace {db_path} ({e}). Close any program that has it open "
                         f"(DB browser, notebook, editor); the finished build is at {tmp}") from e
    db.remove_db(tmp)          # leftover -wal/-shm of the temp name, if any
    if report_path:
        _write_report(report_path, report)
    log(f"published {db_path}")
    return report


def verify(con: sqlite3.Connection) -> dict[str, list]:
    """Each value lists offending rows (first 20); every list must be empty."""
    q = lambda sql: [tuple(r) for r in con.execute(sql + " LIMIT 20")]
    return {
        "v_inv_load_counts": q("SELECT snapshot_id, file_relpath, loader, problem FROM v_inv_load_counts"),
        "v_inv_same_day_conflicts": q("SELECT * FROM v_inv_same_day_conflicts"),
        "v_inv_point_players": q("SELECT * FROM v_inv_point_players"),
        "foreign_key_check": q("SELECT * FROM pragma_foreign_key_check"),
        "overrides_honoured": overrides.verify(con),
        "player_xref_multi_tour": q(
            """SELECT x.source, x.source_key FROM player_xref x JOIN player p USING (player_id)
               GROUP BY x.source, x.source_key HAVING COUNT(DISTINCT p.tour) > 1"""),
    }


def build_report(con, checks, ignored, resolver, mcp) -> dict:
    one = lambda sql, *a: con.execute(sql, a).fetchone()[0]
    tables = ["player", "tournament", "tournament_edition", "matches", "match_player_stats",
              "ranking", "point", "player_xref", "edition_xref", "match_xref", "review_queue",
              "override_link", "ingest_snapshot", "load_run"]
    loads = [dict(zip(("source", "loader", "files", "read", "inserted", "linked", "skipped", "queued"), r))
             for r in con.execute(
                 """SELECT s.source, r.loader, COUNT(*), SUM(rows_read), SUM(rows_inserted),
                           SUM(rows_linked), SUM(rows_skipped), SUM(rows_queued)
                    FROM load_run r JOIN ingest_snapshot s USING (snapshot_id)
                    GROUP BY s.source, r.loader ORDER BY MIN(r.load_run_id)""")]
    skips = [dict(zip(("loader", "reason", "n"), r)) for r in con.execute(
        """SELECT r.loader, k.reason, SUM(k.n) FROM load_skip k JOIN load_run r USING (load_run_id)
           GROUP BY r.loader, k.reason ORDER BY r.loader, SUM(k.n) DESC""")]
    review = [dict(zip(("entity_type", "source", "reason", "n"), r)) for r in con.execute(
        """SELECT entity_type, source, reason, COUNT(*) FROM review_queue
           GROUP BY 1,2,3 ORDER BY 1,2,4 DESC""")]
    coverage = [dict(zip(("source", "tour", "linked_keys", "distinct_matches", "with_points"), r))
                for r in con.execute(
        """SELECT x.source, t.tour, COUNT(*), COUNT(DISTINCT x.match_id),
                  COUNT(DISTINCT CASE WHEN EXISTS (SELECT 1 FROM point p WHERE p.match_id=x.match_id
                        AND p.data_source=x.source) THEN x.match_id END)
           FROM match_xref x JOIN matches m USING (match_id)
           JOIN tournament_edition e USING (edition_id) JOIN tournament t USING (tournament_id)
           WHERE x.source IN ('mcp','sackmann_slam_pbp') GROUP BY 1,2""")]
    return {
        "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "schema_version": one("SELECT value FROM schema_meta WHERE key='schema_version'"),
        "snapshots": [dict(zip(("source", "source_ref", "files", "rows"), r)) for r in con.execute(
            "SELECT source, source_ref, COUNT(*), SUM(row_count) FROM ingest_snapshot GROUP BY 1,2")],
        "backbone_last_edition_start": {t: one(
            """SELECT MAX(start_date) FROM tournament_edition JOIN tournament USING (tournament_id)
               WHERE tour=?""", t) for t in ("atp", "wta")},
        "table_counts": {t: one(f"SELECT COUNT(*) FROM {t}") for t in tables},
        "loads": loads,
        "skip_reasons": skips,
        "review_queue": review,
        "player_resolution": resolver.outcomes,
        "point_source_coverage": coverage,
        "mcp_edition_xref_conflicts": mcp.edition_conflicts,
        "ignored_files": len(ignored),
        "ignored_by_reason": _count(r for _, _, r in ignored),
        "invariants": {k: ("ok" if not v else v) for k, v in checks.items()},
    }


def _count(it):
    out = {}
    for x in it:
        out[x] = out.get(x, 0) + 1
    return out


def _write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=1, ensure_ascii=False, default=str)
        f.write("\n")
