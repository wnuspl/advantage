"""Hand-confirmed links and forced non-matches from data/overrides.csv.

Columns: entity_type,source,source_key,action,target_source,target_key,note
  entity_type  player | edition | match
  action       link (source_key IS target) | forbid (source_key is NOT target)
               | reassign (player only: a backbone player id's matches in a date span
                 belong to the target; source_key = <id>@<from>[..<to>], edition start
                 dates, from inclusive, to exclusive). For a Sackmann id shared by two
                 people: the rest of the record stays where it is.
  target_*     a stable source key of the target entity, e.g. sackmann_atp,104925
Targets are looked up in the *_xref tables, so overrides survive rebuilds even though
surrogate IDs do not. Resolvers consult overrides before any automatic rule; verify()
re-checks at the end of the build that every override was honoured.
"""
from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

from .snapshots import BuildError

BACKBONE = ("sackmann_atp", "sackmann_wta")

_XREF = {"player": ("player_xref", "player_id"),
         "edition": ("edition_xref", "edition_id"),
         "match": ("match_xref", "match_id")}
COLUMNS = ["entity_type", "source", "source_key", "action", "target_source", "target_key", "note"]


class Overrides:
    def __init__(self) -> None:
        self.link: dict[tuple[str, str, str], int] = {}
        self.forbid: dict[tuple[str, str, str], set[int]] = {}
        self.reassign: list[tuple[str, str, str, str | None, int]] = []   # source, id, from, to, target

    def link_for(self, entity: str, source: str, key: str) -> int | None:
        return self.link.get((entity, source, key))

    def forbidden(self, entity: str, source: str, key: str) -> set[int]:
        return self.forbid.get((entity, source, key), set())


def parse_span(key: str) -> tuple[str, str, str | None] | None:
    """'118808@2022-01-01' / '118808@2022-01-01..2024-01-01' -> (id, from, to); None if malformed."""
    sid, sep, span = key.partition("@")
    lo, _, hi = span.partition("..")
    ok = lambda d: len(d) == 10 and d[4] == d[7] == "-" and d.replace("-", "").isdigit()   # noqa: E731
    if not sid or not sep or not ok(lo) or (hi and (not ok(hi) or hi <= lo)):
        return None
    return sid, lo, hi or None


def load(con: sqlite3.Connection, path: Path) -> Overrides:
    ov = Overrides()
    if not path.exists():
        return ov
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != COLUMNS:
            raise BuildError(f"{path}: header must be {','.join(COLUMNS)}; got {reader.fieldnames}")
        for line, row in enumerate(reader, start=2):
            ent, src, key, act = row["entity_type"], row["source"], row["source_key"], row["action"]
            if ent not in _XREF or act not in ("link", "forbid", "reassign"):
                raise BuildError(f"{path}:{line}: bad entity_type/action {ent!r}/{act!r}")
            if act == "reassign" and (ent != "player" or src not in BACKBONE or parse_span(key) is None):
                raise BuildError(f"{path}:{line}: reassign needs entity_type player, a source in "
                                 f"{BACKBONE} and source_key <id>@<from>[..<to>]; got {ent}/{src}/{key!r}")
            table, col = _XREF[ent]
            hit = con.execute(f"SELECT {col} FROM {table} WHERE source=? AND source_key=?",
                              (row["target_source"], row["target_key"])).fetchone()
            if hit is None:
                raise BuildError(f"{path}:{line}: target {row['target_source']}:{row['target_key']} "
                                 f"does not resolve to any {ent}")
            tid = hit[0]
            if act == "reassign":
                sid, lo, hi = parse_span(key)
                ov.reassign.append((src, sid, lo, hi, tid))
            elif act == "link":
                prev = ov.link.get((ent, src, key))
                if prev is not None and prev != tid:
                    raise BuildError(f"{path}:{line}: two different link targets for {ent} {src}:{key}")
                ov.link[(ent, src, key)] = tid
            else:
                ov.forbid.setdefault((ent, src, key), set()).add(tid)
            con.execute(
                """INSERT OR IGNORE INTO override_link (entity_type, source, source_key, action,
                       target_source, target_key, target_id, note) VALUES (?,?,?,?,?,?,?,?)""",
                (ent, src, key, act, row["target_source"], row["target_key"], tid, row.get("note")))
    for k, tid in ov.link.items():
        if tid in ov.forbid.get(k, set()):
            raise BuildError(f"{path}: {k} is both linked and forbidden to the same target")
    return ov


def _span_matches(con, pid: int, lo: str, hi: str | None, source: str) -> list[int]:
    return [r[0] for r in con.execute(
        """SELECT DISTINCT m.match_id FROM matches m JOIN tournament_edition e USING (edition_id)
           JOIN match_xref x ON x.match_id = m.match_id AND x.source = ?
           WHERE ? IN (m.winner_id, m.loser_id) AND e.start_date >= ? AND (? IS NULL OR e.start_date < ?)""",
        (source, pid, lo, hi, hi))]


def apply_reassign(con: sqlite3.Connection, ov: Overrides, log=print) -> int:
    """Move each reassign span's backbone matches (and their stat rows) to the target.
    Runs after the backbone loads and before any other source resolves against it."""
    moved = 0
    for src, sid, lo, hi, tid in ov.reassign:
        hit = con.execute("SELECT player_id FROM player_xref WHERE source=? AND source_key=? AND method='seed'",
                          (src, sid)).fetchone()
        if hit is None:
            raise BuildError(f"reassign {src}:{sid}@{lo}: no such backbone player")
        pid = hit[0]
        if pid == tid:
            raise BuildError(f"reassign {src}:{sid}@{lo}: target is the same player")
        ids = _span_matches(con, pid, lo, hi, src)
        for mid in ids:
            w, l = con.execute("SELECT winner_id, loser_id FROM matches WHERE match_id=?", (mid,)).fetchone()
            if tid in (w, l):
                raise BuildError(f"reassign {src}:{sid}@{lo}: match {mid} would put the target on both sides")
            try:
                con.execute("""UPDATE matches SET winner_id = CASE WHEN winner_id=? THEN ? ELSE winner_id END,
                                   loser_id = CASE WHEN loser_id=? THEN ? ELSE loser_id END WHERE match_id=?""",
                            (pid, tid, pid, tid, mid))
            except sqlite3.IntegrityError as e:
                raise BuildError(f"reassign {src}:{sid}@{lo}: match {mid} collides with an existing match ({e})")
            con.execute("UPDATE match_player_stats SET player_id=? WHERE match_id=? AND player_id=?", (tid, mid, pid))
        log(f"overrides: reassigned {len(ids)} matches of {src}:{sid} from {lo}{'' if hi is None else ' to ' + hi}")
        moved += len(ids)
    return moved


def verify(con: sqlite3.Connection) -> list[str]:
    """Problems where the database disagrees with an override (should be none)."""
    problems = []
    for ent, src, key, act, tid in con.execute(
            "SELECT entity_type, source, source_key, action, target_id FROM override_link"):
        if act == "reassign":
            sid, lo, hi = parse_span(key)
            pid = con.execute("SELECT player_id FROM player_xref WHERE source=? AND source_key=? AND method='seed'",
                              (src, sid)).fetchone()
            left = _span_matches(con, pid[0], lo, hi, src) if pid else []
            if left:
                problems.append(f"reassign not applied: {src}:{key} still has {len(left)} matches")
            continue
        table, col = _XREF[ent]
        hit = con.execute(f"SELECT {col}, method FROM {table} WHERE source=? AND source_key=?",
                          (src, key)).fetchone()
        if act == "link" and (hit is None or hit[0] != tid or hit[1] != "override"):
            problems.append(f"link not applied: {ent} {src}:{key} -> {tid} (have {hit})")
        if act == "forbid" and hit is not None and hit[0] == tid:
            problems.append(f"forbidden link present: {ent} {src}:{key} -> {tid}")
    return problems
