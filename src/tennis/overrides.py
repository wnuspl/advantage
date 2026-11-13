"""Hand-confirmed links and forced non-matches from data/overrides.csv.

Columns: entity_type,source,source_key,action,target_source,target_key,note
  entity_type  player | edition | match
  action       link (source_key IS target) | forbid (source_key is NOT target)
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

_XREF = {"player": ("player_xref", "player_id"),
         "edition": ("edition_xref", "edition_id"),
         "match": ("match_xref", "match_id")}
COLUMNS = ["entity_type", "source", "source_key", "action", "target_source", "target_key", "note"]


class Overrides:
    def __init__(self) -> None:
        self.link: dict[tuple[str, str, str], int] = {}
        self.forbid: dict[tuple[str, str, str], set[int]] = {}

    def link_for(self, entity: str, source: str, key: str) -> int | None:
        return self.link.get((entity, source, key))

    def forbidden(self, entity: str, source: str, key: str) -> set[int]:
        return self.forbid.get((entity, source, key), set())


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
            if ent not in _XREF or act not in ("link", "forbid"):
                raise BuildError(f"{path}:{line}: bad entity_type/action {ent!r}/{act!r}")
            table, col = _XREF[ent]
            hit = con.execute(f"SELECT {col} FROM {table} WHERE source=? AND source_key=?",
                              (row["target_source"], row["target_key"])).fetchone()
            if hit is None:
                raise BuildError(f"{path}:{line}: target {row['target_source']}:{row['target_key']} "
                                 f"does not resolve to any {ent}")
            tid = hit[0]
            if act == "link":
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


def verify(con: sqlite3.Connection) -> list[str]:
    """Problems where the database disagrees with an override (should be none)."""
    problems = []
    for ent, src, key, act, tid in con.execute(
            "SELECT entity_type, source, source_key, action, target_id FROM override_link"):
        table, col = _XREF[ent]
        hit = con.execute(f"SELECT {col}, method FROM {table} WHERE source=? AND source_key=?",
                          (src, key)).fetchone()
        if act == "link" and (hit is None or hit[0] != tid or hit[1] != "override"):
            problems.append(f"link not applied: {ent} {src}:{key} -> {tid} (have {hit})")
        if act == "forbid" and hit is not None and hit[0] == tid:
            problems.append(f"forbidden link present: {ent} {src}:{key} -> {tid}")
    return problems
