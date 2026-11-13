"""Export open review_queue items to a CSV sheet for hand review.

    python scripts/export_review.py            # -> data/review_open.csv

One row per open item, with its top candidates written as stable source keys
(e.g. sackmann_wta:202468) so a confirmed row can be copied straight into
data/overrides.csv:

    entity_type,source,source_key,action,target_source,target_key,note
    player,sackmann_slam_pbp,wta:2018:ausopen:D. Gavrilova,link,sackmann_wta,211148,Daria Saville

Then rebuild with scripts/build_db.py; the override wins over every automatic rule.
"""
from __future__ import annotations

import csv
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

XREF = {"player": ("player_xref", "player_id"), "match": ("match_xref", "match_id"),
        "edition": ("edition_xref", "edition_id")}


def backbone_key(con, entity: str, entity_id: int) -> str:
    table, col = XREF[entity]
    row = con.execute(f"""SELECT source || ':' || source_key FROM {table}
                          WHERE {col}=? AND source LIKE 'sackmann\\_%' ESCAPE '\\' AND method='seed'
                          LIMIT 1""", (entity_id,)).fetchone()
    return row[0] if row else f"?:{entity_id}"


def describe(con, entity: str, entity_id: int) -> str:
    if entity == "player":
        r = con.execute("SELECT canonical_name, dob, country FROM player WHERE player_id=?", (entity_id,)).fetchone()
        return f"{r[0]} ({r[1] or '?'}, {r[2] or '?'})" if r else "?"
    if entity == "match":
        r = con.execute("""SELECT t.canonical_name, e.start_date, m.round, w.canonical_name, l.canonical_name, m.score
                           FROM matches m JOIN tournament_edition e USING (edition_id)
                           JOIN tournament t USING (tournament_id)
                           JOIN player w ON w.player_id=m.winner_id JOIN player l ON l.player_id=m.loser_id
                           WHERE m.match_id=?""", (entity_id,)).fetchone()
        return f"{r[0]} {r[1]} {r[2]}: {r[3]} d. {r[4]} {r[5] or ''}" if r else "?"
    return "?"


def main() -> int:
    db = ROOT / "tennis.db"
    out = ROOT / "data" / "review_open.csv"
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = con.execute("""SELECT entity_type, source, source_key, reason, candidate_ids, detail
                          FROM review_queue WHERE status='open' ORDER BY entity_type, source, reason, source_key""").fetchall()
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["entity_type", "source", "source_key", "reason", "detail",
                    "candidate_1", "candidate_1_desc", "candidate_2", "candidate_2_desc", "candidate_3", "candidate_3_desc"])
        for ent, src, key, reason, cands, detail in rows:
            ids = json.loads(cands) if cands else []
            cells = []
            for cid in ids[:3]:
                cells += [backbone_key(con, ent, cid), describe(con, ent, cid)]
            w.writerow([ent, src, key, reason, detail or "", *cells])
    print(f"{len(rows)} open items -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
