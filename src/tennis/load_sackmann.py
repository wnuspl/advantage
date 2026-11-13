"""Backbone loaders for the Sackmann ATP/WTA files: players, editions + matches + stats, rankings.

Verified against the real files (mirror commit 83733587, 2026-11-13):
- players: player_id,name_first,name_last,hand,dob,ioc,height,wikidata_id; dob YYYYMMDD or blank.
- matches: 49 columns, tourney_date YYYYMMDD (Monday of the event week), no per-match date.
  ATP and WTA player IDs overlap numerically, so they are separate sources.
- (tourney_id, match_num) is NOT unique (WTA main vs qualifying files), and the qual/ITF
  files repeat ~12k matches verbatim. Source key = tid:match_num:round:winner:loser.
- rankings: ranking_date,rank,player,points[,tours].
"""
from __future__ import annotations

import csv
import json
import sqlite3
from dataclasses import dataclass, field

from .loadrun import LoadRun
from .normalize import (ROUND_ORDER, is_placeholder_name, norm_hand, norm_name, norm_round,
                        norm_surface, parse_ymd, score_status, to_int)
from .snapshots import RawFile, open_csv

TOUR_OF_SOURCE = {"sackmann_atp": "atp", "sackmann_wta": "wta"}


@dataclass
class Backbone:
    """In-memory caches shared by the backbone loaders within one build."""
    players: dict[tuple[str, str], int] = field(default_factory=dict)   # (source, sackmann id) -> player_id
    placeholders: set[int] = field(default_factory=set)
    tournaments: dict[tuple[str, str], int] = field(default_factory=dict)  # (tour, key) -> tournament_id
    tournament_names: set[tuple[str, str]] = field(default_factory=set)
    editions: dict[tuple[str, str], int] = field(default_factory=dict)  # (source, tourney_id) -> edition_id
    edition_slots: dict[tuple[int, str], str] = field(default_factory=dict)  # (tournament_id, start) -> tid


def load_players(con: sqlite3.Connection, snapshot_id: int, rf: RawFile, bb: Backbone) -> LoadRun:
    tour = TOUR_OF_SOURCE[rf.source]
    run = LoadRun(con, snapshot_id, rf.loader)
    with open_csv(rf.path) as f:
        for line, row in enumerate(csv.DictReader(f), start=2):
            run.row()
            sid = (row.get("player_id") or "").strip()
            if not sid:
                run.skipped("no_player_id", line)
                continue
            if (rf.source, sid) in bb.players:
                run.skipped("duplicate_player_id", f"{line}:{sid}")
                continue
            first = (row.get("name_first") or "").strip() or None
            last = (row.get("name_last") or "").strip() or None
            name = " ".join(x for x in (first, last) if x) or f"[{rf.source} {sid}]"
            placeholder = is_placeholder_name(first, last)
            cur = con.execute(
                """INSERT INTO player (tour, canonical_name, name_first, name_last, dob, hand,
                       height_cm, country, is_placeholder) VALUES (?,?,?,?,?,?,?,?,?)""",
                (tour, name, first, last, parse_ymd(row.get("dob")), norm_hand(row.get("hand")),
                 to_int(row.get("height")), (row.get("ioc") or "").strip() or None, int(placeholder)),
            )
            pid = cur.lastrowid
            bb.players[(rf.source, sid)] = pid
            if placeholder:
                bb.placeholders.add(pid)
            con.execute(
                """INSERT INTO player_xref (source, source_key, player_id, method, confidence,
                       verified, snapshot_id) VALUES (?,?,?,'seed',1.0,1,?)""",
                (rf.source, sid, pid, snapshot_id))
            alias = norm_name(name)
            if alias and not placeholder:
                con.execute("INSERT OR IGNORE INTO player_alias VALUES (?,?,?)", (alias, pid, rf.source))
            run.inserted()
    run.finish()
    return run


def _tournament(con, bb: Backbone, tour: str, name: str, tid: str) -> int:
    key = (tour, norm_name(name) or tid)
    tour_id = bb.tournaments.get(key)
    if tour_id is None:
        canonical = name or tid
        if (tour, canonical) in bb.tournament_names:       # same text, different normalization key
            canonical = f"{canonical} [{tid}]"
        cur = con.execute("INSERT INTO tournament (tour, canonical_name) VALUES (?,?)", (tour, canonical))
        tour_id = cur.lastrowid
        bb.tournaments[key] = tour_id
        bb.tournament_names.add((tour, canonical))
        con.execute("INSERT OR IGNORE INTO tournament_alias VALUES (?,?)", (key[1], tour_id))
    return tour_id


def _edition(con, bb: Backbone, rf: RawFile, snapshot_id: int, tour: str, row: dict, start: str) -> int:
    tid = row["tourney_id"].strip()
    ed = bb.editions.get((rf.source, tid))
    if ed is not None:
        return ed
    name = (row.get("tourney_name") or "").strip()
    tour_id = _tournament(con, bb, tour, name, tid)
    if (tour_id, start) in bb.edition_slots:
        # Two draws with the same name in the same week (e.g. 'Argentina F5' a/b):
        # they are distinct events, so give the second its own tournament.
        tour_id = _tournament(con, bb, tour, f"{name} [{tid}]", tid)
    year = int(tid[:4]) if tid[:4].isdigit() else int(start[:4])
    bo = to_int(row.get("best_of"))
    cur = con.execute(
        """INSERT INTO tournament_edition (tournament_id, year, start_date, level, surface,
               draw_size, best_of) VALUES (?,?,?,?,?,?,?)""",
        (tour_id, year, start, (row.get("tourney_level") or "").strip() or None,
         norm_surface(row.get("surface")), to_int(row.get("draw_size")),
         bo if bo in (1, 3, 5) else None),
    )
    ed = cur.lastrowid
    bb.editions[(rf.source, tid)] = ed
    bb.edition_slots[(tour_id, start)] = tid
    con.execute(
        """INSERT INTO edition_xref (source, source_key, edition_id, method, confidence, verified,
               snapshot_id) VALUES (?,?,?,'seed',1.0,1,?)""", (rf.source, tid, ed, snapshot_id))
    return ed


_STAT_COLS = ["ace", "df", "svpt", "1stIn", "1stWon", "2ndWon", "SvGms", "bpSaved", "bpFaced"]


def _stats_row(match_id, pid, source, is_winner, row, p, snapshot_id):
    side = "winner" if is_winner else "loser"
    return (match_id, pid, source, int(is_winner),
            *[to_int(row.get(f"{p}_{c}")) for c in _STAT_COLS],
            to_int(row.get(f"{side}_rank")), to_int(row.get(f"{side}_rank_points")),
            to_int(row.get(f"{side}_seed")), (row.get(f"{side}_entry") or "").strip() or None,
            snapshot_id)


def load_matches(con: sqlite3.Connection, snapshot_id: int, rf: RawFile, bb: Backbone) -> LoadRun:
    tour = TOUR_OF_SOURCE[rf.source]
    run = LoadRun(con, snapshot_id, rf.loader)
    stats_buf = []
    with open_csv(rf.path) as f:
        for line, row in enumerate(csv.DictReader(f), start=2):
            run.row()
            tid = (row.get("tourney_id") or "").strip()
            wid_s, lid_s = (row.get("winner_id") or "").strip(), (row.get("loser_id") or "").strip()
            rnd = norm_round(row.get("round"))
            start = parse_ymd(row.get("tourney_date"))
            score = (row.get("score") or "").strip() or None
            status = score_status(score)
            key = f"{tid}:{(row.get('match_num') or '').strip()}:{rnd}:{wid_s}:{lid_s}"
            if not tid or start is None:
                run.skipped("bad_tourney_id_or_date", line); continue
            wid, lid = bb.players.get((rf.source, wid_s)), bb.players.get((rf.source, lid_s))
            if wid is None or lid is None:
                run.skipped("unknown_player_id", line); continue
            if wid == lid:
                run.skipped("same_player_both_sides", line); continue
            if wid in bb.placeholders or lid in bb.placeholders:
                run.skipped("placeholder_player", line); continue
            if rnd is None:
                run.skipped("bad_round", f"{line}:{row.get('round')}"); continue
            if status == "bye":
                run.skipped("bye", line); continue
            ed = _edition(con, bb, rf, snapshot_id, tour, row, start)
            bo = to_int(row.get("best_of"))
            cur = con.execute(
                """INSERT OR IGNORE INTO matches (edition_id, round, round_order, match_date,
                       date_precision, winner_id, loser_id, score, status, best_of, minutes,
                       snapshot_id) VALUES (?,?,?,NULL,'week',?,?,?,?,?,?,?)""",
                (ed, rnd, ROUND_ORDER[rnd], wid, lid, score, status,
                 bo if bo in (1, 3, 5) else None, to_int(row.get("minutes")), snapshot_id))
            if cur.rowcount == 1:
                mid = cur.lastrowid
                con.execute(
                    """INSERT INTO match_xref (source, source_key, match_id, method, confidence,
                           verified, snapshot_id) VALUES (?,?,?,'seed',1.0,1,?)""",
                    (rf.source, key, mid, snapshot_id))
                stats_buf.append(_stats_row(mid, wid, rf.source, True, row, "w", snapshot_id))
                stats_buf.append(_stats_row(mid, lid, rf.source, False, row, "l", snapshot_id))
                if len(stats_buf) >= 20000:
                    _flush_stats(con, stats_buf)
                run.inserted()
                continue
            # Same edition + round + pair already loaded.
            hit = con.execute(
                """SELECT match_id, winner_id, score FROM matches WHERE edition_id=? AND round=?
                       AND pair_lo_id=? AND pair_hi_id=?""",
                (ed, rnd, min(wid, lid), max(wid, lid))).fetchone()
            if hit is None:
                raise RuntimeError(f"{rf.relpath}:{line}: insert ignored but no conflicting match found")
            if hit[1] == wid and hit[2] == score:
                cur = con.execute(
                    """INSERT OR IGNORE INTO match_xref (source, source_key, match_id, method,
                           confidence, verified, snapshot_id) VALUES (?,?,?,'seed',1.0,1,?)""",
                    (rf.source, key, hit[0], snapshot_id))
                if cur.rowcount == 1:
                    run.linked()          # same match listed again under another match_num
                else:
                    run.skipped("exact_duplicate_row", line)
            else:
                con.execute(
                    """INSERT OR IGNORE INTO review_queue (entity_type, source, source_key,
                           candidate_ids, reason, detail, snapshot_id)
                       VALUES ('match',?,?,?,'duplicate_key_conflict',?,?)""",
                    (rf.source, key, json.dumps([hit[0]]),
                     json.dumps({"file": rf.relpath, "line": line, "score": score,
                                 "existing_score": hit[2]}), snapshot_id))
                run.queued()
    _flush_stats(con, stats_buf)
    run.finish()
    return run


def _flush_stats(con, buf):
    if buf:
        con.executemany(
            f"INSERT INTO match_player_stats VALUES ({','.join('?' * 18)})", buf)
        buf.clear()


def load_rankings(con: sqlite3.Connection, snapshot_id: int, rf: RawFile, bb: Backbone) -> LoadRun:
    tour = TOUR_OF_SOURCE[rf.source]
    run = LoadRun(con, snapshot_id, rf.loader)
    buf = []

    def flush():
        before = con.total_changes
        con.executemany("INSERT OR IGNORE INTO ranking VALUES (?,?,?,?,?,?)", buf)
        ins = con.total_changes - before
        run.n_inserted += ins
        if len(buf) - ins:
            run.skipped("duplicate_ranking_row", n=len(buf) - ins)
        buf.clear()

    with open_csv(rf.path) as f:
        for line, row in enumerate(csv.DictReader(f), start=2):
            run.row()
            d = parse_ymd(row.get("ranking_date"))
            rank = to_int(row.get("rank"))
            pid = bb.players.get((rf.source, (row.get("player") or "").strip()))
            if d is None:
                run.skipped("bad_date", line); continue
            if rank is None:
                run.skipped("bad_rank", line); continue
            if pid is None:
                run.skipped("unknown_player_id", line); continue
            # All other constraints are validated above, so OR IGNORE only drops PK duplicates.
            buf.append((tour, d, pid, rank, to_int(row.get("points")), snapshot_id))
            if len(buf) >= 50000:
                flush()
    flush()
    run.finish()
    return run
