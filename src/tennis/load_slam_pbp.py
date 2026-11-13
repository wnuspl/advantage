"""Grand slam point-by-point (singles): resolve matches onto backbone slam editions, load points.

Verified against the real files (mirror commit 83733587, 2026-11-13):
- YYYY-<slam>-matches.csv: match_id,year,slam,match_num,player1,player2,... Other columns
  (round, ids, nations) are blank in recent files, so we rely on match_num:
  1xxx = men, 2xxx = women; the second digit is the round (1 = R128 ... 7 = F).
- YYYY-<slam>-points.csv: columns vary by year. Used: match_id, PointNumber, PointServer,
  PointWinner (1/2 = player1/player2), SetNo, GameNo, ServeNumber or ServeIndicator,
  P{1,2}Ace, P{1,2}DoubleFault, P{1,2}Winner, P{1,2}UnfErr, P{1,2}ForcedError, Rally or
  RallyCount. Rows with PointNumber 0 / '0X' / '0Y' are placeholders.
- AO/FO from 2018 have placeholder rally lengths (README): rally_len is left NULL there.
"""
from __future__ import annotations

import csv
import re
import sqlite3
from dataclasses import dataclass, field

from .load_mcp import flush_points
from .loadrun import LoadRun
from .normalize import norm_name, to_int
from .overrides import Overrides
from .resolve import PlayerResolver, link_match, match_candidates_edition, opponents, queue
from .snapshots import RawFile, open_csv

SOURCE = "sackmann_slam_pbp"
SLAM_NAMES = {"ausopen": {"australian open"}, "frenchopen": {"roland garros", "french open"},
              "wimbledon": {"wimbledon"}, "usopen": {"us open"}}
ROUND_DIGIT = {"1": "R128", "2": "R64", "3": "R32", "4": "R16", "5": "QF", "6": "SF", "7": "F"}
TOUR_DIGIT = {"1": "atp", "2": "wta", "M": "atp", "W": "wta"}


@dataclass
class SlamState:
    known: set[str] = field(default_factory=set)
    linked: dict[str, tuple[int, int, int, str, int]] = field(default_factory=dict)  # id -> (match,p1,p2,slam,year)
    editions: dict[tuple[str, str, int], tuple[int, str]] | None = None
    participants: dict[int, set[int]] = field(default_factory=dict)   # edition -> players


def _participants(con, st: "SlamState", edition_id: int) -> set[int]:
    if edition_id not in st.participants:
        st.participants[edition_id] = {
            pid for row in con.execute(
                "SELECT winner_id, loser_id FROM matches WHERE edition_id=?", (edition_id,))
            for pid in row}
    return st.participants[edition_id]


def parse_match_num(num: str) -> tuple[str | None, str | None]:
    """'1101' -> ('atp','R128'); 'WS301' (2021 AO) -> ('wta','R32'); otherwise (None, None)."""
    m = re.fullmatch(r"([12])(\d)\d\d", num) or re.fullmatch(r"([MW])S(\d)\d\d", num)
    if not m:
        return None, None
    return TOUR_DIGIT.get(m.group(1)), ROUND_DIGIT.get(m.group(2))


def _slam_editions(con) -> dict[tuple[str, str, int], tuple[int, str]]:
    out = {}
    for ed, year, start, name, tour in con.execute(
            """SELECT e.edition_id, e.year, e.start_date, t.canonical_name, t.tour
               FROM tournament_edition e JOIN tournament t USING (tournament_id) WHERE e.level='G'"""):
        n = norm_name(name)
        for slam, names in SLAM_NAMES.items():
            if n in names:
                out[(tour, slam, year)] = (ed, start)
    return out


def load_matches(con: sqlite3.Connection, snapshot_id: int, rf: RawFile, resolver: PlayerResolver,
                 ov: Overrides, st: SlamState) -> LoadRun:
    if st.editions is None:
        st.editions = _slam_editions(con)
    run = LoadRun(con, snapshot_id, rf.loader)
    with open_csv(rf.path) as f:
        rows = list(enumerate(csv.DictReader(f), start=2))
    valid = []
    for line, row in rows:
        run.row()
        mid = (row.get("match_id") or "").strip()
        num = (row.get("match_num") or "").strip()
        slam = (row.get("slam") or "").strip().lower()
        year = to_int(row.get("year"))
        p1, p2 = (row.get("player1") or "").strip(), (row.get("player2") or "").strip()
        if not mid or not num or slam not in SLAM_NAMES or year is None:
            run.skipped("malformed_row", line); continue
        tour, rnd = parse_match_num(num)
        if tour is None or rnd is None:
            run.skipped("not_singles_main_draw", f"{line}:{num}"); continue
        if len(p1) < 3 or len(p2) < 3:
            run.skipped("missing_player_name", line); continue
        if mid in st.known:
            run.skipped("duplicate_match_id", line); continue
        st.known.add(mid)
        ed = st.editions.get((tour, slam, year))
        if ed is None:
            queue(con, "match", SOURCE, mid, "no_edition", None,
                  {"slam": slam, "year": year, "tour": tour}, snapshot_id)
            run.queued(); continue
        valid.append((line, mid, tour, rnd, slam, year, ed, p1, p2))

    # Name keys are scoped to the edition ('atp:2020:frenchopen:A Murray'): this source writes
    # names inconsistently across years, and the edition's participants are strong context.
    def pid_for(tour, slam, year, ed, start, name):
        return resolver.resolve(SOURCE, f"{tour}:{year}:{slam}:{name}", tour, name, [start],
                                snapshot_id, context=_participants(con, st, ed))

    for line, mid, tour, rnd, slam, year, (ed, start), p1, p2 in valid:
        a = pid_for(tour, slam, year, ed, start, p1)
        b = pid_for(tour, slam, year, ed, start, p2)
        if (a is None) != (b is None):
            raw = p1 if a is None else p2
            got = resolver.resolve_by_opponent(
                SOURCE, f"{tour}:{year}:{slam}:{raw}", raw,
                opponents(con, a or b, rnd, edition_id=ed), snapshot_id)
            a, b = (got, b) if a is None else (a, got)
        detail = {"slam": slam, "year": year, "round": rnd, "player1": p1, "player2": p2}
        if a is None or b is None:
            queue(con, "match", SOURCE, mid, "player_unresolved", None, detail, snapshot_id)
            run.queued(); continue
        if a == b:
            run.skipped("same_player_both_sides", line); continue
        match_id, _ = link_match(con, ov, SOURCE, mid, match_candidates_edition(con, a, b, ed), rnd,
                                 snapshot_id, detail)
        if match_id is None:
            run.queued(); continue
        st.linked[mid] = (match_id, a, b, slam, year)
        run.linked()
    run.finish()
    return run


def _flag(row, col) -> int | None:
    v = (row.get(col) or "").strip()
    return None if v == "" else int(v == "1")


def load_points(con: sqlite3.Connection, snapshot_id: int, rf: RawFile, st: SlamState) -> LoadRun:
    run = LoadRun(con, snapshot_id, rf.loader)
    buf = []
    with open_csv(rf.path) as f:
        for line, row in enumerate(csv.DictReader(f), start=2):
            run.row()
            mid = (row.get("match_id") or "").strip()
            if mid not in st.known:
                run.skipped("match_row_skipped_or_missing", mid); continue
            link = st.linked.get(mid)
            if link is None:
                run.skipped("match_not_linked", mid); continue
            idx = to_int(row.get("PointNumber"))
            if not idx or idx < 0:
                run.skipped("placeholder_point", line); continue
            svr, won = (row.get("PointServer") or "").strip(), (row.get("PointWinner") or "").strip()
            if svr not in ("1", "2") or won not in ("1", "2"):
                run.skipped("malformed_row", line); continue
            match_id, p1, p2, slam, year = link
            server, returner = (p1, p2) if svr == "1" else (p2, p1)
            winner = p1 if won == "1" else p2
            loser_side = "2" if won == "1" else "1"
            serve_no = to_int(row.get("ServeNumber")) or to_int(row.get("ServeIndicator"))
            is_ace, is_df = _flag(row, f"P{svr}Ace"), _flag(row, f"P{svr}DoubleFault")
            if is_ace:
                end = "ace"
            elif is_df:
                end = "double_fault"
            elif _flag(row, f"P{won}Winner"):
                end = "winner"
            elif _flag(row, f"P{loser_side}UnfErr"):
                end = "unforced_error"
            elif _flag(row, f"P{loser_side}ForcedError"):
                end = "forced_error"
            else:
                end = None
            rally = to_int(row.get("RallyCount")) if row.get("RallyCount") not in (None, "") \
                else to_int(row.get("Rally"))
            if (slam in ("ausopen", "frenchopen") and year >= 2018) or (rally is not None and rally < 0):
                rally = None
            buf.append((match_id, SOURCE, idx, to_int(row.get("SetNo")), to_int(row.get("GameNo")),
                        server, returner, winner, serve_no if serve_no in (1, 2) else None,
                        is_ace, is_df, rally, end, None, None, None, None, None, snapshot_id))
            if len(buf) >= 20000:
                flush_points(con, run, buf)
    flush_points(con, run, buf)
    run.finish()
    return run
