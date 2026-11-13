"""Match Charting Project: resolve charted matches onto the backbone, then load points.

Verified against the real files (MCP commit 1813a130, 2026-11-13):
- charting-{m,w}-matches.csv: match_id,Player 1,Player 2,Pl 1 hand,Pl 2 hand,Date,Tournament,
  Round,Time,Court,Surface,Umpire,Best of,Final TB?,Charted by. Date is YYYYMMDD.
  match_id = YYYYMMDD-{M,W}-Tournament-Round-Player_1-Player_2. A few dozen rows have
  shifted columns (a hand letter in a name field, a venue in Round): skipped as malformed.
- charting-{m,w}-points-*.csv: match_id,Pt,Set1,Set2,Gm1,Gm2,Pts,Gm#,TbSet,Svr,1st,2nd,
  Notes,PtWinner. Svr/PtWinner are 1/2 = Player 1/Player 2 of the matches file.
- MCP runs past the mirror's cutoff (mirror ends 2026-05-25); later charts cannot resolve
  until the backbone is extended, so they are skipped as after_backbone_cutoff.
"""
from __future__ import annotations

import csv
import re
import sqlite3
from dataclasses import dataclass, field

from .loadrun import LoadRun
from .normalize import add_days, norm_round, parse_ymd, to_int
from .overrides import Overrides
from .resolve import (MAIN_DRAW_LEAD_DAYS, PlayerResolver, link_match, match_candidates_window,
                      opponents, queue)
from .snapshots import RawFile, open_csv

SOURCE = "mcp"
TOUR = {"m": "atp", "w": "wta"}
# Edition start (Sackmann tourney_date, the event's Monday) relative to the charted day:
# qualifying can be ~9 days before it, a two-week final ~16 days after.
WINDOW_BEFORE, WINDOW_AFTER = 16, 9
SHOT_LETTERS = set("fbrsvzopuylmhijktq")


@dataclass
class McpState:
    known: set[str] = field(default_factory=set)            # valid match_ids in matches files
    linked: dict[str, tuple[int, int, int]] = field(default_factory=dict)  # mcp id -> (match, p1, p2)
    primary: set[str] = field(default_factory=set)          # first chart of each backbone match
    charted_matches: set[int] = field(default_factory=set)
    edition_conflicts: int = 0


def gender_of(rf: RawFile) -> str:
    m = re.match(r"charting-([mw])-", rf.relpath.rsplit("/", 1)[-1])
    if not m:
        raise ValueError(f"cannot tell tour from {rf.relpath}")
    return m.group(1)


def load_matches(con: sqlite3.Connection, snapshot_id: int, rf: RawFile, resolver: PlayerResolver,
                 ov: Overrides, st: McpState) -> LoadRun:
    g = gender_of(rf)
    tour = TOUR[g]
    run = LoadRun(con, snapshot_id, rf.loader)
    cutoff = con.execute(
        """SELECT MAX(e.start_date) FROM tournament_edition e JOIN tournament t USING (tournament_id)
           WHERE t.tour=?""", (tour,)).fetchone()[0]
    with open_csv(rf.path) as f:
        rows = list(enumerate(csv.DictReader(f), start=2))

    valid, seen, name_dates = [], set(), {}
    for line, row in rows:
        run.row()
        mid = (row.get("match_id") or "").strip()
        d = parse_ymd(row.get("Date"))
        p1, p2 = (row.get("Player 1") or "").strip(), (row.get("Player 2") or "").strip()
        rnd = norm_round(row.get("Round"))
        if not mid or d is None or len(p1) < 3 or len(p2) < 3 or p1 == p2:
            run.skipped("malformed_row", f"{line}:{mid[:60]}"); continue
        if rnd is None:
            run.skipped("bad_round", f"{line}:{row.get('Round')!r}"); continue
        if mid in seen:
            run.skipped("duplicate_match_id", f"{line}:{mid[:60]}"); continue
        seen.add(mid)
        valid.append((line, mid, d, p1, p2, rnd, row))
        for p in (p1, p2):
            name_dates.setdefault(p, []).append(d)

    ids = {p: resolver.resolve(SOURCE, f"{g}:{p}", tour, p, ds, snapshot_id)
           for p, ds in name_dates.items()}

    for line, mid, d, p1, p2, rnd, row in valid:
        st.known.add(mid)
        a, b = resolver.cache.get((SOURCE, f"{g}:{p1}")), resolver.cache.get((SOURCE, f"{g}:{p2}"))
        if (a is None) != (b is None) and cutoff is not None and d <= add_days(cutoff, WINDOW_BEFORE):
            raw = p1 if a is None else p2
            hi = add_days(d, WINDOW_AFTER if rnd.startswith("Q") else MAIN_DRAW_LEAD_DAYS)
            got = resolver.resolve_by_opponent(
                SOURCE, f"{g}:{raw}", raw,
                opponents(con, a or b, rnd, start_from=add_days(d, -WINDOW_BEFORE), start_to=hi),
                snapshot_id)
            a, b = (got, b) if a is None else (a, got)
        detail = {"date": d, "tournament": row.get("Tournament"), "round": rnd,
                  "player1": p1, "player2": p2}
        if a is None or b is None:
            queue(con, "match", SOURCE, mid, "player_unresolved", None, detail, snapshot_id)
            run.queued(); continue
        if a == b:
            run.skipped("same_player_both_sides", f"{line}:{mid[:60]}"); continue
        if cutoff is None or d > add_days(cutoff, WINDOW_BEFORE):
            run.skipped("after_backbone_cutoff", f"{line}:{mid[:60]}"); continue
        cands = match_candidates_window(con, a, b, add_days(d, -WINDOW_BEFORE), add_days(d, WINDOW_AFTER))
        match_id, outcome = link_match(con, ov, SOURCE, mid, cands, rnd, snapshot_id, detail,
                                       src_date=d, src_tournament=row.get("Tournament"))
        if match_id is None:
            run.queued(); continue
        run.linked()
        st.linked[mid] = (match_id, a, b)
        if match_id not in st.charted_matches:
            st.charted_matches.add(match_id)
            st.primary.add(mid)
        ekey = f"{g}:{d[:4]}:{(row.get('Tournament') or '').strip()}"
        ed = con.execute("SELECT edition_id FROM matches WHERE match_id=?", (match_id,)).fetchone()[0]
        cur = con.execute(
            """INSERT OR IGNORE INTO edition_xref (source, source_key, edition_id, method, confidence,
                   verified, snapshot_id) VALUES (?,?,?,'context',0.95,0,?)""",
            (SOURCE, ekey, ed, snapshot_id))
        if cur.rowcount == 0:
            have = con.execute("SELECT edition_id FROM edition_xref WHERE source=? AND source_key=?",
                               (SOURCE, ekey)).fetchone()[0]
            if have != ed:
                st.edition_conflicts += 1
    run.finish()
    return run


def parse_point_codes(first: str, second: str, server_won: bool) -> dict:
    """Best-effort reading of MCP serve/rally notation (v0; raw codes are kept for re-parsing).

    The deciding code is the 2nd serve's if there was one. Rally length follows the
    convention 'including the serve, excluding errors'.
    """
    decisive = (second or first or "").strip()
    if not decisive:
        return dict(serve_no=None, is_ace=None, is_double_fault=None, rally_len=None, end_type=None)
    d = decisive.lstrip("c")                    # leading c = let
    shots = sum(ch in SHOT_LETTERS for ch in d[1:])
    is_ace = bool(re.fullmatch(r"[0-6][+;^]*\*", d))
    is_df = bool(second.strip()) and shots == 0 and not d.endswith(("*", "#")) and not server_won
    if is_ace:
        end, rally = "ace", 1
    elif is_df:
        end, rally = "double_fault", 0
    else:
        end = {"*": "winner", "#": "forced_error", "@": "unforced_error"}.get(d[-1:])
        rally = 1 + shots - (1 if d.endswith(("@", "#")) and shots > 0 else 0)
    return dict(serve_no=2 if second.strip() else 1, is_ace=int(is_ace), is_double_fault=int(is_df),
                rally_len=rally, end_type=end)


_POINT_SQL = """INSERT OR IGNORE INTO point (match_id, data_source, point_idx, set_idx, game_idx,
    server_id, returner_id, winner_id, serve_no, is_ace, is_double_fault, rally_len, end_type,
    is_tiebreak, is_break_point, score_before, raw_first, raw_second, snapshot_id)
    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"""


def flush_points(con, run: LoadRun, buf: list) -> None:
    """Insert buffered points; rows dropped by the UNIQUE key are counted as duplicates.
    Every CHECK is satisfied by construction, so OR IGNORE only drops duplicate point_idx."""
    if not buf:
        return
    before = con.total_changes
    con.executemany(_POINT_SQL, buf)
    ins = con.total_changes - before
    run.n_inserted += ins
    if len(buf) - ins:
        run.skipped("duplicate_point", n=len(buf) - ins)
    buf.clear()


def load_points(con: sqlite3.Connection, snapshot_id: int, rf: RawFile, st: McpState) -> LoadRun:
    run = LoadRun(con, snapshot_id, rf.loader)
    buf = []
    with open_csv(rf.path) as f:
        for line, row in enumerate(csv.DictReader(f), start=2):
            run.row()
            mid = (row.get("match_id") or "").strip()
            if mid not in st.known:
                run.skipped("match_row_skipped_or_missing", f"{line}:{mid[:60]}"); continue
            link = st.linked.get(mid)
            if link is None:
                run.skipped("match_not_linked", mid[:60]); continue
            if mid not in st.primary:
                run.skipped("duplicate_chart", mid[:60]); continue
            pt, svr, won = to_int(row.get("Pt")), (row.get("Svr") or "").strip(), (row.get("PtWinner") or "").strip()
            if pt is None or svr not in ("1", "2") or won not in ("1", "2"):
                run.skipped("malformed_row", line); continue
            match_id, p1, p2 = link
            server, returner = (p1, p2) if svr == "1" else (p2, p1)
            winner = p1 if won == "1" else p2
            first, second = row.get("1st") or "", row.get("2nd") or ""
            codes = parse_point_codes(first, second, svr == won)
            s1, s2, g1, g2 = (to_int(row.get(k)) for k in ("Set1", "Set2", "Gm1", "Gm2"))
            gm = re.match(r"\s*(\d+)", row.get("Gm#") or "")
            tbset = (row.get("TbSet") or "").strip().lower() in ("true", "1")
            is_tb = None if g1 is None or g2 is None else int(tbset and g1 == g2 and g1 >= 6)
            buf.append((match_id, SOURCE, pt,
                        (s1 + s2 + 1) if s1 is not None and s2 is not None else None,
                        int(gm.group(1)) if gm else None,
                        server, returner, winner, codes["serve_no"], codes["is_ace"],
                        codes["is_double_fault"], codes["rally_len"], codes["end_type"], is_tb, None,
                        (row.get("Pts") or "").strip() or None, first or None, second or None,
                        snapshot_id))
            if len(buf) >= 20000:
                flush_points(con, run, buf)
    flush_points(con, run, buf)
    run.finish()
    return run
