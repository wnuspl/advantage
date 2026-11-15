"""Which edition a match belongs to, from tennis.db (milestone A1; the Kalshi retriever reuses
shared_edition()). Gives MatchQuery its edition_id, surface and best-of.

    find_edition(con, "atp", "Paris Masters", "2026-11-05")
    shared_edition(con, a, b, "2026-11-12")
"""
from __future__ import annotations

from datetime import date, timedelta

from ..normalize import norm_name

RUNNING_DAYS = 21          # an edition counts as running up to 3 weeks after its start


def _day(d: str, n: int) -> str:
    return (date.fromisoformat(d) + timedelta(days=n)).isoformat()


def find_edition(con, tour: str | None, name: str, on: str, alias: str = "data") -> dict | None:
    """The tournament whose name (or an alias) contains `name` (normalized), and its edition
    on day `on`: one that has started by `on` and not ended before it (end_date, else start +
    21 days). With none running, the tournament's latest edition before `on` is returned with
    running=False (its surface and best-of are a guess for a new edition). Among several
    tournaments, the one with the most editions wins."""
    k = norm_name(name)
    if not k:
        return None
    cands = {}
    for tid, t, canon, n in con.execute(
            f"""SELECT t.tournament_id, t.tour, t.canonical_name,
                       (SELECT COUNT(*) FROM {alias}.tournament_edition e WHERE e.tournament_id = t.tournament_id)
                FROM {alias}.tournament t"""):
        if (tour is None or t == tour) and k in norm_name(canon):
            cands[tid] = (n, t, canon)
    for alias_n, tid in con.execute(f"SELECT alias_normalized, tournament_id FROM {alias}.tournament_alias"):
        if k in alias_n and tid not in cands:
            row = con.execute(
                f"""SELECT t.tour, t.canonical_name,
                           (SELECT COUNT(*) FROM {alias}.tournament_edition e WHERE e.tournament_id = t.tournament_id)
                    FROM {alias}.tournament t WHERE t.tournament_id = ?""", (tid,)).fetchone()
            if tour is None or row[0] == tour:
                cands[tid] = (row[2], row[0], row[1])
    if not cands:
        return None
    tid = max(cands, key=lambda t: (cands[t][0], -t))
    row = con.execute(
        f"""SELECT edition_id, start_date, end_date, surface, best_of, level FROM {alias}.tournament_edition
            WHERE tournament_id = ? AND start_date <= ? ORDER BY start_date DESC LIMIT 1""",
        (tid, on)).fetchone()
    if row is None:
        return None
    eid, start, end, surface, best_of, level = row
    running = on <= (end or _day(start, RUNNING_DAYS))
    return {"edition_id": eid if running else None, "latest_edition_id": eid, "tournament_id": tid,
            "tournament": cands[tid][2], "tour": cands[tid][1], "start": start, "end": end,
            "surface": surface, "best_of": best_of, "level": level, "running": running}


def shared_edition(con, a: int, b: int, day: str, alias: str = "data") -> tuple | None:
    """(edition_id, surface, best_of) of an edition both players have a match in that started
    in the 3 weeks before `day` and has not ended before the day before it (v1 kalshi_edges)."""
    return con.execute(
        f"""SELECT e.edition_id, e.surface, e.best_of FROM {alias}.tournament_edition e
            WHERE e.start_date <= ? AND e.start_date >= ? AND COALESCE(e.end_date, e.start_date) >= ?
              AND EXISTS (SELECT 1 FROM {alias}.matches m WHERE m.edition_id = e.edition_id AND ? IN (m.winner_id, m.loser_id))
              AND EXISTS (SELECT 1 FROM {alias}.matches m WHERE m.edition_id = e.edition_id AND ? IN (m.winner_id, m.loser_id))
            ORDER BY e.start_date DESC LIMIT 1""",
        (day, _day(day, -RUNNING_DAYS), _day(day, -1), a, b)).fetchone()
