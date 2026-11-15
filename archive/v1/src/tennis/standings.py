"""Rating standings at a point in time (Elo or Glicko-2): replay a saved run's parameters
up to a date and list every player's ratings, highest first."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date, timedelta

from . import elo

SURFACES = ("hard", "clay", "grass")


@dataclass
class Standing:
    rank: int
    player_id: int
    name: str
    country: str | None
    tour: str
    score: float           # what the list is sorted by (overall, or blend on --surface)
    overall: float
    surface: dict          # surface -> rating (None if never played there)
    matches: int
    last_played: str
    deviation: float | None = None   # Glicko: overall RD grown by inactivity to the as-of date


def compute(con: sqlite3.Connection, m: elo.Matches, params, *, as_of: str | None,
            tour: str | None = None, surface: str | None = None, min_matches: int = 0,
            active_days: int | None = None, alias: str = "data",
            run=elo.run) -> tuple[list[Standing], str]:
    """Ratings after every match in editions starting strictly before `as_of`.

    Returns (standings sorted high to low, last edition start included).
    `surface` ranks by the run's blend of overall and that surface's rating, the number
    the model would use on that surface. Ratings of the two tours are on separate scales.
    `run` is the model's engine (elo.run or glicko2.run); for Glicko the deviation is
    reported as of `as_of` (grown by inactivity).
    """
    sub = elo.until(m, as_of) if as_of else m
    if not len(sub):
        return [], ""
    st = run(sub, params).state
    last: dict[int, str] = {}
    tour_of: dict[int, str] = {}
    for w, l, s, t in zip(sub.winner, sub.loser, sub.start, sub.tour):
        last[w] = last[l] = s
        tour_of[w] = tour_of[l] = t
    last_start = sub.start[-1]
    cutoff = None
    if active_days is not None:
        ref = date.fromisoformat(as_of) if as_of else date.fromisoformat(last_start)
        cutoff = (ref - timedelta(days=active_days)).isoformat()
    names = {pid: (n, c) for pid, n, c in con.execute(
        f"SELECT player_id, canonical_name, country FROM {alias}.player")}

    on = as_of or last_start
    has_rd = hasattr(st, "rd_at")
    rows = []
    for pid, overall in st.rating.items():
        if tour and tour_of[pid] != tour:
            continue
        if st.count.get(pid, 0) < min_matches:
            continue
        if cutoff and last[pid] < cutoff:
            continue
        surf = {s: st.srating.get((pid, s)) for s in SURFACES}
        if surface:
            sr = surf[surface]
            if sr is None:   # never played there: what the engine would start them at
                sr = overall if params.surface_init >= 0.5 else params.init
            score = (1.0 - params.blend) * overall + params.blend * sr
        else:
            score = overall
        name, country = names.get(pid, (f"player {pid}", None))
        rows.append(Standing(0, pid, name, country, tour_of[pid], score, overall, surf,
                             st.count.get(pid, 0), last[pid],
                             st.rd_at(pid, on) if has_rd else None))
    rows.sort(key=lambda r: (-r.score, r.name))
    for i, r in enumerate(rows, 1):
        r.rank = i
    return rows, last_start
