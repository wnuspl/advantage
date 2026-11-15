"""The match table every rating model replays: all backbone matches as column arrays in
walk-forward order, plus the scoring scopes (tour level, scored status).

Pure data access; no model logic. (Extracted from v1 elo.py at the reboot.)
"""
from __future__ import annotations

import sqlite3
from array import array
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field

# Tour-level main draws (verified against the level codes in tennis.db, 2026-11-13).
# AcesAPI events with no Sackmann history get level 'X' (tier unknown). They are rated but
# not scored as tour level: many are WTA 125s, which Sackmann files as 'C'.
TOUR_LEVELS = {"atp": {"G", "M", "A", "F", "O"},
               "wta": {"G", "PM", "P", "I", "W", "T1", "T2", "T3", "T4", "T5", "F", "O"}}
# Not scored: no tennis played (walkover), or result not a contest outcome (default, unfinished).
SCORED_STATUS = {"completed", "retired"}


@dataclass
class Matches:
    """Column arrays in processing order (edition start, round order, match_id)."""
    match_id: array
    winner: array
    loser: array
    start: list            # edition start date, ISO
    status: list
    tour_level: bytearray  # 1 if tour-level main draw (the 'tour' scoring scope)
    tour: list             # 'atp' / 'wta'
    surface: list          # 'hard' / 'clay' / 'grass' / 'carpet' / None
    extra: dict = field(default_factory=dict)   # model-specific per-match arrays, aligned
                                                # (e.g. points_rating.attach); prefixes stay valid

    def __len__(self) -> int:
        return len(self.match_id)


def empty_matches() -> Matches:
    return Matches(array("q"), array("q"), array("q"), [], [], bytearray(), [], [])


def load_matches(con: sqlite3.Connection, alias: str = "data") -> Matches:
    """All backbone matches in walk-forward order. Same-week matches are only ordered by
    round (Sackmann has no match day), then match_id for determinism."""
    m = empty_matches()
    for mid, w, l, start, status, level, rnd, tour, surface in con.execute(
            f"""SELECT m.match_id, m.winner_id, m.loser_id, e.start_date, m.status, e.level,
                       m.round, t.tour, e.surface
                FROM {alias}.matches m JOIN {alias}.tournament_edition e USING (edition_id)
                JOIN {alias}.tournament t USING (tournament_id)
                ORDER BY e.start_date, m.round_order, m.match_id"""):
        m.match_id.append(mid); m.winner.append(w); m.loser.append(l)
        m.start.append(start); m.status.append(status); m.tour.append(tour)
        m.surface.append(surface)
        m.tour_level.append(1 if level in TOUR_LEVELS[tour] and not rnd.startswith("Q") else 0)
    return m


SCOPES = ("tour", "tour_atp", "tour_wta", "all")


def scope_indices(m: Matches, date_from: str, date_to: str, scope: str) -> list[int]:
    """Indices of scored matches (SCORED_STATUS) whose edition starts in [date_from, date_to]
    (ISO, inclusive) and that fall in `scope`: 'tour' (tour-level main draw), 'tour_atp',
    'tour_wta' or 'all'. m.start is sorted, so the date window is two bisections."""
    if scope not in SCOPES:
        raise ValueError(f"unknown scope {scope!r}; one of {SCOPES}")
    lo, hi = bisect_left(m.start, date_from), bisect_right(m.start, date_to)
    tour = scope[5:] if scope.startswith("tour_") else None
    st, tl, tr = m.status, m.tour_level, m.tour
    return [i for i in range(lo, hi) if st[i] in SCORED_STATUS
            and (scope == "all" or (tl[i] and (tour is None or tr[i] == tour)))]


def until(m: Matches, as_of: str) -> Matches:
    """The prefix of `m` whose edition started strictly before `as_of` (ISO date).
    Sackmann dates matches only by their event's start, so an event already under way on
    `as_of` counts in full."""
    cut = bisect_left(m.start, as_of)
    return Matches(m.match_id[:cut], m.winner[:cut], m.loser[:cut], m.start[:cut],
                   m.status[:cut], m.tour_level[:cut], m.tour[:cut], m.surface[:cut], m.extra)
