"""Plain ('naive') Elo: one overall rating per player, walk-forward over the full history.

    P(a beats b) = 1 / (1 + 10 ** ((r_b - r_a) / 400))
    after a match: r += K(n) * weight * (result - expected)
    K(n) = k / (n + offset) ** shape        n = player's rated matches so far
                                            shape = 0 gives a constant K

Pure logic: no database access except load_matches(). Every prediction is made from the
state strictly before the match, then the match updates the state.
"""
from __future__ import annotations

import sqlite3
from array import array
from dataclasses import asdict, dataclass

# Tour-level main draws (verified against the level codes in tennis.db, 2026-11-13).
TOUR_LEVELS = {"atp": {"G", "M", "A", "F", "O"},
               "wta": {"G", "PM", "P", "I", "W", "T1", "T2", "T3", "T4", "T5", "F", "O"}}
# Not scored: no tennis played (walkover), or result not a contest outcome (default, unfinished).
SCORED_STATUS = {"completed", "retired"}


@dataclass(frozen=True)
class EloParams:
    k: float = 250.0              # K numerator
    offset: float = 5.0           # K denominator offset
    shape: float = 0.4            # K decay exponent; 0 = constant K
    init: float = 1500.0          # starting rating
    retired_weight: float = 1.0   # update weight for retirements; 0 = ignore them

    def as_dict(self) -> dict:
        return asdict(self)


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

    def __len__(self) -> int:
        return len(self.match_id)


def load_matches(con: sqlite3.Connection, alias: str = "data") -> Matches:
    """All backbone matches in walk-forward order. Same-week matches are only ordered by
    round (Sackmann has no match day), then match_id for determinism."""
    m = Matches(array("q"), array("q"), array("q"), [], [], bytearray(), [])
    for mid, w, l, start, status, level, rnd, tour in con.execute(
            f"""SELECT m.match_id, m.winner_id, m.loser_id, e.start_date, m.status, e.level,
                       m.round, t.tour
                FROM {alias}.matches m JOIN {alias}.tournament_edition e USING (edition_id)
                JOIN {alias}.tournament t USING (tournament_id)
                ORDER BY e.start_date, m.round_order, m.match_id"""):
        m.match_id.append(mid); m.winner.append(w); m.loser.append(l)
        m.start.append(start); m.status.append(status); m.tour.append(tour)
        m.tour_level.append(1 if level in TOUR_LEVELS[tour] and not rnd.startswith("Q") else 0)
    return m


@dataclass
class EloResult:
    p_winner: array        # pre-match P(actual winner wins), per match
    r_winner: array        # pre-match ratings
    r_loser: array
    n_winner: array        # pre-match rated-match counts
    n_loser: array


def run(m: Matches, p: EloParams) -> EloResult:
    rating: dict[int, float] = {}
    count: dict[int, int] = {}
    n = len(m)
    out = EloResult(array("d", bytes(8 * n)), array("d", bytes(8 * n)), array("d", bytes(8 * n)),
                    array("l", bytes(array("l").itemsize * n)), array("l", bytes(array("l").itemsize * n)))
    init, k, off, shape, rw = p.init, p.k, p.offset, p.shape, p.retired_weight
    for i in range(n):
        w, l = m.winner[i], m.loser[i]
        rw_, rl = rating.get(w, init), rating.get(l, init)
        nw, nl = count.get(w, 0), count.get(l, 0)
        e = 1.0 / (1.0 + 10.0 ** ((rl - rw_) / 400.0))
        out.p_winner[i] = e
        out.r_winner[i], out.r_loser[i] = rw_, rl
        out.n_winner[i], out.n_loser[i] = nw, nl
        status = m.status[i]
        if status == "walkover" or status == "unfinished":
            continue
        weight = rw if status == "retired" else 1.0
        if weight <= 0.0:
            continue
        delta = 1.0 - e
        rating[w] = rw_ + weight * k / (nw + off) ** shape * delta
        rating[l] = rl - weight * k / (nl + off) ** shape * delta
        count[w], count[l] = nw + 1, nl + 1
    return out
