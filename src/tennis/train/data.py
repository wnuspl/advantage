"""The prepared match table for training: matches in walk-forward order with the points
extras attached, the canonical (non-winner-first) player order, and per-match best-of."""
from __future__ import annotations

from dataclasses import dataclass, field

from .. import matches, models_db, points_rating

BLOCKS = {"val": ("2019-01-01", "2021-12-31"), "test": ("2022-01-01", "2026-12-31")}


@dataclass
class Prepared:
    m: matches.Matches
    winner_is_p1: bytearray          # 1 if the winner has the smaller stable player key
    player_key: dict                 # player_id -> stable key ({} in tests that never save)
    best_of: list = field(default_factory=list)   # per match, from the points extras

    def indices(self, date_from: str, date_to: str, scope: str) -> list[int]:
        return matches.scope_indices(self.m, date_from, date_to, scope)


def best_of_of(m) -> list[int]:
    fmt = m.extra["points"].fmt
    return [points_rating.FORMATS[fmt[i]][0] for i in range(len(m))]


def from_matches(m, player_key: dict | None = None) -> Prepared:
    """For a table that already has m.extra['points'] (and 'context'). Without player keys
    the canonical order falls back to the smaller player_id."""
    pk = player_key or {}
    if pk:
        flag = bytearray(1 if pk[w] < pk[l] else 0 for w, l in zip(m.winner, m.loser))
    else:
        flag = bytearray(1 if w < l else 0 for w, l in zip(m.winner, m.loser))
    return Prepared(m, flag, pk, best_of_of(m))


def prepare(con, alias: str = "data") -> Prepared:
    """Load every match from tennis.db (attached as `alias`) with serve stats, scores, rules
    and context features attached."""
    m = matches.load_matches(con, alias)
    points_rating.attach(con, m, alias)
    return from_matches(m, models_db.player_keys(con, alias))
