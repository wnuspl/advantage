"""Elo with an overall rating and one rating per surface, walk-forward over the full history.

    P(a beats b) = 1 / (1 + 10 ** ((R_b - R_a) / 400))
    R = (1 - blend) * overall + blend * surface       (prediction only)

Updates (each rating uses its OWN expected score, so `blend` never changes the history):
    overall  += K(n_all)  * weight * (result - E_overall)      every rated match
    surface  += K(n_surf) * weight * (result - E_surface)      matches on that surface
    K(n) = k / (n + offset) ** shape     n = player's rated matches so far (overall, or on
                                         that surface); shape = 0 gives a constant K
A player's first rating on a surface starts at `init` (surface_init=0) or at their
overall rating at that moment (surface_init=1). Editions with no recorded surface update
only the overall rating and are predicted from it alone.

blend = 0 reproduces plain overall Elo exactly. Because blend is prediction-only, one run
can be scored at many blend values (predict()).

Pure logic: no database access except load_matches(). Every prediction is made from the
state strictly before the match, then the match updates the state.
"""
from __future__ import annotations

import sqlite3
from array import array
from dataclasses import asdict, dataclass, replace

from .matches import (SCORED_STATUS, TOUR_LEVELS, Matches,   # noqa: F401  (re-exported)
                      empty_matches, load_matches, until)

NAN = float("nan")


@dataclass(frozen=True)
class EloParams:
    k: float = 250.0              # K numerator
    offset: float = 5.0           # K denominator offset
    shape: float = 0.4            # K decay exponent; 0 = constant K
    init: float = 1500.0          # starting rating
    retired_weight: float = 1.0   # update weight for retirements; 0 = ignore them
    surface_init: float = 0.0     # first surface rating: 0 = init, 1 = current overall rating
    blend: float = 0.0            # prediction weight on the surface rating (0 = overall only)

    def as_dict(self) -> dict:
        return asdict(self)

    def update_key(self) -> "EloParams":
        """The parameters that shape the rating history (everything but blend)."""
        return replace(self, blend=0.0)



@dataclass
class EloResult:
    p_winner: array        # pre-match P(actual winner wins) at the run's blend
    r_winner: array        # pre-match overall ratings
    r_loser: array
    s_winner: array        # pre-match surface ratings (NaN when the edition has no surface)
    s_loser: array
    n_winner: array        # pre-match overall rated-match counts
    n_loser: array
    state: "EloState | None" = None   # ratings AFTER the last processed match


@dataclass
class EloState:
    rating: dict           # player_id -> overall rating
    count: dict            # player_id -> rated matches
    srating: dict          # (player_id, surface) -> surface rating
    scount: dict           # (player_id, surface) -> rated matches on that surface



def _expected(diff: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))


def predict(res: EloResult, blend: float) -> array:
    """P(actual winner wins) for every match at another blend weight, from stored ratings."""
    out = array("d", bytes(8 * len(res.r_winner)))
    for i in range(len(out)):
        d = res.r_winner[i] - res.r_loser[i]
        sw = res.s_winner[i]
        if blend and sw == sw:                 # sw == sw is False for NaN (no surface)
            d = (1.0 - blend) * d + blend * (sw - res.s_loser[i])
        out[i] = _expected(d)
    return out


def run(m: Matches, p: EloParams) -> EloResult:
    rating: dict[int, float] = {}
    count: dict[int, int] = {}
    srating: dict[tuple[int, str], float] = {}
    scount: dict[tuple[int, str], int] = {}
    n = len(m)
    ints = lambda: array("l", bytes(array("l").itemsize * n))   # noqa: E731
    floats = lambda: array("d", bytes(8 * n))                   # noqa: E731
    out = EloResult(floats(), floats(), floats(), floats(), floats(), ints(), ints())
    init, k, off, shape, rw, blend = p.init, p.k, p.offset, p.shape, p.retired_weight, p.blend
    from_overall = p.surface_init >= 0.5
    for i in range(n):
        w, l, surf = m.winner[i], m.loser[i], m.surface[i]
        rw_, rl = rating.get(w, init), rating.get(l, init)
        nw, nl = count.get(w, 0), count.get(l, 0)
        e = _expected(rw_ - rl)
        out.r_winner[i], out.r_loser[i] = rw_, rl
        out.n_winner[i], out.n_loser[i] = nw, nl
        if surf is not None:
            kw, kl = (w, surf), (l, surf)
            sw = srating.get(kw, rw_ if from_overall else init)
            sl = srating.get(kl, rl if from_overall else init)
            out.s_winner[i], out.s_loser[i] = sw, sl
            out.p_winner[i] = _expected((1.0 - blend) * (rw_ - rl) + blend * (sw - sl)) if blend else e
        else:
            out.s_winner[i] = out.s_loser[i] = NAN
            out.p_winner[i] = e
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
        if surf is not None:
            snw, snl = scount.get(kw, 0), scount.get(kl, 0)
            sdelta = 1.0 - _expected(sw - sl)
            srating[kw] = sw + weight * k / (snw + off) ** shape * sdelta
            srating[kl] = sl - weight * k / (snl + off) ** shape * sdelta
            scount[kw], scount[kl] = snw + 1, snl + 1
    out.state = EloState(rating, count, srating, scount)
    return out

