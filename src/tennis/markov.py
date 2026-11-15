"""Point -> game -> set -> match win probabilities (independent points, Markov chain).

pa = P(A wins a point on A's serve), pb = P(B wins a point on B's serve).

Every function uses only arithmetic on its inputs, so it works on Python floats and on
numpy arrays alike (element-wise). match() is exact; MatchGrid tabulates it over the
logit point probabilities for fast lookup (with gradients) inside rating updates.

Formats (`final` = the deciding set's rule):
    'tb'    tiebreak (first to 7) at 6-6, as in every other set
    'adv'   advantage set: no tiebreak, win by two games
    'tb12'  tiebreak at 12-12 (Wimbledon 2019-2021)
    'mtb'   a match tiebreak (first to 10, win by 2) instead of the deciding set
A 10-point tiebreak at 6-6 (slams from 2022) is modelled as 'tb': the results data
cannot tell 7- from 10-point tiebreaks apart, and the difference in match probability
is a few hundredths of a point.

Who serves first in a set does not change the set's win probability (unit-tested), so
sets are independent and identically distributed apart from the deciding set.
"""
from __future__ import annotations

import math

import numpy as np

FINALS = ("tb", "adv", "tb12", "mtb")


def game(p):
    """P(server holds), server wins each point with probability p."""
    q = 1 - p
    return p ** 4 * (1 + 4 * q + 10 * q * q) + 20 * p ** 3 * q ** 3 * (p * p / (p * p + q * q))


def tiebreak(pa, pb, n: int = 7):
    """P(A wins a first-to-n, win-by-2 tiebreak), A serving the first point.
    Serve order: A, B, B, A, A, B, B, ..."""
    qa, qb = 1 - pa, 1 - pb
    # dist[i][j] = P(score i-j), A on i
    dist = [[0.0] * n for _ in range(n)]
    dist[0][0] = 1.0 + 0 * pa                    # broadcast to pa's shape
    win = 0.0 * pa
    for k in range(2 * n - 2):                   # states with i + j = k, both < n
        a_serves = ((k + 1) // 2) % 2 == 0
        pw = pa if a_serves else qb              # P(A wins point k)
        for i in range(max(0, k - n + 1), min(k, n - 1) + 1):
            j = k - i
            d = dist[i][j]
            if i + 1 == n:
                if j <= n - 2:
                    win = win + d * pw
            else:
                dist[i + 1][j] = dist[i + 1][j] + d * pw
            if j + 1 < n:
                dist[i][j + 1] = dist[i][j + 1] + d * (1 - pw)
    # from (n-1, n-1): pairs of points, one served by each player
    w, l_ = pa * qb, qa * pb
    return win + dist[n - 1][n - 1] * w / (w + l_)


def set_(pa, pb, final: str = "tb"):
    """P(A wins a set), A serving the first game. final: 'tb' / 'adv' / 'tb12'."""
    ga, gb = game(pa), game(pb)
    cap = {"tb": 6, "adv": 5, "tb12": 12}[final]   # DP up to cap-cap, then closed form
    dist = [[0.0] * (cap + 1) for _ in range(cap + 1)]
    dist[0][0] = 1.0 + 0 * pa
    win = 0.0 * pa
    for k in range(2 * cap):
        a_serves = k % 2 == 0
        pw = ga if a_serves else 1 - gb         # P(A wins game k)
        for i in range(max(0, k - cap), min(k, cap) + 1):
            j = k - i
            if i > cap or j > cap:
                continue
            d = dist[i][j]
            # A wins the game
            ni, nj = i + 1, j
            if ni >= 6 and ni - nj >= 2:
                win = win + d * pw
            elif ni <= cap:
                dist[ni][nj] = dist[ni][nj] + d * pw
            # B wins the game
            ni, nj = i, j + 1
            if not (nj >= 6 and nj - ni >= 2) and nj <= cap:
                dist[ni][nj] = dist[ni][nj] + d * (1 - pw)
    at = dist[cap][cap]
    if final == "adv":
        # 5-5, A serves next: pairs of games (A serves, B serves) until someone wins both
        w, l_ = ga * (1 - gb), (1 - ga) * gb
        return win + at * w / (w + l_)
    # tiebreak at cap-cap; game 2*cap is even, so A serves first in it
    return win + at * tiebreak(pa, pb, 7)


def match(pa, pb, best_of: int = 3, final: str = "tb"):
    """P(A wins the match)."""
    s = set_(pa, pb, "tb")
    if final == "mtb":
        f = tiebreak(pa, pb, 10)
    elif final == "tb":
        f = s
    else:
        f = set_(pa, pb, final)
    t = 1 - s
    if best_of == 1:
        return f
    if best_of == 3:
        return s * s + 2 * s * t * f
    if best_of == 5:
        return s ** 3 * (1 + 3 * t) + 6 * s * s * t * t * f
    raise ValueError(f"best_of must be 1, 3 or 5, not {best_of!r}")


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


class MatchGrid:
    """match() tabulated over xa = logit(pa), xb = logit(pb) on [lo, hi]^2, step h.
    Lookups are bilinear; inputs outside the grid are clamped to its edge."""

    def __init__(self, best_of: int = 3, final: str = "tb", lo: float = -2.5, hi: float = 4.5,
                 h: float = 0.01):
        self.best_of, self.final, self.lo, self.h = best_of, final, lo, h
        self.n = int(round((hi - lo) / h)) + 1
        self.hi = lo + (self.n - 1) * h
        x = lo + h * np.arange(self.n)
        pa, pb = np.meshgrid(_sig(x), _sig(x), indexing="ij")
        self.table = np.asarray(match(pa, pb, best_of, final), dtype=float)
        self._flat = self.table.ravel().tolist()     # Python floats: faster scalar lookups

    def lookup(self, xa: float, xb: float):
        """(P, dP/dxa, dP/dxb) for scalars (gradient of the bilinear surface)."""
        lo, h, n = self.lo, self.h, self.n
        fa = (min(max(xa, lo), self.hi) - lo) / h
        fb = (min(max(xb, lo), self.hi) - lo) / h
        i = min(int(fa), n - 2)
        j = min(int(fb), n - 2)
        u, v = fa - i, fb - j
        t, k = self._flat, i * n + j
        p00, p01, p10, p11 = t[k], t[k + 1], t[k + n], t[k + n + 1]
        p = (p00 * (1 - u) + p10 * u) * (1 - v) + (p01 * (1 - u) + p11 * u) * v
        da = ((p10 - p00) * (1 - v) + (p11 - p01) * v) / h
        db = ((p01 - p00) * (1 - u) + (p11 - p10) * u) / h
        return p, da, db

    def lookup_array(self, xa, xb, grad: bool = False):
        """Vectorized lookup: P (and dP/dxa, dP/dxb when grad=True)."""
        lo, h, n = self.lo, self.h, self.n
        fa = (np.clip(xa, lo, self.hi) - lo) / h
        fb = (np.clip(xb, lo, self.hi) - lo) / h
        i = np.minimum(fa.astype(int), n - 2)
        j = np.minimum(fb.astype(int), n - 2)
        u, v = fa - i, fb - j
        t = self.table
        p00, p01, p10, p11 = t[i, j], t[i, j + 1], t[i + 1, j], t[i + 1, j + 1]
        p = (p00 * (1 - u) + p10 * u) * (1 - v) + (p01 * (1 - u) + p11 * u) * v
        if not grad:
            return p
        da = ((p10 - p00) * (1 - v) + (p11 - p01) * v) / h
        db = ((p01 - p00) * (1 - u) + (p11 - p10) * u) / h
        return p, da, db


_GRIDS: dict = {}


def grid(best_of: int, final: str) -> MatchGrid:
    """Cached MatchGrid per format."""
    key = (best_of, final)
    if key not in _GRIDS:
        _GRIDS[key] = MatchGrid(best_of, final)
    return _GRIDS[key]


def logit(p: float) -> float:
    return math.log(p / (1 - p))


# ---------------------------------------------------------------------------- set scores
# Final scores of one set, from A's side, A serving the first game. 14 categories:
#   0-4  A wins 6-0 .. 6-4      5  A wins 7-5      6  A wins 7-6 ('tb') / beyond 7-5 ('adv')
#   7-13 the same scores won by B.   mirror(c) = (c + 7) % 14 is the score seen from B's side.
# 'adv' covers advantage sets (and, approximately, tiebreak-at-12-12 sets): category 6 is
# every score past 6-6 (8-6, 9-7, ...). A tiebreak counts as one game for serve order.
SET_CATS = 14
CAT_GAMES_ODD = (0, 1, 0, 1, 0, 0, 1) * 2     # total games odd -> the other player serves next set
CAT_GAMES_ODD_ADV = (0, 1, 0, 1, 0, 0, 0) * 2  # long advantage sets end n-(n-2): even


def mirror(c: int) -> int:
    return (c + 7) % 14


def set_score_dist(pa, pb, final: str = "tb") -> list:
    """[P(category c)] for c in 0..13, A serving the first game (final: 'tb' or 'adv')."""
    ga, gb = game(pa), game(pb)
    dist = [[0.0] * 7 for _ in range(7)]
    dist[0][0] = 1.0 + 0 * pa
    out = [0.0 * pa] * SET_CATS
    for k in range(12):                         # states i + j = k, both <= 6
        pw = ga if k % 2 == 0 else 1 - gb       # P(A wins game k)
        for i in range(max(0, k - 6), min(k, 6) + 1):
            j = k - i
            d = dist[i][j]
            for a_won, (ni, nj) in ((True, (i + 1, j)), (False, (i, j + 1))):
                pr = d * (pw if a_won else 1 - pw)
                hi, lo = max(ni, nj), min(ni, nj)
                if hi >= 6 and hi - lo >= 2:     # set over: 6-0..6-4 or 7-5
                    c = lo if hi == 6 else 5
                    c = c if ni > nj else c + 7
                    out[c] = out[c] + pr
                else:
                    dist[ni][nj] = dist[ni][nj] + pr
    at = dist[6][6]                              # game 13 is A's: A serves first after 6-6
    if final == "tb":
        t = tiebreak(pa, pb, 7)
    else:
        w, l_ = ga * (1 - gb), (1 - ga) * gb
        t = w / (w + l_)
    out[6] = out[6] + at * t
    out[13] = out[13] + at * (1 - t)
    return out


class ScoreGrid:
    """set_score_dist tabulated on the logit grid (A serving first), plus the Fisher
    information of one set's score about (xa, xb), averaged over who serves first."""

    def __init__(self, final: str = "tb", lo: float = -2.5, hi: float = 4.5, h: float = 0.02):
        self.final, self.lo, self.h = final, lo, h
        self.n = n = int(round((hi - lo) / h)) + 1
        self.hi = lo + (n - 1) * h
        x = lo + h * np.arange(n)
        pa, pb = np.meshgrid(_sig(x), _sig(x), indexing="ij")
        t = np.stack([np.asarray(c, dtype=float) for c in set_score_dist(pa, pb, final)], axis=-1)
        self.table = np.clip(t, 1e-300, None)                      # [i, j, c]
        # average over the first server: B first = A-first table transposed, mirrored categories
        avg = 0.5 * (t + np.transpose(t, (1, 0, 2))[..., [mirror(c) for c in range(SET_CATS)]])
        da, db = np.gradient(avg, h, axis=(0, 1))
        q = np.clip(avg, 1e-12, None)
        self.info = np.stack([(da * da / q).sum(-1), (da * db / q).sum(-1), (db * db / q).sum(-1)], -1)
        self._flat = self.table.ravel().tolist()
        self._iflat = self.info.ravel().tolist()

    def _cell(self, xa, xb):
        lo, h, n = self.lo, self.h, self.n
        fa = (min(max(xa, lo), self.hi) - lo) / h
        fb = (min(max(xb, lo), self.hi) - lo) / h
        i = min(int(fa), n - 2)
        j = min(int(fb), n - 2)
        return i, j, fa - i, fb - j

    def lookup(self, xa: float, xb: float, c: int):
        """(P(category c), d/dxa, d/dxb), A serving first (bilinear)."""
        i, j, u, v = self._cell(xa, xb)
        n, h, t, C = self.n, self.h, self._flat, SET_CATS
        k = (i * n + j) * C + c
        p00, p01, p10, p11 = t[k], t[k + C], t[k + n * C], t[k + n * C + C]
        p = (p00 * (1 - u) + p10 * u) * (1 - v) + (p01 * (1 - u) + p11 * u) * v
        da = ((p10 - p00) * (1 - v) + (p11 - p01) * v) / h
        db = ((p01 - p00) * (1 - u) + (p11 - p10) * u) / h
        return p, da, db

    def information(self, xa: float, xb: float):
        """(I_aa, I_ab, I_bb) of one set's score (bilinear)."""
        i, j, u, v = self._cell(xa, xb)
        n, t = self.n, self._iflat
        out = []
        for e in range(3):
            k = (i * n + j) * 3 + e
            p00, p01, p10, p11 = t[k], t[k + 3], t[k + n * 3], t[k + n * 3 + 3]
            out.append((p00 * (1 - u) + p10 * u) * (1 - v) + (p01 * (1 - u) + p11 * u) * v)
        return out


_SCORE_GRIDS: dict = {}


def score_grid(final: str) -> ScoreGrid:
    if final not in _SCORE_GRIDS:
        _SCORE_GRIDS[final] = ScoreGrid(final)
    return _SCORE_GRIDS[final]
