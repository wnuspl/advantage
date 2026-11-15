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


# ---------------------------------------------------------------------------- scorelines
_A_WINS = ("6-0", "6-1", "6-2", "6-3", "6-4", "7-5", "7-6")


def _label(c: int, adv: bool) -> str:
    """Set score of category c from A's side ('6-3', '4-6', '8-6+' for long advantage sets)."""
    s = _A_WINS[c % 7]
    if adv and c % 7 == 6:
        s = "8-6+"
    if c >= 7:
        a, b = s.rstrip("+").split("-")
        s = f"{b}-{a}" + ("+" if s.endswith("+") else "")
    return s


def match_score_dist(pa: float, pb: float, best_of: int = 3, final: str = "tb",
                     min_prob: float = 1e-7) -> dict:
    """{'6-4 3-6 7-6': probability, ...}: every match scoreline from A's side, A or B
    serving first with equal probability (later sets follow from game parity). The
    deciding set follows `final` ('adv'/'tb12' sets past 6-6 are pooled as '8-6+'; a match
    tiebreak shows as '[10-x]' / '[x-10]'). Paths below min_prob are dropped (best of 5
    has ~100k sequences), so the total is just under 1."""
    need = best_of // 2 + 1
    dists = {"tb": (set_score_dist(pa, pb, "tb"), set_score_dist(pb, pa, "tb"))}
    if final in ("adv", "tb12"):
        dists["adv"] = (set_score_dist(pa, pb, "adv"), set_score_dist(pb, pa, "adv"))
    out: dict = {}

    def go(sa, sb, a_first, prob, sets):
        if sa == need or sb == need:
            key = " ".join(sets)
            out[key] = out.get(key, 0.0) + prob
            return
        deciding = sa == sb == need - 1
        if deciding and final == "mtb":
            t = tiebreak(pa, pb, 10)
            go(sa + 1, sb, a_first, prob * t, sets + ["[10-x]"])
            go(sa, sb + 1, a_first, prob * (1 - t), sets + ["[x-10]"])
            return
        kind = "adv" if deciding and final in ("adv", "tb12") else "tb"
        a_dist, b_dist = dists[kind]
        odd = CAT_GAMES_ODD_ADV if kind == "adv" else CAT_GAMES_ODD
        for c in range(SET_CATS):
            pc = a_dist[c] if a_first else b_dist[mirror(c)]   # B first: B's table, B's side
            pr = prob * pc
            if pr < min_prob:
                continue
            nxt = (not a_first) if odd[c] else a_first
            go(sa + (c < 7), sb + (c >= 7), nxt, pr, sets + [_label(c, kind == "adv")])

    go(0, 0, True, 0.5, [])
    go(0, 0, False, 0.5, [])
    return out


def set_count_dist(pa: float, pb: float, best_of: int = 3, final: str = "tb") -> dict:
    """{'2-0': p, '2-1': p, ...} exactly (no pruning), from A's side."""
    need = best_of // 2 + 1
    dists = {"tb": (set_score_dist(pa, pb, "tb"), set_score_dist(pb, pa, "tb")),
             "adv": (set_score_dist(pa, pb, "adv"), set_score_dist(pb, pa, "adv"))}
    out: dict = {}

    def go(sa, sb, a_first, prob):
        if sa == need or sb == need:
            k = f"{sa}-{sb}"
            out[k] = out.get(k, 0.0) + prob
            return
        deciding = sa == sb == need - 1
        if deciding and final == "mtb":
            t = tiebreak(pa, pb, 10)
            go(sa + 1, sb, a_first, prob * t)
            go(sa, sb + 1, a_first, prob * (1 - t))
            return
        kind = "adv" if deciding and final in ("adv", "tb12") else "tb"
        a_dist, b_dist = dists[kind]
        odd = CAT_GAMES_ODD_ADV if kind == "adv" else CAT_GAMES_ODD
        # pool categories by (A won the set, parity): 4 branches instead of 14
        pool = {}
        for c in range(SET_CATS):
            pc = a_dist[c] if a_first else b_dist[mirror(c)]
            key = (c < 7, odd[c])
            pool[key] = pool.get(key, 0.0) + pc
        for (a_won, flip), pc in pool.items():
            go(sa + a_won, sb + (not a_won), (not a_first) if flip else a_first, prob * pc)

    go(0, 0, True, 0.5)
    go(0, 0, False, 0.5)
    return dict(sorted(out.items(), key=lambda kv: (-int(kv[0].split("-")[0]), kv[0])))


def top_scores(dist: dict, top: int = 5) -> list:
    """The `top` most likely scorelines of match_score_dist, most likely first."""
    return sorted(dist.items(), key=lambda kv: -kv[1])[:top]


# ---------------------------------------------------------------------------- in play
# P(A wins) from any score, same independent-points model. Scalars only (pure Python).
# `a_serving` is who serves the NEXT point (inside a tiebreak the server changes every
# two points, so it is the next point's server, not the server of the set's last game).

TENNIS_POINTS = {"0": 0, "15": 1, "30": 2, "40": 3, "AD": 4, "A": 4}


def game_from(p: float, i: int, j: int) -> float:
    """P(server wins the game) from i-j in points (server's first; 0,1,2,3 = 0,15,30,40;
    past deuce only the difference matters)."""
    if i >= 4 and i - j >= 2:
        return 1.0
    if j >= 4 and j - i >= 2:
        return 0.0
    if i >= 3 and j >= 3:
        d = p * p / (p * p + (1 - p) ** 2)              # P(win from deuce)
        if i == j:
            return d
        return p + (1 - p) * d if i > j else p * d
    return p * game_from(p, i + 1, j) + (1 - p) * game_from(p, i, j + 1)


def tiebreak_from(pa: float, pb: float, i: int, j: int, a_serving: bool, n: int = 7) -> float:
    """P(A wins a first-to-n, win-by-2 tiebreak) from points i-j (A's first), A serving
    the next point if a_serving. Serve order ABBAABB... from the tiebreak's first server."""
    k = i + j
    a_first = a_serving == (((k + 1) // 2) % 2 == 0)    # who served point 0
    w, l_ = pa * (1 - pb), (1 - pa) * pb
    memo: dict = {}

    def f(i, j):
        if i >= n and i - j >= 2:
            return 1.0
        if j >= n and j - i >= 2:
            return 0.0
        if i == j and i >= n - 1:                        # even point count: the next two
            return w / (w + l_)                          # points are served one each
        key = (i, j)
        if key not in memo:
            a_srv = ((((i + j + 1) // 2) % 2) == 0) == a_first
            pw = pa if a_srv else 1 - pb
            memo[key] = pw * f(i + 1, j) + (1 - pw) * f(i, j + 1)
        return memo[key]

    return f(i, j)


def set_from(pa: float, pb: float, gi: int, gj: int, a_serving: bool, final: str = "tb",
             pi: int = 0, pj: int = 0, tb_points: int = 7) -> float:
    """P(A wins the set) from games gi-gj (A's first) and points pi-pj (A's first) in the
    current game or tiebreak; a_serving = A serves the next point. final: 'tb' (tiebreak
    at 6-6), 'tb12' (at 12-12), 'adv' (none). tb_points: the set tiebreak's length."""
    ga, gb = game(pa), game(pb)
    at = {"tb": 6, "tb12": 12, "adv": None}[final]
    memo: dict = {}

    def boundary(gi, gj, a_srv):                         # start of a game
        if (gi >= 6 and gi - gj >= 2) or (at and gi == at + 1):
            return 1.0
        if (gj >= 6 and gj - gi >= 2) or (at and gj == at + 1):
            return 0.0
        if at and gi == gj == at:
            return tiebreak_from(pa, pb, 0, 0, a_srv, tb_points)
        if at is None and gi == gj and gi >= 5:          # pairs of games, one served by each
            w, l_ = ga * (1 - gb), (1 - ga) * gb
            return w / (w + l_)
        key = (gi, gj, a_srv)
        if key not in memo:
            pw = ga if a_srv else 1 - gb                 # P(A wins this game)
            memo[key] = pw * boundary(gi + 1, gj, not a_srv) + (1 - pw) * boundary(gi, gj + 1, not a_srv)
        return memo[key]

    if at and gi == gj == at:
        return tiebreak_from(pa, pb, pi, pj, a_serving, tb_points)
    if pi == pj == 0:
        return boundary(gi, gj, a_serving)
    h = game_from(pa, pi, pj) if a_serving else 1 - game_from(pb, pj, pi)   # P(A wins game)
    return h * boundary(gi + 1, gj, not a_serving) + (1 - h) * boundary(gi, gj + 1, not a_serving)


def match_from(pa: float, pb: float, best_of: int, final: str, sets_a: int, sets_b: int,
               games_a: int = 0, games_b: int = 0, pts_a: int = 0, pts_b: int = 0,
               a_serving: bool = True, deciding_tb_points: int = 7) -> float:
    """P(A wins the match) from any score. Completed sets matter only through their
    count; who serves first in a set does not change its win probability, so later sets
    are identically distributed (the deciding set by `final`). In a match-tiebreak
    deciding set ('mtb') the score is pts_a-pts_b and the games are ignored.
    deciding_tb_points: 10 for the slams' 10-point tiebreak at 6-6 in the deciding set
    (pre-match it is modelled as 7: see the module docstring)."""
    need = best_of // 2 + 1
    if max(sets_a, sets_b) >= need:
        return 1.0 if sets_a >= need else 0.0
    memo: dict = {}

    def regular():
        if "s" not in memo:
            memo["s"] = set_(pa, pb, "tb")
        return memo["s"]

    def decider():              # a future deciding set: its own rule and tiebreak length
        if "f" not in memo:
            if final == "mtb":
                memo["f"] = tiebreak(pa, pb, 10)
            elif final == "tb" and deciding_tb_points == 7:
                memo["f"] = regular()
            else:
                memo["f"] = set_from(pa, pb, 0, 0, True, final, tb_points=deciding_tb_points)
        return memo["f"]

    def rest(sa, sb):
        if sa == need:
            return 1.0
        if sb == need:
            return 0.0
        if (sa, sb) not in memo:
            q = decider() if sa == sb == need - 1 else regular()
            memo[(sa, sb)] = q * rest(sa + 1, sb) + (1 - q) * rest(sa, sb + 1)
        return memo[(sa, sb)]

    deciding = sets_a == sets_b == need - 1
    if deciding and final == "mtb":
        q = tiebreak_from(pa, pb, pts_a, pts_b, a_serving, 10)
    elif deciding:
        q = set_from(pa, pb, games_a, games_b, a_serving, final, pts_a, pts_b, deciding_tb_points)
    else:
        q = set_from(pa, pb, games_a, games_b, a_serving, "tb", pts_a, pts_b)
    return q * rest(sets_a + 1, sets_b) + (1 - q) * rest(sets_a, sets_b + 1)


def set_over(a: int, b: int, final: str = "tb") -> bool:
    """Is a set (games, either order) finished? 'tb': 6-x with a 2-game lead, 7-5, 7-6;
    'tb12': also 13-12; 'adv': a 2-game lead from 6 up."""
    hi, lo = max(a, b), min(a, b)
    if final == "tb" and hi == 7 and lo in (5, 6):
        return True
    if final == "tb12" and hi == 13 and lo == 12:
        return True
    return hi >= 6 and hi - lo >= 2


def parse_live(score: str, points: str = "", best_of: int = 3, final: str = "tb") -> dict:
    """'6-4 3-6 2-1' (A's games first; the last set may be in progress) and '30-15' (A's
    points first: 0/15/30/40/AD; plain numbers inside a tiebreak or match tiebreak) ->
    match_from() keyword arguments. A match-tiebreak deciding set is given by points
    only: score '6-4 4-6', points '5-3'."""
    need = best_of // 2 + 1
    sa = sb = 0
    games = (0, 0)
    toks = [t.strip("[]()") for t in score.split()]
    for k, t in enumerate(toks):
        a, b = (int(x) for x in t.split("-"))
        deciding = sa == sb == need - 1
        if deciding and final == "mtb":
            raise ValueError("deciding match tiebreak: give its score as points, not a set")
        if set_over(a, b, final if deciding else "tb"):
            sa, sb = sa + (a > b), sb + (b > a)
            if max(sa, sb) >= need:
                raise ValueError(f"the match is already over ({sa}-{sb} in sets)")
            continue
        if k != len(toks) - 1:
            raise ValueError(f"set {t!r} is unfinished but is not the last one")
        games = (a, b)
    deciding = sa == sb == need - 1
    tb_at = None if deciding and final == "adv" else 12 if deciding and final == "tb12" else 6
    in_tb = (deciding and final == "mtb") or (tb_at is not None and games == (tb_at, tb_at))
    pa_ = pb_ = 0
    if points:
        x, y = points.upper().split("-")
        pa_, pb_ = (int(x), int(y)) if in_tb else (TENNIS_POINTS[x], TENNIS_POINTS[y])
    return {"sets_a": sa, "sets_b": sb, "games_a": games[0], "games_b": games[1],
            "pts_a": pa_, "pts_b": pb_}
