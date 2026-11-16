"""Reboot item R3: serve form vs return form in play.

Model. In this match A's serve-point logit is x_A + e_sA - e_rB and B's is x_B + e_sB - e_rA,
with e_s ~ N(0, f_s) and e_r ~ N(0, f_r) per player and match. With the four terms
independent, A's direction carries variance f_s + f_r and B's the same, and the two
directions are independent: the posterior predictive of the points still to play depends
on f_s and f_r only through their sum. A within-player link (cov(e_sA, e_rA) != 0) shows
up only as a correlation between the two directions, which is A4's `form_corr`
(in_match_form2; measured, no gain). So the split f_s / f_r is not identified by point
data, in one match or across many, as long as every player has the same f_s, f_r.

What IS identified across matches is per-player heterogeneity: if player i's own serve
varies more from day to day (f_s a_i) or the serve of whoever faces i does (f_r b_i),
the direction variance V(i serving, j returning) = f_s a_i + f_r b_j differs by player
pair. This module estimates it walk-forward from every match with serve stats:

    u = z^2 - 1/(n p q) - v,   z = (k/n - p) / (p q),  p = sigmoid(x)

is an unbiased (delta-method) estimate of one direction's form variance in one match
(k of n serve points won, x / v the pre-match logit and its variance). Per player, the
excess of u over its group's running mean (tour x tour-level) is summed on own serve
(S_s, n_s) and on return (S_r, n_r), from editions that started strictly before the
match's; the shrunk deviations are D_s = S_s / (n_s + kappa_s), D_r = S_r / (n_r + kappa_r),
and the direction's form variance is f * max(floor, 1 + (D_s(server) + D_r(returner)) / F_g),
F_g the group's running mean u. kappa = None turns a side off.

Group form variances (per tour, surface, best-of) are SplitForm.by_group.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .. import points_rating

MIN_SP = 30                     # serve points needed for a match's u


def _u(x: float, v: float, k: int, n: int) -> float:
    p = 1 / (1 + math.exp(-x))
    q = p * (1 - p)
    z = (k / n - p) / q
    return z * z - 1 / (n * q) - v


def match_u(prep, pm) -> tuple[np.ndarray, np.ndarray]:
    """(u of the winner's serve direction, u of the loser's) per match; nan without stats."""
    m, d = prep.m, prep.m.extra["points"]
    n = len(m)
    uw, ul = np.full(n, np.nan), np.full(n, np.nan)
    for i in range(n):
        if m.status[i] != "completed":
            continue
        if d.w_sp[i] >= MIN_SP:
            uw[i] = _u(float(pm["xw"][i]), float(pm["vw"][i]), d.w_sw[i], d.w_sp[i])
        if d.l_sp[i] >= MIN_SP:
            ul[i] = _u(float(pm["xl"][i]), float(pm["vl"][i]), d.l_sw[i], d.l_sp[i])
    return uw, ul


def group_of(m, i: int) -> tuple:
    return (m.tour[i], m.tour_level[i])


def walk(prep, uw, ul, want: set | None = None, on_match=None) -> dict:
    """Walk-forward per-player sums. For each match index i (all, or those whose match_id
    is in `want`) records {winner: (S_s, n_s, S_r, n_r), loser: (...), "F": group mean u}
    from editions that started strictly before match i's. on_match(i, state_before, F_g)
    is called for every match with usable u (for diagnostics)."""
    m = prep.m
    S_s, n_s, S_r, n_r = defaultdict(float), defaultdict(int), defaultdict(float), defaultdict(int)
    g_sum, g_n = defaultdict(float), defaultdict(int)
    pending = []
    out = {}
    cur = None
    for i in range(len(m)):
        if m.start[i] != cur:                       # flush earlier editions
            cur = m.start[i]
            for j in pending:
                g = group_of(m, j)
                for srv, ret, u in ((m.winner[j], m.loser[j], uw[j]), (m.loser[j], m.winner[j], ul[j])):
                    if np.isnan(u):
                        continue
                    g_sum[g] += u
                    g_n[g] += 1
            for j in pending:                       # deviations from the group mean of their time
                g = group_of(m, j)
                F = g_sum[g] / g_n[g] if g_n[g] else 0.0
                for srv, ret, u in ((m.winner[j], m.loser[j], uw[j]), (m.loser[j], m.winner[j], ul[j])):
                    if np.isnan(u):
                        continue
                    S_s[srv] += u - F
                    n_s[srv] += 1
                    S_r[ret] += u - F
                    n_r[ret] += 1
            pending = []
        g = group_of(m, i)
        F = g_sum[g] / g_n[g] if g_n[g] else float("nan")
        if want is None or m.match_id[i] in want:
            out[m.match_id[i]] = {"F": F, **{p: (S_s[p], n_s[p], S_r[p], n_r[p]) for p in (m.winner[i], m.loser[i])}}
        if on_match is not None:
            on_match(i, (S_s, n_s, S_r, n_r), F)
        pending.append(i)
    return out


def predictive_fit(prep, uw, ul, lo: str, hi: str, kappas: list, tour_level_only: bool = True) -> list:
    """Cross-match check of per-player volatility: regress each direction's u - F_g in
    matches with edition start in [lo, hi] on the walk-forward shrunk D_s(server) and
    D_r(returner) (no intercept; the group mean is already removed). Slopes near 1 mean the
    shrinkage is right; R^2 is the share of the noisy u's variance explained.
    Returns [{kappa, n, b_s, se_s, b_r, se_r, r2, r2_s, r2_r}]."""
    m = prep.m
    rows = {k: [] for k in kappas}

    def on(i, st, F):
        if not (lo <= m.start[i] <= hi) or np.isnan(F) or (tour_level_only and not m.tour_level[i]):
            return
        S_s, n_s, S_r, n_r = st
        for srv, ret, u in ((m.winner[i], m.loser[i], uw[i]), (m.loser[i], m.winner[i], ul[i])):
            if np.isnan(u):
                continue
            for k in kappas:
                rows[k].append((u - F, S_s.get(srv, 0.0) / (n_s.get(srv, 0) + k),
                                S_r.get(ret, 0.0) / (n_r.get(ret, 0) + k)))

    walk(prep, uw, ul, want=set(), on_match=on)
    out = []
    for k in kappas:
        a = np.array(rows[k])
        y, X = a[:, 0], a[:, 1:]
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ beta
        cov = np.linalg.inv(X.T @ X) * res.var()
        tot = (y ** 2).sum()
        r2 = 1 - (res ** 2).sum() / tot
        r2_one = []
        for c in range(2):
            b = (X[:, c] @ y) / (X[:, c] @ X[:, c])
            r2_one.append(1 - ((y - b * X[:, c]) ** 2).sum() / tot)
        out.append({"kappa": k, "n": len(y), "b_s": float(beta[0]), "se_s": float(math.sqrt(cov[0, 0])),
                    "b_r": float(beta[1]), "se_r": float(math.sqrt(cov[1, 1])), "r2": float(r2),
                    "r2_s": float(r2_one[0]), "r2_r": float(r2_one[1]),
                    "sd_s": float(X[:, 0].std()), "sd_r": float(X[:, 1].std())})
    return out


def split_half(prep, uw, ul, before: str, min_matches: int = 40) -> dict:
    """Split-half reliability of per-player mean (u - F_g) on serve and on return, from
    matches with edition start before `before`: correlation across players of the mean over
    their odd vs even matches (and Spearman-Brown for the full set). F_g = the group's mean
    over the same matches."""
    m = prep.m
    g_sum, g_n = defaultdict(float), defaultdict(int)
    for i in range(len(m)):
        if m.start[i] >= before:
            break
        for u in (uw[i], ul[i]):
            if not np.isnan(u):
                g_sum[group_of(m, i)] += u
                g_n[group_of(m, i)] += 1
    by = {"serve": defaultdict(list), "return": defaultdict(list)}
    for i in range(len(m)):
        if m.start[i] >= before:
            break
        g = group_of(m, i)
        F = g_sum[g] / g_n[g]
        for srv, ret, u in ((m.winner[i], m.loser[i], uw[i]), (m.loser[i], m.winner[i], ul[i])):
            if np.isnan(u):
                continue
            by["serve"][srv].append(u - F)
            by["return"][ret].append(u - F)
    out = {}
    for side, d in by.items():
        a = np.array([(np.mean(v[0::2]), np.mean(v[1::2])) for v in d.values() if len(v) >= min_matches])
        r = float(np.corrcoef(a[:, 0], a[:, 1])[0, 1])
        se = (1 - r * r) / math.sqrt(len(a) - 1)
        out[side] = {"players": len(a), "r_half": r, "se": se, "r_full": 2 * r / (1 + r),
                     "sd_player_mean": float(a.mean(axis=1).std())}
    return out


# ---------------------------------------------------------------------------- variant

def attach(mds: list, prep, vol: dict) -> None:
    """Give each MatchData its surface, player ids and the walk-forward volatility sums."""
    m = prep.m
    idx = {mid: i for i, mid in enumerate(m.match_id)}
    for md in mds:
        i = idx[md.mid]
        w, l = m.winner[i], m.loser[i]
        p1, p2 = (w, l) if md.y else (l, w)
        md.surface = m.surface[i]
        v = vol.get(md.mid)
        md.r3 = None if v is None else (v["F"], v[p1], v[p2])


@dataclass(frozen=True)
class SplitForm:
    """Independent-stream form update with a form variance per direction."""
    form_var: float = points_rating.IN_PLAY_FORM_VAR
    by_group: tuple = ()             # ((key, f), ...), key = "atp" / "wta/clay" / "atp/bo5" ...
    kappa_s: float | None = None     # None: no per-player serve-side term
    kappa_r: float | None = None
    floor: float = 0.25
    name: str = ""

    def label(self) -> str:
        if self.name:
            return self.name
        s = f"sf{self.form_var:g}"
        if self.by_group:
            s += " [" + ",".join(f"{k}:{f:g}" for k, f in self.by_group) + "]"
        if self.kappa_s is not None:
            s += f" ks{self.kappa_s:g}"
        if self.kappa_r is not None:
            s += f" kr{self.kappa_r:g}"
        return s

    def base_var(self, md) -> float:
        g = dict(self.by_group)
        for key in (f"{md.tour}/{md.surface}/bo{md.best_of}", f"{md.tour}/{md.surface}", f"{md.tour}/bo{md.best_of}", md.tour):
            if key in g:
                return g[key]
        return self.form_var

    def direction_vars(self, md) -> tuple[float, float]:
        f = self.base_var(md)
        if (self.kappa_s is None and self.kappa_r is None) or md.r3 is None or not md.r3[0] > 0:
            return f, f
        F, a, b = md.r3

        def mult(srv, ret):
            d = 0.0
            if self.kappa_s is not None:
                d += srv[0] / (srv[1] + self.kappa_s)
            if self.kappa_r is not None:
                d += ret[2] / (ret[3] + self.kappa_r)
            return max(self.floor, 1 + d / F)
        return f * mult(a, b), f * mult(b, a)

    def match_probs(self, md, params) -> np.ndarray:
        st = points_rating.PointsState(params, *({} for _ in range(13)))
        f1, f2 = self.direction_vars(md)
        out = np.empty(len(md.cps))
        for j, c in enumerate(md.cps):
            x1, v1 = points_rating.in_match_form(md.x1, md.v1, c.k1, c.n1, f1)
            x2, v2 = points_rating.in_match_form(md.x2, md.v2, c.k2, c.n2, f2)
            score = dict(sets_a=c.sets1, sets_b=c.sets2, games_a=c.games1, games_b=c.games2, pts_a=0, pts_b=0)
            out[j] = points_rating.in_play_probability(st, x1, x2, v1, v2, md.best_of, md.final, score,
                                                       c.p1_serving, deciding_tb_points=md.dtb)
        return out
