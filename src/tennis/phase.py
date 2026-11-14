"""Phase profiles from MCP charted points: where a player wins and loses points.

Phases are defined on every charted point from the server's side:
    k = 0 short  (<= 3 strokes incl. the serve; double faults count here)
    k = 1 medium (4-6 strokes)
    k = 2 long   (>= 7 strokes)
    k = 3 server came to the net      (not exclusive with 0-2)
    k = 4 returner came to the net    (not exclusive with 0-2)
Rally length and net detection follow style_features (same parser, same conventions).

Per player and side (serve / return), as of a date D, using only charted matches whose
edition started strictly before D:
    level  d   = sum_s (W_s - N_s P_s) / (N + K_level)                 own level vs population
    edge   e_k = sum_s (w_ks - n_ks (R_ks + d)) / (n_k + K_edge)        phase win rate beyond
                                                                        population + own level
    steer  t_k = sum_s (n_ks - N_s pi_ks) / (N + K_steer)               phase share vs population
with W, N side points won/played, w_k, n_k in phase k, and population rates P (side win
rate), R_k (phase win rate), pi_k (phase share) per tour (and surface s). Win rates are
the player's own (serve points won as server, return points won as returner).

Variants (the surface question):
    pooled   population rates per tour only; all surfaces summed (the 2026-11-14 test)
    surface  population rates per (tour, surface); own level measured against them
    same     like surface, but only points on the target match's surface

Population rates are computed from charted points dated before 1 January of the target
year (point-in-time).

Matchup term for A serving to B (probability units, A's serve-point win rate):
    M_A = sum_k t^ret_{B,k} e^srv_{A,k} - sum_k t^srv_{A,k} e^ret_{B,k}
the cross terms of sum_k (pi_k + t_A + t_B)(R_k + e_A - e_B); all one-player terms are
level, which the ratings already carry.
"""
from __future__ import annotations

import bisect
import sqlite3
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .mcp_notation import NotationError, parse_point

SURFACES = ("hard", "clay", "grass", "carpet")
SURF_INDEX = {s: i for i, s in enumerate(SURFACES)}
NS = len(SURFACES)
NK = 5
PHASES = ("short", "medium", "long", "srv_net", "ret_net")
SIDES = ("serve", "return")
# layout of one stats vector per (side, surface): N, W, n_0..n_4, w_0..w_4
NF = 2 + 2 * NK
VARIANTS = ("pooled", "surface", "same")


def phase_of(n_strokes: int) -> int:
    return 0 if n_strokes <= 3 else (1 if n_strokes <= 6 else 2)


def point_phases(pt) -> tuple[int, bool, bool]:
    """(rally phase, server came to net, returner came to net) for a parsed point."""
    shots = pt.shots
    dec = pt.deciding
    srv_net = bool(dec.serve_and_volley) if not pt.double_fault else False
    ret_net = bool(shots) and (shots[0].approach or shots[0].at_net)
    for k in range(1, len(shots)):
        s = shots[k]
        if s.approach or s.at_net:
            if k % 2 == 1:
                srv_net = True
            else:
                ret_net = True
    return phase_of(pt.n_strokes), srv_net, ret_net


@dataclass
class Points:
    """One row per parsed charted point."""
    match_id: np.ndarray
    server: np.ndarray
    returner: np.ndarray
    server_won: np.ndarray   # 0/1
    phase: np.ndarray        # 0..2
    srv_net: np.ndarray      # 0/1
    ret_net: np.ndarray
    skipped: Counter = field(default_factory=Counter)


def parse_points(rows) -> Points:
    """rows: (match_id, server_id, returner_id, winner_id, raw_first, raw_second)."""
    out = {k: [] for k in ("m", "s", "r", "w", "p", "sn", "rn")}
    skipped = Counter()
    for mid, srv, ret, win, first, second in rows:
        try:
            pt = parse_point(first, second, win == srv)
        except NotationError as e:
            skipped[e.reason] += 1
            continue
        ph, sn, rn = point_phases(pt)
        out["m"].append(mid); out["s"].append(srv); out["r"].append(ret)
        out["w"].append(1 if win == srv else 0); out["p"].append(ph)
        out["sn"].append(1 if sn else 0); out["rn"].append(1 if rn else 0)
    a = lambda k, t: np.asarray(out[k], dtype=t)
    return Points(a("m", np.int64), a("s", np.int64), a("r", np.int64), a("w", np.int8),
                  a("p", np.int8), a("sn", np.int8), a("rn", np.int8), skipped)


def load_points(con: sqlite3.Connection, alias: str = "main") -> Points:
    return parse_points(con.execute(
        f"""SELECT match_id, server_id, returner_id, winner_id, raw_first, raw_second
            FROM {alias}.point WHERE data_source = 'mcp' ORDER BY match_id, point_idx"""))


# ------------------------------------------------------------------ per-match tallies

@dataclass
class Tallies:
    """Per (charted match, player): stats array (2 sides, NS surfaces, NF), surface-resolved.
    Only the match's own surface slot is non-zero."""
    match_id: np.ndarray
    player: np.ndarray
    date: np.ndarray          # edition start, ISO
    tour: np.ndarray
    surf: np.ndarray          # surface index, -1 unknown
    stats: np.ndarray         # (rows, 2, NS, NF)


def tally(pts: Points, meta: dict[int, tuple[str, str, str | None]]) -> Tallies:
    """meta: match_id -> (edition start, tour, surface). Points of matches without a known
    surface are dropped (they cannot enter the surface variants)."""
    surf_of = {m: SURF_INDEX.get(v[2] or "", -1) for m, v in meta.items()}
    ms = np.array([surf_of.get(int(m), -1) for m in pts.match_id], dtype=np.int64) \
        if len(pts.match_id) else np.zeros(0, dtype=np.int64)
    keep = ms >= 0
    rows: dict[tuple[int, int], np.ndarray] = {}
    mid, srv, ret = pts.match_id[keep], pts.server[keep], pts.returner[keep]
    won, ph, sn, rn, sf = (pts.server_won[keep], pts.phase[keep], pts.srv_net[keep],
                           pts.ret_net[keep], ms[keep])
    # index each (match, player) pair
    pairs = {}
    for side, who in ((0, srv), (1, ret)):
        for i in range(len(mid)):
            key = (int(mid[i]), int(who[i]))
            if key not in pairs:
                pairs[key] = len(pairs)
    stats = np.zeros((len(pairs), 2, NS, NF))
    for side, who, wv in ((0, srv, won), (1, ret, 1 - won)):
        r = np.array([pairs[(int(m), int(p))] for m, p in zip(mid, who)], dtype=np.int64)
        np.add.at(stats, (r, side, sf, 0), 1)
        np.add.at(stats, (r, side, sf, 1), wv)
        for k, mask in ((0, ph == 0), (1, ph == 1), (2, ph == 2), (3, sn == 1), (4, rn == 1)):
            np.add.at(stats, (r[mask], side, sf[mask], 2 + k), 1)
            np.add.at(stats, (r[mask], side, sf[mask], 2 + NK + k), wv[mask])
    keys = sorted(pairs, key=lambda k: (meta[k[0]][0], k[0], k[1]))
    order = np.array([pairs[k] for k in keys], dtype=np.int64)
    return Tallies(
        match_id=np.array([k[0] for k in keys], dtype=np.int64),
        player=np.array([k[1] for k in keys], dtype=np.int64),
        date=np.array([meta[k[0]][0] for k in keys], dtype="U10"),
        tour=np.array([meta[k[0]][1] for k in keys], dtype="U3"),
        surf=np.array([surf_of[k[0]] for k in keys], dtype=np.int64),
        stats=stats[order] if len(order) else stats)


# ------------------------------------------------------------------ point-in-time lookup

class History:
    """Cumulative per-player stats by date, and population stats by year (per tour)."""

    def __init__(self, t: Tallies):
        self.by_player: dict[int, tuple[list, np.ndarray]] = {}
        order = np.lexsort((t.match_id, t.date, t.player))
        p_sorted = t.player[order]
        bounds = np.flatnonzero(np.diff(p_sorted)) + 1
        for grp in np.split(order, bounds):
            if not len(grp):
                continue
            pid = int(t.player[grp[0]])
            dates = list(t.date[grp])
            cum = np.concatenate([np.zeros((1, 2, NS, NF)), np.cumsum(t.stats[grp], axis=0)])
            self.by_player[pid] = (dates, cum)
        # population per tour: cumulative by year (stats of years < Y)
        self.pop: dict[tuple[str, int], np.ndarray] = {}
        years = np.array([int(d[:4]) for d in t.date])
        for tour in np.unique(t.tour):
            sel = t.tour == tour
            acc = np.zeros((2, NS, NF))
            for y in range(1970, 2028):
                self.pop[(tour, y)] = acc.copy()
                ys = sel & (years == y)
                if ys.any():
                    acc += t.stats[ys].sum(axis=0)

    def player(self, pid: int, before: str) -> np.ndarray | None:
        """Stats (2, NS, NF) from charted matches with edition start < before."""
        h = self.by_player.get(pid)
        if h is None:
            return None
        i = bisect.bisect_left(h[0], before)
        return h[1][i]

    def population(self, tour: str, before: str) -> np.ndarray:
        return self.pop[(tour, int(before[:4]))]


@dataclass(frozen=True)
class Shrink:
    level: float = 300.0
    edge: float = 150.0
    steer: float = 150.0


def _rates(pop: np.ndarray):
    """Population P (2,NS), R (2,NS,NK), pi (2,NS,NK); NaN where no data."""
    with np.errstate(invalid="ignore", divide="ignore"):
        P = pop[..., 1] / pop[..., 0]
        R = pop[..., 2 + NK:] / pop[..., 2:2 + NK]
        pi = pop[..., 2:2 + NK] / pop[..., 0:1]
    return P, R, pi


def profile(stats: np.ndarray, pop: np.ndarray, variant: str, surf: int,
            sh: Shrink = Shrink()) -> dict:
    """Level d (2,), edges e (2,NK), steering t (2,NK), points N (2,) for one player.
    stats/pop: (2, NS, NF). surf: target match surface index (used by 'same')."""
    if variant == "pooled":
        stats = stats.sum(axis=1, keepdims=True)
        pop = pop.sum(axis=1, keepdims=True)
    elif variant == "same":
        stats = stats[:, surf:surf + 1]
        pop = pop[:, surf:surf + 1]
    elif variant != "surface":
        raise ValueError(variant)
    P, R, pi = _rates(pop)
    P, R, pi = np.nan_to_num(P), np.nan_to_num(R), np.nan_to_num(pi)
    N_s, W_s = stats[..., 0], stats[..., 1]
    n_ks, w_ks = stats[..., 2:2 + NK], stats[..., 2 + NK:]
    N = N_s.sum(axis=1)
    d = (W_s - N_s * P).sum(axis=1) / (N + sh.level)
    e = (w_ks - n_ks * (R + d[:, None, None])).sum(axis=1) / (n_ks.sum(axis=1) + sh.edge)
    t = (n_ks - N_s[..., None] * pi).sum(axis=1) / (N + sh.steer)[:, None]
    return {"d": d, "e": e, "t": t, "N": N}


def matchup(a: dict, b: dict) -> float:
    """M for A serving to B (A's serve-point win rate units)."""
    return float(b["t"][1] @ a["e"][0] - a["t"][0] @ b["e"][1])
