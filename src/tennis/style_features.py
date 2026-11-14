"""Style features from charted points: per-match counts, point-in-time totals, shrinkage.

Every feature is a rate num/den with a known denominator, so thin-data players can be
shrunk toward the tour mean (empirical Bayes, beta-binomial method of moments).

Three feature sets, each later gets its own scaler + PCA (heavy.py):
- serve  : the player's service points
- return : the player's return points
- rally  : the player's own rally shots (everything after the serve and the return) and
           net play on all their points. Kept separate so the same player-level features
           are not fed to the middle layer twice (decision of 2026-11-13).

Rally length = strokes including the serve and the final (error) shot; a double fault is 0.

Timing: a charted match is dated by its backbone edition's start date, the same clock every
other layer uses (Sackmann dates matches only by their event's Monday). A vector "as of D"
uses only matches whose edition started strictly before D, so charts from the event being
predicted (including its qualifying) are never visible.
"""
from __future__ import annotations

import bisect
import sqlite3
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .mcp_notation import (BACKHAND_SIDE, FOREHAND_SIDE, NotationError, Point, SLICE, DROP, LOB,
                           GROUND, parse_point)

FEATURE_VERSION = "style_features_v1"
SIDES = ("serve", "return", "rally")

FEATURES: dict[str, list[str]] = {
    "serve": [
        "first_in",          # 1st serves in / 1st serves
        "second_in",         # 2nd serves in / 2nd serves (1 - double-fault share)
        "ace",               # aces / service points
        "unret_1st",         # unreturned (ace, unreturnable, return error) / 1st serves in
        "unret_2nd",         # same / 2nd serves in
        "serve_volley",      # serve-and-volley / serves in
        "dir1_deuce_wide",   # 1st serves (in or out) with known direction, deuce court
        "dir1_deuce_t",
        "dir1_ad_wide",      # ... ad court
        "dir1_ad_t",
        "dir2_wide",         # 2nd serves with known direction, both courts
        "dir2_t",
        "plus1_forehand",    # serve+1 shot (3rd stroke) hit on the forehand side
        "len_le3",           # service points (no double faults) lasting <= 3 strokes
        "len_4_6",
        "len_7_9",
    ],
    "return": [
        "in_vs_1st",         # returns in play / 1st serves in (aces excluded)
        "in_vs_2nd",         # returns in play / 2nd serves in (aces excluded)
        "depth_shallow",     # returns in play landing in the service boxes (7) / with depth coded
        "depth_deep",        # ... near the baseline (9)
        "forehand",          # returns on the forehand side / returns with a side
        "slice",             # sliced/chipped returns / returns hit f, b, r or s
        "approach",          # return marked as an approach (chip-and-charge) / returns in play
        "len_le3",           # return points (no double faults) lasting <= 3 strokes
        "len_4_6",
        "len_7_9",
    ],
    "rally": [
        "forehand",          # forehand-side rally shots / rally shots with a side
        "slice",             # r, s / f, b, r, s
        "drop",              # drop shots / rally shots
        "lob",               # lobs / rally shots
        "net",               # points where the player came to the net / points played
        "net_won",           # those points won / net points
        "fh_winner",         # winners off forehand-side rally shots / forehand-side rally shots
        "bh_winner",
        "fh_unforced",       # unforced errors off forehand-side rally shots / same denominator
        "bh_unforced",
    ],
}
INDEX = {side: {f: i for i, f in enumerate(fs)} for side, fs in FEATURES.items()}

_SCORE = {"0": 0, "15": 1, "30": 2, "40": 3, "AD": 4, "A": 4}


def is_deuce_court(score_before: str | None, is_tiebreak: int | None) -> bool | None:
    """Deuce court when an even number of points has been played in the game/tiebreak."""
    if not score_before or "-" not in score_before:
        return None
    a, b = (x.strip().upper() for x in score_before.split("-", 1))
    if is_tiebreak:
        if not (a.isdigit() and b.isdigit()):
            return None
        return (int(a) + int(b)) % 2 == 0
    if a in _SCORE and b in _SCORE:
        return (_SCORE[a] + _SCORE[b]) % 2 == 0
    if a.isdigit() and b.isdigit():       # tiebreak-style score without the flag (super-TB)
        return (int(a) + int(b)) % 2 == 0
    return None


class Tally:
    """num/den accumulators for one (player, match): one pair of arrays per side."""
    __slots__ = ("num", "den", "points")

    def __init__(self):
        self.num = {s: [0] * len(FEATURES[s]) for s in SIDES}   # lists: faster than numpy scalars
        self.den = {s: [0] * len(FEATURES[s]) for s in SIDES}
        self.points = {s: 0 for s in SIDES}   # serve points, return points, all points

    def add(self, side: str, feat: str, hit: bool) -> None:
        i = INDEX[side][feat]
        self.den[side][i] += 1
        if hit:
            self.num[side][i] += 1


def _length_bucket(t: Tally, side: str, n: int) -> None:
    t.add(side, "len_le3", n <= 3)
    t.add(side, "len_4_6", 4 <= n <= 6)
    t.add(side, "len_7_9", 7 <= n <= 9)


def tally_point(pt: Point, deuce: bool | None, srv: Tally, ret: Tally) -> None:
    """Add one parsed point to the server's and returner's tallies."""
    srv.points["serve"] += 1
    ret.points["return"] += 1
    srv.points["rally"] += 1
    ret.points["rally"] += 1
    c1, c2, dec = pt.first, pt.second, pt.deciding

    # ---- serve
    first_in = c1.serve_outcome in ("in", "ace", "unreturnable")
    srv.add("serve", "first_in", first_in)
    if c2 is not None:
        srv.add("serve", "second_in", not pt.double_fault)
    srv.add("serve", "ace", dec.serve_outcome == "ace")
    if c1.serve_dir is not None and deuce is not None:
        court = "deuce" if deuce else "ad"
        srv.add("serve", f"dir1_{court}_wide", c1.serve_dir == 4)
        srv.add("serve", f"dir1_{court}_t", c1.serve_dir == 6)
    if c2 is not None and c2.serve_dir is not None:
        srv.add("serve", "dir2_wide", c2.serve_dir == 4)
        srv.add("serve", "dir2_t", c2.serve_dir == 6)
    if pt.double_fault:
        return                                   # nothing below exists on a double fault

    shots = dec.shots
    ret_error = len(shots) == 1 and dec.ending in ("forced", "unforced")
    unreturned = dec.serve_outcome in ("ace", "unreturnable") or ret_error
    srv.add("serve", "unret_1st" if pt.serve_no == 1 else "unret_2nd", unreturned)
    srv.add("serve", "serve_volley", dec.serve_and_volley)
    if len(shots) >= 2 and (shots[1].kind in FOREHAND_SIDE or shots[1].kind in BACKHAND_SIDE):
        srv.add("serve", "plus1_forehand", shots[1].kind in FOREHAND_SIDE)
    _length_bucket(srv, "serve", pt.n_strokes)

    # ---- return
    _length_bucket(ret, "return", pt.n_strokes)
    if dec.serve_outcome != "ace":
        in_play = len(shots) >= 2 or (len(shots) == 1 and dec.ending == "winner")
        ret.add("return", "in_vs_1st" if pt.serve_no == 1 else "in_vs_2nd", in_play)
        if shots:
            r = shots[0]
            if r.kind in FOREHAND_SIDE or r.kind in BACKHAND_SIDE:
                ret.add("return", "forehand", r.kind in FOREHAND_SIDE)
            if r.kind in GROUND or r.kind in SLICE:
                ret.add("return", "slice", r.kind in SLICE)
            if in_play:
                if r.depth is not None:
                    ret.add("return", "depth_shallow", r.depth == 7)
                    ret.add("return", "depth_deep", r.depth == 9)
                ret.add("return", "approach", r.approach)

    # ---- rally: shots[k] is hit by the returner for even k, the server for odd k
    net = {True: dec.serve_and_volley, False: bool(shots) and (shots[0].approach or shots[0].at_net)}
    last = len(shots) - 1
    for k in range(1, len(shots)):
        s = shots[k]
        is_srv = k % 2 == 1
        t = srv if is_srv else ret
        if s.approach or s.at_net:
            net[is_srv] = True
        fh, bh = s.kind in FOREHAND_SIDE, s.kind in BACKHAND_SIDE
        if fh or bh:
            t.add("rally", "forehand", fh)
        if s.kind in GROUND or s.kind in SLICE:
            t.add("rally", "slice", s.kind in SLICE)
        t.add("rally", "drop", s.kind in DROP)
        t.add("rally", "lob", s.kind in LOB)
        if fh or bh:
            hand = "fh" if fh else "bh"
            t.add("rally", f"{hand}_winner", k == last and dec.ending == "winner")
            t.add("rally", f"{hand}_unforced", k == last and dec.ending == "unforced")
    for is_srv, t in ((True, srv), (False, ret)):
        t.add("rally", "net", net[is_srv])
        if net[is_srv]:
            t.add("rally", "net_won", pt.server_won == is_srv)


# ------------------------------------------------------------------ per-match counts

@dataclass
class MatchCounts:
    """Per (charted match, player) totals. Arrays are aligned row-wise."""
    match_id: np.ndarray        # tennis.db match_id
    player: np.ndarray          # tennis.db player_id
    date: np.ndarray            # edition start date, ISO text
    tour: np.ndarray            # 'atp' / 'wta'
    surface: np.ndarray         # edition surface, '' if unknown (for later per-surface features)
    num: dict[str, np.ndarray] = field(default_factory=dict)    # side -> (rows, F)
    den: dict[str, np.ndarray] = field(default_factory=dict)
    points: dict[str, np.ndarray] = field(default_factory=dict)  # side -> (rows,)
    skipped: Counter = field(default_factory=Counter)

    def __len__(self) -> int:
        return len(self.match_id)


def count_points(rows, skipped: Counter | None = None) -> dict[tuple[int, int], Tally]:
    """rows: (match_id, server_id, returner_id, winner_id, raw_first, raw_second,
    score_before, is_tiebreak). Returns {(match_id, player_id): Tally}."""
    skipped = Counter() if skipped is None else skipped
    tallies: dict[tuple[int, int], Tally] = {}
    for match_id, server, returner, winner, first, second, score, tb in rows:
        try:
            pt = parse_point(first, second, winner == server)
        except NotationError as e:
            skipped[e.reason] += 1
            continue
        s = tallies.get((match_id, server)) or tallies.setdefault((match_id, server), Tally())
        r = tallies.get((match_id, returner)) or tallies.setdefault((match_id, returner), Tally())
        tally_point(pt, is_deuce_court(score, tb), s, r)
    return tallies


def load_match_counts(con: sqlite3.Connection, alias: str = "main") -> MatchCounts:
    """Parse every MCP point in tennis.db (attached as `alias`) into per-match tallies."""
    rows = con.execute(
        f"""SELECT match_id, server_id, returner_id, winner_id, raw_first, raw_second,
                   score_before, is_tiebreak
            FROM {alias}.point WHERE data_source = 'mcp' ORDER BY match_id, point_idx""")
    skipped = Counter()
    tallies = count_points(rows, skipped)
    meta = {mid: (d, tour, surf) for mid, d, tour, surf in con.execute(
        f"""SELECT m.match_id, e.start_date, t.tour, e.surface
            FROM {alias}.matches m JOIN {alias}.tournament_edition e USING (edition_id)
            JOIN {alias}.tournament t USING (tournament_id)
            WHERE m.match_id IN (SELECT DISTINCT match_id FROM {alias}.point WHERE data_source='mcp')""")}
    return build_match_counts(tallies, meta, skipped)


def build_match_counts(tallies: dict[tuple[int, int], Tally], meta: dict[int, tuple],
                       skipped: Counter | None = None) -> MatchCounts:
    """meta: match_id -> (edition start date, tour, surface)."""
    keys = sorted(tallies, key=lambda k: (meta[k[0]][0], k[0], k[1]))
    mc = MatchCounts(
        match_id=np.array([k[0] for k in keys], dtype=np.int64),
        player=np.array([k[1] for k in keys], dtype=np.int64),
        date=np.array([meta[k[0]][0] for k in keys], dtype="U10"),
        tour=np.array([meta[k[0]][1] for k in keys], dtype="U3"),
        surface=np.array([meta[k[0]][2] or "" for k in keys], dtype="U6"),
        skipped=skipped or Counter())
    for s in SIDES:
        mc.num[s] = np.array([tallies[k].num[s] for k in keys]).reshape(len(keys), len(FEATURES[s]))
        mc.den[s] = np.array([tallies[k].den[s] for k in keys]).reshape(len(keys), len(FEATURES[s]))
        mc.points[s] = np.array([tallies[k].points[s] for k in keys], dtype=np.int64)
    return mc


# ------------------------------------------------------------------ point-in-time totals

@dataclass
class Totals:
    """A player's summed counts over charted matches strictly before `as_of`."""
    player: np.ndarray       # player ids, sorted
    num: np.ndarray          # (players, F)
    den: np.ndarray
    n_matches: np.ndarray
    n_points: np.ndarray
    max_date: str | None     # latest input row's date (leakage check: must be < as_of)


class History:
    """Cumulative per-player counts, so totals at any date are a bisect away."""

    def __init__(self, mc: MatchCounts, tour: str, side: str):
        sel = np.flatnonzero(mc.tour == tour)
        self.tour = tour
        self.side = side
        self.n_features = len(FEATURES[side])
        self.by_player: dict[int, tuple[list[str], np.ndarray, np.ndarray, np.ndarray]] = {}
        players = mc.player[sel]
        for p in np.unique(players):
            rows = sel[players == p]                       # already in date order
            dates = [str(d) for d in mc.date[rows]]
            cnum = np.cumsum(mc.num[side][rows], axis=0)
            cden = np.cumsum(mc.den[side][rows], axis=0)
            cpts = np.cumsum(mc.points[side][rows])
            self.by_player[int(p)] = (dates, cnum, cden, cpts)

    def player_totals(self, player: int, as_of: str):
        """(num, den, n_matches, n_points, last_date) before as_of, or None if no charted match."""
        h = self.by_player.get(player)
        if h is None:
            return None
        dates, cnum, cden, cpts = h
        i = bisect.bisect_left(dates, as_of)       # rows [0, i) have date < as_of
        if i == 0:
            return None
        assert dates[i - 1] < as_of, "leakage: input row dated on/after as_of"
        return cnum[i - 1], cden[i - 1], i, int(cpts[i - 1]), dates[i - 1]

    def totals(self, as_of: str, min_matches: int = 1) -> Totals:
        ps, nums, dens, nm, npts, last = [], [], [], [], [], None
        for p in sorted(self.by_player):
            got = self.player_totals(p, as_of)
            if got is None or got[2] < min_matches:
                continue
            ps.append(p); nums.append(got[0]); dens.append(got[1]); nm.append(got[2]); npts.append(got[3])
            last = got[4] if last is None or got[4] > last else last
        F = self.n_features
        return Totals(np.array(ps, dtype=np.int64), np.array(nums).reshape(-1, F),
                      np.array(dens).reshape(-1, F), np.array(nm, dtype=np.int64),
                      np.array(npts, dtype=np.int64), last)


# ------------------------------------------------------------------ shrinkage

def estimate_priors(num: np.ndarray, den: np.ndarray, min_den: float = 30.0,
                    k_min: float = 1.0, k_max: float = 1e4) -> tuple[np.ndarray, np.ndarray]:
    """Tour mean and prior strength k per feature (beta-binomial method of moments).

    Uses players with den >= min_den. mean = average of their raw rates (each player counts
    once, so a heavily charted star does not drag the prior); between-player variance of
    true rates = observed variance of raw rates minus the average binomial noise
    mean(1-mean)/n. A Beta prior with that mean and variance has a + b = mean(1-mean)/var - 1 = k.
    With fewer than 3 such players the mean falls back to the pooled rate and k to k_max.
    """
    F = num.shape[1]
    mean = np.full(F, np.nan)
    k = np.full(F, k_max)
    for j in range(F):
        tot = den[:, j].sum()
        if tot <= 0:
            continue
        ok = den[:, j] >= min_den
        if ok.sum() < 3:
            mean[j] = num[:, j].sum() / tot
            continue
        p = num[ok, j] / den[ok, j]
        m = mean[j] = p.mean()
        if m <= 0 or m >= 1:
            continue
        noise = np.mean(m * (1 - m) / den[ok, j])
        tau2 = np.var(p, ddof=1) - noise
        if tau2 > 0:
            k[j] = np.clip(m * (1 - m) / tau2 - 1, k_min, k_max)
    return mean, k


def shrink(num: np.ndarray, den: np.ndarray, mean: np.ndarray, k: np.ndarray) -> np.ndarray:
    """(n * raw + k * mean) / (n + k), with n * raw = num. Rows with den 0 land on the mean."""
    m = np.nan_to_num(mean, nan=0.0)
    return (num + k * m) / (den + k)
