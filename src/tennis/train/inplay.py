"""In-play evaluation (milestone A4; from v1 inplay_form_test.py, extended to game ends).

Data: completed matches with point-by-point rows (slam pbp + MCP, 2011+) whose points
replay to the recorded result under the match's format. A tennis state machine walks
the points (games to 4 by 2, tiebreak at 6-6 / 12-12 / none, match tiebreak), checks set
boundaries against the source's set_idx, and must end exactly at match point with the
right winner; the deciding-set tiebreak length (7 or 10) is whichever replays.

Checkpoints: after every game that does not end the match (set ends are flagged). At each,
P(p1 wins) for p1 = canonical player order (never winner first), from the run's
walk-forward pre-match logits and variances (state strictly before the match) and only
the points already played.

Variants (InPlay): form_var f and form_corr rho of the joint form update
(points_rating.in_match_form2), the in-play uncertainty shrink `unc`, and a calibration
alpha per stage (best-of, sets played) fitted on earlier years. Scored by log loss per
checkpoint; paired differences with SEs clustered by match.

Periods: fit 2011-18 (choose), val 2019-21 (confirm), test 2022-26 (once).
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

from .. import calibrate, models_db, points_rating
from .data import Prepared

PERIODS = {"fit": ("2011", "2018"), "val": ("2019", "2021"), "test": ("2022", "2026")}
CACHE = models_db.ROOT / "models" / "inplay"


# ---------------------------------------------------------------------------- pre-match

def prematch(prep: Prepared, params, log=print) -> dict:
    """Per-match walk-forward pre-match arrays (logits with blend, variances, format),
    cached under models/inplay/ by match table and params."""
    m = prep.m
    h = hashlib.sha256(json.dumps(params.as_dict(), sort_keys=True).encode())
    h.update(f"{len(m)} {m.match_id[-1]} {m.start[-1]}".encode())
    path = CACHE / f"prematch_{h.hexdigest()[:16]}.npz"
    if path.exists():
        z = np.load(path)
        return {k: z[k] for k in z.files}
    log(f"replaying {len(m):,} matches for pre-match logits ...")
    res = points_rating.run(m, params)
    b = params.blend
    out = {"xw": np.asarray(res.xa) + b * np.asarray(res.xa_s), "xl": np.asarray(res.xb) + b * np.asarray(res.xb_s),
           "vw": np.asarray(res.va) + b * np.asarray(res.va_s), "vl": np.asarray(res.vb) + b * np.asarray(res.vb_s),
           "fmt": np.asarray(res.fmt, dtype=np.int64)}
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)
    return out


# ---------------------------------------------------------------------------- replay

@dataclass
class Checkpoint:
    sets1: int
    sets2: int
    games1: int
    games2: int
    p1_serving: bool          # p1 serves the next point
    k1: int                   # p1's serve points won / played so far
    n1: int
    k2: int
    n2: int
    set_end: bool
    a1: int = 0               # aces served so far by p1 / p2
    a2: int = 0


@dataclass
class MatchData:
    mid: int
    year: str
    tour: str
    best_of: int
    final: str
    dtb: int                  # deciding-set tiebreak length that replays (7 / 10)
    x1: float
    x2: float
    v1: float
    v2: float
    y: int                    # p1 won
    cps: list = field(default_factory=list)
    ace1: float = 0.0         # expected aces per serve point (point-in-time), p1 / p2
    ace2: float = 0.0


def replay(points, p1: int, p2: int, best_of: int, final: str, dtb: int):
    """points: [(set_idx, server, winner[, is_ace])] in order. Returns (checkpoints, p1 won) if the
    points replay to a finished match under the format, else None."""
    need = best_of // 2 + 1
    sets = {p1: 0, p2: 0}
    games = {p1: 0, p2: 0}
    pts = {p1: 0, p2: 0}
    sv = {p1: [0, 0], p2: [0, 0]}
    aces = {p1: 0, p2: 0}
    set_no = 1
    cps = []
    n = len(points)
    for k, pt in enumerate(points):
        s_idx, srv, win = pt[0], pt[1], pt[2]
        if srv not in sets or win not in sets:
            return None
        if len(pt) > 3 and pt[3] and win == srv:
            aces[srv] += 1
        if s_idx is not None and s_idx != set_no:
            return None
        if max(sets.values()) >= need:
            return None                                  # points after match point
        other = p2 if win == p1 else p1
        sv[srv][0] += win == srv
        sv[srv][1] += 1
        deciding = sets[p1] == sets[p2] == need - 1
        tb_at = None if (deciding and final == "adv") else 12 if (deciding and final == "tb12") else 6
        in_mtb = deciding and final == "mtb"
        in_tb = in_mtb or (tb_at is not None and games[p1] == games[p2] == tb_at)
        pts[win] += 1
        target = (10 if in_mtb else dtb if deciding else 7) if in_tb else 4
        if pts[win] >= target and pts[win] - pts[other] >= 2:
            pts = {p1: 0, p2: 0}
            games[win] += 1
            g_w, g_o = games[win], games[other]
            set_won = in_tb or (g_w >= 6 and g_w - g_o >= 2)
            if set_won:
                sets[win] += 1
                games = {p1: 0, p2: 0}
                set_no += 1
            if max(sets.values()) >= need:
                if k != n - 1:
                    return None
                return cps, int(sets[p1] >= need)
            if k + 1 < n:
                nxt_srv = points[k + 1][1]
                cps.append(Checkpoint(sets[p1], sets[p2], games[p1], games[p2], nxt_srv == p1,
                                      sv[p1][0], sv[p1][1], sv[p2][0], sv[p2][1], set_won,
                                      aces[p1], aces[p2]))
    return None


def load_points(con, ids: set, alias: str = "data") -> dict:
    """match_id -> [(set_idx, server, winner, is_ace)] from the source with the most points."""
    by = defaultdict(lambda: defaultdict(list))
    for mid, src, s, srv, win, ace in con.execute(
            f"SELECT match_id, data_source, set_idx, server_id, winner_id, COALESCE(is_ace, 0) "
            f"FROM {alias}.point "
            "ORDER BY match_id, data_source, point_idx"):
        if mid in ids:
            by[mid][src].append((s, srv, win, ace))
    return {mid: max(d.values(), key=len) for mid, d in by.items()}


def dataset(con, prep: Prepared, pm: dict, start: str = "2011", alias: str = "data") -> tuple[list, int]:
    """(matches that replay, number dropped)."""
    m = prep.m
    idx = {mid: i for i, mid in enumerate(m.match_id)}
    cand = {mid for mid, i in idx.items() if m.start[i] >= start and m.status[i] == "completed"}
    pts = load_points(con, cand, alias)
    rates = ace_rates(con, m, set(pts), alias)
    out, bad = [], 0
    for mid, plist in pts.items():
        i = idx[mid]
        w, l = m.winner[i], m.loser[i]
        bo, final = points_rating.FORMATS[int(pm["fmt"][i])]
        p1, p2 = (w, l) if prep.winner_is_p1[i] else (l, w)
        got = None
        for dtb in (7, 10):
            got = replay(plist, p1, p2, bo, final, dtb)
            if got is not None:
                break
        if got is None or got[1] != int(p1 == w):
            bad += 1
            continue
        x = {w: float(pm["xw"][i]), l: float(pm["xl"][i])}
        v = {w: float(pm["vw"][i]), l: float(pm["vl"][i])}
        r = rates.get(mid, {})
        out.append(MatchData(mid, m.start[i][:4], m.tour[i], bo, final, dtb, x[p1], x[p2], v[p1], v[p2],
                             got[1], got[0], r.get(p1, 0.0), r.get(p2, 0.0)))
    return out, bad


ACE_PRIOR_POINTS = 300       # pseudo serve points pulling a player's ace rate to the tour mean


def ace_rates(con, m, mids: set, alias: str = "data", prior: float = ACE_PRIOR_POINTS) -> dict:
    """match_id -> {player_id: expected aces per serve point} for matches in `mids`, from
    match_player_stats of strictly earlier matches (walk-forward order of m), shrunk to the
    tour's running mean with `prior` pseudo points."""
    stats = defaultdict(dict)
    for mid, pid, a, sp in con.execute(
            f"SELECT match_id, player_id, aces, serve_points FROM {alias}.match_player_stats "
            "WHERE aces IS NOT NULL AND serve_points > 0"):
        stats[mid][pid] = (a, sp)
    own = defaultdict(lambda: [0, 0])
    tour = defaultdict(lambda: [0, 0])
    out = {}
    for i, mid in enumerate(m.match_id):
        t = m.tour[i]
        if mid in mids:
            ta, ts = tour[t]
            mean = ta / ts if ts else 0.07
            out[mid] = {pid: (own[pid][0] + prior * mean) / (own[pid][1] + prior)
                        for pid in (m.winner[i], m.loser[i])}
        for pid, (a, sp) in stats.get(mid, {}).items():
            own[pid][0] += a
            own[pid][1] += sp
            tour[t][0] += a
            tour[t][1] += sp
    return out


# ---------------------------------------------------------------------------- variants

@dataclass(frozen=True)
class InPlay:
    form_var: float = points_rating.IN_PLAY_FORM_VAR
    form_corr: float = 0.0
    unc: float | None = None          # None: the run's unc
    use_form: bool = True             # False: score only
    ace_bonus: float = 0.0            # extra points won per ace above the player's expected rate

    def label(self) -> str:
        if not self.use_form:
            return "score" + ("" if self.unc is None else f" u{self.unc:g}")
        return (f"f{self.form_var:g} r{self.form_corr:g}" + ("" if self.unc is None else f" u{self.unc:g}")
                + (f" ace{self.ace_bonus:g}" if self.ace_bonus else ""))


def match_probs(md: MatchData, var: InPlay, params) -> np.ndarray:
    st = points_rating.PointsState(params if var.unc is None else replace(params, unc=var.unc),
                                   *({} for _ in range(13)))
    out = np.empty(len(md.cps))
    for j, c in enumerate(md.cps):
        x1, x2, v1, v2 = md.x1, md.x2, md.v1, md.v2
        if var.use_form:
            k1, k2 = c.k1, c.k2
            if var.ace_bonus:
                k1 = min(max(k1 + var.ace_bonus * (c.a1 - c.n1 * md.ace1), 0.0), c.n1)
                k2 = min(max(k2 + var.ace_bonus * (c.a2 - c.n2 * md.ace2), 0.0), c.n2)
            x1, x2, v1, v2 = points_rating.in_match_form2(x1, x2, v1, v2, k1, c.n1, k2, c.n2,
                                                          var.form_var, var.form_corr)
        score = dict(sets_a=c.sets1, sets_b=c.sets2, games_a=c.games1, games_b=c.games2, pts_a=0, pts_b=0)
        out[j] = points_rating.in_play_probability(st, x1, x2, v1, v2, md.best_of, md.final, score,
                                                   c.p1_serving, deciding_tb_points=md.dtb)
    return out


def _chunk(args):
    mds, variants, params = args
    return [[match_probs(md, v, params) for md in mds] for v in variants]


def evaluate(mds: list, variants: list, params, workers: int = 14) -> dict:
    """{variant label: [np.array of P(p1 wins) per checkpoint, one per match]}."""
    chunks = [mds[k::workers] for k in range(workers)]
    out = {v.label(): [None] * len(mds) for v in variants}
    with ProcessPoolExecutor(workers) as ex:
        for k, res in enumerate(ex.map(_chunk, [(c, variants, params) for c in chunks])):
            pos = list(range(k, len(mds), workers))
            for v, per in zip(variants, res):
                for p, arr in zip(pos, per):
                    out[v.label()][p] = arr
    return out


# ---------------------------------------------------------------------------- scoring

def _ll(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def score(mds: list, probs: dict, ref: str, sel=None, cp_sel=None) -> dict:
    """{label: {n, matches, log_loss, d, se}} over checkpoints of matches where sel(md),
    checkpoints where cp_sel(cp); d = paired vs `ref`, SE clustered by match."""
    keep = [i for i, md in enumerate(mds) if sel is None or sel(md)]
    out = {}
    for lab, per in probs.items():
        tot = n = 0.0
        diffs = []
        for i in keep:
            md = mds[i]
            mask = np.array([cp_sel is None or cp_sel(c) for c in md.cps], dtype=bool)
            if not mask.any():
                continue
            a = _ll(per[i][mask], md.y)
            b = _ll(probs[ref][i][mask], md.y)
            tot += a.sum()
            n += mask.sum()
            diffs.append((a - b).sum())
        if not n:
            continue
        d = np.array(diffs)
        out[lab] = {"n": int(n), "matches": len(diffs), "log_loss": tot / n, "d": float(d.sum() / n),
                    "se": float(d.std(ddof=1) * math.sqrt(len(d)) / n) if len(d) > 1 else 0.0}
    return out


def stage(md: MatchData, c: Checkpoint) -> tuple:
    return (md.best_of, c.sets1 + c.sets2)


def fit_stage_alphas(mds: list, probs: list, min_rows: int = 2000) -> dict:
    """alpha per stage (best-of, sets played) on these matches; stages with fewer rows
    than min_rows get the overall alpha."""
    lg, y, keys = [], [], []
    for md, p in zip(mds, probs):
        for c, pj in zip(md.cps, p):
            lg.append(float(calibrate.logit(pj)))
            y.append(md.y)
            keys.append(stage(md, c))
    lg, y = np.array(lg), np.array(y, float)
    keys_arr = np.array([f"{a}/{b}" for a, b in keys])
    out = {"all": calibrate.fit_alpha(lg, y)}
    for k in np.unique(keys_arr):
        s = keys_arr == k
        out[k] = calibrate.fit_alpha(lg[s], y[s]) if s.sum() >= min_rows else out["all"]
    return out


def apply_stage_alphas(mds: list, probs: list, alphas: dict) -> list:
    out = []
    for md, p in zip(mds, probs):
        a = np.array([alphas.get(f"{stage(md, c)[0]}/{stage(md, c)[1]}", alphas["all"]) for c in md.cps])
        out.append(1 / (1 + np.exp(-a * calibrate.logit(p))))
    return out


# ---------------------------------------------------------------------------- moments

def moments(mds: list, min_points: int = 40) -> dict:
    """Shared serve deviations within a match, in logit units r = (k/n - p) / (p (1-p)):
    within a stream (first vs second half of a player's serve points: what persists) and
    across streams (A's serve vs B's serve over the whole match). Centred covariances."""
    within, cross = [], []
    for md in mds:
        if not md.cps:
            continue
        last = md.cps[-1]
        r = []
        for x, k, n in ((md.x1, last.k1, last.n1), (md.x2, last.k2, last.n2)):
            p = 1 / (1 + math.exp(-x))
            r.append((k / n - p) / (p * (1 - p)) if n else None)
        if all(v is not None for v in r) and min(last.n1, last.n2) >= min_points:
            cross.append((r[0], r[1], md.v1 + md.v2))
    # halves need the point sequence; approximate with the checkpoint at mid-match
    for md in mds:
        if len(md.cps) < 4:
            continue
        mid, last = md.cps[len(md.cps) // 2], md.cps[-1]
        for x, k0, n0, k1, n1, v in ((md.x1, mid.k1, mid.n1, last.k1, last.n1, md.v1),
                                     (md.x2, mid.k2, mid.n2, last.k2, last.n2, md.v2)):
            na, nb = n0, n1 - n0
            if min(na, nb) < min_points // 2:
                continue
            p = 1 / (1 + math.exp(-x))
            q = p * (1 - p)
            within.append(((k0 / na - p) / q, ((k1 - k0) / nb - p) / q, v))
    out = {}
    for name, rows in (("within (first vs second half)", within), ("cross (A serve vs B serve)", cross)):
        d = np.array(rows)
        a, b = d[:, 0] - d[:, 0].mean(), d[:, 1] - d[:, 1].mean()
        prod = a * b
        out[name] = {"n": len(d), "cov": float(prod.mean()), "se": float(prod.std(ddof=1) / math.sqrt(len(d))),
                     "mean_v": float(d[:, 2].mean())}
    return out
