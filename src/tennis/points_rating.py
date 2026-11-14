"""Points-based serve/return ratings with a Markov roll-up to match probability (`points_v1`).

State per player: a serve rating s and a return rating r (logit scale of a serve point),
each with a variance; per (player, surface) a serve and a return deviation ds, dr with
their own variances (a hierarchical surface effect, prior mean 0). Per (tour, surface) a
base logit b (the average serve-point win rate), learned online.

    x_A = logit P(A wins a point on A's serve) = b + s_A + ds_A - r_B - dr_B
    x_B = b + s_B + ds_B - r_A - dr_A
    P(A wins) = Markov roll-up of (x_A, x_B) with the match's best-of and deciding-set rule
                (markov.MatchGrid; the rule is inferred per edition from its scores)

Variances grow with time since the player's last rated match (q per day, capped at
v_init); surface variances with time since their last match on that surface (q_surf per
day, capped at surf_var).

Updates, from the pre-match state:
- **Serve/return points** (match_player_stats has both players' serve points and points
  won): each server's point record is a noisy observation of x. In logit space,
      z = (k/n - p) / (p (1 - p)),   R = 1 / (n p (1 - p)) + obs_var,   p = sigmoid(x)
  obs_var is the extra match-to-match noise (form, conditions) beyond binomial sampling.
  Kalman update of the four terms in x (independent, as in Glicko):
      S = sum(v_j) + R / w,   theta_j += sign_j v_j z / S,   v_j -= v_j^2 / S
  The two observations (A serving, B serving) involve disjoint terms. b moves by base_lr * z.
- **Result only** (no stats, e.g. before 1991, WTA before ~2010, most lower levels,
  AcesAPI): extended Kalman update on the Bernoulli result with the Markov gradient H_j:
      S = sum(H_j^2 v_j) + P (1 - P) / w,   theta_j += v_j H_j (y - P) / S,
      v_j -= v_j^2 H_j^2 / S
  w = nostat_weight. With stats, the same result update runs afterwards with weight
  result_weight (0 = points only).
- Retirements scale w by retired_weight (0 skips them); walkovers and unfinished matches
  never update.

Prediction (prediction-only parameters, scored from one pass):
    blend   multiplier on the surface deviations (and their variances); 1 = full model
    unc     uncertainty shrink: logit P / sqrt(1 + unc * 3 var(logit P) / pi^2), with
            var(logit P) from the gradient and the variances of x_A, x_B (Glicko's g()).

Reported "ratings" (r_/s_ arrays, state.rating) are s + r (overall) and s + r + ds + dr
(on the match surface), in serve-logit units.

Pure logic: no database access except attach().
"""
from __future__ import annotations

import math
import re
import sqlite3
from array import array
from dataclasses import asdict, dataclass, replace
from datetime import date

import numpy as np

from . import markov

NAN = float("nan")
_PI2 = math.pi ** 2
# Starting base logits (serve-point win rate ~0.63 ATP, ~0.56 WTA); learned online after.
BASE_INIT = {"atp": 0.53, "wta": 0.24}
FORMATS = [(bo, f) for bo in (1, 3, 5) for f in markov.FINALS]
FORMAT_INDEX = {k: i for i, k in enumerate(FORMATS)}


@dataclass(frozen=True)
class PointsParams:
    v_init: float = 0.04          # variance of a new player's s and r (also the cap)
    q: float = 2e-5               # variance growth per day of inactivity
    obs_var: float = 0.02         # extra per-match variance of a serve-point logit
    surf_var: float = 0.01        # prior variance of a surface deviation (also the cap)
    q_surf: float = 2e-6          # surface-deviation variance growth per day off that surface
    init_offset: float = 0.0      # new players start at s = r = -init_offset
    nostat_weight: float = 1.0    # weight of the result update when there are no stats
    result_weight: float = 0.0    # weight of the extra result update when there are stats
    retired_weight: float = 1.0   # weight multiplier for retirements; 0 = ignore them
    base_lr: float = 0.0005       # learning rate of the (tour, surface) base logit
    blend: float = 1.0            # prediction: multiplier on the surface deviations
    unc: float = 1.0              # prediction: uncertainty shrink (0 = off)
    # registry compatibility (standings, Elo/Glicko-style fields); unused here
    init: float = 0.0
    surface_init: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)

    def update_key(self) -> "PointsParams":
        return replace(self, blend=1.0, unc=1.0)


# ---------------------------------------------------------------------------- data

@dataclass
class PointsData:
    """Per-match extras aligned with elo.Matches (same order, same length)."""
    fmt: bytearray          # FORMAT_INDEX of (best_of, deciding-set rule)
    w_sp: array             # winner's serve points (-1 = no usable stats)
    w_sw: array             # winner's serve points won
    l_sp: array
    l_sw: array
    rule_counts: dict       # diagnostics: edition rules, how inferred


_SET = re.compile(r"^\[?(\d+)-(\d+)\]?(\(\d+\))?$")


def deciding_set_evidence(score: str | None, best_of: int | None) -> str | None:
    """What one completed match's score says about the edition's deciding-set rule:
    'mtb' / 'tb12' / 'adv' / 'tb', or None (no deciding set, or unreadable)."""
    if not score or best_of not in (3, 5):
        return None
    sets = score.split()
    if len(sets) != best_of:
        return None
    m = _SET.match(sets[-1])
    if not m:
        return None
    a, b = int(m.group(1)), int(m.group(2))
    hi, lo = max(a, b), min(a, b)
    if sets[-1].startswith("[") or (hi >= 10 and hi - lo >= 2 and lo <= 7) or (hi, lo) == (1, 0):
        return "mtb"
    if (hi, lo) == (13, 12):
        return "tb12"
    if hi >= 8 and hi - lo == 2:
        return "adv"
    if (hi, lo) == (7, 6):
        return "tb"
    return None


def infer_rules(rows) -> tuple[dict, dict]:
    """rows: (edition_id, tournament_id, start_date, score, best_of) for completed matches.
    Rule per edition: strongest evidence in its own scores (mtb > tb12 > adv > tb; the
    rule is announced before the event, so this is not outcome leakage). An edition
    without evidence takes its tournament's latest earlier inferred rule, else 'tb'."""
    rank = {"tb": 0, "adv": 1, "tb12": 2, "mtb": 3}
    ev: dict[int, str] = {}
    meta: dict[int, tuple] = {}
    for eid, tid, start, score, bo in rows:
        meta.setdefault(eid, (start, tid))
        e = deciding_set_evidence(score, bo)
        if e and (eid not in ev or rank[e] > rank[ev[eid]]):
            ev[eid] = e
    out, last, counts = {}, {}, {}
    for eid in sorted(meta, key=lambda e: meta[e]):
        start, tid = meta[eid]
        if eid in ev:
            rule, how = ev[eid], "evidence"
        elif tid in last:
            rule, how = last[tid], "tournament_history"
        else:
            rule, how = "tb", "default"
        out[eid] = rule
        last[tid] = rule
        counts[(rule, how)] = counts.get((rule, how), 0) + 1
    return out, counts


def attach(con: sqlite3.Connection, m, alias: str = "data") -> PointsData:
    """Load best-of, deciding-set rule and serve stats aligned with `m` (cached on m.extra)."""
    if "points" in m.extra:
        return m.extra["points"]
    rule, counts = infer_rules(con.execute(
        f"""SELECT m.edition_id, e.tournament_id, e.start_date, m.score, m.best_of
            FROM {alias}.matches m JOIN {alias}.tournament_edition e USING (edition_id)
            WHERE m.status = 'completed'"""))
    info = {mid: (eid, bo) for mid, eid, bo in con.execute(
        f"SELECT match_id, edition_id, best_of FROM {alias}.matches")}
    stats: dict[int, dict[int, tuple[int, int]]] = {}
    for mid, w, sp, fw, sw in con.execute(
            f"""SELECT match_id, is_winner, serve_points, first_won, second_won
                FROM {alias}.match_player_stats
                WHERE serve_points > 0 AND first_won IS NOT NULL AND second_won IS NOT NULL
                  AND first_won + second_won <= serve_points"""):
        stats.setdefault(mid, {})[w] = (sp, fw + sw)
    n = len(m)
    d = PointsData(bytearray(n), array("l", [-1]) * n, array("l", [-1]) * n,
                   array("l", [-1]) * n, array("l", [-1]) * n, counts)
    for i, mid in enumerate(m.match_id):
        eid, bo = info[mid]
        bo = bo if bo in (1, 3, 5) else 3
        d.fmt[i] = FORMAT_INDEX[(bo, rule.get(eid, "tb"))]
        st = stats.get(mid)
        if st and 0 in st and 1 in st:
            d.w_sp[i], d.w_sw[i] = st[1]
            d.l_sp[i], d.l_sw[i] = st[0]
    m.extra["points"] = d
    return d


# ---------------------------------------------------------------------------- model

@dataclass
class PointsResult:
    p_winner: array        # pre-match P(actual winner wins) at the run's blend/unc
    r_winner: array        # pre-match s + r (overall)
    r_loser: array
    s_winner: array        # pre-match s + r + ds + dr on the match surface (NaN: no surface)
    s_loser: array
    d_winner: array        # pre-match sqrt(var s + var r)
    d_loser: array
    n_winner: array
    n_loser: array
    # for predict(): x = base part + surface part; variances likewise. a = winner serving.
    xa: np.ndarray
    xa_s: np.ndarray
    va: np.ndarray
    va_s: np.ndarray
    xb: np.ndarray
    xb_s: np.ndarray
    vb: np.ndarray
    vb_s: np.ndarray
    fmt: np.ndarray
    state: "PointsState | None" = None


@dataclass
class PointsState:
    params: PointsParams
    serve: dict            # player_id -> s
    ret: dict              # player_id -> r
    vs: dict
    vr: dict
    last: dict             # player_id -> date ordinal of last rated match
    count: dict
    sserve: dict           # (player_id, surface) -> ds
    sret: dict
    svs: dict
    svr: dict
    slast: dict
    scount: dict
    base: dict             # (tour, surface) -> b

    @property
    def rating(self) -> dict:
        return {k: v + self.ret[k] for k, v in self.serve.items()}

    @property
    def srating(self) -> dict:
        return {k: v + self.sret[k] + self.serve[k[0]] + self.ret[k[0]] for k, v in self.sserve.items()}

    def rd_at(self, pid: int, on: str) -> float:
        p, el = self.params, max(date.fromisoformat(on).toordinal() - self.last[pid], 0)
        return math.sqrt(min(self.vs[pid] + p.q * el, p.v_init) + min(self.vr[pid] + p.q * el, p.v_init))


def _shrink(p, da, db, va, vb, unc):
    """Glicko-style shrink of a match probability toward 0.5 for rating uncertainty."""
    if not unc:
        return p
    p = np.clip(p, 1e-12, 1 - 1e-12)
    w = p * (1 - p)
    var_l = ((da / w) ** 2 * va + (db / w) ** 2 * vb)
    lg = np.log(p / (1 - p)) / np.sqrt(1 + unc * 3 * var_l / _PI2)
    return 1 / (1 + np.exp(-lg))


def predict(res: PointsResult, p: PointsParams) -> array:
    """P(actual winner wins) per match at p.blend / p.unc, from stored pre-match values."""
    xa = res.xa + p.blend * res.xa_s
    xb = res.xb + p.blend * res.xb_s
    va = res.va + p.blend * res.va_s
    vb = res.vb + p.blend * res.vb_s
    out = np.empty(len(xa))
    for k in np.unique(res.fmt):
        idx = np.flatnonzero(res.fmt == k)
        g = markov.grid(*FORMATS[k])
        pr, da, db = g.lookup_array(xa[idx], xb[idx], grad=True)
        out[idx] = _shrink(pr, da, db, va[idx], vb[idx], p.unc)
    return array("d", out.tolist())


def run(m, p: PointsParams) -> PointsResult:
    d: PointsData = m.extra["points"]
    n = len(m)
    floats = lambda: array("d", bytes(8 * n))                   # noqa: E731
    ints = lambda: array("l", bytes(array("l").itemsize * n))   # noqa: E731
    vec = lambda: np.zeros(n)                                   # noqa: E731
    out = PointsResult(*(floats() for _ in range(7)), ints(), ints(), *(vec() for _ in range(8)),
                       np.frombuffer(bytes(d.fmt[:n]), dtype=np.uint8).astype(np.int64))
    grids = {k: markov.grid(*FORMATS[k]) for k in set(d.fmt[:n])}   # only formats present
    v0, q, ov, sv0, qs = p.v_init, p.q, p.obs_var, p.surf_var, p.q_surf
    init, nsw, rsw, rtw, blr = -p.init_offset, p.nostat_weight, p.result_weight, p.retired_weight, p.base_lr
    S, R, VS, VR, LAST, CNT = {}, {}, {}, {}, {}, {}
    DS, DR, DVS, DVR, SLAST, SCNT = {}, {}, {}, {}, {}, {}
    base: dict = {}
    days: dict[str, int] = {}
    w_sp, w_sw, l_sp, l_sw = d.w_sp, d.w_sw, d.l_sp, d.l_sw
    sig = lambda x: 1.0 / (1.0 + math.exp(-x))                   # noqa: E731

    for i in range(n):
        start = m.start[i]
        day = days.get(start)
        if day is None:
            day = days[start] = date.fromisoformat(start).toordinal()
        w, l, surf, tour = m.winner[i], m.loser[i], m.surface[i], m.tour[i]
        # pre-match state, variances inflated to today
        if w in S:
            el = day - LAST[w]
            sw_, rw_ = S[w], R[w]
            vsw, vrw = min(VS[w] + q * el, v0), min(VR[w] + q * el, v0)
        else:
            sw_ = rw_ = init
            vsw = vrw = v0
        if l in S:
            el = day - LAST[l]
            sl_, rl_ = S[l], R[l]
            vsl, vrl = min(VS[l] + q * el, v0), min(VR[l] + q * el, v0)
        else:
            sl_ = rl_ = init
            vsl = vrl = v0
        if surf is not None:
            kw, kl = (w, surf), (l, surf)
            if kw in DS:
                el = day - SLAST[kw]
                dsw, drw = DS[kw], DR[kw]
                dvsw, dvrw = min(DVS[kw] + qs * el, sv0), min(DVR[kw] + qs * el, sv0)
            else:
                dsw = drw = 0.0
                dvsw = dvrw = sv0
            if kl in DS:
                el = day - SLAST[kl]
                dsl, drl = DS[kl], DR[kl]
                dvsl, dvrl = min(DVS[kl] + qs * el, sv0), min(DVR[kl] + qs * el, sv0)
            else:
                dsl = drl = 0.0
                dvsl = dvrl = sv0
        else:
            dsw = drw = dsl = drl = dvsw = dvrw = dvsl = dvrl = 0.0
        bkey = (tour, surf)
        b = base.get(bkey)
        if b is None:
            b = base[bkey] = BASE_INIT.get(tour, 0.4)
        xa0, xa1 = b + sw_ - rl_, dsw - drl          # winner serving
        xb0, xb1 = b + sl_ - rw_, dsl - drw          # loser serving
        va0, va1 = vsw + vrl, dvsw + dvrl
        vb0, vb1 = vsl + vrw, dvsl + dvrw
        out.xa[i], out.xa_s[i], out.va[i], out.va_s[i] = xa0, xa1, va0, va1
        out.xb[i], out.xb_s[i], out.vb[i], out.vb_s[i] = xb0, xb1, vb0, vb1
        out.r_winner[i], out.r_loser[i] = sw_ + rw_, sl_ + rl_
        out.d_winner[i], out.d_loser[i] = math.sqrt(vsw + vrw), math.sqrt(vsl + vrl)
        if surf is not None:
            out.s_winner[i] = sw_ + rw_ + dsw + drw
            out.s_loser[i] = sl_ + rl_ + dsl + drl
        else:
            out.s_winner[i] = out.s_loser[i] = NAN
        out.n_winner[i], out.n_loser[i] = CNT.get(w, 0), CNT.get(l, 0)

        status = m.status[i]
        if status == "walkover" or status == "unfinished":
            continue
        weight = rtw if status == "retired" else 1.0
        if weight <= 0.0:
            continue
        xa, xb = xa0 + xa1, xb0 + xb1
        grid = grids[d.fmt[i]]
        if w_sp[i] > 0:
            # winner serving: terms s_w, ds_w (+), r_l, dr_l (-)
            na, ka = w_sp[i], w_sw[i]
            pa = sig(xa)
            ia = na * pa * (1 - pa)
            za = (ka - na * pa) / ia
            sa = vsw + dvsw + vrl + dvrl + (1.0 / ia + ov) / weight
            sw_ += vsw * za / sa
            dsw += dvsw * za / sa
            rl_ -= vrl * za / sa
            drl -= dvrl * za / sa
            vsw -= vsw * vsw / sa
            dvsw -= dvsw * dvsw / sa
            vrl -= vrl * vrl / sa
            dvrl -= dvrl * dvrl / sa
            # loser serving: terms s_l, ds_l (+), r_w, dr_w (-)
            nb, kb = l_sp[i], l_sw[i]
            pb = sig(xb)
            ib = nb * pb * (1 - pb)
            zb = (kb - nb * pb) / ib
            sb = vsl + dvsl + vrw + dvrw + (1.0 / ib + ov) / weight
            sl_ += vsl * zb / sb
            dsl += dvsl * zb / sb
            rw_ -= vrw * zb / sb
            drw -= dvrw * zb / sb
            vsl -= vsl * vsl / sb
            dvsl -= dvsl * dvsl / sb
            vrw -= vrw * vrw / sb
            dvrw -= dvrw * dvrw / sb
            base[bkey] = b + blr * (za + zb)
            rweight = rsw * weight
            if rweight > 0.0:
                xa = b + sw_ - rl_ + dsw - drl
                xb = b + sl_ - rw_ + dsl - drw
        else:
            rweight = nsw * weight
        if rweight > 0.0:
            # extended Kalman update on the result (winner: y = 1)
            pr, ha, hb = grid.lookup(xa, xb)
            pr = min(max(pr, 1e-9), 1 - 1e-9)
            # gradients: x_a = .. + s_w + ds_w - r_l - dr_l ; x_b = .. + s_l + ds_l - r_w - dr_w
            ss = (ha * ha * (vsw + dvsw + vrl + dvrl) + hb * hb * (vsl + dvsl + vrw + dvrw)
                  + pr * (1 - pr) / rweight)
            e = (1 - pr) / ss
            sw_ += vsw * ha * e
            dsw += dvsw * ha * e
            rl_ -= vrl * ha * e
            drl -= dvrl * ha * e
            sl_ += vsl * hb * e
            dsl += dvsl * hb * e
            rw_ -= vrw * hb * e
            drw -= dvrw * hb * e
            ha2, hb2 = ha * ha / ss, hb * hb / ss
            vsw -= vsw * vsw * ha2
            dvsw -= dvsw * dvsw * ha2
            vrl -= vrl * vrl * ha2
            dvrl -= dvrl * dvrl * ha2
            vsl -= vsl * vsl * hb2
            dvsl -= dvsl * dvsl * hb2
            vrw -= vrw * vrw * hb2
            dvrw -= dvrw * dvrw * hb2
        S[w], R[w], VS[w], VR[w], LAST[w] = sw_, rw_, vsw, vrw, day
        S[l], R[l], VS[l], VR[l], LAST[l] = sl_, rl_, vsl, vrl, day
        CNT[w], CNT[l] = CNT.get(w, 0) + 1, CNT.get(l, 0) + 1
        if surf is not None:
            DS[kw], DR[kw], DVS[kw], DVR[kw], SLAST[kw] = dsw, drw, dvsw, dvrw, day
            DS[kl], DR[kl], DVS[kl], DVR[kl], SLAST[kl] = dsl, drl, dvsl, dvrl, day
            SCNT[kw], SCNT[kl] = SCNT.get(kw, 0) + 1, SCNT.get(kl, 0) + 1

    pw = predict(out, p)
    out.p_winner = pw
    out.state = PointsState(p, S, R, VS, VR, LAST, CNT, DS, DR, DVS, DVR, SLAST, SCNT, base)
    return out
