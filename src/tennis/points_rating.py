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
- **Result only** (no stats and no usable score, or score_weight = 0; e.g. before 1991, WTA before ~2010, most lower levels,
  AcesAPI): extended Kalman update on the Bernoulli result with the Markov gradient H_j:
      S = sum(H_j^2 v_j) + P (1 - P) / w,   theta_j += v_j H_j (y - P) / S,
      v_j -= v_j^2 H_j^2 / S
  w = nostat_weight. With stats, the same result update runs afterwards with weight
  result_weight (0 = points only).
- **Score** (no stats, but a usable score; score_weight > 0): Newton step on the
  log-likelihood of the set scores (markov.set_score_dist: 6-0..7-6 per set, advantage
  sets, match tiebreaks), summed over who served the first set (later sets follow from
  game parity). With prior covariance V = diag(var x_A, var x_B) and Fisher information
  I of the sets (markov.ScoreGrid, averaged over the first server):
      delta_x = (V^-1 + w I)^-1 w grad log L,   w = score_weight
  spread over the four terms of each x in proportion to their variances. Replaces the
  result update for those matches (the score implies the result). parse_score() decides
  usability; retirements keep their completed sets.
- Retirements scale w by retired_weight (0 skips them); walkovers and unfinished matches
  never update.

Context offsets (milestone 9; context.py): with cs_*/cr_* parameters, each serve-point
logit gets x_A += beta_serve . c_A - beta_ret . c_B (rust after a layoff, recent load, age)
BEFORE the roll-up. The adjusted x is used for the prediction and for every update tier,
so a player who performs as their context predicts is not re-rated; the offset itself is
never stored in the state. rust_weight scales the update weight of matches where either
player is 1-5 matches back from a > 12-week break. All zero = runs 605/694 exactly.
matchup() (live predictions) does not apply context yet.

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

from . import context, markov

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
    score_weight: float = 0.0     # weight of the score update when there are no stats
                                  # (0 = off: result-only fallback, as in run 605)
    # context offsets (context.FEATURES): cs_* on the player's own serve logit, cr_* on
    # their own return; all 0 = no context (runs 605/694). Used for prediction AND update.
    cs_back1: float = 0.0
    cs_back2_5: float = 0.0
    cs_gap5_12: float = 0.0
    cs_gap1: float = 0.0
    cs_load28: float = 0.0
    cs_age_lt22: float = 0.0
    cs_age_lt26: float = 0.0
    cs_age_gt28: float = 0.0
    cs_age_gt32: float = 0.0
    cr_back1: float = 0.0
    cr_back2_5: float = 0.0
    cr_gap5_12: float = 0.0
    cr_gap1: float = 0.0
    cr_load28: float = 0.0
    cr_age_lt22: float = 0.0
    cr_age_lt26: float = 0.0
    cr_age_gt28: float = 0.0
    cr_age_gt32: float = 0.0
    # per-tour context: with ctx_per_tour = 1, WTA matches use the *_w coefficients and
    # ATP matches the plain ones; 0 = the plain ones for both tours
    ctx_per_tour: float = 0.0
    cs_back1_w: float = 0.0
    cs_back2_5_w: float = 0.0
    cs_gap5_12_w: float = 0.0
    cs_gap1_w: float = 0.0
    cs_load28_w: float = 0.0
    cs_age_lt22_w: float = 0.0
    cs_age_lt26_w: float = 0.0
    cs_age_gt28_w: float = 0.0
    cs_age_gt32_w: float = 0.0
    cr_back1_w: float = 0.0
    cr_back2_5_w: float = 0.0
    cr_gap5_12_w: float = 0.0
    cr_gap1_w: float = 0.0
    cr_load28_w: float = 0.0
    cr_age_lt22_w: float = 0.0
    cr_age_lt26_w: float = 0.0
    cr_age_gt28_w: float = 0.0
    cr_age_gt32_w: float = 0.0
    rust_weight: float = 1.0      # update weight of a match where a player is 1-5 matches
                                  # back from a > 12-week break (1 = full weight)
    blend: float = 1.0           # prediction: multiplier on the surface deviations
    unc: float = 1.0              # prediction: uncertainty shrink (0 = off)
    # registry compatibility (standings, Elo/Glicko-style fields); unused here
    init: float = 0.0
    surface_init: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)

    def update_key(self) -> "PointsParams":
        return replace(self, blend=1.0, unc=1.0)

    def context_betas(self, tour: str = "atp") -> tuple[np.ndarray, np.ndarray]:
        """(beta_serve, beta_ret) aligned with context.FEATURES for a tour's matches."""
        sfx = "_w" if tour == "wta" and self.ctx_per_tour else ""
        bs = np.array([getattr(self, f"cs_{f}{sfx}") for f in context.FEATURES])
        br = np.array([getattr(self, f"cr_{f}{sfx}") for f in context.FEATURES])
        return bs, br

    def has_context(self) -> bool:
        return any(b.any() for t in ("atp", "wta") for b in self.context_betas(t))

    def with_context(self, beta_serve, beta_ret, tour: str | None = None) -> "PointsParams":
        """Set the coefficients: of both tours (tour None, shared), or of one tour's set."""
        sfx = "_w" if tour == "wta" else ""
        kw = {f"cs_{f}{sfx}": float(b) for f, b in zip(context.FEATURES, beta_serve)}
        kw.update({f"cr_{f}{sfx}": float(b) for f, b in zip(context.FEATURES, beta_ret)})
        return replace(self, **kw)


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
_END_MARKS = {"RET", "ret.", "DEF", "Def.", "W/O", "w/o", "ABD", "abandoned"}
# set types for the score tier: 0 = tiebreak set, 1 = advantage set, 2 = match tiebreak
_FINAL_TYPE = {"tb": 0, "adv": 1, "tb12": 1, "mtb": 2}


def _set_category(a: int, b: int, typ: int) -> int | None:
    """markov.SET_CATS category of a set the match winner scored a-b (None: not a
    complete set of that type). Type 2 (match tiebreak): 0 = winner took it, 1 = lost it."""
    if typ == 2:
        if (max(a, b) >= 10 and abs(a - b) >= 2) or (max(a, b), min(a, b)) == (1, 0):
            return 0 if a > b else 1
        return None
    hi, lo = max(a, b), min(a, b)
    if hi == 6 and lo <= 4:
        c = lo
    elif (hi, lo) == (7, 5):
        c = 5
    elif typ == 0 and (hi, lo) == (7, 6):
        c = 6
    elif typ == 1 and hi >= 8 and hi - lo == 2:
        c = 6
    else:
        return None
    return c if a > b else markov.mirror(c)


def parse_score(score: str | None, best_of: int, final: str, status: str) -> tuple | None:
    """Sets of a winner-first score as ((type, category), ...) for the score tier, or None
    when the score cannot be used (missing, unknown format, inconsistent with the result).
    Sets before the deciding one are tiebreak sets; the deciding set follows `final`.
    Retirements keep their completed sets and drop the unfinished last one."""
    if not score:
        return None
    toks = score.split()
    while toks and toks[-1] in _END_MARKS:
        toks.pop()
    out = []
    for k, t in enumerate(toks):
        mt = _SET.match(t)
        if not mt:
            return None
        typ = _FINAL_TYPE[final] if k == best_of - 1 else 0
        c = _set_category(int(mt.group(1)), int(mt.group(2)), typ)
        if c is None:
            if status == "retired" and k == len(toks) - 1:
                break                            # the set in progress at the retirement
            return None
        out.append((typ, c))
    if not out or len(out) > best_of:
        return None
    if status == "completed":
        won = sum(1 for typ, c in out if (c == 0 if typ == 2 else c < 7))
        if won != best_of // 2 + 1 or len(out) - won >= won:
            return None
    return tuple(out)


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
    """Load best-of, deciding-set rule and serve stats aligned with `m` (cached on m.extra),
    and the parsed scores for the score tier (m.extra['scores'])."""
    if "points" in m.extra:
        if "scores" not in m.extra:
            attach_scores(con, m, alias)
        context.attach(con, m, alias)
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
    attach_scores(con, m, alias)
    context.attach(con, m, alias)
    return d


def attach_scores(con: sqlite3.Connection, m, alias: str = "data") -> list:
    """Parsed scores aligned with `m` (parse_score per match; None = not usable), using the
    formats already in m.extra['points']. Kept apart from PointsData so blackout() leaves
    scores in place: hiding stats leaves the score tier to do the work."""
    d = m.extra["points"]
    raw = dict(con.execute(f"SELECT match_id, score FROM {alias}.matches"))
    out = [parse_score(raw.get(mid), *FORMATS[d.fmt[i]], m.status[i])
           for i, mid in enumerate(m.match_id)]
    m.extra["scores"] = out
    return out


def blackout(m, d: PointsData, after: str) -> PointsData:
    """Copy of d with serve stats removed for every match whose event starts on or after
    `after` (ISO date), so those matches take the no-stats update. Simulates losing the
    stats source on that date (the stats blackout test)."""
    out = PointsData(bytearray(d.fmt), array("l", d.w_sp), array("l", d.w_sw),
                     array("l", d.l_sp), array("l", d.l_sw), d.rule_counts)
    for i, start in enumerate(m.start):
        if start >= after:
            out.w_sp[i] = out.w_sw[i] = out.l_sp[i] = out.l_sw[i] = -1
    return out


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
    scw = p.score_weight
    scores = m.extra.get("scores") if scw > 0.0 else None
    if scw > 0.0 and scores is None:
        raise ValueError("score_weight > 0 needs parsed scores: call attach() first")
    sgrids = (markov.score_grid("tb"), markov.score_grid("adv"))
    mtb = markov.grid(1, "mtb") if scw > 0.0 else None
    odd = (markov.CAT_GAMES_ODD, markov.CAT_GAMES_ODD_ADV)
    log = math.log
    # context offsets on x_a / x_b (prediction and update; never stored in the state)
    rustw = p.rust_weight
    if p.has_context() or rustw != 1.0:
        C = m.extra.get("context")
        if C is None:
            raise ValueError("context parameters need context features: call context.attach() first")
        oa_, ob_ = context.offsets(C, *p.context_betas("atp"))
        if p.ctx_per_tour:
            wa, wb = context.offsets(C, *p.context_betas("wta"))
            is_w = np.array([t == "wta" for t in m.tour[:n]] + [False] * (len(oa_) - n))
            oa_, ob_ = np.where(is_w, wa, oa_), np.where(is_w, wb, ob_)
        OA, OB = oa_[:n].tolist(), ob_[:n].tolist()
        rusty = ((C[0][:n, 0] + C[0][:n, 1] + C[1][:n, 0] + C[1][:n, 1]) > 0).tolist()
    else:
        OA = OB = rusty = None

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
        if OA is not None:
            oa, ob = OA[i], OB[i]
            xa0 += oa
            xb0 += ob
        else:
            oa = ob = 0.0
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
        if rusty is not None and rusty[i]:
            weight *= rustw
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
                xa = b + sw_ - rl_ + dsw - drl + oa
                xb = b + sl_ - rw_ + dsl - drw + ob
        elif scores is not None and scores[i] is not None:
            # score tier: Newton step on the log-likelihood of the set scores (winner = A),
            # summed over who served first in set 1; Fisher information from the set tables
            rweight = 0.0
            sets = scores[i]
            iaa = iab = ibb = 0.0
            for typ, c in sets:
                if typ == 2:
                    pm, dpa, dpb = mtb.lookup(xa, xb)
                    pm = min(max(pm, 1e-9), 1 - 1e-9)
                    v_ = pm * (1 - pm)
                    iaa += dpa * dpa / v_
                    iab += dpa * dpb / v_
                    ibb += dpb * dpb / v_
                else:
                    e1, e2, e3 = sgrids[typ].information(xa, xb)
                    iaa += e1
                    iab += e2
                    ibb += e3
            lps, gas, gbs = [], [], []
            for first_a in (True, False):
                lp = gsa = gsb = 0.0
                a_serves = first_a
                for typ, c in sets:
                    if typ == 2:
                        pm, dpa, dpb = mtb.lookup(xa, xb)
                        if c == 0:
                            pp, g1, g2 = pm, dpa, dpb
                        else:
                            pp, g1, g2 = 1 - pm, -dpa, -dpb
                    else:
                        if a_serves:
                            pp, g1, g2 = sgrids[typ].lookup(xa, xb, c)
                        else:
                            pp, g2, g1 = sgrids[typ].lookup(xb, xa, markov.mirror(c))
                        if odd[typ][c]:
                            a_serves = not a_serves
                    pp = max(pp, 1e-300)
                    lp += log(pp)
                    gsa += g1 / pp
                    gsb += g2 / pp
                lps.append(lp)
                gas.append(gsa)
                gbs.append(gsb)
            mx = max(lps)
            w0, w1 = math.exp(lps[0] - mx), math.exp(lps[1] - mx)
            gsa = (w0 * gas[0] + w1 * gas[1]) / (w0 + w1)
            gsb = (w0 * gbs[0] + w1 * gbs[1]) / (w0 + w1)
            sw8 = scw * weight
            va = vsw + dvsw + vrl + dvrl
            vb = vsl + dvsl + vrw + dvrw
            a11, a12, a22 = 1.0 / va + sw8 * iaa, sw8 * iab, 1.0 / vb + sw8 * ibb
            det = a11 * a22 - a12 * a12
            c11, c12, c22 = a22 / det, -a12 / det, a11 / det
            dxa = sw8 * (c11 * gsa + c12 * gsb)
            dxb = sw8 * (c12 * gsa + c22 * gsb)
            # spread the change in x_a over its four terms in proportion to their variances
            fa, fb = dxa / va, dxb / vb
            sw_ += vsw * fa
            dsw += dvsw * fa
            rl_ -= vrl * fa
            drl -= dvrl * fa
            sl_ += vsl * fb
            dsl += dvsl * fb
            rw_ -= vrw * fb
            drw -= dvrw * fb
            ra, rb = (va - c11) / (va * va), (vb - c22) / (vb * vb)
            vsw -= vsw * vsw * ra
            dvsw -= dvsw * dvsw * ra
            vrl -= vrl * vrl * ra
            dvrl -= dvrl * dvrl * ra
            vsl -= vsl * vsl * rb
            dvsl -= dvsl * dvsl * rb
            vrw -= vrw * vrw * rb
            dvrw -= dvrw * dvrw * rb
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


def matchup(state: PointsState, a: int, b: int, tour: str, surface: str | None, on: str):
    """(x_a, x_b, var_a, var_b) for A vs B on ISO date `on` from a run's final state:
    each serve-point logit with the run's surface blend, variances grown by inactivity
    to `on` (the same arithmetic run() uses before a match). A player never seen starts
    as a new player."""
    p, day = state.params, date.fromisoformat(on).toordinal()

    def part(i):
        if i in state.serve:
            el = max(day - state.last[i], 0)
            s, r = state.serve[i], state.ret[i]
            vs, vr = min(state.vs[i] + p.q * el, p.v_init), min(state.vr[i] + p.q * el, p.v_init)
        else:
            s = r = -p.init_offset
            vs = vr = p.v_init
        k = (i, surface)
        if surface is None:
            return s, r, vs, vr, 0.0, 0.0, 0.0, 0.0
        if k in state.sserve:
            el = max(day - state.slast[k], 0)
            return (s, r, vs, vr, state.sserve[k], state.sret[k],
                    min(state.svs[k] + p.q_surf * el, p.surf_var), min(state.svr[k] + p.q_surf * el, p.surf_var))
        return s, r, vs, vr, 0.0, 0.0, p.surf_var, p.surf_var

    sa, ra, vsa, vra, dsa, dra, dvsa, dvra = part(a)
    sb, rb, vsb, vrb, dsb, drb, dvsb, dvrb = part(b)
    base = state.base.get((tour, surface), BASE_INIT.get(tour, 0.4))
    xa = base + sa - rb + p.blend * (dsa - drb)
    xb = base + sb - ra + p.blend * (dsb - dra)
    va = vsa + vrb + p.blend * (dvsa + dvrb)
    vb = vsb + vra + p.blend * (dvsb + dvra)
    return xa, xb, va, vb


def win_probability(state: PointsState, xa: float, xb: float, va: float, vb: float,
                    best_of: int = 3, final: str = "tb") -> float:
    """P(A wins) exactly as the model predicts it (Markov roll-up + uncertainty shrink)."""
    pr, da, db = markov.grid(best_of, final).lookup_array(np.array([xa]), np.array([xb]), grad=True)
    return float(_shrink(pr, da, db, np.array([va]), np.array([vb]), state.params.unc)[0])
