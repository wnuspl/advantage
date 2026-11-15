"""Context report: do home country, rest, layoffs, in-tournament workload and age move
results away from what points_v1 expects?

    python archive/v1/scripts/context_report.py                 # run 605, tour-level 2005..2026
    python archive/v1/scripts/context_report.py --run 605 --from 2005-01-01

Reads the stored walk-forward predictions of a points_v1 run (models.db) and calibrates
them walk-forward (calibrate.py: per quarter and tour, alpha from the previous 10 years).
Context features for each player come from tennis.db, all levels (qualifying, Challenger,
ITF, Davis Cup count as matches played), strictly from matches before the one predicted:
earlier editions, or earlier rounds of the same edition. Sackmann dates matches only by
their event's start, so all gaps are edition start to edition start.

Three analyses, all on tour-level main-draw matches (the 'tour' scope):
  1. Residuals by bucket: per player-match row, actual win minus calibrated points_v1
     probability, by the player's bucket of each feature. Rows whose opponent is in the
     same bucket are dropped (the two residuals cancel and say nothing about the bucket).
  2. Calibration slope by match context (level, round, best-of, surface): alpha in
     sigmoid(alpha * logit p_cal), 1 = calibrated; <1 = overconfident in that context.
  3. A joint antisymmetric context model, logit P = logit p_cal + beta . (x_A - x_B),
     fit on train, scored on val (2019-21), then refit through 2021 and scored on test
     (2022-26) once. Paired log-loss difference vs calibrated points_v1.

Writes results/context_report.json and reports/context_residuals.png.
Reads only (models.db and tennis.db are not written).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sqlite3
import sys
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np

ARCHIVE = Path(__file__).resolve().parents[1]   # archive/v1: results, reports, docs
ROOT = Path(__file__).resolve().parents[3]      # project root: tennis.db, data, src
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ARCHIVE / "src"))   # archived modules, merged into tennis

from tennis import calibrate, models_db, tuning          # noqa: E402
from tennis.composite_eval import block_of               # noqa: E402

PLAYED = {"completed", "retired"}
SET_RE = re.compile(r"(\d+)-(\d+)")


def games_of(score: str | None) -> int:
    if not score:
        return 0
    g = 0
    for tok in score.split():
        m_ = SET_RE.match(tok)
        if m_:
            g += int(m_.group(1)) + int(m_.group(2))
    return g


def days(a: str, b: str) -> int:
    return (date.fromisoformat(b) - date.fromisoformat(a)).days


# ---------------------------------------------------------------- features
def load_info(con):
    ed = {}
    for eid, start, level, surface, indoor, bo, name, tour in con.execute(
            """SELECT e.edition_id, e.start_date, e.level, e.surface, e.indoor, e.best_of,
                      t.canonical_name, t.tour
               FROM data.tournament_edition e JOIN data.tournament t USING (tournament_id)"""):
        ed[eid] = (start, level, surface, indoor, bo, name, tour)
    mt = {}
    for mid, eid, rnd, ro, score, status, bo, minutes in con.execute(
            "SELECT match_id, edition_id, round, round_order, score, status, best_of, minutes FROM data.matches"):
        mt[mid] = (eid, rnd, ro, score, status, bo, minutes)
    pl = {pid: (c, dob) for pid, c, dob in con.execute("SELECT player_id, country, dob FROM data.player")}
    return ed, mt, pl


def host_map() -> dict:
    out = {}
    with open(ROOT / "data" / "tournament_country.csv", encoding="utf-8") as f:
        rows = [r for r in f if not r.startswith("#")]
    for r in csv.DictReader(rows):
        out[r["canonical_name"]] = r["ioc"]
    return out


class PState:
    __slots__ = ("cur_ed", "cur_start", "cur_n", "cur_games", "cur_min", "cur_min_ok", "last_games",
                 "last_free", "prev_start", "prev_n", "eds", "n_total", "since_return", "first_start")

    def __init__(self):
        self.cur_ed = None; self.cur_start = None; self.cur_n = 0; self.cur_games = 0
        self.cur_min = 0; self.cur_min_ok = True; self.last_games = 0; self.last_free = 0
        self.prev_start = None; self.prev_n = 0
        self.eds = []            # (start, matches played) of finished editions, recent first-out
        self.n_total = 0; self.since_return = 99; self.first_start = None


FEATS = ("home", "gap", "prev_n", "load28", "ed_n", "ed_games", "last_games", "last_free",
         "ed_min", "age", "n_total", "since_return")


def build_features(m, ed, mt, pl, hosts):
    """Per match i: dict of arrays for winner (w) and loser (l), computed before match i."""
    n = len(m)
    F = {side: {k: np.full(n, np.nan) for k in FEATS} for side in ("w", "l")}
    st = defaultdict(PState)
    for i in range(n):
        mid = m.match_id[i]
        eid, rnd, ro, score, status, bo, minutes = mt[mid]
        start, level, surface, indoor, ebo, name, tour = ed[eid]
        host = hosts.get(name)
        g = games_of(score)
        for side, pid in (("w", m.winner[i]), ("l", m.loser[i])):
            s = st[pid]
            if s.cur_ed != eid:                     # new edition for this player: roll over
                if s.cur_ed is not None:
                    s.eds.append((s.cur_start, s.cur_n))
                    if len(s.eds) > 12:
                        s.eds.pop(0)
                    gap_prev = days(s.cur_start, start)
                    s.prev_start, s.prev_n = s.cur_start, s.cur_n
                    if gap_prev > 84:
                        s.since_return = 0
                s.cur_ed, s.cur_start = eid, start
                s.cur_n = 0; s.cur_games = 0; s.cur_min = 0; s.cur_min_ok = True
                s.last_games = 0; s.last_free = 0
                if s.first_start is None:
                    s.first_start = start
            f = F[side]
            country, dob = pl.get(pid, (None, None))
            if host and country:
                f["home"][i] = 1.0 if country == host else 0.0
            if s.prev_start is not None:
                f["gap"][i] = days(s.prev_start, start)
                f["prev_n"][i] = s.prev_n
            f["load28"][i] = sum(k for d0, k in s.eds if 0 < days(d0, start) <= 28)
            f["ed_n"][i] = s.cur_n
            f["ed_games"][i] = s.cur_games
            f["last_games"][i] = s.last_games
            f["last_free"][i] = s.last_free
            if s.cur_min_ok:
                f["ed_min"][i] = s.cur_min
            if dob:
                try:
                    f["age"][i] = days(dob, start) / 365.25
                except ValueError:
                    pass
            f["n_total"][i] = s.n_total
            f["since_return"][i] = s.since_return
        # update both players after the match
        for pid in (m.winner[i], m.loser[i]):
            s = st[pid]
            if status in PLAYED:
                s.cur_n += 1; s.cur_games += g; s.last_games = g
                s.last_free = 1 if (status == "retired" and g < 12) else 0
                if minutes is None:
                    s.cur_min_ok = False
                else:
                    s.cur_min += minutes
                s.n_total += 1
            else:                                   # walkover / default: advanced for free
                s.last_games = 0; s.last_free = 1
        for pid in (m.winner[i], m.loser[i]):
            s = st[pid]
            # matches since return counts matches *played* after a >12-week break
            if status in PLAYED and s.since_return < 99:
                s.since_return += 1
    return F


# ---------------------------------------------------------------- stats helpers
def logit(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def ll_vec(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def fit_offset_logit(X, off, y, l2=1.0, iters=50):
    """logit P = off + X beta, Newton with ridge l2. Returns beta, se."""
    X = np.atleast_2d(X)
    if X.shape[0] != len(y):
        X = X.T
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(off + X @ b)))
        g = X.T @ (y - p) - l2 * b
        H = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.eye(len(b))
        step = np.linalg.solve(H, g)
        b += step
        if np.abs(step).max() < 1e-9:
            break
    p = 1 / (1 + np.exp(-(off + X @ b)))
    H = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.eye(len(b))
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    return b, se


def fit_slope(lg, y):
    """alpha in sigmoid(alpha * lg) with SE."""
    a = 1.0
    for _ in range(50):
        p = 1 / (1 + np.exp(-a * lg))
        g = np.sum(lg * (y - p)); h = np.sum(lg * lg * p * (1 - p))
        a += g / h
        if abs(g / h) < 1e-10:
            break
    p = 1 / (1 + np.exp(-a * lg))
    return float(a), float(1 / math.sqrt(np.sum(lg * lg * p * (1 - p))))


def residual_row(p, y, mask):
    p, y = p[mask], y[mask]
    n = len(p)
    if n == 0:
        return None
    var = np.sum(p * (1 - p))
    res = float(np.mean(y - p))
    se = float(math.sqrt(var) / n)
    # one-parameter logit shift on these rows
    b, bse = fit_offset_logit(np.ones((n, 1)), logit(p), y, l2=0.0)
    return {"n": int(n), "expected": float(p.mean()), "actual": float(y.mean()),
            "resid_pp": 100 * res, "se_pp": 100 * se, "z": res / se if se else float("nan"),
            "logit_shift": float(b[0]), "logit_se": float(bse[0])}


# ---------------------------------------------------------------- buckets
def bucketers():
    def gap(x):
        if np.isnan(x): return "first event"
        if x <= 7: return "1 wk"
        if x <= 14: return "2 wk"
        if x <= 28: return "3-4 wk"
        if x <= 56: return "5-8 wk"
        if x <= 84: return "9-12 wk"
        if x <= 182: return "13-26 wk"
        return "> 26 wk"

    def edn(x):
        return {0: "0 (first match)", 1: "1", 2: "2", 3: "3"}.get(int(x), "4+")

    def edg(x):
        if x == 0: return "0"
        if x < 20: return "1-19"
        if x < 40: return "20-39"
        if x < 60: return "40-59"
        if x < 80: return "60-79"
        return "80+"

    def last(x):
        if x == 0: return "0 / first match"
        if x < 18: return "1-17"
        if x < 24: return "18-23"
        if x < 30: return "24-29"
        if x < 40: return "30-39"
        return "40+"

    def edmin(x):
        if np.isnan(x): return None
        if x == 0: return "0"
        if x < 120: return "<2 h"
        if x < 240: return "2-4 h"
        if x < 360: return "4-6 h"
        return "6 h+"

    def load(x):
        if x == 0: return "0"
        if x <= 2: return "1-2"
        if x <= 4: return "3-4"
        if x <= 7: return "5-7"
        return "8+"

    def prevn(x):
        if np.isnan(x): return None
        return {0: "0", 1: "1", 2: "2", 3: "3"}.get(int(x), "4+")

    def age(x):
        if np.isnan(x): return None
        for hi, lab in ((19, "<19"), (21, "19-20"), (23, "21-22"), (26, "23-25"), (29, "26-28"),
                        (32, "29-31"), (35, "32-34")):
            if x < hi: return lab
        return "35+"

    def ret(x):
        if x >= 99: return "no layoff"
        if x == 0: return "1st match back"
        if x <= 4: return "2-5 back"
        if x <= 14: return "6-15 back"
        return "16+ back"

    def ntot(x):
        if x < 20: return "<20"
        if x < 50: return "20-49"
        if x < 100: return "50-99"
        if x < 200: return "100-199"
        if x < 400: return "200-399"
        return "400+"

    def home(x):
        if np.isnan(x): return None
        return "home" if x == 1 else "away"

    def free(x):
        return "walkover/early ret." if x == 1 else "played"

    return {
        "home": ("Home country", home, None),
        "gap": ("Weeks since previous event started", gap, None),
        "prev_n": ("Matches at the previous event (if within 2 weeks)", prevn, "prev2wk"),
        "load28": ("Matches in events of the previous 4 weeks", load, None),
        "since_return": ("Matches since a >12-week break", ret, None),
        "ed_n": ("Matches already played in this event (incl. qualifying)", edn, None),
        "ed_games": ("Games already played in this event", edg, None),
        "last_games": ("Games in the player's previous round", last, None),
        "last_free": ("Previous round", free, "inevent"),
        "ed_min": ("Minutes already on court in this event", edmin, None),
        "age": ("Age", age, None),
        "n_total": ("Career matches before (all levels)", ntot, None),
    }


ORDER = {
    "gap": ["1 wk", "2 wk", "3-4 wk", "5-8 wk", "9-12 wk", "13-26 wk", "> 26 wk", "first event"],
    "since_return": ["1st match back", "2-5 back", "6-15 back", "16+ back", "no layoff"],
    "ed_n": ["0 (first match)", "1", "2", "3", "4+"],
    "ed_games": ["0", "1-19", "20-39", "40-59", "60-79", "80+"],
    "last_games": ["0 / first match", "1-17", "18-23", "24-29", "30-39", "40+"],
    "ed_min": ["0", "<2 h", "2-4 h", "4-6 h", "6 h+"],
    "load28": ["0", "1-2", "3-4", "5-7", "8+"],
    "prev_n": ["0", "1", "2", "3", "4+"],
    "age": ["<19", "19-20", "21-22", "23-25", "26-28", "29-31", "32-34", "35+"],
    "n_total": ["<20", "20-49", "50-99", "100-199", "200-399", "400+"],
    "home": ["home", "away"],
    "last_free": ["walkover/early ret.", "played"],
}


# ---------------------------------------------------------------- joint model
def design(Fw, Fl, idx_arr, p1_is_w, start_arr, bo_arr):
    """Antisymmetric features x_p1 - x_p2 and their names."""
    def side(k):
        w, l = Fw[k][idx_arr], Fl[k][idx_arr]
        return np.where(p1_is_w, w, l), np.where(p1_is_w, l, w)

    cols, names = [], []

    def add(name, a, b):
        cols.append(np.nan_to_num(a - b)); names.append(name)

    h1, h2 = side("home")
    add("home", np.nan_to_num(h1), np.nan_to_num(h2))
    g1, g2 = side("gap")
    for lab, lo, hi in (("gap 1wk", 0, 7), ("gap 13-26wk", 85, 182), ("gap >26wk", 183, 1e9)):
        add(lab, ((g1 > lo) & (g1 <= hi)).astype(float) if lo else (g1 <= hi).astype(float),
            ((g2 > lo) & (g2 <= hi)).astype(float) if lo else (g2 <= hi).astype(float))
    r1, r2 = side("since_return")
    add("1st match back", (r1 == 0).astype(float), (r2 == 0).astype(float))
    add("2-5 matches back", ((r1 >= 1) & (r1 <= 4)).astype(float), ((r2 >= 1) & (r2 <= 4)).astype(float))
    l1, l2 = side("load28")
    add("load28 /5", l1 / 5, l2 / 5)
    e1, e2 = side("ed_games")
    add("event games /20", e1 / 20, e2 / 20)
    q1, q2 = side("ed_n")
    add("event matches", q1, q2)
    lg1, lg2 = side("last_games")
    add("prev-round games /10", lg1 / 10, lg2 / 10)
    a1, a2 = side("age")
    a1 = np.nan_to_num(a1, nan=26.0); a2 = np.nan_to_num(a2, nan=26.0)
    add("age<21", (a1 < 21).astype(float), (a2 < 21).astype(float))
    add("age 21-22", ((a1 >= 21) & (a1 < 23)).astype(float), ((a2 >= 21) & (a2 < 23)).astype(float))
    add("age 32+", (a1 >= 32).astype(float), (a2 >= 32).astype(float))
    n1, n2 = side("n_total")
    add("career<50", (n1 < 50).astype(float), (n2 < 50).astype(float))
    return np.column_stack(cols), names


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=int, default=605)
    ap.add_argument("--from", dest="date_from", default="2005-01-01")
    ap.add_argument("--to", dest="date_to", default="2026-12-31")
    ap.add_argument("--json", type=Path, default=ARCHIVE / "results" / "context_report.json")
    ap.add_argument("--png", type=Path, default=ARCHIVE / "reports" / "context_residuals.png")
    ap.add_argument("--models", type=Path, default=models_db.MODELS_DB)
    ap.add_argument("--data", type=Path, default=models_db.DATA_DB)
    args = ap.parse_args(argv)
    t = time.perf_counter()

    con = models_db.connect(args.models)
    models_db.attach_data(con, args.data)
    prep = tuning.prepare(con)
    m = prep.m
    # Replay the run from its saved parameters on the current tennis.db (stored predictions
    # miss matches added since, e.g. AcesAPI pulls).
    spec, params = tuning.model_of(con, args.run)
    if spec.model_id != "points_v1":
        raise SystemExit(f"run {args.run} is {spec.model_id}, not points_v1")
    spec.attach(con, m)
    pw = np.asarray(spec.predict(spec.run(m, params), params), dtype=float)
    ed, mt, pl = load_info(con)
    con.close()
    pts = m.extra["points"]
    stats_end = max(m.start[i] for i in range(len(m)) if pts.w_sp[i] > 0)
    if args.date_to > stats_end:
        args.date_to = stats_end         # after the stats end points_v1 is result-only
    print(f"replayed run {args.run} on {len(m):,} matches; scoring to {args.date_to} "
          f"(last event with stats) [{time.perf_counter() - t:.0f}s]")

    hosts = host_map()
    F = build_features(m, ed, mt, pl, hosts)
    print(f"features built [{time.perf_counter() - t:.0f}s]")

    p1cal, y, idx, _ = calibrate.calibrated(prep, pw, args.date_from, args.date_to)
    idx = np.array(idx)
    p1_is_w = np.array([prep.winner_is_p1[i] for i in idx], dtype=bool)
    starts = np.array([m.start[i] for i in idx])
    tours = np.array([m.tour[i] for i in idx])
    rows_meta = [mt[m.match_id[i]] for i in idx]
    eds = [ed[r[0]] for r in rows_meta]
    test = starts >= "2022-01-01"
    print(f"scored tour-level matches {args.date_from}..{args.date_to}: {len(idx):,} "
          f"(test 2022+: {test.sum():,}, calibrated log loss {ll_vec(p1cal[test], y[test]).mean():.4f})")

    # player-perspective rows: winner rows (y=1, p=pw_cal) and loser rows (y=0, 1-p)
    p_w = np.where(p1_is_w, p1cal, 1 - p1cal)
    P = np.concatenate([p_w, 1 - p_w])
    Y = np.concatenate([np.ones(len(idx)), np.zeros(len(idx))])
    TOUR = np.concatenate([tours, tours])
    TEST = np.concatenate([test, test])
    report = {"run": args.run, "date_from": args.date_from, "date_to": args.date_to,
              "n_matches": int(len(idx)), "residuals": {}, "slopes": {}, "joint": {}}

    print("\n== 1. residuals by player context (actual - expected, pp; opponent in another bucket) ==")
    for feat, (title, fn, _) in bucketers().items():
        bw = [fn(v) for v in F["w"][feat][idx]]
        bl = [fn(v) for v in F["l"][feat][idx]]
        B = np.array(bw + bl, dtype=object)
        O = np.array(bl + bw, dtype=object)
        out = {"title": title, "buckets": {}}
        print(f"\n{title}")
        print(f"  {'bucket':<22}{'rows':>8}{'expected':>9}{'actual':>8}{'resid pp':>9}{'SE':>6}{'z':>6}"
              f"{'  test pp':>9}{'SE':>6}")
        for lab in ORDER[feat]:
            sel = (B == lab) & (O != lab)
            r = residual_row(P, Y, sel)
            if r is None or r["n"] < 30:
                continue
            rt = residual_row(P, Y, sel & TEST)
            r["by_tour"] = {tt: residual_row(P, Y, sel & (TOUR == tt)) for tt in ("atp", "wta")}
            r["test"] = rt
            out["buckets"][lab] = r
            tp = f"{rt['resid_pp']:+9.2f}{rt['se_pp']:6.2f}" if rt and rt["n"] >= 30 else ""
            print(f"  {lab:<22}{r['n']:>8,}{r['expected']:9.3f}{r['actual']:8.3f}{r['resid_pp']:+9.2f}"
                  f"{r['se_pp']:6.2f}{r['z']:+6.1f}{tp}")
        report["residuals"][feat] = out

    print("\n== 1b. home residual by period ==")
    START2 = np.concatenate([starts, starts])
    hw, hl = F["w"]["home"][idx], F["l"]["home"][idx]
    H = np.concatenate([hw, hl]); HO = np.concatenate([hl, hw])
    report["home_by_period"] = {}
    for lo_, hi_ in (("2005", "2010"), ("2010", "2015"), ("2015", "2020-03"), ("2020-03", "2021-07"),
                     ("2021-07", "2024"), ("2024", "2027")):
        sel = (H == 1) & (HO == 0) & (START2 >= lo_) & (START2 < hi_)
        r = residual_row(P, Y, sel)
        if r:
            report["home_by_period"][f"{lo_}..{hi_}"] = r
            print(f"  {lo_}..{hi_:<8}{r['n']:>7,}  {r['resid_pp']:+.2f} ± {r['se_pp']:.2f} pp")
    for tt in ("atp", "wta"):
        r = report["residuals"]["home"]["buckets"]["home"]["by_tour"][tt]
        print(f"  {tt}: {r['resid_pp']:+.2f} ± {r['se_pp']:.2f} pp (n {r['n']:,})")

    print("\n== 2. calibration slope of points_v1 (calibrated) by match context ==")
    lg = logit(p1cal)
    ctx = {
        "level": [e[1] for e in eds],
        "round": [r[1] for r in rows_meta],
        "best_of": [str(r[5] or e[4]) for r, e in zip(rows_meta, eds)],
        "surface": [str(e[2]) for e in eds],
        "status": [r[4] for r in rows_meta],
    }
    for name, vals in ctx.items():
        vals = np.array(vals, dtype=object)
        out = {}
        print(f"\n{name}")
        for v in sorted(set(vals), key=str):
            s = vals == v
            if s.sum() < 300:
                continue
            a, se = fit_slope(lg[s], y[s])
            llr = float(ll_vec(p1cal[s], y[s]).mean())
            out[str(v)] = {"n": int(s.sum()), "alpha": a, "se": se, "log_loss": llr}
            print(f"  {str(v):<10}{s.sum():>8,}  alpha {a:5.3f} ± {se:.3f}  ll {llr:.4f}")
        report["slopes"][name] = out

    print("\n== 3. joint antisymmetric context model on top of calibrated points_v1 ==")
    bo = np.array([r[5] or e[4] or 3 for r, e in zip(rows_meta, eds)])
    X, names = design(F["w"], F["l"], idx, p1_is_w, starts, bo)
    off = logit(p1cal)
    quarters = block_of(starts, 3)

    # bo5 x logit: a separate calibration slope for best-of-5 (antisymmetric, since the
    # logit flips sign with player order)
    X = np.column_stack([X, (bo == 5) * off])
    names = names + ["bo5 x logit"]
    groups = {
        "home": ["home"],
        "rest / layoff": ["gap 1wk", "gap 13-26wk", "gap >26wk", "1st match back", "2-5 matches back"],
        "recent load": ["load28 /5"],
        "in-event workload": ["event games /20", "event matches", "prev-round games /10"],
        "age / experience": ["age<21", "age 21-22", "age 32+", "career<50"],
        "bo5 slope": ["bo5 x logit"],
    }
    col = {n_: j for j, n_ in enumerate(names)}

    def evaluate(train, score, label, cols, verbose=True):
        cj = [col[c] for c in cols]
        Xs = X[:, cj]
        b, se = fit_offset_logit(Xs[train], off[train], y[train], l2=10.0)
        pc = 1 / (1 + np.exp(-(off[score] + Xs[score] @ b)))
        d = ll_vec(pc, y[score]) - ll_vec(p1cal[score], y[score])
        q = quarters[score]
        wins = sum(1 for qq in np.unique(q) if d[q == qq].mean() < 0)
        res = {"n": int(score.sum()), "ll_base": float(ll_vec(p1cal[score], y[score]).mean()),
               "ll_context": float(ll_vec(pc, y[score]).mean()), "diff": float(d.mean()),
               "se": float(d.std(ddof=1) / math.sqrt(len(d))), "quarters_won": wins,
               "quarters": int(len(np.unique(q))),
               "coef": {n_: [float(bb), float(ss)] for n_, bb, ss in zip(cols, b, se)}}
        if verbose:
            print(f"  {label}: n={res['n']:,}  base {res['ll_base']:.4f}  context {res['ll_context']:.4f}  "
                  f"diff {res['diff']:+.4f} ± {res['se']:.4f}  quarters won {wins}/{res['quarters']}")
        return res

    val = (starts >= "2019-01-01") & (starts <= "2021-12-31")
    tr_val, tr_test = starts < "2019-01-01", starts < "2022-01-01"
    ctx_cols = [c for g, cs in groups.items() if g != "bo5 slope" for c in cs]
    specs = {"context": ctx_cols, "context + bo5 slope": names}
    print("  validation (fit 2005-18, score 2019-21):")
    report["joint"]["val"] = {k: evaluate(tr_val, val, k, cs) for k, cs in specs.items()}
    print("  ablation on validation: group alone / all minus group")
    report["joint"]["ablation_val"] = {}
    for g, cs in groups.items():
        alone = evaluate(tr_val, val, "", cs, verbose=False)
        minus = evaluate(tr_val, val, "", [c for c in names if c not in cs], verbose=False)
        report["joint"]["ablation_val"][g] = {"alone": alone["diff"], "alone_se": alone["se"],
                                              "minus": minus["diff"], "minus_se": minus["se"]}
        print(f"    {g:<20} alone {alone['diff']:+.4f} ± {alone['se']:.4f}   "
              f"all but {minus['diff']:+.4f} ± {minus['se']:.4f}")
    # selection on validation only: keep a group if dropping it makes val worse
    full_val = report["joint"]["val"]["context + bo5 slope"]["diff"]
    keep = [g for g, a in report["joint"]["ablation_val"].items() if a["minus"] > full_val]
    specs["val-selected"] = [c for g in keep for c in groups[g]]
    report["joint"]["val_selected_groups"] = keep
    report["joint"]["val"]["val-selected"] = evaluate(tr_val, val, f"val-selected {keep}", specs["val-selected"])
    print("  test (fit 2005-21, score 2022-26):")
    report["joint"]["test"] = {}
    for k, cs in specs.items():
        r = evaluate(tr_test, test, k, cs)
        r["by_tour"] = {}
        for tt in ("atp", "wta"):
            r["by_tour"][tt] = evaluate(tr_test, test & (tours == tt), f"    {tt}", cs)
        report["joint"]["test"][k] = r
    print("  coefficients (fit 2005-21, logit units per unit of x_A - x_B):")
    for n_, (bb, ss) in report["joint"]["test"]["context + bo5 slope"]["coef"].items():
        print(f"    {n_:<22}{bb:+.4f} ± {ss:.4f}  (z {bb / ss:+.1f})")

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.json}")
    plot(report, args.png)
    return 0


def plot(report, png):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: no graph")
        return
    feats = ["home", "gap", "since_return", "load28", "ed_n", "last_games", "age", "n_total"]
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.5))
    for ax, f in zip(axes.flat, feats):
        r = report["residuals"][f]
        labs = list(r["buckets"])
        x = np.arange(len(labs))
        v = np.array([r["buckets"][k]["resid_pp"] for k in labs])
        e = np.array([r["buckets"][k]["se_pp"] for k in labs])
        ax.bar(x, v, color=np.where(v >= 0, "#3b7dd8", "#d8613b"), alpha=0.85)
        ax.errorbar(x, v, yerr=2 * e, fmt="none", ecolor="k", lw=1, capsize=3)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(labs, rotation=35, ha="right", fontsize=8)
        ax.set_title(r["title"], fontsize=9)
        ax.set_ylabel("win rate − points_v1 expected (pp)", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"Context vs points_v1 run {report['run']} (calibrated), tour-level "
                 f"{report['date_from'][:4]}–2026, ±2 SE", fontsize=11)
    fig.tight_layout()
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=130)
    print(f"wrote {png}")


if __name__ == "__main__":
    sys.exit(main())
