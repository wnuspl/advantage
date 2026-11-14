"""Phase-profile test, surface-controlled, and the milestone-10 correction on points_v1.

    python scripts/phase_test.py [--points-run 273] [--glicko-run 187] [--out results/phase_test.json]

Step 1  stability: are phase edges (win rate by phase beyond the player's own level) stable
        between periods? Pooled (2026-11-14 test) vs surface-adjusted profiles, each with a
        null band (wins redrawn within player x side x surface, hypergeometric).
Step 2  residual test: do the player part (charted level) and the matchup part (A's phases x
        B's phases) predict points won beyond (a) Glicko-2 run 187 and (b) points_v1? For
        every variant (pooled / surface / same-surface).
Step 3  correction: shift each server's point logit by beta * M / (p (1 - p)), beta fitted
        on serve points before 2019; scored on val 2019-21 and test 2022-26 (tour-level),
        raw and calibrated, paired against points_v1 on identical matches.

Profiles use only charted matches whose edition started before the match (point-in-time);
players need MIN_PTS charted serve and return points, else no correction (never imputed).
Reads tennis.db and models.db read-only; parsed points cached in models/phase/.
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tennis import calibrate, composite_eval, markov, phase, points_rating, ratingmodels, tuning  # noqa: E402

DATA_DB = ROOT / "tennis.db"
MODELS_DB = ROOT / "models.db"
CACHE = ROOT / "models" / "phase"
MIN_PTS = 300
START = "2011-01-01"
PERIODS = [("2015-01-01", "2018-12-31"), ("2019-01-01", "2022-12-31"), ("2023-01-01", "2026-12-31")]
EDGE_ROWS = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (0, 3), (1, 4)]   # (side, k)
EDGE_NAMES = ["serve short", "serve medium", "serve long", "return short", "return medium",
              "return long", "serve net", "return net"]
SEED = 20261114


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------ loading

def connect():
    con = sqlite3.connect("file::memory:", uri=True)
    con.execute(f"ATTACH DATABASE 'file:{DATA_DB.as_posix()}?mode=ro' AS data")
    con.execute(f"ATTACH DATABASE 'file:{MODELS_DB.as_posix()}?mode=ro' AS mdl")
    return con


def load_tallies(con) -> phase.Tallies:
    CACHE.mkdir(parents=True, exist_ok=True)
    tag = int(DATA_DB.stat().st_mtime)
    f = CACHE / f"tallies_{tag}.npz"
    if f.exists():
        z = np.load(f)
        return phase.Tallies(z["match_id"], z["player"], z["date"], z["tour"], z["surf"], z["stats"])
    log("parsing MCP points ...")
    pts = phase.load_points(con, "data")
    log(f"  {len(pts.match_id):,} points parsed, {sum(pts.skipped.values()):,} skipped")
    meta = {mid: (d, tour, surf) for mid, d, tour, surf in con.execute(
        """SELECT m.match_id, e.start_date, t.tour, e.surface
           FROM data.matches m JOIN data.tournament_edition e USING (edition_id)
           JOIN data.tournament t USING (tournament_id)
           WHERE m.match_id IN (SELECT DISTINCT match_id FROM data.point WHERE data_source='mcp')""")}
    t = phase.tally(pts, meta)
    np.savez(f, match_id=t.match_id, player=t.player, date=t.date, tour=t.tour, surf=t.surf, stats=t.stats)
    return t


def run_params(con, run_id):
    model_id, pj = con.execute("SELECT model_id, params_json FROM mdl.model_run WHERE run_id=?",
                               (run_id,)).fetchone()
    spec = ratingmodels.get(model_id)
    return spec, spec.params_cls(**json.loads(pj))


# ------------------------------------------------------------------ step 1

def _period_stats(t: phase.Tallies, lo, hi):
    sel = (t.date >= lo) & (t.date <= hi)
    out = {}
    for pid, tour, st in zip(t.player[sel], t.tour[sel], t.stats[sel]):
        key = (int(pid), tour)
        out[key] = out.get(key, 0) + st
    pop = {}
    for (pid, tour), st in out.items():
        pop[tour] = pop.get(tour, 0) + st
    return out, pop


def _edges(stats, pop, variant):
    pr = phase.profile(stats, pop, variant, 0)
    return np.array([pr["e"][s, k] for s, k in EDGE_ROWS]), pr["N"]


def _redraw(stats, rng):
    """Null: wins within each (side, surface) redrawn so phases carry no edge."""
    st = stats.copy()
    N, W = st[..., 0], st[..., 1]
    for k in range(phase.NK):
        n = st[..., 2 + k]
        ok = N > 0
        draw = np.zeros_like(n)
        draw[ok] = rng.hypergeometric(W[ok].astype(np.int64), (N[ok] - W[ok]).astype(np.int64),
                                      n[ok].astype(np.int64))
        st[..., 2 + phase.NK + k] = draw
    return st


def step1(t: phase.Tallies, reps: int = 40) -> dict:
    rng = np.random.default_rng(SEED)
    per = [_period_stats(t, lo, hi) for lo, hi in PERIODS]
    out = {}
    for a, b in ((0, 1), (1, 2)):
        (sa, pa), (sb, pb) = per[a], per[b]
        keys = [k for k in sa if k in sb]
        label = f"{PERIODS[a][0][:4]}-{PERIODS[a][1][2:4]} -> {PERIODS[b][0][:4]}-{PERIODS[b][1][2:4]}"
        res = {}
        for variant in ("pooled", "surface"):
            ea, eb, keep = [], [], []
            for k in keys:
                x, na = _edges(sa[k], pa[k[1]], variant)
                y, nb = _edges(sb[k], pb[k[1]], variant)
                if na.min() >= MIN_PTS and nb.min() >= MIN_PTS:
                    ea.append(x); eb.append(y); keep.append(k)
            ea, eb = np.array(ea), np.array(eb)
            r = [float(np.corrcoef(ea[:, j], eb[:, j])[0, 1]) for j in range(len(EDGE_NAMES))]
            null = []
            for _ in range(reps):
                xa = np.array([_edges(_redraw(sa[k], rng), pa[k[1]], variant)[0] for k in keep])
                xb = np.array([_edges(_redraw(sb[k], rng), pb[k[1]], variant)[0] for k in keep])
                null.append([np.corrcoef(xa[:, j], xb[:, j])[0, 1] for j in range(len(EDGE_NAMES))])
            null = np.array(null)
            res[variant] = {"n_players": len(keep), "r": dict(zip(EDGE_NAMES, r)),
                            "null_lo": dict(zip(EDGE_NAMES, np.percentile(null, 2.5, axis=0).tolist())),
                            "null_hi": dict(zip(EDGE_NAMES, np.percentile(null, 97.5, axis=0).tolist()))}
        # surface mix stability (clay share of serve points), the suspected confounder
        mix = []
        for k in keys:
            na, nb = sa[k][0, :, 0], sb[k][0, :, 0]
            if na.sum() >= MIN_PTS and nb.sum() >= MIN_PTS:
                mix.append((na[1] / na.sum(), nb[1] / nb.sum(), na[2] / na.sum(), nb[2] / nb.sum()))
        mix = np.array(mix)
        res["clay_share_r"] = float(np.corrcoef(mix[:, 0], mix[:, 1])[0, 1])
        res["grass_share_r"] = float(np.corrcoef(mix[:, 2], mix[:, 3])[0, 1])
        out[label] = res
        log(f"step 1 {label}: " + "  ".join(
            f"{n}: {res['pooled']['r'][n]:+.2f}/{res['surface']['r'][n]:+.2f}" for n in EDGE_NAMES))
    return out


# ------------------------------------------------------------------ step 2 rows

def build_rows(prep, pdata, hist: phase.History, variants) -> dict:
    """One row per match with stats, both players profiled, edition start >= START."""
    m = prep.m
    rng = np.random.default_rng(SEED)
    rows = {k: [] for k in ("i", "a_is_w", "date", "tour", "tl", "surf", "spa", "swa", "spb", "swb")}
    prof = {v: {k: [] for k in ("Ma", "Mb", "La", "Lb", "Pa_srv", "Pb_ret", "Pb_srv", "Pa_ret")} for v in variants}
    for i in range(len(m)):
        if m.start[i] < START or m.status[i] != "completed" or pdata.w_sp[i] <= 0 or pdata.l_sp[i] <= 0:
            continue
        s = phase.SURF_INDEX.get(m.surface[i] or "", -1)
        if s < 0:
            continue
        sw, sl = hist.player(m.winner[i], m.start[i]), hist.player(m.loser[i], m.start[i])
        if sw is None or sl is None:
            continue
        pop = hist.population(m.tour[i], m.start[i])
        profs = {v: (phase.profile(sw, pop, v, s), phase.profile(sl, pop, v, s)) for v in variants}
        # eligibility judged on the pooled counts (same rows for every variant)
        if min(profs[variants[0]][0]["N"].min(), profs[variants[0]][1]["N"].min()) < MIN_PTS:
            continue
        if "same" in profs and min(profs["same"][0]["N"].min(), profs["same"][1]["N"].min()) < 1:
            pass  # same-surface profile with no points is all-zero: no correction, kept
        a_is_w = rng.random() < 0.5
        rows["i"].append(i); rows["a_is_w"].append(a_is_w); rows["date"].append(m.start[i])
        rows["tour"].append(m.tour[i]); rows["tl"].append(m.tour_level[i]); rows["surf"].append(s)
        wsp, wsw, lsp, lsw = pdata.w_sp[i], pdata.w_sw[i], pdata.l_sp[i], pdata.l_sw[i]
        if a_is_w:
            rows["spa"].append(wsp); rows["swa"].append(wsw); rows["spb"].append(lsp); rows["swb"].append(lsw)
        else:
            rows["spa"].append(lsp); rows["swa"].append(lsw); rows["spb"].append(wsp); rows["swb"].append(wsw)
        for v, (pw, pl) in profs.items():
            a, b = (pw, pl) if a_is_w else (pl, pw)
            P = prof[v]
            P["Ma"].append(phase.matchup(a, b)); P["Mb"].append(phase.matchup(b, a))
            P["La"].append(a["d"].sum()); P["Lb"].append(b["d"].sum())
            P["Pa_srv"].append(a["d"][0]); P["Pb_ret"].append(b["d"][1])
            P["Pb_srv"].append(b["d"][0]); P["Pa_ret"].append(a["d"][1])
    out = {k: np.asarray(v) for k, v in rows.items()}
    out["prof"] = {v: {k: np.asarray(x, dtype=float) for k, x in P.items()} for v, P in prof.items()}
    return out


def _ols(y, X, w=None):
    """Coefficients, t-stats, df (weighted least squares, HC1 robust SEs)."""
    w = np.ones(len(y)) if w is None else w
    sw = np.sqrt(w)
    Xw, yw = X * sw[:, None], y * sw
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    e = yw - Xw @ beta
    n, k = Xw.shape
    XtX_inv = np.linalg.pinv(Xw.T @ Xw)
    meat = (Xw * e[:, None]).T @ (Xw * e[:, None])
    cov = XtX_inv @ meat @ XtX_inv * n / (n - k)
    se = np.sqrt(np.diag(cov))
    return beta, beta / se, n - k


def _controls(ref, tour):
    cols = []
    for t in ("atp", "wta"):
        d = (tour == t).astype(float)
        cols += [d, d * ref, d * ref ** 3]
    return np.column_stack(cols)


def step2(R, glicko_pw, xa_w, xb_w, variants) -> dict:
    """Total point share of A beyond each reference, on the player and matchup parts."""
    i = R["i"]
    a_w = R["a_is_w"]
    spa, swa, spb, swb = (R[k].astype(float) for k in ("spa", "swa", "spb", "swb"))
    tot = spa + spb
    y = (swa + (spb - swb)) / tot
    pg = np.asarray([glicko_pw[j] for j in i])
    pg = np.where(a_w, pg, 1 - pg)
    g_logit = np.log(pg / (1 - pg))
    # points_v1 serve-point probabilities for A and B
    xA = np.where(a_w, xa_w[i], xb_w[i])
    xB = np.where(a_w, xb_w[i], xa_w[i])
    qA, qB = 1 / (1 + np.exp(-xA)), 1 / (1 + np.exp(-xB))
    exp_share = (spa * qA + spb * (1 - qB)) / tot
    refs = {"glicko2": g_logit, "points_v1": exp_share}
    subsets = {"all": np.ones(len(i), bool), "atp": R["tour"] == "atp", "wta": R["tour"] == "wta",
               "tour-level": R["tl"] == 1, "2022-26": R["date"] >= "2022-01-01"}
    out = {}
    for v in variants:
        P = R["prof"][v]
        player = (P["La"] - P["Lb"]) / 2
        mu = (P["Ma"] - P["Mb"]) / 2
        q80 = np.abs(mu) >= np.quantile(np.abs(mu), 0.8)
        subs = dict(subsets, **{"largest 20% matchup": q80})
        out[v] = {}
        for rname, ref in refs.items():
            out[v][rname] = {}
            for sname, sel in subs.items():
                X = np.column_stack([_controls(ref[sel], R["tour"][sel]), player[sel], mu[sel]])
                b, t, df = _ols(y[sel], X, tot[sel])
                pc = lambda tt: float(tt / math.sqrt(tt * tt + df))
                out[v][rname][sname] = {"n": int(sel.sum()), "b_player": float(b[-2]), "t_player": float(t[-2]),
                                        "r_player": pc(t[-2]), "b_matchup": float(b[-1]),
                                        "t_matchup": float(t[-1]), "r_matchup": pc(t[-1])}
            s = out[v][rname]["all"]
            log(f"step 2 {v:8s} vs {rname:9s}: player t={s['t_player']:+.1f} (r={s['r_player']:+.3f})"
                f"  matchup t={s['t_matchup']:+.1f} (r={s['r_matchup']:+.3f})  n={s['n']}")
    return out


# ------------------------------------------------------------------ step 3

def serve_obs(R, xa_w, xb_w, v):
    """Serve-side observations (two per row): residual vs points_v1, M, own-level parts."""
    i, a_w = R["i"], R["a_is_w"]
    P = R["prof"][v]
    xA = np.where(a_w, xa_w[i], xb_w[i]); xB = np.where(a_w, xb_w[i], xa_w[i])
    qA, qB = 1 / (1 + np.exp(-xA)), 1 / (1 + np.exp(-xB))
    sp = np.concatenate([R["spa"], R["spb"]]).astype(float)
    sw = np.concatenate([R["swa"], R["swb"]]).astype(float)
    q = np.concatenate([qA, qB])
    M = np.concatenate([P["Ma"], P["Mb"]])
    L = np.concatenate([P["Pa_srv"] - P["Pb_ret"], P["Pb_srv"] - P["Pa_ret"]])
    date = np.concatenate([R["date"], R["date"]])
    tour = np.concatenate([R["tour"], R["tour"]])
    return sw / sp - q, q, M, L, sp, date, tour


def corrected_pwinner(res, params, R, v, beta, gamma=0.0):
    """points_v1 P(winner wins) with the phase correction on rows of R (others unchanged)."""
    P = R["prof"][v]
    i, a_w = R["i"], R["a_is_w"]
    # winner-serving and loser-serving M (and level part)
    M_w = np.where(a_w, P["Ma"], P["Mb"]); M_l = np.where(a_w, P["Mb"], P["Ma"])
    L_w = np.where(a_w, P["Pa_srv"] - P["Pb_ret"], P["Pb_srv"] - P["Pa_ret"])
    L_l = np.where(a_w, P["Pb_srv"] - P["Pa_ret"], P["Pa_srv"] - P["Pb_ret"])
    xa = res.xa + params.blend * res.xa_s
    xb = res.xb + params.blend * res.xb_s
    qa, qb = 1 / (1 + np.exp(-xa[i])), 1 / (1 + np.exp(-xb[i]))
    shift_a = (beta * M_w + gamma * L_w) / (qa * (1 - qa))
    shift_b = (beta * M_l + gamma * L_l) / (qb * (1 - qb))
    r2 = replace(res, xa=res.xa.copy(), xb=res.xb.copy())
    r2.xa[i] += shift_a
    r2.xb[i] += shift_b
    return points_rating.predict(r2, params)


def score(prep, pw, ref_pw, lo, hi, cal: bool, only=None):
    if cal:
        p, y, idx, _ = calibrate.calibrated(prep, pw, lo, hi)
        pr, _, _, _ = calibrate.calibrated(prep, ref_pw, lo, hi)
    else:
        idx = prep.indices(lo, hi, "tour")
        p, y = calibrate.oriented(prep, pw, idx)
        pr, _ = calibrate.oriented(prep, ref_pw, idx)
    if only is not None:
        s = np.isin(np.asarray(idx), only)
        p, y, pr, idx = p[s], y[s], pr[s], np.asarray(idx)[s]
    blocks = composite_eval.block_of(np.array([prep.m.start[j] for j in idx]))
    mt = composite_eval.metrics(p, y, ref=pr, block=blocks)
    return {"n": mt.n, "log_loss": mt.log_loss, "ref_log_loss": float(mt.log_loss - mt.d_ref),
            "d": mt.d_ref, "se": mt.d_se, "blocks_won": mt.blocks_won, "blocks": mt.blocks}


def step3(prep, res, params, R, variants) -> dict:
    out = {}
    base_pw = points_rating.predict(res, params)
    xa_w = res.xa + params.blend * res.xa_s
    xb_w = res.xb + params.blend * res.xb_s
    for v in variants:
        r, q, M, L, sp, date, tour = serve_obs(R, xa_w, xb_w, v)
        fit = date < "2019-01-01"
        X = np.column_stack([np.ones(fit.sum()), M[fit]])
        b, t, _ = _ols(r[fit], X, sp[fit])
        X2 = np.column_stack([np.ones(fit.sum()), M[fit], L[fit]])
        b2, t2, _ = _ols(r[fit], X2, sp[fit])
        beta, gamma = float(b[1]), float(b2[2])
        o = {"beta": beta, "t_beta_fit": float(t[1]), "gamma_level": gamma, "t_gamma_fit": float(t2[2]),
             "n_serve_obs_fit": int(fit.sum())}
        on = R["i"]
        for name, (bb, gg) in {"matchup": (beta, 0.0), "matchup+level": (float(b2[1]), gamma)}.items():
            pw = corrected_pwinner(res, params, R, v, bb, gg)
            for blk, (lo, hi) in tuning.DEFAULT_BLOCKS.items():
                for cal in (False, True):
                    key = f"{name}/{blk}/{'cal' if cal else 'raw'}"
                    o[key] = score(prep, pw, base_pw, lo, hi, cal)
                    o[key + "/profiled"] = score(prep, pw, base_pw, lo, hi, cal, only=on)
        out[v] = o
        tv = o["matchup/test/cal"]
        log(f"step 3 {v:8s}: beta={beta:.3f} (t={o['t_beta_fit']:.1f})  test cal d={tv['d']:+.5f} ± {tv['se']:.5f}"
            f"  (profiled rows {o['matchup/test/cal/profiled']['n']}: d={o['matchup/test/cal/profiled']['d']:+.5f})")
    return out


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--points-run", type=int, default=273)
    ap.add_argument("--glicko-run", type=int, default=187)
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--out", default=str(ROOT / "results" / "phase_test.json"))
    a = ap.parse_args()
    variants = list(phase.VARIANTS)
    con = connect()
    t0 = time.perf_counter()
    prep = tuning.prepare(con)
    log(f"{len(prep.m):,} matches loaded")
    t = load_tallies(con)
    log(f"{len(t.match_id):,} charted (match, player) tallies")
    hist = phase.History(t)

    s1 = step1(t, a.reps)

    gspec, gpar = run_params(con, a.glicko_run)
    log(f"running {gspec.model_id} (run {a.glicko_run}) ...")
    gres = gspec.run(prep.m, gpar)
    pspec, ppar = run_params(con, a.points_run)
    pdata = pspec.attach(con, prep.m)
    log(f"running {pspec.model_id} (run {a.points_run}) ...")
    pres = pspec.run(prep.m, ppar)
    xa_w = pres.xa + ppar.blend * pres.xa_s
    xb_w = pres.xb + ppar.blend * pres.xb_s

    R = build_rows(prep, pdata, hist, variants)
    log(f"{len(R['i']):,} matches with stats and both players profiled (>= {MIN_PTS} pts)")
    s2 = step2(R, gres.p_winner, xa_w, xb_w, variants)
    s3 = step3(prep, pres, ppar, R, variants)

    out = {"params": {"points_run": a.points_run, "glicko_run": a.glicko_run, "min_pts": MIN_PTS,
                      "start": START, "shrink": phase.Shrink().__dict__},
           "step1": s1, "step2": s2, "step3": s3, "seconds": time.perf_counter() - t0}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    log(f"wrote {a.out}")


if __name__ == "__main__":
    main()
