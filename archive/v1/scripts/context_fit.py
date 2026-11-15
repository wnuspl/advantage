"""Fit the points_v1 context offsets (milestone 9) on serve-point residuals.

    python archive/v1/scripts/context_fit.py                      # from run 694, fit < 2019, score val
    python archive/v1/scripts/context_fit.py --iters 4 --record   # record the final setting in models.db
    python archive/v1/scripts/context_fit.py --set rust_weight=0.7 --set q=2e-5

Iteration 0 replays the base run (context off). Each iteration then regresses the
serve-point residuals of the current run on the context features (weighted least squares,
serve and return coefficients separately) and adds the result to the coefficients:

    z = (k - n p) / (n p (1 - p)),   var z = 1 / (n p (1 - p)) + obs_var + var(x)
    z ~ beta_serve . c_server - beta_ret . c_returner

with p from the run's pre-match x (context offset included, surface blend 1, as in the
update). Because the offsets feed the rating updates, a run with offsets has different
ratings, so the residual fit is repeated on the new run until the step is small: the fixed
point is where context explains none of what is left. Only matches with serve stats dated
[--fit-from, --cutoff) are used (default cutoff 2019-01-01, so validation 2019-21 is
out of sample for the coefficients).

Each iteration is scored on the block (default val) as calibrated tour-level log loss,
paired against the calibrated base run. Never choose on --block test.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ARCHIVE = Path(__file__).resolve().parents[1]   # archive/v1: results, reports, docs
ROOT = Path(__file__).resolve().parents[3]      # project root: tennis.db, data, src
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ARCHIVE / "src"))   # archived modules, merged into tennis

from tennis import calibrate, context, models_db, tuning      # noqa: E402
from tennis.composite_eval import block_of, metrics            # noqa: E402

PLAYED = {"completed", "retired"}


def residual_fit(m, res, params, fit_from: str, cutoff: str, tour_only: bool, l2: float = 1.0,
                 tour: str | None = None):
    """(delta_serve, delta_ret, se_serve, se_ret, n_obs) from one run's residuals
    (one tour's matches only if `tour`)."""
    d = m.extra["points"]
    cw, cl = m.extra["context"]
    n = len(m)
    sel = np.array([d.w_sp[i] > 0 and d.l_sp[i] > 0 and fit_from <= m.start[i] < cutoff
                    and m.status[i] in PLAYED and (m.tour_level[i] or not tour_only)
                    and (tour is None or m.tour[i] == tour)
                    for i in range(n)])
    idx = np.flatnonzero(sel)
    rows_x, rows_z, rows_w = [], [], []
    for xa, va, k, nn, cs, cr in (
            (res.xa + res.xa_s, res.va + res.va_s, d.w_sw, d.w_sp, cw, cl),     # winner serving
            (res.xb + res.xb_s, res.vb + res.vb_s, d.l_sw, d.l_sp, cl, cw)):    # loser serving
        x = xa[idx]
        p = 1 / (1 + np.exp(-x))
        npts = np.array([nn[i] for i in idx], dtype=float)
        kpts = np.array([k[i] for i in idx], dtype=float)
        info = npts * p * (1 - p)
        rows_z.append((kpts - npts * p) / info)
        rows_w.append(1.0 / (1.0 / info + params.obs_var + va[idx]))
        rows_x.append(np.hstack([cs[idx], -cr[idx]]))
    X, z, w = np.vstack(rows_x), np.concatenate(rows_z), np.concatenate(rows_w)
    XtW = X.T * w
    A = XtW @ X + l2 * np.eye(X.shape[1])
    beta = np.linalg.solve(A, XtW @ z)
    se = np.sqrt(np.diag(np.linalg.inv(A)))
    K = context.K
    return beta[:K], beta[K:], se[:K], se[K:], len(idx)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=int, default=694, help="base points_v1 run (context off)")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--fit-from", default="2000-01-01")
    ap.add_argument("--cutoff", default="2019-01-01", help="fit on matches before this date")
    ap.add_argument("--tour-only", action="store_true", help="fit on tour-level matches only")
    ap.add_argument("--per-tour", action="store_true",
                    help="separate ATP and WTA coefficients (WTA starts from the shared ones)")
    ap.add_argument("--block", choices=["val", "test"], default="val")
    ap.add_argument("--set", action="append", default=[], help="NAME=VALUE: override a parameter")
    ap.add_argument("--start", type=Path, help="JSON {param: value} of starting context coefficients")
    ap.add_argument("--record", action="store_true", help="record the final setting as a model_run")
    ap.add_argument("--json", type=Path, default=ARCHIVE / "results" / "context_fit.json")
    ap.add_argument("--models", type=Path, default=models_db.MODELS_DB)
    ap.add_argument("--data", type=Path, default=models_db.DATA_DB)
    args = ap.parse_args(argv)
    lo, hi = tuning.DEFAULT_BLOCKS[args.block]
    t0 = time.perf_counter()

    con = models_db.connect(args.models)
    models_db.attach_data(con, args.data)
    prep = tuning.prepare(con)
    m = prep.m
    spec, base_p = tuning.model_of(con, args.run)
    if spec.model_id != "points_v1":
        raise SystemExit(f"run {args.run} is {spec.model_id}, not points_v1")
    spec.attach(con, m)
    print(f"loaded {len(m):,} matches, features attached [{time.perf_counter() - t0:.0f}s]")

    def run_and_score(p):
        t = time.perf_counter()
        res = spec.run(m, p)
        pw = spec.predict(res, p)
        cal, y, idx, _ = calibrate.calibrated(prep, pw, lo, hi)
        return res, cal, y, idx, time.perf_counter() - t

    base_res, base_cal, y, idx, secs = run_and_score(base_p)
    starts = np.array([m.start[i] for i in idx])
    tours = np.array([m.tour[i] for i in idx])
    blocks = block_of(starts)
    print(f"base run {args.run} replayed in {secs:.0f}s; {args.block} {lo}..{hi}: {len(idx):,} tour matches, "
          f"calibrated log loss {metrics(base_cal, y).log_loss:.4f}")

    overrides = {}
    for s in args.set:
        k, v = s.split("=")
        overrides[k] = float(v)
    p = replace(base_p, **overrides)
    if args.start:
        p = replace(p, **json.loads(args.start.read_text(encoding="utf-8")))
    if args.per_tour and not p.ctx_per_tour:
        p = replace(p.with_context(*p.context_betas("atp"), tour="wta"), ctx_per_tour=1.0)
    tours_fit = ("atp", "wta") if p.ctx_per_tour else (None,)
    betas = {t: list(p.context_betas(t or "atp")) for t in tours_fit}
    res = base_res if p == base_p else None
    history = []

    def score(p, res, cal, it):
        out = {"iter": it, "params": p.as_dict()}
        for scope, sel in (("tour", np.ones(len(idx), bool)), ("tour_atp", tours == "atp"),
                           ("tour_wta", tours == "wta")):
            mt = metrics(cal[sel], y[sel], ref=base_cal[sel], block=blocks[sel])
            out[scope] = {"log_loss": mt.log_loss, "d": mt.d_ref, "se": mt.d_se,
                          "blocks_won": mt.blocks_won, "blocks": mt.blocks, "brier": mt.brier}
        r = out["tour"]
        print(f"  iter {it}: {args.block} tour log loss {r['log_loss']:.4f}  vs base {r['d']:+.4f} ± {r['se']:.4f}"
              f"  quarters {r['blocks_won']}/{r['blocks']}   atp {out['tour_atp']['d']:+.4f}  "
              f"wta {out['tour_wta']['d']:+.4f}")
        return out

    for it in range(args.iters + 1):
        if res is None:
            res, cal, _, _, secs = run_and_score(p)
            print(f"  replayed in {secs:.0f}s")
            history.append(score(p, res, cal, it))
        if it == args.iters:
            break
        for t in tours_fit:
            ds, dr, ses, ser, nobs = residual_fit(m, res, p, args.fit_from, args.cutoff, args.tour_only,
                                                  tour=t)
            print(f"\niter {it + 1}{' ' + t if t else ''}: residual fit on {nobs:,} matches with stats "
                  f"[{args.fit_from}, {args.cutoff}){' tour-level' if args.tour_only else ''}")
            print(f"  {'feature':<10}{'serve':>9}{'step':>9}{'SE':>8}{'return':>9}{'step':>9}{'SE':>8}")
            bs, br = betas[t][0] + ds, betas[t][1] + dr
            betas[t] = [bs, br]
            for j, f in enumerate(context.FEATURES):
                print(f"  {f:<10}{bs[j]:+9.4f}{ds[j]:+9.4f}{ses[j]:8.4f}{br[j]:+9.4f}{dr[j]:+9.4f}{ser[j]:8.4f}")
            p = p.with_context(bs, br, tour=t)
        res = None

    final = history[-1]
    if args.record:
        run_id = models_db.start_run(con, "points_v1", p.as_dict(), sweep_id="context-v1",
                                     notes=f"context offsets fit on residuals, base run {args.run}, "
                                           f"cutoff {args.cutoff}, {args.iters} iterations")
        if args.block == "val":
            scores = tuning.score_block(prep, spec.predict(res, p), lo, hi)
            tuning._write_metrics(con, run_id, "val", lo, hi, scores)
        models_db.finish_run(con, run_id)
        final["run_id"] = run_id
        print(f"recorded run {run_id}")
    con.close()
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps({"base_run": args.run, "block": args.block, "cutoff": args.cutoff,
                                     "fit_from": args.fit_from, "tour_only": args.tour_only,
                                     "history": history}, indent=2), encoding="utf-8")
    print(f"wrote {args.json} [{time.perf_counter() - t0:.0f}s]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
