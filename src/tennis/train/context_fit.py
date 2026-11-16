"""Fit the context coefficients (cs_*/cr_*) on serve-point residuals (from v1 context_fit.py).

Iteration 0 replays the starting setting. Each iteration regresses the serve-point
residuals of the current run on the context features (weighted least squares, serve and
return coefficients together) and adds the step to the coefficients:

    z = (k - n p) / (n p (1 - p)),   var z = 1 / (n p (1 - p)) + obs_var + var(x)
    z ~ beta_serve . c_server - beta_ret . c_returner         (ridge l2 = 1)

with p from the run's pre-match x (context offset included, surface blend 1, as in the
update). Offsets feed the rating updates, so the next run has different ratings and the
fit repeats until the step is small: the fixed point is where context explains none of
what is left. Only matches with serve stats dated [fit_from, cutoff) are used (default
cutoff 2019-01-01: validation stays out of sample for the coefficients).
"""
from __future__ import annotations

import numpy as np

from .. import context, points_rating
from .data import BLOCKS, Prepared
from .runs import Scored

PLAYED = {"completed", "retired"}


def residual_fit(m, res, params, fit_from: str, cutoff: str, tour_only: bool = False, l2: float = 1.0,
                 tour: str | None = None):
    """(delta_serve, delta_ret, se_serve, se_ret, n_matches) from one run's residuals."""
    d = m.extra["points"]
    cw, cl = m.extra["context"]
    n = len(m)
    idx = np.array([i for i in range(n)
                    if d.w_sp[i] > 0 and d.l_sp[i] > 0 and fit_from <= m.start[i] < cutoff
                    and m.status[i] in PLAYED and (m.tour_level[i] or not tour_only)
                    and (tour is None or m.tour[i] == tour)], dtype=np.int64)
    rows_x, rows_z, rows_w = [], [], []
    for xa, va, k, nn, cs, cr in (
            (res.xa + res.xa_s, res.va + res.va_s, d.w_sw, d.w_sp, cw, cl),     # winner serving
            (res.xb + res.xb_s, res.vb + res.vb_s, d.l_sw, d.l_sp, cl, cw)):    # loser serving
        p = 1 / (1 + np.exp(-xa[idx]))
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


def fit(prep: Prepared, start: points_rating.PointsParams, *, iters: int = 3, fit_from: str = "2000-01-01",
        cutoff: str = "2019-01-01", tour_only: bool = False, per_tour: bool = False,
        block=BLOCKS["val"], ref: Scored | None = None, log=print):
    """Iterate the residual fit from `start`. Returns (params, history, last result, last
    Scored). history: one dict per scored iteration (params, val metrics vs `ref`)."""
    m = prep.m
    if "context" not in m.extra:
        raise ValueError("context features missing: prepare() attaches them")
    p = start
    if per_tour and not p.ctx_per_tour:
        p = p.with_context(*p.context_betas("atp"), tour="wta")
        p = points_rating.PointsParams(**{**p.as_dict(), "ctx_per_tour": 1.0})
    tours = ("atp", "wta") if p.ctx_per_tour else (None,)
    history = []
    res = sc = None
    for it in range(iters + 1):
        res = points_rating.run(m, p)
        sc = Scored(prep, res.p_winner, block)
        s = sc.scores(ref)
        history.append({"iter": it, "params": p.as_dict(),
                        "scores": {k: {"log_loss": v.log_loss, "d": v.d_ref, "se": v.d_se,
                                       "blocks_won": v.blocks_won, "blocks": v.blocks} for k, v in s.items()}})
        t = s["tour_cal"]
        log(f"  iter {it}: val cal {t.log_loss:.4f} raw {s['tour'].log_loss:.4f}"
            + ("" if ref is None else f"  vs ref {t.d_ref:+.4f} ± {t.d_se:.4f}, quarters {t.blocks_won}/{t.blocks}"))
        if it == iters:
            break
        for tr in tours:
            ds, dr, ses, ser, nobs = residual_fit(m, res, p, fit_from, cutoff, tour_only, tour=tr)
            bs, br = p.context_betas(tr or "atp")
            bs, br = bs + ds, br + dr
            log(f"  fit {it + 1}{' ' + tr if tr else ''} on {nobs:,} matches: max step "
                f"{max(np.abs(ds).max(), np.abs(dr).max()):.4f}")
            p = p.with_context(bs, br, tour=tr)
    return p, history, res, sc
