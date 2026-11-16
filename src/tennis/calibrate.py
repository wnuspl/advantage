"""Walk-forward calibration of match probabilities (rewritten from v1 calibrate.py, milestone A1).

Per block (a quarter starting S) and tour,
    P_cal = sigmoid(alpha * logit P)
with alpha a no-intercept logistic regression on that tour's tour-level scored matches whose
edition started in [S - years, S): past data only. With best-of per match, alpha is fitted per
(block, tour, best-of) on the history of that best-of; a best-of with fewer than `min_rows`
history matches falls back to the tour alpha. (points_v1's single alpha per tour shrank
best-of-5 predictions the Markov roll-up already got about right.)

Everything is winner-oriented: P is P(actual winner wins), y = 1. The no-intercept fit is
symmetric in player order (flipping a row maps logit -> -logit and y -> 1 - y, which leaves
its likelihood term unchanged), so this gives the same alpha as any canonical orientation.

Pure logic over a matches.Matches and an aligned array of P(winner).
"""
from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np

from .matches import Matches, scope_indices

YEARS = 10
MIN_ROWS = 500
TOURS = ("atp", "wta")
BEST_OF = (1, 3, 5)


def fit_alpha(logit: np.ndarray, y: np.ndarray, iters: int = 50) -> float:
    """Scalar logistic regression y ~ sigmoid(alpha * logit), no intercept (Newton from 1).
    No rows (or no information) gives 1: no calibration."""
    logit, y = np.asarray(logit, float), np.asarray(y, float)
    a = 1.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-a * logit))
        g = np.sum((p - y) * logit)
        h = np.sum(p * (1 - p) * logit * logit)
        if h <= 0:
            break
        step = g / h
        a -= step
        if abs(step) < 1e-10:
            break
    if not math.isfinite(a):
        raise FloatingPointError("alpha diverged")
    return float(a)


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def apply(p, alpha: float):
    """sigmoid(alpha * logit p); scalar in, scalar out."""
    out = 1 / (1 + np.exp(-alpha * logit(p)))
    return float(out) if np.ndim(out) == 0 else out


def add_years(d: str, years: int) -> str:
    x = date.fromisoformat(d)
    try:
        return x.replace(year=x.year + years).isoformat()
    except ValueError:                       # 29 February
        return x.replace(year=x.year + years, day=28).isoformat()


def block_start(day: str, step_months: int = 3) -> str:
    """First day of the block (quarter by default) containing ISO date `day`."""
    y, mth = int(day[:4]), int(day[5:7])
    return f"{y:04d}-{(mth - 1) // step_months * step_months + 1:02d}-01"


def block_of(dates, step_months: int = 3) -> np.ndarray:
    return np.array([block_start(d, step_months) for d in dates])


def alphas_at(m: Matches, p_winner, block: str, *, years: int = YEARS, best_of=None,
              min_rows: int = MIN_ROWS, tours=TOURS) -> dict:
    """Alphas for predictions in the block starting `block`, fitted on [block - years, block):
    {tour: alpha} and, with best_of (per match, aligned with m), {(tour, bo): alpha} for
    bo in 1/3/5 (the tour alpha when that best-of has < min_rows history matches)."""
    day_before = (date.fromisoformat(block) - timedelta(days=1)).isoformat()
    out: dict = {}
    for tour in tours:
        hist = scope_indices(m, add_years(block, -years), day_before, f"tour_{tour}")
        hl = logit([p_winner[i] for i in hist])
        a = fit_alpha(hl, np.ones(len(hist)))
        out[tour] = a
        if best_of is None:
            continue
        hb = np.array([best_of[i] for i in hist], dtype=int)
        for bo in BEST_OF:
            h = hb == bo
            out[(tour, bo)] = fit_alpha(hl[h], np.ones(int(h.sum()))) if h.sum() >= min_rows else a
    return out


def alpha_for(alphas: dict, tour: str, best_of: int | None = None) -> float:
    """The alpha a prediction uses: per (tour, best-of) if fitted, else per tour, else 1."""
    if best_of is not None and (tour, best_of) in alphas:
        return alphas[(tour, best_of)]
    return alphas.get(tour, 1.0)


def calibrated(m: Matches, p_winner, date_from: str, date_to: str, *, years: int = YEARS,
               step_months: int = 3, best_of=None, min_rows: int = MIN_ROWS, raw_tours=()
               ) -> tuple[np.ndarray, list[int], dict]:
    """Calibrated P(winner wins) for the tour-level scored matches with edition start in
    [date_from, date_to], each with the alpha of its own block and tour (and best-of).
    Tours in `raw_tours` get alpha 1 (left raw).
    Returns (p_cal, idx, alphas {(tour, block) or (tour, block, best_of): alpha})."""
    idx = scope_indices(m, date_from, date_to, "tour")
    tours = np.array([m.tour[i] for i in idx])
    blocks = block_of([m.start[i] for i in idx], step_months)
    bos = np.array([best_of[i] for i in idx], dtype=int) if best_of is not None else None
    lg = logit([p_winner[i] for i in idx])
    out = np.empty(len(idx))
    alphas: dict = {}
    for blk in np.unique(blocks):
        here = blocks == blk
        tb = sorted(set(tours[here].tolist()))
        al = alphas_at(m, p_winner, str(blk), years=years, best_of=best_of, min_rows=min_rows, tours=tb)
        for t in raw_tours:
            al = {k: (1.0 if k == t or (isinstance(k, tuple) and k[0] == t) else v) for k, v in al.items()}
        for tour in tb:
            s = here & (tours == tour)
            alphas[(tour, str(blk))] = al[tour]
            if bos is None:
                out[s] = 1 / (1 + np.exp(-al[tour] * lg[s]))
                continue
            for bo in np.unique(bos[s]).tolist():
                sb = s & (bos == bo)
                a = alpha_for(al, tour, bo)
                alphas[(tour, str(blk), bo)] = a
                out[sb] = 1 / (1 + np.exp(-a * lg[sb]))
    return out, idx, alphas
