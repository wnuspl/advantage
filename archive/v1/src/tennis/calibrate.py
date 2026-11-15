"""Walk-forward calibration and paired comparison of rating runs on identical matches.

Calibration (the same for every model): per test block (quarter starting S) and tour,
    P_cal = sigmoid(alpha * logit P)
with alpha a no-intercept logistic regression on that tour's tour-level scored matches
dated [S - years, S), i.e. past data only. Run 203 used the same scalar on Glicko-2 (fit
on rows with style vectors before S - 6 months; here every tour-level row before S).

With best_of (per-match best-of, aligned with prep.m), alpha is fitted per (block, tour,
best-of) on that tour's history of the same best-of; a group with fewer than min_rows
history matches falls back to the tour-wide alpha. points_v1's single alpha per tour
shrinks best-of-5 predictions the Markov roll-up already got about right (context report:
slope 1.19 in best-of-5 vs 0.93 in best-of-3).

Comparisons run on numpy arrays oriented p1 = smaller stable player key, so they can be
fed to composite_eval.metrics (paired log-loss difference with SE, blocks won).
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from .composite_eval import block_of
from .matchup.data import fit_alpha


def _add_years(d: str, years: int) -> str:
    x = date.fromisoformat(d)
    try:
        return x.replace(year=x.year + years).isoformat()
    except ValueError:                       # 29 February
        return x.replace(year=x.year + years, day=28).isoformat()


def oriented(prep, p_winner, idx):
    """(p1 win probability, y) at positions idx."""
    pw = np.asarray([p_winner[i] for i in idx], dtype=float)
    y = np.asarray([prep.winner_is_p1[i] for i in idx], dtype=float)
    return np.where(y == 1, pw, 1 - pw), y


def best_of_of(m) -> np.ndarray:
    """Best-of per match from points_rating.attach() (m.extra['points'].fmt)."""
    from .points_rating import FORMATS
    return np.array([FORMATS[k][0] for k in m.extra["points"].fmt[:len(m)]])


def calibrated(prep, p_winner, date_from: str, date_to: str, *, years: int = 10,
               step_months: int = 3, best_of=None, min_rows: int = 500
               ) -> tuple[np.ndarray, np.ndarray, list, dict]:
    """Calibrated P(p1 wins) for the tour-level test matches in [date_from, date_to].
    Returns (p_cal, y, idx, alphas {(tour, block) or (tour, block, best_of): alpha})."""
    idx = prep.indices(date_from, date_to, "tour")
    starts = np.array([prep.m.start[i] for i in idx])
    tours = np.array([prep.m.tour[i] for i in idx])
    blocks = block_of(starts, step_months)
    bos = np.array([best_of[i] for i in idx]) if best_of is not None else np.zeros(len(idx), int)
    p, y = oriented(prep, p_winner, idx)
    lg = _logit(p)
    out = np.empty_like(p)
    alphas = {}
    for tour in np.unique(tours):
        for blk in np.unique(blocks[tours == tour]):
            day_before = (date.fromisoformat(blk) - timedelta(days=1)).isoformat()
            hist = prep.indices(_add_years(blk, -years), day_before, f"tour_{tour}")
            hp, hy = oriented(prep, p_winner, hist)
            hl = _logit(hp)
            a = fit_alpha(hl, hy)
            alphas[(tour, blk)] = a
            s = (tours == tour) & (blocks == blk)
            out[s] = 1 / (1 + np.exp(-a * lg[s]))
            if best_of is None:
                continue
            hb = np.array([best_of[i] for i in hist])
            for bo in np.unique(bos[s]):
                h = hb == bo
                ab = fit_alpha(hl[h], hy[h]) if h.sum() >= min_rows else a
                alphas[(tour, blk, int(bo))] = ab
                sb = s & (bos == bo)
                out[sb] = 1 / (1 + np.exp(-ab * lg[sb]))
    return out, y, idx, alphas


def _logit(p):
    return np.log(np.clip(p, 1e-12, 1 - 1e-12) / np.clip(1 - p, 1e-12, 1))
