"""Walk-forward calibration and paired comparison of rating runs on identical matches.

Calibration (the same for every model): per test block (quarter starting S) and tour,
    P_cal = sigmoid(alpha * logit P)
with alpha a no-intercept logistic regression on that tour's tour-level scored matches
dated [S - years, S), i.e. past data only. Run 203 used the same scalar on Glicko-2 (fit
on rows with style vectors before S - 6 months; here every tour-level row before S).

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


def calibrated(prep, p_winner, date_from: str, date_to: str, *, years: int = 10,
               step_months: int = 3) -> tuple[np.ndarray, np.ndarray, list, dict]:
    """Calibrated P(p1 wins) for the tour-level test matches in [date_from, date_to].
    Returns (p_cal, y, idx, alphas {(tour, block): alpha})."""
    idx = prep.indices(date_from, date_to, "tour")
    starts = np.array([prep.m.start[i] for i in idx])
    tours = np.array([prep.m.tour[i] for i in idx])
    blocks = block_of(starts, step_months)
    p, y = oriented(prep, p_winner, idx)
    lg = np.log(np.clip(p, 1e-12, 1 - 1e-12) / np.clip(1 - p, 1e-12, 1))
    out = np.empty_like(p)
    alphas = {}
    for tour in np.unique(tours):
        for blk in np.unique(blocks[tours == tour]):
            day_before = (date.fromisoformat(blk) - timedelta(days=1)).isoformat()
            hist = prep.indices(_add_years(blk, -years), day_before, f"tour_{tour}")
            hp, hy = oriented(prep, p_winner, hist)
            hl = np.log(np.clip(hp, 1e-12, 1 - 1e-12) / np.clip(1 - hp, 1e-12, 1))
            a = fit_alpha(hl, hy)
            alphas[(tour, blk)] = a
            s = (tours == tour) & (blocks == blk)
            out[s] = 1 / (1 + np.exp(-a * lg[s]))
    return out, y, idx, alphas
