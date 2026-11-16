"""Probabilistic scoring (from v1 evaluate.py and composite_eval.metrics).

Predictions come in as P(actual winner wins). Log loss and Brier are symmetric in player
order; calibration needs an order that is not winner-first, so each match is oriented by
Prepared.winner_is_p1 (p1 = the smaller stable player key) and bins mix wins and losses.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

EPS = 1e-15


@dataclass
class Metrics:
    n: int
    log_loss: float
    brier: float
    accuracy: float                  # reported, never used to choose
    ece: float
    cal_intercept: float             # y ~ sigmoid(a + b logit p): perfect = (0, 1)
    cal_slope: float                 # b < 1 = overconfident
    calibration: list = field(default_factory=list)   # [lo, hi, n, mean p, win rate]
    d_ref: float = float("nan")      # log loss minus the reference's, same rows
    d_se: float = float("nan")
    blocks_won: int = 0
    blocks: int = 0


def oriented(p_winner, winner_is_p1, idx) -> tuple[np.ndarray, np.ndarray]:
    """(P(p1 wins), y) at positions idx."""
    pw = np.asarray([p_winner[i] for i in idx], dtype=float)
    y = np.asarray([winner_is_p1[i] for i in idx], dtype=float)
    return np.where(y == 1, pw, 1 - pw), y


def log_losses(p, y) -> np.ndarray:
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def calibration_bins(p, y, bins: int = 10) -> list:
    b = np.minimum((p * bins).astype(int), bins - 1)
    out = []
    for k in range(bins):
        s = b == k
        out.append([round(k / bins, 3), round((k + 1) / bins, 3), int(s.sum()),
                    float(p[s].mean()) if s.any() else None, float(y[s].mean()) if s.any() else None])
    return out


def ece(p, y, bins: int = 10) -> float:
    if not len(p):
        return float("nan")
    return sum(c[2] * abs(c[3] - c[4]) for c in calibration_bins(p, y, bins) if c[2]) / len(p)


def recalibration(p, y, iters: int = 50) -> tuple[float, float]:
    """(a, b) of y ~ sigmoid(a + b logit p) by Newton."""
    z = np.log(np.clip(p, 1e-9, 1 - 1e-9)) - np.log1p(-np.clip(p, 1e-9, 1 - 1e-9))
    X = np.column_stack([np.ones_like(z), z])
    w = np.array([0.0, 1.0])
    for _ in range(iters):
        q = 0.5 * (1.0 + np.tanh(0.5 * (X @ w)))
        g = X.T @ (q - y)
        H = (X * (q * (1 - q))[:, None]).T @ X
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        w = w - step
        if np.abs(step).max() < 1e-10:
            break
    return float(w[0]), float(w[1])


def metrics(p, y, *, ref=None, block=None) -> Metrics:
    """p = P(p1 wins), y = p1 won (canonical orientation). With `ref` (same rows): paired
    log-loss difference, its SE, and the blocks (e.g. quarters) where p beats ref."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    n = len(y)
    if not n:
        return Metrics(0, *(float("nan"),) * 6)
    ll = log_losses(p, y)
    a, b = recalibration(p, y) if n >= 50 else (float("nan"), float("nan"))
    acc = np.where(p == 0.5, 0.5, ((p > 0.5) == (y == 1)).astype(float)).mean()
    out = Metrics(n, float(ll.mean()), float(np.mean((p - y) ** 2)), float(acc), ece(p, y), a, b,
                  calibration=calibration_bins(p, y))
    if ref is not None:
        d = ll - log_losses(ref, y)
        out.d_ref = float(d.mean())
        out.d_se = float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
        if block is not None:
            block = np.asarray(block)
            for blk in np.unique(block):
                s = block == blk
                out.blocks += 1
                out.blocks_won += int(d[s].mean() < 0)
    return out
