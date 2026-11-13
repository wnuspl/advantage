"""Probabilistic scoring for walk-forward predictions: log loss, Brier, calibration.

Predictions are given as P(actual winner wins). Log loss and Brier are symmetric in
player order, so that is enough for them. Calibration needs an order that is not
'winner first': each match is oriented by a fixed flag (p1 = the player with the
smaller stable key), so bins mix wins and losses honestly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

EPS = 1e-15


@dataclass
class Score:
    n: int
    log_loss: float
    brier: float
    accuracy: float
    calibration: list   # [[bin_lo, bin_hi, n, mean_p, win_rate], ...]


def score(p_winner, winner_is_p1, idx, bins: int = 10) -> Score:
    """Score the predictions at positions `idx`.

    p_winner[i]     P(actual winner of match i wins), made before the match
    winner_is_p1[i] 1 if the winner is p1 in the canonical (non-winner-first) order
    """
    n = ll = br = acc = 0.0
    cal = [[0, 0.0, 0.0] for _ in range(bins)]      # n, sum p1, sum p1_won
    for i in idx:
        pw = min(max(p_winner[i], EPS), 1 - EPS)
        n += 1
        ll -= math.log(pw)
        br += (1.0 - pw) ** 2
        acc += 1.0 if pw > 0.5 else (0.5 if pw == 0.5 else 0.0)
        y = winner_is_p1[i]
        p1 = pw if y else 1.0 - pw
        b = min(int(p1 * bins), bins - 1)
        cal[b][0] += 1
        cal[b][1] += p1
        cal[b][2] += y
    if not n:
        return Score(0, float("nan"), float("nan"), float("nan"), [])
    calibration = [[round(b / bins, 3), round((b + 1) / bins, 3), c[0],
                    round(c[1] / c[0], 4) if c[0] else None, round(c[2] / c[0], 4) if c[0] else None]
                   for b, c in enumerate(cal)]
    return Score(int(n), ll / n, br / n, acc / n, calibration)
