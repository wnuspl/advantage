"""Diagnostics for a run on one block (milestone A3).

Serve dominance: x_A + x_B (sum of the two pre-match serve-point logits, surface blend
applied) says how serve-dominated a match is expected to be. High values mean long sets
and tiebreaks, where the Markov roll-up amplifies small serve differences. The v1 open
question: is the model miscalibrated along this axis? Matches are split into quantile
bins; each bin reports log loss raw and calibrated and the recalibration slope of the
raw and calibrated probabilities (1 = calibrated; < 1 = overconfident).
"""
from __future__ import annotations

import numpy as np

from .runs import Scored
from .scoring import metrics


def serve_dominance(res, params, idx) -> np.ndarray:
    xa = res.xa + params.blend * res.xa_s
    xb = res.xb + params.blend * res.xb_s
    idx = np.asarray(idx, dtype=np.int64)
    return np.asarray(xa)[idx] + np.asarray(xb)[idx]


def by_dominance(sc: Scored, dom: np.ndarray, bins: int = 5) -> list[dict]:
    """Rows per quantile bin of dominance (tour-level matches of `sc`)."""
    if len(dom) < bins:
        return []
    edges = np.quantile(dom, np.linspace(0, 1, bins + 1))
    k = np.clip(np.searchsorted(edges, dom, side="right") - 1, 0, bins - 1)
    out = []
    for b in range(bins):
        s = k == b
        raw = metrics(sc.raw[s], sc.y[s])
        cal = metrics(sc.cal[s], sc.y[s], ref=sc.raw[s])
        out.append({"bin": b, "lo": float(edges[b]), "hi": float(edges[b + 1]), "n": int(s.sum()),
                    "atp_share": float((sc.tour[s] == "atp").mean()),
                    "ll_raw": raw.log_loss, "ll_cal": cal.log_loss, "cal_minus_raw": cal.d_ref,
                    "se": cal.d_se, "slope_raw": raw.cal_slope, "slope_cal": cal.cal_slope})
    return out


def format_rows(rows: list[dict]) -> str:
    lines = [f"  {'x_A+x_B':>15} {'n':>6} {'atp':>5} {'ll raw':>7} {'ll cal':>7} {'cal-raw':>8} "
             f"{'SE':>7} {'slope raw':>9} {'slope cal':>9}"]
    for r in rows:
        lines.append(f"  {r['lo']:+6.2f}..{r['hi']:+6.2f} {r['n']:6d} {r['atp_share']:5.2f} {r['ll_raw']:7.4f} "
                     f"{r['ll_cal']:7.4f} {r['cal_minus_raw']:+8.4f} {r['se']:7.4f} {r['slope_raw']:9.3f} "
                     f"{r['slope_cal']:9.3f}")
    return "\n".join(lines)
