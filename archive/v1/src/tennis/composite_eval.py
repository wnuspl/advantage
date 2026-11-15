"""Accuracy of the composite model against every baseline, on identical matches.

Inputs are walk-forward predictions already stored in models.db (`prediction` rows, p1 =
smaller stable key): the surface-Elo baseline (run 132), the Glicko-2 baseline (run 187)
and the matchup runs of one walk-forward run (scripts/matchup.py; `matchup_glicko` is the
alpha-calibrated Glicko benchmark, the others are composite = Glicko + matchup delta).
Only matches predicted by EVERY compared run are scored, so differences are paired.

Per model and scope: log loss (primary), Brier, accuracy, expected calibration error and a
logistic recalibration fit (intercept, slope: slope < 1 = overconfident), plus the paired
log-loss difference against a reference model with its standard error (per-match
differences; n is large, so the normal approximation is fine) and the number of test
blocks the model wins.

Verdict (MIDDLE_LAYER_SPEC.md §8): a model is better than the reference only if its pooled
log loss is lower AND its calibration is not worse (ECE no more than ECE_TOL above the
reference's). A lower log loss with |z| < 2 (z = mean paired difference / SE) is reported
as "within noise" rather than "better".
"""
from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass, field

import numpy as np

EPS = 1e-15
ECE_TOL = 0.002
SURF = ("hard", "clay", "grass")


@dataclass
class Rows:
    """Aligned per-match arrays of the compared runs (p1 orientation)."""
    match_key: np.ndarray
    date: np.ndarray            # edition start
    y: np.ndarray               # 1.0 if p1 won
    p: dict                     # label -> P(p1 wins)
    tour: np.ndarray
    tour_level: np.ndarray      # bool
    surface: np.ndarray         # 'hard' / 'clay' / 'grass' / 'carpet' / ''
    has: np.ndarray             # both players have style vectors (charted before the date)
    n_min: np.ndarray           # min charted matches before the date of the two players
    block: np.ndarray           # test block label (quarter start)

    def __len__(self) -> int:
        return len(self.y)


@dataclass
class Metrics:
    n: int
    log_loss: float
    brier: float
    accuracy: float
    ece: float
    cal_intercept: float
    cal_slope: float
    d_ref: float = float("nan")      # log loss minus the reference's, same rows
    d_se: float = float("nan")
    blocks_won: int = 0
    blocks: int = 0
    calibration: list = field(default_factory=list)   # [lo, hi, n, mean p, win rate]


def _ll_vec(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def calibration_bins(p, y, bins: int = 10) -> list:
    b = np.minimum((p * bins).astype(int), bins - 1)
    out = []
    for k in range(bins):
        s = b == k
        out.append([k / bins, (k + 1) / bins, int(s.sum()),
                    float(p[s].mean()) if s.any() else None, float(y[s].mean()) if s.any() else None])
    return out


def ece(p, y, bins: int = 10) -> float:
    if not len(p):
        return float("nan")
    return sum(c[2] * abs(c[3] - c[4]) for c in calibration_bins(p, y, bins) if c[2]) / len(p)


def recalibration(p, y, iters: int = 50) -> tuple[float, float]:
    """(a, b) of y ~ sigmoid(a + b * logit(p)) by Newton. Perfect calibration: (0, 1)."""
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
    n = len(y)
    if not n:
        return Metrics(0, *(float("nan"),) * 6)
    ll = _ll_vec(p, y)
    a, b = recalibration(p, y) if n >= 50 else (float("nan"), float("nan"))
    acc = np.where(p == 0.5, 0.5, ((p > 0.5) == (y == 1)).astype(float)).mean()
    out = Metrics(n, float(ll.mean()), float(np.mean((p - y) ** 2)), float(acc), ece(p, y), a, b,
                  calibration=calibration_bins(p, y))
    if ref is not None:
        d = ll - _ll_vec(ref, y)
        out.d_ref = float(d.mean())
        out.d_se = float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
        if block is not None:
            for blk in np.unique(block):
                s = block == blk
                out.blocks += 1
                out.blocks_won += int(d[s].mean() < 0)
    return out


def scopes(r: Rows, min_charted: int = 10) -> dict[str, np.ndarray]:
    t = r.tour_level
    well = r.has & (r.n_min >= min_charted)
    out = {"tour": t, "tour_both": t & r.has, "tour_well": t & well, "tour_none": t & ~r.has}
    for tr in ("atp", "wta"):
        out[f"tour_{tr}"] = t & (r.tour == tr)
        out[f"tour_both_{tr}"] = t & r.has & (r.tour == tr)
    for s in SURF:
        out[f"tour_{s}"] = t & (r.surface == s)
        out[f"tour_both_{s}"] = t & r.has & (r.surface == s)
    out["all"] = np.ones(len(r), dtype=bool)
    out["all_both"] = r.has
    years = sorted({d[:4] for d in r.date})
    for yv in years:
        out[f"tour_{yv}"] = t & np.array([d[:4] == yv for d in r.date])
    return {k: v for k, v in out.items() if v.any()}


def evaluate(r: Rows, ref: str, min_charted: int = 10) -> dict[str, dict[str, Metrics]]:
    """{scope: {label: Metrics}} with paired differences against `ref`."""
    out = {}
    for sc, mask in scopes(r, min_charted).items():
        y, blk = r.y[mask], r.block[mask]
        out[sc] = {lab: metrics(p[mask], y, ref=r.p[ref][mask], block=blk) for lab, p in r.p.items()}
    return out


def verdict(res: dict, label: str, ref: str, scope: str = "tour") -> str:
    m, b = res[scope][label], res[scope][ref]
    if label == ref:
        return "reference"
    better_ll = m.log_loss < b.log_loss
    cal_ok = m.ece <= b.ece + ECE_TOL
    z = m.d_ref / m.d_se if m.d_se and m.d_se == m.d_se and m.d_se > 0 else float("nan")
    sig = f"z = {z:.1f}" if z == z else ""
    if better_ll and cal_ok:
        return f"better ({sig})" if z < -2 else f"lower log loss, within noise ({sig})"
    if better_ll:
        return f"lower log loss but worse calibration ({sig})"
    return f"not better ({sig})"


# ------------------------------------------------------------------ loading

def block_of(dates: np.ndarray, step_months: int = 3) -> np.ndarray:
    def f(d):
        y, mth = int(d[:4]), int(d[5:7])
        q = (mth - 1) // step_months * step_months + 1
        return f"{y:04d}-{q:02d}-01"
    return np.array([f(d) for d in dates])


def load_predictions(con: sqlite3.Connection, runs: dict[str, int], date_from: str, date_to: str):
    """Inner join of the runs' prediction rows on match_key within [date_from, date_to].
    Returns (keys, dates, y, {label: p}, p1_keys, p2_keys)."""
    per = {}
    for lab, rid in runs.items():
        rows = con.execute("""SELECT match_key, edition_start, p1_key, p2_key, p_p1_wins, p1_won
                              FROM prediction WHERE run_id=? AND edition_start BETWEEN ? AND ?
                              AND p1_won IS NOT NULL""", (rid, date_from, date_to)).fetchall()
        if not rows:
            raise ValueError(f"run {rid} ({lab}) has no predictions in [{date_from}, {date_to}]")
        per[lab] = {r[0]: r for r in rows}
    common = set.intersection(*(set(v) for v in per.values()))
    if not common:
        raise ValueError("the runs share no predicted matches")
    keys = sorted(common, key=lambda k: (per[next(iter(per))][k][1], k))
    first = per[next(iter(per))]
    y = np.array([first[k][5] for k in keys], dtype=float)
    for lab, d in per.items():
        yy = np.array([d[k][5] for k in keys], dtype=float)
        if not np.array_equal(y, yy) or any(d[k][2] != first[k][2] for k in keys):
            raise ValueError(f"run {lab} disagrees on orientation or outcome")
    return (np.array(keys), np.array([first[k][1] for k in keys]), y,
            {lab: np.array([d[k][4] for k in keys]) for lab, d in per.items()},
            [first[k][2] for k in keys], [first[k][3] for k in keys])


def attach_attributes(con: sqlite3.Connection, keys, dates, p1_keys, p2_keys, src=None) -> dict:
    """tour, tour_level, surface from tennis.db; style-vector availability and charted
    counts from the heavy layer's charted totals (all sides, editions before the date)."""
    from . import elo, models_db
    mk = {v: k for k, v in models_db.match_keys(con).items()}
    pid = {v: k for k, v in models_db.player_keys(con).items()}
    m = elo.load_matches(con)
    pos = {mid: i for i, mid in enumerate(m.match_id)}
    idx = [pos[mk[k]] for k in keys]
    out = {"tour": np.array([m.tour[i] for i in idx]),
           "tour_level": np.array([bool(m.tour_level[i]) for i in idx]),
           "surface": np.array([m.surface[i] or "" for i in idx])}
    n = len(keys)
    has = np.zeros(n, dtype=bool)
    n_min = np.zeros(n, dtype=np.int64)
    if src is not None:
        from .style_features import SIDES
        for j in range(n):
            tour = out["tour"][j]
            counts = []
            for pk in (p1_keys[j], p2_keys[j]):
                p = pid.get(pk)
                got = [src.model.hist[(tour, s)].player_totals(p, dates[j]) if p is not None else None
                       for s in SIDES]
                counts.append(None if any(g is None for g in got) else int(got[0][2]))
            if counts[0] is not None and counts[1] is not None:
                has[j] = True
                n_min[j] = min(counts)
    out["has"], out["n_min"] = has, n_min
    return out


# ------------------------------------------------------------------ report

def _f(x, nd=4):
    return "" if x is None or x != x else f"{x:.{nd}f}"


def _sd(x, nd=4):
    return "" if x != x else ("+" if x > 0 else "") + f"{x:.{nd}f}"


SCOPE_DESC = {
    "tour": "tour-level main draws, every match (no vectors -> Glicko-only)",
    "tour_both": "tour-level, both players have style vectors (where the matchup layer acts)",
    "tour_well": "tour-level, both players well charted",
    "tour_none": "tour-level, at least one player without a vector (composite = calibrated Glicko)",
    "all": "every scored match the runs share",
    "all_both": "every scored match, both with vectors",
}


def report(r: Rows, res: dict, ref: str, runs: dict[str, int], *, min_charted: int, title: str,
           notes: list[str] = ()) -> str:
    labels = list(r.p)
    L = [f"# {title}", "",
         "Generated by `scripts/composite.py eval`. Every model is scored on the SAME matches "
         f"(inner join of stored walk-forward predictions): {len(r):,} matches, "
         f"{min(r.date)} to {max(r.date)} (edition starts).", "",
         "Runs: " + ", ".join(f"`{lab}` = run {rid}" for lab, rid in runs.items()) + ".", "",
         f"Reference for paired differences: `{ref}`. delta = log loss minus the reference's on the same "
         "rows (negative is better), SE = standard error of the per-match difference, blocks = "
         "quarterly test blocks where the model's mean log loss is lower. ECE = expected "
         "calibration error (10 bins); recalibration = (a, b) of y ~ sigmoid(a + b logit p), "
         "perfect = (0, 1), b < 1 means overconfident.", ""]
    L += list(notes) + ([""] if notes else [])
    L += ["## Verdict (pooled `tour`)", "",
          f"Better = lower pooled log loss than `{ref}` and ECE no more than {ECE_TOL} above it.", "",
          "| model | log loss | delta | SE | verdict | also on `tour_both` |", "|---|---|---|---|---|---|"]
    for lab in labels:
        m = res["tour"][lab]
        L.append(f"| {lab} | {_f(m.log_loss)} | {_sd(m.d_ref)} | {_f(m.d_se)} | {verdict(res, lab, ref)} | "
                 + (verdict(res, lab, ref, "tour_both") if "tour_both" in res else "") + " |")
    order = ["tour", "tour_both", "tour_well", "tour_none", "tour_atp", "tour_wta", "tour_both_atp",
             "tour_both_wta", "tour_hard", "tour_clay", "tour_grass", "tour_both_hard", "tour_both_clay",
             "tour_both_grass", "all", "all_both"]
    L += ["", "## Pooled metrics by scope", ""]
    for sc in order:
        if sc not in res:
            continue
        n = res[sc][labels[0]].n
        d = SCOPE_DESC.get(sc, "")
        if sc == "tour_well":
            d += f" (>= {min_charted} charted matches each)"
        L += [f"### `{sc}` (n = {n:,})" + (f": {d}" if d else ""), "",
              "| model | log loss | delta | SE | blocks | Brier | accuracy | ECE | recalibration (a, b) |",
              "|---|---|---|---|---|---|---|---|---|"]
        for lab in labels:
            m = res[sc][lab]
            L.append(f"| {lab} | {_f(m.log_loss)} | {_sd(m.d_ref)} | {_f(m.d_se)} | "
                     f"{m.blocks_won}/{m.blocks} | {_f(m.brier)} | {_f(m.accuracy, 3)} | {_f(m.ece)} | "
                     f"({_f(m.cal_intercept, 3)}, {_f(m.cal_slope, 3)}) |")
        L.append("")
    years = sorted(k for k in res if k.startswith("tour_") and k[5:].isdigit())
    if years:
        L += ["## Log loss by year (`tour`)", "", "| year | n | " + " | ".join(labels) + " |",
              "|---|---|" + "---|" * len(labels)]
        for k in years:
            L.append(f"| {k[5:]} | {res[k][labels[0]].n:,} | " + " | ".join(_f(res[k][lab].log_loss) for lab in labels)
                     + " |")
        L.append("")
    if "tour_both" in res:
        L += ["## Calibration, `tour_both` (mean predicted / win rate / n, p1 orientation)", "",
              "| bin | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
        cals = [res["tour_both"][lab].calibration for lab in labels]
        for bi in range(len(cals[0])):
            cells = [f"{_f(c[bi][3], 3)} / {_f(c[bi][4], 3)} / {c[bi][2]:,}" if c[bi][2] else "" for c in cals]
            L.append(f"| {cals[0][bi][0]:.1f}-{cals[0][bi][1]:.1f} | " + " | ".join(cells) + " |")
        L.append("")
    L += ["## Attribution", "", "Sackmann-derived data (tennis_atp, tennis_wta via the "
          "`Aneeshers/tennis-sackmann-archive` mirror; Match Charting Project) is CC BY-NC-SA 4.0, "
          "by Jeff Sackmann and MCP contributors.", ""]
    return "\n".join(L)
