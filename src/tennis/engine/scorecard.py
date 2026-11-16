"""Live scorecard (track B5, docs/PLAN.md): model vs market on what the watch recorded.

    sc = scorecard(con)            # con on live.db (engine/recorder.py)

Reads only live.db: snapshots of settled matches (a winner A or B), joined to the watch that
recorded them, so every number is per predictor (watch.predictor).

pre-match   the last priced snapshot before the match went live (status not_started), one
            per (predictor, match): log loss of model and market, the paired difference
            with its SE. This is the live counterpart of the closing-odds benchmark.
in play     every priced live snapshot, weighted so each match counts once (mean log loss
            per match, then averaged over matches; SE across matches). Snapshots are not
            independent within a match, so n is the number of matches.
feed lag    whether the market moves before the score feed shows the point. For
            consecutive live polls of one match, the model's change (driven only by the
            feed's score) is correlated with the market's change k polls later. A peak at
            k < 0 means the market moved first: the feed lags the court (or the market
            anticipates). Resolution is the poll spacing (median gap reported), so a
            60-second watch measures only lags of a minute or more.

Small samples are expected for weeks: every number carries its n and SE.
"""
from __future__ import annotations

import math
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime

EPS = 1e-6


def _ll(p: float, y: int) -> float:
    p = min(max(p, EPS), 1 - EPS)
    return -math.log(p if y else 1 - p)


@dataclass
class Paired:
    n: int = 0
    model: float = float("nan")         # mean log loss
    market: float = float("nan")
    d: float = float("nan")             # model - market (negative: the model did better)
    se: float = float("nan")

    @staticmethod
    def of(pairs: list[tuple[float, float]]) -> "Paired":
        """pairs: (model log loss, market log loss) per unit (a match)."""
        n = len(pairs)
        if not n:
            return Paired()
        ds = [a - b for a, b in pairs]
        mean = sum(ds) / n
        se = math.sqrt(sum((x - mean) ** 2 for x in ds) / (n - 1) / n) if n > 1 else float("nan")
        return Paired(n, sum(a for a, _ in pairs) / n, sum(b for _, b in pairs) / n, mean, se)


@dataclass
class Lag:
    n_steps: int = 0                    # consecutive live poll pairs with both prices
    n_matches: int = 0
    median_gap_s: float = float("nan")
    corr: dict = field(default_factory=dict)   # k -> corr(d_model[t], d_market[t+k])


@dataclass
class PredictorCard:
    predictor: str
    settled: int                        # settled matches this predictor priced at least once
    pre_match: Paired
    in_play: Paired
    lag: Lag


def _settled_rows(con: sqlite3.Connection):
    return con.execute(
        """SELECT w.predictor, s.event, p.as_of, s.status, s.p_model, s.p_market,
                  CASE st.winner WHEN 'a' THEN 1 ELSE 0 END AS y
           FROM snapshot s JOIN poll p USING (poll_id) JOIN watch w USING (watch_id)
           JOIN settlement st ON st.event = s.event
           WHERE st.winner IS NOT NULL AND s.skip = '' AND s.p_model IS NOT NULL AND s.p_market IS NOT NULL
           ORDER BY w.predictor, s.event, p.as_of""").fetchall()


def _corr(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return sxy / (sx * sy) if sx > 0 and sy > 0 else float("nan")


def feed_lag(series: dict[str, list[tuple[str, float, float]]], ks=(-2, -1, 0, 1, 2),
             max_gap_s: float = 300.0) -> Lag:
    """series: event -> [(as_of, p_model, p_market)] of live snapshots in time order. Steps
    across a gap over max_gap_s (a missed poll, a sleeping laptop) are dropped."""
    dm_all, dk_all, gaps = [], {k: ([], []) for k in ks}, []
    n_steps, n_matches = 0, 0
    for rows in series.values():
        t = [datetime.fromisoformat(r[0]) for r in rows]
        steps = []                                     # (d_model, d_market) or None across a gap
        for i in range(1, len(rows)):
            g = (t[i] - t[i - 1]).total_seconds()
            if g > max_gap_s:
                steps.append(None)
                continue
            gaps.append(g)
            steps.append((rows[i][1] - rows[i - 1][1], rows[i][2] - rows[i - 1][2]))
        used = False
        for i, s in enumerate(steps):
            if s is None:
                continue
            n_steps += 1
            used = True
            dm_all.append(s[0])
            for k in ks:
                j = i + k
                if 0 <= j < len(steps) and steps[j] is not None and all(
                        steps[x] is not None for x in range(min(i, j), max(i, j) + 1)):
                    dk_all[k][0].append(s[0])
                    dk_all[k][1].append(steps[j][1])
        n_matches += used
    gaps.sort()
    med = gaps[len(gaps) // 2] if gaps else float("nan")
    return Lag(n_steps, n_matches, med, {k: _corr(*dk_all[k]) for k in ks})


def scorecard(con: sqlite3.Connection) -> list[PredictorCard]:
    by_pred: dict[str, dict[str, list]] = {}
    for pred, ev, as_of, status, pm, pk, y in _settled_rows(con):
        by_pred.setdefault(pred, {}).setdefault(ev, []).append((as_of, status, pm, pk, y))
    out = []
    for pred, events in sorted(by_pred.items()):
        pre, inplay, live_series = [], [], {}
        for ev, rows in events.items():
            nots = [r for r in rows if r[1] == "not_started"]
            lives = [r for r in rows if r[1] == "live"]
            first_live = lives[0][0] if lives else None
            nots = [r for r in nots if first_live is None or r[0] < first_live]
            if nots:
                _, _, pm, pk, y = nots[-1]
                pre.append((_ll(pm, y), _ll(pk, y)))
            if lives:
                y = lives[0][4]
                inplay.append((sum(_ll(r[2], y) for r in lives) / len(lives),
                               sum(_ll(r[3], y) for r in lives) / len(lives)))
                live_series[ev] = [(r[0], r[2], r[3]) for r in lives]
        out.append(PredictorCard(pred, len(events), Paired.of(pre), Paired.of(inplay), feed_lag(live_series)))
    return out


def to_json(cards: list[PredictorCard]) -> list[dict]:
    return [asdict(c) for c in cards]


def format_cards(cards: list[PredictorCard]) -> str:
    if not cards:
        return "no settled, priced matches in live.db yet"
    lines = []
    for c in cards:
        lines.append(f"## {c.predictor}: {c.settled} settled matches priced")
        lines.append(f"  {'':10s} {'n':>4s} {'model':>7s} {'market':>7s} {'diff':>8s} {'SE':>7s}")
        for name, p in (("pre-match", c.pre_match), ("in play", c.in_play)):
            lines.append(f"  {name:10s} {p.n:4d} {p.model:7.4f} {p.market:7.4f} {p.d:+8.4f} {p.se:7.4f}")
        lg = c.lag
        cs = "  ".join(f"k={k:+d}: {v:+.2f}" for k, v in lg.corr.items())
        lines.append(f"  feed lag: {lg.n_steps} live poll steps in {lg.n_matches} matches, median gap "
                     f"{lg.median_gap_s:.0f}s; corr(model move, market move k polls later): {cs}")
    lines.append("diff = model - market log loss (negative: model better); in play weights each match once.")
    lines.append("feed lag: a peak at k < 0 means the market moved before the feed showed the score.")
    return "\n".join(lines)


def phone_summary(cards: list[PredictorCard], counts: dict | None = None) -> tuple[str, str]:
    """(title, message) for a daily push: short enough for a phone notification."""
    if not cards:
        return "advantage: scorecard", "no settled, priced matches in live.db yet"
    c = max(cards, key=lambda x: x.settled)          # the predictor with the most data
    lines = [f"{c.predictor}: {c.settled} settled matches"]
    for name, p in (("pre-match", c.pre_match), ("in play", c.in_play)):
        if p.n:
            se = "" if math.isnan(p.se) else f" ± {p.se:.3f}"
            lines.append(f"{name}: model - market {p.d:+.3f}{se} (n {p.n})")
    lg = c.lag
    if lg.n_steps:
        lines.append(f"feed lag corr k=-1 {lg.corr.get(-1, float('nan')):+.2f}, "
                     f"k=0 {lg.corr.get(0, float('nan')):+.2f} ({lg.n_steps} steps)")
    if counts:
        lines.append(f"last poll {counts.get('last_poll', '?')}; alerts sent {counts.get('alerts_sent', 0)}")
    lines.append("negative diff = model beat the market")
    return f"advantage: scorecard, {c.settled} settled", "\n".join(lines)
