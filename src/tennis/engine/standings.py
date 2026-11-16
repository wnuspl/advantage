"""Rating standings at weekly cutoffs, for the ratings viewer (milestone C, "rankings at
any date"; v1 eoy_serve_return.py's logic, weekly instead of yearly).

    for week in weekly(m, params, mondays("2010-01-04", today), players): ...

A cutoff C (a Monday) is the state after every edition that started before C, the same
rule the predictor and the A5 snapshots use: an edition is never split, and nothing from
an edition starting on or after C is seen. One replay gives every week (run(on_snap=)).

Values per player (logits of a serve point; +0.1 is about +2 pp of serve points won):
    s, r          stored serve / return ratings (age-neutral, as the state holds them)
    fs, fr        the run's age offset at the cutoff (0 without a birth date); s + fs is
                  what the model expects of the player now
    sd_s, sd_r    standard deviations, grown by inactivity to the cutoff
    surf[x]       (ds, dr, n) per surface: deviations and matches there; None if never
                  played there (the model then uses ds = dr = 0)
    n, n_stat     rated matches; of those, with serve-point stats
    last          start date of the last rated edition
Rust/load offsets belong to a specific match and are left out. Tours are on separate scales.
"""
from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from .. import context, points_rating

SURFACES = ("hard", "clay", "grass")
AGE = slice(5, 9)   # the age features in context.FEATURES


def mondays(since: str, until: str) -> list[str]:
    """Every Monday from the first on/after `since` to the first on/after `until`."""
    d = date.fromisoformat(since)
    d += timedelta(days=(7 - d.weekday()) % 7)
    end = date.fromisoformat(until)
    out = []
    while True:
        out.append(d.isoformat())
        if d >= end:
            return out
        d += timedelta(days=7)


def age_offsets(p: points_rating.PointsParams, tour: str, age: float | None) -> tuple[float, float]:
    """(serve, return) age offset of the run at `age` ((0, 0) when unknown)."""
    if age is None or not p.has_context():
        return 0.0, 0.0
    c = np.zeros(context.K)
    c[5], c[6], c[7], c[8] = max(0.0, 22 - age), max(0.0, 26 - age), max(0.0, age - 28), max(0.0, age - 32)
    bs, br = p.context_betas(tour)
    return float(bs[AGE] @ c[AGE]), float(br[AGE] @ c[AGE])


@dataclass
class Player:
    tour: str
    dob: int | None   # date ordinal


@dataclass
class Week:
    cutoff: str
    rows: dict        # tour -> list of row dicts, sorted by s + r + fs + fr (best first)
    base: dict        # tour -> surface -> base serve-point logit


def rows_at(st: points_rating.PointsState, cutoff: str, players: dict, n_stat: dict, *,
            min_matches: int = 10, active_days: int = 365, top: int | None = 500) -> dict:
    """tour -> rows for players with min_matches rated matches and a match in the
    active_days before the cutoff, the top `top` per tour by age-adjusted s + r."""
    p, day = st.params, date.fromisoformat(cutoff).toordinal()
    by_tour: dict[str, list] = {}
    for pid, s in st.serve.items():
        n, last = st.count.get(pid, 0), st.last[pid]
        if n < min_matches or last < day - active_days:
            continue
        info = players.get(pid)
        if info is None:
            continue
        r, el = st.ret[pid], max(day - last, 0)
        age = (day - info.dob) / 365.25 if info.dob else None
        fs, fr = age_offsets(p, info.tour, age)
        surf = {}
        for x in SURFACES:
            k = (pid, x)
            surf[x] = (st.sserve[k], st.sret[k], st.scount[k]) if k in st.sserve else None
        by_tour.setdefault(info.tour, []).append({
            "player_id": pid, "s": s, "r": r, "fs": fs, "fr": fr,
            "sd_s": math.sqrt(min(st.vs[pid] + p.q * el, p.v_init)),
            "sd_r": math.sqrt(min(st.vr[pid] + p.q * el, p.v_init)),
            "n": n, "n_stat": n_stat.get(pid, 0), "last": date.fromordinal(last).isoformat(),
            "age": age, "surf": surf})
    for rows in by_tour.values():
        rows.sort(key=lambda x: (-(x["s"] + x["r"] + x["fs"] + x["fr"]), x["player_id"]))
        if top is not None:
            del rows[top:]
    return by_tour


def base_at(st: points_rating.PointsState, tours) -> dict:
    return {t: {x: st.base.get((t, x), points_rating.BASE_INIT.get(t, 0.4)) for x in SURFACES} for t in tours}


def weekly(m, params: points_rating.PointsParams, cutoffs: list[str], players: dict, *,
           min_matches: int = 10, active_days: int = 365, top: int | None = 500, on_week) -> None:
    """Replay `m` (attached, in processing order) once and call on_week(Week) at each
    cutoff (sorted ISO dates), before the first match of an edition starting on or after it."""
    idx = [bisect_left(m.start, c) for c in cutoffs]
    d = m.extra["points"]
    n_stat: dict[int, int] = {}
    done = [0]   # matches counted into n_stat so far

    def snap(k: int, st: points_rating.PointsState) -> None:
        for i in range(done[0], min(idx[k], len(m))):
            if d.w_sp[i] > 0 and d.l_sp[i] > 0 and m.status[i] not in ("walkover", "unfinished"):
                for pid in (m.winner[i], m.loser[i]):
                    n_stat[pid] = n_stat.get(pid, 0) + 1
        done[0] = max(done[0], idx[k])
        rows = rows_at(st, cutoffs[k], players, n_stat, min_matches=min_matches,
                       active_days=active_days, top=top)
        on_week(Week(cutoffs[k], rows, base_at(st, ("atp", "wta"))))

    points_rating.run(m, params, snap_before=idx, on_snap=snap)
