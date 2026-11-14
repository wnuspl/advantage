"""Inputs of the matchup layer: one row per scored match, oriented p1 vs p2, plus style vectors.

Orientation: p1 is the player with the smaller stable key (same rule as evaluate.py), never
the winner. y = 1 if p1 won.

glicko_logit: logit of the light layer's pre-match P(p1 wins) from a frozen Glicko-2 run
(default run 187), i.e. the Glicko-2 expected-score formula with both RDs and the surface
blend. Recomputed from the run's parameters (deterministic), never modified.

Style vectors come from the heavy layer's fitted objects (heavy.StyleModel, heavy.Fit), not
from its tables. Within one fold every vector is projected through ONE fit per side
(fit-consistency, spec §1): the player's charted totals before the match date are
transformed by that fit's shrinkage priors and pipeline. heavy.get_style picks the latest
fit at each date instead, which would mix PCA axes inside a fold, so StyleSource.vectors
below is the batch API the spec calls `get_style_vectors(requests, fit_ids)`.
TODO: move it into heavy.py once the heavy layer is committed.
"""
from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass

import numpy as np

from .. import elo, glicko2, heavy, models_db, tuning
from ..style_features import SIDES

SURFACES = ("hard", "clay", "grass")          # carpet folds into hard; unknown -> all zeros
SURFACE_IDX = {"hard": 0, "carpet": 0, "clay": 1, "grass": 2}
LOGIT_CLIP = 1e-6


@dataclass
class MatchTable:
    """Scored matches (completed or retired) in walk-forward order. Arrays are row-aligned."""
    match_id: np.ndarray        # tennis.db match_id
    date: np.ndarray            # edition start, ISO text (the clock every layer uses)
    tour: np.ndarray            # 'atp' / 'wta'
    surface: np.ndarray         # index into SURFACES, -1 unknown
    tour_level: np.ndarray      # bool: tour-level main draw (the project's primary scope)
    best_of: np.ndarray         # 3 / 5 (0 unknown); for the point-level phase
    p1: np.ndarray              # tennis.db player_id, smaller stable key
    p2: np.ndarray
    y: np.ndarray               # 1.0 if p1 won
    glicko_logit: np.ndarray    # pre-match logit P(p1 wins) from the light layer
    player_key: dict            # player_id -> stable key
    glicko_run_id: int | None = None

    def __len__(self) -> int:
        return len(self.match_id)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, LOGIT_CLIP, 1 - LOGIT_CLIP)
    return np.log(p) - np.log1p(-p)


def build_table(m: elo.Matches, p_winner, winner_is_p1, player_key: dict,
                best_of: dict | None = None, glicko_run_id: int | None = None) -> MatchTable:
    """Orient and filter the light layer's output. Pure: no database access."""
    keep = np.array([s in elo.SCORED_STATUS for s in m.status], dtype=bool)
    idx = np.flatnonzero(keep)
    flag = np.frombuffer(bytes(winner_is_p1), dtype=np.uint8)[idx].astype(bool)
    w = np.asarray(m.winner, dtype=np.int64)[idx]
    l = np.asarray(m.loser, dtype=np.int64)[idx]
    pw = np.asarray(p_winner, dtype=float)[idx]
    mids = np.asarray(m.match_id, dtype=np.int64)[idx]
    bo = best_of or {}
    return MatchTable(
        match_id=mids,
        date=np.array(m.start, dtype=object)[idx].astype(str),
        tour=np.array(m.tour, dtype=object)[idx].astype(str),
        surface=np.array([SURFACE_IDX.get(m.surface[i], -1) for i in idx], dtype=np.int8),
        tour_level=np.frombuffer(bytes(m.tour_level), dtype=np.uint8)[idx].astype(bool),
        best_of=np.array([bo.get(int(x)) or 0 for x in mids], dtype=np.int8),
        p1=np.where(flag, w, l), p2=np.where(flag, l, w),
        y=flag.astype(float),
        glicko_logit=_logit(np.where(flag, pw, 1.0 - pw)),
        player_key=player_key, glicko_run_id=glicko_run_id)


def load_table(con: sqlite3.Connection, glicko_run_id: int = 187) -> MatchTable:
    """Run the frozen Glicko-2 configuration of a recorded run and orient every match.
    tennis.db must be attached as 'data'."""
    spec, params = tuning.model_of(con, glicko_run_id)
    if spec.model_id != "glicko2_v1":
        raise ValueError(f"run {glicko_run_id} is {spec.model_id}, expected a Glicko-2 run")
    prep = tuning.prepare(con)
    res = glicko2.run(prep.m, params)
    best_of = dict(con.execute("SELECT match_id, best_of FROM data.matches"))
    return build_table(prep.m, res.p_winner, prep.winner_is_p1, prep.player_key, best_of,
                       glicko_run_id)


# ------------------------------------------------------------------ style vectors

@dataclass
class Vectors:
    """Style vectors of one fold for a set of (player, date) requests."""
    x: dict                     # side -> (n, K_side) float; zeros where `has` is False
    has: np.ndarray             # bool: player has a vector for every side
    n_matches: np.ndarray       # charted matches behind the vector (0 if none)
    fit_ids: dict               # side -> fit_id used for every row


class StyleSource:
    """Player charted totals (fit-independent, cached) + projection through chosen fits."""

    def __init__(self, model: heavy.StyleModel):
        self.model = model
        self.fit_by_id = {f.fit_id: f for fl in model.fits.values() for f in fl}
        self._cache: dict = {}  # (side, player, date) -> (num, den, n_matches, last) | None

    def fold_fit_ids(self, tour: str, cutoff: str) -> dict | None:
        """{side: fit_id} of the latest fit per side with fit_date <= cutoff (each saw only
        charted data before its fit_date, so before cutoff). None if any side has no fit."""
        out = {}
        for side in SIDES:
            f = self.model.fit_for(tour, side, cutoff)
            if f is None:
                return None
            out[side] = f.fit_id
        return out

    def _totals(self, tour: str, side: str, player: int, as_of: str):
        key = (side, player, as_of)
        if key not in self._cache:
            got = self.model.hist[(tour, side)].player_totals(player, as_of)
            self._cache[key] = None if got is None else (got[0], got[1], got[2], got[4])
        return self._cache[key]

    def vectors(self, tour: str, players, dates, fit_ids: dict) -> Vectors:
        """get_style_vectors(requests, fit_ids): every vector through the given fits.
        A row has a vector only if the player has charted matches before its date."""
        players = np.asarray(players, dtype=np.int64)
        dates = np.asarray(dates).astype(str)
        n = len(players)
        fits = {s: self.fit_by_id[fit_ids[s]] for s in SIDES}
        for s, f in fits.items():
            if f.tour != tour or f.side != s:
                raise ValueError(f"fit {f.fit_id} is not a {tour}/{s} fit")
        has = np.ones(n, dtype=bool)
        nm = np.zeros(n, dtype=np.int64)
        got = {s: [None] * n for s in SIDES}
        for i in range(n):
            p, d = int(players[i]), dates[i]
            for s in SIDES:
                t = self._totals(tour, s, p, d)
                if t is None:
                    has[i] = False
                    break
                assert t[3] < d, "leakage: style input dated on/after the match"
                got[s][i] = t
            if has[i]:
                nm[i] = got["serve"][i][2]
        x = {}
        rows = np.flatnonzero(has)
        for s in SIDES:
            f = fits[s]
            out = np.zeros((n, f.n_components))
            if len(rows):
                num = np.array([got[s][i][0] for i in rows])
                den = np.array([got[s][i][1] for i in rows])
                out[rows] = f.transform(num, den)
            x[s] = out
        return Vectors(x=x, has=has, n_matches=nm, fit_ids=dict(fit_ids))


def load_style_source(con: sqlite3.Connection, cfg: heavy.HeavyConfig = heavy.HeavyConfig()) -> StyleSource:
    """The heavy layer's stored fits + charted counts (tennis.db attached as 'data')."""
    built = con.execute("SELECT value FROM data.schema_meta WHERE key='built_at'").fetchone()
    mc = heavy.load_counts(con, built_at=built[0] if built else None)
    fits = heavy.load_fits(con, cfg)
    if not fits:
        raise RuntimeError(f"no heavy-layer fits for config {cfg.config_hash}; "
                           "run scripts/style_fit.py fit first")
    return StyleSource(heavy.StyleModel(mc, cfg, fits))


# ------------------------------------------------------------------ fold data

@dataclass
class FoldData:
    """Model inputs for a set of match rows, all vectors from one fit per side."""
    rows: np.ndarray            # indices into the MatchTable
    y: np.ndarray
    glicko_logit: np.ndarray
    offset: np.ndarray          # alpha * glicko_logit (frozen)
    surf: np.ndarray            # (n, 3) one-hot, zeros if unknown
    xa: dict                    # side -> (n, K) for p1
    xb: dict                    # side -> (n, K) for p2
    has: np.ndarray             # both players have vectors
    na: np.ndarray              # charted matches behind p1's / p2's vectors
    nb: np.ndarray
    fit_ids: dict

    def __len__(self) -> int:
        return len(self.rows)

    def subset(self, mask: np.ndarray) -> "FoldData":
        return FoldData(self.rows[mask], self.y[mask], self.glicko_logit[mask], self.offset[mask],
                        self.surf[mask], {s: v[mask] for s, v in self.xa.items()},
                        {s: v[mask] for s, v in self.xb.items()}, self.has[mask], self.na[mask],
                        self.nb[mask], self.fit_ids)

    def dims(self) -> dict:
        return {s: v.shape[1] for s, v in self.xa.items()}


def surface_onehot(idx: np.ndarray) -> np.ndarray:
    out = np.zeros((len(idx), len(SURFACES)))
    ok = idx >= 0
    out[np.flatnonzero(ok), idx[ok]] = 1.0
    return out


def fold_data(t: MatchTable, src: StyleSource, rows: np.ndarray, tour: str, fit_ids: dict,
              alpha: float) -> FoldData:
    rows = np.asarray(rows, dtype=np.int64)
    if len(rows) and not np.all(t.tour[rows] == tour):
        raise ValueError("fold rows span several tours")
    va = src.vectors(tour, t.p1[rows], t.date[rows], fit_ids)
    vb = src.vectors(tour, t.p2[rows], t.date[rows], fit_ids)
    gl = t.glicko_logit[rows]
    return FoldData(rows=rows, y=t.y[rows], glicko_logit=gl, offset=alpha * gl,
                    surf=surface_onehot(t.surface[rows]), xa=va.x, xb=vb.x,
                    has=va.has & vb.has, na=va.n_matches, nb=vb.n_matches, fit_ids=dict(fit_ids))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(z, dtype=float)))


def fit_alpha(logit: np.ndarray, y: np.ndarray, iters: int = 50) -> float:
    """Scalar logistic regression y ~ sigmoid(alpha * logit), no intercept (Newton)."""
    a = 1.0
    for _ in range(iters):
        p = sigmoid(a * logit)
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
