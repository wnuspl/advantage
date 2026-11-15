"""Composite model: light + heavy + matchup layers -> P(A beats B) for any pair and date.

    p_glicko_raw = Glicko-2 P(A wins)        light layer: frozen run (default 187), both RDs
                                             inflated to the match date, surface blend
    glicko_logit = logit(p_glicko_raw)
    offset       = alpha * glicko_logit      alpha frozen in the matchup bundle
    delta        = matchup model(s_A, s_B, surface), s_* projected through the bundle's
                   heavy-layer fit_ids (one fit per side), 0 if either player has no vector
    p            = sigmoid(offset + delta)

This is exactly the chain the walk-forward harness (matchup/harness.py) scores, rebuilt
from what is stored: the Glicko-2 run's parameters (models.db model_run), the heavy-layer
fits (style_fit + joblib), and the matchup bundles (matchup_bundle + model files). Nothing
here trains or modifies a layer.

Bundle choice: for a date D, the bundle of the chosen matchup run whose test block contains
D (test_from <= D < test_to) -- the model the harness used for that match. For dates after
the last block, the latest bundle (its training cutoff is printed; a production retrain on
all data is a separate step, MIDDLE_LAYER_SPEC.md §10). For dates before the first block
there is no bundle and the prediction is an error.

Point-in-time: the Glicko-2 state is replayed over editions starting strictly before D
(elo.until), style vectors use charted matches from editions starting before D, and the
bundle saw only matches before its train cutoff. Within an event already under way on D,
earlier rounds are NOT included (the harness's per-match sequence does include them), so
`check_replay` compares only each player's first match of an edition.
"""
from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import elo, glicko2, models_db, tuning
from .matchup import data as mdata
from .matchup import models as mmodels
from .matchup.data import SURFACE_IDX, FoldData, StyleSource, sigmoid
from .style_features import SIDES

ROOT = Path(__file__).resolve().parents[2]
SURFACES = ("hard", "clay", "grass", "carpet")


# ------------------------------------------------------------------ light layer

def glicko_prob(state: glicko2.Glicko2State | None, a: int, b: int, on: str, surface: str | None) -> dict:
    """Glicko-2 P(a beats b) on date `on` from the ratings after every match before it.

    Same arithmetic as glicko2.run's pre-match step: RDs inflated by inactivity, a player
    with no rating (or no rating on that surface) starts at init / rd_init (surface_init=1:
    at their overall rating), then the run's surface blend."""
    p = state.params if state is not None else None
    if p is None:
        raise ValueError("no Glicko-2 state")

    def overall(pid):
        if pid in state.rating:
            return state.rating[pid], state.rd_at(pid, on), True
        return p.init, p.rd_init, False

    def surf(pid, r_overall):
        if surface is None:
            return math.nan, math.nan
        if (pid, surface) in state.srating:
            return state.srating[(pid, surface)], state.srd_at(pid, surface, on)
        return (r_overall if p.surface_init >= 0.5 else p.init), p.rd_init

    ra, da, ka = overall(a)
    rb, db, kb = overall(b)
    sa, sda = surf(a, ra)
    sb, sdb = surf(b, rb)
    prob = glicko2._blend_prob(ra, da, rb, db, sa, sda, sb, sdb, p.blend)
    return {"p": prob, "a": {"rating": ra, "rd": da, "surface_rating": sa, "surface_rd": sda,
                             "rated": ka, "matches": state.count.get(a, 0)},
            "b": {"rating": rb, "rd": db, "surface_rating": sb, "surface_rd": sdb,
                  "rated": kb, "matches": state.count.get(b, 0)}}


def glicko_state(m: elo.Matches, params, as_of: str | None) -> glicko2.Glicko2State:
    """Ratings after every match in editions starting strictly before `as_of` (all if None)."""
    sub = elo.until(m, as_of) if as_of else m
    return glicko2.run(sub, params).state


# ------------------------------------------------------------------ matchup bundles

@dataclass
class BundleRow:
    run_id: int
    model: str                  # matchup model name, e.g. 'bilinear'
    tour: str
    test_from: str
    test_to: str
    train_cutoff: str
    fit_ids: dict
    alpha: float
    path: str | None


def load_bundles(con: sqlite3.Connection, run_id: int) -> tuple[dict, dict]:
    """({tour: [BundleRow sorted by test_from]}, run params) of a matchup run."""
    row = con.execute("SELECT model_id, params_json FROM model_run WHERE run_id=?", (run_id,)).fetchone()
    if row is None or not row[0].startswith("matchup_"):
        raise ValueError(f"run {run_id} is not a matchup run")
    params = json.loads(row[1])
    out: dict[str, list[BundleRow]] = {}
    for tour, tf, tt, cut, fj, alpha, path in con.execute(
            """SELECT tour, test_from, test_to, train_cutoff, fit_ids_json, alpha, path
               FROM matchup_bundle WHERE run_id=? ORDER BY tour, test_from""", (run_id,)):
        out.setdefault(tour, []).append(BundleRow(run_id, params["model"], tour, tf, tt, cut,
                                                  json.loads(fj), alpha, path))
    if not out:
        raise ValueError(f"run {run_id} has no matchup bundles")
    return out, params


def pick_bundle(bundles: list[BundleRow], on: str) -> BundleRow:
    """The bundle whose test block contains `on`; after the last block, the latest one."""
    if not bundles or on < bundles[0].test_from:
        first = bundles[0].test_from if bundles else "?"
        raise ValueError(f"no matchup bundle for {on}: the run's first test block starts {first}")
    best = bundles[0]
    for b in bundles:
        if b.test_from <= on:
            best = b
        if b.test_from <= on < b.test_to:
            return b
    return best


def rebuild_model(name: str, blob: dict | None) -> mmodels.MatchupModel:
    """A trained matchup model from its bundle file contents (harness.save's format)."""
    if name == "glicko":
        return mmodels.GlickoOnly()
    if blob is None:
        raise ValueError(f"bundle of model {name!r} has no model file")
    cfg, state = blob["config"], blob["state"]
    if name.startswith("bilinear"):
        m = mmodels.BilinearModel(rank=cfg["rank"], terms=tuple(cfg["terms"]),
                                  train_cfg=mmodels.TrainConfig(**cfg["train"]))
    elif name == "siamese":
        m = mmodels.SiameseModel(hidden=cfg["hidden"], head_hidden=cfg["head_hidden"], dropout=cfg["dropout"],
                                 train_cfg=mmodels.TrainConfig(**cfg["train"]))
    elif name == "lgbm":
        import lightgbm as lgb
        m = mmodels.LGBMModel(mmodels.LGBMConfig(**cfg))
        m.booster = lgb.Booster(model_str=state)
        return m
    else:
        raise ValueError(f"don't know how to rebuild matchup model {name!r}")
    m.dims = state["dims"]
    m.module = m.build(m.dims)
    m.module.load_state_dict(state["state_dict"])
    m.module.eval()
    return m


class BundleCache:
    def __init__(self, root: Path = ROOT):
        self.root = Path(root)
        self._models: dict = {}

    def model(self, b: BundleRow) -> mmodels.MatchupModel:
        key = (b.run_id, b.tour, b.test_from)
        if key not in self._models:
            blob = None
            if b.path:
                import torch
                p = Path(b.path)
                blob = torch.load(p if p.is_absolute() else self.root / p, weights_only=False)
                if blob.get("fit_ids") != b.fit_ids:
                    raise ValueError(f"bundle file {b.path} records other fit_ids than matchup_bundle")
            self._models[key] = rebuild_model(b.model, blob)
        return self._models[key]


# ------------------------------------------------------------------ the composite

@dataclass
class Prediction:
    a: int
    b: int
    date: str
    tour: str
    surface: str | None
    p: float                     # composite P(a wins)
    p_glicko_raw: float          # light layer alone (run's probabilities, alpha = 1)
    p_glicko: float              # alpha-calibrated light layer (the matchup benchmark)
    glicko_logit: float
    alpha: float
    delta: float                 # matchup layer's logit adjustment (0 without both vectors)
    has_vectors: bool
    n_charted_a: int
    n_charted_b: int
    model: str
    bundle: dict                 # run_id, test_from, test_to, train_cutoff, fit_ids
    light: dict = field(default_factory=dict)   # ratings / RDs of both players

    def as_dict(self) -> dict:
        return asdict(self)


class CompositePredictor:
    """Light-layer state + heavy-layer style source + matchup bundles of one run."""

    def __init__(self, state: glicko2.Glicko2State, src: StyleSource, bundles: dict,
                 cache: BundleCache | None = None, state_as_of: str | None = None):
        self.state = state
        self.src = src
        self.bundles = bundles
        self.cache = cache or BundleCache()
        self.state_as_of = state_as_of   # the Glicko state covers editions before this date

    def predict_many(self, a, b, dates, surfaces, tour: str) -> list[Prediction]:
        a = [int(x) for x in a]
        b = [int(x) for x in b]
        dates = [str(d) for d in dates]
        n = len(a)
        if self.state_as_of is not None and any(d < self.state_as_of for d in dates):
            raise ValueError(f"Glicko state includes editions up to {self.state_as_of}; "
                             "it cannot predict earlier dates")
        light = [glicko_prob(self.state, a[i], b[i], dates[i], surfaces[i]) for i in range(n)]
        gl = mdata._logit(np.array([x["p"] for x in light]))
        brows = [pick_bundle(self.bundles[tour], d) for d in dates]
        out: list[Prediction | None] = [None] * n
        groups: dict = {}
        for i, br in enumerate(brows):
            groups.setdefault((br.test_from,), []).append(i)
        for _, idx in groups.items():
            br = brows[idx[0]]
            rows = np.array(idx)
            va = self.src.vectors(tour, np.array(a)[rows], np.array(dates)[rows], br.fit_ids)
            vb = self.src.vectors(tour, np.array(b)[rows], np.array(dates)[rows], br.fit_ids)
            sidx = np.array([SURFACE_IDX.get(surfaces[i], -1) if surfaces[i] else -1 for i in idx], dtype=np.int8)
            fd = FoldData(rows=rows, y=np.zeros(len(rows)), glicko_logit=gl[rows], offset=br.alpha * gl[rows],
                          surf=mdata.surface_onehot(sidx), xa=va.x, xb=vb.x, has=va.has & vb.has,
                          na=va.n_matches, nb=vb.n_matches, fit_ids=dict(br.fit_ids))
            p = self.cache.model(br).predict_proba(fd)
            z = np.log(p) - np.log1p(-p)
            for j, i in enumerate(idx):
                out[i] = Prediction(
                    a=a[i], b=b[i], date=dates[i], tour=tour, surface=surfaces[i], p=float(p[j]),
                    p_glicko_raw=float(light[i]["p"]), p_glicko=float(sigmoid(fd.offset[j])),
                    glicko_logit=float(gl[i]), alpha=br.alpha, delta=float(z[j] - fd.offset[j]),
                    has_vectors=bool(fd.has[j]), n_charted_a=int(va.n_matches[j]), n_charted_b=int(vb.n_matches[j]),
                    model=br.model, bundle={"run_id": br.run_id, "test_from": br.test_from, "test_to": br.test_to,
                                            "train_cutoff": br.train_cutoff, "fit_ids": br.fit_ids},
                    light={"a": light[i]["a"], "b": light[i]["b"]})
        return out

    def predict(self, a: int, b: int, on: str, surface: str | None, tour: str) -> Prediction:
        return self.predict_many([a], [b], [on], [surface], tour)[0]


# ------------------------------------------------------------------ loading from the databases

def latest_matchup_run(con: sqlite3.Connection, model: str) -> int:
    row = con.execute("""SELECT r.run_id FROM model_run r WHERE r.model_id=? AND r.finished_at IS NOT NULL
                         AND EXISTS (SELECT 1 FROM matchup_bundle b WHERE b.run_id=r.run_id)
                         ORDER BY r.run_id DESC LIMIT 1""", (f"matchup_{model}_{_version()}",)).fetchone()
    if row is None:
        raise SystemExit(f"no finished matchup_{model} run with bundles in models.db; "
                         "run scripts/matchup.py run first")
    return row[0]


def _version() -> str:
    from .matchup.harness import MODEL_VERSION
    return MODEL_VERSION


def load_predictor(con: sqlite3.Connection, run_id: int, as_of: str | None, *,
                   m: elo.Matches | None = None, src: StyleSource | None = None,
                   log=print) -> CompositePredictor:
    """Everything a prediction on/after `as_of` needs (tennis.db attached as 'data')."""
    from . import heavy
    bundles, params = load_bundles(con, run_id)
    h = params["harness"]
    spec, gparams = tuning.model_of(con, h["glicko_run_id"])
    if spec.model_id != "glicko2_v1":
        raise ValueError(f"light-layer run {h['glicko_run_id']} is not Glicko-2")
    m = m if m is not None else elo.load_matches(con)
    state = glicko_state(m, gparams, as_of)
    if src is None:
        hcfg = heavy.HeavyConfig()
        if h.get("heavy_config") and h["heavy_config"] != hcfg.config_hash:
            raise ValueError(f"matchup run {run_id} used heavy config {h['heavy_config']}, "
                             f"the code default is {hcfg.config_hash}")
        src = mdata.load_style_source(con, hcfg)
    return CompositePredictor(state, src, bundles, state_as_of=as_of)


def resolve_player(con: sqlite3.Connection, text: str, tour: str | None = None) -> tuple[int, str, str]:
    """(player_id, name, tour) from a stable key ('sackmann_atp:104925'), an exact name
    (case-insensitive) or a unique name fragment. Ambiguity is an error listing candidates."""
    if ":" in text:
        row = con.execute("""SELECT p.player_id, p.canonical_name, p.tour FROM data.player_xref x
                             JOIN data.player p USING (player_id)
                             WHERE x.source || ':' || x.source_key = ?""", (text,)).fetchone()
        if row:
            return row
    q = """SELECT p.player_id, p.canonical_name, p.tour,
                  (SELECT MAX(e.start_date) FROM data.matches mm JOIN data.tournament_edition e USING (edition_id)
                   WHERE mm.winner_id = p.player_id OR mm.loser_id = p.player_id) AS last
           FROM data.player p WHERE p.is_placeholder = 0 AND {} {}"""
    tour_sql = "AND p.tour = ?" if tour else ""
    targs = [tour] if tour else []
    rows = con.execute(q.format("LOWER(p.canonical_name) = LOWER(?)", tour_sql), [text, *targs]).fetchall()
    if not rows:
        rows = con.execute(q.format("LOWER(p.canonical_name) LIKE LOWER(?)", tour_sql),
                           [f"%{text}%", *targs]).fetchall()
    rows = [r for r in rows if r[3] is not None] or rows      # prefer players with matches
    if len(rows) == 1:
        return rows[0][:3]
    if not rows:
        raise SystemExit(f"no player matches {text!r}")
    rows.sort(key=lambda r: r[3] or "", reverse=True)
    listing = "\n".join(f"  {r[1]} ({r[2]}, last match {r[3]})" for r in rows[:15])
    raise SystemExit(f"{text!r} is ambiguous; candidates:\n{listing}\n(use the full name or --tour)")


# ------------------------------------------------------------------ replay check

def first_appearances(m: elo.Matches, start: str, scored_only: bool = True) -> list[int]:
    """Indices of matches at edition start `start` where neither player played an earlier
    match (any status) starting that day: their pre-match Glicko state equals the state
    after every edition starting before `start`."""
    from bisect import bisect_left, bisect_right
    lo, hi = bisect_left(m.start, start), bisect_right(m.start, start)
    seen: set = set()
    out = []
    for i in range(lo, hi):
        w, l = m.winner[i], m.loser[i]
        if w not in seen and l not in seen and (not scored_only or m.status[i] in elo.SCORED_STATUS):
            out.append(i)
        seen.add(w)
        seen.add(l)
    return out


@dataclass
class ReplayCheck:
    n: int
    max_abs_diff: float          # composite vs the harness's stored prediction
    max_abs_diff_light: float    # light layer vs the Glicko-2 run's stored prediction
    dates: list


def check_replay(con: sqlite3.Connection, run_id: int, dates: list[str], *, m: elo.Matches | None = None,
                 src: StyleSource | None = None, log=print) -> ReplayCheck:
    """Rebuild predictions for each player's first match of the editions starting on
    `dates` through the composite chain and compare with what the walk-forward run stored
    (models.db prediction rows of the matchup run and of its Glicko-2 run)."""
    bundles, params = load_bundles(con, run_id)
    glicko_run = params["harness"]["glicko_run_id"]
    m = m if m is not None else elo.load_matches(con)
    mkeys = models_db.match_keys(con)
    pkeys = models_db.player_keys(con)
    if src is None:
        src = mdata.load_style_source(con)
    worst = worst_light = 0.0
    n = 0
    for d in dates:
        pred = load_predictor(con, run_id, d, m=m, src=src, log=log)
        idx = [i for i in first_appearances(m, d) if m.tour[i] in bundles]
        for tour in sorted({m.tour[i] for i in idx}):
            ti = [i for i in idx if m.tour[i] == tour]
            stored, light = {}, {}
            keys = [mkeys[m.match_id[i]] for i in ti]
            for k, pp in _stored(con, run_id, keys):
                stored[k] = pp
            for k, pp in _stored(con, glicko_run, keys):
                light[k] = pp
            ti = [i for i in ti if mkeys[m.match_id[i]] in stored]
            if not ti:
                continue
            p1 = [m.winner[i] if pkeys[m.winner[i]] < pkeys[m.loser[i]] else m.loser[i] for i in ti]
            p2 = [m.loser[i] if pkeys[m.winner[i]] < pkeys[m.loser[i]] else m.winner[i] for i in ti]
            got = pred.predict_many(p1, p2, [d] * len(ti), [m.surface[i] for i in ti], tour)
            for i, g in zip(ti, got):
                k = mkeys[m.match_id[i]]
                worst = max(worst, abs(g.p - stored[k]))
                if k in light:
                    worst_light = max(worst_light, abs(g.p_glicko_raw - light[k]))
                n += 1
        log(f"  {d}: checked {n} matches so far, max |diff| composite {worst:.2e}, light {worst_light:.2e}")
    return ReplayCheck(n, worst, worst_light, list(dates))


def _stored(con, run_id: int, keys: list[str]):
    out = []
    for i in range(0, len(keys), 500):
        chunk = keys[i:i + 500]
        out += con.execute(f"SELECT match_key, p_p1_wins FROM prediction WHERE run_id=? AND match_key IN "
                           f"({','.join('?' * len(chunk))})", [run_id, *chunk]).fetchall()
    return out
