"""Heavy layer: PCA style vectors per player, refit on a schedule, point-in-time correct.

For each tour (ATP, WTA: never pooled, their serve stats differ by construction) and each
side (serve, return, rally) a fit at date F:
  1. totals of every player's charted matches with edition start < F (style_features.History)
  2. tour mean and prior strength k per feature from those totals (empirical Bayes)
  3. shrunk rates (n * raw + k * mean) / (n + k)
  4. StandardScaler -> PCA on players with >= min_fit_matches charted matches; keep the
     fewest components reaching `variance` explained variance, capped at max_components;
     flip each component so its largest-magnitude loading is positive.
A vector "as of D" uses the latest fit with fit_date <= D (so the fit saw only data < D),
that fit's priors and pipeline, and the player's totals over editions starting before D.
Players with no charted match before D get no vector (None), never an imputed one.

Fits and vectors are stored in models.db (schema v2, keyed by stable player keys); fitted
pipelines are also written with joblib under models/style/ (gitignored).
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import models_db
from .style_features import (FEATURE_VERSION, FEATURES, SIDES, History, MatchCounts,
                             estimate_priors, load_match_counts, shrink)

ROOT = Path(__file__).resolve().parents[2]
STYLE_DIR = ROOT / "models" / "style"
TOURS = ("atp", "wta")
MAX_STORED_COMPONENTS = 10        # style_vectors has c1..c10
Side = Literal["serve", "return", "rally"]


@dataclass(frozen=True)
class HeavyConfig:
    model_id: str = "style_pca_v1"
    variance: float = 0.80          # keep the fewest components reaching this explained variance
    max_components: int = 10
    refit_months: int = 3           # 3 = quarterly (Jan/Apr/Jul/Oct 1), 1 = monthly
    min_fit_matches: int = 5        # charted matches for a player to enter the PCA fit
    min_fit_players: int = 50       # no fit (so no vectors) until this many qualify
    prior_min_den: float = 30.0     # players used to estimate the prior variance
    k_min: float = 1.0
    k_max: float = 1e4
    random_state: int = 0

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def config_hash(self) -> str:
        blob = json.dumps({**self.as_dict(), "feature_version": FEATURE_VERSION}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:10]

    def __post_init__(self):
        if not 0 < self.variance <= 1:
            raise ValueError("variance must be in (0, 1]")
        if not 1 <= self.max_components <= MAX_STORED_COMPONENTS:
            raise ValueError(f"max_components must be 1..{MAX_STORED_COMPONENTS}")
        if self.refit_months not in (1, 2, 3, 4, 6, 12):
            raise ValueError("refit_months must divide 12")


@dataclass
class Fit:
    tour: str
    side: str
    fit_date: str
    features: list[str]
    tour_mean: np.ndarray
    prior_k: np.ndarray
    pipeline: Pipeline
    explained: np.ndarray           # explained-variance ratio of kept components
    n_players: int
    data_from: str | None
    data_to: str | None
    fit_id: str = ""

    @property
    def n_components(self) -> int:
        return len(self.explained)

    def transform(self, num: np.ndarray, den: np.ndarray) -> np.ndarray:
        x = shrink(np.atleast_2d(num), np.atleast_2d(den), self.tour_mean, self.prior_k)
        return self.pipeline.transform(x)

    def loadings(self) -> np.ndarray:
        """(features, components) in standardized-feature units."""
        return self.pipeline.named_steps["pca"].components_.T


@dataclass
class StyleVector:
    vector: np.ndarray
    n_matches: int
    n_points: int
    fit_id: str
    as_of: str


def fit_dates(first: str, last: str, months: int) -> list[str]:
    """Refit dates (1st of every `months`-th month, aligned to January) in [first, last]."""
    y, m = int(first[:4]), 1
    out = []
    while True:
        d = f"{y:04d}-{m:02d}-01"
        if d > last:
            return out
        if d >= first:
            out.append(d)
        m += months
        if m > 12:
            y, m = y + 1, m - 12


def _sign_fix(pca: PCA) -> None:
    """Flip each component so its largest-magnitude loading is positive (stable across refits)."""
    comps = pca.components_
    idx = np.argmax(np.abs(comps), axis=1)
    signs = np.sign(comps[np.arange(len(comps)), idx])
    signs[signs == 0] = 1
    pca.components_ = comps * signs[:, None]


def fit_side(hist: History, fit_date: str, cfg: HeavyConfig) -> Fit | None:
    """Fit one (tour, side) pipeline on data strictly before fit_date, or None if too thin."""
    tot = hist.totals(fit_date)
    if len(tot.player) == 0:
        return None
    assert tot.max_date < fit_date, "leakage: fit input dated on/after fit_date"
    mean, k = estimate_priors(tot.num, tot.den, cfg.prior_min_den, cfg.k_min, cfg.k_max)
    pop = tot.n_matches >= cfg.min_fit_matches
    if pop.sum() < cfg.min_fit_players:
        return None
    x = shrink(tot.num[pop], tot.den[pop], mean, k)
    F = x.shape[1]
    probe = Pipeline([("scale", StandardScaler()),
                      ("pca", PCA(svd_solver="full", random_state=cfg.random_state))]).fit(x)
    cum = np.cumsum(probe.named_steps["pca"].explained_variance_ratio_)
    n = int(min(np.searchsorted(cum, cfg.variance - 1e-12) + 1, cfg.max_components, F))
    pipe = Pipeline([("scale", StandardScaler()),
                     ("pca", PCA(n_components=n, svd_solver="full", random_state=cfg.random_state))]).fit(x)
    _sign_fix(pipe.named_steps["pca"])
    used = [hist.player_totals(int(p), fit_date)[4] for p in tot.player]
    first_dates = [hist.by_player[int(p)][0][0] for p in tot.player]
    return Fit(tour=hist.tour, side=hist.side, fit_date=fit_date, features=list(FEATURES[hist.side]),
               tour_mean=mean, prior_k=k, pipeline=pipe,
               explained=pipe.named_steps["pca"].explained_variance_ratio_.copy(),
               n_players=int(pop.sum()), data_from=min(first_dates), data_to=max(used))


# ------------------------------------------------------------------ persistence

def fit_id_of(cfg: HeavyConfig, tour: str, side: str, fit_date: str) -> str:
    return f"{cfg.model_id}:{cfg.config_hash}:{tour}:{side}:{fit_date}"


def save_fit(con: sqlite3.Connection, fit: Fit, cfg: HeavyConfig, *, style_dir: Path = STYLE_DIR,
             data_built_at: str | None = None) -> str:
    fid = fit_id_of(cfg, fit.tour, fit.side, fit.fit_date)
    fit.fit_id = fid
    path = Path(style_dir) / cfg.config_hash / f"{fit.tour}_{fit.side}_{fit.fit_date}.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": fit.pipeline, "tour_mean": fit.tour_mean, "prior_k": fit.prior_k,
                 "features": fit.features}, path)
    try:
        rel = path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        rel = str(path)
    con.execute("DELETE FROM style_fit WHERE fit_id=?", (fid,))
    con.execute(
        """INSERT INTO style_fit (fit_id, model_id, config_hash, tour, side, fit_date, data_from,
               data_to, n_players, n_components, explained_json, config_json, feature_version,
               pipeline_path, data_built_at, code_commit, created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (fid, cfg.model_id, cfg.config_hash, fit.tour, fit.side, fit.fit_date, fit.data_from,
         fit.data_to, fit.n_players, fit.n_components,
         json.dumps([round(float(v), 6) for v in fit.explained]),
         json.dumps(cfg.as_dict(), sort_keys=True), FEATURE_VERSION, rel, data_built_at,
         models_db._git_commit(), models_db._now()))
    sc = fit.pipeline.named_steps["scale"]
    con.executemany(
        """INSERT INTO style_fit_feature (fit_id, feature, position, tour_mean, prior_k,
               scaler_mean, scaler_scale) VALUES (?,?,?,?,?,?,?)""",
        [(fid, f, i, float(np.nan_to_num(fit.tour_mean[i])), float(fit.prior_k[i]),
          float(sc.mean_[i]), float(sc.scale_[i])) for i, f in enumerate(fit.features)])
    L = fit.loadings()
    con.executemany(
        "INSERT INTO style_loading (fit_id, feature, component, loading) VALUES (?,?,?,?)",
        [(fid, f, c + 1, float(L[i, c])) for i, f in enumerate(fit.features)
         for c in range(L.shape[1])])
    return fid


def load_fits(con: sqlite3.Connection, cfg: HeavyConfig) -> dict[tuple[str, str], list[Fit]]:
    """All stored fits of this configuration, per (tour, side), sorted by fit_date."""
    out: dict[tuple[str, str], list[Fit]] = {}
    rows = con.execute(
        """SELECT fit_id, tour, side, fit_date, pipeline_path, explained_json, n_players,
                  data_from, data_to FROM style_fit WHERE config_hash=? ORDER BY fit_date""",
        (cfg.config_hash,)).fetchall()
    for fid, tour, side, fd, ppath, expl, n, dfrom, dto in rows:
        path = Path(ppath)
        blob = joblib.load(path if path.is_absolute() else ROOT / path)
        out.setdefault((tour, side), []).append(Fit(
            tour=tour, side=side, fit_date=fd, features=blob["features"],
            tour_mean=blob["tour_mean"], prior_k=blob["prior_k"], pipeline=blob["pipeline"],
            explained=np.array(json.loads(expl)), n_players=n, data_from=dfrom, data_to=dto,
            fit_id=fid))
    return out


# ------------------------------------------------------------------ counts cache

def counts_cache_path(built_at: str | None, style_dir: Path = STYLE_DIR) -> Path:
    tag = (built_at or "unknown").replace(":", "").replace("-", "")
    return Path(style_dir) / f"counts_{FEATURE_VERSION}_{tag}.npz"


def load_counts(con: sqlite3.Connection, alias: str = "data", *, built_at: str | None = None,
                style_dir: Path = STYLE_DIR, use_cache: bool = True) -> MatchCounts:
    """Per-match counts from tennis.db, cached per (feature version, tennis.db build)."""
    path = counts_cache_path(built_at, style_dir)
    if use_cache and built_at and path.exists():
        z = np.load(path, allow_pickle=False)
        mc = MatchCounts(match_id=z["match_id"], player=z["player"], date=z["date"], tour=z["tour"],
                         surface=z["surface"])
        for s in SIDES:
            mc.num[s], mc.den[s], mc.points[s] = z[f"num_{s}"], z[f"den_{s}"], z[f"points_{s}"]
        mc.skipped.update(json.loads(str(z["skipped"])))
        return mc
    mc = load_match_counts(con, alias)
    if use_cache and built_at:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, match_id=mc.match_id, player=mc.player, date=mc.date, tour=mc.tour,
            surface=mc.surface, skipped=np.array(json.dumps(dict(mc.skipped))),
            **{f"num_{s}": mc.num[s] for s in SIDES}, **{f"den_{s}": mc.den[s] for s in SIDES},
            **{f"points_{s}": mc.points[s] for s in SIDES})
    return mc


# ------------------------------------------------------------------ the model

class StyleModel:
    """Fits + per-player history: computes any player's vector at any date."""

    def __init__(self, mc: MatchCounts, cfg: HeavyConfig = HeavyConfig(),
                 fits: dict[tuple[str, str], list[Fit]] | None = None):
        self.cfg = cfg
        self.mc = mc
        self.hist = {(t, s): History(mc, t, s) for t in TOURS for s in SIDES}
        self.tour_of = {int(p): str(t) for p, t in zip(mc.player, mc.tour)}
        self.fits: dict[tuple[str, str], list[Fit]] = fits or {}

    def fit_all(self, first: str = "1990-01-01", last: str | None = None,
                progress=None) -> list[Fit]:
        """Fit every scheduled date in [first, last] where enough players qualify."""
        last = last or max(str(d) for d in self.mc.date)
        new = []
        for d in fit_dates(first, last, self.cfg.refit_months):
            for t in TOURS:
                for s in SIDES:
                    f = fit_side(self.hist[(t, s)], d, self.cfg)
                    if f is not None:
                        f.fit_id = fit_id_of(self.cfg, t, s, d)
                        self.fits.setdefault((t, s), []).append(f)
                        new.append(f)
            if progress:
                progress(d)
        for v in self.fits.values():
            v.sort(key=lambda f: f.fit_date)
        return new

    def fit_for(self, tour: str, side: str, as_of: str) -> Fit | None:
        fits = self.fits.get((tour, side), [])
        best = None
        for f in fits:                  # few hundred at most
            if f.fit_date <= as_of:
                best = f
            else:
                break
        return best

    def style(self, player: int, as_of: str, side: str) -> StyleVector | None:
        tour = self.tour_of.get(player)
        if tour is None:
            return None
        got = self.hist[(tour, side)].player_totals(player, as_of)
        if got is None:
            return None
        fit = self.fit_for(tour, side, as_of)
        if fit is None:
            return None
        num, den, n_matches, n_points, last = got
        assert last < as_of
        vec = fit.transform(num, den)[0]
        return StyleVector(vector=vec, n_matches=int(n_matches), n_points=int(n_points),
                           fit_id=fit.fit_id, as_of=as_of)


def materialize(con: sqlite3.Connection, model: StyleModel, pairs, pkeys: dict[int, str]) -> int:
    """Write vectors for (player_id, as_of) pairs into style_vectors. Returns rows written.

    Same numbers as StyleModel.style, batched: one transform per fit instead of per row."""
    groups: dict[str, tuple[Fit, list]] = {}
    for player, as_of in pairs:
        key = pkeys.get(player)
        tour = model.tour_of.get(player)
        if key is None or tour is None:
            continue
        for side in SIDES:
            got = model.hist[(tour, side)].player_totals(player, as_of)
            fit = model.fit_for(tour, side, as_of) if got is not None else None
            if fit is None:
                continue
            groups.setdefault(fit.fit_id, (fit, []))[1].append((key, as_of, side, got))
    sql = f"""INSERT OR REPLACE INTO style_vectors (player_id, as_of_date, side, fit_id, n_matches,
                  n_points, {', '.join(f'c{i}' for i in range(1, 11))})
              VALUES (?,?,?,?,?,?,{','.join('?' * 10)})"""
    n = 0
    for fid, (fit, rows) in groups.items():
        V = fit.transform(np.array([r[3][0] for r in rows]), np.array([r[3][1] for r in rows]))
        pad = [None] * (MAX_STORED_COMPONENTS - V.shape[1])
        con.executemany(sql, [(key, as_of, side, fid, int(got[2]), int(got[3]),
                               *[float(x) for x in v], *pad)
                              for (key, as_of, side, got), v in zip(rows, V)])
        n += len(rows)
    con.commit()
    return n


def backbone_pairs(con: sqlite3.Connection, players: set[int], alias: str = "data",
                   date_from: str = "1900-01-01"):
    """(player_id, edition start) for every backbone match a charted player played."""
    rows = con.execute(
        f"""SELECT DISTINCT p, e.start_date FROM (
                SELECT winner_id AS p, edition_id FROM {alias}.matches
                UNION ALL SELECT loser_id, edition_id FROM {alias}.matches) x
            JOIN {alias}.tournament_edition e USING (edition_id)
            WHERE e.start_date >= ? ORDER BY e.start_date, p""", (date_from,))
    return [(p, d) for p, d in rows if p in players]


# ------------------------------------------------------------------ public API

_default: dict = {}


def _vector_from_row(row) -> np.ndarray:
    return np.array([c for c in row if c is not None], dtype=float)


def get_style(player_id: str, as_of: date | str, side: Side, *, cfg: HeavyConfig = HeavyConfig(),
              models_path: Path | None = None, data_path: Path | None = None,
              style_dir: Path = STYLE_DIR) -> StyleVector | None:
    """Style vector of a player (stable key, e.g. 'sackmann_atp:104925') as of a date.

    Uses only charted matches from editions that started strictly before `as_of`. Reads the
    materialized row if there is one, otherwise computes it from the stored fits.
    Returns None when the player has no charted match before `as_of` (or no fit exists yet).
    """
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}")
    as_of = as_of.isoformat() if isinstance(as_of, date) else str(as_of)[:10]
    st = _state(cfg, models_path, data_path, style_dir)
    row = st["con"].execute(
        f"""SELECT v.fit_id, v.n_matches, v.n_points, {', '.join(f'c{i}' for i in range(1, 11))}
            FROM style_vectors v JOIN style_fit f USING (fit_id)
            WHERE v.player_id=? AND v.as_of_date=? AND v.side=? AND f.config_hash=?""",
        (player_id, as_of, side, cfg.config_hash)).fetchone()
    if row is not None:
        return StyleVector(_vector_from_row(row[3:]), row[1], row[2], row[0], as_of)
    pid = st["pids"].get(player_id)
    if pid is None:
        return None
    return st["model"].style(pid, as_of, side)


def get_style_vector(player_id: str, as_of: date | str, side: Side, **kw) -> np.ndarray | None:
    """The spec's API: just the vector (see get_style for n_matches / n_points)."""
    sv = get_style(player_id, as_of, side, **kw)
    return None if sv is None else sv.vector


def _state(cfg: HeavyConfig, models_path, data_path, style_dir) -> dict:
    key = (cfg.config_hash, str(models_path), str(data_path), str(style_dir))
    st = _default.get(key)
    if st is None:
        con = models_db.connect(models_path or models_db.MODELS_DB)
        models_db.attach_data(con, data_path or models_db.DATA_DB)
        built = con.execute("SELECT value FROM data.schema_meta WHERE key='built_at'").fetchone()
        mc = load_counts(con, built_at=built[0] if built else None, style_dir=style_dir)
        model = StyleModel(mc, cfg, load_fits(con, cfg))
        pids = {v: k for k, v in models_db.player_keys(con).items()}
        st = _default[key] = {"con": con, "model": model, "pids": pids}
    return st
