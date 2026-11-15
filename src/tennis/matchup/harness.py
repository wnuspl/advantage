"""Walk-forward harness for the matchup layer (spec §7-8). Every model runs on identical folds.

A fold (per tour) is a test block [S, S + step):
  val    = [S - val_months, S)          early stopping only
  train  = everything before the cutoff C = S - val_months
  fits   = per side, the latest heavy-layer fit with fit_date <= C; train, val and test
           vectors ALL come from these fits (one fit per side per fold)
  alpha  = scalar logistic regression of y on glicko_logit over training rows with both
           vectors dated [C - alpha_years, C) (alpha_rows='train'; 'scope' uses every
           train-scope row of the tour). Frozen; sigmoid(alpha * glicko_logit) is the
           Glicko-only benchmark.
Dates are edition starts (the project's clock). Glicko logits are walk-forward pre-match
values, style vectors use only charted matches before the match's edition, and model
weights / alpha / fits see only matches before C <= S: nothing is dated on/after a
predicted match.

Training rows: matches where both players have a vector (any level, or tour-level only
with train_scope='tour'); the match itself needs no charting. Test rows: every scored
match in the block; rows without both vectors get exactly sigmoid(alpha * glicko_logit).
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

import numpy as np

from .. import models_db
from ..evaluate import Score, score
from .data import MatchTable, StyleSource, fit_alpha, fold_data, sigmoid
from .models import make

ROOT = Path(__file__).resolve().parents[3]
BUNDLE_DIR = ROOT / "models" / "matchup"
MODEL_VERSION = "v1"


@dataclass(frozen=True)
class HarnessConfig:
    glicko_run_id: int = 187
    test_from: str = "2022-01-01"
    test_to: str = "2026-12-31"
    step_months: int = 3
    val_months: int = 6
    alpha_years: int = 10
    alpha_rows: str = "train"         # 'train' = the fold's training rows with both vectors
                                      # (spec §4); 'scope' = every train-scope row of the tour;
                                      # 'tour' = the tour's tour-level rows (as calibrate.py)
    train_scope: str = "all"          # 'all' scored matches, or 'tour' (tour-level only)
    min_charted: int = 10             # N for the well-charted evaluation subset
    tours: tuple = ("atp", "wta")
    heavy_config: str = ""            # heavy-layer config hash the fits came from

    def as_dict(self) -> dict:
        d = asdict(self)
        d["tours"] = list(self.tours)
        return d


def add_months(d: str, k: int) -> str:
    y, m = int(d[:4]), int(d[5:7]) - 1 + k
    y, m = y + m // 12, m % 12
    return date(y, m + 1, 1).isoformat()


@dataclass
class Fold:
    tour: str
    test_from: str
    test_to: str                      # exclusive
    val_from: str                     # = train cutoff
    fit_ids: dict
    alpha: float
    train: np.ndarray                 # MatchTable rows (both players charted at some point)
    val: np.ndarray
    test: np.ndarray                  # every scored row of the block
    n_alpha: int = 0

    @property
    def cutoff(self) -> str:
        return self.val_from

    @property
    def label(self) -> str:
        return f"{self.tour}:{self.test_from}"


def make_folds(t: MatchTable, src: StyleSource, cfg: HarnessConfig, log=print) -> list[Fold]:
    charted = np.array(sorted(src.model.tour_of), dtype=np.int64)
    both_charted = np.isin(t.p1, charted) & np.isin(t.p2, charted)
    scope = t.tour_level if cfg.train_scope == "tour" else np.ones(len(t), dtype=bool)
    last = max(t.date)
    folds = []
    s = cfg.test_from
    while s <= min(cfg.test_to, last):
        e = add_months(s, cfg.step_months)
        c = add_months(s, -cfg.val_months)
        for tour in cfg.tours:
            in_tour = t.tour == tour
            fit_ids = src.fold_fit_ids(tour, c)
            if fit_ids is None:
                log(f"  {tour} {s}: no heavy-layer fit on or before {c}; fold skipped")
                continue
            amask = in_tour & (t.tour_level if cfg.alpha_rows == "tour" else scope)
            arow = np.flatnonzero(amask & (t.date < c) & (t.date >= add_months(c, -12 * cfg.alpha_years)))
            alpha = fit_alpha(t.glicko_logit[arow], t.y[arow])
            base = in_tour & scope & both_charted
            folds.append(Fold(
                tour=tour, test_from=s, test_to=e, val_from=c, fit_ids=fit_ids, alpha=alpha,
                train=np.flatnonzero(base & (t.date < c)),
                val=np.flatnonzero(base & (t.date >= c) & (t.date < s)),
                test=np.flatnonzero(in_tour & (t.date >= s) & (t.date < e)),
                n_alpha=len(arow)))
        s = e
    return folds


def check_fold(t: MatchTable, src: StyleSource, f: Fold) -> None:
    """Point-in-time guards (also exercised by the leakage test)."""
    for side, fid in f.fit_ids.items():
        fit = src.fit_by_id[fid]
        assert fit.fit_date <= f.cutoff, f"fit {fid} after cutoff {f.cutoff}"
        assert fit.data_to is None or fit.data_to < fit.fit_date, f"fit {fid} saw data >= fit_date"
    if len(f.train):
        assert max(t.date[f.train]) < f.cutoff, "train row on/after cutoff"
    if len(f.val):
        assert min(t.date[f.val]) >= f.cutoff and max(t.date[f.val]) < f.test_from
    if len(f.test):
        assert min(t.date[f.test]) >= f.test_from and max(t.date[f.test]) < f.test_to


@dataclass
class RunResult:
    cfg: HarnessConfig
    models: list
    folds: list
    pred: dict                        # model name -> P(p1 wins) per MatchTable row (NaN off-test)
    has: np.ndarray                   # test rows: both players had vectors under the fold's fits
    n_min: np.ndarray                 # test rows: min charted matches of the two players
    fold_of: np.ndarray               # test rows: index into folds (-1 elsewhere)
    trained: list = field(default_factory=list)   # (model, fold index, model object)


def run(t: MatchTable, src: StyleSource, cfg: HarnessConfig, model_names, log=print,
        keep_models: bool = True) -> RunResult:
    folds = make_folds(t, src, cfg, log)
    n = len(t)
    names = ["glicko_raw", *model_names]
    pred = {m: np.full(n, np.nan) for m in names}
    has = np.zeros(n, dtype=bool)
    n_min = np.zeros(n, dtype=np.int64)
    fold_of = np.full(n, -1, dtype=np.int64)
    res = RunResult(cfg, names, folds, pred, has, n_min, fold_of)
    for fi, f in enumerate(folds):
        check_fold(t, src, f)
        t0 = time.perf_counter()
        tr = fold_data(t, src, f.train, f.tour, f.fit_ids, f.alpha)
        if cfg.alpha_rows == "train":
            sel = tr.has & (t.date[tr.rows] >= add_months(f.cutoff, -12 * cfg.alpha_years))
            f.alpha, f.n_alpha = fit_alpha(tr.glicko_logit[sel], tr.y[sel]), int(sel.sum())
            tr.offset = f.alpha * tr.glicko_logit
        va = fold_data(t, src, f.val, f.tour, f.fit_ids, f.alpha)
        te = fold_data(t, src, f.test, f.tour, f.fit_ids, f.alpha)
        has[f.test], n_min[f.test], fold_of[f.test] = te.has, np.minimum(te.na, te.nb), fi
        pred["glicko_raw"][f.test] = sigmoid(te.glicko_logit)
        msg = []
        for name in model_names:
            m = make(name)
            m.fit(tr, va)
            p = m.predict_proba(te)
            assert np.allclose(p[~te.has], sigmoid(te.offset[~te.has])), "null-vector row moved"
            pred[name][f.test] = p
            if keep_models:
                res.trained.append((name, fi, m))
            ll = _ll(p[te.has], te.y[te.has])
            msg.append(f"{name} {ll:.4f}")
        log(f"[{fi + 1}/{len(folds)}] {f.label} alpha={f.alpha:.3f} train {tr.has.sum():,} "
            f"val {va.has.sum():,} test {len(te):,} ({te.has.sum():,} with vectors)  "
            f"both-vector test ll: glicko_raw {_ll(sigmoid(te.glicko_logit[te.has]), te.y[te.has]):.4f} "
            + " ".join(msg) + f"  ({time.perf_counter() - t0:.0f}s)")
    return res


def _ll(p, y) -> float:
    if not len(p):
        return float("nan")
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))


# ------------------------------------------------------------------ scoring

SURF_NAMES = ("hard", "clay", "grass")


def scope_masks(t: MatchTable, res: RunResult) -> dict[str, np.ndarray]:
    test = res.fold_of >= 0
    tour = test & t.tour_level
    well = res.has & (res.n_min >= res.cfg.min_charted)
    out = {"tour": tour, "tour_both": tour & res.has, "tour_well": tour & well,
           "all": test, "all_both": test & res.has}
    for tr in res.cfg.tours:
        out[f"tour_{tr}"] = tour & (t.tour == tr)
        out[f"tour_both_{tr}"] = tour & res.has & (t.tour == tr)
    for i, s in enumerate(SURF_NAMES):
        out[f"tour_{s}"] = tour & (t.surface == i)
        out[f"tour_both_{s}"] = tour & res.has & (t.surface == i)
    return out


def ece(s: Score) -> float:
    """Expected calibration error from evaluate.score's 10 bins (p1 orientation)."""
    if not s.n:
        return float("nan")
    return sum(c[2] * abs(c[3] - c[4]) for c in s.calibration if c[2]) / s.n


def score_rows(t: MatchTable, p: np.ndarray, mask: np.ndarray) -> Score:
    idx = np.flatnonzero(mask)
    pw = np.where(t.y == 1.0, p, 1.0 - p)
    return score(pw, t.y.astype(np.int8), idx)


def metrics(t: MatchTable, res: RunResult) -> dict:
    """{model: {scope: Score}} pooled over all test blocks."""
    masks = scope_masks(t, res)
    return {m: {sc: score_rows(t, res.pred[m], mk) for sc, mk in masks.items()} for m in res.models}


def fold_metrics(t: MatchTable, res: RunResult, scopes=("tour", "tour_both")) -> dict:
    """{model: {(test_from, scope): Score}} per test block (both tours pooled)."""
    masks = scope_masks(t, res)
    starts = sorted({f.test_from for f in res.folds})
    in_block = {s: np.isin(res.fold_of, [i for i, f in enumerate(res.folds) if f.test_from == s])
                for s in starts}
    return {m: {(s, sc): score_rows(t, res.pred[m], masks[sc] & in_block[s]) for s in starts for sc in scopes}
            for m in res.models}


# ------------------------------------------------------------------ persistence

def save(con, t: MatchTable, res: RunResult, *, sweep_id: str, notes: str | None = None,
         bundle_dir: Path = BUNDLE_DIR, save_predictions: bool = True) -> dict[str, int]:
    """One model_run per model: pooled metrics (block 'wf_test'), per-block metrics,
    test predictions, and per-fold bundles (weights + fit_ids + alpha)."""
    import torch

    pooled, per_fold = metrics(t, res), fold_metrics(t, res)
    mkeys = models_db.match_keys(con) if save_predictions else {}
    test_rows = np.flatnonzero(res.fold_of >= 0)
    d_from, d_to = min(t.date[test_rows]), max(t.date[test_rows])
    trained = {(n, fi): m for n, fi, m in res.trained}
    run_ids = {}
    for name in res.models:
        model_cfg = trained[(name, 0)].config() if (name, 0) in trained else {}
        params = {"model": name, "harness": res.cfg.as_dict(), "model_config": model_cfg}
        run_id = models_db.start_run(con, f"matchup_{name}_{MODEL_VERSION}", params,
                                     sweep_id=sweep_id, notes=notes)
        run_ids[name] = run_id
        rows = [(run_id, "wf_test", sc, d_from, d_to, s.n, s.log_loss, s.brier, s.accuracy,
                 json.dumps(s.calibration)) for sc, s in pooled[name].items() if s.n]
        rows += [(run_id, f"test:{blk}", sc, blk, blk, s.n, s.log_loss, s.brier, s.accuracy,
                  json.dumps(s.calibration)) for (blk, sc), s in per_fold[name].items() if s.n]
        con.executemany("""INSERT OR REPLACE INTO run_metric (run_id, block, scope, date_from, date_to, n,
                               log_loss, brier, accuracy, calibration_json) VALUES (?,?,?,?,?,?,?,?,?,?)""", rows)
        if save_predictions:
            pk, p = t.player_key, res.pred[name]
            con.executemany("INSERT OR REPLACE INTO prediction VALUES (?,?,?,?,?,?,?)",
                            [(run_id, mkeys[int(t.match_id[i])], t.date[i], pk[int(t.p1[i])], pk[int(t.p2[i])],
                              float(p[i]), int(t.y[i])) for i in test_rows])
        for fi, f in enumerate(res.folds):
            m = trained.get((name, fi))
            path = None
            if m is not None and m.state() is not None:
                path = Path(bundle_dir) / str(run_id) / f"{f.tour}_{f.test_from}.pt"
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model": name, "config": m.config(), "state": m.state(),
                            "fit_ids": f.fit_ids, "alpha": f.alpha, "train_cutoff": f.cutoff,
                            "heavy_config": res.cfg.heavy_config}, path)
                try:
                    path = path.resolve().relative_to(ROOT).as_posix()
                except ValueError:
                    path = str(path)
            info = getattr(m, "info", {}) if m is not None else {}
            con.execute(
                """INSERT OR REPLACE INTO matchup_bundle (run_id, tour, test_from, test_to, train_cutoff,
                       fit_ids_json, alpha, n_alpha, n_train, n_val, n_test, info_json, path)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (run_id, f.tour, f.test_from, f.test_to, f.cutoff, json.dumps(f.fit_ids, sort_keys=True),
                 f.alpha, f.n_alpha, int(len(f.train)), int(len(f.val)), int(len(f.test)),
                 json.dumps(info, default=float), path))
        con.commit()
        models_db.finish_run(con, run_id)
    return run_ids


# ------------------------------------------------------------------ report

def _fmt(x, nd=4):
    return "" if x is None or x != x else f"{x:.{nd}f}"


def report(t: MatchTable, res: RunResult, run_ids: dict | None = None, title: str = "Matchup layer") -> str:
    pooled, per_fold = metrics(t, res), fold_metrics(t, res)
    base = "glicko"
    lines = [f"# {title}: walk-forward results", "",
             f"Generated by `scripts/matchup.py`. Harness config: `{json.dumps(res.cfg.as_dict())}`.", "",
             f"Folds: {len(res.folds)} (tour x {res.cfg.step_months}-month test block), "
             f"{min(f.test_from for f in res.folds)} to {max(f.test_to for f in res.folds)} (exclusive). "
             f"Base rating run: {getattr(t, 'base_model_id', 'glicko2_v1')} run "
             f"{res.cfg.glicko_run_id}. `glicko_raw` = that run's probabilities unchanged; "
             "`glicko` = alpha-calibrated (the benchmark every interaction model must beat). "
             "The names are historical: with a points_v1 base they mean points_v1 raw / calibrated.", ""]
    if run_ids:
        lines += ["models.db runs: " + ", ".join(f"{m} = {r}" for m, r in run_ids.items()), ""]
    order = ["tour", "tour_both", "tour_well", "tour_atp", "tour_wta", "tour_both_atp", "tour_both_wta",
             "tour_hard", "tour_clay", "tour_grass", "tour_both_hard", "tour_both_clay", "tour_both_grass",
             "all", "all_both"]
    desc = {"tour": "tour-level main draws (every match; no vector -> Glicko-only)",
            "tour_both": "tour-level, both players have style vectors",
            "tour_well": f"tour-level, both players >= {res.cfg.min_charted} charted matches",
            "all": "every scored match", "all_both": "every scored match, both with vectors"}
    lines += ["## Pooled test metrics", "",
              "Delta = log loss minus the alpha-calibrated Glicko benchmark on the same rows "
              "(negative is better). ECE = expected calibration error (10 bins).", ""]
    for sc in order:
        if sc not in pooled[res.models[0]]:
            continue
        n = pooled[res.models[0]][sc].n
        lines += [f"### `{sc}` (n = {n:,})" + (f": {desc[sc]}" if sc in desc else ""), "",
                  "| model | log loss | delta | Brier | accuracy | ECE |", "|---|---|---|---|---|---|"]
        b = pooled.get(base, {}).get(sc)
        for m in res.models:
            s = pooled[m][sc]
            d = s.log_loss - b.log_loss if b is not None and b.n else float("nan")
            lines.append(f"| {m} | {_fmt(s.log_loss)} | {('+' if d > 0 else '') + _fmt(d)} | {_fmt(s.brier)} | "
                         f"{_fmt(s.accuracy, 3)} | {_fmt(ece(s))} |")
        lines.append("")
    lines += ["## Per test block, `tour_both` log loss", "",
              "| block | n | " + " | ".join(res.models) + " |", "|---|---|" + "---|" * len(res.models)]
    blocks = sorted({b for (b, sc) in per_fold[res.models[0]] if sc == "tour_both"})
    for blk in blocks:
        ss = [per_fold[m][(blk, "tour_both")] for m in res.models]
        lines.append(f"| {blk} | {ss[0].n:,} | " + " | ".join(_fmt(s.log_loss) for s in ss) + " |")
    lines += ["", "## Calibration, `tour_both` (p1 orientation: mean predicted / win rate / n)", "",
              "| bin | " + " | ".join(res.models) + " |", "|---|" + "---|" * len(res.models)]
    cals = [pooled[m]["tour_both"].calibration for m in res.models]
    for bi in range(len(cals[0])):
        cells = [f"{_fmt(c[bi][3], 3)} / {_fmt(c[bi][4], 3)} / {c[bi][2]:,}" if c[bi][2] else "" for c in cals]
        lines.append(f"| {cals[0][bi][0]:.1f}-{cals[0][bi][1]:.1f} | " + " | ".join(cells) + " |")
    lines += ["", "## Folds", "", "| fold | train cutoff | alpha | fits (serve / return / rally fit_date) | "
              "train rows | val rows | test rows |", "|---|---|---|---|---|---|---|"]
    for f in res.folds:
        fd = " / ".join(f.fit_ids[s].rsplit(":", 1)[1] for s in ("serve", "return", "rally"))
        lines.append(f"| {f.label} | {f.cutoff} | {f.alpha:.3f} | {fd} | {len(f.train):,} | {len(f.val):,} | "
                     f"{len(f.test):,} |")
    return "\n".join(lines) + "\n"
