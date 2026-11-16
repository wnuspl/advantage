"""Replay, score and record points-model runs in models.db (from v1 tuning.py, points only).

A run = one PointsParams setting replayed over the full history. Validation (2019-21) is
scored raw and calibrated; test (2022-26) only by finalize(), once per finished model.
blend and unc are prediction-only, so a sweep replays each update setting once and scores
every (blend, unc) from that pass.

run_metric scopes: tour, tour_atp, tour_wta, all (raw, as in v1) and tour_cal, tour_atp_cal,
tour_wta_cal (walk-forward calibrated: per quarter, tour and best-of, 10-year history;
calibrate.calibrated). Calibration needs the full history of predictions, so it is computed
from the same pass.
"""
from __future__ import annotations

import itertools
import json
from dataclasses import fields

import numpy as np

from .. import calibrate, models_db, points_rating
from .data import BLOCKS, Prepared
from .scoring import Metrics, metrics, oriented

RAW_SCOPES = ("tour", "tour_atp", "tour_wta", "all")
CAL_SCOPES = ("tour_cal", "tour_atp_cal", "tour_wta_cal")
PRIMARY = "tour_cal"
MODEL_ID = "points_v1"


def params_of(con, run_id: int) -> tuple[str, points_rating.PointsParams]:
    row = con.execute("SELECT model_id, params_json FROM model_run WHERE run_id=?", (run_id,)).fetchone()
    if row is None:
        raise ValueError(f"run {run_id} not found in models.db")
    if not row[0].startswith("points_"):
        raise ValueError(f"run {run_id} is {row[0]}, not a points model")
    return row[0], points_rating.PointsParams(**json.loads(row[1]))


def expand_grid(spec, base: points_rating.PointsParams | None = None) -> list[points_rating.PointsParams]:
    """{'q': [1e-5, 2e-5], 'unc': [3]} (or a list of such dicts) -> settings; unlisted
    parameters keep `base` (default PointsParams())."""
    base = base or points_rating.PointsParams()
    valid = {f.name for f in fields(points_rating.PointsParams)}
    out, seen = [], set()
    for s in (spec if isinstance(spec, list) else [spec]):
        unknown = set(s) - valid
        if unknown:
            raise ValueError(f"unknown parameter(s) {sorted(unknown)}")
        names = list(s)
        for combo in itertools.product(*(v if isinstance(v, list) else [v] for v in s.values())):
            p = points_rating.PointsParams(**{**base.as_dict(), **{n: float(v) for n, v in zip(names, combo)}})
            if p not in seen:
                seen.add(p)
                out.append(p)
    return out


class Scored:
    """One run's predictions on one block: raw and calibrated, canonical orientation."""

    def __init__(self, prep: Prepared, p_winner, block: tuple[str, str], by_best_of: bool = True):
        lo, hi = block
        self.block = block
        cal, idx, self.alphas = calibrate.calibrated(prep.m, p_winner, lo, hi,
                                                     best_of=prep.best_of if by_best_of else None)
        self.idx = idx
        self.raw, self.y = oriented(p_winner, prep.winner_is_p1, idx)
        flag = np.array([prep.winner_is_p1[i] for i in idx], dtype=float)
        self.cal = np.where(flag == 1, cal, 1 - cal)
        self.tour = np.array([prep.m.tour[i] for i in idx])
        self.quarter = calibrate.block_of([prep.m.start[i] for i in idx])
        all_idx = prep.indices(lo, hi, "all")
        self.all_raw, self.all_y = oriented(p_winner, prep.winner_is_p1, all_idx)

    def select(self, scope: str) -> np.ndarray:
        base = scope.removesuffix("_cal")
        if base == "tour":
            return np.ones(len(self.idx), bool)
        return self.tour == base[5:]

    def scores(self, ref: "Scored | None" = None) -> dict[str, Metrics]:
        out = {}
        for sc in RAW_SCOPES[:3] + CAL_SCOPES:
            s = self.select(sc)
            p = (self.cal if sc.endswith("_cal") else self.raw)[s]
            r = None if ref is None else (ref.cal if sc.endswith("_cal") else ref.raw)[s]
            out[sc] = metrics(p, self.y[s], ref=r, block=self.quarter[s])
        out["all"] = metrics(self.all_raw, self.all_y)
        return out


def write_metrics(con, run_id: int, label: str, block: tuple[str, str], scores: dict[str, Metrics]) -> None:
    con.executemany(
        """INSERT OR REPLACE INTO run_metric (run_id, block, scope, date_from, date_to, n, log_loss,
               brier, accuracy, calibration_json) VALUES (?,?,?,?,?,?,?,?,?,?)""",
        [(run_id, label, sc, block[0], block[1], s.n, s.log_loss, s.brier, s.accuracy,
          json.dumps(s.calibration)) for sc, s in scores.items()])
    con.commit()


def record(con, params, scores: dict[str, Metrics], *, label: str = "val", block=BLOCKS["val"],
           model_id: str = MODEL_ID, sweep_id: str | None = None, notes: str | None = None) -> int:
    run_id = models_db.start_run(con, model_id, params.as_dict(), sweep_id=sweep_id, notes=notes)
    write_metrics(con, run_id, label, block, scores)
    models_db.finish_run(con, run_id)
    return run_id


def sweep(con, prep: Prepared, grid: list, *, sweep_id: str, block=BLOCKS["val"], model_id: str = MODEL_ID,
          log=print) -> list[tuple[int, Metrics]]:
    groups: dict = {}
    for p in grid:
        groups.setdefault(p.update_key(), []).append(p)
    out, done = [], 0
    for g, (key, members) in enumerate(groups.items(), 1):
        res = points_rating.run(prep.m, key)
        for p in members:
            sc = Scored(prep, points_rating.predict(res, p), block).scores()
            rid = record(con, p, sc, block=block, model_id=model_id, sweep_id=sweep_id)
            done += 1
            out.append((rid, sc[PRIMARY]))
            log(f"[{done}/{len(grid)} pass {g}/{len(groups)}] run {rid}  val cal {sc[PRIMARY].log_loss:.4f}  "
                f"raw {sc['tour'].log_loss:.4f}  {json.dumps(diff_params(p))}")
    return out


def diff_params(p, base=None) -> dict:
    """Parameters that differ from `base` (default PointsParams()), for display."""
    b = (base or points_rating.PointsParams()).as_dict()
    return {k: v for k, v in p.as_dict().items() if b[k] != v}


def leaderboard(con, *, sweep_id: str | None = None, scope: str = PRIMARY, top: int = 15,
                model_id: str | None = None) -> list:
    """(run_id, sweep_id, params_json, n, val log loss, brier, test log loss, model_id),
    best val first."""
    sql = """SELECT r.run_id, r.sweep_id, r.params_json, v.n, v.log_loss, v.brier, t.log_loss, r.model_id
             FROM model_run r JOIN run_metric v ON v.run_id=r.run_id AND v.block='val' AND v.scope=?
             LEFT JOIN run_metric t ON t.run_id=r.run_id AND t.block='test' AND t.scope=?
             WHERE r.finished_at IS NOT NULL AND r.model_id LIKE 'points_%'"""
    args: list = [scope, scope]
    if model_id:
        sql += " AND r.model_id=?"
        args.append(model_id)
    if sweep_id:
        sql += " AND r.sweep_id=?"
        args.append(sweep_id)
    return con.execute(sql + " ORDER BY v.log_loss LIMIT ?", args + [top]).fetchall()


def has_test(con, run_id: int) -> bool:
    return con.execute("SELECT 1 FROM run_metric WHERE run_id=? AND block='test'", (run_id,)).fetchone() is not None


def finalize(con, prep: Prepared, run_id: int, *, block=BLOCKS["test"], force: bool = False,
             save_predictions: bool = False) -> dict[str, Metrics]:
    """Score the held-out test block for a finished model, once (force to redo)."""
    if has_test(con, run_id) and not force:
        raise ValueError(f"run {run_id} already has test metrics: test is scored once per model")
    _, params = params_of(con, run_id)
    res = points_rating.run(prep.m, params)
    scores = Scored(prep, res.p_winner, block).scores()
    write_metrics(con, run_id, "test", block, scores)
    if save_predictions:
        save(con, prep, run_id, res)
    return scores


def save(con, prep: Prepared, run_id: int, res) -> None:
    """Per-match predictions (canonical order) and pre-match ratings of a run."""
    mk = models_db.match_keys(con)
    m, pk = prep.m, prep.player_key
    con.execute("DELETE FROM prediction WHERE run_id=?", (run_id,))
    con.execute("DELETE FROM pre_match_rating WHERE run_id=?", (run_id,))
    preds, rats = [], []
    for i in range(len(m)):
        w, l, key = m.winner[i], m.loser[i], mk[m.match_id[i]]
        pw = res.p_winner[i]
        if prep.winner_is_p1[i]:
            preds.append((run_id, key, m.start[i], pk[w], pk[l], pw, 1))
        else:
            preds.append((run_id, key, m.start[i], pk[l], pk[w], 1.0 - pw, 0))
        sw, sl = res.s_winner[i], res.s_loser[i]
        rats.append((run_id, key, pk[w], res.r_winner[i], sw if sw == sw else None, res.d_winner[i], res.n_winner[i]))
        rats.append((run_id, key, pk[l], res.r_loser[i], sl if sl == sl else None, res.d_loser[i], res.n_loser[i]))
        if len(preds) >= 100_000:
            _flush(con, preds, rats)
    _flush(con, preds, rats)
    con.commit()


def _flush(con, preds, rats):
    con.executemany("INSERT INTO prediction VALUES (?,?,?,?,?,?,?)", preds)
    con.executemany("INSERT INTO pre_match_rating VALUES (?,?,?,?,?,?,?)", rats)
    preds.clear()
    rats.clear()
