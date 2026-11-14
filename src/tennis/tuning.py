"""Parameter sweeps for walk-forward rating models (Elo, Glicko-2), recorded in models.db.

Workflow (scripts/sweep.py; scripts/elo_sweep.py is the same with model elo_v1):
  sweep     run every parameter setting over the full history; score ONLY the
            validation block; record model_run + run_metric rows.
  show      leaderboard by validation log loss.
  finalize  for the chosen run(s): score the test block, optionally save per-match
            predictions and pre-match ratings. The test block is never used to choose.
The model of an existing run always comes from its model_run.model_id.
"""
from __future__ import annotations

import itertools
import json
import sqlite3
from dataclasses import dataclass, field, fields

from . import elo, models_db, ratingmodels
from .evaluate import Score, score

MODEL_ID = "elo_v1"                       # default model
DEFAULT_BLOCKS = {"val": ("2019-01-01", "2021-12-31"), "test": ("2022-01-01", "2026-12-31")}
SCOPES = ("tour", "tour_atp", "tour_wta", "all")
PRIMARY_SCOPE = "tour"
DEFAULT_GRID = ratingmodels.get(MODEL_ID).default_grid


@dataclass
class Prepared:
    m: elo.Matches
    winner_is_p1: bytearray            # canonical orientation by stable player key
    player_key: dict[int, str]
    index_cache: dict = field(default_factory=dict)

    def indices(self, date_from: str, date_to: str, scope: str) -> list[int]:
        key = (date_from, date_to, scope)
        if key not in self.index_cache:
            m = self.m
            test = {"tour": lambda i: m.tour_level[i],
                    "tour_atp": lambda i: m.tour_level[i] and m.tour[i] == "atp",
                    "tour_wta": lambda i: m.tour_level[i] and m.tour[i] == "wta",
                    "all": lambda i: True}[scope]
            self.index_cache[key] = [i for i in range(len(m))
                                     if date_from <= m.start[i] <= date_to
                                     and m.status[i] in elo.SCORED_STATUS and test(i)]
        return self.index_cache[key]


def prepare(con: sqlite3.Connection) -> Prepared:
    """Load matches once (tennis.db must be attached as 'data')."""
    m = elo.load_matches(con)
    pk = models_db.player_keys(con)
    flag = bytearray(1 if pk[w] < pk[l] else 0 for w, l in zip(m.winner, m.loser))
    return Prepared(m, flag, pk)


def expand_grid(spec, params_cls=elo.EloParams) -> list:
    """{'k': [100, 250], 'shape': [0.4]} (or a list of such dicts) -> parameter sets.
    Unlisted parameters keep their defaults."""
    specs = spec if isinstance(spec, list) else [spec]
    valid = {f.name for f in fields(params_cls)}
    out, seen = [], set()
    for s in specs:
        unknown = set(s) - valid
        if unknown:
            raise ValueError(f"unknown parameter(s) {sorted(unknown)}; valid: {sorted(valid)}")
        names = list(s)
        for combo in itertools.product(*(v if isinstance(v, list) else [v] for v in s.values())):
            p = params_cls(**{n: float(v) for n, v in zip(names, combo)})
            if p not in seen:
                seen.add(p)
                out.append(p)
    return out


def _write_metrics(con, run_id, block, date_from, date_to, scores: dict[str, Score]) -> None:
    con.executemany(
        """INSERT OR REPLACE INTO run_metric (run_id, block, scope, date_from, date_to, n, log_loss,
               brier, accuracy, calibration_json) VALUES (?,?,?,?,?,?,?,?,?,?)""",
        [(run_id, block, scope, date_from, date_to, s.n, s.log_loss, s.brier, s.accuracy,
          json.dumps(s.calibration)) for scope, s in scores.items()])
    con.commit()


def score_block(prep: Prepared, p_winner, date_from: str, date_to: str) -> dict[str, Score]:
    return {sc: score(p_winner, prep.winner_is_p1, prep.indices(date_from, date_to, sc))
            for sc in SCOPES}


def sweep(con, prep: Prepared, grid: list, *, sweep_id: str,
          val: tuple[str, str] = DEFAULT_BLOCKS["val"], progress=print,
          model: str = MODEL_ID) -> list[tuple[int, Score]]:
    """One full-history pass per distinct update setting; every blend value of that
    setting is scored from the same pass. Each (setting, blend) is its own model_run."""
    spec = ratingmodels.get(model)
    if spec.attach:
        spec.attach(con, prep.m)
    groups: dict = {}
    for p in grid:
        if not isinstance(p, spec.params_cls):
            raise TypeError(f"{model} expects {spec.params_cls.__name__}, got {type(p).__name__}")
        groups.setdefault(p.update_key(), []).append(p)
    out, done = [], 0
    for g, (key, members) in enumerate(groups.items(), 1):
        res = spec.run(prep.m, key)
        for p in members:
            run_id = models_db.start_run(con, model, p.as_dict(), sweep_id=sweep_id)
            scores = score_block(prep, spec.predict(res, p), *val)
            _write_metrics(con, run_id, "val", *val, scores)
            models_db.finish_run(con, run_id)
            s = scores[PRIMARY_SCOPE]
            out.append((run_id, s))
            done += 1
            progress(f"[{done}/{len(grid)} pass {g}/{len(groups)}] run {run_id} {json.dumps(p.as_dict())}  "
                     f"val {PRIMARY_SCOPE} log_loss={s.log_loss:.4f} brier={s.brier:.4f} acc={s.accuracy:.3f}")
    return out


def leaderboard(con, *, sweep_id: str | None = None, scope: str = PRIMARY_SCOPE, top: int = 15,
                model: str | None = None) -> list:
    """Rows: run_id, sweep_id, params_json, n, val log_loss, brier, accuracy, test log_loss, model_id.
    model=None lists every model together (same blocks and scopes, so directly comparable)."""
    sql = """SELECT r.run_id, r.sweep_id, r.params_json, v.n, v.log_loss, v.brier, v.accuracy,
                    t.log_loss AS test_log_loss, r.model_id
             FROM model_run r JOIN run_metric v ON v.run_id=r.run_id AND v.block='val' AND v.scope=?
             LEFT JOIN run_metric t ON t.run_id=r.run_id AND t.block='test' AND t.scope=?
             WHERE r.finished_at IS NOT NULL"""
    args = [scope, scope]
    if model:
        sql += " AND r.model_id=?"
        args.append(model)
    if sweep_id:
        sql += " AND r.sweep_id=?"
        args.append(sweep_id)
    return con.execute(sql + " ORDER BY v.log_loss LIMIT ?", args + [top]).fetchall()


def model_of(con, run_id: int):
    """(registry entry, params) of a recorded run."""
    row = con.execute("SELECT model_id, params_json FROM model_run WHERE run_id=?", (run_id,)).fetchone()
    if row is None:
        raise ValueError(f"run {run_id} not found in models.db")
    spec = ratingmodels.get(row[0])
    return spec, spec.params_cls(**json.loads(row[1]))


def params_of(con, run_id: int):
    return model_of(con, run_id)[1]


def finalize(con, prep: Prepared, run_id: int, *, test: tuple[str, str] = DEFAULT_BLOCKS["test"],
             save_predictions: bool = False) -> dict[str, Score]:
    """Score the held-out test block for one chosen run; optionally save per-match rows."""
    spec, params = model_of(con, run_id)
    if spec.attach:
        spec.attach(con, prep.m)
    res = spec.run(prep.m, params)
    scores = score_block(prep, res.p_winner, *test)
    _write_metrics(con, run_id, "test", *test, scores)
    if save_predictions:
        _save(con, prep, run_id, res)
    return scores


def _save(con, prep: Prepared, run_id: int, res) -> None:
    mk = models_db.match_keys(con)
    m, pk = prep.m, prep.player_key
    dw_arr, dl_arr = getattr(res, "d_winner", None), getattr(res, "d_loser", None)
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
        dw = dw_arr[i] if dw_arr is not None else None
        dl = dl_arr[i] if dl_arr is not None else None
        rats.append((run_id, key, pk[w], res.r_winner[i], sw if sw == sw else None, dw, res.n_winner[i]))
        rats.append((run_id, key, pk[l], res.r_loser[i], sl if sl == sl else None, dl, res.n_loser[i]))
        if len(preds) >= 100_000:
            _flush(con, preds, rats)
    _flush(con, preds, rats)
    con.commit()


def _flush(con, preds, rats):
    con.executemany("INSERT INTO prediction VALUES (?,?,?,?,?,?,?)", preds)
    con.executemany("INSERT INTO pre_match_rating VALUES (?,?,?,?,?,?,?)", rats)
    preds.clear()
    rats.clear()
