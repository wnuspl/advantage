"""The one canonical prediction path for points models (milestone A1): `PointsPredictor`.

    pred = PointsPredictor.load(790)                  # replay cached under models/engine/
    pred.predict(MatchQuery("atp", a, b, on="2026-11-15", surface="hard", best_of=3))

Pre-match, for A vs B (exactly the arithmetic run() uses before a match):
  1. serve-point logits x_A, x_B and variances from the final rating state, variances grown
     by inactivity to the rating day, surface deviations weighted by `blend`
     (points_rating.matchup);
  2. context offsets (rust, load, age) from each player's bookkeeping after the last replayed
     match: x_A += beta_serve . c_A - beta_ret . c_B (and likewise x_B);
  3. Markov roll-up with the match's best-of and deciding-set rule, then the `unc` shrink
     (points_rating.win_probability) = p_raw;
  4. calibration: sigmoid(alpha * logit p_raw), alpha per (tour, best-of) for the quarter of
     the predictor's date, fitted on the 10 years before it (calibrate.alphas_at).
The rating day is the edition's start when the query names an edition the data knows (run()
rates every match of an edition on its start date), else the match day.

In play (query.live with status 'live'): each serve logit is first updated on the serve
points won so far (points_rating.in_match_form, form_var 0.02), then rolled up from the
current score (points_rating.in_play_probability: Markov from the score + the same `unc`
shrink). No calibration in play: the alphas are fitted on pre-match predictions. A live
state with a `problem`, or no known server, gives the pre-match probability with
in_play=False and the reason in notes: callers must not act on it as an in-play price.

Any query.live (live or finished: a settled match's final stats) also puts each serve
stream's in-match form in detail (`stream_form`: form_a / form_b, serve_a_form / serve_b_form,
form_points_a / form_points_b); the recorder stores the final values at settlement.

`detail["p_sd"]`: the SD of the output probability under the ratings' own variances
(p_spread; in play, the form-updated variances), for display as the model's certainty;
`p_sd_pre` keeps the pre-match one. `rating_a` / `rating_b`: each player's serve and return
rating going into the match (points_rating.player_ratings, logits relative to `base`), never
touched by the in-play update; with a player index, each carries its rank (1 = best overall
serve / return) among the tour's players rated in the year before the match, and `ref` holds that
population's mean ratings (the ratings are not centred: an average player is not 0).

`explain` (engine/explain.py): the probability built up step by step from two tour-average
players (serve, return, surface, context per player and group, rating doubt, calibration;
in play: score, each stream's form), the steps summing to the output. `ctx_c_a` /
`ctx_c_b`: each player's context feature vector (context.FEATURES).

`stale`: a player's last rated match is more than 120 days before the match (or there is
none). A query dated before the latest replayed edition gets a note: its ratings already
include later results (use `as_of` for a point-in-time prediction).

The replay (~5 min over all of tennis.db) is pickled to models/engine/ and reused while the
cache key holds: tennis.db contents, the model code, the run's params, as_of, and the
calibration quarter.
"""
from __future__ import annotations

import bisect
import hashlib
import json
import math
import pickle
import sqlite3
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from .. import calibrate, context, markov, matches, models_db, points_rating
from ..kalshi import LiveState
from . import explain as explain_mod
from . import players as players_mod
from . import state as state_mod
from .predictor import MatchQuery, Prediction

ROOT = models_db.ROOT
CACHE_DIR = ROOT / "models" / "engine"
STALE_DAYS = 120
REF_DAYS = 365          # the tour reference population: players rated in the year before the match
RATING_KEYS = ("serve", "ret", "serve_sd", "ret_sd", "seen", "serve_rank", "ret_rank")
CACHE_VERSION = b"engine-points-cache-v1"
CODE = ("points_rating.py", "context.py", "markov.py", "matches.py", "calibrate.py",
        "engine/predict.py", "engine/players.py", "engine/state.py")


@dataclass
class ModelState:
    """Everything predict() needs from a replay. Plain data: picklable."""
    run_id: int | None
    model_id: str
    params: points_rating.PointsParams
    state: points_rating.PointsState
    ctx_states: dict            # context.player_states() after the last replayed match ({} = no context)
    dob: dict                   # player_id -> ISO birth date
    alphas: dict                # calibrate.alphas_at() for block `alpha_block`
    alpha_block: str
    editions: dict              # edition_id -> (start ISO, tournament_id, surface, best_of)
    edition_rule: dict          # edition_id -> deciding-set rule seen in its matches
    tournament_rule: dict       # tournament_id -> rule of its latest edition with matches
    last_start: str | None      # latest edition start replayed
    n_matches: int
    as_of: str | None
    players: tuple | None = None   # engine.players index


def build_model(m, params: points_rating.PointsParams, edition_of, dob: dict, editions: dict, *,
                today: str, run_id: int | None = None, model_id: str = "points_v1",
                as_of: str | None = None, players_index: tuple | None = None) -> ModelState:
    """Replay `m` (points_rating.attach()ed; context.attach()ed when params have context) and
    keep what predictions need. edition_of[i] = edition id of match i; editions: edition_id
    -> (start, tournament_id, surface, best_of) for every edition a query may name."""
    res = points_rating.run(m, params)
    fmt = m.extra["points"].fmt
    n = len(m)
    best_of = [points_rating.FORMATS[fmt[i]][0] for i in range(n)]
    block = calibrate.block_start(today)
    alphas = calibrate.alphas_at(m, res.p_winner, block, best_of=best_of)
    erule: dict = {}
    for i in range(n):
        erule[edition_of[i]] = points_rating.FORMATS[fmt[i]][1]
    trule = tournament_rules(editions, erule)
    ctx = context.player_states(m, edition_of) if params.has_context() else {}
    return ModelState(run_id=run_id, model_id=model_id, params=params, state=res.state,
                      ctx_states=ctx, dob=dob, alphas=alphas, alpha_block=block, editions=editions,
                      edition_rule=erule, tournament_rule=trule,
                      last_start=m.start[n - 1] if n else None, n_matches=n, as_of=as_of,
                      players=players_index)


def tournament_rules(editions: dict, erule: dict) -> dict:
    """tournament_id -> deciding-set rule of its latest edition with matches."""
    trule: dict = {}
    for eid, (start, tid, _s, _b) in sorted(editions.items(), key=lambda kv: (kv[1][0], kv[0])):
        if eid in erule:
            trule[tid] = erule[eid]
    return trule


def _sig(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def p_spread(f, xa: float, xb: float, va: float, vb: float, h: float = 0.01) -> float:
    """SD of the output probability f(xa, xb) under the two serve logits' rating variances
    (delta method, central differences): how far the price could move if the ratings are
    off by their own uncertainty. Inactivity grows va/vb, so stale ratings get wider spreads.
    It leaves out the market, the feed and model error beyond the ratings."""
    da = (f(xa + h, xb) - f(xa - h, xb)) / (2 * h)
    db = (f(xa, xb + h) - f(xa, xb - h)) / (2 * h)
    return math.sqrt(da * da * va + db * db * vb)


class PointsPredictor:
    """A Predictor (engine.predictor) for points_v1-family runs."""

    def __init__(self, model: ModelState, *, calibrated: bool = True, live_form: bool = True,
                 form_var: float = points_rating.IN_PLAY_FORM_VAR, max_stale_days: int = STALE_DAYS,
                 explain: bool = True):
        self.model = model
        self.calibrated = calibrated
        self.live_form = live_form
        self.form_var = form_var
        self.max_stale_days = max_stale_days
        self.explain = explain
        tag = f"run{model.run_id}" if model.run_id is not None else "unrecorded"
        self.name = f"{model.model_id}/{tag}" + (f"@{model.as_of}" if model.as_of else "") \
            + ("" if calibrated else "/raw")
        self._players = None
        self._ref: dict = {}

    # ------------------------------------------------------------------ loading

    @classmethod
    def load(cls, run_id: int = 790, *, as_of: str | None = None, today: str | None = None,
             models_path: Path = models_db.MODELS_DB, data_path: Path = models_db.DATA_DB,
             cache_dir: Path | None = CACHE_DIR, rebuild: bool = False, log=print,
             state_dir: Path | None = None, use_state: bool = True, **kw) -> "PointsPredictor":
        """Replay run `run_id` from models.db over tennis.db (editions starting before `as_of`
        only, if given), or reuse the cached replay. `today` (default as_of, else the system
        date) picks the calibration quarter and the name resolver's 'recent' window."""
        today = today or as_of or date.today().isoformat()
        con = sqlite3.connect(models_db._uri(models_path, "ro"), uri=True)
        try:
            models_db.attach_data(con, data_path)
            model_id, params = run_params(con, run_id)
            key = cache_key(con, run_id, params, as_of, today)
            path = None if cache_dir is None else \
                Path(cache_dir) / f"run{run_id}{'_' + as_of if as_of else ''}.pkl"
            model = None if (rebuild or path is None) else _read_cache(path, key, log)
            if model is None:
                t = time.perf_counter()
                log(f"model: replaying run {run_id} over tennis.db (cached afterwards) ...")
                model = model_from_db(con, run_id, model_id, params, as_of=as_of, today=today,
                                      use_state=use_state, state_dir=state_dir, log=log)
                log(f"model: replay done in {time.perf_counter() - t:.0f}s "
                    f"({model.n_matches:,} matches, latest edition {model.last_start})")
                if path is not None:
                    _write_cache(path, key, model, log)
        finally:
            con.close()
        return cls(model, **kw)

    @property
    def players(self) -> players_mod.Resolver:
        if self._players is None:
            if self.model.players is None:
                raise ValueError("this model was built without a player index")
            self._players = players_mod.Resolver(self.model.players)
        return self._players

    def deciding_rule(self, edition_id: int | None = None, tournament_id: int | None = None) -> str:
        """Deciding-set rule as the replay inferred it: the edition's own, else its
        tournament's latest, else 'tb'."""
        md = self.model
        if edition_id in md.edition_rule:
            return md.edition_rule[edition_id]
        if tournament_id is None and edition_id in md.editions:
            tournament_id = md.editions[edition_id][1]
        return md.tournament_rule.get(tournament_id, "tb")

    # ------------------------------------------------------------------ prediction

    def reference(self, tour: str, on: str) -> dict | None:
        """The tour's rated population on ISO date `on`: players of `tour` (engine.players
        index) rated in the REF_DAYS before it. Mean overall serve and return ratings (the
        ratings are not centred: an average player is not 0) and the sorted values for ranks.
        None without a player index."""
        if self.model.players is None:
            return None
        key = (tour, on)
        if key not in self._ref:
            st, info = self.model.state, self.model.players[2]
            lo = date.fromisoformat(on).toordinal() - REF_DAYS
            hi = date.fromisoformat(on).toordinal()
            pids = [i for i in st.serve if lo <= st.last[i] <= hi and (info.get(i) or (None,))[0] == tour]
            if not pids:
                self._ref[key] = None
            else:
                s = sorted(st.serve[i] for i in pids)
                r = sorted(st.ret[i] for i in pids)
                self._ref[key] = {"serve": sum(s) / len(s), "ret": sum(r) / len(r), "n": len(pids),
                                  "sorted_serve": s, "sorted_ret": r}
            if len(self._ref) > 64:
                self._ref.pop(next(iter(self._ref)))
        return self._ref[key]

    def _ranks(self, ref: dict, pid: int) -> tuple[int | None, int | None]:
        """1 = best overall serve / return rating in the reference population (None if the
        player is not in it, e.g. new or inactive)."""
        st = self.model.state
        if pid not in st.serve:
            return None, None
        n = ref["n"]
        return (n - bisect.bisect_right(ref["sorted_serve"], st.serve[pid]) + 1,
                n - bisect.bisect_right(ref["sorted_ret"], st.ret[pid]) + 1)

    def prematch(self, q: MatchQuery) -> dict:
        """The pre-match pieces: logits, variances, context offsets, p_raw, p_cal, alpha."""
        md, p = self.model, self.model.params
        st = md.state
        ed = md.editions.get(q.edition_id) if q.edition_id is not None else None
        rating_day = ed[0] if ed and ed[0] <= q.on else q.on
        surface = q.surface if q.surface is not None else (ed[2] if ed else None)
        a, b = q.player_a, q.player_b
        if a == b:
            raise ValueError("player_a and player_b are the same player")
        xa, xb, va, vb = points_rating.matchup(st, a, b, q.tour, surface, rating_day)
        oa = ob = 0.0
        ca = cb = None
        if p.has_context():
            d0 = date.fromisoformat(rating_day).toordinal()
            c = context.features_from_states(md.ctx_states, md.dob, {a: (q.edition_id, d0), b: (q.edition_id, d0)})
            bs, br = p.context_betas(q.tour)
            oa = float(c[a] @ bs - c[b] @ br)
            ob = float(c[b] @ bs - c[a] @ br)
            ca, cb = [float(v) for v in c[a]], [float(v) for v in c[b]]
            xa, xb = xa + oa, xb + ob
        p_raw = points_rating.win_probability(st, xa, xb, va, vb, q.best_of, q.final)
        cal = self.calibrated and q.tour not in p.raw_tours()
        alpha = calibrate.alpha_for(md.alphas, q.tour, q.best_of) if cal else 1.0
        p_cal = calibrate.apply(p_raw, alpha) if cal else p_raw
        def f(a, b):
            pr = points_rating.win_probability(st, a, b, va, vb, q.best_of, q.final)
            return calibrate.apply(pr, alpha) if cal else pr
        p_sd = p_spread(f, xa, xb, va, vb)
        ra = points_rating.player_ratings(st, a, q.tour, surface, rating_day)
        rb = points_rating.player_ratings(st, b, q.tour, surface, rating_day)
        ref = self.reference(q.tour, rating_day)
        if ref is not None:
            for pid, rr in ((a, ra), (b, rb)):
                rr["serve_rank"], rr["ret_rank"] = self._ranks(ref, pid)
        return {"x_a": xa, "x_b": xb, "v_a": va, "v_b": vb, "ctx_a": oa, "ctx_b": ob, "p_sd": p_sd,
                "p_sd_pre": p_sd, "base": ra["base"],
                "ref": None if ref is None else {"serve": ref["serve"], "ret": ref["ret"], "n": ref["n"]},
                "rating_a": {k: ra[k] for k in RATING_KEYS if k in ra},
                "rating_b": {k: rb[k] for k in RATING_KEYS if k in rb},
                "serve_a": _sig(xa), "serve_b": _sig(xb), "p_raw": p_raw, "p_prematch": p_cal,
                "alpha": alpha, "rating_day": rating_day, "surface": surface,
                "best_of": q.best_of, "final": q.final, "ctx_c_a": ca, "ctx_c_b": cb}

    def predict(self, q: MatchQuery) -> Prediction:
        md, st = self.model, self.model.state
        d = self.prematch(q)
        notes: list[str] = []
        day = date.fromisoformat(q.on).toordinal()
        cut = day - self.max_stale_days
        last = {pid: st.last.get(pid) for pid in (q.player_a, q.player_b)}
        stale = any(v is None or v < cut for v in last.values())
        for pid, v in last.items():
            if v is None:
                notes.append(f"player {pid} has no rated match: new-player prior")
        d["last_a"], d["last_b"] = (None if last[k] is None else date.fromordinal(last[k]).isoformat()
                                    for k in (q.player_a, q.player_b))
        if stale:
            notes.append(f"stale: a player's last rated match is over {self.max_stale_days} days old")
        if md.last_start and q.on < md.last_start:
            notes.append(f"ratings include editions up to {md.last_start}, after this match "
                         "(not point-in-time; use as_of)")
        if q.surface is None and d["surface"] is None:
            notes.append("surface unknown: no surface deviations")
        p_out, p_raw, in_play = d["p_prematch"], d["p_raw"], False

        live = q.live
        if live is not None and self.live_form:
            d.update(self.stream_form(d, live))
        if live is not None:
            status = getattr(live, "status", "live")
            problem = getattr(live, "problem", "") or ("" if getattr(live, "a_serving", None) is not None
                                                       else "server unknown")
            if status != "live":
                notes.append(f"live state '{status}': pre-match probability")
            elif problem:
                notes.append(f"live score unusable ({problem}): pre-match probability, not an in-play price")
                d["live_problem"] = problem
            else:
                xa, xb, va, vb = d["x_a"], d["x_b"], d["v_a"], d["v_b"]
                if self.live_form:
                    xa, va = d["x_a"] + d["form_a"], d["v_a_form"]
                    xb, vb = d["x_b"] + d["form_b"], d["v_b_form"]
                def f(a, b):
                    return points_rating.in_play_probability(
                        st, a, b, va, vb, q.best_of, q.final, live.score(), live.a_serving,
                        deciding_tb_points=getattr(live, "deciding_tb_points", 7))
                p_out = p_raw = f(xa, xb)
                d["p_sd"] = p_spread(f, xa, xb, va, vb) if 0 < p_out < 1 else 0.0
                in_play = True
                d.update(x_a_live=xa, x_b_live=xb, v_a_live=va, v_b_live=vb,
                         serve_a_live=_sig(xa), serve_b_live=_sig(xb), form=self.live_form)
                notes.append("in play: not calibrated")
        if self.explain:
            live_x = (d["x_a_live"], d["x_b_live"], d["v_a_live"], d["v_b_live"]) if in_play else None
            d["explain"] = explain_mod.explain(md, q, d, live if in_play else None, live_x)
        return Prediction(p_a=p_out, p_raw=p_raw, in_play=in_play, model=self.name, stale=stale,
                          notes=notes, detail=d)

    def stream_form(self, d: dict, live) -> dict:
        """Each serve stream's in-match form on the serve points in `live` (any status: a
        finished match's final stats give the model's last word on the day). Stream A = A
        serving = A's serve vs B's return; stream B likewise. form_* is the shift of that
        stream's serve-point logit (points_rating.in_match_form, prior N(x, v + form_var)),
        serve_*_form the serve-point probability after it; v_*_form the updated variance."""
        out = {}
        for s in ("a", "b"):
            won, played = (int(v) for v in (getattr(live, "serve_" + s, None) or (0, 0)))
            x, v = points_rating.in_match_form(d["x_" + s], d["v_" + s], won, played, self.form_var)
            out.update({f"form_{s}": x - d["x_" + s], f"v_{s}_form": v, f"serve_{s}_form": _sig(x),
                        f"form_points_{s}": [won, played]})
        out["form_var"] = self.form_var
        return out

    def set_scores(self, q: MatchQuery) -> dict:
        """Pre-match set-score distribution from the two serve-point probabilities (no
        shrink, not calibrated): {'2-0': p, ...} from A's side."""
        d = self.prematch(q)
        return markov.set_count_dist(d["serve_a"], d["serve_b"], q.best_of, q.final)


# ---------------------------------------------------------------------------- helpers

def live_from_score(score: str, points: str = "", a_serving: bool = True, *, serve_a=(0, 0),
                    serve_b=(0, 0), best_of: int = 3, final: str = "tb",
                    deciding_tb_points: int = 7) -> LiveState:
    """A typed score ('6-4 3-2', A's games first; points '30-15') as a live state for
    MatchQuery.live. serve_a / serve_b: (won, played) on serve so far this match."""
    sc = markov.parse_live(score, points, best_of, final)
    ls = LiveState(status="live", a_serving=a_serving, serve_a=tuple(serve_a), serve_b=tuple(serve_b),
                   raw=f"{score} {points}".strip(), **sc)
    ls.deciding_tb_points = deciding_tb_points       # read by PointsPredictor via getattr
    return ls


def run_params(con, run_id: int) -> tuple[str, points_rating.PointsParams]:
    row = con.execute("SELECT model_id, params_json FROM model_run WHERE run_id=?", (run_id,)).fetchone()
    if row is None:
        raise ValueError(f"run {run_id} not found in models.db")
    if not row[0].startswith("points_"):
        raise ValueError(f"run {run_id} is {row[0]}, not a points model")
    return row[0], points_rating.PointsParams(**json.loads(row[1]))


def model_from_db(con, run_id: int | None, model_id: str, params, *, as_of: str | None,
                  today: str, alias: str = "data", use_state: bool = True, state_dir: Path | None = None,
                  log=print) -> ModelState:
    """From the latest valid saved state (engine/state.py) plus the editions after it, or
    from a full replay when there is none."""
    if use_state and run_id is not None:
        snap = state_mod.latest_valid(con, run_id, params, before=as_of, state_dir=state_dir, alias=alias, log=log)
        if snap is not None:
            log(f"model: resuming from saved state {snap.cutoff} ({snap.n_matches:,} matches)")
            return model_from_snapshot(con, snap, run_id, model_id, as_of=as_of, today=today, alias=alias)
    m = matches.load_matches(con, alias)
    points_rating.attach(con, m, alias)              # also context.attach
    if as_of:
        m = matches.until(m, as_of)
    edmap = dict(con.execute(f"SELECT match_id, edition_id FROM {alias}.matches"))
    dob = {k: v for k, v in con.execute(f"SELECT player_id, dob FROM {alias}.player") if v}
    sql = f"SELECT edition_id, start_date, tournament_id, surface, best_of FROM {alias}.tournament_edition"
    eds = {e: (s, t, sf, bo) for e, s, t, sf, bo in con.execute(sql)
           if not as_of or s < as_of}
    return build_model(m, params, [edmap[i] for i in m.match_id], dob, eds, today=today,
                       run_id=run_id, model_id=model_id, as_of=as_of,
                       players_index=players_mod.build_index(con, today, alias))


def model_from_snapshot(con, snap, run_id: int, model_id: str, *, as_of: str | None, today: str,
                        alias: str = "data") -> ModelState:
    """The same ModelState build_model() gives for the full history, from a snapshot plus
    the replay of the editions after its cutoff (tests/test_state.py)."""
    w, erule_before, _keys = state_mod.resume(con, snap, before=as_of, alias=alias)
    params = w.res.state.params
    calib = state_mod.merge_calib(snap.calib, state_mod.calib_rows(w))
    block = calibrate.block_start(today)
    erule = {**erule_before, **w.erule}
    dob = {k: v for k, v in con.execute(f"SELECT player_id, dob FROM {alias}.player") if v}
    sql = f"SELECT edition_id, start_date, tournament_id, surface, best_of FROM {alias}.tournament_edition"
    eds = {e: (s, t, sf, bo) for e, s, t, sf, bo in con.execute(sql) if not as_of or s < as_of}
    return ModelState(run_id=run_id, model_id=model_id, params=params, state=w.res.state,
                      ctx_states=w.ctx if params.has_context() else {}, dob=dob,
                      alphas=state_mod.calib_alphas(calib, block), alpha_block=block, editions=eds,
                      edition_rule=erule, tournament_rule=tournament_rules(eds, erule),
                      last_start=w.m.start[-1] if len(w.m) else snap.last_start,
                      n_matches=snap.n_matches + len(w.m), as_of=as_of,
                      players=players_mod.build_index(con, today, alias))


def cache_key(con, run_id: int, params, as_of: str | None, today: str, alias: str = "data") -> str:
    """Changes with tennis.db's contents, the model code, the run's params, as_of, and the
    calibration quarter."""
    h = hashlib.sha256(CACHE_VERSION)
    h.update(repr(con.execute(
        f"SELECT COUNT(*), MAX(match_id), MAX(snapshot_id), TOTAL(winner_id), TOTAL(loser_id), "
        f"TOTAL(LENGTH(score)) FROM {alias}.matches").fetchone()).encode())
    h.update(repr(con.execute(
        f"SELECT COUNT(*), MAX(player_id), TOTAL(LENGTH(dob)) FROM {alias}.player").fetchone()).encode())
    h.update(repr(con.execute(
        f"SELECT COUNT(*), TOTAL(LENGTH(sha256)) FROM {alias}.ingest_snapshot").fetchone()).encode())
    h.update(repr(con.execute(
        f"SELECT COUNT(*), MAX(start_date), TOTAL(LENGTH(surface)) FROM {alias}.tournament_edition").fetchone()).encode())
    src = Path(__file__).resolve().parents[1]
    for f in CODE:                       # line endings normalized: a Windows-built cache loads on Linux
        h.update((src / f).read_bytes().replace(b"\r\n", b"\n"))
    h.update(json.dumps(params.as_dict(), sort_keys=True).encode())
    h.update(f"{run_id} {as_of} {calibrate.block_start(today)} {today[:4]}".encode())
    return h.hexdigest()


def _read_cache(path: Path, key: str, log) -> ModelState | None:
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            cached = pickle.load(f)
        if cached.get("key") == key:
            md = cached["model"]
            log(f"model: cached replay of run {md.run_id} ({md.n_matches:,} matches, latest edition {md.last_start})")
            return md
    except Exception as e:                       # corrupt / old format: rebuild
        log(f"model: cache unreadable ({e}); rebuilding")
    return None


def _write_cache(path: Path, key: str, model: ModelState, log) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        pickle.dump({"key": key, "model": model}, f, protocol=pickle.HIGHEST_PROTOCOL)
    for _ in range(10):                          # Windows: a scanner may hold the old file briefly
        try:
            tmp.replace(path)
            return
        except PermissionError:
            time.sleep(0.5)
    log(f"warning: could not replace {path} (file in use); not cached this time")
