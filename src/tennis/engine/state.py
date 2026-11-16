"""Saved rating state at monthly cutoffs, and resuming the replay from it (milestone A5).

A snapshot at cutoff C holds everything the replay knows after every edition that started
before C: the PointsState, context bookkeeping, the tournament deciding-set rule history,
edition rules, and the tour-level predictions of the last 11 years (for calibration).
Resuming = load only editions starting on or after C from tennis.db, attach them with the
saved rule history and context states, and run() from the saved state. Because a cutoff
never splits an edition, the result equals a full replay (tests/test_state.py).

tennis.db is rebuilt on every data update and its surrogate IDs change, so snapshots store
stable keys only: players by their seed xref key, editions by theirs, tournaments by
(tour, name). They are translated back to the current IDs on load.

Validity (a snapshot is used only while the data before its cutoff is unchanged):
  static  hash of data/overrides.csv, the raw-file hashes of every non-AcesAPI source, and
          player birth dates: old history only changes when one of these does;
  recent  stable-key hash of every match (players, score, status, format, stats) of
          editions starting in the 400 days before the cutoff: covers late results, which
          only AcesAPI adds (from 2026-05);
  totals  match count and total score length before the cutoff (cheap backstop);
  code    the model code and the run's params.
The latest valid snapshot is used; a stale one is skipped, never patched.

Files: models/engine/state/run<id>/<cutoff>.pkl (gitignored).
"""
from __future__ import annotations

import copy
import hashlib
import json
import pickle
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

from .. import calibrate, context, matches, models_db, points_rating

STATE_DIR = models_db.ROOT / "models" / "engine" / "state"
RECENT_DAYS = 400
CALIB_YEARS = 11
CODE = ("points_rating.py", "context.py", "markov.py", "matches.py", "engine/state.py")
VERSION = "state-v1"


@dataclass
class Snapshot:
    version: str
    run_id: int
    params: dict
    cutoff: str                   # state after every edition starting before this date
    fingerprint: dict
    n_matches: int
    last_start: str | None
    state: dict                   # PointsState fields, keyed by stable player keys
    ctx: dict                     # player key -> context._P fields (edition as stable key)
    rule_last: dict               # "tour|tournament name" -> rule
    edition_rule: dict            # edition key -> rule
    calib: dict = field(default_factory=dict)   # tour-level scored matches: start/tour/best_of/p


# ---------------------------------------------------------------------------- keys

def _keys(con, table: str, col: str, alias: str = "data") -> dict:
    return dict(con.execute(
        f"""SELECT {col}, MIN(source || ':' || source_key) FROM {alias}.{table}
            WHERE method = 'seed' GROUP BY {col}"""))


def key_maps(con, alias: str = "data") -> dict:
    """{'player': id->key, 'edition': id->key, 'tournament': id->key} for the current db."""
    return {"player": _keys(con, "player_xref", "player_id", alias),
            "edition": _keys(con, "edition_xref", "edition_id", alias),
            "tournament": {tid: f"{t}|{nm}" for tid, t, nm in con.execute(
                f"SELECT tournament_id, tour, canonical_name FROM {alias}.tournament")}}


# ---------------------------------------------------------------------------- fingerprint

def code_hash(params) -> str:
    h = hashlib.sha256(VERSION.encode())
    src = Path(__file__).resolve().parents[1]
    for f in CODE:
        h.update((src / f).read_bytes())
    h.update(json.dumps(params.as_dict(), sort_keys=True).encode())
    return h.hexdigest()


def fingerprint(con, cutoff: str, params, alias: str = "data",
                overrides: Path = models_db.ROOT / "data" / "overrides.csv") -> dict:
    h = hashlib.sha256()
    h.update(overrides.read_bytes() if overrides.exists() else b"")
    for row in con.execute(f"""SELECT source, file_relpath, sha256 FROM {alias}.ingest_snapshot
                               WHERE source NOT LIKE 'acesapi%' ORDER BY source, file_relpath, sha256"""):
        h.update(repr(row).encode())
    h.update(repr(con.execute(f"SELECT COUNT(*), TOTAL(LENGTH(dob)), COUNT(dob) FROM {alias}.player").fetchone()).encode())
    static = h.hexdigest()

    since = (date.fromisoformat(cutoff) - timedelta(days=RECENT_DAYS)).isoformat()
    h = hashlib.sha256()
    rows = []
    for row in con.execute(
            f"""SELECT (SELECT MIN(source || ':' || source_key) FROM {alias}.match_xref x
                        WHERE x.match_id = m.match_id AND x.method = 'seed') mk,
                       (SELECT MIN(source || ':' || source_key) FROM {alias}.player_xref x
                        WHERE x.player_id = m.winner_id AND x.method = 'seed'),
                       (SELECT MIN(source || ':' || source_key) FROM {alias}.player_xref x
                        WHERE x.player_id = m.loser_id AND x.method = 'seed'),
                       m.score, m.status, m.best_of, m.round_order, e.start_date, e.surface,
                       (SELECT MIN(source || ':' || source_key) FROM {alias}.edition_xref x
                        WHERE x.edition_id = e.edition_id AND x.method = 'seed'),
                       (SELECT GROUP_CONCAT(is_winner || ':' || serve_points || '/' || first_won || '/' ||
                                            second_won || '/' || aces, ' ')
                        FROM {alias}.match_player_stats s WHERE s.match_id = m.match_id)
                FROM {alias}.tournament_edition e JOIN {alias}.matches m USING (edition_id)
                WHERE e.start_date >= ? AND e.start_date < ?""", (since, cutoff)):
        rows.append(repr(row))
    for r in sorted(rows):
        h.update(r.encode())
    recent = h.hexdigest()
    totals = list(con.execute(
        f"""SELECT COUNT(*), TOTAL(LENGTH(m.score)), TOTAL(m.best_of) FROM {alias}.matches m
            JOIN {alias}.tournament_edition e USING (edition_id) WHERE e.start_date < ?""", (cutoff,)).fetchone())
    return {"static": static, "recent": recent, "totals": totals, "code": code_hash(params)}


# ---------------------------------------------------------------------------- translate

_STATE_PLAYER = ("serve", "ret", "vs", "vr", "last", "count")
_STATE_SURF = ("sserve", "sret", "svs", "svr", "slast", "scount")
_P_FIELDS = ("ed", "start", "n", "prev_start", "eds", "since_return")


def to_stable(st: points_rating.PointsState, ctx: dict, rule_last: dict, erule: dict, keys: dict) -> tuple:
    pk, ek, tk = keys["player"], keys["edition"], keys["tournament"]
    state = {f: {pk[p]: v for p, v in getattr(st, f).items()} for f in _STATE_PLAYER}
    state.update({f: {(pk[p], s): v for (p, s), v in getattr(st, f).items()} for f in _STATE_SURF})
    state["base"] = dict(st.base)
    cx = {}
    for p, s in ctx.items():
        d = {f: copy.copy(getattr(s, f)) for f in _P_FIELDS}
        d["ed"] = ek.get(d["ed"]) if d["ed"] is not None else None
        cx[pk[p]] = d
    return state, cx, {tk[t]: r for t, r in rule_last.items()}, {ek[e]: r for e, r in erule.items()}


def from_stable(snap: Snapshot, keys: dict) -> tuple:
    """(PointsState, ctx states, rule_last, edition_rule) with current IDs. Keys missing
    from the current db raise KeyError: the fingerprint should have rejected the snapshot."""
    pid = {k: i for i, k in keys["player"].items()}
    eid = {k: i for i, k in keys["edition"].items()}
    tid = {k: i for i, k in keys["tournament"].items()}
    params = points_rating.PointsParams(**snap.params)
    f = {n: {pid[k]: v for k, v in snap.state[n].items()} for n in _STATE_PLAYER}
    f.update({n: {(pid[k], s): v for (k, s), v in snap.state[n].items()} for n in _STATE_SURF})
    st = points_rating.PointsState(params, f["serve"], f["ret"], f["vs"], f["vr"], f["last"], f["count"],
                                   f["sserve"], f["sret"], f["svs"], f["svr"], f["slast"], f["scount"],
                                   dict(snap.state["base"]))
    ctx = {}
    for k, d in snap.ctx.items():
        p = context._P()
        for name in _P_FIELDS:
            setattr(p, name, copy.copy(d[name]))
        p.ed = eid[p.ed] if p.ed is not None else None
        ctx[pid[k]] = p
    return st, ctx, {tid[k]: r for k, r in snap.rule_last.items()}, {eid[k]: r for k, r in snap.edition_rule.items()}


# ---------------------------------------------------------------------------- replay windows

@dataclass
class Window:
    """The replay of one window of editions [since, before), resumed from the state before it."""
    m: matches.Matches
    res: points_rating.PointsResult
    edition_of: list
    ctx: dict                     # context states after the window
    rule_last: dict               # after the window
    erule: dict                   # edition -> rule, this window's editions


def replay_window(con, params, since: str | None, before: str | None, state=None, ctx=None,
                  rule_last=None, alias: str = "data") -> Window:
    m = matches.load_matches(con, alias, since, before)
    rl = dict(rule_last or {})
    points_rating.attach(con, m, alias, since=since, before=before, rule_last=rl, ctx_init=ctx)
    res = points_rating.run(m, params, start_state=state)
    cond, args = matches.window_sql(since, before)
    edmap = dict(con.execute(f"""SELECT m.match_id, m.edition_id FROM {alias}.matches m
                                 JOIN {alias}.tournament_edition e USING (edition_id) WHERE {cond}""", args))
    edition_of = [edmap[i] for i in m.match_id]
    new_ctx = context.player_states(m, edition_of, init=ctx) if params.has_context() else {}
    fmt = m.extra["points"].fmt
    erule = {edition_of[i]: points_rating.FORMATS[fmt[i]][1] for i in range(len(m))}
    return Window(m, res, edition_of, new_ctx, rl, erule)


def calib_rows(w: Window, since: str | None = None) -> dict:
    """Tour-level scored matches of a window (for calibration alphas), columnar."""
    m, fmt = w.m, w.m.extra["points"].fmt
    keep = [i for i in range(len(m)) if m.tour_level[i] and m.status[i] in matches.SCORED_STATUS
            and (since is None or m.start[i] >= since)]
    return {"start": [m.start[i] for i in keep], "tour": [m.tour[i] for i in keep],
            "best_of": [points_rating.FORMATS[fmt[i]][0] for i in keep],
            "p": [w.res.p_winner[i] for i in keep]}


def merge_calib(a: dict, b: dict, since: str | None = None) -> dict:
    out = {k: list(a.get(k, [])) + list(b.get(k, [])) for k in ("start", "tour", "best_of", "p")}
    if since:
        keep = [j for j, s in enumerate(out["start"]) if s >= since]
        out = {k: [v[j] for j in keep] for k, v in out.items()}
    return out


def calib_alphas(calib: dict, block: str) -> dict:
    """calibrate.alphas_at() on the stored tour-level rows (all scored, all tour level)."""
    m = matches.empty_matches()
    n = len(calib["start"])
    m.start.extend(calib["start"]); m.tour.extend(calib["tour"])
    m.status.extend(["completed"] * n); m.tour_level.extend(b"\x01" * n)
    m.match_id.extend(range(n)); m.winner.extend([0] * n); m.loser.extend([0] * n); m.surface.extend([None] * n)
    return calibrate.alphas_at(m, calib["p"], block, best_of=calib["best_of"])


# ---------------------------------------------------------------------------- build / load

def month_starts(today: str, n: int) -> list[str]:
    d = date.fromisoformat(today).replace(day=1)
    out = []
    for _ in range(n):
        out.append(d.isoformat())
        d = (d - timedelta(days=1)).replace(day=1)
    return sorted(out)


def build(con, run_id: int, params, cutoffs: list[str], *, out_dir: Path | None = None, alias: str = "data",
          log=print) -> list[Path]:
    """Replay the history window by window and save a snapshot at each cutoff (sorted)."""
    out_dir = Path(out_dir or STATE_DIR / f"run{run_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = key_maps(con, alias)
    state = ctx = None
    rule_last: dict = {}
    erule: dict = {}
    calib: dict = {}
    n = 0
    last_start = None
    paths = []
    prev = None
    for c in sorted(cutoffs):
        t = time.perf_counter()
        w = replay_window(con, params, prev, c, state, ctx, rule_last, alias)
        state, ctx, rule_last = w.res.state, w.ctx, w.rule_last
        erule.update(w.erule)
        keep_from = calibrate.add_years(c, -CALIB_YEARS)
        calib = merge_calib(calib, calib_rows(w, keep_from), keep_from)
        n += len(w.m)
        last_start = w.m.start[-1] if len(w.m) else last_start
        st, cx, rl, er = to_stable(state, ctx, rule_last, erule, keys)
        snap = Snapshot(VERSION, run_id, params.as_dict(), c, fingerprint(con, c, params, alias), n, last_start,
                        st, cx, rl, er, calib)
        path = out_dir / f"{c}.pkl"
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps(snap, protocol=pickle.HIGHEST_PROTOCOL))
        tmp.replace(path)
        paths.append(path)
        log(f"snapshot {c}: {n:,} matches, window {len(w.m):,} [{time.perf_counter() - t:.0f}s]")
        prev = c
    return paths


def snapshots(run_id: int, state_dir: Path | None = None) -> list[Path]:
    d = Path(state_dir or STATE_DIR / f"run{run_id}")
    return sorted(d.glob("*.pkl")) if d.exists() else []


def latest_valid(con, run_id: int, params, *, before: str | None = None, state_dir: Path | None = None,
                 alias: str = "data", log=print) -> Snapshot | None:
    """The newest snapshot (cutoff <= `before` if given) whose fingerprint matches the db."""
    for path in reversed(snapshots(run_id, state_dir)):
        cutoff = path.stem
        if before and cutoff > before:
            continue
        try:
            snap = pickle.loads(path.read_bytes())
        except Exception as e:                                   # noqa: BLE001
            log(f"state: {path.name} unreadable ({e})")
            continue
        if snap.version != VERSION or snap.params != params.as_dict():
            continue
        fp = fingerprint(con, cutoff, params, alias)
        bad = [k for k in fp if fp[k] != snap.fingerprint.get(k)]
        if bad:
            log(f"state: snapshot {cutoff} stale ({', '.join(bad)} changed)")
            continue
        return snap
    return None


def resume(con, snap: Snapshot, *, before: str | None = None, alias: str = "data"):
    """(Window after the snapshot, PointsState before it, ctx before it, current keys) - the
    replay of editions from the snapshot's cutoff (up to `before`)."""
    keys = key_maps(con, alias)
    st, ctx, rule_last, erule = from_stable(snap, keys)
    w = replay_window(con, st.params, snap.cutoff, before, st, ctx, rule_last, alias)
    return w, erule, keys
