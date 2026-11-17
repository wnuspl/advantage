"""What the live page shows (web/live.html via scripts/web_server.py /api/live): the watch's
latest poll from live.db, read-only. Nothing here talks to Kalshi or runs the model.

    con = connect(LIVE_DB)              # read-only
    view = live_view(con)               # dict, JSON-ready

The view:
  watch      the running watch (predictor, policy, poll interval)
  poll       the latest poll (as_of, age) and, if the newest poll failed, its error
  matches    each open match in the latest successful poll: event, players, tour label,
             status, live score, market (p_market, asks), model (p_model, p_sd), staleness,
             the policy's decision, and a short history of p_model / p_market
  settled    matches settled in the last `settled_hours`: winner, final score, the last
             pre-match model and market numbers, and whether the policy flagged a bet then
  alerts     the latest decision alerts
  sizing     the policy the decisions use: the watch's own, or the watch's with the viewer's
             bankroll / Kelly fraction / model weight (`overrides`), in which case every
             match is re-decided with engine.decide on the poll's recorded order books
             (the same code the watch runs; its alerts keep using its own policy)
"""
from __future__ import annotations

import json
import math
import re
import sqlite3
import zlib
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace as NS

from .decide import Policy, decide_all

TOUR_LABEL = {"KXATPMATCH": "ATP", "KXWTAMATCH": "WTA",
              "KXATPCHALLENGERMATCH": "ATP Challenger", "KXWTACHALLENGERMATCH": "WTA 125"}

SNAP = ("status", "skip", "p_market", "ask_a", "ask_b", "score", "sets_a", "sets_b", "games_a", "games_b",
        "pts_a", "pts_b", "a_serving", "serve_a_won", "serve_a_played", "serve_b_won", "serve_b_played",
        "live_problem", "p_model", "p_raw", "in_play", "stale", "a_resolved", "b_resolved", "a_last_rated",
        "b_last_rated", "a_stale_days", "b_stale_days", "day", "surface", "surface_from", "best_of", "final")
DECISION = ("side", "p_use", "p_side", "ask", "contracts", "cost", "fees", "ev", "reason")
OVERRIDABLE = {"bankroll": (1.0, 1e6), "kelly": (0.01, 1.0), "model_weight": (0.0, 1.0)}


def policy_with(watch_policy: dict, overrides: dict | None) -> Policy:
    """The watch's policy with the viewer's bankroll / kelly / model_weight (each clamped)."""
    known = {k: v for k, v in watch_policy.items() if k in Policy.__dataclass_fields__}
    for k, v in (overrides or {}).items():
        lo, hi = OVERRIDABLE[k]
        known[k] = min(max(float(v), lo), hi)
    return Policy(**known)


def redecide(rows, books: dict, pol: Policy, placeholder: bool = False) -> dict:
    """engine.decide on recorded snapshot rows (sqlite rows with the snapshot + event columns):
    event -> Decision. The rows carry what decide() reads from a live MarketSnapshot."""
    snaps = []
    for r in rows:
        priced = not r["skip"] and r["p_model"] is not None
        pred = NS(p_a=r["p_model"], stale=bool(r["stale"]), detail={"placeholder": True} if placeholder else {})
        ev = NS(event=r["event"], a=NS(ticker=r["a_ticker"], name=r["a_name"]),
                b=NS(ticker=r["b_ticker"], name=r["b_name"]))
        snaps.append(NS(event=ev, skip=r["skip"], prediction=pred if priced else None, status=r["status"],
                        p_market=r["p_market"], p_model=r["p_model"]))
    return {d.event: d for d in decide_all(snaps, books, pol)}


def connect(path: Path | str) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True, timeout=10,
                          check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def _iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).isoformat(timespec="seconds")


def _parse(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _thin(points: list, keep: int) -> list:
    """Evenly spaced subset of at most `keep` points, always keeping the last."""
    if len(points) <= keep:
        return points
    step = (len(points) - 1) / (keep - 1)
    return [points[round(i * step)] for i in range(keep)]


def latest_poll_id(con: sqlite3.Connection) -> int | None:
    r = con.execute("SELECT MAX(poll_id) FROM poll WHERE error IS NULL").fetchone()
    return r[0]


def live_view(con: sqlite3.Connection, now: datetime | None = None, *, history_hours: float = 6,
              history_points: int = 72, settled_hours: float = 48, n_alerts: int = 30,
              overrides: dict | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    cols = {r[1] for r in con.execute("PRAGMA table_info(snapshot)")}
    has_sd = "p_sd" in cols
    w = con.execute("SELECT * FROM watch ORDER BY watch_id DESC LIMIT 1").fetchone()
    out: dict = {"now": _iso(now), "watch": None, "poll": None, "error": None,
                 "matches": [], "settled": [], "alerts": [], "sizing": None}
    if w is not None:
        out["watch"] = {"predictor": w["predictor"], "started_at": w["started_at"], "every_s": w["every_s"],
                        "policy": json.loads(w["policy"]), "series": w["series"].split(",")}
        pol = policy_with(out["watch"]["policy"], overrides)
        out["sizing"] = {**asdict(pol), "custom": bool(overrides)}
    newest = con.execute("SELECT poll_id, as_of, error FROM poll ORDER BY poll_id DESC LIMIT 1").fetchone()
    if newest is not None and newest["error"]:
        out["error"] = {"as_of": newest["as_of"], "error": newest["error"][:300]}
    pid = latest_poll_id(con)
    if pid is None:
        return out
    p = con.execute("SELECT poll_id, as_of, done_at, n_events, books_sha FROM poll WHERE poll_id=?", (pid,)).fetchone()
    out["poll"] = {"poll_id": pid, "as_of": p["as_of"], "done_at": p["done_at"], "n_events": p["n_events"],
                   "age_s": round((now - _parse(p["as_of"])).total_seconds())}

    sd = "s.p_sd" if has_sd else "NULL AS p_sd"
    rows = con.execute(
        f"""SELECT s.*, {sd}, e.series, e.tour, e.title, e.tournament, e.start, e.a_name, e.b_name,
                   e.a_ticker, e.b_ticker, e.first_seen,
                   d.side, d.p_use, d.p_side, d.ask, d.contracts, d.cost, d.fees, d.ev, d.reason, d.fills
            FROM snapshot s JOIN event e USING (event)
            LEFT JOIN decision d ON d.poll_id = s.poll_id AND d.event = s.event
            WHERE s.poll_id = ?""", (pid,)).fetchall()

    mine = None
    if overrides and w is not None and rows:
        b = con.execute("SELECT zlib FROM blob WHERE sha256 = ?", (p["books_sha"],)).fetchone()
        books = json.loads(zlib.decompress(b[0]).decode("utf-8")) if b else {}
        mine = redecide(rows, books, pol, placeholder=w["predictor"].startswith("fixed"))

    since = _iso(now - timedelta(hours=history_hours))
    hist: dict[str, list] = {}
    if rows:
        events = [r["event"] for r in rows]
        q = ",".join("?" * len(events))
        for h in con.execute(
                f"""SELECT s.event, p.as_of, s.p_model, s.p_market, s.status FROM snapshot s
                    JOIN poll p USING (poll_id)
                    WHERE s.event IN ({q}) AND p.as_of >= ? ORDER BY s.poll_id""", (*events, since)):
            hist.setdefault(h["event"], []).append(
                [h["as_of"], _r(h["p_model"]), _r(h["p_market"]), h["status"] == "live"])

    for r in rows:
        m = {"event": r["event"], "series": r["series"], "tour": r["tour"],
             "tour_label": TOUR_LABEL.get(r["series"], r["series"]), "title": r["title"],
             "tournament": r["tournament"], "start": r["start"], "first_seen": r["first_seen"],
             "a": {"name": r["a_name"], "ticker": r["a_ticker"]},
             "b": {"name": r["b_name"], "ticker": r["b_ticker"]},
             "p_sd": _r(r["p_sd"]), "hist": _thin(hist.get(r["event"], []), history_points)}
        for k in SNAP:
            m[k] = _r(r[k]) if isinstance(r[k], float) else r[k]
        if mine is not None:
            d = mine[r["event"]]
            sz = d.sizing
            m["decision"] = {"side": d.side, "p_use": _r(d.p_use), "p_side": _r(d.p_side), "ask": _r(d.ask),
                             "contracts": sz.contracts, "cost": _r(sz.cost), "fees": _r(sz.fees),
                             "ev": _r(sz.ev), "reason": d.reason,
                             "fills": [{"price": f.price, "contracts": f.contracts, "route": f.route} for f in sz.fills]}
        else:
            m["decision"] = None if r["reason"] is None else {k: (_r(r[k]) if isinstance(r[k], float) else r[k])
                                                              for k in DECISION}
            if m["decision"] is not None:
                m["decision"]["fills"] = json.loads(r["fills"] or "[]")
        out["matches"].append(m)

    cut = _iso(now - timedelta(hours=settled_hours))
    for s in con.execute(
            """SELECT st.*, e.series, e.title, e.tournament, e.a_name, e.b_name, e.start FROM settlement st
               JOIN event e USING (event) WHERE st.seen_at >= ? ORDER BY st.seen_at DESC""", (cut,)).fetchall():
        pre = con.execute(
            f"""SELECT s.p_model, s.p_market, {sd}, d.side, d.ask, d.contracts, d.cost, d.ev, p.as_of
                FROM snapshot s JOIN poll p USING (poll_id)
                LEFT JOIN decision d ON d.poll_id = s.poll_id AND d.event = s.event
                WHERE s.event = ? AND s.status = 'not_started' ORDER BY s.poll_id DESC LIMIT 1""",
            (s["event"],)).fetchone()
        flagged = con.execute(
            """SELECT d.side, d.ask, d.contracts, d.cost, d.ev, p.as_of FROM decision d JOIN poll p USING (poll_id)
               WHERE d.event = ? AND d.contracts > 0 ORDER BY d.poll_id DESC LIMIT 1""", (s["event"],)).fetchone()
        out["settled"].append({
            "event": s["event"], "tour_label": TOUR_LABEL.get(s["series"], s["series"]), "title": s["title"],
            "tournament": s["tournament"], "start": s["start"], "seen_at": s["seen_at"],
            "a": {"name": s["a_name"]}, "b": {"name": s["b_name"]}, "winner": s["winner"],
            "final_score": s["final_score"], "match_status": s["match_status"],
            "pre": None if pre is None else {"p_model": _r(pre["p_model"]), "p_market": _r(pre["p_market"]),
                                             "p_sd": _r(pre["p_sd"]), "as_of": pre["as_of"]},
            "bet": None if flagged is None else {"side": flagged["side"], "ask": _r(flagged["ask"]),
                                                 "contracts": flagged["contracts"], "cost": _r(flagged["cost"]),
                                                 "ev": _r(flagged["ev"]), "as_of": flagged["as_of"]}})

    for a in con.execute(
            """SELECT a.alert_id, a.event, a.kind, a.title, a.message, a.status, p.as_of FROM alert a
               JOIN poll p USING (poll_id) ORDER BY a.alert_id DESC LIMIT ?""", (n_alerts,)):
        out["alerts"].append(dict(a))
    return out


def _r(x, nd: int = 4):
    return None if x is None else round(x, nd)


def _round(v, nd: int = 4):
    if isinstance(v, float):
        return round(v, nd)
    if isinstance(v, dict):
        return {k: _round(x, nd) for k, x in v.items()}
    return v


# ---------------------------------------------------------------------------- one match

EVENT_RE = re.compile(r"^[A-Z0-9][A-Z0-9._-]{2,79}$")      # Kalshi event tickers: KXATPMATCH-26NOV16ALCTIA


def valid_event(event: str | None) -> bool:
    return bool(event) and EVENT_RE.match(event) is not None


def set_scores(score: str | None, n: int = 99) -> list[tuple[int, int]]:
    """The leading 'g-g' tokens of a score string ('6-7 6-2 0-0 *A serving', or a settlement's
    '6-7 6-2 5-7'), at most n: with n = sets played, the completed sets, A first."""
    out = []
    for tok in (score or "").split():
        m = re.fullmatch(r"(\d+)-(\d+)", tok)
        if m is None:
            break
        out.append((int(m[1]), int(m[2])))
    return out[:n]


def _tb_set(ga: int, gb: int) -> bool:
    """A completed set that ended in a tiebreak (7-6) or was a match tiebreak (1-0, 10-8)."""
    return (max(ga, gb) == 7 and min(ga, gb) == 6) or max(ga, gb) == 1 or min(ga, gb) >= 8


def score_steps(prev: dict, cur: dict) -> list[dict]:
    """Set ends and breaks of serve between two consecutive recorded scores (dicts with
    sets_a/b, games_a/b, a_serving, score; `cur` may carry `done`, the completed sets, instead
    of a score string). Games alternate servers, so over k games starting with prev's server
    the breaks by a side are at least (games it won) - (games it served): exact when k = 1,
    a lower bound (`exact` False) when polls missed games. Nothing is guessed when the step
    is unreadable (score went back, a set skipped, a tiebreak among missed games, a serve
    order that does not add up); tiebreak games are never breaks."""
    ps, cs = prev["sets_a"] + prev["sets_b"], cur["sets_a"] + cur["sets_b"]
    if cs < ps or cs > ps + 1:
        return []
    marks = []
    tb = False
    if cs == ps:
        na, nb = cur["games_a"] - prev["games_a"], cur["games_b"] - prev["games_b"]
        if na < 0 or nb < 0:
            return []
    else:
        done = cur.get("done") or set_scores(cur.get("score"), cs)
        if len(done) < cs:
            return []
        fa, fb = done[cs - 1]
        if fa < prev["games_a"] or fb < prev["games_b"]:
            return []
        side = "a" if cur["sets_a"] > prev["sets_a"] else "b"
        marks.append({"kind": "set", "side": side, "set": cs, "games": [fa, fb]})
        tb = _tb_set(fa, fb)
        na = fa - prev["games_a"] + cur["games_a"] - (1 if tb and side == "a" else 0)
        nb = fb - prev["games_b"] + cur["games_b"] - (1 if tb and side == "b" else 0)
    k = na + nb
    server = prev.get("a_serving")
    if k <= 0 or server is None or tb or "tiebreak" in (prev.get("score") or ""):
        return marks
    if cur.get("a_serving") is not None and bool(cur["a_serving"]) != (bool(server) != (k % 2 == 1)):
        return marks
    a_srv = (k + 1) // 2 if server else k // 2
    for side, won, own in (("a", na, a_srv), ("b", nb, k - a_srv)):
        if won - own > 0:
            marks.append({"kind": "break", "side": side, "n": won - own, "exact": k == 1})
    return marks


def match_marks(polls: list[dict], final_score: str | None = None) -> list[dict]:
    """Breaks of serve and set ends over a match's polls (oldest first; those without a
    score are passed over). final_score (the settlement's, A first) adds the last set end
    when the feed stopped before it. Each mark: {i (index into polls), as_of, kind, side, ...}."""
    out, prev = [], None
    for i, p in enumerate(polls):
        if p.get("sets_a") is None or p.get("games_a") is None:
            continue
        if prev is not None:
            got = score_steps(prev, p)
            if not got and (p["sets_a"] + p["sets_b"], p["games_a"] + p["games_b"]) < \
                    (prev["sets_a"] + prev["sets_b"], prev["games_a"] + prev["games_b"]):
                continue            # a feed glitch went backwards: keep the earlier score as the base
            out += [{"i": i, "as_of": p["as_of"], **m} for m in got]
        prev = p
    if final_score and prev is not None:
        done = set_scores(final_score)
        sa = sum(1 for a, b in done if a > b)
        if len(done) == prev["sets_a"] + prev["sets_b"] + 1:
            last = len(polls) - 1
            cur = {"sets_a": sa, "sets_b": len(done) - sa, "games_a": 0, "games_b": 0, "a_serving": None,
                   "done": done}
            out += [{"i": last, "as_of": polls[last]["as_of"], **m} for m in score_steps(prev, cur)]
    return out


MATCH_SNAP = ("status", "p_model", "p_market", "score", "sets_a", "sets_b", "games_a", "games_b", "pts_a",
              "pts_b", "a_serving", "serve_a_won", "serve_a_played", "serve_b_won", "serve_b_played", "in_play")


def _settled_live(con: sqlite3.Connection, st) -> dict | None:
    """The final live details kept in the settlement's blob (None for old or bare rows)."""
    if not st["raw_sha"]:
        return None
    b = con.execute("SELECT zlib FROM blob WHERE sha256 = ?", (st["raw_sha"],)).fetchone()
    blob = json.loads(zlib.decompress(b[0]).decode("utf-8")) if b else None
    return blob.get("live") if isinstance(blob, dict) else None


def _final_form(con: sqlite3.Connection, event: str) -> dict | None:
    if not con.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'final_form'").fetchone():
        return None
    r = con.execute("SELECT * FROM final_form WHERE event = ?", (event,)).fetchone()
    return None if r is None else dict(r)


def match_view(con: sqlite3.Connection, event: str, now: datetime | None = None) -> dict | None:
    """Everything the live page's match panel shows for one event, or None if live.db has
    never seen it: every poll that listed it (not thinned), the break / set-end marks, the
    pre-match block (each player's ratings going into the match as the predictor recorded
    them, the last pre-match model and market), the settlement, if any, and for a settled
    match with final stats `after` (after_match: final stats vs the pre-match expectation, the
    model's final read of each serve stream)."""
    now = now or datetime.now(timezone.utc)
    e = con.execute("SELECT * FROM event WHERE event = ?", (event,)).fetchone()
    if e is None:
        return None
    cols = {r[1] for r in con.execute("PRAGMA table_info(snapshot)")}
    sd = "s.p_sd" if "p_sd" in cols else "NULL AS p_sd"
    polls = []
    for r in con.execute(
            f"""SELECT p.as_of, {sd}, {', '.join('s.' + c for c in MATCH_SNAP)} FROM snapshot s
                JOIN poll p USING (poll_id) WHERE s.event = ? ORDER BY s.poll_id""", (event,)):
        polls.append({k: _round(r[k]) for k in r.keys()})
    st = con.execute("SELECT * FROM settlement WHERE event = ?", (event,)).fetchone()
    started = next((i for i, p in enumerate(polls) if p["status"] in ("live", "interrupted")), None)
    upto = len(polls) if started is None else started
    last_pre = next((p for p in reversed(polls[:upto]) if p["status"] == "not_started"), None)
    has_pm = con.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'prematch'").fetchone()
    pm = con.execute("SELECT * FROM prematch WHERE event = ?", (event,)).fetchone() if has_pm else None
    pre = {k: (last_pre or {}).get(k) for k in ("p_model", "p_market", "p_sd", "as_of")}
    pre["ratings"] = None
    if pm is not None:
        det = _round(json.loads(pm["detail"]))
        pre["ratings"] = {"model": pm["model"], "taken_at": pm["status"], **det}
        if pre["p_sd"] is None:
            pre["p_sd"] = det.get("p_sd_pre")
    after = None
    if st is not None:
        after = after_match(_settled_live(con, st), e["a_comp"], e["b_comp"],
                            json.loads(pm["detail"]) if pm is not None else None, _final_form(con, event))
        if after is not None:
            after = _round(after)
    marks = []
    if started is not None:
        marks = [{**m, "i": m["i"] + started}
                 for m in match_marks(polls[started:], st["final_score"] if st else None)]
    return {"now": _iso(now), "event": event, "series": e["series"],
            "tour_label": TOUR_LABEL.get(e["series"], e["series"]), "title": e["title"],
            "tournament": e["tournament"], "start": e["start"], "first_seen": e["first_seen"],
            "last_seen": e["last_seen"], "a": {"name": e["a_name"]}, "b": {"name": e["b_name"]},
            "status": "settled" if st else (polls[-1]["status"] if polls else None),
            "started_at": None if started is None else polls[started]["as_of"],
            "polls": polls, "marks": marks, "pre": pre, "after": after,
            "settlement": None if st is None else {
                "winner": st["winner"], "final_score": st["final_score"], "match_status": st["match_status"],
                "seen_at": st["seen_at"]}}


# ---------------------------------------------------------------------------- after the match

NOISE_Z = 2.0           # a gap beyond NOISE_Z SDs is "clearly" above / below expectation


def _sig(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def final_stats(details: dict | None, a_comp: str, b_comp: str) -> dict | None:
    """Kalshi's final per-player statistics (Sportradar `competitor{1,2}_statistics`) mapped
    to A / B by competitor id, never by order. None when the ids do not match the event or
    nobody served a point (voids, walkovers)."""
    d = details or {}
    side = {d.get("competitor1_id"): "1", d.get("competitor2_id"): "2"}
    if a_comp not in side or b_comp not in side or a_comp == b_comp:
        return None
    raw = {s: d.get(f"competitor{side[c]}_statistics") or {} for s, c in (("a", a_comp), ("b", b_comp))}
    g = lambda s, k: int(raw[s].get(k) or 0)                         # noqa: E731
    out = {}
    for s, o in (("a", "b"), ("b", "a")):
        won, lost = g(s, "service_points_won"), g(s, "service_points_lost")
        n = won + lost
        first_in = g(s, "first_serve_successful")
        held, broken = g(s, "service_games_won"), g(o, "breakpoints_won")
        out[s] = {"serve_won": won, "serve_played": n,
                  "first_in": first_in, "first_won": g(s, "first_serve_points_won"),
                  # second-serve points = serve points not started with a first serve in
                  # (Kalshi's second_serve_successful does not count them consistently)
                  "second_played": max(n - first_in, 0), "second_won": g(s, "second_serve_points_won"),
                  "aces": g(s, "aces"), "double_faults": g(s, "double_faults"),
                  "bp_won": g(s, "breakpoints_won"), "bp_total": g(s, "total_breakpoints"),
                  "service_games": held + broken, "service_games_won": held,
                  "tiebreaks_won": g(s, "tiebreaks_won"), "points_won": g(s, "points_won")}
    if out["a"]["serve_played"] <= 0 or out["b"]["serve_played"] <= 0:
        return None
    return out


def versus(won: int, n: int, expected: float, v: float | None, z: float = NOISE_Z) -> dict:
    """Points won out of n against a pre-match expected rate. The spread of the gap combines
    the binomial spread over n points, p(1-p)/n, and the rating's own uncertainty carried to
    the rate by the delta method, (p(1-p))^2 v with v the serve-point logit's variance. The
    label: 'above' / 'below' when the gap is beyond z SDs, else 'within' (noise)."""
    p = expected
    var_b = p * (1 - p) / n
    var_r = (p * (1 - p)) ** 2 * v if v else 0.0
    sd = math.sqrt(var_b + var_r)
    rate = won / n
    gap = rate - p
    label = "above" if gap > z * sd else "below" if gap < -z * sd else "within"
    return {"won": won, "played": n, "rate": rate, "expected": p, "gap": gap, "pts": won - n * p,
            "sd": sd, "sd_binom": math.sqrt(var_b), "sd_rating": math.sqrt(var_r), "z": gap / sd if sd else None,
            "label": label}


def after_match(details: dict | None, a_comp: str, b_comp: str, pre: dict | None = None,
                form: dict | None = None, z: float = NOISE_Z) -> dict | None:
    """The match panel's "After the match" block from the settlement's final Kalshi stats.
    pre: the prematch detail (serve_a / serve_b expected serve-point rates, v_a / v_b their
    logit variances), or None (settled before prematch was recorded: actuals only).
    form: the final_form row, or None. Serve and return of one stream are the same points:
    A's return gap is minus B's serve gap, with the same spread."""
    st = final_stats(details, a_comp, b_comp)
    if st is None:
        return None
    out = {"stats": st, "serve": None, "ret": None, "form": None, "z": z}
    exp = {}
    if pre:
        for s in ("a", "b"):
            p = pre.get("serve_" + s)
            if p is None and pre.get("x_" + s) is not None:
                p = _sig(pre["x_" + s])
            exp[s] = p
    if exp and all(exp.get(s) is not None for s in ("a", "b")):
        out["serve"], out["ret"] = {}, {}
        for s, o in (("a", "b"), ("b", "a")):
            out["serve"][s] = versus(st[s]["serve_won"], st[s]["serve_played"], exp[s], pre.get("v_" + s), z)
            # s's return points = o's serve points s won
            out["ret"][s] = versus(st[o]["serve_played"] - st[o]["serve_won"], st[o]["serve_played"],
                                   1 - exp[o], pre.get("v_" + o), z)
    if form:
        out["form"] = {s: {"form": form["form_" + s], "rate": form[f"serve_{s}_form"],
                           "pre": None if form.get("x_" + s) is None else _sig(form["x_" + s]),
                           "won": form[f"serve_{s}_won"], "played": form[f"serve_{s}_played"]}
                       for s in ("a", "b")}
        out["form"]["model"], out["form"]["form_var"] = form["model"], form.get("form_var")
    return out
