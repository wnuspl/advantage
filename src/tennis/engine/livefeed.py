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
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

TOUR_LABEL = {"KXATPMATCH": "ATP", "KXWTAMATCH": "WTA",
              "KXATPCHALLENGERMATCH": "ATP Challenger", "KXWTACHALLENGERMATCH": "WTA 125"}

SNAP = ("status", "skip", "p_market", "ask_a", "ask_b", "score", "sets_a", "sets_b", "games_a", "games_b",
        "pts_a", "pts_b", "a_serving", "serve_a_won", "serve_a_played", "serve_b_won", "serve_b_played",
        "live_problem", "p_model", "p_raw", "in_play", "stale", "a_resolved", "b_resolved", "a_last_rated",
        "b_last_rated", "a_stale_days", "b_stale_days", "day", "surface", "surface_from", "best_of", "final")
DECISION = ("side", "p_use", "p_side", "ask", "contracts", "cost", "fees", "ev", "reason")


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
              history_points: int = 72, settled_hours: float = 48, n_alerts: int = 30) -> dict:
    now = now or datetime.now(timezone.utc)
    cols = {r[1] for r in con.execute("PRAGMA table_info(snapshot)")}
    has_sd = "p_sd" in cols
    w = con.execute("SELECT * FROM watch ORDER BY watch_id DESC LIMIT 1").fetchone()
    out: dict = {"now": _iso(now), "watch": None, "poll": None, "error": None,
                 "matches": [], "settled": [], "alerts": []}
    if w is not None:
        out["watch"] = {"predictor": w["predictor"], "started_at": w["started_at"], "every_s": w["every_s"],
                        "policy": json.loads(w["policy"]), "series": w["series"].split(",")}
    newest = con.execute("SELECT poll_id, as_of, error FROM poll ORDER BY poll_id DESC LIMIT 1").fetchone()
    if newest is not None and newest["error"]:
        out["error"] = {"as_of": newest["as_of"], "error": newest["error"][:300]}
    pid = latest_poll_id(con)
    if pid is None:
        return out
    p = con.execute("SELECT poll_id, as_of, done_at, n_events FROM poll WHERE poll_id=?", (pid,)).fetchone()
    out["poll"] = {"poll_id": pid, "as_of": p["as_of"], "done_at": p["done_at"], "n_events": p["n_events"],
                   "age_s": round((now - _parse(p["as_of"])).total_seconds())}

    sd = "s.p_sd" if has_sd else "NULL AS p_sd"
    rows = con.execute(
        f"""SELECT s.*, {sd}, e.series, e.tour, e.title, e.tournament, e.start, e.a_name, e.b_name,
                   e.a_ticker, e.b_ticker, e.first_seen,
                   d.side, d.p_use, d.p_side, d.ask, d.contracts, d.cost, d.fees, d.ev, d.reason
            FROM snapshot s JOIN event e USING (event)
            LEFT JOIN decision d ON d.poll_id = s.poll_id AND d.event = s.event
            WHERE s.poll_id = ?""", (pid,)).fetchall()

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
        m["decision"] = None if r["reason"] is None else {k: (_r(r[k]) if isinstance(r[k], float) else r[k])
                                                          for k in DECISION}
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
