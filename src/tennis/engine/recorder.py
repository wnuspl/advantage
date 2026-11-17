"""live.db: the Kalshi watch's append-only record (track B2, docs/PLAN.md; schema in
live_schema.sql).

    rec = Recorder()                                   # creates / opens live.db
    wid = rec.start_watch(predictor.name, policy, series, every_s=60)
    pid = rec.record_poll(wid, pull, rows, decisions)  # rows: live.snapshots, decisions: decide.decide_all
    pull = rec.load_poll(pid)                          # the same Pull back: reprice / re-decide (B3b)
    rec.at("KXATPMATCH-26NOV16ALCTIA", "2026-11-16T12:00:00+00:00")   # state at time t

Settlement: `settlement_from()` reads GET /events/<ticker> (markets with `status`
determined/finalized and `result` yes/no) plus the last live details (`match_status`,
final set scores, mapped to A/B by competitor id). Nothing is settled until Kalshi has
determined both markets. `record_final_form()` then asks the predictor for each serve
stream's final in-match form on the settlement's final stats (table `final_form`).
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import zlib
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from .. import kalshi
from .decide import Decision, Policy
from .live import MarketSnapshot, Pull
from .predictor import MatchQuery

ROOT = Path(__file__).resolve().parents[3]
LIVE_DB = ROOT / "live.db"
LIVE_SCHEMA = ROOT / "live_schema.sql"
SETTLED = ("determined", "finalized", "settled")
GIVE_UP_DAYS = 15        # Kalshi: a postponed match's market closes within two weeks


# Columns added after live.db was first created: (table, column, type). CREATE TABLE IF NOT
# EXISTS leaves an existing table alone, so these are added in place (additive only).
ADDED_COLUMNS = (("snapshot", "p_sd", "REAL"),)


def migrate(con: sqlite3.Connection) -> None:
    for table, col, typ in ADDED_COLUMNS:
        have = {r[1] for r in con.execute(f"PRAGMA table_info({table})")}
        if col not in have:
            con.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")
    con.commit()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def settlement_from(ev: dict, event_json: dict, details: dict | None) -> dict | None:
    """ev: an `event` row (dict). event_json: GET /events/<ticker> response. details: the
    match's last live details, or None. Returns the settlement row, or None while Kalshi
    has not determined both markets."""
    mk = {m.get("ticker"): m for m in event_json.get("markets") or []}
    ma, mb = mk.get(ev["a_ticker"]), mk.get(ev["b_ticker"])
    if ma is None or mb is None or ma.get("status") not in SETTLED or mb.get("status") not in SETTLED:
        return None
    ra, rb = ma.get("result") or "", mb.get("result") or ""
    winner = "a" if ra == "yes" and rb != "yes" else "b" if rb == "yes" and ra != "yes" else None

    def val(m):
        v = m.get("settlement_value_dollars")
        return None if v in (None, "") else float(v)
    match_status = final = None
    d = details or {}
    side = {d.get("competitor1_id"): "1", d.get("competitor2_id"): "2"}
    if ev["a_comp"] in side and ev["b_comp"] in side and ev["a_comp"] != ev["b_comp"]:
        match_status = d.get("match_status") or d.get("status")
        xa = d.get(f"competitor{side[ev['a_comp']]}_round_scores") or []
        xb = d.get(f"competitor{side[ev['b_comp']]}_round_scores") or []
        final = " ".join(f"{int(x.get('score') or 0)}-{int(y.get('score') or 0)}" for x, y in zip(xa, xb)) or None
    return {"winner": winner, "result_a": ra, "result_b": rb, "value_a": val(ma), "value_b": val(mb),
            "market_status": ma.get("status"), "match_status": match_status, "final_score": final}


class Recorder:
    def __init__(self, path: Path | str = LIVE_DB):
        self.path = Path(path) if str(path) != ":memory:" else path
        self.con = sqlite3.connect(str(path), timeout=30)
        self.con.row_factory = sqlite3.Row
        if str(path) != ":memory:":
            self.con.execute("PRAGMA journal_mode = WAL")
        self.con.executescript(LIVE_SCHEMA.read_text(encoding="utf-8"))
        migrate(self.con)
        self.con.execute("PRAGMA foreign_keys = ON")

    def close(self) -> None:
        self.con.close()

    # ---------------------------------------------------------------- blobs
    def put_blob(self, obj) -> str:
        raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")     # key order kept: replay = what was polled
        sha = hashlib.sha256(raw).hexdigest()
        self.con.execute("INSERT OR IGNORE INTO blob (sha256, bytes, zlib) VALUES (?,?,?)",
                         (sha, len(raw), zlib.compress(raw, 6)))
        return sha

    def get_blob(self, sha: str | None):
        if sha is None:
            return None
        r = self.con.execute("SELECT zlib FROM blob WHERE sha256 = ?", (sha,)).fetchone()
        return None if r is None else json.loads(zlib.decompress(r[0]).decode("utf-8"))

    # ---------------------------------------------------------------- watches and polls
    def start_watch(self, predictor: str, policy: Policy, series, every_s: float,
                    data_db: str | None = None, code_rev: str | None = None) -> int:
        with self.con:
            cur = self.con.execute(
                "INSERT INTO watch (started_at, predictor, policy, series, every_s, data_db, code_rev) "
                "VALUES (?,?,?,?,?,?,?)",
                (_now(), predictor, json.dumps(vars(policy)), ",".join(series), every_s, data_db, code_rev))
        return cur.lastrowid

    def record_error(self, watch_id: int, as_of: str, error: str) -> int:
        with self.con:
            cur = self.con.execute("INSERT INTO poll (watch_id, as_of, done_at, error) VALUES (?,?,?,?)",
                                   (watch_id, as_of, _now(), error[:2000]))
        return cur.lastrowid

    def record_poll(self, watch_id: int, pull: Pull, rows: list[MarketSnapshot],
                    decisions: list[Decision]) -> int:
        """One transaction: raw parts, events, snapshots, decisions."""
        with self.con:
            shas = [self.put_blob(x) for x in (pull.events, pull.milestones, pull.live, pull.books)]
            pid = self.con.execute(
                "INSERT INTO poll (watch_id, as_of, done_at, events_sha, milestones_sha, live_sha, books_sha, n_events) "
                "VALUES (?,?,?,?,?,?,?,?)", (watch_id, pull.as_of, _now(), *shas, len(rows))).lastrowid
            for s in rows:
                self._event(s, pull.as_of)
                self.con.execute(f"INSERT INTO snapshot ({', '.join(SNAP_COLS)}) VALUES "
                                 f"({', '.join('?' * len(SNAP_COLS))})", _snapshot_row(pid, s))
                self._prematch(pid, s)
                self._explain(pid, s)
            for d in decisions:
                sz = d.sizing
                self.con.execute(
                    "INSERT INTO decision (poll_id, event, side, p_use, p_side, ask, contracts, cost, fees, ev, "
                    "fills, reason) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (pid, d.event, d.side, d.p_use, d.p_side, d.ask, sz.contracts, sz.cost, sz.fees, sz.ev,
                     json.dumps([vars(f) for f in sz.fills]), d.reason))
        return pid

    def _event(self, s: MarketSnapshot, as_of: str) -> None:
        e = s.event
        self.con.execute(
            """INSERT INTO event (event, series, tour, title, tournament, start, a_ticker, b_ticker, a_name, b_name,
                                  a_comp, b_comp, milestone, first_seen, last_seen)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT (event) DO UPDATE SET last_seen = excluded.last_seen, start = excluded.start,
                 tournament = COALESCE(excluded.tournament, event.tournament),
                 milestone = COALESCE(excluded.milestone, event.milestone)""",
            (e.event, e.series, e.tour, e.title, s.tournament or None, e.start, e.a.ticker, e.b.ticker, e.a.name,
             e.b.name, e.a.comp, e.b.comp, s.milestone, as_of, as_of))

    def _prematch(self, poll_id: int, s: MarketSnapshot) -> None:
        """Keep the latest pre-match detail while the match has not started; once it has,
        the first one seen stays (a match first seen live keeps that poll's, whose pre-match
        fields are still pre-match: the predictor's in-play update goes into other keys)."""
        pr = s.prediction
        det = pr.detail if pr else {}
        if not det or det.get("placeholder") or "rating_a" not in det:
            return
        blob = json.dumps({k: det[k] for k in PREMATCH_KEYS if k in det}, ensure_ascii=False)
        row = (s.event.event, poll_id, pr.model, s.status, blob)
        if s.status == "not_started":
            self.con.execute(
                """INSERT INTO prematch (event, poll_id, model, status, detail) VALUES (?,?,?,?,?)
                   ON CONFLICT (event) DO UPDATE SET poll_id = excluded.poll_id, model = excluded.model,
                     status = excluded.status, detail = excluded.detail
                   WHERE prematch.status = 'not_started'""", row)
        else:
            self.con.execute("INSERT OR IGNORE INTO prematch (event, poll_id, model, status, detail) "
                             "VALUES (?,?,?,?,?)", row)

    def _explain(self, poll_id: int, s: MarketSnapshot) -> None:
        """The latest explain block: replaced by each pre-match or in-play pricing, never
        by one that priced a live match without its score (the in-play one stays)."""
        pr = s.prediction
        ex = (pr.detail or {}).get("explain") if pr else None
        if not ex or s.status not in ("not_started", "live"):
            return
        if s.status == "live" and not pr.in_play:
            return
        self.con.execute(
            """INSERT INTO explain (event, poll_id, model, status, in_play, detail) VALUES (?,?,?,?,?,?)
               ON CONFLICT (event) DO UPDATE SET poll_id = excluded.poll_id, model = excluded.model,
                 status = excluded.status, in_play = excluded.in_play, detail = excluded.detail""",
            (s.event.event, poll_id, pr.model, s.status, int(bool(ex.get("in_play"))),
             json.dumps(ex, ensure_ascii=False, separators=(",", ":"))))

    def record_alerts(self, poll_id: int, alerts) -> None:
        """engine.alerts.Alert list for one poll."""
        with self.con:
            self.con.executemany(
                "INSERT INTO alert (poll_id, event, kind, title, message, priority, status, detail) "
                "VALUES (?,?,?,?,?,?,?,?)",
                [(poll_id, a.event, a.kind, a.title, a.message, a.priority, a.status, a.detail) for a in alerts])

    def load_poll(self, poll_id: int) -> Pull:
        r = self.con.execute("SELECT * FROM poll WHERE poll_id = ?", (poll_id,)).fetchone()
        if r is None or r["error"]:
            raise KeyError(f"poll {poll_id} missing or failed")
        return Pull(r["as_of"], self.get_blob(r["events_sha"]), self.get_blob(r["milestones_sha"]) or {},
                    self.get_blob(r["live_sha"]) or {}, self.get_blob(r["books_sha"]) or {},
                    f"live.db poll {poll_id}")

    def milestones(self) -> dict:
        """event -> {'id': milestone} for every event with a known milestone (reused by
        fetch_pull, so links are requested once per event, not once per poll)."""
        r = self.con.execute("SELECT milestones_sha FROM poll WHERE error IS NULL ORDER BY poll_id DESC LIMIT 1").fetchone()
        return {k: v for k, v in ((self.get_blob(r[0]) or {}).items() if r else []) if v}

    # ---------------------------------------------------------------- settlement
    def to_settle(self, open_events: set[str], now: str | None = None) -> list[dict]:
        """Unsettled events that are no longer open or whose last snapshot says closed, seen
        in the last GIVE_UP_DAYS."""
        now = now or _now()
        cut = (datetime.fromisoformat(now) - timedelta(days=GIVE_UP_DAYS)).isoformat(timespec="seconds")
        rows = self.con.execute(
            """SELECT e.*, (SELECT s.status FROM snapshot s WHERE s.event = e.event
                            ORDER BY s.poll_id DESC LIMIT 1) AS last_status
               FROM event e LEFT JOIN settlement t USING (event)
               WHERE t.event IS NULL AND e.last_seen >= ?""", (cut,)).fetchall()
        return [dict(r) for r in rows if r["event"] not in open_events or r["last_status"] == "closed"]

    def settle(self, ev: dict, event_json: dict, details: dict | None, seen_at: str | None = None,
               milestone: dict | None = None) -> dict | None:
        """The blob keeps everything a later tennis.db loader needs (kalshi_results.py):
        the markets, the final live details (score, status, serve stats) and the milestone
        (tournament, round, best-of, start, Sportradar id). Blobs written before 2026-11-16
        lack the milestone; the export recovers it from the polls."""
        st = settlement_from(ev, event_json, details)
        if st is None:
            return None
        with self.con:
            sha = self.put_blob({"event": event_json, "live": details, "milestone": milestone})
            self.con.execute(
                """INSERT OR IGNORE INTO settlement (event, seen_at, winner, result_a, result_b, value_a, value_b,
                       market_status, match_status, final_score, raw_sha) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (ev["event"], seen_at or _now(), st["winner"], st["result_a"], st["result_b"], st["value_a"],
                 st["value_b"], st["market_status"], st["match_status"], st["final_score"], sha))
        return st

    def last_query(self, event: str, live=None) -> MatchQuery | None:
        """The MatchQuery the watch last priced `event` with, rebuilt from its latest priced
        snapshot (players, day, surface, rules, edition), with `live` attached; None if the
        match was never priced."""
        r = self.con.execute(
            """SELECT e.tour, s.a_pid, s.b_pid, s.day, s.surface, s.best_of, s.final, s.edition_id
               FROM snapshot s JOIN event e USING (event)
               WHERE s.event = ? AND s.skip = '' AND s.p_model IS NOT NULL AND s.a_pid IS NOT NULL
                 AND s.b_pid IS NOT NULL AND s.day IS NOT NULL
               ORDER BY s.poll_id DESC LIMIT 1""", (event,)).fetchone()
        if r is None:
            return None
        return MatchQuery(tour=r["tour"], player_a=r["a_pid"], player_b=r["b_pid"], on=r["day"],
                          surface=r["surface"], best_of=r["best_of"] or 3, final=r["final"] or "tb",
                          edition_id=r["edition_id"], live=live)

    def record_final_form(self, ev: dict, details: dict | None, predictor, now: str | None = None) -> dict | None:
        """The model's final read of each serve stream: Predictor.predict on the match's last
        query with the settlement's final Kalshi stats as `live`. Kept only when the predictor
        reports stream form (detail form_a / form_b), both servers played points (voids and
        walkovers have none), and neither player's last rated match is on or after the match
        day (a backfill after tennis.db took in the match would measure form against ratings
        that already include it). Returns the stored row, or None."""
        if not details:
            return None
        ls = kalshi.live_state(details, ev["a_comp"], ev["b_comp"])
        if ls.problem == "competitor ids do not match the markets":
            return None
        if ls.serve_a[1] <= 0 or ls.serve_b[1] <= 0:
            return None
        q = self.last_query(ev["event"], ls)
        if q is None:
            return None
        pr = predictor.predict(q)
        d = pr.detail or {}
        if "form_a" not in d or "form_b" not in d:
            return None
        if any((d.get(k) or "") >= q.on for k in ("last_a", "last_b")):
            return None          # the ratings already include this match day (a late backfill): not pre-match
        row = {"event": ev["event"], "recorded_at": now or _now(), "model": pr.model, "form_var": d.get("form_var"),
               "x_a": d.get("x_a"), "x_b": d.get("x_b"),
               "serve_a_won": ls.serve_a[0], "serve_a_played": ls.serve_a[1],
               "serve_b_won": ls.serve_b[0], "serve_b_played": ls.serve_b[1],
               "form_a": d["form_a"], "form_b": d["form_b"],
               "serve_a_form": d["serve_a_form"], "serve_b_form": d["serve_b_form"]}
        with self.con:
            self.con.execute(f"INSERT OR REPLACE INTO final_form ({', '.join(row)}) VALUES ({', '.join('?' * len(row))})",
                             tuple(row.values()))
        return row

    def missing_final_form(self) -> list[tuple[dict, dict | None]]:
        """Settled events with no final_form row: (event row, final live details) pairs, for
        a backfill after the fact (watch.py final-form)."""
        out = []
        for r in self.con.execute(
                """SELECT e.*, t.raw_sha FROM settlement t JOIN event e USING (event)
                   LEFT JOIN final_form f USING (event) WHERE f.event IS NULL ORDER BY t.seen_at""").fetchall():
            blob = self.get_blob(r["raw_sha"]) or {}
            out.append((dict(r), blob.get("live") if isinstance(blob, dict) else None))
        return out

    # ---------------------------------------------------------------- queries
    def at(self, event: str, t: str) -> dict | None:
        """Price, score, model and decision for `event` at time t: the last poll at or
        before t that listed it, plus its settlement if known."""
        r = self.con.execute(
            """SELECT p.as_of, s.*, d.side, d.p_use, d.contracts, d.cost, d.ev, d.reason
               FROM snapshot s JOIN poll p USING (poll_id) LEFT JOIN decision d USING (poll_id, event)
               WHERE s.event = ? AND p.as_of <= ? ORDER BY p.as_of DESC LIMIT 1""", (event, t)).fetchone()
        if r is None:
            return None
        out = dict(r)
        e = self.con.execute("SELECT a_name, b_name, title FROM event WHERE event = ?", (event,)).fetchone()
        out.update(dict(e))
        st = self.con.execute("SELECT * FROM settlement WHERE event = ?", (event,)).fetchone()
        out["settlement"] = dict(st) if st else None
        return out

    def counts(self) -> dict:
        q = lambda sql: self.con.execute(sql).fetchone()[0]          # noqa: E731
        return {"watches": q("SELECT COUNT(*) FROM watch"), "polls": q("SELECT COUNT(*) FROM poll"),
                "failed_polls": q("SELECT COUNT(*) FROM poll WHERE error IS NOT NULL"),
                "events": q("SELECT COUNT(*) FROM event"), "settled": q("SELECT COUNT(*) FROM settlement"),
                "snapshots": q("SELECT COUNT(*) FROM snapshot"), "bets": q("SELECT COUNT(*) FROM decision WHERE contracts > 0"),
                "alerts_sent": q("SELECT COUNT(*) FROM alert WHERE status = 'sent'"),
                "first_poll": q("SELECT MIN(as_of) FROM poll"), "last_poll": q("SELECT MAX(as_of) FROM poll"),
                "blob_mb": round(q("SELECT COALESCE(SUM(LENGTH(zlib)), 0) FROM blob") / 1e6, 2)}


SNAP_COLS = ("poll_id", "event", "status", "skip", "p_market", "ask_a", "ask_b", "score", "sets_a", "sets_b",
             "games_a", "games_b", "pts_a", "pts_b", "a_serving", "serve_a_won", "serve_a_played", "serve_b_won",
             "serve_b_played", "live_problem", "p_model", "p_raw", "in_play", "stale", "a_pid", "b_pid",
             "a_resolved", "b_resolved", "a_last_rated", "b_last_rated", "a_stale_days", "b_stale_days", "day",
             "surface", "surface_from", "best_of", "best_of_from", "final", "edition_id", "p_sd")


# Prediction.detail keys that are pre-match by construction (engine.predict.prematch):
# the in-play update writes *_live keys and overwrites only p_sd, so p_sd_pre is kept.
PREMATCH_KEYS = ("rating_a", "rating_b", "base", "ref", "x_a", "x_b", "v_a", "v_b", "ctx_a", "ctx_b", "serve_a",
                 "serve_b", "p_raw", "p_prematch", "p_sd_pre", "alpha", "rating_day", "surface", "best_of",
                 "final", "last_a", "last_b")


def _days(day: str, last: str | None) -> int | None:
    if not day or not last:
        return None
    return (date.fromisoformat(day) - date.fromisoformat(last)).days


def _snapshot_row(poll_id: int, s: MarketSnapshot) -> tuple:
    e, ls, pr = s.event, s.live, s.prediction
    det = pr.detail if pr else {}
    la, lb = det.get("last_a"), det.get("last_b")
    priced = bool(s.day)
    return (poll_id, e.event, s.status, s.skip, s.p_market, s.ask_a, s.ask_b,
            ls.raw if ls else None,
            *((ls.sets_a, ls.sets_b, ls.games_a, ls.games_b, ls.pts_a, ls.pts_b,
               None if ls.a_serving is None else int(ls.a_serving), *ls.serve_a, *ls.serve_b, ls.problem or None)
              if ls else (None,) * 12),
            pr.p_a if pr else None, pr.p_raw if pr else None,
            None if pr is None else int(pr.in_play), None if pr is None else int(pr.stale),
            e.a.player_id, e.b.player_id, e.a.resolved or None, e.b.resolved or None, la, lb,
            _days(s.day, la), _days(s.day, lb), s.day or None,
            s.surface if priced else None, s.surface_from or None, s.best_of if priced else None,
            s.best_of_from or None, s.final if priced else None, s.edition_id, det.get("p_sd"))
