"""Kalshi retriever: open match-winner markets + live scores -> priced `MarketSnapshot`s.

Track B1 (docs/PLAN.md), rewritten from v1 `kalshi_edges.py` against the predictor seam:
the retriever never touches a model, it builds a `MatchQuery` per market and asks
whatever `Predictor` it is given. Read-only: no order code anywhere.

    pull = fetch_pull()                       # network: events, milestones, live, books
    save_pull(pull, RAW_KALSHI)               # immutable data/raw/kalshi/<UTC>/
    rows = snapshots(pull, predictor, resolver, editions)

A pull is everything one poll saw, kept raw so it can be repriced offline (tests run on
saved pulls, never the network):

    events.json      series -> open events, markets nested            (GET /events)
    milestones.json  event ticker -> its tennis milestone, or null    (GET /milestones)
    live.json        milestone id -> live details                     (GET /live_data/batch)
    orderbooks.json  market ticker -> orderbook_fp                    (GET /markets/<t>/orderbook)
    pull.json        when, which series, counts

Pulls made before 2026-11-15 22:00 have no milestones.json / pull.json, and the earliest
no live.json; they load with those parts empty.

Live data maps to an event by its **pair of competitor ids** (the markets'
`custom_strike.tennis_competitor`), never by name or order. When the milestone link is
known it must agree with that pair, otherwise the row is refused.

Rules and surface for the query, in order:
  best-of   milestone details.best_of > the running edition in tennis.db > 5 for a men's
            slam main draw (by competition name) > 3
  final     the edition's deciding-set rule inferred from its tournament's scores
            (points_rating.infer_rules) > 'tb'
  surface   the edition's > the tournament's history (engine/surfaces.py: Kalshi gives no
            surface, only the tournament name) > the caller's fallback > None (unknown)
The running edition is one both players already have a match in, started within 21 days
and not ended before the day before the match (as v1). A first-round match has none.
"""
from __future__ import annotations

import json
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .. import kalshi
from ..normalize import norm_name
from ..points_rating import infer_rules
from .editions import shared_edition
from .predictor import MatchQuery, Prediction, Predictor
from .surfaces import TournamentSurfaces, kalshi_tournament

ROOT = Path(__file__).resolve().parents[3]
RAW_KALSHI = ROOT / "data" / "raw" / "kalshi"
SLAMS = ("australian open", "roland garros", "french open", "wimbledon", "us open")
WORKERS = 6                       # parallel requests, well under Kalshi's read rate limit
_BORN = re.compile(r"\s*\(b\.\s*\d{4}\)\s*$")


def plain_name(name: str) -> str:
    """Kalshi tells namesakes apart with a birth-year suffix ("Cezar Cretu (b. 2001)",
    seen 2026-11-15); the resolver gets the bare name."""
    return _BORN.sub("", name)


# ---------------------------------------------------------------------------- events

@dataclass
class Side:
    ticker: str
    name: str                     # yes_sub_title: the player's full name
    comp: str | None              # custom_strike.tennis_competitor (Sportradar-fed id)
    player_id: int | None = None
    resolved: str = ""            # how the name resolved (players.Resolver)


@dataclass
class Event:
    event: str
    series: str
    tour: str
    title: str
    competition: str
    rules: str
    start: str | None             # occurrence_datetime (UTC ISO)
    a: Side
    b: Side


def parse_events(events: dict[str, list[dict]]) -> tuple[list[Event], list[tuple[str, str]]]:
    """series -> raw events  ->  (one Event per match, [(event ticker, why rejected)])."""
    out, rejected = [], []
    for series, evs in events.items():
        tour = kalshi.TOUR_OF_SERIES.get(series)
        for e in evs:
            et = e.get("event_ticker", "?")
            if tour is None:
                rejected.append((et, f"unknown series {series}"))
                continue
            # A/B by ticker: stable across polls (the API's market order is not guaranteed)
            mk = sorted((m for m in e.get("markets", []) if m.get("status") == "active"), key=lambda m: m["ticker"])
            if len(mk) != 2:
                rejected.append((et, f"{len(mk)} active markets, expected 2"))
                continue
            sides = [Side(m["ticker"], m.get("yes_sub_title", ""),
                          (m.get("custom_strike") or {}).get("tennis_competitor")) for m in mk]
            if not all(s.comp for s in sides) or sides[0].comp == sides[1].comp:
                rejected.append((et, "markets lack two distinct competitor ids"))
                continue
            a = mk[0]
            out.append(Event(et, series, tour, e.get("title", ""),
                             (e.get("product_metadata") or {}).get("competition", ""),
                             a.get("rules_primary", ""),
                             a.get("occurrence_datetime") or a.get("expected_expiration_time"),
                             sides[0], sides[1]))
    return out, rejected


# ---------------------------------------------------------------------------- pulls

@dataclass
class Pull:
    as_of: str                                    # UTC ISO time the pull started
    events: dict[str, list[dict]]                 # series -> raw events
    milestones: dict[str, dict | None] = field(default_factory=dict)   # event -> milestone
    live: dict[str, dict] = field(default_factory=dict)               # milestone id -> details
    books: dict[str, dict] = field(default_factory=dict)              # ticker -> orderbook_fp
    source: str = ""                              # folder it was loaded from / saved to


def fetch_pull(series=kalshi.MATCH_SERIES, workers: int = WORKERS, known_milestones: dict | None = None) -> Pull:
    """One poll of Kalshi (public endpoints, no key). Books are fetched for every event
    whose live status is not 'closed'. `known_milestones` (event -> milestone, e.g. from the
    previous poll) are reused instead of re-requested; events without one are asked again."""
    as_of = datetime.now(timezone.utc).isoformat(timespec="seconds")
    raw = {s: kalshi.open_events(s) for s in series}
    evs, _ = parse_events(raw)
    known = known_milestones or {}
    ask = [e for e in evs if not known.get(e.event)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        fresh = dict(zip([e.event for e in ask], pool.map(lambda e: kalshi.milestone(e.event), ask)))
    ms = {e.event: known.get(e.event) or fresh.get(e.event) for e in evs}
    live = kalshi.live_batch([m["id"] for m in ms.values() if m])
    pull = Pull(as_of, raw, ms, live)
    tickers = [s.ticker for e in evs if _status(_live_for(e, pull)[1]) != "closed" for s in (e.a, e.b)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pull.books = dict(zip(tickers, pool.map(kalshi.orderbook, tickers)))
    return pull


def save_pull(pull: Pull, root: Path = RAW_KALSHI) -> Path:
    """Write the pull to root/<UTC stamp>/ (never overwrites: raw files are immutable)."""
    stamp = datetime.fromisoformat(pull.as_of).astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    d = Path(root) / stamp
    d.mkdir(parents=True, exist_ok=False)
    for name, obj in (("events", pull.events), ("milestones", pull.milestones),
                      ("live", pull.live), ("orderbooks", pull.books)):
        (d / f"{name}.json").write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    meta = {"as_of": pull.as_of, "series": list(pull.events),
            "events": sum(map(len, pull.events.values())), "milestones": sum(1 for m in pull.milestones.values() if m),
            "live": len(pull.live), "orderbooks": len(pull.books)}
    (d / "pull.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    pull.source = str(d)
    return d


def load_pull(d: Path) -> Pull:
    """A saved pull. Older pulls lack milestones/pull.json (and the first ones live.json);
    their as_of comes from the folder name."""
    d = Path(d)

    def part(name):
        p = d / f"{name}.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    meta = part("pull")
    as_of = meta.get("as_of") or datetime.strptime(d.name, "%Y%m%dT%H%M%SZ").replace(
        tzinfo=timezone.utc).isoformat(timespec="seconds")
    return Pull(as_of, part("events"), part("milestones"), part("live"), part("orderbooks"), str(d))


def _live_for(e: Event, pull: Pull) -> tuple[str | None, dict | None, str]:
    """(milestone id, live details, problem) for one event, by competitor-id pair."""
    pair = {e.a.comp, e.b.comp}
    ms = pull.milestones.get(e.event)
    if ms:
        mid = ms.get("id")
        det = ms.get("details") or {}
        mpair = {det.get("first_competitor_id"), det.get("second_competitor_id")}
        if None not in mpair and mpair != pair:
            return mid, None, "milestone competitors do not match the markets"
        d = pull.live.get(mid)
        if d is not None:
            if {d.get("competitor1_id"), d.get("competitor2_id")} != pair:
                return mid, None, "live competitors do not match the markets"
            return mid, d, ""
    for mid, d in pull.live.items():                 # legacy pulls: no milestone links saved
        if {d.get("competitor1_id"), d.get("competitor2_id")} == pair:
            return mid, d, ""
    return (ms or {}).get("id"), None, ""


def _status(d: dict | None) -> str:
    return (d or {}).get("status") or "no_feed"


# ---------------------------------------------------------------------------- editions

@dataclass(frozen=True)
class EditionInfo:
    edition_id: int
    surface: str | None
    best_of: int | None
    final: str


EditionLookup = Callable[[int, int, str], "EditionInfo | None"]


class DbEditions:
    """The running edition both players are in (see module doc), from tennis.db."""

    def __init__(self, con: sqlite3.Connection, alias: str = "main"):
        self.con, self.alias = con, alias
        self._rules: dict[int, dict] = {}

    def _final(self, tournament_id: int, edition_id: int) -> str:
        if tournament_id not in self._rules:
            a = self.alias
            rows = self.con.execute(
                f"""SELECT m.edition_id, e.tournament_id, e.start_date, m.score, COALESCE(m.best_of, e.best_of)
                    FROM {a}.matches m JOIN {a}.tournament_edition e USING (edition_id)
                    WHERE e.tournament_id = ? AND m.status = 'completed'""", (tournament_id,)).fetchall()
            self._rules[tournament_id] = infer_rules(rows)[0]
        return self._rules[tournament_id].get(edition_id, "tb")

    def __call__(self, a: int, b: int, day: str) -> EditionInfo | None:
        r = shared_edition(self.con, a, b, day, alias=self.alias)
        if r is None:
            return None
        eid, surface, best_of = r
        tid = self.con.execute(f"SELECT tournament_id FROM {self.alias}.tournament_edition WHERE edition_id = ?",
                               (eid,)).fetchone()[0]
        return EditionInfo(eid, surface, best_of, self._final(tid, eid))


# ---------------------------------------------------------------------------- snapshots

@dataclass
class MarketSnapshot:
    event: Event
    as_of: str
    status: str                       # not_started / live / closed / no_feed
    milestone: str | None = None
    tournament: str = ""              # Kalshi's tournament name, round stripped ("ATP Challenger Kobe")
    live: kalshi.LiveState | None = None
    day: str = ""
    surface: str | None = None
    surface_from: str = ""
    best_of: int = 3
    best_of_from: str = ""
    final: str = "tb"
    edition_id: int | None = None
    p_market: float | None = None     # de-overrounded mid (kalshi.fair_prob), P(A wins)
    ask_a: float | None = None        # cheapest way to buy "A wins" (YES A or NO B), before fee
    ask_b: float | None = None
    query: MatchQuery | None = None
    prediction: Prediction | None = None
    skip: str = ""                    # why there is no prediction ("" = priced)

    @property
    def p_model(self) -> float | None:
        return self.prediction.p_a if self.prediction else None

    @property
    def deviation(self) -> float | None:
        if self.p_model is None or self.p_market is None:
            return None
        return self.p_model - self.p_market

    def to_json(self) -> dict:
        e, ls, pr = self.event, self.live, self.prediction
        return {
            "event": e.event, "series": e.series, "tour": e.tour, "title": e.title,
            "competition": e.competition, "start": e.start, "as_of": self.as_of, "status": self.status,
            "milestone": self.milestone, "tournament": self.tournament,
            "a": vars(e.a), "b": vars(e.b),
            "day": self.day, "surface": self.surface, "surface_from": self.surface_from,
            "best_of": self.best_of, "best_of_from": self.best_of_from, "final": self.final,
            "edition_id": self.edition_id,
            "live": None if ls is None else {"score": ls.raw, "problem": ls.problem, "a_serving": ls.a_serving,
                                             "serve_a": list(ls.serve_a), "serve_b": list(ls.serve_b), **ls.score()},
            "p_market": self.p_market, "ask_a": self.ask_a, "ask_b": self.ask_b,
            "prediction": None if pr is None else {"p_a": pr.p_a, "p_raw": pr.p_raw, "in_play": pr.in_play,
                                                   "model": pr.model, "stale": pr.stale, "notes": pr.notes},
            "deviation": self.deviation, "skip": self.skip,
        }


def _is_slam_main(e: Event) -> bool:
    comp = norm_name(e.competition + " " + e.rules)
    return e.tour == "atp" and any(s in comp for s in SLAMS) and "qualif" not in comp


def snapshots(pull: Pull, predictor: Predictor, resolve: Callable[[str, str], tuple[int | None, str]],
              editions: EditionLookup | None = None, surface: str | None = None,
              include_started: bool = False, surfaces: TournamentSurfaces | None = None) -> list[MarketSnapshot]:
    """One MarketSnapshot per open match event in the pull, priced by `predictor` unless
    `skip` says why not. Rows come back in event order; nothing is dropped silently
    (events that fail to parse come back as `rejected` from parse_events)."""
    now = datetime.fromisoformat(pull.as_of)
    events, _ = parse_events(pull.events)
    out = []
    for e in events:
        mid, details, problem = _live_for(e, pull)
        s = MarketSnapshot(e, pull.as_of, _status(details), mid,
                           tournament=kalshi_tournament(pull.milestones.get(e.event), e.rules))
        ba, bb = pull.books.get(e.a.ticker), pull.books.get(e.b.ticker)
        if ba is not None and bb is not None:
            s.p_market = kalshi.fair_prob(ba, bb)
            la, lb = kalshi.ladder(ba, e.a.ticker, bb, e.b.ticker), kalshi.ladder(bb, e.b.ticker, ba, e.a.ticker)
            s.ask_a = la[0].price if la else None
            s.ask_b = lb[0].price if lb else None
        out.append(s)
        if problem:
            s.skip = problem
            continue
        if s.status == "closed":
            s.skip = "finished"
            continue
        started = e.start is not None and datetime.fromisoformat(e.start.replace("Z", "+00:00")) <= now
        if s.status == "no_feed" and started and not include_started:
            s.skip = "started, no live score"
            continue
        for side in (e.a, e.b):
            side.player_id, side.resolved = resolve(e.tour, plain_name(side.name))
        missing = [x.name for x in (e.a, e.b) if x.player_id is None]
        if missing:
            s.skip = "unresolved: " + ", ".join(missing)
            continue
        if e.a.player_id == e.b.player_id:
            s.skip = "both names resolve to one player"
            continue

        s.day = (e.start or pull.as_of)[:10]
        ed = editions(e.a.player_id, e.b.player_id, s.day) if editions else None
        ms_bo = ((pull.milestones.get(e.event) or {}).get("details") or {}).get("best_of")
        if ms_bo and str(ms_bo).isdigit() and int(ms_bo) in (1, 3, 5):
            s.best_of, s.best_of_from = int(ms_bo), "milestone"
        elif ed and ed.best_of:
            s.best_of, s.best_of_from = ed.best_of, "edition"
        elif _is_slam_main(e):
            s.best_of, s.best_of_from = 5, "slam name"
        else:
            s.best_of, s.best_of_from = 3, "default"
        s.final = ed.final if ed else "tb"
        s.edition_id = ed.edition_id if ed else None
        guess = surfaces.lookup(e.tour, s.tournament, s.day) if surfaces and not (ed and ed.surface) else None
        if ed and ed.surface:
            s.surface, s.surface_from = ed.surface, "edition"
        elif guess:
            s.surface, s.surface_from = guess.surface, "history: " + guess.how
        elif surface:
            s.surface, s.surface_from = surface, "fallback"
        else:
            s.surface, s.surface_from = None, "unknown"

        if s.status == "live":
            s.live = kalshi.live_state(details, e.a.comp, e.b.comp, s.best_of, s.final)
            if s.live.problem:
                s.skip = "live score unreadable: " + s.live.problem
                continue
        s.query = MatchQuery(tour=e.tour, player_a=e.a.player_id, player_b=e.b.player_id, on=s.day,
                             surface=s.surface, best_of=s.best_of, final=s.final,
                             edition_id=s.edition_id, live=s.live)
        s.prediction = predictor.predict(s.query)
    return out


# ---------------------------------------------------------------------------- setup

def load_predictor(kind: str, *, fixed_p: float = 0.5, run: int = 790, raw: bool = False,
                   data_path: Path | None = None, today: str | None = None) -> Predictor:
    """'fixed' (placeholder) or 'points' (engine.predict.PointsPredictor of `run`)."""
    if kind == "fixed":
        from .predictor import FixedPredictor
        return FixedPredictor(fixed_p)
    from .predict import PointsPredictor
    kw = {"data_path": data_path} if data_path else {}
    return PointsPredictor.load(run, today=today, calibrated=not raw, **kw)


class Retriever:
    """Everything snapshots() needs besides the pull, built once from tennis.db (an open
    read-only connection): the predictor, its name resolver (or one from tennis.db), the
    running-edition lookup and the tournament surface history."""

    def __init__(self, predictor: Predictor, con: sqlite3.Connection, today: str,
                 surface: str | None = None, include_started: bool = False):
        from .players import Resolver
        self.predictor, self.con = predictor, con
        self.surface, self.include_started = surface, include_started
        self.resolve = getattr(predictor, "players", None) or Resolver.from_db(con, today, alias="main")
        self.editions = DbEditions(con)
        self.surfaces = TournamentSurfaces.from_db(con)

    def __call__(self, pull: Pull) -> list[MarketSnapshot]:
        return snapshots(pull, self.predictor, self.resolve, self.editions, self.surface,
                         self.include_started, self.surfaces)
