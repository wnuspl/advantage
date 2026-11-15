"""Kalshi tennis match markets: fetch (public, read-only endpoints), order-book ladders,
taker fees, and Kelly sizing that walks the book.

Market structure (verified 2026-11-15 on KXATPMATCH / KXWTAMATCH / KXATPCHALLENGERMATCH):
one *event* per match (`mutually_exclusive`), with two binary *markets*, one per player
(`yes_sub_title` = that player's full name). Each pays $1 if YES. The order book lists
bids only: `orderbook_fp.yes_dollars` (bids to buy YES) and `.no_dollars` (bids to buy
NO), as [price, size] with price in dollars. Buying YES at x fills against a NO bid at
1 - x, so the YES asks are the NO bids mirrored.

"A wins" can be bought two ways: YES on A's market, or NO on B's market (NO on B pays
$1 iff B loses, i.e. A wins, since a match only settles if a ball is played). `ladder()`
merges both into one ask ladder.

Fee (taker, Kalshi general fee schedule): ceil(rate * C * P * (1 - P)) dollars per order,
rate 0.07. Sizing uses the per-contract fee rate * P * (1 - P) without the per-order
rounding up to the cent (at most 1 cent per order).

Nothing here places orders. Reads only.
"""
from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field

BASE = "https://api.elections.kalshi.com/trade-api/v2"
MATCH_SERIES = ("KXATPMATCH", "KXWTAMATCH", "KXATPCHALLENGERMATCH", "KXWTACHALLENGERMATCH")
TOUR_OF_SERIES = {"KXATPMATCH": "atp", "KXWTAMATCH": "wta",
                  "KXATPCHALLENGERMATCH": "atp", "KXWTACHALLENGERMATCH": "wta"}
FEE_RATE = 0.07
UA = "advantage-research/0.1"


# ---------------------------------------------------------------------------- fetch

def get(path: str, params: dict | None = None, retries: int = 3) -> dict:
    url = BASE + path + ("?" + urllib.parse.urlencode(params) if params else "")
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    for k in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429 and k + 1 < retries:      # rate limited: back off
                time.sleep(1.5 * (k + 1))
                continue
            raise
    raise RuntimeError("unreachable")


def open_events(series: str) -> list[dict]:
    """All open events of a series, markets nested (follows the cursor)."""
    out, cursor = [], None
    while True:
        params = {"series_ticker": series, "status": "open", "with_nested_markets": "true", "limit": 200}
        if cursor:
            params["cursor"] = cursor
        r = get("/events", params)
        out.extend(r.get("events", []))
        cursor = r.get("cursor")
        if not cursor or not r.get("events"):
            return out


def orderbook(ticker: str) -> dict:
    return get(f"/markets/{ticker}/orderbook").get("orderbook_fp", {})


# ---------------------------------------------------------------------------- live scores
# Each match event links to a Sportradar-fed "milestone" (verified 2026-11-15):
#   GET /milestones?related_event_ticker=<event>   -> id, details.status, details.best_of
#   GET /live_data/batch?milestone_ids=<id>&milestone_ids=<id>...   -> live_datas[].details:
#     status 'not_started' | 'live' | 'closed'; competitor{1,2}_id (= the markets'
#     custom_strike.tennis_competitor); competitor{1,2}_round_scores [{score, outcome
#     'winner'|'loser'|'ongoing'}] (games per set); competitor{1,2}_current_round_score
#     (current game: 0/15/30/40, advantage shown as 50 against 40; a separate `advantage`
#     field exists but was empty when 50 appeared); server (competitor id of the next server);
#     competitor{1,2}_statistics incl. service_points_won / service_points_lost.
# The live data carries no timestamp of its own: how far it lags the court is unknown.

def milestone(event_ticker: str) -> dict | None:
    for m in get("/milestones", {"related_event_ticker": event_ticker, "limit": 10}).get("milestones", []):
        if m.get("type", "").startswith("tennis") and event_ticker in (m.get("primary_event_tickers") or []):
            return m
    return None


def live_batch(ids: list[str], chunk: int = 20) -> dict[str, dict]:
    """milestone id -> live details (one request per `chunk` ids)."""
    out = {}
    for k in range(0, len(ids), chunk):
        q = urllib.parse.urlencode([("milestone_ids", i) for i in ids[k:k + chunk]])
        for x in get("/live_data/batch?" + q).get("live_datas") or []:
            out[x["milestone_id"]] = x.get("details") or {}
    return out


_GAME_PTS = {0: 0, 15: 1, 30: 2, 40: 3}


@dataclass
class LiveState:
    status: str                       # not_started / live / closed / unknown
    sets_a: int = 0
    sets_b: int = 0
    games_a: int = 0
    games_b: int = 0
    pts_a: int = 0
    pts_b: int = 0
    a_serving: bool | None = None
    serve_a: tuple[int, int] = (0, 0)  # (won, played) on serve this match
    serve_b: tuple[int, int] = (0, 0)
    problem: str = ""                 # set when the score cannot be read safely
    raw: str = ""                     # human-readable score

    def score(self) -> dict:
        return {"sets_a": self.sets_a, "sets_b": self.sets_b, "games_a": self.games_a,
                "games_b": self.games_b, "pts_a": self.pts_a, "pts_b": self.pts_b}


def live_state(d: dict, comp_a: str, comp_b: str, best_of: int = 3, final: str = "tb") -> LiveState:
    """Kalshi live details -> the in-play model's inputs, A/B = the event's two markets
    (matched by competitor id, never by name or by competitor1/2 order). Anything
    unexpected leaves `problem` set: callers must not bet on such a state."""
    st = LiveState(status=d.get("status") or "unknown")
    side = {d.get("competitor1_id"): "1", d.get("competitor2_id"): "2"}
    if comp_a not in side or comp_b not in side or comp_a == comp_b:
        st.problem = "competitor ids do not match the markets"
        return st
    ka, kb = side[comp_a], side[comp_b]

    def stat(k, name):
        return int((d.get(f"competitor{k}_statistics") or {}).get(name) or 0)
    for k, attr in ((ka, "serve_a"), (kb, "serve_b")):
        won = stat(k, "service_points_won")
        setattr(st, attr, (won, won + stat(k, "service_points_lost")))
    if st.status != "live":
        return st

    ra, rb = d.get(f"competitor{ka}_round_scores") or [], d.get(f"competitor{kb}_round_scores") or []
    if len(ra) != len(rb):
        st.problem = "set lists differ in length"
        return st
    need = best_of // 2 + 1
    for x, y in zip(ra, rb):
        if x.get("outcome") == "ongoing" or y.get("outcome") == "ongoing":
            st.games_a, st.games_b = int(x.get("score") or 0), int(y.get("score") or 0)
        elif x.get("outcome") == "winner":
            st.sets_a += 1
        elif y.get("outcome") == "winner":
            st.sets_b += 1
    if max(st.sets_a, st.sets_b) >= need:
        st.problem = "match already decided"
        return st
    deciding = st.sets_a == st.sets_b == need - 1
    tb_at = None if deciding and final == "adv" else 12 if deciding and final == "tb12" else 6
    in_tb = (deciding and final == "mtb") or (tb_at is not None and st.games_a == st.games_b == tb_at)
    pa, pb = d.get(f"competitor{ka}_current_round_score"), d.get(f"competitor{kb}_current_round_score")
    try:
        pa, pb = int(pa or 0), int(pb or 0)
    except (TypeError, ValueError):
        st.problem = f"unreadable points {pa!r}-{pb!r}"
        return st
    adv = d.get("advantage") or ""
    if in_tb:
        st.pts_a, st.pts_b = pa, pb
        shown = f"{pa}-{pb} (tiebreak)"
    else:
        if (pa, pb) in ((50, 40), (40, 50)):           # Kalshi shows advantage as 50 (seen 2026-11-15)
            st.pts_a, st.pts_b = (4, 3) if pa == 50 else (3, 4)
            shown = "AD-40" if pa == 50 else "40-AD"
            adv = ""
        elif pa not in _GAME_PTS or pb not in _GAME_PTS:
            st.problem = f"unexpected game points {pa}-{pb}"
            return st
        else:
            st.pts_a, st.pts_b = _GAME_PTS[pa], _GAME_PTS[pb]
            shown = f"{pa}-{pb}"
        if adv:
            if st.pts_a != 3 or st.pts_b != 3:
                pass                                   # stale advantage flag at a non-deuce score
            elif adv in (comp_a, f"competitor{ka}"):
                st.pts_a, shown = 4, "AD-40"
            elif adv in (comp_b, f"competitor{kb}"):
                st.pts_b, shown = 4, "40-AD"
            else:
                st.problem = f"unknown advantage value {adv!r}"
                return st
    srv = d.get("server")
    if srv not in (comp_a, comp_b):
        st.problem = "server unknown"
        return st
    st.a_serving = srv == comp_a
    done = " ".join(f"{int(x.get('score') or 0)}-{int(y.get('score') or 0)}" for x, y in zip(ra, rb)
                    if x.get("outcome") != "ongoing")
    st.raw = (done + " " if done else "") + f"{st.games_a}-{st.games_b} {shown}" + (" *A serving" if st.a_serving else " *B serving")
    return st


# ---------------------------------------------------------------------------- books

@dataclass
class Level:
    price: float          # dollars per contract, before fee
    size: float           # contracts available
    route: str            # "YES <ticker>" or "NO <ticker>"


def _levels(side: list | None) -> list[tuple[float, float]]:
    return [(float(p), float(s)) for p, s in (side or []) if float(s) > 0]


def asks(book: dict, side: str) -> list[tuple[float, float]]:
    """Ask ladder (price, size), cheapest first, to BUY `side` ('yes'|'no') of one market:
    the opposite side's bids mirrored."""
    other = "no_dollars" if side == "yes" else "yes_dollars"
    lv = [(round(1 - p, 4), s) for p, s in _levels(book.get(other))]
    return sorted(lv)


def best_bid(book: dict, side: str) -> float | None:
    lv = _levels(book.get(f"{side}_dollars"))
    return max(p for p, _ in lv) if lv else None


def ladder(book_a: dict, ticker_a: str, book_b: dict | None, ticker_b: str | None) -> list[Level]:
    """Every way to buy 'A wins', cheapest first: YES on A's market, NO on B's market."""
    out = [Level(p, s, f"YES {ticker_a}") for p, s in asks(book_a, "yes")]
    if book_b is not None:
        out += [Level(p, s, f"NO {ticker_b}") for p, s in asks(book_b, "no")]
    out = [lv for lv in out if 0 < lv.price < 1]
    out.sort(key=lambda lv: (lv.price, lv.route))
    return out


def fair_prob(book_a: dict, book_b: dict | None) -> float | None:
    """Market's P(A wins): mid of A's YES market and of 1 - B's YES mid, averaged
    (each mid = (best YES bid + best YES ask) / 2). The two markets' mids usually sum
    to ~1.01-1.03; averaging A and 1-B removes that overround symmetrically."""
    def mid(book):
        bid, ask_lv = best_bid(book, "yes"), asks(book, "yes")
        if bid is None or not ask_lv:
            return None
        return (bid + ask_lv[0][0]) / 2
    ma = mid(book_a)
    mb = mid(book_b) if book_b is not None else None
    vals = [v for v in (ma, None if mb is None else 1 - mb) if v is not None]
    return sum(vals) / len(vals) if vals else None


def fee_per_contract(price: float, rate: float = FEE_RATE) -> float:
    return rate * price * (1 - price)


# ---------------------------------------------------------------------------- sizing

@dataclass
class Fill:
    price: float
    contracts: int
    route: str


@dataclass
class Sizing:
    contracts: int = 0
    cost: float = 0.0                 # dollars incl. fees
    fees: float = 0.0
    fills: list[Fill] = field(default_factory=list)
    ev: float = 0.0                   # expected profit, dollars
    log_growth: float = 0.0           # expected log growth of the (fraction-scaled) bankroll

    @property
    def avg_price(self) -> float:
        return (self.cost - self.fees) / self.contracts if self.contracts else float("nan")


def kelly_walk(p: float, levels: list[Level], bankroll: float, fraction: float = 1.0,
               fee_rate: float = FEE_RATE, max_cost: float | None = None) -> Sizing:
    """Contracts on 'A wins' (each pays $1) maximizing expected log wealth, walking the
    ask ladder. `fraction` < 1 is fractional Kelly: the full-Kelly solve on
    fraction * bankroll (exact for a single price level, since the Kelly stake is
    linear in bankroll). Wealth is L = W - spent if A loses and H = L + contracts if A
    wins; at a level costing c per contract (incl. fee) the log-optimal extra amount is
        x* = (p (1 - c) L - (1 - p) c H) / (c (1 - c)),
    which reduces to the textbook stake c x* = W (p - c) / (1 - c) from flat. Stops at
    the first level where x* <= 0 (later levels cost more). Integer contracts."""
    W = bankroll * fraction
    out = Sizing()
    if W <= 0 or not 0 < p < 1:
        return out
    L, H = W, W
    for lv in levels:
        c = lv.price + fee_rate * lv.price * (1 - lv.price)
        if c >= 1 or c >= p:
            break
        x = (p * (1 - c) * L - (1 - p) * c * H) / (c * (1 - c))
        n = int(math.floor(min(x, lv.size) + 1e-9))
        if max_cost is not None:
            n = min(n, int(math.floor((max_cost - out.cost) / c + 1e-9)))
        if n <= 0:
            break
        out.fills.append(Fill(lv.price, n, lv.route))
        out.contracts += n
        out.cost += n * c
        out.fees += n * (c - lv.price)
        L -= n * c
        H = L + out.contracts
        if n < math.floor(lv.size + 1e-9):            # optimum reached inside this level
            break
    if out.contracts:
        out.ev = p * out.contracts - out.cost
        out.log_growth = p * math.log(H / W) + (1 - p) * math.log(L / W)
    return out


def scale_sizing(s: Sizing, p: float, scale: float, bankroll: float, fee_rate: float = FEE_RATE) -> Sizing:
    """Shrink a sizing pro rata (portfolio cap): each fill's contracts * scale, floored.
    ev and log_growth recomputed; log_growth relative to `bankroll`."""
    out = Sizing()
    for f in s.fills:
        n = int(math.floor(f.contracts * scale + 1e-9))
        if n <= 0:
            continue
        c = f.price + fee_rate * f.price * (1 - f.price)
        out.fills.append(Fill(f.price, n, f.route))
        out.contracts += n
        out.cost += n * c
        out.fees += n * (c - f.price)
    if out.contracts and bankroll > out.cost:
        out.ev = p * out.contracts - out.cost
        L = bankroll - out.cost
        out.log_growth = p * math.log((L + out.contracts) / bankroll) + (1 - p) * math.log(L / bankroll)
    return out


def blend(p_model: float, p_market: float | None, w: float) -> float:
    """Logit-space blend: w = 1 model only, 0 market only."""
    if p_market is None or w >= 1:
        return p_model
    lg = lambda q: math.log(q / (1 - q))                       # noqa: E731
    z = w * lg(min(max(p_model, 1e-6), 1 - 1e-6)) + (1 - w) * lg(min(max(p_market, 1e-6), 1 - 1e-6))
    return 1 / (1 + math.exp(-z))
