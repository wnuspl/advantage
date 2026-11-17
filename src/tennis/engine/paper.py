"""Paper-trading replays on a watch's live.db (track B5): what following the flagged bets
would have made, and whether a selling rule beats holding to settlement.

    con = livefeed.connect(LIVE_DB)               # read-only
    bets = first_bets(con, lambda wid: pol)       # event -> the first flagged bet
    pos, state = sell_replay(con, pol)            # the same entries, then the sell rules
    view = pnl_view(con, bankroll=1000)           # what /pnl shows (JSON-ready), one pass

Both re-decide on the order books recorded with each poll (engine.livefeed.redecide), at a
bankroll you choose, and count only matches with a settlement (a winner).

  first bets - each settled match's FIRST flagged bet (one bet per match, placed when first
          recommended), under a policy per watch. $1 per contract if the side won; cost
          includes fees.
  sell  - the same entries under one policy, then at every later poll an exit for the held
          "side wins" contracts: sell into their own market's bids, or buy the other side
          (YES on the opponent / NO on our player). Either way it is "buy the other side at
          q", worth 1 - q - fee(q) per contract now (the pair pays $1 whoever wins). Rules:
          hold; model (sell while the exit beats our blended win probability by a margin:
          the condition under which decide() would flag the other side); take profit at
          +15c; take profit / stop loss at +-15c.

Every consumer reads the same pass over the polls (`replay`), and a poll's order books are
decompressed only when some consumer needs them, so pnl_view costs one scan of live.db.
scripts/paper_replay.py prints these; scripts/web_server.py serves pnl_view at /api/pnl.
"""
from __future__ import annotations

import json
import math
import sqlite3
import zlib
from typing import Callable

from .. import kalshi
from .decide import Policy
from .livefeed import TOUR_LABEL, policy_with, redecide

TP = 0.15                       # take-profit / stop-loss distance, dollars per contract


def settled(con: sqlite3.Connection) -> dict:
    return {r["event"]: r for r in con.execute("SELECT * FROM settlement WHERE winner IS NOT NULL")}


def policies(con: sqlite3.Connection) -> dict[int, dict]:
    return {r["watch_id"]: json.loads(r["policy"]) for r in con.execute("SELECT * FROM watch")}


class Books:
    """A poll's order books, decompressed on first use."""

    def __init__(self, con: sqlite3.Connection, sha: str | None):
        self.con, self.sha, self._v = con, sha, None

    def get(self) -> dict:
        if self._v is None:
            b = self.con.execute("SELECT zlib FROM blob WHERE sha256 = ?", (self.sha,)).fetchone()
            self._v = json.loads(zlib.decompress(b[0]).decode("utf-8")) if b else {}
        return self._v


def polls(con: sqlite3.Connection, events: set):
    """(poll row, snapshot rows of `events`, Books) per successful poll, in order."""
    for p in con.execute("SELECT poll_id, watch_id, as_of, books_sha FROM poll WHERE error IS NULL "
                         "ORDER BY poll_id").fetchall():
        rows = [r for r in con.execute("""SELECT s.*, e.a_name, e.b_name, e.a_ticker, e.b_ticker FROM snapshot s
                                          JOIN event e USING (event) WHERE s.poll_id = ?""", (p["poll_id"],))
                if r["event"] in events]
        if rows:
            yield p, rows, Books(con, p["books_sha"])


def _entry(d, r, p) -> dict:
    return {"event": r["event"], "side": d.side, "n": d.sizing.contracts, "cost": d.sizing.cost,
            "ev": d.sizing.ev, "ask": d.ask, "p_side": d.p_side, "paid": d.sizing.cost / d.sizing.contracts,
            "as_of": p["as_of"], "poll_id": p["poll_id"], "status": r["status"],
            "name": r["a_name"] if d.side == "a" else r["b_name"],
            "opp": r["b_name"] if d.side == "a" else r["a_name"]}


class FirstBets:
    """Consumer: event -> its first flagged bet, `policy_for(watch_id)` giving the Policy per poll."""

    def __init__(self, policy_for: Callable[[int], Policy]):
        self.policy_for = policy_for
        self.out: dict[str, dict] = {}

    def step(self, p, rows, books: Books) -> None:
        new = [r for r in rows if r["event"] not in self.out]
        if not new:
            return
        for ev, d in redecide(new, books.get(), self.policy_for(p["watch_id"])).items():
            if d.sizing.contracts > 0:
                self.out[ev] = _entry(d, next(r for r in new if r["event"] == ev), p)

    def finish(self, st: dict) -> dict[str, dict]:
        for ev, e in self.out.items():
            e["won"] = st[ev]["winner"] == e["side"]
            e["pnl"] = (e["n"] if e["won"] else 0) - e["cost"]
            e["final_score"] = st[ev]["final_score"]
            e["settled_at"] = st[ev]["seen_at"]
        return self.out


def model_rule(margin: float):
    return lambda pos, snap, e, p_side: p_side is not None and e > p_side + margin


RULES = {
    "hold": lambda pos, snap, e, p_side: False,
    "model, sell above value": model_rule(0.0),
    "model, sell 2 pts above value": model_rule(0.02),
    "take profit +15c": lambda pos, snap, e, p_side: e >= pos["paid"] + TP,
    "take profit +15c / stop -15c": lambda pos, snap, e, p_side: abs(e - pos["paid"]) >= TP,
}


class Sell:
    """Consumer: entries as FirstBets under `pol`; state[rule][event] = {held, cash, sells, pnl}
    after every later poll's exit offers."""

    def __init__(self, pol: Policy, rules: dict = RULES):
        self.pol, self.rules = pol, rules
        self.pos: dict[str, dict] = {}
        self.state: dict[str, dict] = {k: {} for k in rules}

    def step(self, p, rows, books: Books) -> None:
        pol, rules, pos, state = self.pol, self.rules, self.pos, self.state
        new = [r for r in rows if r["event"] not in pos]
        if new:
            for ev, d in redecide(new, books.get(), pol).items():
                if d.sizing.contracts > 0:
                    pos[ev] = _entry(d, next(r for r in new if r["event"] == ev), p)
                    for k in rules:
                        state[k][ev] = {"held": d.sizing.contracts, "cash": 0.0, "sells": []}
        for r in rows:
            ev = r["event"]
            if ev not in pos or pos[ev]["poll_id"] == p["poll_id"] or r["status"] in ("closed", "cancelled"):
                continue
            if all(state[k][ev]["held"] <= 0 for k in rules):
                continue                                 # sold out under every rule: nothing to price
            side = pos[ev]["side"]
            own, opp = (r["a_ticker"], r["b_ticker"]) if side == "a" else (r["b_ticker"], r["a_ticker"])
            bk = books.get()
            bo, bp = bk.get(own), bk.get(opp)
            if bo is None or bp is None:
                continue
            exits = kalshi.ladder(bp, opp, bo, own)          # ways to buy the other side, cheapest first
            p_side = None
            if not r["skip"] and r["p_model"] is not None and not r["stale"]:
                pu = kalshi.blend(r["p_model"], r["p_market"], pol.model_weight)
                p_side = pu if side == "a" else 1 - pu
            for k, rule in rules.items():
                s = state[k][ev]
                for lv in exits:
                    if s["held"] <= 0:
                        break
                    e = 1 - lv.price - kalshi.fee_per_contract(lv.price, pol.fee_rate)
                    if not rule(pos[ev], r, e, p_side):
                        break
                    n = min(s["held"], int(lv.size))
                    if n <= 0:
                        continue
                    s["held"] -= n
                    s["cash"] += n * e
                    s["sells"].append({"as_of": p["as_of"], "status": r["status"], "contracts": n,
                                       "value": round(e, 4), "score": r["score"]})

    def finish(self, st: dict) -> tuple[dict, dict]:
        for ev, e in self.pos.items():
            e["won"] = st[ev]["winner"] == e["side"]
            e["settled_at"] = st[ev]["seen_at"]
            for k in self.rules:
                s = self.state[k][ev]
                s["pnl"] = s["cash"] + (s["held"] if e["won"] else 0) - e["cost"]
        return self.pos, self.state


def replay(con: sqlite3.Connection, consumers: list, st: dict | None = None) -> list:
    """One pass over the polls of settled matches, feeding every consumer; their results."""
    st = settled(con) if st is None else st
    for p, rows, books in polls(con, set(st)):
        for c in consumers:
            c.step(p, rows, books)
    return [c.finish(st) for c in consumers]


def first_bets(con: sqlite3.Connection, policy_for) -> dict[str, dict]:
    """event -> its first flagged bet, `policy_for(watch_id)` giving the Policy per poll."""
    return replay(con, [FirstBets(policy_for)])[0]


def sell_replay(con: sqlite3.Connection, pol: Policy, rules: dict = RULES) -> tuple[dict, dict]:
    """(entries, state): entries as first_bets under `pol`; state[rule][event] =
    {held, cash, sells, pnl} after every later poll's exit offers."""
    return replay(con, [Sell(pol, rules)])[0]


def variants(con: sqlite3.Connection, overrides: dict) -> dict[str, Callable[[int], Policy]]:
    """The two pnl policies: each watch's own, and the newest watch's throughout, both with
    the viewer's `overrides` (bankroll / kelly / model_weight, clamped by policy_with)."""
    pols = policies(con)
    newest = pols[max(pols)]
    return {
        "as recommended (each watch's policy)": lambda wid: policy_with(pols[wid], overrides),
        "newest watch's policy throughout": lambda wid: policy_with(newest, overrides),
    }


def newest_policy(con: sqlite3.Connection, overrides: dict) -> Policy:
    pols = policies(con)
    return policy_with(pols[max(pols)], overrides)


def totals(bets) -> dict[str, dict]:
    """'pre' / 'in play' / 'all' -> n, won, cost, ev, pnl (bets as first_bets values)."""
    tot: dict[str, dict] = {}
    for e in sorted(bets, key=lambda e: e["as_of"]):
        for key in ("all", "in play" if e["status"] == "live" else "pre"):
            t = tot.setdefault(key, {"n": 0, "won": 0, "cost": 0.0, "ev": 0.0, "pnl": 0.0})
            t["n"] += 1
            t["won"] += e["won"]
            t["cost"] += e["cost"]
            t["ev"] += e["ev"]
            t["pnl"] += e["pnl"]
    return tot


def stats(bets: list[dict]) -> dict:
    """Totals with honest error bars for one group of bets:
    roi_se     SE of ROI = sum pnl / sum cost (ratio estimator over bets)
    hit_se     binomial SE of the hit rate
    pnl_sd_model  SD of the P&L if the policy's own p_side were right (sum n^2 p (1-p)): the
               spread around `ev` that luck alone gives."""
    n = len(bets)
    cost = sum(e["cost"] for e in bets)
    pnl = sum(e["pnl"] for e in bets)
    won = sum(1 for e in bets if e["won"])
    out = {"n": n, "won": won, "cost": cost, "ev": sum(e["ev"] for e in bets), "pnl": pnl,
           "roi": pnl / cost if cost else None, "roi_se": None, "hit": won / n if n else None, "hit_se": None,
           "pnl_sd_model": math.sqrt(sum(e["n"] ** 2 * e["p_side"] * (1 - e["p_side"]) for e in bets)) if n else None}
    if n > 1 and cost:
        r = pnl / cost
        out["roi_se"] = math.sqrt(sum((e["pnl"] - r * e["cost"]) ** 2 for e in bets) * n / (n - 1)) / cost
        h = won / n
        out["hit_se"] = math.sqrt(h * (1 - h) / n)
    return out


def _tour(event: str) -> str:
    return TOUR_LABEL.get(event.split("-")[0], event.split("-")[0])


def unsettled_flagged(con: sqlite3.Connection) -> int:
    return con.execute("""SELECT COUNT(DISTINCT d.event) FROM decision d LEFT JOIN settlement s USING (event)
                          WHERE d.contracts > 0 AND s.event IS NULL""").fetchone()[0]


def pnl_view(con: sqlite3.Connection, bankroll: float = 1000.0, kelly: float | None = None,
             rules: dict = RULES) -> dict:
    """What /pnl shows, from one replay pass: per variant the bets (in settlement order), the
    bankroll after each, and stats (all / pre / in play); the sell rules on the newest
    policy's entries; the policies used. JSON-ready (no NaN)."""
    if not con.execute("SELECT 1 FROM watch LIMIT 1").fetchone():
        return {"bankroll": bankroll, "variants": [], "sell": None, "unsettled_flagged": 0, "settled": 0}
    ov = {"bankroll": bankroll} | ({} if kelly is None else {"kelly": kelly})
    vs = variants(con, ov)
    pol = newest_policy(con, ov)
    st = settled(con)
    fbs = [FirstBets(pf) for pf in vs.values()]
    *bets_by, (pos, state) = replay(con, fbs + [Sell(pol, rules)], st)
    out_vars = []
    for name, bets in zip(vs, bets_by):
        rows = sorted(bets.values(), key=lambda e: (e["settled_at"], e["as_of"], e["event"]))
        bank, curve = bankroll, [{"t": None, "bankroll": bankroll}]
        for e in rows:
            bank += e["pnl"]
            curve.append({"t": e["settled_at"], "bankroll": round(bank, 4), "event": e["event"]})
        out_vars.append({
            "name": name,
            "bets": [{**{k: e[k] for k in ("event", "side", "name", "opp", "n", "ask", "p_side", "cost", "ev",
                                           "won", "pnl", "as_of", "settled_at", "status", "final_score")},
                      "tour": _tour(e["event"]), "phase": "in play" if e["status"] == "live" else "pre"}
                     for e in rows],
            "curve": curve,
            "stats": {k: stats([e for e in rows if k == "all" or (k == "in play") == (e["status"] == "live")])
                      for k in ("all", "pre", "in play")},
        })
    hold = sum(s["pnl"] for s in state["hold"].values()) if "hold" in state else 0.0
    cost = sum(e["cost"] for e in pos.values())
    sell = {"entries": len(pos), "cost": cost, "rules": [
        {"rule": k, "pnl": (tot := sum(s["pnl"] for s in state[k].values())), "vs_hold": tot - hold,
         "roi": tot / cost if cost else None,
         "sold_in": sum(s["held"] < pos[ev]["n"] for ev, s in state[k].items())}
        for k in rules]}
    pols = policies(con)
    return {"bankroll": bankroll, "kelly": kelly, "settled": len(st),
            "policies": {"newest": vars(pol), "watches": {str(w): p for w, p in pols.items()}},
            "variants": out_vars, "sell": sell, "unsettled_flagged": unsettled_flagged(con)}
