"""Paper-trading replays on a watch's live.db (track B5): what following the flagged bets
would have made, and whether a selling rule beats holding to settlement.

Both re-decide on the order books recorded with each poll (engine.livefeed.redecide), at a
bankroll you choose, and count only matches with a settlement (a winner).

  pnl   - each settled match's FIRST flagged bet (one bet per match, placed when first
          recommended), under (1) the policy each watch was running and (2) the newest
          watch's policy throughout. $1 per contract if the side won; cost includes fees.
  sell  - the same entries under the newest watch's policy, then at every later poll an
          exit for the held "side wins" contracts: sell into their own market's bids, or
          buy the other side (YES on the opponent / NO on our player). Either way it is
          "buy the other side at q", worth 1 - q - fee(q) per contract now (the pair pays $1
          whoever wins). Rules: hold; model (sell while the exit beats our blended win
          probability by a margin: the condition under which decide() would flag the other
          side); take profit at +15c; take profit / stop loss at +-15c.

The server's live.db is in use by the watch; replay a copy (sqlite backup API):
  ssh <server> 'cd ~/advantage && .venv/bin/python -c "import sqlite3; s=sqlite3.connect(\\"file:live.db?mode=ro\\", uri=True); d=sqlite3.connect(\\"/tmp/live_copy.db\\"); s.backup(d)"'
  scp <server>:/tmp/live_copy.db <dir>/live.db
  python scripts/paper_replay.py <dir>/live.db pnl
  python scripts/paper_replay.py <dir>/live.db sell
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import zlib
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tennis import kalshi                                    # noqa: E402
from tennis.engine.decide import Policy                      # noqa: E402
from tennis.engine.livefeed import policy_with, redecide     # noqa: E402

TP = 0.15                       # take-profit / stop-loss distance, dollars per contract


def connect(path) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def settled(con) -> dict:
    return {r["event"]: r for r in con.execute("SELECT * FROM settlement WHERE winner IS NOT NULL")}


def policies(con) -> dict[int, dict]:
    return {r["watch_id"]: json.loads(r["policy"]) for r in con.execute("SELECT * FROM watch")}


def polls(con, events: set):
    """(poll row, snapshot rows of `events`, order books) per successful poll, in order."""
    for p in con.execute("SELECT poll_id, watch_id, as_of, books_sha FROM poll WHERE error IS NULL "
                         "ORDER BY poll_id").fetchall():
        rows = [r for r in con.execute("""SELECT s.*, e.a_name, e.b_name, e.a_ticker, e.b_ticker FROM snapshot s
                                          JOIN event e USING (event) WHERE s.poll_id = ?""", (p["poll_id"],))
                if r["event"] in events]
        if not rows:
            continue
        b = con.execute("SELECT zlib FROM blob WHERE sha256 = ?", (p["books_sha"],)).fetchone()
        yield p, rows, json.loads(zlib.decompress(b[0]).decode("utf-8")) if b else {}


def _entry(d, r, p) -> dict:
    return {"event": r["event"], "side": d.side, "n": d.sizing.contracts, "cost": d.sizing.cost,
            "ev": d.sizing.ev, "ask": d.ask, "p_side": d.p_side, "paid": d.sizing.cost / d.sizing.contracts,
            "as_of": p["as_of"], "poll_id": p["poll_id"], "status": r["status"],
            "name": r["a_name"] if d.side == "a" else r["b_name"]}


def first_bets(con, policy_for) -> dict[str, dict]:
    """event -> its first flagged bet, `policy_for(watch_id)` giving the Policy per poll."""
    st = settled(con)
    out: dict[str, dict] = {}
    for p, rows, books in polls(con, set(st)):
        new = [r for r in rows if r["event"] not in out]
        if not new:
            continue
        for ev, d in redecide(new, books, policy_for(p["watch_id"])).items():
            if d.sizing.contracts > 0:
                out[ev] = _entry(d, next(r for r in new if r["event"] == ev), p)
    for ev, e in out.items():
        e["won"] = st[ev]["winner"] == e["side"]
        e["pnl"] = (e["n"] if e["won"] else 0) - e["cost"]
        e["final_score"] = st[ev]["final_score"]
    return out


def model_rule(margin: float):
    return lambda pos, snap, e, p_side: p_side is not None and e > p_side + margin


RULES = {
    "hold": lambda pos, snap, e, p_side: False,
    "model, sell above value": model_rule(0.0),
    "model, sell 2 pts above value": model_rule(0.02),
    "take profit +15c": lambda pos, snap, e, p_side: e >= pos["paid"] + TP,
    "take profit +15c / stop -15c": lambda pos, snap, e, p_side: abs(e - pos["paid"]) >= TP,
}


def sell_replay(con, pol: Policy, rules: dict = RULES) -> tuple[dict, dict]:
    """(entries, state): entries as first_bets under `pol`; state[rule][event] =
    {held, cash, sells, pnl} after every later poll's exit offers."""
    st = settled(con)
    pos: dict[str, dict] = {}
    state: dict[str, dict] = {k: {} for k in rules}
    for p, rows, books in polls(con, set(st)):
        new = [r for r in rows if r["event"] not in pos]
        if new:
            for ev, d in redecide(new, books, pol).items():
                if d.sizing.contracts > 0:
                    pos[ev] = _entry(d, next(r for r in new if r["event"] == ev), p)
                    for k in rules:
                        state[k][ev] = {"held": d.sizing.contracts, "cash": 0.0, "sells": []}
        for r in rows:
            ev = r["event"]
            if ev not in pos or pos[ev]["poll_id"] == p["poll_id"] or r["status"] in ("closed", "cancelled"):
                continue
            side = pos[ev]["side"]
            own, opp = (r["a_ticker"], r["b_ticker"]) if side == "a" else (r["b_ticker"], r["a_ticker"])
            bo, bp = books.get(own), books.get(opp)
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
    for ev, e in pos.items():
        e["won"] = st[ev]["winner"] == e["side"]
        for k in rules:
            s = state[k][ev]
            s["pnl"] = s["cash"] + (s["held"] if e["won"] else 0) - e["cost"]
    return pos, state


def print_pnl(con, bankroll: float) -> None:
    pols = policies(con)
    newest = pols[max(pols)]
    variants = {
        "as recommended (each watch's policy)": lambda wid: policy_with(pols[wid], {"bankroll": bankroll}),
        "newest watch's policy throughout": lambda wid: policy_with(newest, {"bankroll": bankroll}),
    }
    for name, pf in variants.items():
        bets = first_bets(con, pf)
        print(f"\n=== {name}, ${bankroll:g} bankroll: {len(bets)} settled bets ===")
        tot = defaultdict(lambda: defaultdict(float))
        for e in sorted(bets.values(), key=lambda e: e["as_of"]):
            print(f"{e['as_of'][5:16]} {e['status'][:4]:4} {e['name'][:22]:22} {e['n']:4d} @ {e['ask']:.2f} "
                  f"p={e['p_side']:.2f} cost {e['cost']:7.2f} ev {e['ev']:+6.2f} "
                  f"{'WON ' if e['won'] else 'lost'} {e['pnl']:+8.2f}  [{e['final_score']}]")
            for key in ("all", "in play" if e["status"] == "live" else "pre"):
                t = tot[key]
                t["n"] += 1
                t["won"] += e["won"]
                t["cost"] += e["cost"]
                t["ev"] += e["ev"]
                t["pnl"] += e["pnl"]
        for key in ("pre", "in play", "all"):
            t = tot.get(key)
            if t:
                print(f"  {key:7}: {int(t['n'])} bets, {int(t['won'])} won, staked ${t['cost']:.2f}, "
                      f"expected {t['ev']:+.2f}, actual {t['pnl']:+.2f} (ROI {100 * t['pnl'] / t['cost']:+.1f}%)")
    print("\nflagged matches not settled yet:",
          con.execute("""SELECT COUNT(DISTINCT d.event) FROM decision d LEFT JOIN settlement s USING (event)
                         WHERE d.contracts > 0 AND s.event IS NULL""").fetchone()[0])


def print_sell(con, bankroll: float) -> None:
    pols = policies(con)
    pol = policy_with(pols[max(pols)], {"bankroll": bankroll})
    pos, state = sell_replay(con, pol)
    print(f"{len(pos)} settled entries (newest watch's policy, ${bankroll:g}); P&L per rule:\n")
    names = list(RULES)
    for i, k in enumerate(names):
        print(f"  [{i}] {k}")
    for ev, e in sorted(pos.items(), key=lambda x: x[1]["as_of"]):
        cells = " ".join(f"{state[k][ev]['pnl']:+8.2f}" for k in names)
        print(f"{e['as_of'][5:16]} {e['name'][:20]:20} {e['n']:4d} paid {e['paid']:.2f} "
              f"{'WON ' if e['won'] else 'lost'} {cells}")
    hold = sum(s["pnl"] for s in state["hold"].values())
    print()
    for k in names:
        tot = sum(s["pnl"] for s in state[k].values())
        n_sold = sum(s["held"] < pos[ev]["n"] for ev, s in state[k].items())
        print(f"{k:30} P&L {tot:+8.2f}  vs hold {tot - hold:+7.2f}  sold in {n_sold} matches")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("live_db", help="a copy of the watch's live.db")
    ap.add_argument("what", choices=("pnl", "sell"))
    ap.add_argument("--bankroll", type=float, default=1000.0)
    a = ap.parse_args(argv)
    con = connect(a.live_db)
    try:
        (print_pnl if a.what == "pnl" else print_sell)(con, a.bankroll)
    finally:
        con.close()


if __name__ == "__main__":
    main()
