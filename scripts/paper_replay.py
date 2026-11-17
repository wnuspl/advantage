"""Paper-trading replays on a watch's live.db (track B5): what following the flagged bets
would have made, and whether a selling rule beats holding to settlement.

The replay logic lives in src/tennis/engine/paper.py (shared with the web server's /pnl);
this script only prints it.

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
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tennis.engine.paper import (RULES, TP, first_bets, newest_policy, policies, polls,  # noqa: E402,F401
                                 sell_replay, settled, totals, unsettled_flagged, variants)


def connect(path) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def print_pnl(con, bankroll: float) -> None:
    for name, pf in variants(con, {"bankroll": bankroll}).items():
        bets = first_bets(con, pf)
        print(f"\n=== {name}, ${bankroll:g} bankroll: {len(bets)} settled bets ===")
        for e in sorted(bets.values(), key=lambda e: e["as_of"]):
            print(f"{e['as_of'][5:16]} {e['status'][:4]:4} {e['name'][:22]:22} {e['n']:4d} @ {e['ask']:.2f} "
                  f"p={e['p_side']:.2f} cost {e['cost']:7.2f} ev {e['ev']:+6.2f} "
                  f"{'WON ' if e['won'] else 'lost'} {e['pnl']:+8.2f}  [{e['final_score']}]")
        tot = totals(bets.values())
        for key in ("pre", "in play", "all"):
            t = tot.get(key)
            if t:
                print(f"  {key:7}: {int(t['n'])} bets, {int(t['won'])} won, staked ${t['cost']:.2f}, "
                      f"expected {t['ev']:+.2f}, actual {t['pnl']:+.2f} (ROI {100 * t['pnl'] / t['cost']:+.1f}%)")
    print("\nflagged matches not settled yet:", unsettled_flagged(con))


def print_sell(con, bankroll: float) -> None:
    pos, state = sell_replay(con, newest_policy(con, {"bankroll": bankroll}))
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
