"""scripts/paper_replay.py on a fixture live.db: two identical polls of the saved Kalshi
pull, every match settled with side A winning."""
from __future__ import annotations

import contextlib
import io
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE.parent / "scripts"))
sys.path.insert(0, str(HERE))

from tennis.engine.decide import Policy, decide_all                     # noqa: E402
from tennis.engine.live import load_pull, snapshots                     # noqa: E402
from tennis.engine.livefeed import policy_with                          # noqa: E402
from tennis.engine.recorder import Recorder                             # noqa: E402
from test_live import FIX, NoNetwork, resolver                          # noqa: E402
from test_livefeed import Spread                                        # noqa: E402

import paper_replay                                                     # noqa: E402

POL = Policy(bankroll=100, model_weight=0.6, in_play=True)


def a_wins(ev: dict) -> dict:
    return {"markets": [
        {"ticker": ev["a_ticker"], "status": "finalized", "result": "yes", "settlement_value_dollars": "1.0000"},
        {"ticker": ev["b_ticker"], "status": "finalized", "result": "no", "settlement_value_dollars": "0.0000"}]}


class PaperReplayTest(NoNetwork):
    def setUp(self):
        super().setUp()
        self.tmp = Path(tempfile.mkdtemp())
        self.db = self.tmp / "live.db"
        pull = load_pull(FIX)
        rec = Recorder(self.db)
        wid = rec.start_watch("spread", POL, ["KXATPMATCH"], 20)
        rows = snapshots(pull, Spread(), resolver())
        for _ in range(2):                                   # prices unchanged between the polls
            rec.record_poll(wid, pull, rows, decide_all(rows, pull.books, POL))
        for ev in rec.con.execute("SELECT * FROM event").fetchall():
            rec.settle(dict(ev), a_wins(dict(ev)), None, seen_at="2026-11-16T00:00:00+00:00")
        rec.close()
        self.con = paper_replay.connect(self.db)
        self.pol = policy_with(paper_replay.policies(self.con)[wid], {"bankroll": 1000})

    def tearDown(self):
        self.con.close()
        shutil.rmtree(self.tmp, ignore_errors=True)
        super().tearDown()

    def test_first_bets_settle_at_a_dollar(self):
        bets = paper_replay.first_bets(self.con, lambda wid: self.pol)
        self.assertTrue(bets)
        for e in bets.values():
            self.assertEqual(e["won"], e["side"] == "a")
            self.assertAlmostEqual(e["pnl"], (e["n"] if e["side"] == "a" else 0) - e["cost"])
            self.assertEqual(e["poll_id"], 1)                # placed when first flagged
        small = paper_replay.first_bets(self.con, lambda wid: policy_with(paper_replay.policies(self.con)[wid], {}))
        self.assertLessEqual(sum(e["n"] for e in small.values()), sum(e["n"] for e in bets.values()))

    def test_unchanged_prices_never_sell(self):
        """Exiting = buying the other side, which costs more than the bet is worth when the
        book hasn't moved: no rule sells, and holding equals the first-bet P&L."""
        pos, state = paper_replay.sell_replay(self.con, self.pol)
        bets = paper_replay.first_bets(self.con, lambda wid: self.pol)
        self.assertEqual(set(pos), set(bets))
        for ev, e in bets.items():
            self.assertAlmostEqual(state["hold"][ev]["pnl"], e["pnl"])
            for k in paper_replay.RULES:
                self.assertEqual(state[k][ev]["sells"], [], k)
                self.assertEqual(state[k][ev]["held"], e["n"])

    def test_cli(self):
        for what in ("pnl", "sell"):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                paper_replay.main([str(self.db), what])
            self.assertIn("P&L" if what == "sell" else "settled bets", out.getvalue())


if __name__ == "__main__":
    import unittest
    unittest.main()
