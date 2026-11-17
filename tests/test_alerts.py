"""Track B4: change detection, alert limits, ntfy publishing, recording, and the watch hook,
on the saved Kalshi fixture. ntfy is faked: nothing leaves the machine."""
from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

from tennis.engine import Prediction                                   # noqa: E402
from tennis.engine.alerts import (AlertConfig, Alerter, ChangeDetector,  # noqa: E402
                                  NtfyNotifier, env_value, format_alert)
from tennis.engine.decide import Policy, decide_all                    # noqa: E402
from tennis.engine.live import load_pull, snapshots                    # noqa: E402
from tennis.engine.recorder import Recorder                            # noqa: E402
from test_live import FIX, NoNetwork, resolver                         # noqa: E402
from test_recorder import load_watch                                   # noqa: E402

ALC = "KXATPMATCH-26NOV16ALCTIA"              # Alcaraz (A, players 1) v Tiafoe (B, 2); market ~0.885 on A
POL = Policy(in_play=True, allow_stale=True)
T0 = datetime(2026, 11, 15, 22, 31, 32, tzinfo=timezone.utc)


class ByPair:
    """P(A) = `p` for Alcaraz v Tiafoe (players 1, 2), 0.5 for every other match (most
    tests keep only the Alcaraz row)."""
    name = "bypair"

    def __init__(self, p, last_a=None):
        self.p, self.last_a = p, last_a

    def predict(self, q):
        p = self.p if (q.player_a, q.player_b) == (1, 2) else 0.5
        det = {"last_a": self.last_a, "last_b": "2026-11-10"} if self.last_a else {}
        return Prediction(p_a=p, model=self.name, detail=det)


class FakeNtfy:
    def __init__(self, fail=False):
        self.sent, self.fail = [], fail

    def send(self, alert):
        if self.fail:
            raise OSError("ntfy down")
        self.sent.append(alert)


class Scenario(NoNetwork):
    def setUp(self):
        super().setUp()
        self.pull = load_pull(FIX)

    def poll(self, p, only_alc=True, minutes=0, last_a=None):
        pull = copy.copy(self.pull)
        pull.as_of = (T0 + timedelta(minutes=minutes)).isoformat(timespec="seconds")
        rows = snapshots(pull, ByPair(p, last_a), resolver())
        if only_alc:
            rows = [r for r in rows if r.event.event == ALC]
        return pull.as_of, rows, decide_all(rows, pull.books, POL)

    def stake(self, p):
        return next(d for d in self.poll(p)[2] if d.event == ALC).sizing.cost


class DetectorTest(Scenario):
    def test_new_grew_side_gone(self):
        self.assertTrue(5 <= self.stake(0.93) and self.stake(0.99) >= 1.5 * self.stake(0.93))   # preconditions
        det = ChangeDetector(AlertConfig())

        def kinds(p, **kw):
            _, rows, ds = self.poll(p, **kw)
            chs = det.update(rows, ds)
            for c in chs:
                det.mark(c)
            return [c.kind for c in chs]
        self.assertEqual(kinds(0.93), ["new"])
        self.assertEqual(kinds(0.93), [])                     # unchanged: nothing
        self.assertEqual(kinds(0.94), [])                     # small growth: nothing
        self.assertEqual(kinds(0.99), ["grew"])
        self.assertEqual(kinds(0.99), [])
        self.assertEqual(kinds(0.5), ["side"])                # now Tiafoe at ~0.12
        self.assertEqual(kinds(0.885), ["gone"])              # model = market: no edge
        self.assertEqual(kinds(0.885), [])

    def test_small_bets_and_closed_matches(self):
        det = ChangeDetector(AlertConfig(min_stake=1e6))
        _, rows, ds = self.poll(0.99)
        self.assertEqual(det.update(rows, ds), [])            # below min_stake: not actionable
        det = ChangeDetector(AlertConfig())
        _, rows, ds = self.poll(0.99)
        for c in det.update(rows, ds):
            det.mark(c)
        self.assertEqual(det.update([], []), [])              # match left the open list: no 'gone', state dropped
        self.assertEqual(det.alerted, {})


class AlerterTest(Scenario):
    def run_polls(self, alerter, ps, gap_min=1):
        out = []
        for k, p in enumerate(ps):
            as_of, rows, ds = self.poll(p, minutes=k * gap_min)
            out.append([(a.kind, a.status, a.detail) for a in alerter.process(as_of, rows, ds)])
        return out

    def test_sent_and_cooldown(self):
        n = FakeNtfy()
        al = Alerter(ChangeDetector(AlertConfig(cooldown_s=900)), n)
        got = self.run_polls(al, [0.93, 0.99, 0.885])
        self.assertEqual(got[0], [("new", "sent", "")])
        self.assertEqual(got[1], [("grew", "suppressed", "cooldown (900s per match)")])
        self.assertEqual(got[2], [("gone", "sent", "")])      # gone bypasses the cooldown
        self.assertEqual(len(n.sent), 2)
        al = Alerter(ChangeDetector(AlertConfig(cooldown_s=900)), FakeNtfy())
        got = self.run_polls(al, [0.93, 0.99], gap_min=20)     # after the cooldown: sent
        self.assertEqual(got[1], [("grew", "sent", "")])

    def test_gone_only_for_pushed_bets(self):
        al = Alerter(ChangeDetector(AlertConfig(max_per_hour=0)), FakeNtfy())
        got = self.run_polls(al, [0.99, 0.885])
        self.assertEqual(got[0][0][:2], ("new", "suppressed"))
        self.assertEqual(got[1], [("gone", "suppressed", "its bet was never pushed")])

    def test_hourly_cap(self):
        n = FakeNtfy()
        al = Alerter(ChangeDetector(AlertConfig(max_per_hour=2)), n)
        as_of, rows, ds = self.poll(0.99, only_alc=False)       # p 0.5 elsewhere: underdog bets everywhere
        alerts = al.process(as_of, rows, ds)
        self.assertGreater(len(alerts), 2)
        self.assertEqual([a.status for a in alerts].count("sent"), 2)
        self.assertTrue(all(a.detail == "hourly cap (2)" for a in alerts if a.status == "suppressed"))
        evs = [a.change.decision.sizing.ev for a in alerts if a.kind == "new"]
        self.assertEqual(evs, sorted(evs, reverse=True))           # the largest EV goes out first
        later = (T0 + timedelta(minutes=61)).isoformat(timespec="seconds")
        _, rows, ds = self.poll(0.5, only_alc=False)               # Alcaraz -> Tiafoe: a 'side' change
        side = [a for a in al.process(later, rows, ds) if a.kind == "side"]
        self.assertEqual([a.status for a in side], ["sent"])       # the hour rolled over

    def test_failure_and_print_only(self):
        al = Alerter(ChangeDetector(AlertConfig()), FakeNtfy(fail=True))
        got = self.run_polls(al, [0.99])
        self.assertEqual(got[0], [("new", "failed", "OSError: ntfy down")])
        al = Alerter(ChangeDetector(AlertConfig()), None)
        self.assertEqual(self.run_polls(al, [0.99, 0.885]), [[("new", "printed", "")], [("gone", "printed", "")]])


class FormatTest(Scenario):
    def test_message(self):
        det = ChangeDetector(AlertConfig())
        _, rows, ds = self.poll(0.99, last_a="2026-05-20")
        a = format_alert(det.update(rows, ds)[0], AlertConfig(priority_ev=1e9))
        d = a.change.decision
        self.assertTrue(a.title.startswith(f"BET Carlos Alcaraz {d.sizing.contracts} @"))
        self.assertIn(f"(${d.sizing.cost:.0f})", a.title)
        self.assertIn("Carlos Alcaraz v Frances Tiafoe · ATP World Tour Finals Men Singles · 2026-11-16 16:00 UTC",
                      a.message)
        self.assertIn("check: A last rated 180d ago; surface unknown", a.message)
        self.assertRegex(a.message, r"fills via (YES ALC|NO TIA)")   # either route buys "Alcaraz wins"
        self.assertEqual((a.priority, a.tags), (3, ["tennis"]))
        self.assertEqual(format_alert(a.change, AlertConfig(priority_ev=0)).priority, 4)


class NtfyTest(unittest.TestCase):
    def test_publish_json(self):
        calls = []
        n = NtfyNotifier("adv-x1y2", "https://ntfy.example/", token="tk_1",
                         post=lambda url, body, headers: calls.append((url, body, headers)))
        from tennis.engine.alerts import Alert
        n.send(Alert(change=None, title="BET Dominik Palán 5 @0.38 ($2)", message="line 1\nline 2",   # type: ignore[arg-type]
                     priority=4, tags=["tennis"]))
        url, body, headers = calls[0]
        self.assertEqual(url, "https://ntfy.example/")
        self.assertEqual(json.loads(body.decode("utf-8")),
                         {"topic": "adv-x1y2", "title": "BET Dominik Palán 5 @0.38 ($2)", "message": "line 1\nline 2",
                          "priority": 4, "tags": ["tennis"]})
        self.assertEqual(headers, {"Content-Type": "application/json", "Authorization": "Bearer tk_1"})
        with self.assertRaises(ValueError):
            NtfyNotifier("")

    def test_env_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / ".env"
            p.write_text('# x\nACESAPI_KEY=abc\nexport NTFY_TOPIC="adv-secret"\n', encoding="utf-8")
            old = os.environ.pop("NTFY_TOPIC", None)
            try:
                self.assertEqual(env_value("NTFY_TOPIC", p), "adv-secret")
                self.assertIsNone(env_value("NOPE", p))
                os.environ["NTFY_TOPIC"] = "from-env"
                self.assertEqual(env_value("NTFY_TOPIC", p), "from-env")
            finally:
                os.environ.pop("NTFY_TOPIC", None)
                if old is not None:
                    os.environ["NTFY_TOPIC"] = old


class WatchHookTest(Scenario):
    def test_poll_once_records_and_prints_alerts(self):
        watch = load_watch()
        pulls = iter([copy.deepcopy(self.pull), copy.deepcopy(self.pull)])
        watch.fetch_pull = lambda series, known_milestones=None: next(pulls)
        watch.kalshi.get = lambda *a, **k: {}
        watch.kalshi.live_batch = lambda ids: {}
        p = {"v": 0.99}

        class Retr:
            def __call__(self, pull):
                return [r for r in snapshots(pull, ByPair(p["v"]), resolver()) if r.event.event == ALC]
        rec = Recorder(":memory:")
        wid = rec.start_watch("bypair", POL, ["x"], 60)
        n = FakeNtfy()
        al = Alerter(ChangeDetector(AlertConfig()), n)
        line1 = watch.poll_once(rec, wid, Retr(), POL, ("x",), {}, al)
        p["v"] = 0.885
        self.pull.as_of = "2026-11-15T22:40:00+00:00"
        line2 = watch.poll_once(rec, wid, Retr(), POL, ("x",), {}, al)
        self.assertIn("BET Carlos Alcaraz", line1)
        self.assertIn("[sent]", line1)
        self.assertIn("GONE Alcaraz vs Tiafoe", line2)
        self.assertIn("(holders: hold)", line2)
        rows = [dict(r) for r in rec.con.execute("SELECT kind, status, event FROM alert ORDER BY alert_id")]
        self.assertEqual([(r["kind"], r["status"]) for r in rows], [("new", "sent"), ("gone", "sent")])
        self.assertEqual(rec.counts()["alerts_sent"], 2)
        self.assertEqual(len(n.sent), 2)
        rec.close()


if __name__ == "__main__":
    unittest.main()
