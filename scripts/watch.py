"""Watch Kalshi tennis markets and record every poll in live.db (track B2, docs/PLAN.md).
Read-only against Kalshi: nothing here places orders.

    python scripts/watch.py run --predictor points --every 60 --hours 6   # record (Ctrl+C stops)
    python scripts/watch.py run --ntfy                                     # + push alerts (topic: NTFY_TOPIC in .env)
    python scripts/watch.py test-alert                                     # send one test push
    python scripts/watch.py status                                         # what live.db holds
    python scripts/watch.py at KXATPMATCH-26NOV16ALCTIA --time 2026-11-16T12:00:00Z

Each poll: open events of every match series, milestone links (requested once per event),
live scores, full order books; the B1 snapshots, the predictor's prediction and the B3a
decision per match; all written in one transaction (engine/recorder.py, live_schema.sql).
Matches that leave the open list (or show closed) are settled from Kalshi's market results
plus the last live details (how the match ended, final score). A failed poll is recorded
with its error and the loop carries on, backing off.

Alerts (engine/alerts.py): every poll, each match's decision is compared with the last
one; changes (new bet, side flipped, stake grew, alerted bet gone) are printed and
recorded in live.db's `alert` table. With --ntfy they are also pushed, within a per-match
cooldown and an hourly cap.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tennis import kalshi                                                     # noqa: E402
from tennis.engine.alerts import (NTFY_SERVER, Alert, AlertConfig, Alerter,   # noqa: E402
                                  ChangeDetector, NtfyNotifier, env_value)
from tennis.engine.decide import add_policy_args, decide_all, policy_from_args   # noqa: E402
from tennis.engine.live import Retriever, fetch_pull, load_predictor          # noqa: E402
from tennis.engine.recorder import LIVE_DB, Recorder                          # noqa: E402

MAX_SETTLE = 20          # settlement lookups per poll


def _rev() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                              timeout=10).stdout.strip() or None
    except Exception:
        return None


def poll_once(rec: Recorder, wid: int, retr: Retriever, policy, series, ms_cache: dict,
              alerter: Alerter | None = None) -> str:
    """One poll + settlement checks. Returns a one-line summary."""
    t0 = time.monotonic()
    pull = fetch_pull(series, known_milestones=ms_cache)
    ms_cache.update({k: v for k, v in pull.milestones.items() if v})
    rows = retr(pull)
    ds = decide_all(rows, pull.books, policy)
    poll_id = rec.record_poll(wid, pull, rows, ds)
    alerts = alerter.process(pull.as_of, rows, ds) if alerter else []
    if alerts:
        rec.record_alerts(poll_id, alerts)
    settled = []
    todo = rec.to_settle({r.event.event for r in rows})[:MAX_SETTLE]
    if todo:
        mids = [t["milestone"] for t in todo if t["milestone"]]
        live = kalshi.live_batch(mids) if mids else {}
        for t in todo:
            try:
                st = rec.settle(t, kalshi.get(f"/events/{t['event']}"), live.get(t["milestone"]))
            except Exception as e:                       # one bad lookup must not stop the watch
                print(f"  settle {t['event']}: {type(e).__name__}: {e}")
                continue
            if st:
                w = t["a_name"] if st["winner"] == "a" else t["b_name"] if st["winner"] == "b" else "no winner"
                settled.append(f"{t['title']}: {w} ({st['match_status'] or '?'}, {st['final_score'] or '?'})")
    n_live = sum(r.status == "live" for r in rows)
    bets = [d for d in ds if d.bet]
    line = (f"{pull.as_of[11:19]}  {len(rows)} open, {n_live} live, {sum(not r.skip for r in rows)} priced, "
            f"{len(bets)} bets (${sum(d.sizing.cost for d in bets):,.0f}), {len(todo)} to settle, "
            f"{time.monotonic() - t0:.1f}s")
    for a in alerts:
        line += f"\n  {a.title}  [{a.status}{': ' + a.detail if a.detail else ''}]"
    for s in settled:
        line += f"\n  settled {s}"
    return line


def alert_config(args) -> AlertConfig:
    return AlertConfig(min_stake=args.alert_min_stake, min_ev=args.alert_min_ev, grow_ratio=args.alert_grow,
                       cooldown_s=args.alert_cooldown, max_per_hour=args.alert_max_per_hour)


def notifier(args) -> NtfyNotifier:
    topic = args.ntfy_topic or env_value("NTFY_TOPIC")
    if not topic:
        raise SystemExit("no ntfy topic: set NTFY_TOPIC in .env (or pass --ntfy-topic)")
    return NtfyNotifier(topic, args.ntfy_server, env_value("NTFY_TOKEN"))


def run(args) -> int:
    policy = policy_from_args(args)
    series = tuple(args.series.split(","))
    today = datetime.now(timezone.utc).date().isoformat()
    predictor = load_predictor(args.predictor, fixed_p=args.fixed_p, run=args.run, raw=args.raw,
                               data_path=args.data, today=today)
    con = sqlite3.connect(f"file:{args.data.resolve().as_posix()}?mode=ro", uri=True)
    retr = Retriever(predictor, con, today, args.surface, args.include_started)
    rec = Recorder(args.db)
    st = args.data.stat()
    wid = rec.start_watch(predictor.name, policy, series, args.every, f"{st.st_size}:{int(st.st_mtime)}", _rev())
    ms_cache = rec.milestones()
    alerter = Alerter(ChangeDetector(alert_config(args)), notifier(args) if args.ntfy else None)
    stop_at = datetime.now(timezone.utc) + timedelta(hours=args.hours) if args.hours else None
    print(f"watch {wid}: {predictor.name}, every {args.every:g}s"
          + (f" until {stop_at:%Y-%m-%d %H:%M} UTC" if stop_at else "") + f", recording to {args.db}"
          + (", pushing alerts to ntfy" if args.ntfy else ", alerts printed only"))
    fails = 0
    try:
        while stop_at is None or datetime.now(timezone.utc) < stop_at:
            t0 = time.monotonic()
            as_of = datetime.now(timezone.utc).isoformat(timespec="seconds")
            try:
                print(poll_once(rec, wid, retr, policy, series, ms_cache, alerter), flush=True)
                fails = 0
            except Exception as e:                       # network, rate limit, ...: record and back off
                fails += 1
                rec.record_error(wid, as_of, f"{type(e).__name__}: {e}")
                print(f"{as_of[11:19]}  poll failed ({fails} in a row): {type(e).__name__}: {e}", flush=True)
            wait = args.every * min(2 ** fails, 10) if fails else args.every
            time.sleep(max(0.0, wait - (time.monotonic() - t0)))
    except KeyboardInterrupt:
        print("stopped")
    finally:
        print(json.dumps(rec.counts()))
        rec.close()
        con.close()
    return 0


def test_alert(args) -> int:
    """One push to the configured topic, to check the phone receives it."""
    n = notifier(args)
    n.send(Alert(change=None, title="advantage: test alert",          # type: ignore[arg-type]
                 message="If this reached your phone, watch alerts will too. Read-only: no orders are placed.",
                 tags=["white_check_mark"]))
    print(f"sent to {n.server}/<topic>")
    return 0


def at(args) -> int:
    rec = Recorder(args.db)
    t = args.time or datetime.now(timezone.utc).isoformat(timespec="seconds")
    if t.endswith("Z"):
        t = t[:-1] + "+00:00"
    r = rec.at(args.event, t)
    rec.close()
    if r is None:
        print(f"{args.event}: nothing recorded at or before {t}")
        return 1
    print(json.dumps(r, indent=1, ensure_ascii=False, default=str))
    return 0


def status(args) -> int:
    rec = Recorder(args.db)
    print(json.dumps(rec.counts(), indent=1))
    rec.close()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=LIVE_DB)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="poll and record until --hours or Ctrl+C")
    r.add_argument("--every", type=float, default=60.0, help="seconds between polls")
    r.add_argument("--hours", type=float, default=None, help="stop after this long (default: never)")
    r.add_argument("--predictor", choices=["fixed", "points"], default="points")
    r.add_argument("--fixed-p", type=float, default=0.5)
    r.add_argument("--run", type=int, default=790)
    r.add_argument("--raw", action="store_true")
    r.add_argument("--series", default=",".join(kalshi.MATCH_SERIES))
    r.add_argument("--surface", choices=["hard", "clay", "grass", "carpet"])
    r.add_argument("--include-started", action="store_true")
    r.add_argument("--data", type=Path, default=ROOT / "tennis.db")
    add_policy_args(r)
    g = r.add_argument_group("alerts (engine/alerts.py)")
    g.add_argument("--ntfy", action="store_true", help="push alerts with ntfy (otherwise printed and recorded only)")
    g.add_argument("--alert-min-stake", type=float, default=AlertConfig.min_stake, help="dollars")
    g.add_argument("--alert-min-ev", type=float, default=AlertConfig.min_ev, help="dollars")
    g.add_argument("--alert-grow", type=float, default=AlertConfig.grow_ratio,
                   help="re-alert when the stake grows by this factor")
    g.add_argument("--alert-cooldown", type=float, default=AlertConfig.cooldown_s, help="seconds per match")
    g.add_argument("--alert-max-per-hour", type=int, default=AlertConfig.max_per_hour)
    r.set_defaults(fn=run)
    ta = sub.add_parser("test-alert", help="send one test push to the ntfy topic")
    ta.set_defaults(fn=test_alert)
    for sp in (r, ta):
        sp.add_argument("--ntfy-topic", help="default: NTFY_TOPIC from the environment or .env")
        sp.add_argument("--ntfy-server", default=NTFY_SERVER)
    a = sub.add_parser("at", help="price, score, model and decision of one event at a time")
    a.add_argument("event")
    a.add_argument("--time", help="UTC ISO time (default: now)")
    a.set_defaults(fn=at)
    s = sub.add_parser("status", help="what live.db holds")
    s.set_defaults(fn=status)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
