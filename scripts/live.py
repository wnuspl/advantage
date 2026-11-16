"""Open Kalshi tennis match markets vs a predictor, and what the policy would bet
(tracks B1 + B3a, docs/PLAN.md). Read-only: nothing here places orders.

    python scripts/live.py                                  # pull now, FixedPredictor(0.5)
    python scripts/live.py --pull data/raw/kalshi/20261115T223132Z   # reprice a saved pull, no network
    python scripts/live.py --predictor points --bankroll 10 --kelly 0.25 --model-weight 0.5
    python scripts/live.py --predictor fixed --fixed-p 0.6 --surface hard --json out.json

Each poll is saved raw under data/raw/kalshi/<UTC>/ (events, milestones, live scores,
order books). Every open match is listed: priced rows sorted by |model - market|, each
with the bet the policy would place or why it places none (engine/decide.py), then the
ones not priced, each with its reason. `--predictor points` is the run-790 / points_v2
predictor (engine/predict.py, `--run`, `--raw`). FixedPredictor is a placeholder and is
never bet on.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tennis import kalshi                                         # noqa: E402
from tennis.engine.decide import add_policy_args, decide_all, policy_from_args   # noqa: E402
from tennis.engine.live import (RAW_KALSHI, Retriever, fetch_pull, load_predictor,   # noqa: E402
                                load_pull, parse_events, save_pull)


def fmt(x, spec):
    return "-" if x is None else format(x, spec)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictor", choices=["fixed", "points"], default="fixed")
    ap.add_argument("--fixed-p", type=float, default=0.5, help="FixedPredictor's P(A wins)")
    ap.add_argument("--run", type=int, default=790, help="points run (models.db) for --predictor points")
    ap.add_argument("--raw", action="store_true", help="points predictor without calibration")
    ap.add_argument("--series", default=",".join(kalshi.MATCH_SERIES))
    ap.add_argument("--pull", type=Path, help="reprice a saved pull folder instead of fetching")
    ap.add_argument("--surface", choices=["hard", "clay", "grass", "carpet"],
                    help="when neither a running edition nor the tournament's history gives one (default: unknown)")
    ap.add_argument("--include-started", action="store_true",
                    help="price matches past their start time that have no live feed, as if pre-match")
    add_policy_args(ap)
    ap.add_argument("--data", type=Path, default=ROOT / "tennis.db")
    ap.add_argument("--json", type=Path, help="also write every row and decision as JSON here")
    args = ap.parse_args(argv)
    policy = policy_from_args(args)

    if args.pull:
        pull = load_pull(args.pull)
    else:
        pull = fetch_pull(tuple(args.series.split(",")))
        save_pull(pull, RAW_KALSHI)
    events, rejected = parse_events(pull.events)
    print(f"pull {pull.source} ({pull.as_of}): {len(events)} matches, {len(pull.live)} live-data entries, "
          f"{len(pull.books)} books")

    if not args.data.exists():
        raise SystemExit(f"{args.data} not found; run scripts/build_db.py first")
    predictor = load_predictor(args.predictor, fixed_p=args.fixed_p, run=args.run, raw=args.raw,
                               data_path=args.data, today=pull.as_of[:10])
    con = sqlite3.connect(f"file:{args.data.resolve().as_posix()}?mode=ro", uri=True)
    try:
        rows = Retriever(predictor, con, pull.as_of[:10], args.surface, args.include_started)(pull)
    finally:
        con.close()
    decisions = decide_all(rows, pull.books, policy)
    dec = {d.event: d for d in decisions}

    priced = sorted((r for r in rows if not r.skip), key=lambda r: -abs(r.deviation or 0))
    print(f"\npredictor {predictor.name}; bankroll ${policy.bankroll:,.0f}, {policy.kelly:g} Kelly, "
          f"model weight {policy.model_weight:g}, in-play bets {'on' if policy.in_play else 'off'}\n")
    hdr = (f"{'match':38} {'start (UTC)':16} {'surf':6} {'bo':>2} {'model':>6} {'mkt':>6} "
           f"{'dev':>6} {'ask A':>5} {'ask B':>5}  {'decision':44} score")
    print(hdr)
    print("-" * len(hdr))
    for r in priced:
        e, d = r.event, dec[r.event.event]
        start = "LIVE" if r.status == "live" else (e.start or "")[:16].replace("T", " ")
        sz = d.sizing
        what = (f"BET {d.name[:16]} {sz.contracts} @{sz.avg_price:.2f} ${sz.cost:.0f} EV{sz.ev:+.2f}"
                if d.bet else d.reason)
        print(f"{(e.a.name + ' v ' + e.b.name)[:38]:38} {start:16} {(r.surface or '?')[:6]:6} "
              f"{r.best_of:2d} {r.p_model:6.3f} {fmt(r.p_market, '6.3f'):>6} {fmt(r.deviation, '+6.3f'):>6} "
              f"{fmt(r.ask_a, '5.2f'):>5} {fmt(r.ask_b, '5.2f'):>5}  {what[:44]:44} {r.live.raw if r.live else ''}")
    surf = sorted({(r.event.tour, r.tournament, r.surface or '?', r.surface_from) for r in rows if r.day})
    if surf:
        print("\nsurfaces:")
        for tour, t, sf, how in surf:
            print(f"  {tour} {t[:40]:40} {sf:6} {how}")
    loose = [(s.name, s.resolved) for r in priced for s in (r.event.a, r.event.b) if s.resolved != "exact"]
    if loose:
        print("\nnames resolved loosely (check):")
        for n, how in loose:
            print(f"  {n:32} {how}")
    skipped = [r for r in rows if r.skip]
    if skipped or rejected:
        print(f"\nnot priced ({len(skipped) + len(rejected)}):")
        for r in skipped:
            print(f"  {r.event.title[:40]:40} {r.skip}")
        for et, why in rejected:
            print(f"  {et[:40]:40} {why}")
    bets = [d for d in decisions if d.bet]
    stake = sum(d.sizing.cost for d in bets)
    reasons: dict[str, int] = {}
    for d in decisions:
        if not d.bet:
            k = d.reason.split(":")[0]
            reasons[k] = reasons.get(k, 0) + 1
    print(f"\n{len(priced)} priced, {len(skipped)} skipped, {len(rejected)} rejected events")
    print(f"{len(bets)} bets, stake ${stake:,.2f} ({stake / policy.bankroll:.1%} of bankroll), "
          f"EV ${sum(d.sizing.ev for d in bets):+,.2f} under the sizing probability; no bet: "
          + (", ".join(f"{k} {n}" for k, n in sorted(reasons.items(), key=lambda t: -t[1])) or "none"))
    if args.json:
        args.json.write_text(json.dumps({"pull": pull.source, "as_of": pull.as_of, "predictor": predictor.name,
                                         "policy": vars(policy), "rows": [r.to_json() for r in rows],
                                         "decisions": [d.to_json() for d in decisions], "rejected": rejected},
                                        indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"json: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
