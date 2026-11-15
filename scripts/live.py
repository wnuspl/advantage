"""Open Kalshi tennis match markets vs a predictor (track B1, docs/PLAN.md). Read-only.

    python scripts/live.py                                  # pull now, FixedPredictor(0.5)
    python scripts/live.py --pull data/raw/kalshi/20261115T223132Z   # reprice a saved pull, no network
    python scripts/live.py --predictor fixed --fixed-p 0.6 --surface hard --json out.json

Each poll is saved raw under data/raw/kalshi/<UTC>/ (events, milestones, live scores,
order books). Every open match is listed: priced rows sorted by |model - market|, then
the ones not priced, each with its reason. `--predictor points` is the run-790 /
points_v2 predictor (engine/predict.py, `--run`, `--raw`).
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
from tennis.engine import FixedPredictor                          # noqa: E402
from tennis.engine.live import (RAW_KALSHI, DbEditions, fetch_pull, load_pull,   # noqa: E402
                                parse_events, save_pull, snapshots)
from tennis.engine.players import Resolver                        # noqa: E402
from tennis.engine.surfaces import TournamentSurfaces             # noqa: E402


def make_predictor(args, today: str):
    if args.predictor == "fixed":
        return FixedPredictor(args.fixed_p)
    from tennis.engine.predict import PointsPredictor
    return PointsPredictor.load(args.run, today=today, data_path=args.data, calibrated=not args.raw)


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
                    help="when no running edition in tennis.db gives one (default: unknown)")
    ap.add_argument("--include-started", action="store_true",
                    help="price matches past their start time that have no live feed, as if pre-match")
    ap.add_argument("--data", type=Path, default=ROOT / "tennis.db")
    ap.add_argument("--json", type=Path, help="also write every row as JSON here")
    args = ap.parse_args(argv)

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
    predictor = make_predictor(args, pull.as_of[:10])
    con = sqlite3.connect(f"file:{args.data.resolve().as_posix()}?mode=ro", uri=True)
    try:
        resolver = getattr(predictor, "players", None) or Resolver.from_db(con, pull.as_of[:10], alias="main")
        rows = snapshots(pull, predictor, resolver, DbEditions(con), args.surface, args.include_started,
                         TournamentSurfaces.from_db(con))
    finally:
        con.close()

    priced = sorted((r for r in rows if not r.skip), key=lambda r: -abs(r.deviation or 0))
    print(f"\npredictor {predictor.name}\n")
    hdr = (f"{'match':38} {'start (UTC)':16} {'tour':4} {'surf':6} {'bo':>2} {'model':>6} {'mkt':>6} "
           f"{'dev':>6} {'ask A':>5} {'ask B':>5}  score")
    print(hdr)
    print("-" * len(hdr))
    for r in priced:
        e = r.event
        start = "LIVE" if r.status == "live" else (e.start or "")[:16].replace("T", " ")
        flag = " stale" if r.prediction.stale else ""
        print(f"{(e.a.name + ' v ' + e.b.name)[:38]:38} {start:16} {e.tour:4} {(r.surface or '?')[:6]:6} "
              f"{r.best_of:2d} {r.p_model:6.3f} {fmt(r.p_market, '6.3f'):>6} {fmt(r.deviation, '+6.3f'):>6} "
              f"{fmt(r.ask_a, '5.2f'):>5} {fmt(r.ask_b, '5.2f'):>5}  {r.live.raw if r.live else ''}{flag}")
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
    print(f"\n{len(priced)} priced, {len(skipped)} skipped, {len(rejected)} rejected events")
    if args.json:
        args.json.write_text(json.dumps({"pull": pull.source, "as_of": pull.as_of, "predictor": predictor.name,
                                         "rows": [r.to_json() for r in rows], "rejected": rejected},
                                        indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"json: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
