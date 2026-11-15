"""Ratings at a given date (Elo or Glicko-2), highest first, with player names.

    python archive/v1/scripts/ratings.py                                   # latest saved Elo run, end of data
    python archive/v1/scripts/ratings.py --model glicko2_v1                # latest saved Glicko-2 run
    python archive/v1/scripts/ratings.py --run 140 --as-of 2024-06-01      # any run; its model is read from models.db
    python archive/v1/scripts/ratings.py --as-of 2019-07-01 --tour atp --surface grass --top 20
    python archive/v1/scripts/ratings.py --run 132 --all --csv ratings.csv

Replays the chosen run's parameters (from models.db) over every match in editions that
started strictly before --as-of, then lists each player's rating after their last match.
Sackmann dates matches only by their event's start, so an event under way on --as-of
counts in full. Any run works, saved per-match rows or not.

Glicko-2 runs also show RD (the rating's uncertainty), grown by inactivity up to --as-of:
a player who has not played for months has a wide RD even if their rating is high.

By default the list is per tour (ratings of the two tours are on separate scales),
sorted by overall rating, and limited to players with 10+ rated matches who played in
the year before --as-of (otherwise long-retired players keep their old peaks). --surface
sorts by the run's blend of overall and that surface's rating: the number the model
would actually use on that surface. Reads tennis.db and models.db; writes nothing
except an optional --csv file.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import date
from pathlib import Path

ARCHIVE = Path(__file__).resolve().parents[1]   # archive/v1: results, reports, docs
ROOT = Path(__file__).resolve().parents[3]      # project root: tennis.db, data, src
sys.path.insert(0, str(ROOT / "src"))

from tennis import elo, models_db, ratingmodels, standings, tuning   # noqa: E402


def _date(s: str) -> str:
    return date.fromisoformat(s[:10]).isoformat()   # accepts YYYY-MM-DD or a full timestamp


def default_run(con, model: str) -> int:
    row = con.execute("""SELECT r.run_id FROM model_run r
                         WHERE r.model_id=? AND EXISTS (SELECT 1 FROM prediction p WHERE p.run_id=r.run_id)
                         ORDER BY r.run_id DESC LIMIT 1""", (model,)).fetchone()
    if row is None:
        raise SystemExit(f"no saved {model} run in models.db; pass --run RUN_ID "
                         "(or save one: sweep.py finalize RUN_ID --save-predictions)")
    return row[0]


def fmt(x):
    return f"{x:7.1f}" if x is not None else "      -"


def main(argv=None, default_model: str = tuning.MODEL_ID) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--as-of", type=_date, help="date or timestamp (default: end of data)")
    ap.add_argument("--run", type=int, help="model_run id whose parameters to use "
                                            "(default: latest saved run of --model)")
    ap.add_argument("--model", choices=sorted(ratingmodels.MODELS),
                    help=f"which model's latest saved run to use when --run is not given "
                         f"(default {default_model})")
    ap.add_argument("--tour", choices=["atp", "wta"], help="one tour only (default: both, listed separately)")
    ap.add_argument("--surface", choices=list(standings.SURFACES), help="rank by the blended rating on this surface")
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--min-matches", type=int, default=10)
    ap.add_argument("--active-days", type=int, default=365, help="must have played within N days of --as-of")
    ap.add_argument("--all", action="store_true", help="no top/min-matches/activity limits")
    ap.add_argument("--csv", type=Path, help="also write the full list to this CSV file")
    ap.add_argument("--models", type=Path, default=models_db.MODELS_DB)
    ap.add_argument("--data", type=Path, default=models_db.DATA_DB)
    args = ap.parse_args(argv)

    con = models_db.connect(args.models)
    models_db.attach_data(con, args.data)
    try:
        run_id = args.run or default_run(con, args.model or default_model)
        spec, params = tuning.model_of(con, run_id)
        if args.run and args.model and args.model != spec.model_id:
            raise SystemExit(f"run {run_id} is a {spec.model_id} run, not {args.model}")
        t = time.perf_counter()
        m = elo.load_matches(con)
        rows, last_start = standings.compute(
            con, m, params, as_of=args.as_of, tour=args.tour, surface=args.surface,
            min_matches=0 if args.all else args.min_matches,
            active_days=None if args.all else args.active_days, run=spec.run)
    finally:
        con.close()

    rd = spec.has_deviation
    print(f"run {run_id} ({spec.model_id}) {params.as_dict()}")
    print(f"as of {args.as_of or 'end of data'} (last event included started {last_start}); "
          f"{time.perf_counter() - t:.0f}s")
    if args.surface:
        print(f"sorted by {1 - params.blend:.2f} x overall + {params.blend:.2f} x {args.surface}")

    for tour in ([args.tour] if args.tour else ["atp", "wta"]):
        shown = [r for r in rows if r.tour == tour]
        top = shown if args.all else shown[:args.top]
        print(f"\n{tour.upper()}  ({len(shown):,} players match the filters; showing {len(top)})")
        print(f"{'#':>4}  {'player':28s} {'ctry':4s} {'sort by':>7} {'overall':>7} "
              + (f"{'RD':>6} " if rd else "")
              + f"{'hard':>7} {'clay':>7} {'grass':>7} {'matches':>7}  last event")
        for i, r in enumerate(top, 1):
            print(f"{i:>4}  {r.name[:28]:28s} {r.country or '':4s} {r.score:7.1f} {r.overall:7.1f} "
                  + (f"{r.deviation:6.1f} " if rd else "")
                  + f"{fmt(r.surface['hard'])} {fmt(r.surface['clay'])} {fmt(r.surface['grass'])} "
                  f"{r.matches:>7}  {r.last_played}")

    if args.csv:
        with open(args.csv, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["tour", "rank", "player_id", "player", "country", "sort_by", "overall", "rd",
                        "hard", "clay", "grass", "matches", "last_event_start"])
            for tour in ("atp", "wta"):
                for i, r in enumerate((r for r in rows if r.tour == tour), 1):
                    w.writerow([tour, i, r.player_id, r.name, r.country, round(r.score, 2),
                                round(r.overall, 2), round(r.deviation, 2) if r.deviation is not None else "",
                                *(round(r.surface[s], 2) if r.surface[s] is not None else ""
                                  for s in standings.SURFACES),
                                r.matches, r.last_played])
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
