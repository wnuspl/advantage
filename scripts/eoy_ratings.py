"""End-of-year ratings: the top N players per tour at the end of each year, one CSV per year.

    python scripts/eoy_ratings.py                               # latest saved Glicko-2 run, every complete year
    python scripts/eoy_ratings.py --from 2000 --to 2025
    python scripts/eoy_ratings.py --model elo_v1 --out results/eoy-surface-elo
    python scripts/eoy_ratings.py --run 132 --top 50 --include-partial

Same replay as scripts/ratings.py: for year Y, every match in editions that started
strictly before Y-12-20 counts (--cutoff; events starting 27-31 Dec are next season's
openers, e.g. United Cup / Brisbane, so a 1 January cutoff would leak them in), with the same filters (10+ rated matches,
played in the year before the cutoff). Writes <out>/atp/<Y>.csv and <out>/wta/<Y>.csv.

By default only years the data fully covers are written (the mirror ends mid-2026, so a
2026 file would be a mid-season snapshot); --include-partial also writes the last year.
Each year is a fresh replay from the start of the data, so a full run takes a while.
Reads tennis.db and models.db; writes only the CSV files.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tennis import elo, models_db, ratingmodels, standings, tuning   # noqa: E402
from ratings import default_run   # noqa: E402

TOURS = ("atp", "wta")
HEADER = ["rank", "player_id", "player", "country", "sort_by", "overall", "rd",
          "hard", "clay", "grass", "matches", "last_event_start"]


def _r(x):
    return round(x, 2) if x is not None else ""


def write_year(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        for i, r in enumerate(rows, 1):
            w.writerow([i, r.player_id, r.name, r.country, _r(r.score), _r(r.overall), _r(r.deviation),
                        *(_r(r.surface[s]) for s in standings.SURFACES), r.matches, r.last_played])


def main(argv=None, default_model: str = "glicko2_v1") -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=int, help="model_run id whose parameters to use "
                                            "(default: latest saved run of --model)")
    ap.add_argument("--model", choices=sorted(ratingmodels.MODELS),
                    help=f"which model's latest saved run to use when --run is not given "
                         f"(default {default_model})")
    ap.add_argument("--from", dest="first", type=int, help="first year (default: first year in the data)")
    ap.add_argument("--to", dest="last", type=int, help="last year (default: last complete year)")
    ap.add_argument("--include-partial", action="store_true",
                    help="also write the last year in the data even though it is incomplete")
    ap.add_argument("--cutoff", default="12-20", help="MM-DD in year Y: events starting on or after it "
                                                      "are left out of year Y (default 12-20)")
    ap.add_argument("--surface", choices=list(standings.SURFACES), help="rank by the blended rating on this surface")
    ap.add_argument("--top", type=int, default=100)
    ap.add_argument("--min-matches", type=int, default=10)
    ap.add_argument("--active-days", type=int, default=365, help="must have played within N days of the cutoff")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "eoy-elos")
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
        m = elo.load_matches(con)
        if not len(m):
            raise SystemExit("no matches in tennis.db")
        data_first, data_last = int(m.start[0][:4]), int(m.start[-1][:4])
        first = args.first or data_first
        last = args.last or (data_last if args.include_partial else data_last - 1)
        print(f"run {run_id} ({spec.model_id}) {params.as_dict()}")
        print(f"years {first}-{last}, top {args.top} per tour -> {args.out}/{{atp,wta}}/<year>.csv")

        t0 = time.perf_counter()
        for year in range(first, last + 1):
            t = time.perf_counter()
            rows, last_start = standings.compute(
                con, m, params, as_of=f"{year}-{args.cutoff}", surface=args.surface,
                min_matches=args.min_matches, active_days=args.active_days, run=spec.run)
            counts = []
            for tour in TOURS:
                top = [r for r in rows if r.tour == tour][:args.top]
                if top:
                    write_year(args.out / tour / f"{year}.csv", top)
                counts.append(f"{tour} {len(top)}")
            print(f"  {year}: {', '.join(counts)}  (last event {last_start}; {time.perf_counter() - t:.0f}s)")
    finally:
        con.close()
    print(f"done in {time.perf_counter() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
