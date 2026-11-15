"""End-of-year serve/return ratings from a points_v1 run: the top N players per tour at the
end of each year, one CSV per year (the serve/return counterpart of eoy_ratings.py).

    python scripts/eoy_serve_return.py                       # run 790, every complete year
    python scripts/eoy_serve_return.py --from 1991 --to 2025 --top 200
    python scripts/eoy_serve_return.py --sort serve          # rank by serve rating

Same cutoff and filters as eoy_ratings.py: year Y counts every match in editions that
started strictly before Y-12-20, players need 10+ rated matches and a match in the year
before the cutoff. Unlike eoy_ratings.py this is ONE replay with a state snapshot at each
cutoff (points_rating.run(snap_before=...)), so all years take one pass.

Columns (ratings are logits of a serve point, so +0.1 of serve rating is roughly +2 pp of
serve points won):
    overall          serve + return (what the list is sorted by unless --sort)
    serve, return    s + f_s(age) and r + f_r(age) at the cutoff: the rating plus the run's
                     age offset, i.e. what the model expects of the player now (higher is
                     better for both). The stored ratings are age-neutral: the age offset
                     is applied per match and never stored (see points_rating.py).
    age              at the cutoff (blank: no birth date, so no age offset)
    serve_sd, return_sd   standard deviations, grown by inactivity to the cutoff
    serve_pts_won    P(win a serve point) on hard vs the average of the year's top 100, %
    return_pts_won   P(win a return point) on hard vs the same reference, %
                     (a 0 rating is the all-levels average, far below tour level)
    <surface>_serve, <surface>_return   serve/return + blend * surface deviation (what the
                     model uses on that surface; blank if never played there)
    overall_neutral, serve_neutral, return_neutral   the stored, age-neutral s + r, s, r
    matches          rated matches; stat_matches: of those, with serve/return point stats
                     (before ~1991 ATP / ~2010s WTA there are none, and the serve/return
                     split rests on results and scores only: read it loosely there)
    last_event_start

Rust/load offsets are left out (they belong to a specific match). The two tours are on
separate scales.
Reads tennis.db and models.db; writes only the CSV files.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from bisect import bisect_left
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np   # noqa: E402

from tennis import context, elo, models_db, points_rating, tuning   # noqa: E402

TOURS = ("atp", "wta")
SURFACES = ("hard", "clay", "grass")
HEADER = ["rank", "player_id", "player", "country", "age", "overall", "serve", "return",
          "serve_sd", "return_sd", "serve_pts_won", "return_pts_won",
          *(f"{s}_{k}" for s in SURFACES for k in ("serve", "return")),
          "overall_neutral", "serve_neutral", "return_neutral",
          "matches", "stat_matches", "last_event_start"]
AGE = slice(5, 9)   # the age features in context.FEATURES
SORT = {"overall": lambda r: r["overall"], "serve": lambda r: r["serve"], "return": lambda r: r["return"]}


def _sig(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def age_offsets(p, tour: str, age: float | None) -> tuple[float, float]:
    """(serve, return) age offset of the run at `age` (0, 0 when unknown)."""
    if age is None:
        return 0.0, 0.0
    c = np.zeros(context.K)
    c[5], c[6], c[7], c[8] = max(0.0, 22 - age), max(0.0, 26 - age), max(0.0, age - 28), max(0.0, age - 32)
    bs, br = p.context_betas(tour)
    return float(bs[AGE] @ c[AGE]), float(br[AGE] @ c[AGE])


def add_point_rates(st: points_rating.PointsState, rows: list[dict], ref_n: int = 100) -> None:
    """serve/return points won on hard vs the mean serve and return of the top ref_n
    (rows: one tour, sorted by overall)."""
    tour = rows[0]["tour"]
    b = st.base.get((tour, "hard"), points_rating.BASE_INIT.get(tour, 0.4))
    ref = rows[:ref_n]
    ms = sum(r["serve"] for r in ref) / len(ref)
    mr = sum(r["return"] for r in ref) / len(ref)
    for r in rows:
        r["serve_pts_won"] = 100 * _sig(b + r["serve"] - mr)
        r["return_pts_won"] = 100 * (1 - _sig(b + ms - r["return"]))


def standings(st: points_rating.PointsState, as_of: str, tour_of: dict, last: dict, nstat: dict,
              names: dict, birth: dict, min_matches: int, active_days: int) -> list[dict]:
    p, day = st.params, date.fromisoformat(as_of).toordinal()
    active = (date.fromisoformat(as_of) - timedelta(days=active_days)).isoformat()
    rows = []
    for pid, s in st.serve.items():
        if st.count.get(pid, 0) < min_matches or last[pid] < active:
            continue
        r, el, tour = st.ret[pid], max(day - st.last[pid], 0), tour_of[pid]
        age = (day - birth[pid]) / 365.25 if birth.get(pid) else None
        fs, fr = age_offsets(p, tour, age)
        name, country = names.get(pid, (f"player {pid}", None))
        row = {"player_id": pid, "player": name, "country": country, "tour": tour, "age": age,
               "overall": s + r + fs + fr, "serve": s + fs, "return": r + fr,
               "serve_sd": math.sqrt(min(st.vs[pid] + p.q * el, p.v_init)),
               "return_sd": math.sqrt(min(st.vr[pid] + p.q * el, p.v_init)),
               "overall_neutral": s + r, "serve_neutral": s, "return_neutral": r,
               "matches": st.count[pid], "stat_matches": nstat.get(pid, 0), "last_event_start": last[pid]}
        for surf in SURFACES:
            k = (pid, surf)
            have = k in st.sserve
            row[f"{surf}_serve"] = s + fs + p.blend * st.sserve[k] if have else None
            row[f"{surf}_return"] = r + fr + p.blend * st.sret[k] if have else None
        rows.append(row)
    return rows


def write_year(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = {"serve_pts_won": 1, "return_pts_won": 1, "age": 1}
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        for i, r in enumerate(rows, 1):
            out = []
            for h in HEADER:
                v = i if h == "rank" else r[h]
                if isinstance(v, float):
                    v = round(v, fmt.get(h, 4))
                out.append("" if v is None else v)
            w.writerow(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=int, default=790, help="points_v1 model_run id (default 790, the baseline)")
    ap.add_argument("--from", dest="first", type=int, help="first year (default: first year in the data)")
    ap.add_argument("--to", dest="last", type=int, help="last year (default: last complete year)")
    ap.add_argument("--include-partial", action="store_true", help="also write the last, incomplete year")
    ap.add_argument("--cutoff", default="12-20", help="MM-DD; events starting on/after it go to next year")
    ap.add_argument("--sort", choices=list(SORT), default="overall")
    ap.add_argument("--top", type=int, default=100)
    ap.add_argument("--min-matches", type=int, default=10)
    ap.add_argument("--active-days", type=int, default=365)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "eoy-serve-return")
    ap.add_argument("--models", type=Path, default=models_db.MODELS_DB)
    ap.add_argument("--data", type=Path, default=models_db.DATA_DB)
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    con = models_db.connect(args.models)
    models_db.attach_data(con, args.data)
    try:
        spec, params = tuning.model_of(con, args.run)
        if spec.model_id != "points_v1":
            raise SystemExit(f"run {args.run} is {spec.model_id}, not points_v1")
        m = elo.load_matches(con)
        if not len(m):
            raise SystemExit("no matches in tennis.db")
        spec.attach(con, m)
        names, birth = {}, {}
        for pid, n, c, dob in con.execute("SELECT player_id, canonical_name, country, dob FROM data.player"):
            names[pid] = (n, c)
            try:
                birth[pid] = date.fromisoformat(dob).toordinal() if dob else None
            except ValueError:
                birth[pid] = None
    finally:
        con.close()

    data_first, data_last = int(m.start[0][:4]), int(m.start[-1][:4])
    first = args.first or data_first
    last_year = args.last or (data_last if args.include_partial else data_last - 1)
    years = list(range(first, last_year + 1))
    cuts = [bisect_left(m.start, f"{y}-{args.cutoff}") for y in years]
    print(f"run {args.run} (points_v1), years {first}-{last_year}, top {args.top} per tour "
          f"by {args.sort} -> {args.out}/{{atp,wta}}/<year>.csv")

    res = points_rating.run(m, params, snap_before=cuts)
    print(f"replay done ({time.perf_counter() - t0:.0f}s)")

    # per-player bookkeeping up to each cutoff, in one pass
    d = m.extra["points"]
    last, tour_of, nstat = {}, {}, {}
    i = 0
    for year, cut, st in zip(years, cuts, res.snapshots):
        for i in range(i, cut):
            w, l, s = m.winner[i], m.loser[i], m.start[i]
            last[w] = last[l] = s
            tour_of[w] = tour_of[l] = m.tour[i]
            if d.w_sp[i] > 0 and d.l_sp[i] > 0:
                nstat[w], nstat[l] = nstat.get(w, 0) + 1, nstat.get(l, 0) + 1
        i = cut
        rows = standings(st, f"{year}-{args.cutoff}", tour_of, last, nstat, names, birth,
                         args.min_matches, args.active_days)
        counts = []
        for tour in TOURS:
            tr = sorted((r for r in rows if r["tour"] == tour), key=lambda r: (-r["overall"], r["player"]))
            if tr:
                add_point_rates(st, tr)
            top = sorted(tr, key=lambda r: (-SORT[args.sort](r), r["player"]))[:args.top]
            if top:
                write_year(args.out / tour / f"{year}.csv", top)
            counts.append(f"{tour} {len(top)}")
        print(f"  {year}: {', '.join(counts)}")
    print(f"done in {time.perf_counter() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
