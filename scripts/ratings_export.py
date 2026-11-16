"""Export weekly rating standings for the ratings viewer (web/ratings.html, served by
scripts/ratings_server.py).

    python scripts/ratings_export.py                          # run 790, weekly from 2010
    python scripts/ratings_export.py --since 2000-01-03 --top 1000

One full replay (~5 min), then one gzipped JSON file per week:
    models/ratings/run<id>/index.json           run, fields, weeks, settings
    models/ratings/run<id>/players.json         [stable key, name, country, tour, dob] by index
    models/ratings/run<id>/weeks/<monday>.json.gz
The folder is written next to the old one and swapped in at the end, so a running server
never mixes two exports. Rerun after each data update (build_db.py).
Reads tennis.db and models.db; writes only models/ratings/.
"""
from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sqlite3
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tennis import matches, models_db, points_rating                   # noqa: E402
from tennis.engine import standings                                    # noqa: E402
from tennis.engine.predict import run_params                           # noqa: E402
from tennis.engine.state import key_maps                               # noqa: E402

OUT = ROOT / "models" / "ratings"
FIELDS = ["player", "s", "r", "sd_s", "sd_r", "fs", "fr", "n", "n_stat", "last",
          *(f"{x}_{k}" for x in standings.SURFACES for k in ("ds", "dr", "n"))]


def _r(x: float) -> float:
    return round(x, 4)


def export(con, run_id: int, out_root: Path, *, since: str, today: str, top: int, min_matches: int,
           active_days: int, log=print) -> Path:
    model_id, params = run_params(con, run_id)
    t = time.perf_counter()
    m = matches.load_matches(con)
    points_rating.attach(con, m)
    log(f"loaded {len(m):,} matches [{time.perf_counter() - t:.0f}s]")
    players = {pid: standings.Player(tour, date.fromisoformat(dob).toordinal() if dob else None)
               for pid, tour, dob in con.execute("SELECT player_id, tour, dob FROM data.player WHERE is_placeholder = 0")}
    info = {pid: (name, country, dob) for pid, name, country, dob in
            con.execute("SELECT player_id, canonical_name, country, dob FROM data.player")}
    pkey = key_maps(con)["player"]
    data_through = max(m.start) if len(m) else None
    cutoffs = standings.mondays(since, today)

    final = out_root / f"run{run_id}"
    tmp = out_root / f"run{run_id}.new"
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "weeks").mkdir(parents=True)
    index: dict[int, int] = {}    # player_id -> index in players.json
    weeks: list[str] = []

    def on_week(w: standings.Week) -> None:
        rows = {}
        for tour, rs in w.rows.items():
            out = []
            for x in rs:
                pid = x["player_id"]
                if pid not in index:
                    index[pid] = len(index)
                row = [index[pid], _r(x["s"]), _r(x["r"]), _r(x["sd_s"]), _r(x["sd_r"]), _r(x["fs"]), _r(x["fr"]),
                       x["n"], x["n_stat"], x["last"]]
                for surf in standings.SURFACES:
                    v = x["surf"][surf]
                    row += [_r(v[0]), _r(v[1]), v[2]] if v else [None, None, None]
                out.append(row)
            rows[tour] = out
        body = {"cutoff": w.cutoff, "base": {t: {k: _r(v) for k, v in b.items()} for t, b in w.base.items()},
                "rows": rows}
        with gzip.open(tmp / "weeks" / f"{w.cutoff}.json.gz", "wt", encoding="utf-8") as f:
            json.dump(body, f, separators=(",", ":"))
        weeks.append(w.cutoff)
        if len(weeks) % 100 == 0:
            log(f"  {len(weeks)}/{len(cutoffs)} weeks ({w.cutoff}) [{time.perf_counter() - t:.0f}s]")

    standings.weekly(m, params, cutoffs, players, min_matches=min_matches, active_days=active_days, top=top,
                     on_week=on_week)
    plist = [None] * len(index)
    for pid, i in index.items():
        name, country, dob = info[pid]
        plist[i] = [pkey.get(pid, f"id:{pid}"), name, country, players[pid].tour, dob]
    (tmp / "players.json").write_text(json.dumps(plist, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    meta = {"run_id": run_id, "model_id": model_id, "blend": params.blend, "fields": FIELDS,
            "surfaces": list(standings.SURFACES), "weeks": weeks, "data_through": data_through,
            "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "min_matches": min_matches, "active_days": active_days, "top": top}
    (tmp / "index.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    old = out_root / f"run{run_id}.old"
    shutil.rmtree(old, ignore_errors=True)
    if final.exists():
        final.rename(old)
    tmp.rename(final)
    shutil.rmtree(old, ignore_errors=True)
    log(f"{len(weeks)} weeks, {len(plist):,} players -> {final} [{time.perf_counter() - t:.0f}s]")
    return final


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=int, default=790)
    ap.add_argument("--since", default="2010-01-04", help="first week (rounded up to a Monday)")
    ap.add_argument("--today", default=date.today().isoformat(), help="last week: the first Monday on/after this")
    ap.add_argument("--top", type=int, default=500, help="players per tour per week")
    ap.add_argument("--min-matches", type=int, default=10)
    ap.add_argument("--active-days", type=int, default=365)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--models", type=Path, default=models_db.MODELS_DB)
    ap.add_argument("--data", type=Path, default=models_db.DATA_DB)
    args = ap.parse_args(argv)
    con = sqlite3.connect(models_db._uri(args.models, "ro"), uri=True)
    models_db.attach_data(con, args.data)
    try:
        export(con, args.run, args.out, since=args.since, today=args.today, top=args.top,
               min_matches=args.min_matches, active_days=args.active_days)
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
