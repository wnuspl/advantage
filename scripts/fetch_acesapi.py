"""Pull tournament results from AcesAPI into a new immutable snapshot folder.

    python scripts/fetch_acesapi.py                  # incremental pull since the Sackmann cutoff
    python scripts/fetch_acesapi.py --since 2026-01-01 --max-credits 300   # wider overlap
    python scripts/fetch_acesapi.py --plan           # spend only list calls; show what a pull would fetch
    python scripts/fetch_acesapi.py --pin            # pull, then pin it in data/build_manifest.json
    python scripts/fetch_acesapi.py --inspect data/raw/acesapi-20261114T120000Z   # no API calls

What a pull fetches (src/tennis/acesapi_fetch.py):
  1. list_tournaments per tour and season from --since (1 credit per page).
  2. get_tournament draw=all (2 credits) for each tournament that has started and overlaps
     [--since, today], unless an earlier pull already saw it final more than --refetch-days
     after it ended. AcesAPI rewrites the current season daily (10:00 UTC), so tournaments
     inside that window are pulled again to pick up corrections.
  Stops before --max-credits would be exceeded. The folder is written under a temp name
  and renamed when done, so a crash leaves no half-written snapshot. Each pull is logged
  to data/SOURCES.md.

The key is ACESAPI_KEY from the environment, else from .env at the project root (gitignored).

Then: pin the folder in data/build_manifest.json (--pin does it), run
python scripts/build_db.py, and read the "acesapi" section of data/build_report.json.
AcesAPI terms: storing and research use allowed; no redistribution of the data as a dataset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tennis import acesapi_fetch as af                     # noqa: E402
from tennis.acesapi import LEAGUE, pull_folder, unwrap      # noqa: E402

RAW = ROOT / "data" / "raw"
MANIFEST = ROOT / "data" / "build_manifest.json"
SOURCES = ROOT / "data" / "SOURCES.md"
DEFAULT_SINCE = "2026-05-18"   # a week before the Sackmann cutoff: Roland Garros 2026 overlaps


def read_dotenv(path: Path) -> dict[str, str]:
    """KEY=value lines; '#' comments and blank lines ignored; optional quotes stripped."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip().removeprefix("export ").strip()
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
            v = v[1:-1]
        out[k] = v
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", default=DEFAULT_SINCE, help="first day of interest (ISO date)")
    ap.add_argument("--tours", nargs="+", default=["atp", "wta"], choices=["atp", "wta"])
    ap.add_argument("--max-credits", type=int, default=200, help="credit budget for this pull")
    ap.add_argument("--refetch-days", type=int, default=10)
    ap.add_argument("--plan", action="store_true", help="list calls only; print the plan; save nothing")
    ap.add_argument("--pin", action="store_true", help="append the pull to data/build_manifest.json")
    ap.add_argument("--inspect", type=Path, help="print field values found in a saved pull folder")
    args = ap.parse_args(argv)

    if args.inspect:
        print(json.dumps(af.inspect(args.inspect), indent=1, ensure_ascii=False))
        return 0
    key = os.environ.get("ACESAPI_KEY") or read_dotenv(ROOT / ".env").get("ACESAPI_KEY")
    if not key:
        print("ACESAPI_KEY not found in the environment or in .env")
        return 2
    since, today = date.fromisoformat(args.since), datetime.now(timezone.utc).date()
    client = af.Client(key)
    pull_id = af.new_pull_id()
    tmp = RAW / f".tmp-{pull_folder(pull_id)}"
    final = RAW / pull_folder(pull_id)
    tmp.mkdir(parents=True)
    log = {"pull_id": pull_id, "since": args.since, "tours": args.tours,
           "max_credits": args.max_credits, "refetch_days": args.refetch_days,
           "base_url": af.BASE_URL, "calls": [], "complete": False}
    spent = 0

    def call(tool: str, body: dict, rel: str | None) -> bytes | None:
        nonlocal spent
        entry = {"tool": tool, "body": body, "file": rel,
                 "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
        log["calls"].append(entry)
        try:
            raw = client.call(tool, body)
        except af.ApiError as e:
            entry.update(http=e.status, error=str(e))
            if e.status in (401, 402, 403):
                raise
            print(f"  {tool} {body}: {e}")
            return None
        spent += af.COST[tool]
        entry.update(http=200, sha256=hashlib.sha256(raw).hexdigest(),
                     credits_remaining=client.credits_remaining)
        if rel and not args.plan:
            path = tmp / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        return raw

    prev = af.previous_pulls(RAW)
    todo: list[tuple[str, dict, str]] = []
    status = 0
    try:
        for tour in args.tours:
            for season in range(since.year, today.year + 1):
                offset = 0
                while offset is not None:
                    body = {"season": season, "limit": 100, "offset": offset, "league": LEAGUE[tour]}
                    raw = call("list_tournaments", body, f"tournaments/{tour}-{season}-{offset}.json")
                    if raw is None:
                        break
                    items, nxt = af.list_items(raw)
                    for t in items:
                        why = af.wanted(t, prev.get((tour, str(t.get("tournament_id")))), since,
                                        today, args.refetch_days)
                        if why:
                            todo.append((tour, t, why))
                    offset = nxt if (nxt and nxt != offset and items) else None
        todo.sort(key=lambda x: (str(x[1].get("start_date")), x[0], str(x[1].get("tournament_id"))))
        print(f"{len(todo)} draw(s) to fetch, {2 * len(todo)} credits; "
              f"{spent} spent on lists; credits left: {client.credits_remaining}")
        for tour, t, why in todo:
            print(f"  {tour} {t.get('start_date')} {t.get('tournament_id'):>10} {why:17s} {t.get('name')}")
        if args.plan:
            return 0
        skipped = 0
        for tour, t, why in todo:
            left = client.credits_remaining
            if spent + af.COST["get_tournament"] > args.max_credits or (left is not None and left < 2):
                skipped += 1
                continue
            tid = str(t.get("tournament_id"))
            call("get_tournament", {"tournament_id": tid, "draw": "all", "league": LEAGUE[tour]},
                 f"draws/{tour}/{tid}.json")
        log["complete"] = skipped == 0
        if skipped:
            print(f"budget reached: {skipped} draw(s) left for the next pull")
    except af.ApiError as e:
        print(f"STOPPED: {e}")
        status = 1
    finally:
        log["credits_spent"] = spent
        log["credits_remaining"] = client.credits_remaining
        if args.plan:
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            (tmp / "pull.json").write_text(json.dumps(log, indent=1, ensure_ascii=False) + "\n",
                                           encoding="utf-8", newline="\n")
            tmp.rename(final)

    n_draws = len(list(final.glob("draws/*/*.json")))
    with open(SOURCES, "a", encoding="utf-8") as f:
        f.write(f"acesapi  {final.name}  pull={pull_id}  draws={n_draws}  credits_spent={spent}  "
                f"credits_left={client.credits_remaining}  complete={log['complete']}  "
                f"api={af.BASE_URL}\n")
    print(f"saved {final.relative_to(ROOT)}: {n_draws} draws, {spent} credits spent, "
          f"{client.credits_remaining} left")
    if n_draws == 0:
        print("no draws saved: nothing to pin")
        return status
    if args.pin:
        try:
            if af.pin(MANIFEST, af.manifest_entry(pull_id, final)):
                print(f"pinned in {MANIFEST.relative_to(ROOT)}; now run python scripts/build_db.py")
        except ValueError as e:
            print(f"saved but NOT pinned: {e}")
            return 1
    else:
        print("not pinned: rerun with --pin, or add this entry to data/build_manifest.json:")
        print(json.dumps(af.manifest_entry(pull_id, final), indent=2))
    return status


if __name__ == "__main__":
    sys.exit(main())
