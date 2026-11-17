"""Export settled Kalshi matches from a live.db into an immutable raw folder (a future
tennis.db source; nothing is pinned in data/build_manifest.json, so tennis.db and the
state snapshots are untouched). Format: src/tennis/kalshi_results.py.

    python scripts/kalshi_export.py                         # live.db -> data/raw/kalshi_results-<UTC>/
    python scripts/kalshi_export.py --live-db <copy>/live.db
    python scripts/kalshi_export.py --check                 # parse what would be exported, write nothing
    python scripts/kalshi_export.py --check-raw             # parse every exported file

The watch runs on the server: export there (live.db is opened read-only, safe while the
watch writes), then copy the new folder to data/raw/ here, or copy live.db first (see
scripts/paper_replay.py) and export locally. Matches already in any kalshi_results-*
folder under --raw-root are not exported again.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tennis import kalshi_results as kr                      # noqa: E402


def report(docs) -> str:
    reasons, stats, exact, series = Counter(), 0, 0, Counter()
    for d in docs:
        res, why = kr.parse(d)
        series[d["event"]["series"]] += 1
        reasons[why or "ok"] += 1
        both = bool(res and res["winner_stats"] and res["loser_stats"])
        stats += both
        exact += both and res["winner_stats"]["exact"] and res["loser_stats"]["exact"]
    lines = [f"{len(docs)} matches: " + ", ".join(f"{k} {v}" for k, v in sorted(series.items())),
             "parse: " + ", ".join(f"{k} {v}" for k, v in reasons.most_common()),
             f"both players' serve stats: {stats} ({exact} with an exact serve breakdown)",
             f"milestone missing: {sum(d['milestone'] is None for d in docs)}, "
             f"live from polls: {sum((d['live_from'] or '').startswith('poll') for d in docs)}, "
             f"live missing: {sum(d['live'] is None for d in docs)}"]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live-db", default=str(ROOT / "live.db"))
    ap.add_argument("--raw-root", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--check", action="store_true", help="parse what would be exported; write nothing")
    ap.add_argument("--check-raw", action="store_true", help="parse every file already exported")
    a = ap.parse_args()
    raw_root = Path(a.raw_root)
    if a.check_raw:
        docs = [kr.read(p) for p in sorted(raw_root.glob(kr.FOLDER_PREFIX + "*/matches/*/*.json"))]
        print(report(docs))
        return 0
    path = Path(a.live_db).resolve()
    if not path.exists():
        print(f"no live.db at {path}")
        return 1
    con = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    docs = kr.export_docs(con, kr.exported_events(raw_root))
    if not docs:
        print("nothing new to export")
        return 0
    print(report(docs))
    if a.check:
        return 0
    out = kr.write_export(docs, raw_root, kr.live_db_meta(con, str(path)))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
