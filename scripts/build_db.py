"""Rebuild tennis.db from the pinned raw snapshots. The database is disposable.

    python scripts/build_db.py                 # full build -> tennis.db
    python scripts/build_db.py --db other.db   # build somewhere else

Protocol (details in src/tennis/build.py and docs/DESIGN.md, "Build protocol"):
  0 preflight  -> 1 schema -> 2 register snapshots -> 3 Sackmann backbone -> 4 overrides
  -> 5 MCP -> 6 slam point-by-point -> 7 verify invariants -> 8 publish (atomic swap)

Inputs:  data/build_manifest.json (which folder per source), data/raw/<pinned folders>,
         data/overrides.csv (hand-confirmed links), data/snapshot_lock.json (file hashes).
Outputs: tennis.db, data/build_report.json. On failure the previous tennis.db is untouched
         and the partial build is left at tennis.db.building.
Exit status: 0 success, 1 build stopped (layout drift, changed raw bytes, failed invariant).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tennis.build import run_build          # noqa: E402
from tennis.snapshots import BuildError     # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=ROOT / "tennis.db")
    ap.add_argument("--manifest", type=Path, default=ROOT / "data" / "build_manifest.json")
    ap.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--lock", type=Path, default=ROOT / "data" / "snapshot_lock.json")
    ap.add_argument("--overrides", type=Path, default=ROOT / "data" / "overrides.csv")
    ap.add_argument("--report", type=Path, default=ROOT / "data" / "build_report.json")
    args = ap.parse_args(argv)
    try:
        report = run_build(project_root=ROOT, manifest_path=args.manifest, raw_root=args.raw,
                           db_path=args.db, lock_path=args.lock, overrides_path=args.overrides,
                           report_path=args.report)
    except BuildError as e:
        print(f"BUILD STOPPED: {e}", file=sys.stderr)
        return 1
    print("\nTable counts:")
    for t, n in report["table_counts"].items():
        print(f"  {t:20s} {n:>10,}")
    print("\nReview queue:")
    for r in report["review_queue"]:
        print(f"  {r['entity_type']:7s} {r['source']:18s} {r['reason']:24s} {r['n']:>6}")
    print(f"\nReport: {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
