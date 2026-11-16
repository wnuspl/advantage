"""Saved rating state for fast predictor loads (milestone A5).

    python scripts/state.py build [--run 790] [--months 3]   # snapshots at the last N month starts
    python scripts/state.py status [--run 790]               # which snapshots are valid for tennis.db now
    python scripts/state.py verify [--run 790]               # resume from the latest valid one == full replay

After a data update (fetch_acesapi.py --pin, build_db.py), the predictor resumes from the
latest snapshot whose history is unchanged and replays only the editions after it
(seconds instead of minutes). Rebuild snapshots monthly, or when status shows none valid
(e.g. after a change to an older source or to data/overrides.csv).
Reads tennis.db and models.db; writes models/engine/state/.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tennis import context, matches, models_db, points_rating           # noqa: E402
from tennis.engine import state                                        # noqa: E402
from tennis.engine.predict import run_params                           # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build", "status", "verify"])
    ap.add_argument("--run", type=int, default=790)
    ap.add_argument("--months", type=int, default=3, help="build: snapshots at this many month starts")
    ap.add_argument("--today", default=date.today().isoformat())
    ap.add_argument("--models", type=Path, default=models_db.MODELS_DB)
    ap.add_argument("--data", type=Path, default=models_db.DATA_DB)
    args = ap.parse_args(argv)
    t0 = time.perf_counter()
    con = sqlite3.connect(models_db._uri(args.models, "ro"), uri=True)
    models_db.attach_data(con, args.data)
    try:
        _, params = run_params(con, args.run)
        if args.cmd == "build":
            cutoffs = state.month_starts(args.today, args.months)
            print(f"run {args.run}: building snapshots at {', '.join(cutoffs)}")
            for p in state.build(con, args.run, params, cutoffs):
                print(f"  {p.relative_to(ROOT)} ({p.stat().st_size / 1e6:.0f} MB)")
        elif args.cmd == "status":
            paths = state.snapshots(args.run)
            if not paths:
                print(f"run {args.run}: no snapshots (python scripts/state.py build)")
            for p in paths:
                ok = state.latest_valid(con, args.run, params, before=p.stem, log=print)
                print(f"  {p.stem}: {'valid' if ok is not None and ok.cutoff == p.stem else 'stale'}")
        else:
            snap = state.latest_valid(con, args.run, params)
            if snap is None:
                raise SystemExit("no valid snapshot")
            t = time.perf_counter()
            w, _, _ = state.resume(con, snap)
            print(f"resumed from {snap.cutoff}: {len(w.m):,} matches after it in {time.perf_counter() - t:.1f}s")
            t = time.perf_counter()
            m = matches.load_matches(con)
            points_rating.attach(con, m)
            full = points_rating.run(m, params)
            edmap = dict(con.execute("SELECT match_id, edition_id FROM data.matches"))
            ctx = context.player_states(m, [edmap[i] for i in m.match_id]) if params.has_context() else {}
            print(f"full replay {len(m):,} matches in {time.perf_counter() - t:.0f}s")
            a, b = w.res.state, full.state
            bad = [f for f in ("serve", "ret", "vs", "vr", "last", "count", "sserve", "sret", "svs", "svr",
                               "slast", "scount", "base") if getattr(a, f) != getattr(b, f)]
            ctx_ok = set(w.ctx) == set(ctx) and all(
                all(getattr(w.ctx[k], f) == getattr(ctx[k], f) for f in state._P_FIELDS) for k in ctx)
            k = len(m) - len(w.m)
            pred_ok = list(w.res.p_winner) == list(full.p_winner[k:])
            print(f"state equal: {not bad}{'' if not bad else ' (differs: ' + ', '.join(bad) + ')'}; "
                  f"context equal: {ctx_ok}; window predictions equal: {pred_ok}")
            if bad or not ctx_ok or not pred_ok:
                return 1
    finally:
        con.close()
    print(f"[{time.perf_counter() - t0:.0f}s]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
