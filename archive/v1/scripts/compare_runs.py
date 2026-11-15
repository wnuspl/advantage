"""Compare rating runs on identical tour-level matches, raw and walk-forward calibrated.

    python archive/v1/scripts/compare_runs.py 187 RUN_ID                     # test block 2022-2026
    python archive/v1/scripts/compare_runs.py 187 RUN_ID --block val         # validation 2019-2021
    python archive/v1/scripts/compare_runs.py 187 RUN_ID --stored glicko_cal=203

Each rating run (any model in ratingmodels) is replayed from its saved parameters. The
reference is the CALIBRATED first run. Calibration: per quarter and tour, a scalar alpha
on the logit fitted on the previous 10 years of that tour's tour-level matches
(src/tennis/calibrate.py; --by-best-of: per tour and best-of). --stored adds runs whose predictions are stored in models.db
(e.g. the matchup runs); then only matches every run predicted are scored.

Columns: log loss, delta vs reference with SE (per-match paired differences) and quarterly
blocks won, Brier, ECE, recalibration slope (b < 1 = overconfident). Reads only.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ARCHIVE = Path(__file__).resolve().parents[1]   # archive/v1: results, reports, docs
ROOT = Path(__file__).resolve().parents[3]      # project root: tennis.db, data, src
sys.path.insert(0, str(ROOT / "src"))

from tennis import calibrate, models_db, tuning                       # noqa: E402
from tennis.composite_eval import block_of, metrics                   # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_ids", type=int, nargs="+")
    ap.add_argument("--block", choices=["val", "test"], default="test")
    ap.add_argument("--stored", action="append", default=[], help="LABEL=RUN_ID with stored predictions")
    ap.add_argument("--years", type=int, default=10, help="calibration history (years)")
    ap.add_argument("--by-best-of", action="store_true",
                    help="calibration alpha per (tour, best-of) instead of per tour")
    ap.add_argument("--models", type=Path, default=models_db.MODELS_DB)
    ap.add_argument("--data", type=Path, default=models_db.DATA_DB)
    args = ap.parse_args(argv)
    lo, hi = tuning.DEFAULT_BLOCKS[args.block]

    con = models_db.connect(args.models)
    models_db.attach_data(con, args.data)
    t = time.perf_counter()
    prep = tuning.prepare(con)
    preds, y, idx = {}, None, None
    for rid in args.run_ids:
        spec, params = tuning.model_of(con, rid)
        if spec.attach:
            spec.attach(con, prep.m)
        res = spec.run(prep.m, params)
        pw = spec.predict(res, params)
        bo = calibrate.best_of_of(prep.m) if args.by_best_of else None
        cal, y, idx, alphas = calibrate.calibrated(prep, pw, lo, hi, years=args.years, best_of=bo)
        raw, _ = calibrate.oriented(prep, pw, idx)
        lab = f"{rid}:{spec.model_id}"
        preds[lab + " raw"], preds[lab + " cal"] = raw, cal
        a = np.array(list(alphas.values()))
        print(f"run {rid} ({spec.model_id}) replayed; alpha {a.min():.3f}..{a.max():.3f} "
              f"(mean {a.mean():.3f}) [{time.perf_counter() - t:.0f}s]")
    keep = np.ones(len(idx), dtype=bool)
    if args.stored:
        mk = models_db.match_keys(con)
        keys = np.array([mk[prep.m.match_id[i]] for i in idx])
        pos = {k: j for j, k in enumerate(keys)}
        for spec_ in args.stored:
            lab, rid = spec_.split("=")
            rows = con.execute("SELECT match_key, p_p1_wins, p1_won FROM prediction WHERE run_id=? "
                               "AND edition_start BETWEEN ? AND ?", (int(rid), lo, hi)).fetchall()
            p = np.full(len(idx), np.nan)
            for k, pp, won in rows:
                j = pos.get(k)
                if j is not None:
                    if won is not None and won != y[j]:
                        raise SystemExit(f"stored run {rid} disagrees on the outcome of {k}")
                    p[j] = pp
            keep &= ~np.isnan(p)
            preds[f"{rid}:{lab}"] = p
    con.close()

    starts = np.array([prep.m.start[i] for i in idx])
    tours = np.array([prep.m.tour[i] for i in idx])
    blocks = block_of(starts)
    ref = [k for k in preds if k.endswith(" cal")][0]
    print(f"\n{args.block} {lo}..{hi}, tour-level, {keep.sum():,} of {len(idx):,} matches "
          f"(predicted by every run). Reference: {ref}.")
    for scope, sel in (("tour", keep), ("tour_atp", keep & (tours == "atp")), ("tour_wta", keep & (tours == "wta"))):
        print(f"\n[{scope}] n={sel.sum():,}")
        print(f"  {'model':34s} {'log loss':>8} {'delta':>8} {'SE':>7} {'blocks':>7} {'Brier':>7} {'ECE':>7} {'slope':>6}")
        for lab, p in preds.items():
            mt = metrics(p[sel], y[sel], ref=preds[ref][sel], block=blocks[sel])
            print(f"  {lab:34s} {mt.log_loss:8.4f} {mt.d_ref:+8.4f} {mt.d_se:7.4f} "
                  f"{mt.blocks_won:>3}/{mt.blocks:<3} {mt.brier:7.4f} {mt.ece:7.4f} {mt.cal_slope:6.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
