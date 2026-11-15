"""Tune a walk-forward rating model (Elo or Glicko-2) and record every run in models.db.

    python archive/v1/scripts/sweep.py sweep --model glicko2_v1                # that model's default grid
    python archive/v1/scripts/sweep.py sweep --model elo_v1 --grid '{\"k\": [150], \"blend\": [0.3, 0.4]}' --name my-test
    python archive/v1/scripts/sweep.py show                                     # all models, ranked together
    python archive/v1/scripts/sweep.py show --model glicko2_v1 --scope tour_atp
    python archive/v1/scripts/sweep.py finalize RUN_ID [--save-predictions]     # model read from the run

Models: elo_v1 (k, offset, shape, init, retired_weight, surface_init, blend)
        glicko2_v1 (tau, rd_init, vol_init, period_days, rd_max, init, retired_weight,
                    surface_init, blend)
In Windows PowerShell 5.1 write --grid with single quotes outside and \\" inside.

Every setting runs over the full history, so ratings are warm by the evaluation years.
'sweep' scores only the validation block (default 2019-2021). 'finalize' scores the
held-out test block (default 2022-2026) for runs already chosen. Never pick parameters
by test score. blend is prediction-only: each update setting runs once and every blend
value is scored from that pass.

Scopes: tour (ATP+WTA tour-level main draws), tour_atp, tour_wta, all (every level).
Scored statuses: completed and retired; walkovers, defaults and unfinished are not scored.
Reads tennis.db read-only; writes models.db only.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ARCHIVE = Path(__file__).resolve().parents[1]   # archive/v1: results, reports, docs
ROOT = Path(__file__).resolve().parents[3]      # project root: tennis.db, data, src
sys.path.insert(0, str(ROOT / "src"))

from tennis import models_db, ratingmodels, tuning   # noqa: E402


def _block(s: str) -> tuple[str, str]:
    a, b = s.split(":")
    return a, b


def _prepare(con):
    t = time.perf_counter()
    prep = tuning.prepare(con)
    print(f"loaded {len(prep.m):,} matches in {time.perf_counter() - t:.1f}s")
    return prep


def _print_board(rows, scope):
    print(f"\n{'run':>5}  {'model':10s} {'val log_loss':>12} {'brier':>7} {'acc':>6} {'n':>7}  {'test ll':>8}  params"
          f"   (scope={scope}; coin flip = 0.6931)")
    for run_id, sweep_id, params, n, ll, br, acc, tll, model_id in rows:
        p = json.loads(params)
        short = " ".join(f"{k}={v:g}" for k, v in p.items() if k != "init" or v != 1500)
        tl = f"{tll:.4f}" if tll is not None else "-"
        print(f"{run_id:>5}  {model_id:10s} {ll:>12.4f} {br:>7.4f} {acc:>6.3f} {n:>7}  {tl:>8}  {short}  [{sweep_id}]")


def main(argv=None, default_model: str | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", type=Path, default=models_db.MODELS_DB)
    ap.add_argument("--data", type=Path, default=models_db.DATA_DB)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sweep", help="run a grid; score the validation block")
    s.add_argument("--model", default=default_model or tuning.MODEL_ID, choices=sorted(ratingmodels.MODELS))
    s.add_argument("--grid", help="JSON dict (or list of dicts) of parameter -> list of values")
    s.add_argument("--name", help="sweep id (default: timestamp)")
    s.add_argument("--val", type=_block, default=tuning.DEFAULT_BLOCKS["val"], help="FROM:TO, ISO dates")

    sh = sub.add_parser("show", help="leaderboard by validation log loss")
    sh.add_argument("--model", default=default_model, choices=sorted(ratingmodels.MODELS),
                    help="one model only (default: all models together)")
    sh.add_argument("--name")
    sh.add_argument("--scope", default=tuning.PRIMARY_SCOPE, choices=tuning.SCOPES)
    sh.add_argument("--top", type=int, default=15)

    f = sub.add_parser("finalize", help="score the test block for chosen runs")
    f.add_argument("run_ids", type=int, nargs="+")
    f.add_argument("--test", type=_block, default=tuning.DEFAULT_BLOCKS["test"], help="FROM:TO, ISO dates")
    f.add_argument("--save-predictions", action="store_true",
                   help="also write per-match predictions and pre-match ratings (~5M rows per run)")
    args = ap.parse_args(argv)

    con = models_db.connect(args.models)
    models_db.attach_data(con, args.data)
    try:
        if args.cmd == "sweep":
            spec = ratingmodels.get(args.model)
            grid = tuning.expand_grid(json.loads(args.grid) if args.grid else spec.default_grid, spec.params_cls)
            name = args.name or datetime.now().strftime("sweep-%Y%m%d-%H%M%S")
            n_pass = len({p.update_key() for p in grid})
            print(f"sweep '{name}' ({args.model}): {len(grid)} settings in {n_pass} passes, "
                  f"validation {args.val[0]}..{args.val[1]}")
            prep = _prepare(con)
            t = time.perf_counter()
            tuning.sweep(con, prep, grid, sweep_id=name, val=args.val, model=args.model)
            print(f"done in {time.perf_counter() - t:.0f}s")
            _print_board(tuning.leaderboard(con, sweep_id=name), tuning.PRIMARY_SCOPE)
        elif args.cmd == "show":
            _print_board(tuning.leaderboard(con, sweep_id=args.name, scope=args.scope, top=args.top,
                                            model=args.model), args.scope)
        else:
            prep = _prepare(con)
            for run_id in args.run_ids:
                spec, _ = tuning.model_of(con, run_id)
                scores = tuning.finalize(con, prep, run_id, test=args.test, save_predictions=args.save_predictions)
                print(f"run {run_id} ({spec.model_id}) test {args.test[0]}..{args.test[1]}:")
                for scope, sc in scores.items():
                    print(f"  {scope:9s} n={sc.n:>7}  log_loss={sc.log_loss:.4f}  brier={sc.brier:.4f}  acc={sc.accuracy:.3f}")
                if args.save_predictions:
                    print("  saved predictions and pre-match ratings")
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
