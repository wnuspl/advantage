"""Train and evaluate points models; every run is recorded in models.db (milestone A2).

    python scripts/train.py score 790                        # replay a recorded run, print val (no write)
    python scripts/train.py sweep --base 790 --grid '{"q": [5e-6, 1e-5], "unc": [2, 3, 4]}' --name q-unc
    python scripts/train.py context --base 694 --set q=7e-6 --set unc=3 --iters 3 --record
    python scripts/train.py show [--name q-unc] [--scope tour_atp_cal]
    python scripts/train.py compare 694 790 [--block test] [--per-tour-alpha]
    python scripts/train.py finalize RUN_ID [--save-predictions]   # test 2022-26: once per model

Validation 2019-21 is scored raw and calibrated (per quarter, tour and best-of, 10-year
history, past data only); the primary number is tour-level calibrated log loss
(scope tour_cal). Choose on val only. 'finalize' scores the held-out test block and refuses
to do it twice. --model-id names new runs (points_v1 by default; the retrain is points_v2).
In Windows PowerShell 5.1 write --grid with single quotes outside and \\" inside.
Reads tennis.db read-only; writes models.db only (and --json files).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tennis import context, models_db, points_rating                 # noqa: E402
from tennis.train import compare, context_fit, runs                   # noqa: E402
from tennis.train.data import BLOCKS, prepare                         # noqa: E402


def _overrides(sets: list[str]) -> dict:
    out = {}
    for s in sets:
        k, v = s.split("=", 1)
        out[k] = float(v)
    return out


def _print_scores(sc: dict) -> None:
    for scope in ("tour_cal", "tour_atp_cal", "tour_wta_cal", "tour", "tour_atp", "tour_wta", "all"):
        s = sc[scope]
        d = "" if s.d_ref != s.d_ref else f"  vs ref {s.d_ref:+.4f} ± {s.d_se:.4f} ({s.blocks_won}/{s.blocks} quarters)"
        print(f"  {scope:13s} n={s.n:>7,}  log loss {s.log_loss:.4f}  brier {s.brier:.4f}  "
              f"slope {s.cal_slope:.3f}{d}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", type=Path, default=models_db.MODELS_DB)
    ap.add_argument("--data", type=Path, default=models_db.DATA_DB)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sc = sub.add_parser("score", help="replay a recorded run and print its val scores (writes nothing)")
    sc.add_argument("run_id", type=int)

    sw = sub.add_parser("sweep", help="grid over parameters from a base run; score val; record each")
    sw.add_argument("--base", type=int, help="run whose params unlisted parameters keep (default: defaults)")
    sw.add_argument("--grid", required=True, help="JSON dict (or list of dicts): parameter -> list of values")
    sw.add_argument("--name", help="sweep id (default: timestamp)")
    sw.add_argument("--model-id", default=runs.MODEL_ID)

    cx = sub.add_parser("context", help="fit context coefficients by iterated residual WLS")
    cx.add_argument("--base", type=int, default=694, help="starting run (its context coefficients too)")
    cx.add_argument("--set", action="append", default=[], help="NAME=VALUE override, e.g. q=7e-6")
    cx.add_argument("--zero-context", action="store_true", help="start from zero coefficients")
    cx.add_argument("--start", type=Path, help="JSON of starting params (a --json history: its last iteration), "
                                                  "applied before --set")
    cx.add_argument("--iters", type=int, default=3)
    cx.add_argument("--fit-from", default="2000-01-01")
    cx.add_argument("--cutoff", default="2019-01-01", help="fit on matches before this date")
    cx.add_argument("--tour-only", action="store_true")
    cx.add_argument("--per-tour", action="store_true", help="separate ATP / WTA coefficients")
    cx.add_argument("--record", action="store_true", help="record the final setting as a model_run")
    cx.add_argument("--name", default="context")
    cx.add_argument("--model-id", default=runs.MODEL_ID)
    cx.add_argument("--json", type=Path, help="write the iteration history here")

    sh = sub.add_parser("show", help="leaderboard by val log loss")
    sh.add_argument("--name")
    sh.add_argument("--scope", default=runs.PRIMARY, choices=runs.RAW_SCOPES + runs.CAL_SCOPES)
    sh.add_argument("--model-id")
    sh.add_argument("--top", type=int, default=15)

    cp = sub.add_parser("compare", help="runs on identical matches, raw and calibrated, paired")
    cp.add_argument("run_ids", type=int, nargs="+")
    cp.add_argument("--block", choices=["val", "test"], default="val")
    cp.add_argument("--per-tour-alpha", action="store_true", help="calibrate per tour, not per (tour, best-of)")

    fz = sub.add_parser("finalize", help="score the held-out test block (once per model)")
    fz.add_argument("run_ids", type=int, nargs="+")
    fz.add_argument("--save-predictions", action="store_true")
    fz.add_argument("--force", action="store_true", help="re-score a run that already has test metrics")
    args = ap.parse_args(argv)

    con = models_db.connect(args.models)
    models_db.attach_data(con, args.data)
    t0 = time.perf_counter()

    def loaded():
        prep = prepare(con)
        print(f"loaded {len(prep.m):,} matches [{time.perf_counter() - t0:.0f}s]")
        return prep

    try:
        if args.cmd == "show":
            print(f"{'run':>5} {'model':10} {'val':>7} {'brier':>7} {'n':>6} {'test':>7}  params (vs defaults)  [{args.scope}]")
            for rid, swid, pj, n, ll, br, tll, mid in runs.leaderboard(
                    con, sweep_id=args.name, scope=args.scope, top=args.top, model_id=args.model_id):
                p = runs.diff_params(points_rating.PointsParams(**json.loads(pj)))
                p = {k: v for k, v in p.items() if not k.startswith(("cs_", "cr_"))}
                has_ctx = any(k.startswith(("cs_", "cr_")) and v for k, v in json.loads(pj).items())
                print(f"{rid:>5} {mid:10} {ll:7.4f} {br:7.4f} {n:>6} {tll if tll is None else round(tll, 4)!s:>7}  "
                      f"{json.dumps(p)}{' +context' if has_ctx else ''}  [{swid}]")
            return 0

        prep = loaded()
        if args.cmd == "score":
            _, params = runs.params_of(con, args.run_id)
            res = points_rating.run(prep.m, params)
            print(f"run {args.run_id}: val {BLOCKS['val'][0]}..{BLOCKS['val'][1]}")
            _print_scores(runs.Scored(prep, res.p_winner, BLOCKS["val"]).scores())
        elif args.cmd == "sweep":
            base = runs.params_of(con, args.base)[1] if args.base else None
            grid = runs.expand_grid(json.loads(args.grid), base)
            name = args.name or datetime.now().strftime("sweep-%Y%m%d-%H%M%S")
            print(f"sweep '{name}': {len(grid)} settings in {len({p.update_key() for p in grid})} passes")
            runs.sweep(con, prep, grid, sweep_id=name, model_id=args.model_id)
        elif args.cmd == "context":
            _, base = runs.params_of(con, args.base)
            start = base
            if args.start:
                js = json.loads(args.start.read_text(encoding="utf-8"))
                start = points_rating.PointsParams(**(js["history"][-1]["params"] if "history" in js else js))
            start = replace(start, **_overrides(args.set))
            if args.zero_context:
                start = start.with_context([0.0] * context.K, [0.0] * context.K)
            ref = runs.Scored(prep, points_rating.run(prep.m, base).p_winner, BLOCKS["val"])
            print(f"reference: run {args.base} as recorded, val cal {ref.scores()['tour_cal'].log_loss:.4f}")
            p, hist, res, scd = context_fit.fit(prep, start, iters=args.iters, fit_from=args.fit_from,
                                                cutoff=args.cutoff, tour_only=args.tour_only,
                                                per_tour=args.per_tour, ref=ref)
            print(f"\n{'feature':<10}{'serve':>9}{'return':>9}")
            for tr in (("atp", "wta") if p.ctx_per_tour else ("atp",)):
                bs, br = p.context_betas(tr)
                for j, f in enumerate(context.FEATURES):
                    print(f"{f:<10}{bs[j]:+9.4f}{br[j]:+9.4f}" + (f"  {tr}" if p.ctx_per_tour else ""))
            if args.record:
                rid = runs.record(con, p, scd.scores(), model_id=args.model_id, sweep_id=args.name,
                                  notes=f"context fit: start run {args.base} {json.dumps(_overrides(args.set))}"
                                        f"{' zero' if args.zero_context else ''}, cutoff {args.cutoff}, "
                                        f"{args.iters} iterations")
                print(f"recorded run {rid}")
                hist[-1]["run_id"] = rid
            if args.json:
                args.json.parent.mkdir(parents=True, exist_ok=True)
                args.json.write_text(json.dumps({"base": args.base, "set": args.set, "cutoff": args.cutoff,
                                                 "history": hist}, indent=1), encoding="utf-8")
        elif args.cmd == "compare":
            sc_ = compare.replay(con, prep, args.run_ids, BLOCKS[args.block], not args.per_tour_alpha)
            print(f"\n{args.block} {BLOCKS[args.block][0]}..{BLOCKS[args.block][1]}; reference: "
                  f"{next(iter(sc_))} calibrated")
            print(compare.format_table(compare.table(sc_)))
        elif args.cmd == "finalize":
            for rid in args.run_ids:
                s = runs.finalize(con, prep, rid, force=args.force, save_predictions=args.save_predictions)
                print(f"run {rid}: test {BLOCKS['test'][0]}..{BLOCKS['test'][1]}")
                _print_scores(s)
        print(f"[{time.perf_counter() - t0:.0f}s]")
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
