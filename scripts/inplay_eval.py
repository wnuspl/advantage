"""In-play evaluation of a points run (milestone A4): game-end checkpoints on slam pbp + MCP.

    python scripts/inplay_eval.py moments                      # shared serve deviations within / across streams
    python scripts/inplay_eval.py grid --period fit            # choose on 2011-18
    python scripts/inplay_eval.py grid --period val --forms 0.02 --rhos 0,-0.3 --uncs 3
    python scripts/inplay_eval.py grid --period val --stage-cal ...   # + per-stage alpha fitted on 2011-18

Baseline = score + form 0.02, rho 0, run's unc (what the live predictor does). Every variant
is paired against it (log loss per checkpoint, SE clustered by match). Also reported:
set ends only, and by tour. Choose on fit, confirm on val, score test once.
Reads tennis.db / models.db; writes models/inplay/ (cache, json).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tennis import models_db                                      # noqa: E402
from tennis.train import inplay, runs                             # noqa: E402
from tennis.train.data import prepare                             # noqa: E402

BASE = inplay.InPlay()


def floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x != ""]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["moments", "grid"])
    ap.add_argument("--run", type=int, default=790)
    ap.add_argument("--period", choices=list(inplay.PERIODS), default="fit")
    ap.add_argument("--forms", default="0.01,0.02,0.03,0.05")
    ap.add_argument("--rhos", default="-0.5,-0.25,0,0.25")
    ap.add_argument("--uncs", default="", help="in-play unc values (default: the run's only)")
    ap.add_argument("--aces", default="0", help="ace_bonus values")
    ap.add_argument("--stage-cal", action="store_true", help="also per-stage alphas fitted on the fit period")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--models", type=Path, default=models_db.MODELS_DB)
    ap.add_argument("--data", type=Path, default=models_db.DATA_DB)
    args = ap.parse_args(argv)
    t0 = time.perf_counter()

    con = models_db.connect(args.models)
    models_db.attach_data(con, args.data)
    try:
        _, params = runs.params_of(con, args.run)
        prep = prepare(con)
        pm = inplay.prematch(prep, params)
        key = hashlib.sha256(repr(("v2-aces", args.run, len(prep.m), int(prep.m.match_id[-1]),
                                   con.execute("SELECT COUNT(*), MAX(point_id) FROM data.point").fetchone())).encode())
        cache = inplay.CACHE / f"dataset_{key.hexdigest()[:16]}.pkl"
        if cache.exists():
            mds, bad = pickle.loads(cache.read_bytes())
        else:
            mds, bad = inplay.dataset(con, prep, pm)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(pickle.dumps((mds, bad), protocol=pickle.HIGHEST_PROTOCOL))
    finally:
        con.close()
    n_cp = sum(len(md.cps) for md in mds)
    print(f"run {args.run}: {len(mds):,} matches replay ({bad:,} dropped), {n_cp:,} game-end checkpoints "
          f"[{time.perf_counter() - t0:.0f}s]")

    if args.cmd == "moments":
        for per, (a, b) in inplay.PERIODS.items():
            sub = [md for md in mds if a <= md.year <= b]
            for name, r in inplay.moments(sub).items():
                print(f"  {per:4} {name:32} n={r['n']:6d}  cov {r['cov']:+.4f} ± {r['se']:.4f}  (mean rating var {r['mean_v']:.4f})")
        return 0

    a, b = inplay.PERIODS[args.period]
    sub = [md for md in mds if a <= md.year <= b]
    uncs = floats(args.uncs) or [None]
    variants = [BASE, inplay.InPlay(use_form=False)]
    for f in floats(args.forms):
        for r in floats(args.rhos):
            for u in uncs:
                for ab in floats(args.aces):
                    v = inplay.InPlay(form_var=f, form_corr=r, unc=u, ace_bonus=ab)
                    if v not in variants:
                        variants.append(v)
    print(f"{args.period} {a}-{b}: {len(sub):,} matches, {len(variants)} variants")
    t = time.perf_counter()
    probs = inplay.evaluate(sub, variants, params, args.workers)
    print(f"evaluated [{time.perf_counter() - t:.0f}s]")

    stage_alphas = {}
    if args.stage_cal:
        fa, fb = inplay.PERIODS["fit"]
        fit = [md for md in mds if fa <= md.year <= fb]
        fit_probs = probs if args.period == "fit" else inplay.evaluate(fit, variants, params, args.workers)
        for v in variants:
            al = inplay.fit_stage_alphas(fit, fit_probs[v.label()])
            stage_alphas[v.label()] = al
            probs[v.label() + " +stage"] = inplay.apply_stage_alphas(sub, probs[v.label()], al)
            if v == BASE:
                print("stage alphas (base, fitted on fit period): " + json.dumps({k: round(x, 3) for k, x in al.items()}))

    ref = BASE.label()
    report = {}
    for scope, sel, cps in (("all", None, None), ("atp", lambda md: md.tour == "atp", None),
                            ("wta", lambda md: md.tour == "wta", None), ("set ends", None, lambda c: c.set_end)):
        res = inplay.score(sub, probs, ref, sel, cps)
        report[scope] = res
        print(f"\n[{scope}] {res[ref]['n']:,} checkpoints, {res[ref]['matches']:,} matches; vs {ref}")
        for lab, r in sorted(res.items(), key=lambda kv: kv[1]["log_loss"])[:args.top if scope == "all" else 5]:
            print(f"  {lab:22} {r['log_loss']:.5f}  d {r['d']:+.5f} ± {r['se']:.5f}")
    out = inplay.CACHE / f"grid_{args.period}_run{args.run}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"run": args.run, "period": args.period, "report": report, "stage_alphas": stage_alphas}, indent=1), encoding="utf-8")
    print(f"\nwritten {out} [{time.perf_counter() - t0:.0f}s]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
