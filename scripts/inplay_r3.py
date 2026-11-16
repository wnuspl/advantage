"""Reboot item R3: serve form vs return form in play (train/inplay_r3.py).

    python scripts/inplay_r3.py moments                 # u by group, split-half, cross-match slopes
    python scripts/inplay_r3.py grid --period fit       # per-group f, per-player kappas (A4 harness)
    python scripts/inplay_r3.py grid --period val --variants "..."   # confirm chosen ones

Every variant is paired against the live setting (one form term 0.02, independent streams,
run's unc), log loss per game-end checkpoint, SE clustered by match; also set ends and by
tour. Choose on fit 2011-18, confirm on val 2019-21, test 2022-26 once.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tennis import models_db                                      # noqa: E402
from tennis.train import inplay, inplay_r3, runs                  # noqa: E402
from tennis.train.data import prepare                             # noqa: E402

SF = inplay_r3.SplitForm


def load(run: int):
    con = models_db.connect(models_db.MODELS_DB)
    models_db.attach_data(con, models_db.DATA_DB)
    try:
        _, params = runs.params_of(con, run)
        prep = prepare(con)
        pm = inplay.prematch(prep, params)
        key = hashlib.sha256(repr(("v2-aces", run, len(prep.m), int(prep.m.match_id[-1]),
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
    return params, prep, pm, mds


def moments(prep, pm) -> dict:
    m = prep.m
    uw, ul = inplay_r3.match_u(prep, pm)
    out = {}
    # group means of u (fit and val years, tour level), with SE
    for per in ("fit", "val"):
        a, b = inplay.PERIODS[per]
        g = defaultdict(list)
        for i in range(len(m)):
            y = m.start[i][:4]
            if not (a <= y <= b) or not m.tour_level[i]:
                continue
            bo = prep.best_of[i]
            for u in (uw[i], ul[i]):
                if not np.isnan(u):
                    for key in (m.tour[i], f"{m.tour[i]}/{m.surface[i]}", f"{m.tour[i]}/bo{bo}"):
                        g[key].append(u)
        out[f"u {per}"] = {k: {"n": len(v), "mean": float(np.mean(v)), "se": float(np.std(v) / np.sqrt(len(v)))}
                           for k, v in sorted(g.items())}
        print(f"\nmean u (direction form variance) by group, tour level, {per} {a}-{b}:")
        for k, r in out[f"u {per}"].items():
            print(f"  {k:14} n={r['n']:7d}  {r['mean']:+.4f} ± {r['se']:.4f}")
    out["split_half"] = inplay_r3.split_half(prep, uw, ul, "2019-01-01")
    print("\nsplit-half reliability of per-player mean u - F_g (all matches with stats before 2019, 40+):")
    for side, r in out["split_half"].items():
        print(f"  {side:7} players={r['players']:5d}  r_half {r['r_half']:+.3f} ± {r['se']:.3f}  "
              f"full {r['r_full']:+.3f}  sd of player means {r['sd_player_mean']:.4f}")
    kappas = [25, 50, 100, 200, 400, 800]
    for per in ("fit", "val"):
        a, b = inplay.PERIODS[per]
        res = inplay_r3.predictive_fit(prep, uw, ul, f"{a}-01-01", f"{b}-12-31", kappas)
        out[f"predictive {per}"] = res
        print(f"\ncross-match: u - F_g on walk-forward D_s(server), D_r(returner), tour level {per}:")
        for r in res:
            print(f"  kappa {r['kappa']:4d} n={r['n']:6d}  b_s {r['b_s']:+.3f} ± {r['se_s']:.3f}  "
                  f"b_r {r['b_r']:+.3f} ± {r['se_r']:.3f}  R2 {r['r2']*1e4:6.1f}e-4 (serve only {r['r2_s']*1e4:5.1f}, "
                  f"return only {r['r2_r']*1e4:5.1f})  sd D_s {r['sd_s']:.4f} D_r {r['sd_r']:.4f}")
    return out


def parse_variants(spec: str) -> list:
    """';'-separated SplitForm(...) keyword lists, e.g.
    "form_var=0.02,kappa_s=200; by_group=(('atp',0.025),('wta',0.015))" """
    out = []
    for part in spec.split(";"):
        part = part.strip()
        if part:
            out.append(eval(f"SF({part})", {"SF": SF}))
    return out


def default_grid() -> list:
    v = [SF(name="sf0.02 (=live)")]
    for f in (0.015, 0.025):
        v.append(SF(form_var=f))
    for fa in (0.015, 0.02, 0.025, 0.03):
        for fw in (0.015, 0.02, 0.025, 0.03):
            if fa != fw:
                v.append(SF(by_group=(("atp", fa), ("wta", fw))))
    for sa, sb in (("clay", 0.015), ("clay", 0.025), ("grass", 0.015), ("grass", 0.025), ("hard", 0.015), ("hard", 0.025)):
        v.append(SF(by_group=tuple((f"{t}/{sa}", sb) for t in ("atp", "wta"))))
    for bo, f in ((5, 0.01), (5, 0.015), (5, 0.025), (3, 0.015), (3, 0.025)):
        v.append(SF(by_group=(("atp/bo%d" % bo, f),)))
    for k in (100, 400, 1600):
        v.append(SF(kappa_s=k))
        v.append(SF(kappa_r=k))
        v.append(SF(kappa_s=k, kappa_r=k))
    return v


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["moments", "grid"])
    ap.add_argument("--run", type=int, default=790)
    ap.add_argument("--period", choices=list(inplay.PERIODS), default="fit")
    ap.add_argument("--variants", default="", help="SplitForm keyword lists separated by ';' (default: the R3 grid)")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args(argv)
    t0 = time.perf_counter()
    params, prep, pm, mds = load(args.run)
    print(f"run {args.run}: {len(mds):,} pbp matches [{time.perf_counter() - t0:.0f}s]")

    if args.cmd == "moments":
        out = moments(prep, pm)
        path = inplay.CACHE / f"r3_moments_run{args.run}.json"
        path.write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"\nwritten {path} [{time.perf_counter() - t0:.0f}s]")
        return 0

    a, b = inplay.PERIODS[args.period]
    sub = [md for md in mds if a <= md.year <= b]
    uw, ul = inplay_r3.match_u(prep, pm)
    vol = inplay_r3.walk(prep, uw, ul, want={md.mid for md in sub})
    inplay_r3.attach(sub, prep, vol)
    base = SF(name="sf0.02 (=live)")
    variants = [base] + [v for v in (parse_variants(args.variants) if args.variants else default_grid()) if v.label() != base.label()]
    variants.append(inplay.InPlay())          # A4's live variant: must equal base exactly
    print(f"{args.period} {a}-{b}: {len(sub):,} matches, {sum(len(md.cps) for md in sub):,} checkpoints, "
          f"{len(variants)} variants")
    t = time.perf_counter()
    probs = inplay.evaluate(sub, variants, params, args.workers)
    print(f"evaluated [{time.perf_counter() - t:.0f}s]")
    ref = base.label()
    gap = max(float(np.abs(p - q).max()) for p, q in zip(probs[ref], probs[inplay.InPlay().label()]) if len(p))
    print(f"check: SplitForm base vs InPlay live, max |dp| = {gap:.2e}")

    report = {}
    scopes = [("all", None, None), ("atp", lambda md: md.tour == "atp", None),
              ("wta", lambda md: md.tour == "wta", None), ("set ends", None, lambda c: c.set_end)]
    for sname in ("hard", "clay", "grass"):
        scopes.append((sname, (lambda s: lambda md: md.surface == s)(sname), None))
    for scope, sel, cps in scopes:
        res = inplay.score(sub, probs, ref, sel, cps)
        report[scope] = res
        print(f"\n[{scope}] {res[ref]['n']:,} checkpoints, {res[ref]['matches']:,} matches; vs {ref}")
        for lab, r in sorted(res.items(), key=lambda kv: kv[1]["log_loss"])[:args.top if scope == "all" else 6]:
            print(f"  {lab:46} {r['log_loss']:.5f}  d {r['d']:+.5f} ± {r['se']:.5f}")
    out = inplay.CACHE / f"r3_grid_{args.period}_run{args.run}{'_custom' if args.variants else ''}.json"
    out.write_text(json.dumps({"run": args.run, "period": args.period, "report": report}, indent=1), encoding="utf-8")
    print(f"\nwritten {out} [{time.perf_counter() - t0:.0f}s]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
