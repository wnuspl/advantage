"""Stats blackout test: how fast does points_v1 degrade once per-match serve stats stop?

    python archive/v1/scripts/blackout_test.py                       # run 605, blackouts 2022-01-01 and 2023-01-01
    python archive/v1/scripts/blackout_test.py --blackout 2024-01-01 --run 605 --glicko 187

For each blackout date D, run 605 is replayed with every match's serve stats hidden from
D on (all levels), so from D those matches take the no-stats update (result only: the
score tier does not exist yet). That mirrors live use after the mirror's cutoff, when
AcesAPI supplies results without stats. Scored on tour-level matches from D to the last
date that has stats, against
  - full:    run 605 with its stats (what we lose),
  - glicko:  run 187 (Glicko-2, results only; unaffected by the blackout),
each walk-forward calibrated (calibrate.py: per quarter and tour, alpha fit on the run's
own predictions over the previous 10 years, so the blackout run's alpha also adapts).

Writes results/blackout_test.json and reports/blackout_test.png (if matplotlib is present).
Reads only (models.db and tennis.db are not written).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

ARCHIVE = Path(__file__).resolve().parents[1]   # archive/v1: results, reports, docs
ROOT = Path(__file__).resolve().parents[3]      # project root: tennis.db, data, src
sys.path.insert(0, str(ROOT / "src"))

from tennis import calibrate, models_db, points_rating, tuning     # noqa: E402
from tennis.composite_eval import block_of                         # noqa: E402


def ll_vec(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def summarize(ll: dict, sel, ref="full") -> dict:
    out = {"n": int(sel.sum())}
    for k, v in ll.items():
        out[k] = float(v[sel].mean())
        if k != ref:
            d = v[sel] - ll[ref][sel]
            out[f"d_{k}"] = float(d.mean())
            out[f"se_{k}"] = float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else float("nan")
    return out


def quarters_since(start: str, d0: str) -> int:
    a, b = date.fromisoformat(d0), date.fromisoformat(start)
    return ((b.year - a.year) * 12 + b.month - a.month) // 3


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=int, default=605, help="points_v1 run to replay")
    ap.add_argument("--glicko", type=int, default=187, help="results-only reference run")
    ap.add_argument("--blackout", action="append", default=None, help="ISO date (repeatable)")
    ap.add_argument("--json", type=Path, default=ARCHIVE / "results" / "blackout_test.json")
    ap.add_argument("--png", type=Path, default=ARCHIVE / "reports" / "blackout_test.png")
    ap.add_argument("--models", type=Path, default=models_db.MODELS_DB)
    ap.add_argument("--data", type=Path, default=models_db.DATA_DB)
    args = ap.parse_args(argv)
    dates = sorted(args.blackout or ["2022-01-01", "2023-01-01"])

    t = time.perf_counter()
    con = models_db.connect(args.models)
    models_db.attach_data(con, args.data)
    prep = tuning.prepare(con)
    m = prep.m
    spec, params = tuning.model_of(con, args.run)
    if spec.model_id != "points_v1":
        raise SystemExit(f"run {args.run} is {spec.model_id}, not points_v1")
    spec.attach(con, m)
    gspec, gparams = tuning.model_of(con, args.glicko)
    if gspec.attach:
        gspec.attach(con, m)
    con.close()
    full_data = m.extra["points"]
    stats_end = max(m.start[i] for i in range(len(m)) if full_data.w_sp[i] > 0)
    lo, hi = dates[0], stats_end
    print(f"last event start with stats: {stats_end}; scoring {lo}..{hi}")

    runs = {"full": spec.predict(spec.run(m, params), params),
            "glicko": gspec.predict(gspec.run(m, gparams), gparams)}
    print(f"full and glicko replayed [{time.perf_counter() - t:.0f}s]")
    for d0 in dates:
        m.extra["points"] = points_rating.blackout(m, full_data, d0)
        runs[f"blackout_{d0}"] = spec.predict(spec.run(m, params), params)
        print(f"blackout {d0} replayed [{time.perf_counter() - t:.0f}s]")
    m.extra["points"] = full_data

    cal, raw = {}, {}
    for k, pw in runs.items():
        cal[k], y, idx, _ = calibrate.calibrated(prep, pw, lo, hi)
        raw[k], _ = calibrate.oriented(prep, pw, idx)
    starts = np.array([m.start[i] for i in idx])
    has_stats = np.array([full_data.w_sp[i] > 0 for i in idx])
    tours = np.array([m.tour[i] for i in idx])

    report = {"run": args.run, "glicko_run": args.glicko, "stats_end": stats_end,
              "share_with_stats": float(has_stats.mean()), "blackouts": {}}
    for d0 in dates:
        key = f"blackout_{d0}"
        sel0 = starts >= d0
        llc = {k: ll_vec(cal[k], y) for k in ("full", key, "glicko")}
        llr = {k: ll_vec(raw[k], y) for k in ("full", key, "glicko")}
        llc = {("blackout" if k == key else k): v for k, v in llc.items()}
        llr = {("blackout" if k == key else k): v for k, v in llr.items()}
        qs = np.array([quarters_since(s, d0) if s >= d0 else -1 for s in starts])
        rep = {"overall_cal": summarize(llc, sel0), "overall_raw": summarize(llr, sel0),
               "by_tour_cal": {t_: summarize(llc, sel0 & (tours == t_)) for t_ in ("atp", "wta")},
               "by_year": {}, "by_quarter": {}}
        for yr in range(int(qs.max()) // 4 + 1):
            rep["by_year"][yr + 1] = summarize(llc, (qs // 4) == yr)
        for q in range(int(qs.max()) + 1):
            s = qs == q
            if s.sum() >= 50:
                rep["by_quarter"][q] = summarize(llc, s)
        # how much of the full model's edge over Glicko is left
        o = rep["overall_cal"]
        rep["edge_kept"] = float((o["glicko"] - o["blackout"]) / (o["glicko"] - o["full"]))
        for yr, r in rep["by_year"].items():
            r["edge_kept"] = float((r["glicko"] - r["blackout"]) / (r["glicko"] - r["full"]))
        report["blackouts"][d0] = rep

        print(f"\n== blackout from {d0}: tour-level {d0}..{hi}, n={o['n']:,} (calibrated) ==")
        print(f"  full {o['full']:.4f}  blackout {o['blackout']:.4f}  glicko {o['glicko']:.4f}")
        print(f"  blackout - full {o['d_blackout']:+.4f} ± {o['se_blackout']:.4f}; "
              f"glicko - full {o['d_glicko']:+.4f} ± {o['se_glicko']:.4f}; edge kept {rep['edge_kept']:.0%}")
        r = rep["overall_raw"]
        print(f"  raw: full {r['full']:.4f}  blackout {r['blackout']:.4f}  glicko {r['glicko']:.4f}")
        print(f"  {'year':>4} {'n':>6} {'full':>7} {'blackout':>8} {'glicko':>7} {'bo-full':>8} {'SE':>6} {'gl-full':>8} {'kept':>5}")
        for yr, r in rep["by_year"].items():
            print(f"  {yr:>4} {r['n']:>6,} {r['full']:7.4f} {r['blackout']:8.4f} {r['glicko']:7.4f} "
                  f"{r['d_blackout']:+8.4f} {r['se_blackout']:6.4f} {r['d_glicko']:+8.4f} {r['edge_kept']:5.0%}")

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {args.json}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: no graph")
        return 0
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    for c, d0 in zip(colors, dates):
        bq = report["blackouts"][d0]["by_quarter"]
        x = np.array([int(q) for q in bq]) + 0.5
        yb = np.array([v["d_blackout"] for v in bq.values()])
        sb = np.array([v["se_blackout"] for v in bq.values()])
        yg = np.array([v["d_glicko"] for v in bq.values()])
        ax.plot(x / 4, yb, "-o", color=c, ms=3, label=f"stats hidden from {d0}")
        ax.fill_between(x / 4, yb - 2 * sb, yb + 2 * sb, color=c, alpha=0.12)
        ax.plot(x / 4, yg, "--", color=c, alpha=0.6, lw=1, label=f"Glicko-2 (same matches, from {d0})")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("years since the stats stopped")
    ax.set_ylabel("log loss minus full-stats points_v1 (calibrated)")
    ax.set_title(f"Stats blackout: points_v1 run {args.run} on result-only updates, tour-level, per quarter (±2 SE)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    args.png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.png, dpi=130)
    print(f"wrote {args.png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
