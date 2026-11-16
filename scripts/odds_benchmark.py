"""Closing-odds benchmark (track B5, docs/PLAN.md): the de-vigged Pinnacle market vs our
models on identical tour-level matches.

    python scripts/odds_benchmark.py                       # run 790, raw and calibrated
    python scripts/odds_benchmark.py --runs 790,930,203    # points runs are replayed, others read

Source: Valuebetennis season CSVs (CC BY 4.0, https://www.valuebetennis.com/en/donnees.htm),
Pinnacle opening and closing ("last odds recorded before the match") odds, pinned in
data/raw/valuebetennis-20261114/. Rows are linked to tour-level main-draw backbone matches
(odds.link); every row ends linked or with a reason.

Points runs (points_v1/v2) are replayed in process (~5 min each) and scored raw and
calibrated (walk-forward, as in training); other runs are read from models.db's prediction
table. Nothing is written to models.db. Test 2022-26 is used as a benchmark against the
market for finished models, not to choose between them.

Outputs (stdout + models/odds/benchmark_<runs>.json):
  1. link accounting and coverage;
  2. log loss / Brier / accuracy / ECE of market (close, open) and models on IDENTICAL
     matches, overall, per tour, per year;
  3. blend y ~ a*logit(market) + b*logit(model): in-sample (b, z) and walk-forward by year
     (fit on earlier years only), the held-out gain over the market, and the implied
     engine.decide model_weight b / (a + b);
  4. flat-stake backtest of each model against Pinnacle close and open.
Odds are a benchmark only: nothing here feeds a feature or a rating.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tennis import models_db, odds, points_rating                   # noqa: E402
from tennis.train import runs as truns                               # noqa: E402
from tennis.train.data import prepare                                # noqa: E402
from tennis.train.scoring import metrics                             # noqa: E402

RAW = ROOT / "data/raw/valuebetennis-20261114"
FIRST = "2021-01-01"          # rows from here feed the walk-forward blend; scoring starts at --from


def model_probs(con, prep, run_ids: list[int]) -> dict[str, dict[str, float]]:
    """label -> {match_key: P(p1 wins)} on tour-level matches from FIRST (canonical p1)."""
    mk = models_db.match_keys(con)
    out = {}
    for rid in run_ids:
        model_id = con.execute("SELECT model_id FROM model_run WHERE run_id=?", (rid,)).fetchone()
        if model_id is None:
            raise SystemExit(f"run {rid} not in models.db")
        if model_id[0].startswith("points_"):
            t0 = time.monotonic()
            _, params = truns.params_of(con, rid)
            res = points_rating.run(prep.m, params)
            sc = truns.Scored(prep, res.p_winner, (FIRST, "2026-12-31"), raw_tours=params.raw_tours())
            keys = [mk[prep.m.match_id[i]] for i in sc.idx]
            out[f"run{rid}_raw"] = dict(zip(keys, sc.raw))
            out[f"run{rid}_cal"] = dict(zip(keys, sc.cal))
            print(f"replayed run {rid} ({model_id[0]}) in {time.monotonic() - t0:.0f}s", flush=True)
        else:
            out[f"run{rid}"] = {k: p for k, p in con.execute(
                "SELECT match_key, p_p1_wins FROM prediction WHERE run_id=? AND edition_start >= ?", (rid, FIRST))}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=RAW)
    ap.add_argument("--runs", default="790")
    ap.add_argument("--from", dest="date_from", default="2022-01-01")
    ap.add_argument("--to", dest="date_to", default="2026-12-31")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    run_ids = [int(x) for x in a.runs.split(",")]
    out_path = a.out or ROOT / f"models/odds/benchmark_{'_'.join(map(str, run_ids))}.json"

    con = models_db.connect()
    models_db.attach_data(con)
    prep = prepare(con)
    ours, names = odds.load_our_matches(con, "2020-11-01", alias="data")

    # ------------------------------------------------------------ read + link
    rows, read_counts = [], Counter()
    for f in sorted(a.raw.glob("valuebetennis-matchs-*.csv")):
        r, c = odds.read_vb_file(f)
        rows += r
        read_counts.update(c)
    links = odds.link(rows, ours, names)
    reasons = Counter(lk.reason for lk in links)
    print("read:", dict(read_counts))
    print("link:", dict(reasons))

    mk = models_db.match_keys(con)
    our_by_id = {m.match_id: m for m in ours}
    flag = {mk[prep.m.match_id[i]]: prep.winner_is_p1[i] for i in range(len(prep.m))}
    start = {mk[prep.m.match_id[i]]: prep.m.start[i] for i in range(len(prep.m))}
    tour_of = {mk[mid]: m.tour for mid, m in our_by_id.items()}
    odd_by_key = {mk[lk.match_id]: lk.row for lk in links
                  if lk.match_id is not None and "PS_close" in lk.row.odds}

    # ------------------------------------------------------------ aligned rows
    P = model_probs(con, prep, run_ids)
    models = list(P)
    base = [k for k in P[models[0]] if a.date_from <= start[k] <= a.date_to]
    cov = sum(k in odd_by_key for k in base)
    print(f"coverage: {cov:,}/{len(base):,} tour-level matches {a.date_from}..{a.date_to} have closing odds "
          f"({cov / max(len(base), 1):.1%})")
    keys = sorted(set(odd_by_key) & set.intersection(*(set(P[m]) for m in models)), key=lambda k: (start[k], k))
    keys = [k for k in keys if start[k] <= a.date_to]
    y = np.array([flag[k] for k in keys], float)                      # p1 won
    st = np.array([start[k] for k in keys])
    year = np.array([s[:4] for s in st])
    tour = np.array([tour_of[k] for k in keys])
    Pm = {m: np.array([P[m][k] for k in keys]) for m in models}

    def side(kind):
        o1, o2 = np.full(len(keys), np.nan), np.full(len(keys), np.nan)
        for j, k in enumerate(keys):
            od = odd_by_key[k].odds.get(kind)
            if od:
                ow, ol = od                                           # winner, loser
                o1[j], o2[j] = (ow, ol) if y[j] == 1 else (ol, ow)
        return o1, o2
    c1, c2 = side("PS_close")
    o1, o2 = side("PS_open")
    has_open = ~np.isnan(o1)
    mkt = {f"close_{meth}": odds.devig(c1, c2, meth) for meth in ("power", "multiplicative", "shin")}
    mkt["open_power"] = np.where(has_open, odds.devig(np.where(has_open, o1, 2), np.where(has_open, o2, 2)), np.nan)
    allp = {**mkt, **Pm}
    win = (st >= a.date_from) & (st <= a.date_to)
    print(f"scored rows (all models + close): {int(win.sum()):,}; Pinnacle median margin "
          f"{np.median(odds.overround(c1[win], c2[win])):.4f}")

    out = {"source": "Valuebetennis, CC BY 4.0, https://www.valuebetennis.com/en/donnees.htm",
           "raw": str(a.raw), "runs": run_ids, "read": dict(read_counts), "link": dict(reasons),
           "coverage": [cov, len(base)], "n": int(win.sum()), "tables": {}, "blend": {}, "backtest": []}

    def table(name, mask):
        print(f"\n## {name} (n = {int(mask.sum()):,})")
        print(f"{'':16s} {'log loss':>8s} {'vs close':>9s} {'SE':>7s} {'Brier':>7s} {'acc':>6s} {'ECE':>7s}")
        res = {}
        for lab, p in allp.items():
            m = mask & ~np.isnan(p)
            r = metrics(p[m], y[m], ref=mkt["close_power"][m])
            res[lab] = {"n": r.n, "log_loss": r.log_loss, "d": r.d_ref, "se": r.d_se, "brier": r.brier,
                        "acc": r.accuracy, "ece": r.ece}
            print(f"{lab:16s} {r.log_loss:8.4f} {r.d_ref:+9.4f} {r.d_se:7.4f} {r.brier:7.4f} {r.accuracy:6.3f} "
                  f"{r.ece:7.4f}" + ("  (rows with an opening line)" if lab == "open_power" else ""))
        out["tables"][name] = res

    table(f"tour level, {a.date_from[:4]}-{a.date_to[2:4]}", win)
    for t in ("atp", "wta"):
        table(t, win & (tour == t))
    for yr in sorted(set(year[win])):
        table(f"year {yr}", win & (year == yr))

    # ------------------------------------------------------------ blend
    print("\n## blend: y ~ a*logit(close) + b*logit(model); model_weight = b/(a+b)")
    lm = odds.logit(mkt["close_power"])
    for lab in models:
        X = np.column_stack([lm, odds.logit(Pm[lab])])
        (ba, bb), (_, sb) = odds.blend_fit(y[win], X[win])
        held, per_year = [], {}
        for yr in sorted(set(year[win])):
            tr, te = year < yr, win & (year == yr)
            if tr.sum() < 2000:
                continue
            (fa, fb), _ = odds.blend_fit(y[tr], X[tr])
            pb = odds.sigmoid(X[te] @ np.array([fa, fb]))
            d = odds.log_losses(pb, y[te]) - odds.log_losses(mkt["close_power"][te], y[te])
            per_year[yr] = {"a": fa, "b": fb, "w": odds.model_weight(fa, fb), "n": int(te.sum()),
                            "gain": float(-d.mean())}
            held.append(d)
        held = np.concatenate(held) if held else np.array([])
        g = float(-held.mean()) if len(held) else float("nan")
        gse = float(held.std(ddof=1) / np.sqrt(len(held))) if len(held) > 1 else float("nan")
        w = odds.model_weight(ba, bb)
        print(f"{lab:14s} in-sample a={ba:.3f} b={bb:+.3f} (z={bb / sb:+.1f}) model_weight {w:+.3f}; held-out gain "
              f"{g:+.5f} ± {gse:.5f} nats over {len(held):,}")
        for yr, v in per_year.items():
            print(f"{'':14s}   {yr}: b={v['b']:+.3f} w={v['w']:+.3f} gain={v['gain']:+.5f} (n {v['n']:,})")
        out["blend"][lab] = {"a": float(ba), "b": float(bb), "b_se": float(sb), "model_weight": float(w),
                             "held_gain": g, "held_gain_se": gse, "held_n": int(len(held)), "per_year": per_year}

    # ------------------------------------------------------------ backtest
    print("\n## flat-stake backtest: bet when p*odds - 1 > threshold")
    for lab in models:
        for kind, (x1, x2, ok) in (("close", (c1, c2, np.ones(len(keys), bool))), ("open", (o1, o2, has_open))):
            for thr in (0.0, 0.05):
                m = win & ok
                bt = odds.backtest(Pm[lab][m], y[m], x1[m], x2[m], threshold=thr, label=lab)
                print(f"{lab:14s} {kind:5s} >{thr:.2f}: {bt.n_bets:6,} bets ({bt.n_bets / max(bt.n_matches, 1):5.1%}) "
                      f"ROI {bt.roi:+.2%} ± {bt.roi_se:.2%}  hit {bt.hit_rate:.3f} @ {bt.mean_odds:.2f}")
                out["backtest"].append({"odds": kind, "threshold": thr, **bt.__dict__})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
