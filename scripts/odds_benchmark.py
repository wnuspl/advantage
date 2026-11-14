"""Closing-odds benchmark: the de-vigged Pinnacle market vs our walk-forward models.

    python scripts/odds_benchmark.py [--raw data/raw/valuebetennis-20261114] [--runs 187,203,207]

Source: Valuebetennis season CSVs (CC BY 4.0, https://www.valuebetennis.com/en/donnees.htm),
Pinnacle opening and closing ("last odds recorded before the match") odds. Rows are linked
to tour-level main-draw backbone matches (odds.link); every row ends linked or with a reason.

Outputs (stdout + models/odds/benchmark.json):
  1. link accounting and coverage of our tour-level predictions;
  2. log loss / Brier / accuracy / ECE of market (close and open, several devig methods)
     and models on IDENTICAL matches, overall, per tour, per year;
  3. blend test: y ~ a*logit(market) + b*logit(model), in-sample (b, z) and walk-forward by
     year (fit on earlier years only, score the next) -> held-out gain over the market;
  4. flat-stake backtest of each model against Pinnacle close and open prices.
Odds are a benchmark only: nothing here feeds a feature or a rating.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tennis import models_db, odds                       # noqa: E402
from tennis.composite_eval import metrics                # noqa: E402

LABELS = {132: "elo_surface", 187: "glicko2_raw", 203: "glicko2_cal", 207: "siamese", 602: "points_v1_602"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "data/raw/valuebetennis-20261114"))
    ap.add_argument("--runs", default="132,187,203,207")
    ap.add_argument("--from", dest="date_from", default="2022-01-01")
    ap.add_argument("--to", dest="date_to", default="2026-12-31")
    ap.add_argument("--out", default=str(ROOT / "models/odds/benchmark.json"))
    a = ap.parse_args()
    runs = [int(x) for x in a.runs.split(",")]

    con = models_db.connect()
    models_db.attach_data(con)
    ours, names = odds.load_our_matches(con, "2020-11-01", alias="data")

    # ---------------------------------------------------------------- read + link
    rows, read_counts = [], Counter()
    for f in sorted(Path(a.raw).glob("valuebetennis-matchs-*.csv")):
        r, c = odds.read_vb_file(f)
        rows += r
        read_counts.update(c)
    links = odds.link(rows, ours, names, scorer=odds.name_score_full)
    reasons = Counter(l.reason for l in links)
    print("read:", dict(read_counts))
    print("link:", dict(reasons))

    mk = models_db.match_keys(con)
    pk = models_db.player_keys(con)
    our_by_id = {m.match_id: m for m in ours}
    odd_by_key = {}
    for l in links:
        if l.match_id is None or "PS_close" not in l.row.odds:
            continue
        m = our_by_id[l.match_id]
        odd_by_key[mk[l.match_id]] = (l, pk[m.winner], pk[m.loser])

    # ---------------------------------------------------------------- predictions
    tour_keys = {mk[m.match_id]: m for m in ours if m.match_id in mk}
    preds = {}
    for rid in runs:
        preds[rid] = {k: (st, p1, p, y) for k, st, p1, p, y in con.execute(
            """SELECT match_key, edition_start, p1_key, p_p1_wins, p1_won FROM prediction
               WHERE run_id=? AND edition_start BETWEEN ? AND ? AND p1_won IS NOT NULL""",
            (rid, "2021-01-01", a.date_to))}
    in_window = lambda st: a.date_from <= st <= a.date_to                                 # noqa: E731
    base = [k for k, v in preds[runs[0]].items() if k in tour_keys and in_window(v[0])]
    cov = sum(1 for k in base if k in odd_by_key)
    print(f"coverage: {cov}/{len(base)} tour-level matches of run {runs[0]} in window have closing odds "
          f"({cov / max(len(base), 1):.1%})")

    # aligned rows (all years from 2021, for the walk-forward blend); window flag for scoring
    keys = sorted(set(odd_by_key) & set.intersection(*(set(preds[r]) for r in runs)),
                  key=lambda k: (preds[runs[0]][k][0], k))
    keys = [k for k in keys if k in tour_keys]
    st = np.array([preds[runs[0]][k][0] for k in keys])
    y = np.array([preds[runs[0]][k][3] for k in keys], float)
    tour = np.array([tour_keys[k].tour for k in keys])
    P = {LABELS.get(r, f"run{r}"): np.array([preds[r][k][2] for k in keys]) for r in runs}
    # market in p1 orientation
    def side(kind):
        o1, o2, ok = [], [], []
        for k in keys:
            l, wkey, lkey = odd_by_key[k]
            od = l.row.odds.get(kind)
            p1 = preds[runs[0]][k][1]
            if od is None:
                o1.append(np.nan); o2.append(np.nan); ok.append(False); continue
            ow, ol = od
            if p1 == wkey:
                o1.append(ow); o2.append(ol)
            else:
                o1.append(ol); o2.append(ow)
            ok.append(True)
            # sanity: the winner by odds file agrees with our outcome
            assert (p1 == wkey) == (preds[runs[0]][k][3] == 1), k
        return np.array(o1), np.array(o2), np.array(ok)
    c1, c2, _ = side("PS_close")
    o1, o2, has_open = side("PS_open")
    for meth in ("multiplicative", "power", "shin"):
        P[f"close_{meth}"] = odds.devig(c1, c2, meth)
    P["open_power"] = np.where(has_open, odds.devig(np.where(has_open, o1, 2), np.where(has_open, o2, 2), "power"), np.nan)
    win = np.array([in_window(s) for s in st])
    year = np.array([s[:4] for s in st])
    print(f"scored rows (window, all models + close): {win.sum()}  margin median "
          f"{np.median(odds.overround(c1[win], c2[win])):.4f}")

    out = {"source": "valuebetennis CC BY 4.0", "raw": a.raw, "read": dict(read_counts),
           "link": dict(reasons), "coverage": [cov, len(base)], "n": int(win.sum()), "tables": {}}

    def table(name, mask, ref="close_power"):
        print(f"\n## {name} (n = {int(mask.sum())})")
        print(f"{'model':16s} {'logloss':>8s} {'d_vs_mkt':>9s} {'SE':>7s} {'Brier':>7s} {'acc':>6s} {'ECE':>7s}")
        res = {}
        for lab, p in P.items():
            m = mask & ~np.isnan(p)
            if lab == "open_power":
                ref_p = P[ref][m]
            else:
                ref_p = P[ref][m]
            r = metrics(p[m], y[m], ref=ref_p)
            res[lab] = {"n": r.n, "log_loss": r.log_loss, "d": r.d_ref, "se": r.d_se, "brier": r.brier,
                        "acc": r.accuracy, "ece": r.ece}
            print(f"{lab:16s} {r.log_loss:8.4f} {r.d_ref:+9.4f} {r.d_se:7.4f} {r.brier:7.4f} "
                  f"{r.accuracy:6.3f} {r.ece:7.4f}" + ("  (open rows only)" if lab == "open_power" else ""))
        out["tables"][name] = res

    table("tour, 2022-26", win)
    for t in ("atp", "wta"):
        table(f"tour_{t}", win & (tour == t))
    for yr in sorted(set(year[win])):
        table(f"year {yr}", win & (year == yr))

    # ---------------------------------------------------------------- blend
    print("\n## blend: y ~ a*logit(close_power) + b*logit(model)")
    mkt = odds.logit(P["close_power"])
    out["blend"] = {}
    for lab in [LABELS.get(r, f"run{r}") for r in runs]:
        X = np.column_stack([mkt, odds.logit(P[lab])])
        (ba, bb), (sa, sb) = odds.blend_fit(y[win], X[win])
        # walk-forward: for each test year, fit on all earlier rows available
        held, held_mkt, per_year = [], [], {}
        for yr in sorted(set(year[win])):
            tr = (year < yr) & ~np.isnan(X[:, 1])
            te = win & (year == yr)
            if tr.sum() < 2000:
                continue
            (fa, fb), _ = odds.blend_fit(y[tr], X[tr])
            pb = odds.sigmoid(fa * mkt[te] + fb * X[te, 1])
            ll_b, ll_m = odds.log_loss(pb, y[te]), odds.log_loss(P["close_power"][te], y[te])
            per_year[yr] = {"a": fa, "b": fb, "n": int(te.sum()), "gain": ll_m - ll_b}
            d = -(y[te] * np.log(pb) + (1 - y[te]) * np.log(1 - pb)) - \
                -(y[te] * np.log(P["close_power"][te]) + (1 - y[te]) * np.log(1 - P["close_power"][te]))
            held.append(d)
        held = np.concatenate(held) if held else np.array([])
        g = -held.mean() if len(held) else float("nan")
        gse = held.std(ddof=1) / np.sqrt(len(held)) if len(held) > 1 else float("nan")
        print(f"{lab:14s} in-sample a={ba:.3f} b={bb:+.3f} (z={bb / sb:+.1f})   held-out gain "
              f"{g:+.5f} ± {gse:.5f} nats over {len(held)} matches   per year: "
              + ", ".join(f"{k}: b={v['b']:+.2f} gain={v['gain']:+.4f}" for k, v in per_year.items()))
        out["blend"][lab] = {"a": ba, "b": bb, "b_se": sb, "held_gain": g, "held_gain_se": gse,
                             "held_n": int(len(held)), "per_year": per_year}

    # ---------------------------------------------------------------- backtest
    print("\n## flat-stake backtest, bet if p*odds - 1 > threshold (2022-26)")
    out["backtest"] = []
    for lab in [LABELS.get(r, f"run{r}") for r in runs]:
        for kind, (x1, x2, ok) in (("close", (c1, c2, np.ones_like(win))), ("open", (o1, o2, has_open))):
            for thr in (0.0, 0.05):
                m = win & ok.astype(bool)
                bt = odds.backtest(P[lab][m], y[m], x1[m], x2[m], threshold=thr, label=lab)
                print(f"{lab:14s} {kind:5s} thr={thr:.2f}: bets {bt.n_bets:6d} ({bt.n_bets / bt.n_matches:5.1%}) "
                      f"ROI {bt.roi:+.2%} ± {bt.roi_se:.2%}  hit {bt.hit_rate:.3f} @ {bt.mean_odds:.2f}")
                out["backtest"].append({"model": lab, "odds": kind, "threshold": thr, **bt.__dict__})

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
