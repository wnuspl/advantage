"""Does in-match serve form predict the rest of the match? Two tests on the point-by-point
matches (slam pbp + MCP, 2011+), against a points_v1 run's walk-forward pre-match logits.

    python archive/v1/scripts/inplay_form_test.py [--run 790] [--rebuild]

Test 1, persistence. For each player in each match, residuals of their serve points
against the pre-match expectation p = sigmoid(x), in logit units r = (k/n - p) / (p (1-p)),
for two halves of their serve points. With independent halves given a shared deviation
e (rating error + today's form), Cov(r1, r2) = Var(e). Two splits:
  odd/even points  -> everything shared within the match;
  first/second half -> only what persists from early to late in the match.
The model's in-match prior says Var(e) = v (rating variance, pre-match) + obs_var.
Covariances are centred (a global bias of p on this sample is not counted as form).

Test 2, prediction. At the end of every set that does not end the match, P(p1 wins)
(p1 = canonical player order, never "winner first") from:
  prematch    pre-match probability, no score
  score       markov.match_from on the set score, pre-match logits
  form f      each serve logit first updated on that player's serve points so far,
              points_rating.in_match_form with form variance f (f = run's obs_var is what
              kalshi_edges does live)
Log loss per variant; f chosen on 2011-2018, reported on 2019-21 and 2022-26. Paired
differences vs 'score' with SEs clustered by match.

Pre-match logits and variances are the run's own walk-forward values (state before the
match). Only completed matches whose point data reproduces the result (every set's
last point won by that set's winner, the match winner taking the right number of sets).
Reads only. Output: stdout + results/inplay_form_test.json.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ARCHIVE = Path(__file__).resolve().parents[1]   # archive/v1: results, reports, docs
ROOT = Path(__file__).resolve().parents[3]      # project root: tennis.db, data, src
sys.path.insert(0, str(ROOT / "src"))

from tennis import models_db, points_rating, tuning  # noqa: E402

FORMS = (0.0, 0.001, 0.003, 0.01, 0.02, 0.05, 0.1)
PERIODS = {"fit 2011-18": ("2011", "2018"), "val 2019-21": ("2019", "2021"), "test 2022-26": ("2022", "2026")}


def sig(x):
    return 1 / (1 + math.exp(-x))


def load_prematch(con, run: int, rebuild: bool):
    """Per-match pre-match arrays from the replay (cached in models/inplay/)."""
    path = ROOT / "models" / "inplay" / f"prematch_run{run}.npz"
    prep = tuning.prepare(con)
    m = prep.m
    if path.exists() and not rebuild:
        z = np.load(path)
        if len(z["match_id"]) == len(m) and int(z["match_id"][-1]) == int(m.match_id[-1]):
            print(f"pre-match logits: cached {path.relative_to(ROOT)}")
            return prep, {k: z[k] for k in z.files}
    spec, params = tuning.model_of(con, run)
    spec.attach(con, m)
    t = time.perf_counter()
    print(f"replaying run {run} over {len(m):,} matches ...")
    res = points_rating.run(m, params)
    b = params.blend
    out = {"match_id": np.asarray(m.match_id, dtype=np.int64),
           "xw": np.asarray(res.xa) + b * np.asarray(res.xa_s), "xl": np.asarray(res.xb) + b * np.asarray(res.xb_s),
           "vw": np.asarray(res.va) + b * np.asarray(res.va_s), "vl": np.asarray(res.vb) + b * np.asarray(res.vb_s),
           "fmt": np.asarray(res.fmt, dtype=np.int64), "p_winner": np.asarray(res.p_winner)}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)
    print(f"  done in {time.perf_counter() - t:.0f}s")
    return prep, out


def load_points(con, ids: set) -> dict:
    """match_id -> [(set_idx, server_id, winner_id)] from the source with the most points."""
    by = defaultdict(lambda: defaultdict(list))
    for mid, src, s, srv, win in con.execute(
            "SELECT match_id, data_source, set_idx, server_id, winner_id FROM data.point ORDER BY match_id, data_source, point_idx"):
        if mid in ids:
            by[mid][src].append((s, srv, win))
    return {mid: max(d.values(), key=len) for mid, d in by.items()}


def check(pts, w, l, need):
    """Per-set lists [(server, won_by_server)] if the points reproduce the result, else None."""
    sets, cur, cur_idx = [], [], None
    for s, srv, win in pts:
        if srv not in (w, l) or win not in (w, l):
            return None
        if s != cur_idx:
            if cur:
                sets.append((cur_idx, cur))
            cur, cur_idx = [], s
        cur.append((srv, win))
    if cur:
        sets.append((cur_idx, cur))
    if [k for k, _ in sets] != list(range(1, len(sets) + 1)):
        return None
    sw = sl = 0
    for k, (_, ps) in enumerate(sets):
        if max(sw, sl) >= need:
            return None
        last = ps[-1][1]
        sw, sl = sw + (last == w), sl + (last == l)
    if sw != need or sl >= need:
        return None
    return [ps for _, ps in sets]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=int, default=790)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--json", type=Path, default=ARCHIVE / "results" / "inplay_form_test.json")
    args = ap.parse_args(argv)

    con = models_db.connect(models_db.MODELS_DB)
    models_db.attach_data(con, models_db.DATA_DB)
    try:
        spec, params = tuning.model_of(con, args.run)
        prep, pm = load_prematch(con, args.run, args.rebuild)
        m = prep.m
        status = dict(con.execute("SELECT match_id, status FROM data.matches"))
        idx = {int(mid): i for i, mid in enumerate(pm["match_id"])}
        cand = {mid for mid, i in idx.items() if m.start[i] >= "2011" and status.get(mid) == "completed"}
        t = time.perf_counter()
        pts = load_points(con, cand)
        print(f"point data: {len(pts):,} completed matches from 2011 ({time.perf_counter() - t:.0f}s)")
    finally:
        con.close()

    ov = params.obs_var
    st = points_rating.PointsState(params, *({} for _ in range(13)))
    t1 = {"time": [], "oddeven": []}           # (contribution r1*r2, r1, r2, v + ov, year, tour)
    rows = []                                   # test 2 checkpoints
    n_bad = 0
    t = time.perf_counter()
    for mid, plist in pts.items():
        i = idx[mid]
        w, l = m.winner[i], m.loser[i]
        bo, final = points_rating.FORMATS[int(pm["fmt"][i])]
        need = bo // 2 + 1
        sets = check(plist, w, l, need)
        if sets is None:
            n_bad += 1
            continue
        x = {w: float(pm["xw"][i]), l: float(pm["xl"][i])}
        v = {w: float(pm["vw"][i]), l: float(pm["vl"][i])}
        year, tour = m.start[i][:4], m.tour[i]

        # test 1
        for pl in (w, l):
            seq = [int(win == pl) for ps in sets for srv, win in ps if srv == pl]
            if len(seq) < 40:
                continue
            p = sig(x[pl])
            pq = p * (1 - p)
            h = len(seq) // 2
            for kind, (s1, s2) in (("time", (seq[:h], seq[h:])), ("oddeven", (seq[0::2], seq[1::2]))):
                r1 = (sum(s1) / len(s1) - p) / pq
                r2 = (sum(s2) / len(s2) - p) / pq
                t1[kind].append((r1, r2, v[pl] + ov, v[pl], year, tour))

        # test 2
        p1 = w if prep.winner_is_p1[i] else l
        p2 = l if p1 == w else w
        y = int(p1 == w)
        x1, x2, v1, v2 = x[p1], x[p2], v[p1], v[p2]
        pre = points_rating.win_probability(st, x1, x2, v1, v2, bo, final)
        s1 = s2 = 0
        sv = {p1: [0, 0], p2: [0, 0]}                     # serve points [won, played]
        for k, ps in enumerate(sets[:-1]):
            for srv, win in ps:
                sv[srv][0] += win == srv
                sv[srv][1] += 1
            s1, s2 = s1 + (ps[-1][1] == p1), s2 + (ps[-1][1] == p2)
            score = dict(sets_a=s1, sets_b=s2, games_a=0, games_b=0, pts_a=0, pts_b=0)
            pr = {"prematch": pre,
                  "score": points_rating.in_play_probability(st, x1, x2, v1, v2, bo, final, score, True)}
            for f in FORMS:
                a1, b1 = points_rating.in_match_form(x1, v1, *sv[p1], f)
                a2, b2 = points_rating.in_match_form(x2, v2, *sv[p2], f)
                pr[f"form {f:g}"] = points_rating.in_play_probability(st, a1, a2, b1, b2, bo, final, score, True)
            rows.append({"mid": mid, "year": year, "tour": tour, "bo": bo, "after_set": k + 1, "y": y, "p": pr})
    print(f"checked {len(pts):,} matches: {n_bad:,} dropped (points do not reproduce the result); "
          f"{len(rows):,} set-end checkpoints ({time.perf_counter() - t:.0f}s)")

    out = {"run": args.run, "obs_var": ov, "unc": params.unc, "matches": len(pts) - n_bad, "dropped": n_bad}

    # ---- test 1
    print(f"\nTest 1: shared serve deviation within a match (logit^2); model prior v + obs_var, obs_var = {ov:g}")
    print(f"  {'split':10} {'sample':12} {'n':>6} {'Cov(r1,r2)':>11} {'se':>8} {'raw E[r1 r2]':>12}  {'model v+ov':>10} {'v':>8}")
    out["test1"] = {}
    for kind, data in t1.items():
        for lab, sel in (("all", lambda r: True), ("atp", lambda r: r[5] == "atp"), ("wta", lambda r: r[5] == "wta"),
                         ("2011-18", lambda r: r[4] <= "2018"), ("2019-26", lambda r: r[4] >= "2019")):
            d = np.array([r[:4] for r in data if sel(r)])
            if len(d) < 50:
                continue
            r1, r2 = d[:, 0], d[:, 1]
            cov = float(np.mean((r1 - r1.mean()) * (r2 - r2.mean())))
            prod = (r1 - r1.mean()) * (r2 - r2.mean())
            se = float(prod.std(ddof=1) / math.sqrt(len(d)))
            raw = float(np.mean(r1 * r2))
            print(f"  {kind:10} {lab:12} {len(d):6d} {cov:11.4f} {se:8.4f} {raw:12.4f}  {d[:, 2].mean():10.4f} {d[:, 3].mean():8.4f}")
            out["test1"][f"{kind} {lab}"] = {"n": len(d), "cov": cov, "se": se, "raw": raw,
                                             "model_v_plus_ov": float(d[:, 2].mean()), "v": float(d[:, 3].mean())}

    # ---- test 2
    variants = list(rows[0]["p"]) if rows else []

    def ll(p, y):
        p = min(max(p, 1e-12), 1 - 1e-12)
        return -math.log(p if y else 1 - p)

    def table(sel, label):
        rs = [r for r in rows if sel(r)]
        if not rs:
            return None
        res = {"n": len(rs), "matches": len({r["mid"] for r in rs})}
        base = {}
        for vname in variants:
            per = defaultdict(float)
            for r in rs:
                per[r["mid"]] += ll(r["p"][vname], r["y"]) - ll(r["p"]["score"], r["y"])
            tot = sum(ll(r["p"][vname], r["y"]) for r in rs) / len(rs)
            dif = np.array(list(per.values()))
            res[vname] = {"log_loss": tot, "d_vs_score": float(dif.sum() / len(rs)),
                          "se": float(dif.std(ddof=1) * math.sqrt(len(dif)) / len(rs)) if len(dif) > 1 else 0.0}
            base[vname] = res[vname]
        return res

    print("\nTest 2: log loss at set ends (lower is better); d = vs score-only, paired, SE clustered by match")
    out["test2"] = {}
    for pname, (a, b) in PERIODS.items():
        for lab, sel in (("all", lambda r: True), ("atp", lambda r: r["tour"] == "atp"), ("wta", lambda r: r["tour"] == "wta")):
            res = table(lambda r, a=a, b=b, sel=sel: a <= r["year"] <= b and sel(r), f"{pname} {lab}")
            if res is None:
                continue
            out["test2"][f"{pname} {lab}"] = res
            if lab != "all":
                continue
            print(f"\n  {pname}: {res['n']:,} checkpoints, {res['matches']:,} matches")
            for vname in variants:
                e = res[vname]
                print(f"    {vname:12} {e['log_loss']:.4f}   d {e['d_vs_score']:+.4f} ± {e['se']:.4f}")
    fit = out["test2"].get("fit 2011-18 all", {})
    forms = [v for v in variants if v.startswith("form")]
    if fit and forms:
        best = min(forms, key=lambda v: fit[v]["log_loss"])
        out["best_form_on_fit"] = best
        print(f"\n  best form variance on 2011-18: {best} (the run's obs_var is {ov:g})")
        for pname in ("val 2019-21", "test 2022-26"):
            for lab in ("all", "atp", "wta"):
                e = out["test2"].get(f"{pname} {lab}")
                if e:
                    print(f"    {pname} {lab:3}: {best} d {e[best]['d_vs_score']:+.4f} ± {e[best]['se']:.4f}; "
                          f"form {ov:g} d {e[f'form {ov:g}']['d_vs_score']:+.4f} ± {e[f'form {ov:g}']['se']:.4f}")
    by_set = {}
    for k in sorted({r["after_set"] for r in rows}):
        res = table(lambda r, k=k: r["year"] >= "2019" and r["after_set"] == k, f"after set {k}")
        if res:
            by_set[k] = res
    out["by_set_2019_26"] = by_set
    if forms and by_set:
        print("\n  2019-26 by checkpoint:")
        for k, res in by_set.items():
            print(f"    after set {k}: n {res['n']:,}; score {res['score']['log_loss']:.4f}; "
                  + "; ".join(f"{v.split()[1]} {res[v]['d_vs_score']:+.4f}±{res[v]['se']:.4f}" for v in forms))

    args.json.parent.mkdir(exist_ok=True)
    args.json.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwritten {args.json.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
