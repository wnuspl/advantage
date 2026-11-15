"""Milestone 9 test scoring (run once, 2026-11-14): TEST 2022-26, scored once. Run 790 (context inside points_v1) vs run 694, and vs the
match-level layer on calibrated 694 (val-selected spec, fit on tour-level 2005-2021)."""
import json, sys
from pathlib import Path
import numpy as np
ARCHIVE = Path(__file__).resolve().parents[1]   # archive/v1: results, reports, docs
ROOT = Path(__file__).resolve().parents[3]      # project root: tennis.db, data, src
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ARCHIVE / "scripts"))
import context_report as cr
from tennis import calibrate, models_db, tuning
from tennis.composite_eval import block_of, metrics

con = models_db.connect(); models_db.attach_data(con)
prep = tuning.prepare(con); m = prep.m
pw = {}
for rid in (694, 790):
    spec, p = tuning.model_of(con, rid); spec.attach(con, m)
    pw[rid] = spec.predict(spec.run(m, p), p); print("replayed", rid, flush=True)
bo_all = calibrate.best_of_of(m)
ed, mt, pl = cr.load_info(con); con.close()
F = cr.build_features(m, ed, mt, pl, cr.host_map())

cal = {r: calibrate.calibrated(prep, pw[r], "2005-01-01", "2026-12-31")[0] for r in pw}
_, y, idx, _ = calibrate.calibrated(prep, pw[694], "2005-01-01", "2026-12-31")
calbo = {r: calibrate.calibrated(prep, pw[r], "2005-01-01", "2026-12-31", best_of=bo_all)[0] for r in pw}
idx = np.array(idx)
raw = {r: calibrate.oriented(prep, pw[r], idx)[0] for r in pw}
p1_is_w = np.array([prep.winner_is_p1[i] for i in idx], dtype=bool)
starts = np.array([m.start[i] for i in idx]); tours = np.array([m.tour[i] for i in idx])
rows_meta = [mt[m.match_id[i]] for i in idx]; eds = [ed[r[0]] for r in rows_meta]
bo = np.array([r[5] or e[4] or 3 for r, e in zip(rows_meta, eds)])
X, names = cr.design(F["w"], F["l"], idx, p1_is_w, starts, bo)
cols = ["gap 1wk", "gap 13-26wk", "gap >26wk", "1st match back", "2-5 matches back", "load28 /5",
        "age<21", "age 21-22", "age 32+", "career<50"]
Xs = X[:, [names.index(c) for c in cols]]
train, test = starts < "2022-01-01", starts >= "2022-01-01"
blocks = block_of(starts)

def layer(base):
    off = cr.logit(base); Xb = np.column_stack([Xs, (bo == 5) * off])
    b, _ = cr.fit_offset_logit(Xb[train], off[train], y[train], l2=10.0)
    return 1 / (1 + np.exp(-(off + Xb @ b)))

P = {"694 cal": cal[694], "694 raw": raw[694], "694 cal bo": calbo[694],
     "bar (layer on 694 cal)": layer(cal[694]),
     "790 cal": cal[790], "790 raw": raw[790], "790 cal bo": calbo[790],
     "790 cal + layer": layer(cal[790])}
pairs = [("790 cal", "694 cal"), ("790 raw", "694 raw"), ("790 cal bo", "694 cal bo"),
         ("bar (layer on 694 cal)", "694 cal"), ("790 cal", "bar (layer on 694 cal)"),
         ("790 cal + layer", "790 cal")]
out = {}
for scope, sel in (("tour", test), ("atp", test & (tours == "atp")), ("wta", test & (tours == "wta"))):
    print(f"\n[{scope}] n={sel.sum():,}")
    print("  " + "  ".join(f"{k} {metrics(v[sel], y[sel]).log_loss:.4f}" for k, v in P.items() if k in ("694 cal", "694 raw", "790 cal", "790 raw")))
    out[scope] = {}
    for a, b in pairs:
        r = metrics(P[a][sel], y[sel], ref=P[b][sel], block=blocks[sel])
        out[scope][f"{a} vs {b}"] = {"ll": r.log_loss, "d": r.d_ref, "se": r.d_se, "q": [r.blocks_won, r.blocks]}
        print(f"  {a:>24} vs {b:<24} ll {r.log_loss:.4f}  diff {r.d_ref:+.4f} ± {r.d_se:.4f}  quarters {r.blocks_won}/{r.blocks}")
(ARCHIVE / "results" / "context_test.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print("wrote results/context_test.json")
