"""End-to-end predictions and accuracy of the composite model (light + heavy + matchup).

    # one match: P(A beats B), with each layer's contribution
    python archive/v1/scripts/composite.py predict "Jannik Sinner" "Carlos Alcaraz" --date 2026-06-01 --surface grass
    python archive/v1/scripts/composite.py predict sackmann_atp:206173 sackmann_atp:207989 --surface clay --model siamese

    # accuracy: every matchup model of the latest matchup run vs surface Elo (132) and Glicko-2 (187),
    # on identical matches; writes docs/composite_report.md
    python archive/v1/scripts/composite.py eval
    python archive/v1/scripts/composite.py eval --sweep matchup-20261114T010000 --ref glicko2_raw

    # consistency: rebuild stored walk-forward predictions through the composite chain
    python archive/v1/scripts/composite.py check --model bilinear --dates 3

predict uses the matchup run's bundle whose test block contains --date (after the last
block: the latest bundle). The Glicko-2 state covers editions starting before --date,
so a date past the end of the data (2026-05-25 on the current build) predicts from the
last ratings with RDs grown by the gap. Needs tennis.db, models.db, heavy-layer fits
(scripts/style_fit.py fit) and a saved matchup run (scripts/matchup.py run).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

ARCHIVE = Path(__file__).resolve().parents[1]   # archive/v1: results, reports, docs
ROOT = Path(__file__).resolve().parents[3]      # project root: tennis.db, data, src
sys.path.insert(0, str(ROOT / "src"))

import numpy as np                                                 # noqa: E402

from tennis import composite, composite_eval, elo, models_db      # noqa: E402
from tennis.matchup import data as mdata                           # noqa: E402

REPORT = ARCHIVE / "docs" / "composite_report.md"
BASELINES = {"elo_surface": 132, "glicko2_raw": 187}               # CLAUDE.md milestone 5


def _connect(args):
    con = models_db.connect(args.models_db)
    models_db.attach_data(con, args.data)
    return con


def _run_id(con, args) -> int:
    return args.run or composite.latest_matchup_run(con, args.model)


# ------------------------------------------------------------------ predict

def cmd_predict(args) -> int:
    con = _connect(args)
    run_id = _run_id(con, args)
    on = args.date or date.today().isoformat()
    a_id, a_name, a_tour = composite.resolve_player(con, args.player_a, args.tour)
    b_id, b_name, b_tour = composite.resolve_player(con, args.player_b, args.tour or a_tour)
    if a_tour != b_tour:
        raise SystemExit(f"{a_name} is {a_tour}, {b_name} is {b_tour}: no cross-tour predictions")
    last = con.execute("SELECT MAX(start_date) FROM data.tournament_edition").fetchone()[0]
    t0 = time.perf_counter()
    pred = composite.load_predictor(con, run_id, on)
    p = pred.predict(a_id, b_id, on, args.surface, a_tour)
    took = time.perf_counter() - t0
    if args.json:
        print(json.dumps({**p.as_dict(), "a_name": a_name, "b_name": b_name}, indent=2, default=float))
        return 0
    la, lb = p.light["a"], p.light["b"]
    print(f"{a_name} vs {b_name}  ({a_tour.upper()}, {args.surface or 'surface unknown'}, {on})")
    if on > last:
        print(f"  note: data ends with editions starting {last}; ratings are as of then, RDs grown to {on}")
    print(f"  matchup run {run_id} ({p.model}), bundle test block {p.bundle['test_from']}..{p.bundle['test_to']}, "
          f"trained before {p.bundle['train_cutoff']}")
    print("\n  light (Glicko-2)   rating    RD   surface rating   matches")
    for nm, l in ((a_name, la), (b_name, lb)):
        sr = "" if l["surface_rating"] != l["surface_rating"] else f"{l['surface_rating']:7.1f} ±{l['surface_rd']:5.1f}"
        print(f"    {nm[:16]:16s} {l['rating']:7.1f} {l['rd']:5.1f}   {sr:16s} {l['matches']:6d}"
              + ("" if l["rated"] else "   (unrated)"))
    print(f"\n  heavy (style)      charted matches before {on}: {p.n_charted_a} / {p.n_charted_b}"
          + ("" if p.has_vectors else "  -> no matchup adjustment (a player has no style vector)"))
    print("\n  P(%s wins)" % a_name)
    print(f"    Glicko-2 raw              {p.p_glicko_raw:.3f}")
    print(f"    calibrated (alpha {p.alpha:.3f})  {p.p_glicko:.3f}")
    print(f"    matchup delta             {p.delta:+.3f} logit")
    print(f"    composite                 {p.p:.3f}   (fair odds {1 / p.p:.2f} / {1 / (1 - p.p):.2f})")
    print(f"\n  ({took:.0f}s)")
    return 0


# ------------------------------------------------------------------ eval

def _matchup_runs(con, sweep: str | None) -> tuple[str, dict]:
    if sweep is None:
        row = con.execute("""SELECT sweep_id FROM model_run WHERE model_id LIKE 'matchup\\_%' ESCAPE '\\'
                             AND finished_at IS NOT NULL AND EXISTS (SELECT 1 FROM prediction p WHERE p.run_id=model_run.run_id)
                             ORDER BY run_id DESC LIMIT 1""").fetchone()
        if row is None:
            raise SystemExit("no finished matchup run with saved predictions; run scripts/matchup.py run first")
        sweep = row[0]
    rows = con.execute("""SELECT run_id, model_id FROM model_run WHERE sweep_id=? AND model_id LIKE 'matchup\\_%' ESCAPE '\\'
                          AND finished_at IS NOT NULL ORDER BY run_id""", (sweep,)).fetchall()
    out = {}
    for rid, mid in rows:
        name = mid[len("matchup_"):].rsplit("_", 1)[0]
        out["matchup_" + name] = rid
    if not out:
        raise SystemExit(f"no finished matchup runs with sweep_id {sweep!r}")
    return sweep, out


def cmd_eval(args) -> int:
    con = _connect(args)
    sweep, mruns = _matchup_runs(con, args.sweep)
    runs = {lab: rid for lab, rid in BASELINES.items() if lab not in args.drop}
    runs.update({lab: rid for lab, rid in mruns.items() if lab not in args.drop})
    params = json.loads(con.execute("SELECT params_json FROM model_run WHERE run_id=?",
                                    (next(iter(mruns.values())),)).fetchone()[0])["harness"]
    if params["glicko_run_id"] != BASELINES["glicko2_raw"]:
        print(f"note: matchup sweep used Glicko-2 run {params['glicko_run_id']}, not {BASELINES['glicko2_raw']}")
    d_from = args.date_from or params["test_from"]
    d_to = args.date_to or params["test_to"]
    ref = args.ref if args.ref in runs else ("matchup_glicko" if "matchup_glicko" in runs else "glicko2_raw")
    print(f"matchup sweep {sweep}: " + ", ".join(f"{k}={v}" for k, v in runs.items()) + f"; reference {ref}")
    t0 = time.perf_counter()
    keys, dates, y, p, k1, k2 = composite_eval.load_predictions(con, runs, d_from, d_to)
    print(f"{len(keys):,} matches predicted by every run ({time.perf_counter() - t0:.0f}s)")
    t0 = time.perf_counter()
    src = mdata.load_style_source(con)
    att = composite_eval.attach_attributes(con, keys, dates, k1, k2, src)
    print(f"attributes and charted counts ({time.perf_counter() - t0:.0f}s)")
    rows = composite_eval.Rows(match_key=keys, date=dates, y=y, p=p, block=composite_eval.block_of(dates),
                               **att)
    min_charted = args.min_charted or params.get("min_charted", 10)
    res = composite_eval.evaluate(rows, ref, min_charted)
    notes = [f"Matchup sweep `{sweep}`; harness config `{json.dumps(params)}`."]
    text = composite_eval.report(rows, res, ref, runs, min_charted=min_charted,
                                 title="Composite model: accuracy against the baselines", notes=notes)
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"report -> {out}\n")
    for sc in ("tour", "tour_both", "tour_well"):
        if sc not in res:
            continue
        print(f"{sc} (n={res[sc][ref].n:,})")
        for lab in runs:
            m = res[sc][lab]
            print(f"  {lab:22s} ll {m.log_loss:.4f}  d {m.d_ref:+.4f} ± {m.d_se:.4f}  brier {m.brier:.4f}  "
                  f"ece {m.ece:.4f}  slope {m.cal_slope:.3f}  {composite_eval.verdict(res, lab, ref, sc)}")
    return 0


# ------------------------------------------------------------------ check

def cmd_check(args) -> int:
    con = _connect(args)
    run_id = _run_id(con, args)
    bundles, params = composite.load_bundles(con, run_id)
    m = elo.load_matches(con)
    h = params["harness"]
    starts = sorted({s for s in m.start if h["test_from"] <= s <= h["test_to"]})
    if not starts:
        raise SystemExit("no edition starts inside the run's test window")
    pick = np.linspace(0, len(starts) - 1, args.dates).round().astype(int)
    dates = [starts[i] for i in sorted(set(pick))] if not args.date else args.date
    print(f"run {run_id} ({params['model']}): replaying {len(dates)} edition start(s): {', '.join(dates)}")
    src = mdata.load_style_source(con)
    res = composite.check_replay(con, run_id, dates, m=m, src=src)
    ok = res.n > 0 and res.max_abs_diff < args.tol and res.max_abs_diff_light < args.tol
    print(f"{'OK' if ok else 'MISMATCH'}: {res.n} matches, max |composite - stored| {res.max_abs_diff:.2e}, "
          f"max |light - run {h['glicko_run_id']}| {res.max_abs_diff_light:.2e} (tol {args.tol})")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-db", type=Path, default=models_db.MODELS_DB)
    ap.add_argument("--data", type=Path, default=models_db.DATA_DB)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("predict", help="P(A beats B) on a date and surface")
    p.add_argument("player_a")
    p.add_argument("player_b")
    p.add_argument("--date", help="match date, ISO (default today)")
    p.add_argument("--surface", choices=composite.SURFACES)
    p.add_argument("--tour", choices=["atp", "wta"], help="disambiguate names")
    p.add_argument("--model", default="bilinear", help="matchup model whose latest run to use (default bilinear)")
    p.add_argument("--run", type=int, help="matchup model_run id (overrides --model)")
    p.add_argument("--json", action="store_true")

    e = sub.add_parser("eval", help="composite vs baselines on identical matches")
    e.add_argument("--sweep", help="matchup run tag (sweep_id; default: the latest)")
    e.add_argument("--ref", default="matchup_glicko", help="reference label for paired differences")
    e.add_argument("--drop", nargs="*", default=[], help="labels to leave out")
    e.add_argument("--date-from")
    e.add_argument("--date-to")
    e.add_argument("--min-charted", type=int)
    e.add_argument("--report", default=str(REPORT))

    c = sub.add_parser("check", help="rebuild stored walk-forward predictions through the composite chain")
    c.add_argument("--model", default="bilinear")
    c.add_argument("--run", type=int)
    c.add_argument("--dates", type=int, default=3, help="edition starts to replay, spread over the test window")
    c.add_argument("--date", nargs="*", help="explicit edition start dates instead")
    c.add_argument("--tol", type=float, default=1e-5)

    args = ap.parse_args(argv)
    return {"predict": cmd_predict, "eval": cmd_eval, "check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
