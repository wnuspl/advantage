"""Matchup layer: walk-forward run of the interaction models against the Glicko-2 benchmark.

    python scripts/matchup.py run                                  # every model, both tours
    python scripts/matchup.py run --models glicko bilinear --tours atp --test-from 2024-01-01
    python scripts/matchup.py run --no-save                        # print + report only
    python scripts/matchup.py run --models glicko siamese --base-run 605 --alpha-rows tour         --test-from 2019-01-01 --test-to 2021-12-31 --report docs/matchup_points_val.md   # 10b

Needs tennis.db, the base rating run (default Glicko-2 run 187; any ratingmodels run, e.g.
points_v1 605) in models.db, and heavy-layer fits
(scripts/style_fit.py fit). Saves one model_run per model (sweep_id = the run tag),
per-fold bundles under models/matchup/, and writes docs/matchup_report.md.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tennis import heavy, models_db                            # noqa: E402
from tennis.matchup import data, harness                       # noqa: E402
from tennis.matchup.models import MODEL_NAMES                  # noqa: E402

REPORT = ROOT / "docs" / "matchup_report.md"


def cmd_run(args) -> int:
    con = models_db.connect(args.models_db)
    models_db.attach_data(con, args.data)
    hcfg = heavy.HeavyConfig()
    cfg = harness.HarnessConfig(glicko_run_id=args.glicko_run, test_from=args.test_from, test_to=args.test_to,
                                step_months=args.step_months, val_months=args.val_months,
                                alpha_years=args.alpha_years, alpha_rows=args.alpha_rows, train_scope=args.train_scope,
                                min_charted=args.min_charted, tours=tuple(args.tours),
                                heavy_config=hcfg.config_hash)
    t0 = time.perf_counter()
    table = data.load_table(con, cfg.glicko_run_id)
    print(f"match table: {len(table):,} scored matches, base {table.base_model_id} run {cfg.glicko_run_id} "
          f"({time.perf_counter() - t0:.0f}s)")
    t0 = time.perf_counter()
    src = data.load_style_source(con, hcfg)
    print(f"style source: heavy config {hcfg.config_hash}, {sum(len(v) for v in src.model.fits.values())} fits "
          f"({time.perf_counter() - t0:.0f}s)")
    res = harness.run(table, src, cfg, args.model_list)
    tag = args.tag or "matchup-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_ids = None
    if not args.no_save:
        run_ids = harness.save(con, table, res, sweep_id=tag, notes=args.notes,
                               save_predictions=not args.no_predictions)
        print(f"saved runs ({tag}): {run_ids}")
    text = harness.report(table, res, run_ids)
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"report -> {out}")
    pooled = harness.metrics(table, res)
    for sc in ("tour", "tour_both", "tour_well"):
        print(f"\n{sc}: " + "  ".join(f"{m} {pooled[m][sc].log_loss:.4f}" for m in res.models)
              + f"  (n={pooled[res.models[0]][sc].n:,})")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models-db", type=Path, default=models_db.MODELS_DB)
    ap.add_argument("--data", type=Path, default=models_db.DATA_DB)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--models", dest="model_list", nargs="+", default=list(MODEL_NAMES), choices=MODEL_NAMES)
    r.add_argument("--tours", nargs="+", default=["atp", "wta"], choices=["atp", "wta"])
    r.add_argument("--base-run", "--glicko-run", dest="glicko_run", type=int, default=187,
                   help="rating run giving the base logit (Glicko-2 187, points_v1 605, ...)")
    r.add_argument("--test-from", default="2022-01-01")
    r.add_argument("--test-to", default="2026-12-31")
    r.add_argument("--step-months", type=int, default=3)
    r.add_argument("--val-months", type=int, default=6)
    r.add_argument("--alpha-years", type=int, default=10)
    r.add_argument("--alpha-rows", choices=["train", "scope", "tour"], default="train")
    r.add_argument("--train-scope", choices=["all", "tour"], default="all")
    r.add_argument("--min-charted", type=int, default=10)
    r.add_argument("--tag", help="sweep_id for the saved runs")
    r.add_argument("--notes")
    r.add_argument("--no-save", action="store_true")
    r.add_argument("--no-predictions", action="store_true", help="save metrics and bundles, not per-match rows")
    r.add_argument("--report", default=str(REPORT))
    args = ap.parse_args(argv)
    return {"run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
