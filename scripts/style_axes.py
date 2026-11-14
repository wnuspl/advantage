"""Players at each end of every style axis, for the modern game (JSON for the axis page).

    python scripts/style_axes.py --out axes.json [--as-of 2026-05-26] [--active-since 2024-01-01]

Uses the latest fit per tour/side as of --as-of. Players qualify with >= --min-matches charted
matches (whole career, as the vectors are) and a backbone match in an edition starting on or
after --active-since.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tennis import heavy, models_db                 # noqa: E402
from tennis.style_features import SIDES             # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--as-of", default="2026-05-26")
    ap.add_argument("--active-since", default="2024-01-01")
    ap.add_argument("--min-matches", type=int, default=15)
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args(argv)
    cfg = heavy.HeavyConfig()
    con = models_db.connect()
    models_db.attach_data(con)
    built = con.execute("SELECT value FROM data.schema_meta WHERE key='built_at'").fetchone()[0]
    model = heavy.StyleModel(heavy.load_counts(con, built_at=built), cfg, heavy.load_fits(con, cfg))
    names = dict(con.execute("SELECT player_id, canonical_name FROM data.player"))
    active = {p for (p,) in con.execute(
        """SELECT winner_id FROM data.matches m JOIN data.tournament_edition e USING (edition_id)
           WHERE e.start_date >= ? UNION SELECT loser_id FROM data.matches m
           JOIN data.tournament_edition e USING (edition_id) WHERE e.start_date >= ?""",
        (args.active_since, args.active_since))}
    out = {"as_of": args.as_of, "active_since": args.active_since, "min_matches": args.min_matches,
           "axes": []}
    for tour in heavy.TOURS:
        for side in SIDES:
            fit = model.fit_for(tour, side, args.as_of)
            rows = []
            for p, t in model.tour_of.items():
                if t != tour or p not in active:
                    continue
                sv = model.style(p, args.as_of, side)
                if sv and sv.n_matches >= args.min_matches:
                    rows.append((names[p], sv.n_matches, sv.vector))
            V = np.array([r[2] for r in rows])
            L = fit.loadings()
            for c in range(fit.n_components):
                order = np.argsort(-V[:, c])
                fmt = lambda i: {"name": rows[i][0], "matches": rows[i][1], "score": round(float(V[i, c]), 2)}
                load = sorted(zip(fit.features, L[:, c]), key=lambda x: -abs(x[1]))
                out["axes"].append({
                    "tour": tour, "side": side, "component": c + 1,
                    "explained": round(float(fit.explained[c]), 4), "fit_id": fit.fit_id,
                    "pool": len(rows),
                    "loadings": [[f, round(float(v), 2)] for f, v in load[:6]],
                    "high": [fmt(i) for i in order[:args.top]],
                    "low": [fmt(i) for i in order[::-1][:args.top]]})
    args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.out}: {len(out['axes'])} axes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
