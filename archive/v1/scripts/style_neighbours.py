"""Most similar players by style (serve + return + rally vectors combined), as JSON.

    python archive/v1/scripts/style_neighbours.py --out neighbours.json [--tour atp] [--as-of 2026-05-26]

Subjects: players with >= --min-matches charted matches and a backbone match in an edition
starting on or after --active-since, ordered by their latest ranking before --as-of.
Candidates: every player of the tour with >= --min-matches charted matches, any era.

Similarity: each side's component scores are divided by the square root of that side's
total score variance over the candidate pool, so serve, return and rally weigh equally,
then concatenated; distance is Euclidean. Per-side nearest players are also listed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ARCHIVE = Path(__file__).resolve().parents[1]   # archive/v1: results, reports, docs
ROOT = Path(__file__).resolve().parents[3]      # project root: tennis.db, data, src
sys.path.insert(0, str(ROOT / "src"))

from tennis import heavy, models_db                 # noqa: E402
from tennis.style_features import SIDES             # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tour", choices=heavy.TOURS, default="atp")
    ap.add_argument("--as-of", default="2026-05-26")
    ap.add_argument("--active-since", default="2024-01-01")
    ap.add_argument("--min-matches", type=int, default=15)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args(argv)
    cfg = heavy.HeavyConfig()
    con = models_db.connect()
    models_db.attach_data(con)
    built = con.execute("SELECT value FROM data.schema_meta WHERE key='built_at'").fetchone()[0]
    model = heavy.StyleModel(heavy.load_counts(con, built_at=built), cfg, heavy.load_fits(con, cfg))
    names = dict(con.execute("SELECT player_id, canonical_name FROM data.player"))
    active = {p for (p,) in con.execute(
        """SELECT x.p FROM (SELECT winner_id p, edition_id FROM data.matches
                            UNION ALL SELECT loser_id, edition_id FROM data.matches) x
           JOIN data.tournament_edition e USING (edition_id) WHERE e.start_date >= ?""",
        (args.active_since,))}
    rank = dict(con.execute(
        """SELECT player_id, rank FROM data.ranking r WHERE tour=? AND ranking_date =
             (SELECT MAX(ranking_date) FROM data.ranking WHERE tour=? AND ranking_date < ?)""",
        (args.tour, args.tour, args.as_of)))

    ids, blocks, n_matches, last_year = [], {s: [] for s in SIDES}, [], []
    for p, t in model.tour_of.items():
        if t != args.tour:
            continue
        svs = {s: model.style(p, args.as_of, s) for s in SIDES}
        if any(v is None for v in svs.values()) or svs["serve"].n_matches < args.min_matches:
            continue
        ids.append(p)
        n_matches.append(svs["serve"].n_matches)
        last_year.append(model.hist[(args.tour, "serve")].player_totals(p, args.as_of)[4][:4])
        for s in SIDES:
            blocks[s].append(svs[s].vector)
    scaled = {}
    for s in SIDES:
        V = np.array(blocks[s])
        scaled[s] = V / np.sqrt(V.var(axis=0).sum())
    combined = np.hstack([scaled[s] for s in SIDES])

    def nearest(M, i, k):
        d = np.linalg.norm(M - M[i], axis=1)
        d[i] = np.inf
        order = np.argsort(d)[:k]
        return [{"name": names[ids[j]], "dist": round(float(d[j]), 3), "last": last_year[j],
                 "matches": n_matches[j]} for j in order]

    subjects = [i for i, p in enumerate(ids) if p in active]
    subjects.sort(key=lambda i: (rank.get(ids[i], 10**6), names[ids[i]]))
    out = {"tour": args.tour, "as_of": args.as_of, "active_since": args.active_since,
           "min_matches": args.min_matches, "pool": len(ids), "players": []}
    for i in subjects:
        out["players"].append({
            "name": names[ids[i]], "rank": rank.get(ids[i]), "matches": n_matches[i],
            "overall": nearest(combined, i, args.k),
            **{s: nearest(scaled[s], i, 1)[0] for s in SIDES}})
    args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.out}: {len(out['players'])} players, pool {len(ids)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
