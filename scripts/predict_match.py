"""Predict one match with a points_v1 run: win probability plus the likely scorelines.

    python scripts/predict_match.py "Sinner" "Alcaraz"
    python scripts/predict_match.py "Swiatek" "Sabalenka" --surface clay --date 2026-06-01
    python scripts/predict_match.py "Sinner" "Alcaraz" --best-of 5 --final tb --top 8

The run (default 694, the current baseline) is replayed over every edition starting
before --date, so the ratings are the ones the model would have used that day. Names
match the player table case-insensitively (substring; the busiest match wins ties).

Two numbers can differ slightly: the headline win probability includes the model's
uncertainty shrink toward 50% (what is scored), while the scorelines and set counts come
straight from the two serve-point probabilities (the "simulated sets"), so they add up
to the unshrunk probability shown next to it. The scorelines are not tuned or
calibrated: they assume every point is independent, so treat them as the model's
picture of the match, not a calibrated forecast. Reads only.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tennis import elo, markov, models_db, points_rating, tuning   # noqa: E402


def find_player(con, name: str) -> tuple[int, str, str]:
    rows = con.execute(
        """SELECT p.player_id, p.canonical_name, p.tour,
                  (SELECT COUNT(*) FROM data.matches m WHERE m.winner_id = p.player_id OR m.loser_id = p.player_id) AS n
           FROM data.player p WHERE p.canonical_name LIKE ? ORDER BY n DESC""", (f"%{name}%",)).fetchall()
    if not rows:
        raise SystemExit(f"no player matches {name!r}")
    exact = [r for r in rows if r[1].lower() == name.lower()]
    pid, canon, tour, n = (exact or rows)[0]
    if not exact and len(rows) > 1:
        others = ", ".join(r[1] for r in rows[1:4])
        print(f"note: {name!r} -> {canon} ({n} matches); also matches {others}")
    return pid, canon, tour


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("player_a")
    ap.add_argument("player_b")
    ap.add_argument("--date", default=date.today().isoformat(), help="ISO date (ratings as of this day)")
    ap.add_argument("--surface", default="hard", choices=["hard", "clay", "grass", "carpet"])
    ap.add_argument("--best-of", type=int, default=3, choices=[1, 3, 5])
    ap.add_argument("--final", default="tb", choices=list(markov.FINALS),
                    help="deciding set: tb (tiebreak), adv, tb12, mtb (match tiebreak)")
    ap.add_argument("--run", type=int, default=694)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--models", type=Path, default=models_db.MODELS_DB)
    ap.add_argument("--data", type=Path, default=models_db.DATA_DB)
    args = ap.parse_args(argv)

    con = models_db.connect(args.models)
    models_db.attach_data(con, args.data)
    try:
        spec, params = tuning.model_of(con, args.run)
        if spec.model_id != "points_v1":
            raise SystemExit(f"run {args.run} is {spec.model_id}, not points_v1")
        a, name_a, tour = find_player(con, args.player_a)
        b, name_b, tour_b = find_player(con, args.player_b)
        if tour != tour_b:
            raise SystemExit(f"{name_a} ({tour}) and {name_b} ({tour_b}) are on different tours")
        t = time.perf_counter()
        m = elo.load_matches(con)
        spec.attach(con, m)
    finally:
        con.close()
    sub = elo.until(m, args.date)
    st = points_rating.run(sub, params).state
    xa, xb, va, vb = points_rating.matchup(st, a, b, tour, args.surface, args.date)
    p_win = points_rating.win_probability(st, xa, xb, va, vb, args.best_of, args.final)
    pa, pb = 1 / (1 + markov.math.exp(-xa)), 1 / (1 + markov.math.exp(-xb))
    p_raw = markov.match(pa, pb, args.best_of, args.final)

    fmt = f"best of {args.best_of}, deciding set {args.final}"
    print(f"\n{name_a} vs {name_b}  ({tour.upper()}, {args.surface}, {fmt}; run {args.run}, "
          f"ratings as of {args.date}; {time.perf_counter() - t:.0f}s)")
    for nm, i in ((name_a, a), (name_b, b)):
        if i not in st.serve:
            print(f"  {nm}: no rated matches before {args.date} (treated as a new player)")
    print(f"\n  P({name_a} wins) = {p_win:.1%}   (from the simulated sets alone: {p_raw:.1%})")
    print(f"  serve points won: {name_a} {pa:.1%}, {name_b} {pb:.1%}")
    print("\n  sets:  " + "   ".join(f"{k} {v:.1%}" for k, v in markov.set_count_dist(pa, pb, args.best_of, args.final).items()))
    print(f"\n  most likely scorelines ({name_a}'s games first):")
    for score, pr in markov.top_scores(markov.match_score_dist(pa, pb, args.best_of, args.final), args.top):
        print(f"    {score:24s} {pr:5.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
