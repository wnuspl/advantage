"""Predict one match with a points_v1 run: win probability plus the likely scorelines.

    python archive/v1/scripts/predict_match.py "Sinner" "Alcaraz"
    python archive/v1/scripts/predict_match.py "Swiatek" "Sabalenka" --surface clay --date 2026-06-01
    python archive/v1/scripts/predict_match.py "Sinner" "Alcaraz" --best-of 5 --final tb --top 8
    python archive/v1/scripts/predict_match.py "Shelton" "Fils" --score "4-6 3-2" --points 30-40 --serving A
    python archive/v1/scripts/predict_match.py "Shelton" "Fils" --score "4-6 3-2" --serving B --serve-pts 31/48 36/50

In play (--score, --points, --serving): the same serve-point probabilities rolled up from
the current score (markov.match_from), with the uncertainty shrink. --serve-pts also
updates each serve logit on the serve points won so far this match (today's form:
points_rating.in_match_form). Everything is the independent-points model; it has not
been checked against real in-play outcomes yet.

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

ARCHIVE = Path(__file__).resolve().parents[1]   # archive/v1: results, reports, docs
ROOT = Path(__file__).resolve().parents[3]      # project root: tennis.db, data, src
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ARCHIVE / "src"))   # archived modules, merged into tennis

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
    live = ap.add_argument_group("in play (from mid-match)")
    live.add_argument("--score", help="games so far, A's first, last set may be in progress: '6-4 3-6 2-1'")
    live.add_argument("--points", default="", help="current game, A's first: '30-15', '40-AD'; tiebreak: '5-3'")
    live.add_argument("--serving", choices=["A", "B", "a", "b"],
                      help="who serves the next point (required with --score)")
    live.add_argument("--serve-pts", nargs=2, metavar=("A_WON/PLAYED", "B_WON/PLAYED"),
                      help="serve points won/played so far this match, e.g. 41/60 35/58: updates "
                           "each player's serve logit for today's form")
    live.add_argument("--deciding-tb", type=int, default=7, choices=[7, 10],
                      help="points in the deciding set's tiebreak at 6-6 (10 at the slams since 2022)")
    live.add_argument("--form-var", type=float, default=points_rating.IN_PLAY_FORM_VAR,
                      help="in-match form variance for --serve-pts (default fit by scripts/inplay_form_test.py)")
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
    if args.score is not None:
        in_play(args, st, name_a, name_b, xa, xb, va, vb, p_win)
    return 0


def in_play(args, st, name_a, name_b, xa, xb, va, vb, p_pre) -> None:
    """P(A wins) from the live score, with and without today's serve record."""
    if args.serving is None:
        raise SystemExit("--score needs --serving A|B (who serves the next point)")
    sc = markov.parse_live(args.score, args.points, args.best_of, args.final)
    a_srv = args.serving.upper() == "A"
    kw = dict(best_of=args.best_of, final=args.final, score=sc, a_serving=a_srv,
              deciding_tb_points=args.deciding_tb)
    p_live = points_rating.in_play_probability(st, xa, xb, va, vb, **kw)
    srv = name_a if a_srv else name_b
    print(f"\n  in play: sets {sc['sets_a']}-{sc['sets_b']}, games {sc['games_a']}-{sc['games_b']}, "
          f"points {args.points or '0-0'}, {srv} to serve")
    print(f"  P({name_a} wins) = {p_live:.1%}   (pre-match {p_pre:.1%})")
    if args.serve_pts:
        (ka, na), (kb, nb) = ((int(v) for v in s.split("/")) for s in args.serve_pts)
        xa2, va2 = points_rating.in_match_form(xa, va, ka, na, args.form_var)
        xb2, vb2 = points_rating.in_match_form(xb, vb, kb, nb, args.form_var)
        p_form = points_rating.in_play_probability(st, xa2, xb2, va2, vb2, **kw)
        sig = lambda x: 1 / (1 + markov.math.exp(-x))               # noqa: E731
        print(f"  with today's serve record ({name_a} {ka}/{na}, {name_b} {kb}/{nb}): "
              f"P({name_a} wins) = {p_form:.1%}")
        print(f"    serve points: {name_a} {sig(xa):.1%} -> {sig(xa2):.1%}, "
              f"{name_b} {sig(xb):.1%} -> {sig(xb2):.1%}")
    print("  (independent points from here; not calibrated on in-play outcomes)")


if __name__ == "__main__":
    sys.exit(main())
