"""Predict one match through the engine (milestone A1): pre-match, or in play from a score.

    python scripts/predict.py "Jannik Sinner" "Carlos Alcaraz"
    python scripts/predict.py Sinner Alcaraz --tournament "Paris Masters" --date 2026-11-05
    python scripts/predict.py Swiatek Sabalenka --surface clay --best-of 3 --date 2026-06-01 --as-of 2026-06-01
    python scripts/predict.py Shelton Fils --score "4-6 3-2" --points 30-40 --serving B --serve-pts 31/48 36/50

Names: exact (accents and punctuation ignored), else last name + first initial among recent
players, else a substring of the full name (busiest player first). --tournament finds the
edition running on --date in tennis.db: its surface, best-of and deciding-set rule, and its
id (for context). With no running edition, the tournament's latest edition gives surface and
best-of. Explicit --surface / --best-of / --final always win.

The numbers come from PointsPredictor (engine/predict.py): run 790 by default, final state of
the full replay (cached under models/engine/). --as-of replays only editions that started
before that date, for a point-in-time prediction of a past match. In play (--score,
--serving): Markov roll-up from the score, with today's serve points (--serve-pts) updating
each serve logit; not calibrated. Reads only.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tennis import models_db                                     # noqa: E402
from tennis.engine import MatchQuery                             # noqa: E402
from tennis.engine.editions import find_edition                  # noqa: E402
from tennis.engine.predict import PointsPredictor, live_from_score   # noqa: E402

FINALS = ("tb", "adv", "tb12", "mtb")


def pick_player(pred: PointsPredictor, name: str, tour: str | None) -> tuple[int, str, str]:
    """(player_id, tour, how)."""
    res = pred.players
    for t in ([tour] if tour else ["atp", "wta"]):
        pid, how = res.resolve(t, name)
        if pid is not None:
            return pid, t, how
    hits = res.search(name, tour)
    if not hits:
        raise SystemExit(f"no player matches {name!r}")
    pid, t, canon, n = hits[0]
    how = "substring" + (f" (busiest of {len(hits)}{'+' if len(hits) == 5 else ''})" if len(hits) > 1 else "")
    return pid, t, how


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("player_a")
    ap.add_argument("player_b")
    ap.add_argument("--tour", choices=["atp", "wta"], help="default: from the names")
    ap.add_argument("--date", default=date.today().isoformat(), help="match day (ISO)")
    ap.add_argument("--tournament", help="tournament name: edition, surface, best-of, deciding set")
    ap.add_argument("--surface", choices=["hard", "clay", "grass", "carpet"])
    ap.add_argument("--best-of", type=int, choices=[1, 3, 5])
    ap.add_argument("--final", choices=FINALS, help="deciding set: tb, adv, tb12, mtb (match tiebreak)")
    ap.add_argument("--run", type=int, default=790)
    ap.add_argument("--as-of", help="replay only editions starting before this date (point in time)")
    ap.add_argument("--raw", action="store_true", help="no calibration")
    ap.add_argument("--rebuild-cache", action="store_true")
    live = ap.add_argument_group("in play")
    live.add_argument("--score", help="games so far, A's first; the last set may be in progress: '6-4 3-6 2-1'")
    live.add_argument("--points", default="", help="current game, A's first: '30-15', '40-AD'; tiebreak: '5-3'")
    live.add_argument("--serving", choices=["A", "B", "a", "b"], help="who serves the next point")
    live.add_argument("--serve-pts", nargs=2, metavar=("A_WON/PLAYED", "B_WON/PLAYED"),
                      help="serve points won/played so far this match, e.g. 41/60 35/58")
    live.add_argument("--deciding-tb", type=int, default=7, choices=[7, 10],
                      help="points in the deciding set's tiebreak at 6-6 (10 at the slams since 2022)")
    live.add_argument("--no-form", action="store_true", help="ignore --serve-pts (score only)")
    ap.add_argument("--models", type=Path, default=models_db.MODELS_DB)
    ap.add_argument("--data", type=Path, default=models_db.DATA_DB)
    args = ap.parse_args(argv)
    if args.score is not None and args.serving is None:
        ap.error("--score needs --serving A|B (who serves the next point)")

    pred = PointsPredictor.load(args.run, as_of=args.as_of, models_path=args.models, data_path=args.data,
                                rebuild=args.rebuild_cache, calibrated=not args.raw,
                                live_form=not args.no_form)
    a, tour_a, how_a = pick_player(pred, args.player_a, args.tour)
    b, tour_b, how_b = pick_player(pred, args.player_b, args.tour or tour_a)
    if tour_a != tour_b:
        raise SystemExit(f"{pred.players.name(a)} ({tour_a}) and {pred.players.name(b)} ({tour_b}) "
                         "are on different tours")
    tour = tour_a

    ed = None
    if args.tournament:
        con = sqlite3.connect(models_db._uri(args.data, "ro"), uri=True)
        try:
            ed = find_edition(con, tour, args.tournament, args.date, alias="main")
        finally:
            con.close()
        if ed is None:
            raise SystemExit(f"no {tour.upper()} tournament matches {args.tournament!r} before {args.date}")
    surface = args.surface or (ed and ed["surface"]) or "hard"
    best_of = args.best_of or (ed and ed["best_of"]) or 3
    final = args.final or (pred.deciding_rule(ed["latest_edition_id"], ed["tournament_id"]) if ed else "tb")
    q = MatchQuery(tour=tour, player_a=a, player_b=b, on=args.date, surface=surface, best_of=best_of,
                   final=final, edition_id=ed["edition_id"] if ed else None)

    name_a, name_b = pred.players.name(a), pred.players.name(b)
    for typed, nm, how in ((args.player_a, name_a, how_a), (args.player_b, name_b, how_b)):
        if how != "exact":
            print(f"note: {typed!r} -> {nm} ({how})")
    if ed:
        state = f"edition of {ed['start']}" if ed["running"] else f"no edition running; latest {ed['start']}"
        print(f"tournament: {ed['tournament']} ({state}, {ed['surface']}, best of {ed['best_of']})")

    p = pred.predict(q)
    d = p.detail
    print(f"\n{name_a} vs {name_b}  ({tour.upper()}, {surface}, best of {best_of}, deciding set {final}; "
          f"{p.model}, {args.date})")
    print(f"\n  P({name_a} wins) = {p.p_a:.1%}" + ("" if args.raw else f"   (raw {d['p_raw']:.1%}, alpha {d['alpha']:.3f})"))
    print(f"  serve points won: {name_a} {d['serve_a']:.1%}, {name_b} {d['serve_b']:.1%}"
          + (f"   (context {d['ctx_a']:+.3f} / {d['ctx_b']:+.3f} logit)" if d["ctx_a"] or d["ctx_b"] else ""))
    print(f"  last rated match: {name_a} {d['last_a'] or 'never'}, {name_b} {d['last_b'] or 'never'}")
    print("  sets (simulated, unshrunk): " + "   ".join(f"{k} {v:.1%}" for k, v in pred.set_scores(q).items()))

    if args.score is not None:
        sa = sb = (0, 0)
        if args.serve_pts:
            sa, sb = (tuple(int(v) for v in s.split("/")) for s in args.serve_pts)
        ls = live_from_score(args.score, args.points, args.serving.upper() == "A", serve_a=sa, serve_b=sb,
                             best_of=best_of, final=final, deciding_tb_points=args.deciding_tb)
        lp = pred.predict(MatchQuery(**{**q.__dict__, "live": ls}))
        ld = lp.detail
        srv = name_a if ls.a_serving else name_b
        print(f"\n  in play: sets {ls.sets_a}-{ls.sets_b}, games {ls.games_a}-{ls.games_b}, "
              f"points {args.points or '0-0'}, {srv} to serve")
        print(f"  P({name_a} wins) = {lp.p_a:.1%}   (pre-match {p.p_a:.1%})")
        if args.serve_pts and not args.no_form:
            print(f"  serve points with today's form: {name_a} {ld['serve_a']:.1%} -> {ld['serve_a_live']:.1%}, "
                  f"{name_b} {ld['serve_b']:.1%} -> {ld['serve_b_live']:.1%}")
        p = lp
    for n in p.notes:
        print(f"  note: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
