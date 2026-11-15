"""Kalshi tennis match markets vs a points_v1 run: where the model disagrees most with the
market, and the log-optimal (Kelly) stake on each, walking the live order book.

    python scripts/kalshi_edges.py                          # run 790, $1000 bankroll, 1/4 Kelly
    python scripts/kalshi_edges.py --bankroll 500 --kelly 0.5 --model-weight 0.5
    python scripts/kalshi_edges.py --series KXATPMATCH,KXWTAMATCH --include-started

Steps:
  1. Pull every open event of the match-winner series (public endpoints, no key) and the
     order book of both markets of each match. Raw responses are saved under
     data/raw/kalshi/<UTC time>/ (immutable; useful later to score Kalshi as a benchmark).
  2. Resolve both names to player ids (normalized exact name, else busiest player with
     the same last name and first initial; unresolved matches are listed and skipped).
  3. Replay the run over all of tennis.db, then for each match: serve-point logits from
     the final state (inactivity-grown variance), context offsets if the run has them
     (rust, load, age as of the match day), Markov roll-up, uncertainty shrink, and the
     walk-forward calibration alpha of the latest quarter for that tour (--no-calibrate
     for raw). Surface, best-of and the edition (for context) come from an edition in
     tennis.db that both players are in and that is still running; otherwise --surface,
     best-of 3 (5 for men's slam main draws, detected by name) and a new edition.
  4. Market probability = de-overrounded mid. Deviation = model - market.
  5. Bet sizing per match, for whichever side has p > ask + fee: maximize expected log
     wealth on --kelly * bankroll, walking the merged ladder (YES on the player, or NO on
     the opponent). p used for sizing = logit blend of model and market (--model-weight;
     1 = model only).

Why EV alone can't size a bet: expected profit is linear in stake, so "highest EV" is
always "the whole bankroll" (or all the liquidity) whenever the edge is positive. The
growth-optimal stake (Kelly) is the standard answer; quarter Kelly is the default because
the model's probabilities are uncertain.

Read before betting: on 2022-26 tour-level matches Pinnacle closing odds score 0.5864 log
loss vs 0.6002 for points_v1 (run 602; run 790 ~0.594), and the model's held-out weight
given the market is ~0 (models/odds/benchmark_602.json). Kalshi may be less sharp than
Pinnacle, but a large model-vs-market gap is more often a model miss (injury, stale
ratings: stats end 2026-05-25, Challenger results end there too) than a market miss.
Matches already started are skipped by default: the model is pre-match only.

Reads only; places no orders. Output: table on stdout + models/kalshi/<UTC time>.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tennis import calibrate, context, kalshi, markov, models_db, points_rating, tuning  # noqa: E402
from tennis.normalize import norm_name  # noqa: E402

SLAMS = ("australian open", "roland garros", "french open", "wimbledon", "us open")


# ---------------------------------------------------------------------------- players

def _key(s: str | None) -> str:
    return norm_name(s).replace(" ", "")              # "O'Connell" == "Oconnell"


class Resolver:
    """Kalshi full name -> player_id. Exact (accents, punctuation and spaces ignored),
    else last name + first initial among players with a match in the last 3 years only
    (an unrestricted fallback matched 'Colson Wells' to a 1970s 'Chris Wells')."""

    def __init__(self, con=None, today: str | None = None, index: tuple | None = None):
        """From tennis.db, or from a cached `index` (plain dicts, so the replay cache
        never pickles this script's classes: they would not load under another __main__)."""
        if index is not None:
            self.by_name, self.by_last = index
            return
        self.by_name: dict[tuple[str, str], list[tuple[int, int]]] = {}
        self.by_last: dict[tuple[str, str, str], list[tuple[int, int]]] = {}
        stats = {p: (k, last) for p, k, last in con.execute(
            "SELECT p, COUNT(*), MAX(d) FROM (SELECT winner_id p, e.start_date d FROM data.matches m "
            "JOIN data.tournament_edition e USING (edition_id) UNION ALL SELECT loser_id, e.start_date "
            "FROM data.matches m JOIN data.tournament_edition e USING (edition_id)) GROUP BY p")}
        recent = (date.fromisoformat(today) - timedelta(days=3 * 365)).isoformat()
        for pid, tour, canon, first, last in con.execute(
                "SELECT player_id, tour, canonical_name, name_first, name_last FROM data.player WHERE is_placeholder=0"):
            k, last_d = stats.get(pid, (0, None))
            self.by_name.setdefault((tour, _key(canon)), []).append((k, pid))
            if last and last_d and last_d >= recent:
                self.by_last.setdefault((tour, _key(last), norm_name(first)[:1]), []).append((k, pid))

    def __call__(self, tour: str, name: str) -> tuple[int | None, str]:
        hits = self.by_name.get((tour, _key(name)))
        how = "exact"
        if not hits:                                     # last name + first initial
            parts = norm_name(name).split()
            for cut in range(1, len(parts)):             # multi-word last names first
                hits = self.by_last.get((tour, "".join(parts[cut:]), parts[0][:1]))
                if hits:
                    how = "last+initial"
                    break
        if not hits:
            return None, "unresolved"
        hits = sorted(hits, reverse=True)
        if len(hits) > 1 and hits[1][0] > 0:
            how += f" (busiest of {len(hits)})"
        return hits[0][1], how


# ---------------------------------------------------------------------------- matches

def parse_events(events: list[dict], series: str) -> list[dict]:
    """One dict per match: both markets, names, start time, competition."""
    out = []
    for e in events:
        mk = [m for m in e.get("markets", []) if m.get("status") == "active"]
        if len(mk) != 2:
            continue
        a, b = mk
        out.append({
            "event": e["event_ticker"], "series": series, "tour": kalshi.TOUR_OF_SERIES[series],
            "title": e.get("title", ""), "competition": (e.get("product_metadata") or {}).get("competition", ""),
            "rules": a.get("rules_primary", ""),
            "start": a.get("occurrence_datetime") or a.get("expected_expiration_time"),
            "a": {"ticker": a["ticker"], "name": a.get("yes_sub_title", ""),
                  "comp": (a.get("custom_strike") or {}).get("tennis_competitor")},
            "b": {"ticker": b["ticker"], "name": b.get("yes_sub_title", ""),
                  "comp": (b.get("custom_strike") or {}).get("tennis_competitor")},
        })
    return out


def current_edition(con, a: int, b: int, day: str):
    """(edition_id, surface, best_of) of a still-running edition both players are in."""
    return con.execute(
        """SELECT e.edition_id, e.surface, e.best_of FROM data.tournament_edition e
           WHERE e.start_date <= ? AND e.start_date >= ? AND COALESCE(e.end_date, e.start_date) >= ?
             AND EXISTS (SELECT 1 FROM data.matches m WHERE m.edition_id = e.edition_id AND ? IN (m.winner_id, m.loser_id))
             AND EXISTS (SELECT 1 FROM data.matches m WHERE m.edition_id = e.edition_id AND ? IN (m.winner_id, m.loser_id))
           ORDER BY e.start_date DESC LIMIT 1""",
        (day, (date.fromisoformat(day) - timedelta(days=21)).isoformat(),
         (date.fromisoformat(day) - timedelta(days=1)).isoformat(), a, b)).fetchone()


def tour_alpha(prep, p_winner, tour: str, on: str) -> float:
    """Calibration alpha as the walk-forward eval sets it for the quarter containing `on`."""
    from tennis.matchup.data import fit_alpha     # imports scikit-learn (~6 s): cache builds only
    d = date.fromisoformat(on)
    q0 = date(d.year, 3 * ((d.month - 1) // 3) + 1, 1)
    hist = prep.indices(calibrate._add_years(q0.isoformat(), -10), (q0 - timedelta(days=1)).isoformat(), f"tour_{tour}")
    hp, hy = calibrate.oriented(prep, p_winner, hist)
    return fit_alpha(np.log(np.clip(hp, 1e-12, 1 - 1e-12) / np.clip(1 - hp, 1e-12, 1)), hy)



class Pricer:
    """P(A wins) and the Kelly bet for one match, from its pre-match serve-point logits
    (x["base"]) and, when it is live, Kalshi's live score.

    In play: the score goes to markov.match_from via points_rating.in_play_probability
    (uncertainty shrink, no calibration alpha: the alphas are fit on pre-match
    predictions). Unless --no-live-form, each serve logit is first updated on the serve
    points won so far (points_rating.in_match_form). Live bets are sized only with
    --in-play-bets, and never when the score could not be read cleanly."""

    def __init__(self, args, state, alphas: dict):
        self.args, self.st, self.alphas = args, state, alphas

    def prematch(self, x) -> float:
        """Calibrated pre-match P(A wins), for comparison with the in-play number."""
        return self.model_prob(x, None)[1]

    def model_prob(self, x, details) -> tuple[float, float, "kalshi.LiveState | None"]:
        xa, xb, va, vb = x["base"]
        if not (details and details.get("status") == "live"):
            p_raw = points_rating.win_probability(self.st, xa, xb, va, vb, x["best_of"], "tb")
            al = self.alphas.get(x["tour"], 1.0)
            return p_raw, 1 / (1 + math.exp(-al * math.log(p_raw / (1 - p_raw)))), None
        ls = kalshi.live_state(details, x["a"]["comp"], x["b"]["comp"], x["best_of"], "tb")
        if ls.problem:
            return float("nan"), float("nan"), ls
        if not self.args.no_live_form:
            fv = self.args.form_var
            xa, va = points_rating.in_match_form(xa, va, *ls.serve_a, fv)
            xb, vb = points_rating.in_match_form(xb, vb, *ls.serve_b, fv)
        p = points_rating.in_play_probability(self.st, xa, xb, va, vb, x["best_of"], "tb",
                                              ls.score(), ls.a_serving)
        return p, p, ls

    def price(self, x, books, details) -> dict:
        args = self.args
        p_raw, p_model, ls = self.model_prob(x, details)
        ba, bb = books[x["a"]["ticker"]], books[x["b"]["ticker"]]
        p_mkt = kalshi.fair_prob(ba, bb)
        ok = not math.isnan(p_model)
        p_use = kalshi.blend(p_model, p_mkt, args.model_weight) if ok else float("nan")
        live = ls is not None
        may_bet = ok and x["fresh"] and (not live or args.in_play_bets)
        best = None
        for own, opp, bk_own, bk_opp, p in ((x["a"], x["b"], ba, bb, p_use), (x["b"], x["a"], bb, ba, 1 - p_use)):
            lad = kalshi.ladder(bk_own, own["ticker"], bk_opp, opp["ticker"])
            sz = kalshi.kelly_walk(p, lad, args.bankroll, args.kelly, args.fee_rate, args.max_stake) \
                if may_bet else kalshi.Sizing()
            cand = {"side": own["name"], "p": p, "ask": lad[0].price if lad else None, "sizing": sz,
                    "liquidity_at_ask": lad[0].size if lad else 0}
            if best is None or sz.ev > best["sizing"].ev:
                best = cand
        why = "" if may_bet else ("no bet (live score unreadable)" if not ok else
                                  "no bet (stale rating)" if not x["fresh"] else "no bet (in play)")
        return {**x, "p_raw": p_raw, "p_model": p_model, "p_market": p_mkt, "p_use": p_use,
                "dev": None if (p_mkt is None or not ok) else p_model - p_mkt, "bet": best, "no_bet": why,
                "live_state": ls}


CACHE_DIR = ROOT / "models" / "kalshi"
CODE = ("points_rating.py", "context.py", "elo.py", "markov.py", "calibrate.py", "tuning.py")


def cache_key(con, run: int, params, today: str) -> str:
    """Changes when tennis.db's contents, the model code, the run's params or the
    calibration quarter change."""
    h = hashlib.sha256(b"replay-cache-v2")
    h.update(repr(con.execute(
        "SELECT COUNT(*), MAX(match_id), MAX(snapshot_id), TOTAL(winner_id), TOTAL(loser_id), "
        "TOTAL(LENGTH(score)) FROM data.matches").fetchone()).encode())
    h.update(repr(con.execute("SELECT COUNT(*), MAX(player_id), TOTAL(LENGTH(dob)) FROM data.player").fetchone()).encode())
    h.update(repr(con.execute("SELECT COUNT(*), TOTAL(LENGTH(sha256)) FROM data.ingest_snapshot").fetchone()).encode())
    for f in CODE:
        h.update((ROOT / "src" / "tennis" / f).read_bytes())
    h.update(json.dumps(vars(params), sort_keys=True, default=str).encode())
    d = date.fromisoformat(today)
    h.update(f"{run} {d.year}Q{(d.month - 1) // 3}".encode())
    return h.hexdigest()


def load_model(con, run: int, spec, params, today: str, rebuild: bool = False) -> dict:
    """Everything the live pricing needs from the full replay: the final PointsState,
    every player's context bookkeeping, per-tour calibration alphas for this quarter,
    and the name resolver. The replay takes minutes (pure Python over ~1.8M matches);
    the result is pickled to models/kalshi/ and reused until cache_key() changes."""
    key = cache_key(con, run, params, today)
    path = CACHE_DIR / f"replay_run{run}.pkl"
    if path.exists() and not rebuild:
        try:
            with path.open("rb") as f:
                cached = pickle.load(f)
            if cached.get("key") == key:
                print(f"model: cached replay of run {run} ({cached['n']:,} matches, latest result {cached['last_day']})")
                return cached
        except Exception as e:                         # corrupt / old format: rebuild
            print(f"cache unreadable ({e}); rebuilding")
    t = time.perf_counter()
    prep = tuning.prepare(con)
    m = prep.m
    print(f"model: replaying run {run} over {len(m):,} matches (cached afterwards) ...")
    spec.attach(con, m)
    res = points_rating.run(m, params)
    edmap = dict(con.execute("SELECT match_id, edition_id FROM data.matches"))
    out = {
        "key": key, "n": len(m),
        "last_day": con.execute("SELECT MAX(COALESCE(match_date, '')) FROM data.matches").fetchone()[0],
        "state": res.state,
        "ctx_states": context.player_states(m, [edmap[i] for i in m.match_id]) if params.has_context() else {},
        "alphas": {t_: tour_alpha(prep, res.p_winner, t_, today) for t_ in ("atp", "wta")},
        "resolver_index": (lambda r: (r.by_name, r.by_last))(Resolver(con, today)),
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    for attempt in range(10):          # Windows: a scanner may hold the old file briefly
        try:
            tmp.replace(path)
            break
        except PermissionError:
            time.sleep(0.5)
    else:
        print(f"warning: could not replace {path.relative_to(ROOT)} (file in use); not cached this time")
    print(f"model: replay done in {time.perf_counter() - t:.0f}s")
    return out


# ---------------------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=int, default=790)
    ap.add_argument("--series", default=",".join(kalshi.MATCH_SERIES))
    ap.add_argument("--bankroll", type=float, default=1000.0, help="principal, dollars")
    ap.add_argument("--kelly", type=float, default=0.25, help="Kelly fraction (1 = full Kelly)")
    ap.add_argument("--model-weight", type=float, default=1.0,
                    help="logit blend weight on the model for sizing (1 = model only, 0 = market)")
    ap.add_argument("--max-stake", type=float, default=None, help="cap per match, dollars")
    ap.add_argument("--max-total", type=float, default=0.5,
                    help="cap on the summed stake of all bets, as a fraction of bankroll (scaled down pro rata)")
    ap.add_argument("--max-stale-days", type=int, default=120,
                    help="no bet if either player's last rated match is older than this (or unrated); "
                         "such matches are also hidden unless --show-stale")
    ap.add_argument("--show-stale", action="store_true",
                    help="list matches with stale ratings in the table and watch mode (never bet on)")
    ap.add_argument("--fee-rate", type=float, default=kalshi.FEE_RATE)
    ap.add_argument("--surface", default="hard", choices=["hard", "clay", "grass", "carpet"],
                    help="when no running edition in tennis.db pins it")
    ap.add_argument("--no-calibrate", action="store_true")
    ap.add_argument("--rebuild-cache", action="store_true", help="redo the full replay even if the cache is valid")
    ap.add_argument("--include-started", action="store_true",
                    help="price matches past their start time that have no live-score feed, as if pre-match")
    ap.add_argument("--in-play-bets", action="store_true",
                    help="size bets on live matches too (off by default: the feed's lag behind the court is unknown)")
    ap.add_argument("--no-live-form", action="store_true",
                    help="in play, ignore today's serve points (score only)")
    ap.add_argument("--form-var", type=float, default=points_rating.IN_PLAY_FORM_VAR,
                    help="in-match form variance for the live serve update (default fit by "
                         "scripts/inplay_form_test.py)")
    ap.add_argument("--watch", type=float, default=0, metavar="SECONDS",
                    help="after the report, keep re-pricing live matches every SECONDS; log to models/kalshi/live_<time>.jsonl")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--models", type=Path, default=models_db.MODELS_DB)
    ap.add_argument("--data", type=Path, default=models_db.DATA_DB)
    args = ap.parse_args(argv)

    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    raw_dir = ROOT / "data" / "raw" / "kalshi" / stamp
    raw_dir.mkdir(parents=True, exist_ok=False)

    # 1. Kalshi: events, live status (milestones), order books
    matches, raw_events = [], {}
    for s in args.series.split(","):
        ev = kalshi.open_events(s)
        raw_events[s] = ev
        matches += parse_events(ev, s)
    with ThreadPoolExecutor(max_workers=6) as pool:     # well under Kalshi's read rate limit
        ms = list(pool.map(lambda x: kalshi.milestone(x["event"]), matches))
    for x, mm in zip(matches, ms):
        x["milestone"] = mm["id"] if mm else None
        x["ms_best_of"] = int((mm or {}).get("details", {}).get("best_of") or 0) or None
    live = kalshi.live_batch([x["milestone"] for x in matches if x["milestone"]])
    skipped, todo = [], []
    for x in matches:
        d = live.get(x["milestone"]) if x["milestone"] else None
        x["live_details"] = d
        status = (d or {}).get("status")
        if status == "closed":
            skipped.append((x["title"], "finished"))
            continue
        x["live"] = status == "live"
        if d is None:                                      # no feed: fall back to the start time
            started = x["start"] and datetime.fromisoformat(x["start"].replace("Z", "+00:00")) <= now
            if started and not args.include_started:
                skipped.append((x["title"], "started, no live score"))
                continue
        todo.append(x)
    matches = todo
    tickers = [x[s]["ticker"] for x in matches for s in "ab"]
    with ThreadPoolExecutor(max_workers=6) as pool:
        books = dict(zip(tickers, pool.map(kalshi.orderbook, tickers)))
    (raw_dir / "events.json").write_text(json.dumps(raw_events, ensure_ascii=False), encoding="utf-8")
    (raw_dir / "orderbooks.json").write_text(json.dumps(books), encoding="utf-8")
    (raw_dir / "live.json").write_text(json.dumps(live), encoding="utf-8")
    n_live = sum(x["live"] for x in matches)
    print(f"Kalshi: {sum(len(v) for v in raw_events.values())} open events, {len(matches)} to price "
          f"({n_live} live), {len(skipped)} skipped; raw saved to {raw_dir.relative_to(ROOT)}")

    # 2-3. model (replay cached: see load_model)
    con = models_db.connect(args.models)
    models_db.attach_data(con, args.data)
    try:
        spec, params = tuning.model_of(con, args.run)
        if spec.model_id != "points_v1":
            raise SystemExit(f"run {args.run} is {spec.model_id}, not points_v1")
        cache = load_model(con, args.run, spec, params, now.date().isoformat(), args.rebuild_cache)
        resolve = Resolver(index=cache["resolver_index"])
        priced = []
        for x in matches:
            for side in ("a", "b"):
                pid, how = resolve(x["tour"], x[side]["name"])
                x[side].update(pid=pid, resolved=how)
            if x["a"]["pid"] is None or x["b"]["pid"] is None:
                skipped.append((x["title"], "unresolved: " + ", ".join(
                    x[s]["name"] for s in "ab" if x[s]["pid"] is None)))
                continue
            day = x["start"][:10] if x["start"] else now.date().isoformat()
            x["day"] = day
            ed = current_edition(con, x["a"]["pid"], x["b"]["pid"], day)
            comp = norm_name(x["competition"] + " " + x["rules"])
            slam = x["tour"] == "atp" and any(s in comp for s in SLAMS) and "qualif" not in comp
            x["edition_id"] = ed[0] if ed else None
            x["surface"] = (ed[1] if ed and ed[1] else args.surface)
            x["best_of"] = x["ms_best_of"] or (ed[2] if ed and ed[2] else (5 if slam else 3))
            x["surface_from"] = "tennis.db edition" if ed else "default"
            priced.append(x)
        dob = dict(con.execute("SELECT player_id, dob FROM data.player"))
        names = dict(con.execute("SELECT player_id, canonical_name FROM data.player"))
    finally:
        con.close()

    st = cache["state"]
    alphas = {} if args.no_calibrate else {t: cache["alphas"][t] for t in {x["tour"] for x in priced}}
    stale_cut = (now.date() - timedelta(days=args.max_stale_days)).toordinal()
    for x in priced:                       # pre-match serve-point logits, once per match
        a, b, tour = x["a"]["pid"], x["b"]["pid"], x["tour"]
        xa, xb, va, vb = points_rating.matchup(st, a, b, tour, x["surface"], x["day"])
        if params.has_context():
            d0 = date.fromisoformat(x["day"]).toordinal()
            c = context.features_from_states(cache["ctx_states"], dob,
                                             {a: (x["edition_id"], d0), b: (x["edition_id"], d0)})
            bs, br = params.context_betas(tour)
            xa += float(c[a] @ bs - c[b] @ br)
            xb += float(c[b] @ bs - c[a] @ br)
        x["base"] = (xa, xb, va, vb)
        x["fresh"] = all(i in st.last and st.last[i] >= stale_cut for i in (a, b))

    pricer = Pricer(args, st, alphas)
    rows = []
    for x in priced:
        r = pricer.price(x, books, x["live_details"])
        r.update(last_a=date.fromordinal(st.last[x["a"]["pid"]]).isoformat() if x["a"]["pid"] in st.last else None,
                 last_b=date.fromordinal(st.last[x["b"]["pid"]]).isoformat() if x["b"]["pid"] in st.last else None,
                 db_a=names.get(x["a"]["pid"]), db_b=names.get(x["b"]["pid"]))
        rows.append(r)

    rows.sort(key=lambda r: -abs(r["dev"] or 0))

    # portfolio cap: per-match Kelly stakes are sized independently; scale all down pro rata
    total = sum(r["bet"]["sizing"].cost for r in rows)
    cap = args.max_total * args.bankroll
    scale = min(1.0, cap / total) if total > 0 else 1.0
    if scale < 1.0:
        for r in rows:
            r["bet"]["sizing"] = kalshi.scale_sizing(r["bet"]["sizing"], r["bet"]["p"], scale,
                                                     args.bankroll * args.kelly, args.fee_rate)

    # 4-5. report
    cal = "raw" if args.no_calibrate else "calibrated (" + ", ".join(f"{t} a={v:.3f}" for t, v in sorted(alphas.items())) + ")"
    print(f"\nrun {args.run}, {cal}; bankroll ${args.bankroll:,.0f}, {args.kelly:g} Kelly, "
          f"model weight {args.model_weight:g}, fee rate {args.fee_rate:g}\n")
    hdr = f"{'match':34} {'start (UTC)':16} {'surf':5} {'P(A) mod':>8} {'mkt':>6} {'dev':>6}  {'bet':22} {'ask':>5} {'n':>5} {'stake':>8} {'EV':>7}"
    print(hdr)
    print("-" * len(hdr))
    shown = rows if args.show_stale else [r for r in rows if r["fresh"]]
    for r in shown[:args.top]:
        bt = r["bet"]
        sz = bt["sizing"]
        title = f"{r['a']['name']} v {r['b']['name']}"[:34]
        ls = r["live_state"]
        start = "LIVE" if ls else (r["start"] or "")[:16].replace("T", " ")
        mkt = f"{r['p_market']:.3f}" if r["p_market"] is not None else "  -  "
        dev = f"{r['dev']:+.3f}" if r["dev"] is not None else "  -  "
        bet = bt["side"][:22] if sz.contracts else (r["no_bet"] or "no bet")[:22]
        ask = f"{sz.avg_price:.2f}" if sz.contracts else (f"{bt['ask']:.2f}" if bt["ask"] else "  -")
        print(f"{title:34} {start:16} {r['surface'][:5]:5} {r['p_model']:8.3f} {mkt:>6} {dev:>6}  {bet:22} "
              f"{ask:>5} {sz.contracts:5d} {sz.cost:8.2f} {sz.ev:+7.2f}")
        if ls:
            print(f"{'':34}   {ls.problem or ls.raw}   serve pts A {ls.serve_a[0]}/{ls.serve_a[1]}, "
                  f"B {ls.serve_b[0]}/{ls.serve_b[1]}; pre-match {pricer.prematch(r):.3f}")
    bets = [r for r in rows if r["bet"]["sizing"].contracts]
    tot = sum(r["bet"]["sizing"].cost for r in bets)
    print(f"\n{len(bets)} bets, total stake ${tot:,.2f} ({tot / args.bankroll:.1%} of bankroll"
          + (f", scaled x{scale:.2f} to the --max-total cap" if scale < 1 else "") + "), "
          f"total EV ${sum(r['bet']['sizing'].ev for r in bets):+,.2f} (under the sizing probability)")

    n_hidden = len(rows) - len(shown)
    if n_hidden:
        print(f"{n_hidden} matches hidden: a player's last rated match is over {args.max_stale_days} days old "
              f"or missing (Challenger results end 2026-05-25); --show-stale to list them")
    if args.show_stale:
        cut = (now.date() - timedelta(days=args.max_stale_days)).isoformat()
        stale = [(r[f"db_{s}"], r[f"last_{s}"]) for r in rows for s in "ab" if (r[f"last_{s}"] or "") < cut]
        if stale:
            print(f"\nstale ratings (last rated match > {args.max_stale_days} days ago, or unrated):")
            for nm, d in sorted(set(stale), key=lambda t: t[1] or "")[:30]:
                print(f"  {nm or '?':28} {d or 'never'}")
    fuzzy = [(r[s]["name"], r[f"db_{s}"], r[s]["resolved"]) for r in shown for s in "ab" if r[s]["resolved"] != "exact"]
    if fuzzy:
        print("\nnames resolved loosely (check):")
        for k, v, how in fuzzy:
            print(f"  {k:28} -> {v:28} {how}")
    if skipped:
        print("\nskipped:")
        for t, why in skipped:
            print(f"  {t:40} {why}")

    out_dir = ROOT / "models" / "kalshi"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"as_of": now.isoformat(), "run": args.run, "alphas": alphas, "bankroll": args.bankroll,
              "kelly": args.kelly, "model_weight": args.model_weight, "fee_rate": args.fee_rate,
              "raw": str(raw_dir.relative_to(ROOT)), "skipped": skipped, "matches": [{
                  "event": r["event"], "title": r["title"], "competition": r["competition"], "start": r["start"],
                  "tour": r["tour"], "surface": r["surface"], "surface_from": r["surface_from"], "best_of": r["best_of"],
                  "a": r["a"] | {"db_name": r["db_a"], "last_rated": r["last_a"]},
                  "b": r["b"] | {"db_name": r["db_b"], "last_rated": r["last_b"]},
                  "p_raw": r["p_raw"], "p_model": r["p_model"], "p_market": r["p_market"], "p_use": r["p_use"],
                  "deviation": r["dev"], "no_bet": r["no_bet"],
                  "live": live_json(r["live_state"]),
                  "bet": {"side": r["bet"]["side"], "p": r["bet"]["p"], "best_ask": r["bet"]["ask"],
                          "contracts": r["bet"]["sizing"].contracts, "cost": r["bet"]["sizing"].cost,
                          "fees": r["bet"]["sizing"].fees, "ev": r["bet"]["sizing"].ev,
                          "log_growth": r["bet"]["sizing"].log_growth,
                          "fills": [vars(f) for f in r["bet"]["sizing"].fills]},
              } for r in rows]}
    path = out_dir / f"{stamp}.json"
    path.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nreport: {path.relative_to(ROOT)}")
    if args.watch > 0:
        watch(args, pricer, [x for x in priced if x["fresh"] or args.show_stale], stamp)
    return 0


def live_json(ls) -> dict | None:
    if ls is None:
        return None
    return {"status": ls.status, "score": ls.raw, "problem": ls.problem, "a_serving": ls.a_serving,
            "serve_a": list(ls.serve_a), "serve_b": list(ls.serve_b), **ls.score()}


def watch(args, pricer: Pricer, priced: list[dict], stamp: str) -> None:
    """Re-price every match with a live feed every args.watch seconds until all are
    finished (or Ctrl+C). Prints a line when a match's score, model or market moves;
    logs every poll (fetch times, live details, books' best prices, model, bet) to
    models/kalshi/live_<stamp>.jsonl, for measuring later how the feed and the market
    move relative to each other. The --max-total cap is not re-applied here."""
    log = CACHE_DIR / f"live_{stamp}.jsonl"
    by_ms = {x["milestone"]: x for x in priced if x.get("milestone")}
    last: dict = {}
    print(f"\nwatching {len(by_ms)} matches every {args.watch:g}s (Ctrl+C to stop); log {log.relative_to(ROOT)}")
    with log.open("a", encoding="utf-8") as f:
        try:
            while by_ms:
                t0 = time.monotonic()
                t_live = datetime.now(timezone.utc)
                live = kalshi.live_batch(list(by_ms))
                now_live = []
                for mid, x in list(by_ms.items()):
                    d = live.get(mid) or {}
                    if d.get("status") == "closed":
                        print(f"{t_live:%H:%M:%S}  {x['title']}: finished")
                        del by_ms[mid]
                    elif d.get("status") == "live":
                        now_live.append((mid, x, d))
                tickers = [x[s]["ticker"] for _, x, _ in now_live for s in "ab"]
                with ThreadPoolExecutor(max_workers=6) as pool:
                    books = dict(zip(tickers, pool.map(kalshi.orderbook, tickers)))
                t_book = datetime.now(timezone.utc)
                for mid, x, d in now_live:
                    r = pricer.price(x, books, d)
                    ls, bt = r["live_state"], r["bet"]
                    sz = bt["sizing"]
                    ba, bb = books[x["a"]["ticker"]], books[x["b"]["ticker"]]
                    la, lb = kalshi.ladder(ba, "", bb, ""), kalshi.ladder(bb, "", ba, "")
                    f.write(json.dumps({
                        "t_live": t_live.isoformat(), "t_book": t_book.isoformat(), "event": x["event"],
                        "milestone": mid, "live": live_json(ls), "p_model": r["p_model"], "p_market": r["p_market"],
                        "ask_a": la[0].price if la else None, "ask_b": lb[0].price if lb else None,
                        "bet": {"side": bt["side"], "contracts": sz.contracts, "cost": sz.cost, "ev": sz.ev},
                        "no_bet": r["no_bet"], "details": d}, ensure_ascii=False) + "\n")
                    key = (ls.raw if ls else "", round(r["p_model"], 3) if r["p_model"] == r["p_model"] else None,
                           None if r["p_market"] is None else round(r["p_market"], 2), sz.contracts)
                    if key != last.get(mid):
                        last[mid] = key
                        what = f"BET {bt['side']} {sz.contracts} @ {sz.avg_price:.2f} (${sz.cost:.2f}, EV {sz.ev:+.2f})" \
                            if sz.contracts else r["no_bet"] or "no edge"
                        mkt = f"{r['p_market']:.3f}" if r["p_market"] is not None else "-"
                        print(f"{t_live:%H:%M:%S}  {x['a']['name'][:18]:18} v {x['b']['name'][:18]:18} "
                              f"{(ls.problem or ls.raw) if ls else '':34} model {r['p_model']:.3f} mkt {mkt:>5}  {what}")
                f.flush()
                time.sleep(max(0.0, args.watch - (time.monotonic() - t0)))
        except KeyboardInterrupt:
            print("stopped")


if __name__ == "__main__":
    raise SystemExit(main())
