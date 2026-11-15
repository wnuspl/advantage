"""Predict the 2026 Nitto ATP Finals (Turin, indoor hard, best of 3, tiebreak sets) with a
points_v1 run (default 605; context offsets applied when the run has them, e.g. 790): replay the run over the full history in tennis.db, take each
player's state as of the event's start, compute every pairwise match probability with the
Markov roll-up, apply the latest walk-forward ATP calibration alpha, then Monte-Carlo the
round robin (ATP tiebreak rules approximated: wins, then head-to-head if two tied, then a
random draw standing in for sets/games %) and the knockouts.

    python archive/v1/scripts/predict_finals.py [--run 605] [--sims 200000]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date, timedelta
from itertools import combinations
from pathlib import Path

import numpy as np

ARCHIVE = Path(__file__).resolve().parents[1]   # archive/v1: results, reports, docs
ROOT = Path(__file__).resolve().parents[3]      # project root: tennis.db, data, src
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ARCHIVE / "src"))   # archived modules, merged into tennis

from tennis import calibrate, context, markov, models_db, points_rating, tuning  # noqa: E402
from tennis.matchup.data import fit_alpha  # noqa: E402

START = "2026-11-15"
GROUPS = {  # seed order, from atptour.com (draw of 2026-11-12)
    "John McEnroe": ["Alexander Zverev", "Ben Shelton", "Arthur Fils", "Rafael Jodar"],
    "Ivan Lendl": ["Carlos Alcaraz", "Daniil Medvedev", "Flavio Cobolli", "Frances Tiafoe"],
}


def fav_won(score: str) -> bool:
    """True if the first player won a best-of-3 scoreline like '6-4 3-6 7-6'."""
    won = 0
    for s in score.split():
        x, y = s.strip("[]").split("-")
        won += int(x.rstrip("+")) > int(y.rstrip("+"))
    return won >= 2


def context_at(m, edition, dob, pids, day: int) -> dict:
    """context.FEATURES for each player entering a new edition starting on ordinal `day`,
    replaying context.features()' per-player bookkeeping over all earlier matches."""
    st = {pid: context._P() for pid in pids}
    for i in range(len(m)):
        pair = (m.winner[i], m.loser[i])
        if pids.isdisjoint(pair):
            continue
        d0 = date.fromisoformat(m.start[i]).toordinal()
        for pid in pair:
            if pid not in st:
                continue
            s = st[pid]
            if s.ed != edition[i]:
                if s.ed is not None:
                    s.eds.append((s.start, s.n))
                    s.eds = s.eds[-12:]
                    if d0 - s.start > context.LAYOFF_DAYS:
                        s.since_return = 0
                    s.prev_start = s.start
                s.ed, s.start, s.n = edition[i], d0, 0
            if m.status[i] in context.PLAYED:
                s.n += 1
                if s.since_return < 99:
                    s.since_return += 1
    out = {}
    for pid, s in st.items():
        s.eds.append((s.start, s.n))                         # finals = a new edition
        if day - s.start > context.LAYOFF_DAYS:
            s.since_return = 0
        prev = s.start
        c = np.zeros(context.K)
        c[0] = float(s.since_return == 0)
        c[1] = float(1 <= s.since_return <= 4)
        gap = day - prev
        c[2] = float(28 < gap <= context.LAYOFF_DAYS)
        c[3] = float(0 < gap <= 7)
        c[4] = sum(k for d0, k in s.eds if 0 < day - d0 <= 28) / 5.0
        if dob.get(pid):
            age = (day - date.fromisoformat(dob[pid]).toordinal()) / 365.25
            c[5:9] = max(0, 22 - age), max(0, 26 - age), max(0, age - 28), max(0, age - 32)
        out[pid] = c
    return out


def find_player(con, name: str) -> int:
    first, last = name.split(" ", 1)
    rows = con.execute(
        "SELECT player_id, canonical_name FROM data.player WHERE tour='atp' AND "
        "canonical_name LIKE ? AND canonical_name LIKE ?", (f"%{first}%", f"%{last.replace('o', '_')}%")).fetchall()
    exact = [r for r in rows if r[1] == name or r[1].replace("ó", "o") == name]
    rows = exact or rows
    if len(rows) != 1:
        raise SystemExit(f"{name}: {len(rows)} matches {rows}")
    return rows[0][0]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=int, default=605)
    ap.add_argument("--sims", type=int, default=200_000)
    ap.add_argument("--json", type=Path, default=ARCHIVE / "results" / "atp_finals_2026.json")
    args = ap.parse_args(argv)

    con = models_db.connect(models_db.MODELS_DB)
    models_db.attach_data(con, models_db.DATA_DB)
    prep = tuning.prepare(con)
    m = prep.m
    spec, params = tuning.model_of(con, args.run)
    assert spec.model_id == "points_v1"
    spec.attach(con, m)
    names = [n for g in GROUPS.values() for n in g]
    pid = {n: find_player(con, n) for n in names}
    canon = dict(con.execute("SELECT player_id, canonical_name FROM data.player"))
    edmap = dict(con.execute("SELECT match_id, edition_id FROM data.matches"))
    dob = dict(con.execute("SELECT player_id, dob FROM data.player"))
    con.close()

    res = points_rating.run(m, params)
    st = res.state
    p = params
    day = date.fromisoformat(START).toordinal()
    bs, br = params.context_betas("atp")
    ctx = context_at(m, [edmap[x] for x in m.match_id], dob, set(pid.values()), day)         if params.has_context() else {i: np.zeros(context.K) for i in pid.values()}

    # calibration alpha as the walk-forward eval would set it for this quarter (ATP, 10 years)
    blk = "2026-10-01"
    hist = prep.indices(calibrate._add_years(blk, -10), "2026-09-30", "tour_atp")
    hp, hy = calibrate.oriented(prep, res.p_winner, hist)
    alpha = fit_alpha(np.log(np.clip(hp, 1e-12, 1 - 1e-12) / np.clip(1 - hp, 1e-12, 1)), hy)

    def part(i, surf="hard"):
        el = day - st.last[i]
        s, r = st.serve[i], st.ret[i]
        vs, vr = min(st.vs[i] + p.q * el, p.v_init), min(st.vr[i] + p.q * el, p.v_init)
        k = (i, surf)
        if k in st.sserve:
            el = day - st.slast[k]
            ds, dr = st.sserve[k], st.sret[k]
            dvs, dvr = min(st.svs[k] + p.q_surf * el, p.surf_var), min(st.svr[k] + p.q_surf * el, p.surf_var)
        else:
            ds = dr = 0.0
            dvs = dvr = p.surf_var
        return s, r, vs, vr, ds, dr, dvs, dvr

    b = st.base[("atp", "hard")]
    g = markov.grid(3, "tb")

    def prob(a, c):
        sa, ra, vsa, vra, dsa, dra, dvsa, dvra = part(pid[a])
        sc, rc, vsc, vrc, dsc, drc, dvsc, dvrc = part(pid[c])
        ca, cc = ctx[pid[a]], ctx[pid[c]]
        xa = b + sa - rc + p.blend * (dsa - drc) + ca @ bs - cc @ br
        xb = b + sc - ra + p.blend * (dsc - dra) + cc @ bs - ca @ br
        va = vsa + vrc + p.blend * (dvsa + dvrc)
        vb = vsc + vra + p.blend * (dvsc + dvra)
        pr, da, db = g.lookup_array(np.array([xa]), np.array([xb]), grad=True)
        raw = float(points_rating._shrink(pr, da, db, np.array([va]), np.array([vb]), p.unc)[0])
        cal = 1 / (1 + math.exp(-alpha * math.log(raw / (1 - raw))))
        return raw, cal, xa, xb

    P = {}
    pairs = []
    for a, c in combinations(names, 2):
        raw, cal, xa, xb = prob(a, c)
        P[(a, c)], P[(c, a)] = cal, 1 - cal
        spa, spb = 1 / (1 + math.exp(-xa)), 1 / (1 + math.exp(-xb))
        pairs.append({"a": a, "b": c, "p_a_raw": raw, "p_a": cal, "serve_pts_a": spa, "serve_pts_b": spb,
                      "sets": markov.set_count_dist(spa, spb, 3, "tb")})

    players = []
    for n in names:
        i = pid[n]
        s, r, vs, vr, ds, dr, dvs, dvr = part(i)
        players.append({"name": n, "db_name": canon[i], "matches": st.count.get(i, 0),
                        "last_match": date.fromordinal(st.last[i]).isoformat(),
                        "overall": s + r, "hard": s + r + ds + dr, "serve": s + ds, "return": r + dr,
                        "sd": math.sqrt(vs + vr), "context": dict(zip(context.FEATURES, ctx[i].tolist())),
                        "ctx_serve": float(ctx[i] @ bs), "ctx_return": float(ctx[i] @ br)})

    # Monte Carlo
    rng = np.random.default_rng(20261115)
    titles = {n: 0 for n in names}
    finals = dict(titles)
    semis = dict(titles)
    gnames = list(GROUPS)
    for _ in range(args.sims):
        placed = []
        for gname in gnames:
            gp = GROUPS[gname]
            wins = {n: 0 for n in gp}
            beat = set()
            for a, c in combinations(gp, 2):
                w, l = (a, c) if rng.random() < P[(a, c)] else (c, a)
                wins[w] += 1
                beat.add((w, l))
            tieb = {n: rng.random() for n in gp}

            def key(n):
                same = [o for o in gp if o != n and wins[o] == wins[n]]
                h2h = 1 if len(same) == 1 and (n, same[0]) in beat else 0
                return (wins[n], h2h, tieb[n])
            order = sorted(gp, key=key, reverse=True)
            placed.append(order[:2])
        (a1, a2), (b1, b2) = placed
        for n in (a1, a2, b1, b2):
            semis[n] += 1
        f1 = a1 if rng.random() < P[(a1, b2)] else b2
        f2 = b1 if rng.random() < P[(b1, a2)] else a2
        finals[f1] += 1
        finals[f2] += 1
        titles[f1 if rng.random() < P[(f1, f2)] else f2] += 1
    for pl in players:
        n = pl["name"]
        pl.update(semi=semis[n] / args.sims, final=finals[n] / args.sims, title=titles[n] / args.sims)

    out = {"run": args.run, "as_of": START, "alpha_atp": alpha, "base_atp_hard": b,
           "groups": GROUPS, "players": players, "pairs": pairs, "sims": args.sims,
           "note": "stats end 2026-05-25: later matches update on result only"}
    args.json.parent.mkdir(exist_ok=True)
    args.json.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")

    print(f"run {args.run}, ATP calibration alpha {alpha:.3f}")
    print(f"{'player':20} {'db name':22} {'n':>5} {'last':>10} {'hard':>6} {'sd':>5} {'SF':>6} {'F':>6} {'title':>6}")
    for pl in sorted(players, key=lambda x: -x["title"]):
        print(f"{pl['name']:20} {pl['db_name'][:22]:22} {pl['matches']:5} {pl['last_match']:>10} "
              f"{pl['hard']:6.3f} {pl['sd']:5.3f} {pl['semi']:6.1%} {pl['final']:6.1%} {pl['title']:6.1%}")
    print("\nround robin (calibrated P(first player wins); raw in brackets)")
    for gname, gp in GROUPS.items():
        print(f"  {gname} group")
        for a, c in combinations(gp, 2):
            q = next(x for x in pairs if x["a"] == a and x["b"] == c)
            print(f"    {a:18} vs {c:18} {q['p_a']:6.1%}  [{q['p_a_raw']:.1%}]")
    print("\nmost likely scoreline given the favourite wins (uncalibrated, independent points)")
    for q in pairs:
        a, c = q["a"], q["b"]
        fav, dog, spf, spd = (a, c, q["serve_pts_a"], q["serve_pts_b"]) if q["p_a"] >= 0.5 else \
            (c, a, q["serve_pts_b"], q["serve_pts_a"])
        dist = markov.match_score_dist(spf, spd, 3, "tb")
        wins = {k: v for k, v in dist.items() if fav_won(k)}
        tot = sum(wins.values())
        top = sorted(wins.items(), key=lambda kv: -kv[1])[:3]
        sets = q["sets"] if fav == a else {k[::-1]: v for k, v in q["sets"].items()}
        print(f"    {fav:18} def {dog:18} p={max(q['p_a'], 1 - q['p_a']):.0%}  2-0 {sets['2-0'] / tot:.0%} | "
              + "  ".join(f"{k} ({v:.1%})" for k, v in top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
