"""Closing-odds benchmark (track B5, docs/PLAN.md): Valuebetennis Pinnacle odds linked to
backbone matches, de-vigged, and the blend and betting arithmetic used to compare a model
with the market. Brought back from v1 (`archive/v1/src/tennis/odds.py`); the
tennis-data.co.uk xlsx reader and the betting simulation stay archived.

Pipeline:
  1. `read_vb_file` parses one Valuebetennis season CSV (columns checked, never assumed).
  2. `link` resolves each row to a tour-level main-draw backbone match (same tour, edition
     window around the row's date, both names compatible). Every row ends linked or with a
     reason; nothing is dropped silently.
  3. `devig` turns a pair of decimal odds into fair probabilities (multiplicative, power,
     Shin), so the market is scored like any model.
  4. `blend_fit` asks whether a model adds information to the market (the real test);
     `backtest` bets the model against the quoted (not de-vigged) odds.

Odds are a benchmark only. They never enter a feature or a rating.
"""
from __future__ import annotations

import csv
import math
import re
import sqlite3
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np

from .matches import TOUR_LEVELS

EPS = 1e-12


@dataclass
class OddsRow:
    source_key: str            # "vb:<match_id>", stable for a pinned file
    tour: str                  # 'atp' / 'wta'
    date: str                  # match day, ISO
    tournament: str
    round: str
    winner: str
    loser: str
    comment: str
    surface: str
    odds: dict = field(default_factory=dict)   # 'PS_close' / 'PS_open' -> (odds winner, odds loser)


def _num(v):
    if v is None:
        return None
    try:
        return float(str(v).replace(",", "."))
    except ValueError:
        return None


# Valuebetennis (valuebetennis.com/en/donnees.htm, CC BY 4.0): one CSV per season,
# ';'-separated, UTF-8 with BOM, full names, Pinnacle opening and closing odds.
VB_TOUR_CATEGORIES = {"Main tour", "Masters", "Grand Chelem", "Grand Slam", ""}
VB_COLUMNS = {"match_id", "date", "categorie", "genre", "joueur1", "joueur1_id", "joueur2", "joueur2_id",
              "cote1_cloture", "cote2_cloture", "cote1_ouverture", "cote2_ouverture", "vainqueur_id"}


def read_vb_file(path: Path) -> tuple[list[OddsRow], dict]:
    """Rows of one season file, winner/loser oriented. Returns (rows, counts): rows outside
    tour-level categories or without a winner are counted, not returned."""
    path = Path(path)
    counts = {"read": 0, "not_tour_category": 0, "no_winner": 0, "kept": 0}
    out = []
    with open(path, encoding="utf-8-sig", newline="", errors="replace") as fh:
        rd = csv.DictReader(fh, delimiter=";")
        missing = VB_COLUMNS - set(rd.fieldnames or [])
        if missing:
            raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
        for r in rd:
            counts["read"] += 1
            if r["categorie"] not in VB_TOUR_CATEGORIES:
                counts["not_tour_category"] += 1
                continue
            w = r["vainqueur_id"]
            if not w or w not in (r["joueur1_id"], r["joueur2_id"]):
                counts["no_winner"] += 1
                continue
            first_won = w == r["joueur1_id"]
            od = {}
            for kind, a, b in (("PS_close", "cote1_cloture", "cote2_cloture"),
                               ("PS_open", "cote1_ouverture", "cote2_ouverture")):
                x, y = _num(r[a]), _num(r[b])
                if x and y and x > 1 and y > 1:
                    od[kind] = (x, y) if first_won else (y, x)
            win, lose = (r["joueur1"], r["joueur2"]) if first_won else (r["joueur2"], r["joueur1"])
            out.append(OddsRow(f"vb:{r['match_id']}", r["genre"].strip().lower(), r["date"][:10],
                               r.get("tournoi", ""), r.get("tour", ""), win, lose, r.get("score", ""),
                               r.get("surface", ""), od))
            counts["kept"] += 1
    return out, counts


# ------------------------------------------------------------------ names

def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    return "".join(c for c in s if c.isalpha()).lower()


def _tokens(s: str) -> list[str]:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    return [norm(t) for t in re.split(r"[\s\-'.]+", s) if norm(t)]


def name_score(full: str, last: str, first: str, canonical: str) -> int:
    """Full names ('Felix Auger-Aliassime') against ours. 0 = incompatible.
    4: same tokens; 3: our surname tokens plus a matching first initial; 2: one token set
    contains the other, sharing at least two tokens (middle names, compound surnames)."""
    theirs = _tokens(full)
    if not theirs:
        return 0
    ours = _tokens(canonical) or (_tokens(first) + _tokens(last))
    if sorted(theirs) == sorted(ours) or "".join(theirs) == "".join(ours):
        return 4
    ours_last = _tokens(last)
    st, so = set(theirs), set(ours)
    initials = {t[:1] for t in _tokens(first)}
    rest = st - set(ours_last)
    if ours_last and set(ours_last) <= st and rest and {t[:1] for t in rest} & initials:
        return 3
    if (st <= so or so <= st) and len(st & so) >= 2:
        return 2
    return 0


# ------------------------------------------------------------------ linking

@dataclass
class OurMatch:
    match_id: int
    tour: str
    start: str
    match_date: str | None
    winner: int
    loser: int
    status: str


def load_our_matches(con: sqlite3.Connection, date_from: str, alias: str = "main"):
    """Tour-level main-draw matches (the 'tour' scoring scope) from date_from, plus player
    names {player_id: (last, first, canonical)}."""
    q = f"""SELECT m.match_id, t.tour, e.start_date, m.match_date, m.winner_id, m.loser_id,
                   m.status, e.level, m.round
            FROM {alias}.matches m JOIN {alias}.tournament_edition e USING (edition_id)
            JOIN {alias}.tournament t USING (tournament_id)
            WHERE e.start_date >= ?"""
    out = []
    for mid, tour, st, md, w, l, status, level, rnd in con.execute(q, (date_from,)):
        if level in TOUR_LEVELS[tour] and not rnd.startswith("Q"):
            out.append(OurMatch(mid, tour, st, md, w, l, status))
    names = {pid: (last or "", first or "", canon or "") for pid, last, first, canon in
             con.execute(f"SELECT player_id, name_last, name_first, canonical_name FROM {alias}.player")}
    return out, names


def _d(s: str) -> date:
    return date.fromisoformat(s)


@dataclass
class Link:
    row: OddsRow
    match_id: int | None
    reason: str              # linked / no_candidate / ambiguous / no_date / winner_disagrees / duplicate_claim
    score: int = 0


def link(rows: list[OddsRow], ours: list[OurMatch], names: dict, *, before: int = 17,
         after: int = 1, day_tol: int = 2, scorer=name_score) -> list[Link]:
    """Each odds row -> one of our matches. Candidate = same tour, edition start in
    [date - before, date + after], exact match day (when we have one) within day_tol, both
    names compatible. Best total name score wins; a tie is 'ambiguous'. A pair found with
    winner and loser swapped is 'winner_disagrees' (the sources disagree on the result:
    mostly retirements). A backbone match is used at most once (the best-scoring claim keeps
    it, the others get 'duplicate_claim')."""
    by_tsp: dict[tuple[str, str], dict[int, list[OurMatch]]] = {}
    starts: dict[str, list[str]] = {}
    tok_index: dict[str, set[int]] = {}
    for m in ours:
        d = by_tsp.setdefault((m.tour, m.start), {})
        if not d:
            starts.setdefault(m.tour, []).append(m.start)
        d.setdefault(m.winner, []).append(m)
        d.setdefault(m.loser, []).append(m)
        for pid in (m.winner, m.loser):
            last, _, canon = names.get(pid, ("", "", ""))
            for t in set(_tokens(canon)) | set(_tokens(last)):
                tok_index.setdefault(t, set()).add(pid)
    pcache: dict[str, dict[int, int]] = {}

    def cands(name: str) -> dict[int, int]:
        if name not in pcache:
            pids = set()
            for t in _tokens(name):
                if len(t) > 1:
                    pids |= tok_index.get(t, set())
            got = {}
            for pid in pids:
                sc = scorer(name, *names.get(pid, ("", "", "")))
                if sc:
                    got[pid] = sc
            pcache[name] = got
        return pcache[name]

    results = []
    for r in rows:
        if not r.date:
            results.append(Link(r, None, "no_date"))
            continue
        d = _d(r.date)
        cw, cl = cands(r.winner), cands(r.loser)
        found, swapped = {}, 0
        for st in starts.get(r.tour, []):
            if not (-after <= (d - _d(st)).days <= before):
                continue
            idx = by_tsp[(r.tour, st)]
            for wp in cw:
                for m in idx.get(wp, ()):
                    if m.match_date and abs((d - _d(m.match_date)).days) > day_tol:
                        continue
                    if m.winner == wp and m.loser in cl:
                        found[m.match_id] = cw[wp] + cl[m.loser]
                    elif m.loser == wp and m.winner in cl:
                        swapped += 1
        if not found:
            results.append(Link(r, None, "winner_disagrees" if swapped else "no_candidate"))
            continue
        ranked = sorted(((sc, mid) for mid, sc in found.items()), reverse=True)
        if len(ranked) > 1 and ranked[0][0] == ranked[1][0]:
            results.append(Link(r, None, "ambiguous", ranked[0][0]))
        else:
            results.append(Link(r, ranked[0][1], "linked", ranked[0][0]))
    best: dict[int, Link] = {}
    for lk in results:
        if lk.match_id is not None:
            cur = best.get(lk.match_id)
            if cur is None or lk.score > cur.score:
                best[lk.match_id] = lk
    for lk in results:
        if lk.match_id is not None and best[lk.match_id] is not lk:
            lk.match_id, lk.reason = None, "duplicate_claim"
    return results


# ------------------------------------------------------------------ devig

def overround(o1, o2):
    return 1.0 / np.asarray(o1, float) + 1.0 / np.asarray(o2, float) - 1.0


def devig(o1, o2, method: str = "power"):
    """Fair probability of side 1 from decimal odds (arrays ok)."""
    q1, q2 = 1.0 / np.asarray(o1, float), 1.0 / np.asarray(o2, float)
    if method == "multiplicative":
        return q1 / (q1 + q2)
    if method == "power":       # p_i = q_i^k with sum = 1
        lo, hi = np.full_like(q1, 0.2), np.full_like(q1, 5.0)
        for _ in range(60):
            k = (lo + hi) / 2
            s = q1 ** k + q2 ** k
            hi = np.where(s < 1, k, hi)
            lo = np.where(s >= 1, k, lo)
        k = (lo + hi) / 2
        return q1 ** k / (q1 ** k + q2 ** k)
    if method == "shin":
        Q = q1 + q2

        def pz(q, z):
            return (np.sqrt(z * z + 4 * (1 - z) * q * q / Q) - z) / (2 * (1 - z))
        lo, hi = np.zeros_like(q1), np.full_like(q1, 0.4)
        for _ in range(60):
            z = (lo + hi) / 2
            s = pz(q1, z) + pz(q2, z)
            lo = np.where(s > 1, z, lo)
            hi = np.where(s <= 1, z, hi)
        z = (lo + hi) / 2
        p1, p2 = pz(q1, z), pz(q2, z)
        return p1 / (p1 + p2)
    raise ValueError(method)


# ------------------------------------------------------------------ scoring helpers

def logit(p):
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, float)))


def log_losses(p, y):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def blend_fit(y, X, iters: int = 50):
    """Logistic regression without intercept (orientation is arbitrary, so none belongs).
    Returns (coef, se). Newton-Raphson, starting from (1, 0, ...)."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    b = np.zeros(X.shape[1])
    b[0] = 1.0
    for _ in range(iters):
        p = sigmoid(X @ b)
        H = X.T @ (X * (p * (1 - p))[:, None]) + 1e-9 * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p))
        b += step
        if np.max(np.abs(step)) < 1e-10:
            break
    p = sigmoid(X @ b)
    H = X.T @ (X * (p * (1 - p))[:, None])
    return b, np.sqrt(np.diag(np.linalg.inv(H)))


def model_weight(a: float, b: float) -> float:
    """The blend y ~ a*logit(market) + b*logit(model) as engine.decide's model_weight
    w = b / (a + b) (its blend is (1-w)*logit(market) + w*logit(model); the overall scale
    a + b, a sharpening, is dropped)."""
    return b / (a + b)


# ------------------------------------------------------------------ betting

@dataclass
class Backtest:
    label: str
    n_matches: int
    n_bets: int
    profit: float
    roi: float
    roi_se: float
    hit_rate: float
    mean_odds: float
    kelly_log_growth: float      # mean log bankroll growth per match, quarter Kelly


def backtest(p1, y, o1, o2, *, threshold: float = 0.0, kelly_frac: float = 0.25,
             label: str = "") -> Backtest:
    """Bet the side with the larger expected value p*o - 1 when it exceeds `threshold`.
    Flat 1-unit stakes for ROI; fractional Kelly (not compounded across simultaneous
    matches) for the log-growth figure. y = 1 if side 1 won."""
    p1, y, o1, o2 = (np.asarray(a, float) for a in (p1, y, o1, o2))
    ev1, ev2 = p1 * o1 - 1, (1 - p1) * o2 - 1
    side1 = ev1 >= ev2
    ev = np.where(side1, ev1, ev2)
    o = np.where(side1, o1, o2)
    won = np.where(side1, y == 1, y == 0)
    bet = ev > threshold
    ret = np.where(won, o - 1, -1.0)[bet]
    n = int(bet.sum())
    f = np.clip(kelly_frac * ev / (o - 1), 0, 0.5)
    g = np.where(bet, np.log1p(np.where(won, f * (o - 1), -f)), 0.0)
    return Backtest(label, len(y), n, float(ret.sum()),
                    float(ret.mean()) if n else float("nan"),
                    float(ret.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan"),
                    float(won[bet].mean()) if n else float("nan"),
                    float(o[bet].mean()) if n else float("nan"),
                    float(g.mean()) if len(y) else float("nan"))
