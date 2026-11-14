"""Closing-odds benchmark (tennis-data.co.uk) and betting math.

Pipeline:
  1. `read_td_file` parses one tennis-data.co.uk yearly .xlsx (header names are read, never
     assumed: bookmaker pairs are every `<X>W` / `<X>L` column pair).
  2. `link` resolves each row to a backbone match (tour-level main draw, same tour, edition
     window around the row's date, both names compatible). Every row ends linked or with a
     reason; nothing is dropped silently.
  3. `devig` turns a pair of decimal odds into fair probabilities (multiplicative, power or
     Shin), so the market is scored like any model.
  4. Scoring reuses composite_eval.metrics; `blend_fit` asks whether a model adds
     information to the market (the real test); `backtest` bets the model against the
     quoted (not devigged) odds.
  5. `simulate` answers "how accurate must a model be to profit" on synthetic matches whose
     market-probability distribution is taken from the real closing odds.

Odds are a benchmark only. They never enter a feature or a rating.
"""
from __future__ import annotations

import math
import re
import sqlite3
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np

from .xlsx_lite import read_first_sheet, serial_to_iso

EPS = 1e-12
NON_ODDS_PREFIXES = {"", "Wsets", "Lsets"}   # guard; W1..W5 / WRank etc. never pair as <X>W


# ------------------------------------------------------------------ parsing

@dataclass
class TdRow:
    source_key: str            # "<file stem>:<row number>", stable for a pinned file
    tour: str
    date: str                  # match day, ISO
    tournament: str
    round: str
    winner: str
    loser: str
    comment: str
    surface: str
    odds: dict = field(default_factory=dict)   # bookmaker -> (odds winner, odds loser)


def odds_pairs(header: list[str]) -> list[str]:
    """Bookmaker prefixes X with both XW and XL columns (e.g. PS, B365, Max, Avg)."""
    cols = set(header)
    out = []
    for h in header:
        if h.endswith("W") and len(h) > 1:
            x = h[:-1]
            if x + "L" in cols and x not in NON_ODDS_PREFIXES and not x.isdigit():
                out.append(x)
    return out


def _num(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(str(v).replace(",", "."))
    except ValueError:
        return None


def read_td_file(path: Path, tour: str) -> tuple[list[TdRow], list[str]]:
    """Rows of one tennis-data file plus its header (for the record)."""
    header, rows, d1904 = read_first_sheet(path)
    ix = {h: i for i, h in enumerate(header)}
    need = ["Date", "Tournament", "Round", "Winner", "Loser"]
    missing = [c for c in need if c not in ix]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}; header = {header}")
    books = odds_pairs(header)
    out = []
    for n, r in enumerate(rows, start=2):
        def g(c):
            i = ix.get(c)
            return r[i] if i is not None and i < len(r) else None
        d = g("Date")
        if isinstance(d, float):
            d = serial_to_iso(d, d1904)
        elif isinstance(d, str):
            d = _parse_text_date(d)
        od = {}
        for b in books:
            w, l = _num(g(b + "W")), _num(g(b + "L"))
            if w and l and w > 1 and l > 1:
                od[b] = (w, l)
        out.append(TdRow(f"{Path(path).stem}:{n}", tour, d, str(g("Tournament") or ""),
                         str(g("Round") or ""), str(g("Winner") or ""), str(g("Loser") or ""),
                         str(g("Comment") or ""), str(g("Surface") or ""), od))
    return out, header


def _parse_text_date(s: str) -> str | None:
    for pat in (r"(\d{4})-(\d{2})-(\d{2})", r"(\d{1,2})/(\d{1,2})/(\d{4})"):
        m = re.match(pat, s)
        if m:
            a = [int(x) for x in m.groups()]
            return (date(a[0], a[1], a[2]) if pat.startswith(r"(\d{4})")
                    else date(a[2], a[1], a[0])).isoformat()
    return None


# ------------------------------------------------------------------ names

def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    return "".join(c for c in s if c.isalpha()).lower()


def _tokens(s: str) -> list[str]:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    return [norm(t) for t in re.split(r"[\s\-']+", s) if norm(t)]


def parse_td_name(name: str) -> tuple[list[str], str]:
    """'Auger-Aliassime F.' -> (['auger','aliassime'], 'f'); 'Wolf J.J.' -> (['wolf'], 'j').
    Trailing tokens containing a '.' are initials."""
    parts = (name or "").strip().split()
    while len(parts) > 1 and "." in parts[-1]:
        ini = parts.pop()
        initial = norm(ini)[:1]
        return _tokens(" ".join(parts)), initial
    return _tokens(" ".join(parts)), ""


def name_score(td_name: str, last: str, first: str, canonical: str) -> int:
    """0 = incompatible; higher = better. Surname must match (exactly, or every td surname
    token appears in our full name); the first initial adds confidence."""
    toks, ini = parse_td_name(td_name)
    if not toks:
        return 0
    ours_last = _tokens(last)
    ours_all = set(_tokens(canonical)) | set(ours_last) | set(_tokens(first))
    if "".join(toks) == "".join(ours_last):
        s = 3
    elif all(t in ours_all for t in toks):
        s = 2
    elif ours_last and "".join(ours_last).startswith("".join(toks)):
        s = 1
    else:
        return 0
    firsts = {t[:1] for t in _tokens(first)} | {t[:1] for t in _tokens(canonical)}
    if ini:
        s += 1 if ini in firsts else -1
    return max(s, 0)


def name_score_full(full: str, last: str, first: str, canonical: str) -> int:
    """Scorer for sources with full names ('Felix Auger-Aliassime'). 0 = incompatible.
    4: same token set; 3: our surname tokens and first initial present; 2: one token set
    contains the other (middle names, compound surnames)."""
    theirs = _tokens(full)
    if not theirs:
        return 0
    ours = _tokens(canonical) or (_tokens(first) + _tokens(last))
    if sorted(theirs) == sorted(ours) or "".join(theirs) == "".join(ours):
        return 4
    ours_last = _tokens(last)
    st, so = set(theirs), set(ours)
    initials = {t[:1] for t in _tokens(first)}
    if ours_last and set(ours_last) <= st and (st - set(ours_last)) and             {t[:1] for t in st - set(ours_last)} & initials:
        return 3
    if (st <= so or so <= st) and len(st & so) >= 2:
        return 2
    return 0


# Valuebetennis (valuebetennis.com/en/donnees.htm, CC BY 4.0): one CSV per season,
# ';'-separated, UTF-8 with BOM, full names, Pinnacle opening and closing odds.
VB_TOUR_CATEGORIES = {"Main tour", "Masters", "Grand Chelem", "Grand Slam", ""}


def read_vb_file(path: Path) -> tuple[list[TdRow], dict]:
    """Rows of one Valuebetennis season file, winner/loser oriented. Returns (rows, counts):
    rows outside tour-level categories or without a winner are counted, not returned."""
    import csv
    path = Path(path)
    counts = {"read": 0, "not_tour_category": 0, "no_winner": 0, "kept": 0}
    out = []
    with open(path, encoding="utf-8-sig", newline="", errors="replace") as fh:
        rd = csv.DictReader(fh, delimiter=";")
        need = {"match_id", "date", "categorie", "genre", "joueur1", "joueur1_id", "joueur2",
                "joueur2_id", "cote1_cloture", "cote2_cloture", "cote1_ouverture",
                "cote2_ouverture", "vainqueur_id"}
        missing = need - set(rd.fieldnames or [])
        if missing:
            raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
        for r in rd:
            counts["read"] += 1
            if r["categorie"] not in VB_TOUR_CATEGORIES:
                counts["not_tour_category"] += 1
                continue
            w = r["vainqueur_id"]
            if w not in (r["joueur1_id"], r["joueur2_id"]) or not w:
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
            out.append(TdRow(f"vb:{r['match_id']}", r["genre"], r["date"][:10], r.get("tournoi", ""),
                             r.get("tour", ""), win, lose, r.get("score", ""), r.get("surface", ""), od))
            counts["kept"] += 1
    return out, counts


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
    """Tour-level main-draw matches (the 'tour' scope) from date_from, plus player names."""
    from .elo import TOUR_LEVELS
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
    row: TdRow
    match_id: int | None
    reason: str              # 'linked', 'no_candidate', 'ambiguous', 'no_date', 'winner_disagrees'
    score: int = 0


def link(rows: list[TdRow], ours: list[OurMatch], names: dict, *, before: int = 17,
         after: int = 1, day_tol: int = 2, scorer=None) -> list[Link]:
    """Each odds row -> one of our matches. Candidate = same tour, edition start in
    [date - before, date + after], exact match day (when we have one) within day_tol, and
    both names compatible (`scorer`, default the tennis-data 'Last F.' scorer). Best total
    name score wins; a tie is 'ambiguous'. A pair found with winner and loser swapped is
    'winner_disagrees' (not linked: the sources disagree on the result). A backbone match
    is used at most once (the best-scoring claim keeps it, others get 'duplicate_claim')."""
    scorer = scorer or name_score
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
            last, first, canon = names.get(pid, ("", "", ""))
            for t in set(_tokens(canon)) | set(_tokens(last)):
                tok_index.setdefault(t, set()).add(pid)
    pcache: dict[str, dict[int, int]] = {}

    def cands(name: str) -> dict[int, int]:
        if name not in pcache:
            pids = set()
            for t in _tokens(name.replace(".", " ")):
                if len(t) > 1:
                    pids |= tok_index.get(t, set())
            got = {}
            for pid in pids:
                last, first, canon = names.get(pid, ("", "", ""))
                sc = scorer(name, last, first, canon)
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
            delta = (d - _d(st)).days
            if not (-after <= delta <= before):
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
    return 1.0 / np.asarray(o1) + 1.0 / np.asarray(o2) - 1.0


def devig(o1, o2, method: str = "multiplicative"):
    """Fair probability of side 1 from decimal odds (arrays ok)."""
    q1, q2 = 1.0 / np.asarray(o1, float), 1.0 / np.asarray(o2, float)
    if method == "multiplicative":
        return q1 / (q1 + q2)
    if method == "additive":
        return np.clip(q1 - (q1 + q2 - 1) / 2, EPS, 1 - EPS)
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


def log_loss(p, y):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def blend_fit(y, X, iters: int = 50):
    """Logistic regression without intercept (orientation is arbitrary, so none belongs).
    Returns (coef, se). Newton-Raphson."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    b = np.zeros(X.shape[1])
    b[0] = 1.0
    for _ in range(iters):
        p = sigmoid(X @ b)
        W = p * (1 - p)
        H = X.T @ (X * W[:, None]) + 1e-9 * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p))
        b += step
        if np.max(np.abs(step)) < 1e-10:
            break
    p = sigmoid(X @ b)
    H = X.T @ (X * (p * (1 - p))[:, None])
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    return b, se


# ------------------------------------------------------------------ betting

@dataclass
class Backtest:
    label: str
    n_matches: int
    n_bets: int
    staked: float
    profit: float
    roi: float
    roi_se: float
    hit_rate: float
    mean_odds: float
    kelly_log_growth: float      # mean log bankroll growth per match, quarter Kelly


def backtest(p1, y, o1, o2, *, threshold: float = 0.0, kelly_frac: float = 0.25,
             label: str = "", max_odds: float | None = None) -> Backtest:
    """Bet the side with the larger expected value p*o - 1 when it exceeds `threshold`.
    Flat 1-unit stakes for ROI; quarter Kelly (fraction of bankroll, not compounded across
    simultaneous matches) for the log-growth figure. y = 1 if side 1 won."""
    p1, y, o1, o2 = (np.asarray(a, float) for a in (p1, y, o1, o2))
    ev1, ev2 = p1 * o1 - 1, (1 - p1) * o2 - 1
    side1 = ev1 >= ev2
    ev = np.where(side1, ev1, ev2)
    o = np.where(side1, o1, o2)
    p = np.where(side1, p1, 1 - p1)
    won = np.where(side1, y == 1, y == 0)
    bet = ev > threshold
    if max_odds is not None:
        bet &= o <= max_odds
    ret = np.where(won, o - 1, -1.0)[bet]
    n = int(bet.sum())
    f = np.clip(kelly_frac * ev / (o - 1), 0, 0.5)
    g = np.where(bet, np.log1p(np.where(won, f * (o - 1), -f)), 0.0)
    return Backtest(label, len(y), n, float(n), float(ret.sum()),
                    float(ret.mean()) if n else float("nan"),
                    float(ret.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan"),
                    float(won[bet].mean()) if n else float("nan"),
                    float(o[bet].mean()) if n else float("nan"),
                    float(g.mean()))


# ------------------------------------------------------------------ simulation

@dataclass
class SimPoint:
    kappa: float           # share of the market's missing information the model has
    sigma_noise: float     # model's own noise (logit sd)
    margin: float
    ll_market: float
    ll_model: float
    gap: float             # ll_model - ll_market (negative = model scores better)
    blend_b: float         # coefficient on the model in y ~ a*logit(market) + b*logit(model)
    blend_gain: float      # ll_market - ll(blend): information the model adds to the market
    roi: float
    roi_se: float
    bet_share: float
    growth: float


def simulate(market_logits, *, sigma_hidden: float, kappas, noises, margins,
             threshold: float = 0.0, n: int = 400_000, seed: int = 0) -> list[SimPoint]:
    """Synthetic matches: truth logit z = a + h, a drawn from the real market's logits
    (resampled), h ~ N(0, sigma_hidden^2) = information the closing market does not have.
    The market quotes its calibrated probability E[sigmoid(z) | a] with a multiplicative
    margin. The model sees a + kappa*h + e, e ~ N(0, sigma_noise^2), then is recalibrated
    (1-d logistic fit, as a careful modeller would). For each (kappa, noise, margin): log
    losses, blend coefficient, and the ROI of betting on positive expected value."""
    rng = np.random.default_rng(seed)
    a = rng.choice(np.asarray(market_logits, float), size=n, replace=True)
    h = rng.normal(0, sigma_hidden, n)
    z = a + h
    y = (rng.random(n) < sigmoid(z)).astype(float)
    # market's calibrated probability given a: probit approximation of E[sigmoid(a + h)]
    pm = sigmoid(a / math.sqrt(1 + math.pi * sigma_hidden ** 2 / 8))
    llm = log_loss(pm, y)
    out = []
    half = n // 2
    for kappa in kappas:
        for s in noises:
            sig = a + kappa * h + rng.normal(0, s, n)
            # recalibrate on the first half, evaluate on the second
            (c,), _ = blend_fit(y[:half], sig[:half, None])
            pp = sigmoid(c * sig[half:])
            yy, pmm = y[half:], pm[half:]
            (ba, bb), _ = blend_fit(yy, np.column_stack([logit(pmm), logit(pp)]))
            llp = log_loss(pp, yy)
            gain = log_loss(pmm, yy) - log_loss(sigmoid(ba * logit(pmm) + bb * logit(pp)), yy)
            for mg in margins:
                o1 = 1 / (pmm * (1 + mg))
                o2 = 1 / ((1 - pmm) * (1 + mg))
                bt = backtest(pp, yy, o1, o2, threshold=threshold)
                out.append(SimPoint(kappa, s, mg, log_loss(pmm, yy), llp, llp - log_loss(pmm, yy),
                                    float(bb), gain, bt.roi, bt.roi_se, bt.n_bets / len(yy),
                                    bt.kelly_log_growth))
    return out
