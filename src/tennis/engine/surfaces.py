"""Tournament -> surface from history, for markets whose edition is not in tennis.db yet.

Kalshi gives no surface anywhere (checked 2026-11-15: events, markets, milestones, live
data, /series, /events/<t>/metadata). It does name the tournament: the milestone's
`details.tournament_name` ("ATP Challenger Kobe", "ATP World Tour Finals Men Singles"),
and the market rules ("... match in the 2026 ATP Challenger Metz Round Of 32 after a ball
has been played ..."), which older pulls without milestones still have.

tennis.db has every edition's surface (1968-2026, tour level, Challengers to 2026-05,
ITF). The lookup tries tiers in order and stops at the first with an edition:

  challenger   'ATP/WTA Challenger <city>':
                 1. a Challenger / WTA 125 of that city ("Kobe CH", "Lima Los Inkas CH",
                    "Brisbane 1 CH", "Montevideo 125")
                 2. a tour-level event of that city (Metz 2026 is a Challenger; tennis.db
                    only has the 250 that was there before)
                 3. an ITF event of that city ("M25 Champaign IL")
  finals       '... Finals ...': the tour's season finals (not NextGen, Davis/BJK Cup,
               Challenger finals)
  tour         anything else: the name (minus 'ATP'/'WTA'/'Men Singles'/...) as a
               tournament alias, then as a city

Within the tier, the edition is the one from the last 6 years whose calendar week is
nearest the match (cities that host twice a year, e.g. Lima in June and October), the
latest on ties. `how` says which tier, tournament and edition, so it can be checked.
Surfaces from history are a guess: a venue can change (Metz). The retriever uses them
only when no running edition in tennis.db gives the surface.
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from datetime import date

from ..normalize import norm_name

YEARS = 6
WEEK_WINDOW = 35                       # days either side counted as "the same time of year"
TOUR_LEVELS = {"G", "M", "A", "F", "P", "PM", "I", "W", "X", "O", "T1", "T2", "T3", "T4", "T5"}
NOT_FINALS = ("next gen", "nextgen", "davis", "bjk", "fed cup", "challenger")
_ROUND = re.compile(r"\s+(qualification\s+(round\s+\d+|final)|round\s+of\s+\d+|round\s+\d+|"
                    r"quarterfinals?|semifinals?|final)\s*$", re.I)
_SINGLES = re.compile(r"\b(singles)\b.*$", re.I)    # "... Men Singles Ivan Lendl Group" -> "... Men Singles"
_RULES = re.compile(r"match in the \d{4} (.+?) after a ball", re.I)
_ITF = re.compile(r"^(m|w)\d+\s+|\s+\d+k$")


def kalshi_tournament(milestone: dict | None, rules: str = "") -> str:
    """The tournament name Kalshi gives, without the round: milestone first, else rules text."""
    name = ((milestone or {}).get("details") or {}).get("tournament_name") or ""
    if not name:
        m = _RULES.search(rules or "")
        name = m.group(1) if m else ""
    return _ROUND.sub("", _SINGLES.sub(r"\1", name)).strip()


@dataclass(frozen=True)
class SurfaceGuess:
    surface: str
    how: str                          # "<tier>: <tournament> <edition start>"
    tournament_id: int
    edition_start: str


class TournamentSurfaces:
    def __init__(self, tournaments: dict, aliases: dict):
        """tournaments: id -> (tour, canonical name, {levels}, [(start_date, surface)]);
        aliases: normalized alias -> [tournament ids]."""
        self.t, self.aliases = tournaments, aliases
        self.core = {tid: _core(name) for tid, (_, name, _, _) in tournaments.items()}

    @classmethod
    def from_db(cls, con: sqlite3.Connection, alias: str = "main") -> "TournamentSurfaces":
        t: dict = {}
        for tid, tour, name, level, start, surface in con.execute(
                f"""SELECT t.tournament_id, t.tour, t.canonical_name, e.level, e.start_date, e.surface
                    FROM {alias}.tournament t JOIN {alias}.tournament_edition e USING (tournament_id)"""):
            x = t.setdefault(tid, (tour, name, set(), []))
            x[2].add(level)
            if surface:
                x[3].append((start, surface))
        al: dict = {}
        for a, tid in con.execute(f"SELECT alias_normalized, tournament_id FROM {alias}.tournament_alias"):
            al.setdefault(a, []).append(tid)
        return cls(t, al)

    # -------------------------------------------------------------- candidates
    def _of(self, tour, pred):
        return [tid for tid, (tr, _, lv, _) in self.t.items() if tr == tour and pred(tid, lv)]

    def _city(self, tid, city):
        c = self.core[tid]
        return c == city or c.startswith(city + " ")

    def tiers(self, tour: str, name: str) -> list[tuple[str, list[int]]]:
        n = norm_name(name)
        kind_ch = re.match(r"^(atp|wta) challenger (.+)$", n)
        if kind_ch:
            city = kind_ch.group(2)
            return [("challenger", self._of(tour, lambda t, lv: "C" in lv and self._city(t, city))),
                    ("tour-level same city", self._of(tour, lambda t, lv: lv & TOUR_LEVELS and self._city(t, city))),
                    ("itf same city", self._of(tour, lambda t, lv: not (lv & TOUR_LEVELS) and "C" not in lv
                                               and "D" not in lv and self._city(t, city)))]
        if "finals" in n:
            return [("finals", self._of(tour, lambda t, lv: lv & TOUR_LEVELS and "finals" in self.core[t]
                                        and not any(w in self.core[t] for w in NOT_FINALS)))]
        core = re.sub(r"^(atp|wta)\s+|\s+(men|women)s?\s+singles$", "", n).strip()
        return [("tour alias", [t for t in self.aliases.get(core, []) if self.t[t][0] == tour]),
                ("tour-level same city", self._of(tour, lambda t, lv: lv & TOUR_LEVELS and self._city(t, core)))]

    # -------------------------------------------------------------- lookup
    def lookup(self, tour: str, name: str, on: str) -> SurfaceGuess | None:
        if not name:
            return None
        d = date.fromisoformat(on)
        lo = date(d.year - YEARS, d.month, min(d.day, 28)).isoformat()
        for tier, ids in self.tiers(tour, name):
            eds = [(s, sf, tid) for tid in ids for s, sf in self.t[tid][3] if lo <= s <= on]
            if not eds:
                continue

            def gap(s):
                x = abs(date.fromisoformat(s).timetuple().tm_yday - d.timetuple().tm_yday)
                return min(x, 365 - x)
            near = [e for e in eds if gap(e[0]) <= WEEK_WINDOW] or eds
            s, sf, tid = max(near)                       # latest start among the nearest-week editions
            return SurfaceGuess(sf, f"{tier}: {self.t[tid][1]} {s}", tid, s)
        return None


def _core(name: str) -> str:
    """'Brisbane 1 CH' -> 'brisbane 1', 'Montevideo 125' -> 'montevideo', 'M25 Champaign IL'
    -> 'champaign il', 'Tour Finals' -> 'tour finals'."""
    n = norm_name(name)
    n = re.sub(r"\s+(ch|125)$", "", n)
    return _ITF.sub("", n).strip()
