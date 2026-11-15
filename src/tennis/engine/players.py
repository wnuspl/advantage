"""Name -> tennis.db player_id, shared by the CLI, the web app and the Kalshi retriever
(from v1 kalshi_edges.Resolver, milestone A1).

    r = Resolver.from_db(con, today="2026-11-15")      # tennis.db attached as 'data'
    pid, how = r.resolve("atp", "Felix Auger-Aliassime") # how: exact / last+initial / unresolved

Resolution order:
  1. exact full name, ignoring accents, punctuation and spaces ("O'Connell" == "Oconnell");
  2. last name + first initial, among players with a match in the last 3 years only (an
     unrestricted fallback matched 'Colson Wells' to a 1970s 'Chris Wells'); multi-word last
     names are tried first.
Ties go to the player with the most matches; `how` says "(busiest of n)" when it mattered.
`search()` is a looser substring lookup for people typing names at a CLI.

The index is plain dicts and tuples, so it can be pickled into the predictor's cache.
"""
from __future__ import annotations

from datetime import date, timedelta

from ..normalize import norm_name

RECENT_YEARS = 3


def name_key(s: str | None) -> str:
    return norm_name(s).replace(" ", "")


class Resolver:
    def __init__(self, index: tuple):
        """index: from build_index(); (by_name, by_last, info)."""
        self.by_name, self.by_last, self.info = index

    @classmethod
    def from_db(cls, con, today: str, alias: str = "data") -> "Resolver":
        return cls(build_index(con, today, alias))

    @property
    def index(self) -> tuple:
        return self.by_name, self.by_last, self.info

    def resolve(self, tour: str, name: str) -> tuple[int | None, str]:
        hits = self.by_name.get((tour, name_key(name)))
        how = "exact"
        if not hits:
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

    __call__ = resolve

    def search(self, text: str, tour: str | None = None, limit: int = 5) -> list[tuple[int, str, str, int]]:
        """(player_id, tour, canonical name, matches) whose normalized name contains `text`,
        busiest first."""
        k = norm_name(text)
        out = [(pid, t, nm, n) for pid, (t, nm, n, _last) in self.info.items()
               if (tour is None or t == tour) and k in norm_name(nm)]
        return sorted(out, key=lambda r: (-r[3], r[0]))[:limit]

    def name(self, pid: int) -> str | None:
        x = self.info.get(pid)
        return x[1] if x else None

    def tour(self, pid: int) -> str | None:
        x = self.info.get(pid)
        return x[0] if x else None


def build_index(con, today: str, alias: str = "data") -> tuple:
    """(by_name, by_last, info) from tennis.db:
    by_name (tour, name_key) -> [(matches, pid)]; by_last (tour, last key, first initial) ->
    [(matches, pid)] for players with a match since today - 3 years; info pid -> (tour,
    canonical name, matches, last edition start)."""
    stats = {p: (k, last) for p, k, last in con.execute(
        f"""SELECT p, COUNT(*), MAX(d) FROM (
                SELECT m.winner_id p, e.start_date d FROM {alias}.matches m
                JOIN {alias}.tournament_edition e USING (edition_id)
                UNION ALL SELECT m.loser_id, e.start_date FROM {alias}.matches m
                JOIN {alias}.tournament_edition e USING (edition_id)) GROUP BY p""")}
    recent = (date.fromisoformat(today) - timedelta(days=RECENT_YEARS * 365)).isoformat()
    by_name: dict = {}
    by_last: dict = {}
    info: dict = {}
    for pid, tour, canon, first, last in con.execute(
            f"SELECT player_id, tour, canonical_name, name_first, name_last FROM {alias}.player "
            "WHERE is_placeholder = 0"):
        k, last_d = stats.get(pid, (0, None))
        info[pid] = (tour, canon, k, last_d)
        by_name.setdefault((tour, name_key(canon)), []).append((k, pid))
        if last and last_d and last_d >= recent:
            by_last.setdefault((tour, name_key(last), norm_name(first)[:1]), []).append((k, pid))
    return by_name, by_last, info
