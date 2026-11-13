"""Entity resolution for sources without backbone IDs (MCP, slam point-by-point).

Cascade, in order (see docs/DESIGN.md):
  players: override -> exact normalized name (unique) -> context (career span covers the
           source's match dates) -> otherwise review_queue with fuzzy candidates from a
           block (same surname token or shared tokens). Fuzzy links are never auto-merged.
  matches: override -> unordered pair + edition-start window around the source date
           (or a known edition), round as tiebreaker -> otherwise review_queue.
"""
from __future__ import annotations

import json
import sqlite3
from difflib import SequenceMatcher

from .normalize import add_days, norm_name
from .overrides import Overrides

CONTEXT_MARGIN_DAYS = 60


def queue(con, entity, source, key, reason, candidates=None, detail=None, snapshot_id=None) -> None:
    con.execute(
        """INSERT OR IGNORE INTO review_queue (entity_type, source, source_key, candidate_ids,
               reason, detail, snapshot_id) VALUES (?,?,?,?,?,?,?)""",
        (entity, source, key, json.dumps(candidates) if candidates is not None else None, reason,
         json.dumps(detail, ensure_ascii=False) if detail is not None else None, snapshot_id))


class PlayerResolver:
    def __init__(self, con: sqlite3.Connection, overrides: Overrides):
        self.con = con
        self.ov = overrides
        self.alias: dict[tuple[str, str], list[int]] = {}
        self.blocks: dict[tuple[str, str], list[tuple[int, str]]] = {}
        for alias, pid, tour in con.execute(
                """SELECT a.alias_normalized, a.player_id, p.tour FROM player_alias a
                   JOIN player p USING (player_id) WHERE p.is_placeholder = 0"""):
            self.alias.setdefault((tour, alias), []).append(pid)
            for tok in set(alias.split()):
                if len(tok) >= 3:
                    self.blocks.setdefault((tour, tok), []).append((pid, alias))
        self.parts: dict[int, tuple[str, str]] = {
            pid: (norm_name(f), norm_name(l))
            for pid, f, l in con.execute("SELECT player_id, name_first, name_last FROM player")}
        self.span: dict[int, tuple[str, str]] = {}
        for pid, lo, hi in con.execute(
                """SELECT player_id, MIN(edition_start), MAX(edition_start) FROM v_player_match
                   GROUP BY player_id"""):
            self.span[pid] = (lo, hi)
        self.cache: dict[tuple[str, str], int | None] = {}
        self.outcomes: dict[str, int] = {}

    def _note(self, outcome: str) -> None:
        self.outcomes[outcome] = self.outcomes.get(outcome, 0) + 1

    def _xref(self, source, key, pid, method, conf, snapshot_id, verified=0):
        self.con.execute(
            """INSERT INTO player_xref (source, source_key, player_id, method, confidence, verified,
                   snapshot_id) VALUES (?,?,?,?,?,?,?)""",
            (source, key, pid, method, conf, verified, snapshot_id))

    def _covers(self, pid: int, d: str) -> bool:
        s = self.span.get(pid)
        return s is not None and add_days(s[0], -CONTEXT_MARGIN_DAYS) <= d <= add_days(s[1], CONTEXT_MARGIN_DAYS)

    def resolve(self, source: str, key: str, tour: str, raw_name: str, dates: list[str],
                snapshot_id: int, context: set[int] | None = None) -> int | None:
        """Resolve one source name to a player_id, writing player_xref or review_queue.

        `context`, when given, is the set of players known (from the backbone) to have
        played in the event(s) where this name appears. It disambiguates exact-name
        collisions and resolves 'A Bedene'-style initial + surname names.
        """
        if (source, key) in self.cache:
            return self.cache[(source, key)]
        pid = self._resolve(source, key, tour, raw_name, sorted(set(dates)), snapshot_id, context)
        self.cache[(source, key)] = pid
        return pid

    def _initial_match(self, n: str, context: set[int]) -> list[int]:
        toks = n.split()
        # 'a bedene', 'j l struff', 'a davidovich fokina': leading 1-letter initials + surname
        k = 0
        while k < len(toks) - 1 and len(toks[k]) == 1:
            k += 1
        if k == 0:
            return []
        initials, surname = toks[:k], " ".join(toks[k:])
        out = []
        for pid in context:
            first, last = self.parts.get(pid, ("", ""))
            if last == surname and first[:1] == initials[0]:
                out.append(pid)
        return out

    def _resolve(self, source, key, tour, raw_name, dates, snapshot_id, context=None):
        forced = self.ov.link_for("player", source, key)
        if forced is not None:
            self._xref(source, key, forced, "override", 1.0, snapshot_id, verified=1)
            self._note("override")
            return forced
        forbidden = self.ov.forbidden("player", source, key)
        n = norm_name(raw_name)
        cands = [p for p in self.alias.get((tour, n), []) if p not in forbidden]
        detail = {"name": raw_name, "tour": tour, "first_date": dates[0] if dates else None,
                  "last_date": dates[-1] if dates else None, "n_matches": len(dates)}
        if len(cands) == 1:
            self._xref(source, key, cands[0], "exact_name", 1.0, snapshot_id)
            self._note("exact_name")
            return cands[0]
        if len(cands) > 1 and context:
            in_event = [p for p in cands if p in context]
            if len(in_event) == 1:
                self._xref(source, key, in_event[0], "context", 0.98, snapshot_id)
                self._note("context_event")
                return in_event[0]
        if not cands and context:
            init = [p for p in self._initial_match(n, context) if p not in forbidden]
            if len(init) == 1:
                self._xref(source, key, init[0], "context", 0.95, snapshot_id)
                self._note("context_initials")
                return init[0]
        if len(cands) > 1:
            score = {p: sum(self._covers(p, d) for d in dates) for p in cands}
            best = max(score.values())
            top = [p for p, s in score.items() if s == best]
            if best > 0 and len(top) == 1:
                self._xref(source, key, top[0], "context", 0.95, snapshot_id)
                self._note("context")
                return top[0]
            detail["context_scores"] = {str(p): s for p, s in score.items()}
            queue(self.con, "player", source, key, "ambiguous", sorted(cands), detail, snapshot_id)
            self._note("ambiguous")
            return None
        # No exact candidate: propose fuzzy candidates from blocks, never auto-link.
        pool: dict[int, str] = {}
        for tok in set(n.split()):
            for pid, alias in self.blocks.get((tour, tok), []):
                if pid not in forbidden:
                    pool[pid] = alias
        scored = sorted(((SequenceMatcher(None, n, a).ratio(), p, a) for p, a in pool.items()),
                        reverse=True)[:5]
        scored = [s for s in scored if s[0] >= 0.6]
        if scored:
            detail["candidates"] = [{"player_id": p, "alias": a, "score": round(r, 3),
                                     "span": self.span.get(p),
                                     "covers_dates": sum(self._covers(p, d) for d in dates)}
                                    for r, p, a in scored]
            queue(self.con, "player", source, key, "low_confidence", [p for _, p, _ in scored],
                  detail, snapshot_id)
            self._note("fuzzy_queued")
        else:
            queue(self.con, "player", source, key, "no_candidate", [], detail, snapshot_id)
            self._note("no_candidate")
        return None


_CAND_SQL = """SELECT m.match_id, m.round, m.edition_id, e.start_date, t.canonical_name
    FROM matches m JOIN tournament_edition e USING (edition_id) JOIN tournament t USING (tournament_id)
    WHERE m.pair_lo_id=? AND m.pair_hi_id=? """


def match_candidates_window(con, p1: int, p2: int, start_from: str, start_to: str):
    return con.execute(_CAND_SQL + "AND e.start_date BETWEEN ? AND ?",
                       (min(p1, p2), max(p1, p2), start_from, start_to)).fetchall()


def match_candidates_edition(con, p1: int, p2: int, edition_id: int):
    return con.execute(_CAND_SQL + "AND m.edition_id=?",
                       (min(p1, p2), max(p1, p2), edition_id)).fetchall()


MAIN_DRAW_LEAD_DAYS = 2   # Sackmann dates an event by its Monday; some draws start a day or two early


def _narrow(cands: list, rnd: str | None, src_date: str | None, src_tournament: str | None) -> list:
    """Tiebreaks when a pair has several candidate matches near the source date."""
    if len(cands) > 1 and src_date and rnd and not rnd.startswith("Q"):
        # A main-draw match cannot be played before its event starts.
        started = [c for c in cands if c[3] is None or c[3] <= add_days(src_date, MAIN_DRAW_LEAD_DAYS)]
        cands = started or cands
    if len(cands) > 1 and rnd:
        cands = [c for c in cands if c[1] == rnd] or cands
    if len(cands) > 1 and src_tournament:
        t = norm_name(src_tournament)
        sims = sorted(((SequenceMatcher(None, t, norm_name(c[4])).ratio(), c) for c in cands),
                      key=lambda x: -x[0])
        if sims[0][0] >= 0.8 and sims[0][0] - sims[1][0] >= 0.2:
            cands = [sims[0][1]]
    if len(cands) > 1 and src_date:
        # Older MCP charts often carry the event's start date rather than the match date.
        cands = [c for c in cands if c[3] == src_date] or cands
    return cands


def link_match(con, ov: Overrides, source: str, key: str, cands: list, rnd: str | None,
               snapshot_id: int, detail: dict, src_date: str | None = None,
               src_tournament: str | None = None) -> tuple[int | None, str]:
    """Pick a backbone match from candidates; write match_xref or review_queue.

    Returns (match_id or None, outcome) with outcome in linked / queued:<reason>.
    """
    forced = ov.link_for("match", source, key)
    if forced is not None:
        _match_xref(con, source, key, forced, "override", 1.0, snapshot_id, verified=1)
        return forced, "linked"
    forbidden = ov.forbidden("match", source, key)
    cands = _narrow([c for c in cands if c[0] not in forbidden], rnd, src_date, src_tournament)
    if len(cands) == 1:
        conf = 1.0 if (rnd is None or cands[0][1] == rnd) else 0.9   # a pair meets once per event
        _match_xref(con, source, key, cands[0][0], "pair_tournament_date", conf, snapshot_id)
        return cands[0][0], "linked"
    reason = "ambiguous" if cands else "no_candidate"
    if cands:
        detail = {**detail, "candidates": [{"match_id": c[0], "round": c[1], "start": c[3],
                                            "tournament": c[4]} for c in cands]}
    queue(con, "match", source, key, reason, [c[0] for c in cands], detail, snapshot_id)
    return None, f"queued:{reason}"


def _match_xref(con, source, key, match_id, method, conf, snapshot_id, verified=0):
    con.execute(
        """INSERT INTO match_xref (source, source_key, match_id, method, confidence, verified,
               snapshot_id) VALUES (?,?,?,?,?,?,?)""",
        (source, key, match_id, method, conf, verified, snapshot_id))
