"""AcesAPI tournament draws -> backbone editions/matches after the Sackmann cutoff.

Inputs are pull folders data/raw/acesapi-<YYYYMMDDTHHMMSSZ>/ written by
scripts/fetch_acesapi.py, each pinned in data/build_manifest.json like any other snapshot.
One file per tournament per pull: draws/<tour>/<tournament_id>.json (get_tournament).

Timestamps (see docs/DESIGN.md "AcesAPI"):
- AcesAPI rewrites the current season every day, so the same tournament appears in
  several pulls. Within a build the LATEST pull of each tournament is loaded; its rows in
  earlier pulls are read and skipped as `superseded_by_later_pull` (load_run still
  accounts for every row).
- `match_seen` records, per loaded match, the first and last pull in which it was final
  and whether its result changed between pulls. A pull's time is when we knew the result.
- matches get match_date (day precision) from the API's match day.

Overlap with Sackmann: every match is first matched to the backbone (unordered pair, edition
start +-3 days, round tiebreak). Linked rows are the validation sample. A row that does not
link is inserted only if its edition starts after the Sackmann frontier (last backbone
edition start of that tour minus 7 days); before that it is queued (`no_backbone_match`),
because the backbone should already have it.

New players: a name that does not resolve is created as a new player only in an edition
past the frontier, and only when the resolver found no candidate at all, or every fuzzy
candidate has a different country. Otherwise the match is queued.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field

from .acesapi import (PULL_ID, SOURCE, draw_matches, edition_start, is_qualifying, map_round,
                      match_day, match_status, n_main_rounds, n_qual_rounds, read_json)
from .loadrun import LoadRun
from .normalize import ROUND_ORDER, add_days, norm_name, norm_surface
from .overrides import Overrides
from .resolve import PlayerResolver, link_match, match_candidates_window, opponents, queue
from .snapshots import BuildError, RawFile

LINK_WINDOW_DAYS = 3        # edition start difference allowed when linking to the backbone
FRONTIER_MARGIN_DAYS = 7    # editions starting on/before (last backbone start - this) must link
SAME_EVENT_DAYS = 21        # previous-year edition within +-this of the anniversary = same event
UNKNOWN_TIER = "X"          # tournament_edition.level for AcesAPI events of unknown tier


@dataclass
class AcesState:
    frontier: dict[str, str] = field(default_factory=dict)       # tour -> last date that must link
    editions: dict[str, int] = field(default_factory=dict)       # 'atp:188-2026' -> edition_id
    edition_source: dict[int, str] = field(default_factory=dict)  # edition_id -> 'acesapi' if ours
    seen: dict[str, dict] = field(default_factory=dict)          # match key -> first/last/results
    linked: int = 0
    inserted: int = 0
    players_created: int = 0


def tour_of(rf: RawFile) -> str:
    parts = rf.relpath.split("/")
    if len(parts) != 3 or parts[0] != "draws" or parts[1] not in ("atp", "wta"):
        raise BuildError(f"{rf.folder}/{rf.relpath}: expected draws/<atp|wta>/<tournament_id>.json")
    return parts[1]


def _pull_time(rf: RawFile) -> str:
    if not PULL_ID.match(rf.source_ref):
        raise BuildError(f"{rf.folder}: AcesAPI pull id {rf.source_ref!r} is not YYYYMMDDTHHMMSSZ")
    p = rf.source_ref
    return f"{p[:4]}-{p[4:6]}-{p[6:8]}T{p[9:11]}:{p[11:13]}:{p[13:15]}Z"


def plan(files: list[tuple[int, RawFile]]) -> dict[tuple[str, str], tuple[int, RawFile]]:
    """Latest pull per (tour, tournament file). Pull ids are UTC timestamps, so they sort."""
    latest: dict[tuple[str, str], tuple[int, RawFile]] = {}
    for sid, rf in files:
        key = (tour_of(rf), rf.relpath)
        if key not in latest or rf.source_ref > latest[key][1].source_ref:
            latest[key] = (sid, rf)
    return latest


def init_state(con: sqlite3.Connection) -> AcesState:
    st = AcesState()
    for tour, last in con.execute(
            """SELECT t.tour, MAX(e.start_date) FROM tournament_edition e
               JOIN tournament t USING (tournament_id) GROUP BY t.tour"""):
        st.frontier[tour] = add_days(last, -FRONTIER_MARGIN_DAYS)
    return st


def load_draw(con: sqlite3.Connection, snapshot_id: int, rf: RawFile, resolver: PlayerResolver,
              ov: Overrides, st: AcesState, latest: bool) -> LoadRun:
    tour = tour_of(rf)
    run = LoadRun(con, snapshot_id, rf.loader)
    doc = read_json(rf.path)
    rows = draw_matches(doc)
    pulled = _pull_time(rf)
    _note_seen(st, tour, rows, pulled)
    if not latest:
        for _ in rows:
            run.row()
        run.skipped("superseded_by_later_pull", rf.source_ref, n=len(rows))
        run.finish()
        return run

    tourn = doc.get("tournament") or {}
    tkey = f"{tour}:{tourn.get('tournament_id') or rf.relpath.rsplit('/', 1)[-1][:-5]}"
    start = edition_start(tourn, rows)
    main_rounds, qual_rounds = n_main_rounds(rows), n_qual_rounds(rows)
    # Main draw first: its links identify the backbone edition before qualifying rows,
    # which the backbone often lacks, look for a round to fill.
    rows = sorted(rows, key=lambda x: is_qualifying(*x))
    frontier = st.frontier.get(tour)
    past_frontier = frontier is None or (start is not None and start > frontier)
    context = _context(con, tour, start) if start else set()

    for rnd, m in rows:
        run.row()
        mkey = f"{tour}:{m.get('match_id')}"
        if not m.get("match_id"):
            run.skipped("no_match_id", rf.relpath); continue
        if str(m.get("status") or "").lower() != "final":
            run.skipped("not_final", f"{mkey}:{m.get('status')}"); continue
        if start is None:
            run.skipped("no_edition_date", tkey); continue
        q = is_qualifying(rnd, m)
        code = map_round(m.get("round") or rnd.get("round"), q, main_rounds, qual_rounds)
        if code is None:
            run.skipped("unmapped_round", f"{'Q:' if q else ''}{m.get('round') or rnd.get('round')}"); continue
        status = match_status(m.get("outcome"), m.get("score"))
        if status is None:
            run.skipped("unknown_outcome", f"{mkey}:{m.get('outcome')}"); continue
        ids = (str(m.get("player1_id") or ""), str(m.get("player2_id") or ""))
        names = ((m.get("player1") or "").strip(), (m.get("player2") or "").strip())
        win = str(m.get("winner_id") or "")
        if not all(ids) or not all(names) or win not in ids:
            run.skipped("missing_player_or_winner", mkey); continue
        if ids[0] == ids[1]:
            run.skipped("same_player_both_sides", mkey); continue
        day = match_day(m) or start
        countries = (m.get("player1_country"), m.get("player2_country"))
        pids = [resolver.resolve(SOURCE, f"{tour}:{ids[i]}", tour, names[i], [day], snapshot_id,
                                 context=context or None) for i in (0, 1)]
        if (pids[0] is None) != (pids[1] is None):          # opponent anchoring, as for MCP
            i = 0 if pids[0] is None else 1
            got = resolver.resolve_by_opponent(
                SOURCE, f"{tour}:{ids[i]}", names[i],
                opponents(con, pids[1 - i], code, start_from=add_days(start, -LINK_WINDOW_DAYS),
                          start_to=add_days(start, LINK_WINDOW_DAYS)), snapshot_id)
            pids[i] = got
        if past_frontier:
            for i in (0, 1):
                if pids[i] is None:
                    pids[i] = _maybe_create(con, resolver, st, tour, ids[i], names[i], countries[i],
                                            snapshot_id)
        detail = {"tournament": tourn.get("name"), "tkey": tkey, "start": start, "round": code,
                  "player1": names[0], "player2": names[1], "score": m.get("score")}
        if pids[0] is None or pids[1] is None:
            queue(con, "match", SOURCE, mkey, "player_unresolved", None, detail, snapshot_id)
            run.queued(); continue
        if pids[0] == pids[1]:
            run.skipped("same_player_both_sides", mkey); continue
        winner, loser = (pids[0], pids[1]) if win == ids[0] else (pids[1], pids[0])

        cands, conflict = _backbone_candidates(con, winner, loser, start, code)
        if conflict and not cands:
            queue(con, "match", SOURCE, mkey, "result_conflict", [c[0] for c in conflict],
                  detail, snapshot_id)
            run.queued(); continue
        ed = None
        if cands or not past_frontier:
            if not cands:
                ed = _round_gap_edition(con, st, tkey, code)
                if ed is None:
                    queue(con, "match", SOURCE, mkey, "no_backbone_match", None, detail, snapshot_id)
                    run.queued(); continue
        if cands:
            match_id, _ = link_match(con, ov, SOURCE, mkey, cands, code, snapshot_id, detail,
                                     src_date=start, src_tournament=tourn.get("name"))
            if match_id is None:
                run.queued(); continue
            _edition_xref(con, tkey, match_id, snapshot_id, st)
            st.seen[mkey]["match_id"] = match_id
            st.linked += 1
            run.linked(); continue

        if ed is None:
            ed = _edition(con, st, tour, tkey, tourn, start, main_rounds, snapshot_id)
        bo = 5 if (tour == "atp" and tourn.get("major") and not q) else 3
        cur = con.execute(
            """INSERT OR IGNORE INTO matches (edition_id, round, round_order, match_date,
                   date_precision, winner_id, loser_id, score, status, best_of, minutes,
                   snapshot_id) VALUES (?,?,?,?,?,?,?,?,?,?,NULL,?)""",
            (ed, code, ROUND_ORDER[code], match_day(m), "day" if match_day(m) else "week",
             winner, loser, (m.get("score") or "").strip() or None, status, bo, snapshot_id))
        if cur.rowcount == 0:
            queue(con, "match", SOURCE, mkey, "duplicate_key_conflict", None, detail, snapshot_id)
            run.queued(); continue
        match_id = cur.lastrowid
        con.execute(
            """INSERT INTO match_xref (source, source_key, match_id, method, confidence, verified,
                   snapshot_id) VALUES (?,?,?,'seed',1.0,1,?)""", (SOURCE, mkey, match_id, snapshot_id))
        st.seen[mkey]["match_id"] = match_id
        st.inserted += 1
        run.inserted()
    run.finish()
    return run


def _backbone_candidates(con, winner: int, loser: int, start: str, code: str):
    """Backbone matches this row could be: (same-winner candidates, other-winner candidates).

    Excludes rows AcesAPI itself inserted (its match ids are unique, so two AcesAPI rows are
    never one match) and candidates on the other side of the qualifying/main-draw line (a
    lucky loser can meet the same opponent in both). A candidate with the other winner is a
    result conflict between the sources, never a link.
    """
    cands = match_candidates_window(con, winner, loser, add_days(start, -LINK_WINDOW_DAYS),
                                    add_days(start, LINK_WINDOW_DAYS))
    q = code.startswith("Q")
    keep, conflict = [], []
    for c in cands:
        if c[1].startswith("Q") != q:
            continue
        w, src = con.execute(
            """SELECT m.winner_id, s.source FROM matches m JOIN ingest_snapshot s USING (snapshot_id)
               WHERE m.match_id=?""", (c[0],)).fetchone()
        if src == SOURCE:
            continue
        (keep if w == winner else conflict).append(c)
    return keep, conflict


def _round_gap_edition(con, st: AcesState, tkey: str, code: str) -> int | None:
    """The linked backbone edition, if it has no match at all in this round.

    Fills whole rounds the backbone never had (e.g. slam qualifying), never a partial round,
    so an AcesAPI row can't duplicate a backbone match that merely failed to link.
    """
    ed = st.editions.get(tkey)
    if ed is None or st.edition_source.get(ed) == SOURCE:
        return None
    if con.execute("SELECT 1 FROM matches WHERE edition_id=? AND round=? AND snapshot_id NOT IN "
                   "(SELECT snapshot_id FROM ingest_snapshot WHERE source=?) LIMIT 1",
                   (ed, code, SOURCE)).fetchone():
        return None
    return ed


def _note_seen(st: AcesState, tour: str, rows, pulled: str) -> None:
    """First/last pull in which each match was final, and whether its result changed."""
    for _, m in rows:
        if not m.get("match_id") or str(m.get("status") or "").lower() != "final":
            continue
        key = f"{tour}:{m.get('match_id')}"
        result = (str(m.get("winner_id") or ""), (m.get("score") or "").strip(),
                  str(m.get("outcome") or ""))
        s = st.seen.setdefault(key, {"first": pulled, "last": pulled, "results": set(), "n": 0})
        s["first"], s["last"] = min(s["first"], pulled), max(s["last"], pulled)
        s["results"].add(result)
        s["n"] += 1


def write_seen(con: sqlite3.Connection, st: AcesState) -> None:
    con.executemany(
        """INSERT INTO match_seen (match_id, source, source_key, first_seen_at, last_seen_at,
               n_pulls, revised) VALUES (?,?,?,?,?,?,?)""",
        [(s["match_id"], SOURCE, k, s["first"], s["last"], s["n"], int(len(s["results"]) > 1))
         for k, s in sorted(st.seen.items()) if "match_id" in s])


def _context(con, tour: str, start: str) -> set[int]:
    """Players in backbone editions of this tour starting within the link window."""
    return {pid for row in con.execute(
        """SELECT m.winner_id, m.loser_id FROM matches m JOIN tournament_edition e USING (edition_id)
           JOIN tournament t USING (tournament_id)
           WHERE t.tour=? AND e.start_date BETWEEN ? AND ?""",
        (tour, add_days(start, -LINK_WINDOW_DAYS), add_days(start, LINK_WINDOW_DAYS))) for pid in row}


def _maybe_create(con, resolver: PlayerResolver, st: AcesState, tour: str, aid: str, name: str,
                  country: str | None, snapshot_id: int) -> int | None:
    key = f"{tour}:{aid}"
    if resolver.ov.link_for("player", SOURCE, key) is not None:
        return None
    q = con.execute("""SELECT reason, candidate_ids FROM review_queue
                       WHERE entity_type='player' AND source=? AND source_key=? AND status='open'""",
                    (SOURCE, key)).fetchone()
    if q is None:
        return None
    reason, cands = q[0], json.loads(q[1] or "[]")
    if reason == "low_confidence":
        country = (country or "").strip().upper()
        theirs = [con.execute("SELECT country FROM player WHERE player_id=?", (p,)).fetchone()[0]
                  for p in cands]
        if not country or any(not c or c.upper() == country for c in theirs):
            return None
    elif reason != "no_candidate":
        return None
    parts = name.split()
    first, last = (" ".join(parts[:-1]) or None, parts[-1]) if parts else (None, None)
    pid = con.execute(
        """INSERT INTO player (tour, canonical_name, name_first, name_last, country)
           VALUES (?,?,?,?,?)""", (tour, name, first, last, (country or "").strip() or None)).lastrowid
    con.execute(
        """INSERT INTO player_xref (source, source_key, player_id, method, confidence, verified,
               snapshot_id) VALUES (?,?,?,'seed',1.0,0,?)""", (SOURCE, key, pid, snapshot_id))
    alias = norm_name(name)
    if alias:
        con.execute("INSERT OR IGNORE INTO player_alias VALUES (?,?,?)", (alias, pid, SOURCE))
        resolver.alias.setdefault((tour, alias), []).append(pid)
    resolver.parts[pid] = (norm_name(first), norm_name(last))
    resolver.cache[(SOURCE, key)] = pid
    con.execute("""UPDATE review_queue SET status='resolved',
                       resolved_at=strftime('%Y-%m-%dT%H:%M:%SZ','now')
                   WHERE entity_type='player' AND source=? AND source_key=?""", (SOURCE, key))
    st.players_created += 1
    resolver._note("new_player")
    return pid


def _edition_xref(con, tkey: str, match_id: int, snapshot_id: int, st: AcesState) -> None:
    ed = con.execute("SELECT edition_id FROM matches WHERE match_id=?", (match_id,)).fetchone()[0]
    con.execute("""INSERT OR IGNORE INTO edition_xref (source, source_key, edition_id, method,
                       confidence, verified, snapshot_id) VALUES (?,?,?,'context',0.95,0,?)""",
                (SOURCE, tkey, ed, snapshot_id))
    st.editions.setdefault(tkey, ed)


def _edition_xref_id(con, tkey: str, edition_id: int, snapshot_id: int) -> None:
    con.execute("""INSERT OR IGNORE INTO edition_xref (source, source_key, edition_id, method,
                       confidence, verified, snapshot_id) VALUES (?,?,?,'context',0.95,0,?)""",
                (SOURCE, tkey, edition_id, snapshot_id))


def _tournament(con, tour: str, tourn: dict, start: str) -> tuple[int, str | None]:
    """Existing tournament (and its last level) for this event, else a new one.

    Same event = a tournament with an edition about one year earlier (+-21 days) whose
    aliases match, tried in order: an alias equal to the name, then to the city, then
    distinctive-token overlap ("Rolex Shanghai Masters" ~ "Shanghai Masters", city
    "Cincinnati" ~ "Cincinnati Masters"). At each step exactly one tournament must qualify.
    """
    prev_lo, prev_hi = add_days(start, -365 - SAME_EVENT_DAYS), add_days(start, -365 + SAME_EVENT_DAYS)
    prior = con.execute(
        """SELECT a.alias_normalized, a.tournament_id, e.level, e.start_date FROM tournament_alias a
           JOIN tournament t USING (tournament_id) JOIN tournament_edition e USING (tournament_id)
           WHERE t.tour=? AND e.start_date BETWEEN ? AND ? ORDER BY e.start_date""",
        (tour, prev_lo, prev_hi)).fetchall()
    name, city = norm_name(tourn.get("name")), norm_name(tourn.get("city"))
    mine = _tokens(name) | _tokens(city)
    tests = [lambda al: bool(name) and al == name,
             lambda al: bool(city) and al == city,
             lambda al: bool(mine & _tokens(al))]
    for test in tests:
        hits = {tid: level for al, tid, level, _ in prior if test(al)}
        if len(hits) == 1:
            tid, level = next(iter(hits.items()))
            if name:
                con.execute("INSERT OR IGNORE INTO tournament_alias VALUES (?,?)", (name, tid))
            return tid, level
    name = (tourn.get("name") or "").strip() or "AcesAPI " + str(tourn.get("tournament_id"))
    if con.execute("SELECT 1 FROM tournament WHERE tour=? AND canonical_name=?", (tour, name)).fetchone():
        name = f"{name} [{tourn.get('tournament_id')}]"
    tid = con.execute("INSERT INTO tournament (tour, canonical_name, city, country) VALUES (?,?,?,?)",
                      (tour, name, tourn.get("city"), tourn.get("country"))).lastrowid
    con.execute("INSERT OR IGNORE INTO tournament_alias VALUES (?,?)", (norm_name(name), tid))
    return tid, None


_GENERIC = {"open", "masters", "championships", "championship", "tennis", "international",
            "internationals", "ladies", "women", "womens", "cup", "classic", "grand", "prix",
            "trophy", "presented", "powered", "by", "the", "de", "di", "del", "la", "le", "of",
            "and", "par", "por", "city", "atp", "wta", "tour", "finals", "final", "series"}


def _tokens(alias: str | None) -> set[str]:
    """Distinctive tokens: 4+ letters, not a generic tournament word, no digits."""
    return {t for t in (alias or "").split() if len(t) >= 4 and t not in _GENERIC and t.isalpha()}


def _edition(con, st: AcesState, tour: str, tkey: str, tourn: dict, start: str,
             main_rounds: int | None, snapshot_id: int) -> int:
    if tkey in st.editions:
        return st.editions[tkey]
    tid, prev_level = _tournament(con, tour, tourn, start)
    have = con.execute("SELECT edition_id FROM tournament_edition WHERE tournament_id=? AND start_date=?",
                       (tid, start)).fetchone()
    if have:     # the backbone already has this event (its last weeks overlap the frontier)
        _edition_xref_id(con, tkey, have[0], snapshot_id)
        st.editions[tkey] = have[0]
        return have[0]
    level = prev_level or ("G" if tourn.get("major") else UNKNOWN_TIER)
    year = int(tourn.get("season") or start[:4])
    ed = con.execute(
        """INSERT INTO tournament_edition (tournament_id, year, start_date, end_date, level,
               surface, indoor, draw_size, best_of) VALUES (?,?,?,?,?,?,?,?,?)""",
        (tid, year, start, tourn.get("end_date"), level, norm_surface(tourn.get("surface")),
         None if tourn.get("indoor") is None else int(bool(tourn.get("indoor"))),
         2 ** main_rounds if main_rounds else None,
         5 if (tour == "atp" and tourn.get("major")) else 3)).lastrowid
    con.execute("""INSERT INTO edition_xref (source, source_key, edition_id, method, confidence,
                       verified, snapshot_id) VALUES (?,?,?,'seed',1.0,1,?)""",
                (SOURCE, tkey, ed, snapshot_id))
    st.editions[tkey] = ed
    st.edition_source[ed] = SOURCE
    return ed
