"""Kalshi results as a raw source for tennis.db: export from live.db, and parsing.

Nothing here touches tennis.db. The export writes immutable raw folders that a future
loader can pin in data/build_manifest.json; until then they sit outside the manifest
(the build only checks pinned folders), so tennis.db and the A5 state snapshots
(engine/state.py) stay valid. See docs/DESIGN.md "Kalshi results".

Export (`scripts/kalshi_export.py`): every settled match in a live.db that no earlier
export holds becomes one file

    data/raw/kalshi_results-<YYYYMMDDTHHMMSSZ>/matches/<series>/<event>.json

    {"format": "kalshi_result/1",
     "event":       the live.db event row (tickers, names, competitor ids, start, first/last seen),
     "settlement":  the live.db settlement row (winner, market results, seen_at, ...),
     "markets":     GET /events/<event> at settlement,
     "live":        the final live details (set scores, match_status, serve statistics),
     "live_from":   "settlement" | "poll <id> <as_of>" (when the settlement had none, or not closed),
     "milestone":   the event's milestone (tournament_name, round, best_of, start_date, Sportradar id),
     "milestone_from": "settlement" | "poll <id> <as_of>" | null}

plus `export.json` (when, from which live.db, which watches and code revisions, counts).
Folders are written to a temporary name and renamed when complete; an existing folder is
never written to. Settlements in live.db are insert-once, so a match is exported once.

Parsing (`parse`): one file -> a flat result in tennis.db terms (tour, level, names,
competitor ids, winner, status, score winner first, best-of, round and tournament as
Kalshi names them, and both players' serve stats in match_player_stats columns), or a skip
reason. This is what the loader will read; it is tested now so the recorded format is
known to carry everything.

Stable keys: the event ticker for the match, Kalshi's competitor id (UUID, the same
across events) for players.
"""
from __future__ import annotations

import json
import os
import sqlite3
import zlib
from datetime import datetime, timezone
from pathlib import Path

FORMAT = "kalshi_result/1"
FOLDER_PREFIX = "kalshi_results-"

# Kalshi series -> (tour, level). Level 'tour' is ATP/WTA tour level (incl. slams and
# finals); 'challenger' is ATP Challenger / WTA 125 (Kalshi's KXWTACHALLENGERMATCH).
SERIES = {
    "KXATPMATCH": ("atp", "tour"),
    "KXWTAMATCH": ("wta", "tour"),
    "KXATPCHALLENGERMATCH": ("atp", "challenger"),
    "KXWTACHALLENGERMATCH": ("wta", "challenger"),
}

# Kalshi live `match_status` at the end -> matches.status. Only 'ended' was seen up to
# 2026-11-16; the others are as documented by Sportradar and are unverified.
MATCH_STATUS = {"ended": "completed", "retired": "retired", "walkover": "walkover",
                "defaulted": "default", "interrupted": "unfinished", "abandoned": "unfinished"}


# ------------------------------------------------------------------------------- export

def _blob(con: sqlite3.Connection, sha: str | None):
    if sha is None:
        return None
    r = con.execute("SELECT zlib FROM blob WHERE sha256 = ?", (sha,)).fetchone()
    return None if r is None else json.loads(zlib.decompress(r[0]).decode("utf-8"))


def exported_events(raw_root: Path) -> set[str]:
    """Event tickers already in any export folder under raw_root."""
    out = set()
    for d in Path(raw_root).glob(FOLDER_PREFIX + "*"):
        if d.is_dir() and not d.name.endswith(".partial"):
            out.update(p.stem for p in (d / "matches").glob("*/*.json"))
    return out


class _PollScan:
    """Newest-first fallback lookups in the poll blobs, each distinct blob read once."""

    def __init__(self, con: sqlite3.Connection):
        self.con = con
        self.polls = con.execute(
            """SELECT poll_id, as_of, milestones_sha, live_sha FROM poll
               WHERE error IS NULL ORDER BY poll_id DESC""").fetchall()
        self.cache: dict[str, object] = {}

    def _get(self, sha):
        if sha not in self.cache:
            self.cache[sha] = _blob(self.con, sha) or {}
        return self.cache[sha]

    def milestone(self, event: str):
        for pid, as_of, msha, _ in self.polls:
            m = self._get(msha).get(event) if msha else None
            if m:
                return m, f"poll {pid} {as_of}"
        return None, None

    def live(self, mid: str):
        """The newest closed details of milestone `mid`, else the newest of any status."""
        last = (None, None)
        for pid, as_of, _, lsha in self.polls:
            d = self._get(lsha).get(mid) if lsha else None
            if d:
                if d.get("status") == "closed":
                    return d, f"poll {pid} {as_of}"
                if last[0] is None:
                    last = (d, f"poll {pid} {as_of}")
        return last


def export_docs(con: sqlite3.Connection, skip: set[str] = frozenset()) -> list[dict]:
    """One kalshi_result/1 document per settled event not in `skip`, oldest settlement first."""
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """SELECT e.*, s.seen_at, s.winner, s.result_a, s.result_b, s.value_a, s.value_b,
                  s.market_status, s.match_status, s.final_score, s.raw_sha
           FROM settlement s JOIN event e USING (event) ORDER BY s.seen_at, e.event""").fetchall()
    scan = None
    docs = []
    ev_cols = [c[1] for c in con.execute("PRAGMA table_info(event)")]
    st_cols = [c[1] for c in con.execute("PRAGMA table_info(settlement)") if c[1] not in ("event", "raw_sha")]
    for r in rows:
        if r["event"] in skip:
            continue
        raw = _blob(con, r["raw_sha"]) or {}
        live, live_from = raw.get("live"), "settlement"
        ms, ms_from = raw.get("milestone"), "settlement"
        if not ms:
            scan = scan or _PollScan(con)
            ms, ms_from = scan.milestone(r["event"])
        mid = (ms or {}).get("id") or r["milestone"]
        if (not live or live.get("status") != "closed") and mid:
            scan = scan or _PollScan(con)
            alt, alt_from = scan.live(mid)
            if alt and (not live or alt.get("status") == "closed"):
                live, live_from = alt, alt_from
        docs.append({"format": FORMAT,
                     "event": {c: r[c] for c in ev_cols},
                     "settlement": {c: r[c] for c in st_cols},
                     "markets": raw.get("event"),
                     "live": live, "live_from": live_from if live else None,
                     "milestone": ms, "milestone_from": ms_from if ms else None})
    return docs


def write_export(docs: list[dict], raw_root: Path, meta: dict, stamp: str | None = None) -> Path | None:
    """Write docs to a new raw_root/kalshi_results-<stamp>/ (None when there is nothing new)."""
    if not docs:
        return None
    stamp = stamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    final = Path(raw_root) / f"{FOLDER_PREFIX}{stamp}"
    tmp = Path(str(final) + ".partial")
    if final.exists() or tmp.exists():
        raise FileExistsError(f"{final} (or its .partial) already exists; raw folders are never rewritten")
    for doc in docs:
        p = tmp / "matches" / doc["event"]["series"] / f"{doc['event']['event']}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    meta = {**meta, "format": FORMAT, "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "matches": len(docs),
            "by_series": {s: sum(d["event"]["series"] == s for d in docs) for s in sorted({d["event"]["series"] for d in docs})},
            "settled_from": min(d["settlement"]["seen_at"] for d in docs),
            "settled_to": max(d["settlement"]["seen_at"] for d in docs)}
    (tmp / "export.json").write_text(json.dumps(meta, indent=1), encoding="utf-8", newline="\n")
    os.replace(tmp, final)
    return final


def live_db_meta(con: sqlite3.Connection, path: str) -> dict:
    return {"live_db": path,
            "watches": [dict(zip(("watch_id", "started_at", "predictor", "code_rev"), r)) for r in con.execute(
                "SELECT watch_id, started_at, predictor, code_rev FROM watch ORDER BY watch_id")]}


# ------------------------------------------------------------------------------- parsing

def read(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    if doc.get("format") != FORMAT:
        raise ValueError(f"{path}: format {doc.get('format')!r}, expected {FORMAT!r}")
    return doc


def _int(x) -> int:
    return int(x or 0)


def _stats(me: dict, opp: dict) -> dict | None:
    """Kalshi competitor statistics -> match_player_stats columns, or None when absent or
    when the points the model uses (serve points, first_won + second_won = service points
    won) do not add up. `exact` is False when the serve breakdown is off: first_in +
    second serves in + double faults != serve points (3 of the first 24 settled matches,
    2026-11-15/16, by 1-2 points; Sportradar's counts, not ours). Service games = own holds
    + the opponent's breaks; bp_faced = the opponent's break chances."""
    if not me or not opp or "service_points_won" not in me:
        return None
    sp = _int(me.get("service_points_won")) + _int(me.get("service_points_lost"))
    out = {"aces": _int(me.get("aces")), "double_faults": _int(me.get("double_faults")),
           "serve_points": sp, "first_in": _int(me.get("first_serve_successful")),
           "first_won": _int(me.get("first_serve_points_won")),
           "second_won": _int(me.get("second_serve_points_won")),
           "serve_games": _int(me.get("service_games_won")) + _int(opp.get("breakpoints_won")),
           "bp_faced": _int(opp.get("total_breakpoints"))}
    out["bp_saved"] = out["bp_faced"] - _int(opp.get("breakpoints_won"))
    if sp <= 0 or out["first_won"] + out["second_won"] != _int(me.get("service_points_won")) or out["bp_saved"] < 0:
        return None
    out["exact"] = out["first_in"] + _int(me.get("second_serve_successful")) + out["double_faults"] == sp
    return out


def parse(doc: dict) -> tuple[dict | None, str]:
    """(result, '') or (None, skip reason). Sides: 'w' winner, 'l' loser."""
    ev, st, live, ms = doc["event"], doc["settlement"], doc.get("live") or {}, doc.get("milestone") or {}
    if ev["series"] not in SERIES:
        return None, "unknown_series"
    tour, level = SERIES[ev["series"]]
    if st.get("winner") not in ("a", "b"):
        return None, "no_winner"                      # void / fair-price settlement (no ball played)
    status = MATCH_STATUS.get(live.get("match_status") or "")
    if status is None:
        return None, f"match_status:{live.get('match_status') or 'missing'}"
    side = {live.get("competitor1_id"): "1", live.get("competitor2_id"): "2"}
    if ev["a_comp"] not in side or ev["b_comp"] not in side or ev["a_comp"] == ev["b_comp"]:
        return None, "live_competitors_mismatch"
    k = {"a": side[ev["a_comp"]], "b": side[ev["b_comp"]]}
    w, l = st["winner"], "b" if st["winner"] == "a" else "a"
    if live.get("winner") and live["winner"] != ev[f"{w}_comp"]:
        return None, "winner_disagrees_with_live"
    sw = live.get(f"competitor{k[w]}_round_scores") or []
    sl = live.get(f"competitor{k[l]}_round_scores") or []
    if len(sw) != len(sl):
        return None, "set_lists_differ"
    score = " ".join(f"{_int(x.get('score'))}-{_int(y.get('score'))}" for x, y in zip(sw, sl))
    if status == "retired":
        score = (score + " RET").strip()
    elif status == "walkover":
        score = "W/O"
    det = ms.get("details") or {}
    bo = det.get("best_of")
    start = ms.get("start_date") or ev.get("start")
    stw = live.get(f"competitor{k[w]}_statistics") or {}
    stl = live.get(f"competitor{k[l]}_statistics") or {}
    return {
        "source_key": ev["event"], "series": ev["series"], "tour": tour, "level": level,
        "tournament": det.get("tournament_name") or ev.get("tournament"), "round": det.get("round"),
        "start": start, "day_utc": start[:10] if start else None,
        "best_of": int(bo) if bo and str(bo).isdigit() else None,
        "winner_name": ev[f"{w}_name"], "loser_name": ev[f"{l}_name"],
        "winner_comp": ev[f"{w}_comp"], "loser_comp": ev[f"{l}_comp"],
        "winner_seed": _int(live.get(f"competitor{k[w]}_seed")) or None,
        "loser_seed": _int(live.get(f"competitor{k[l]}_seed")) or None,
        "status": status, "score": score,
        "winner_stats": _stats(stw, stl), "loser_stats": _stats(stl, stw),
        "sportradar_id": ms.get("source_id"),
        "known_at": st["seen_at"],                    # when this project first had the result
    }, ""
