"""AcesAPI response parsing shared by the fetcher and the loader. No database access here.

Shapes from the AcesAPI docs, checked against live responses on 2026-11-14
(list_tournaments ATP 2026; get_tournament 172-2026, Roland Garros, all 237 rows):
- REST: POST https://mcp.acesapi.com/v1/<tool>, JSON body, header x-api-key.
- get_tournament -> {"tour", "tournament": {tournament_id "188-2025", name, season,
  start_date, end_date, city, country, major, surface, indoor, status}, "champion",
  "runner_up", "rounds": [{"round": "Round 1", "round_order": 11, "draw": "main",
  "matches": [...]}], "count"}.
- match rows: match_id, tournament_id, season, surface, indoor, starts_at, match_date,
  draw, round, round_order, player1_id, player1, player1_country, seed1, player2_id, ...,
  winner_id, status (scheduled|live|final), outcome (completed|retired|...), score, sets[].
- Live responses are wrapped in {"data": ...} (docs examples are not): unwrap() accepts both.
- Round labels seen: "Qualifying 1st Round", "Qualifying 2nd Round", "Qualifying Final"
  (draw "qualifying"); "Round 1".."Round 4", "Quarterfinal", "Semifinal", "Final" (draw "main").
- outcome values seen: completed, retired (score "6-3 4-1 ret."), walkover (score "w/o").
  match_date is the local match day; starts_at is UTC.
Unverified: WTA labels, round-robin events, the "incomplete" marker, whether ids are shared
between ATP and WTA (every key is scoped by tour, so it does not matter).
Anything unknown is skipped with a named reason, never guessed. Run
`python scripts/fetch_acesapi.py --inspect <folder>` on the first real pull.
"""
from __future__ import annotations

import json
import math
import re
from datetime import date
from pathlib import Path

from .normalize import ROUND_ORDER, norm_round, score_status

SOURCE = "acesapi"
LEAGUE = {"atp": "tennis", "wta": "wta"}           # the API's "league" parameter per tour
PULL_ID = re.compile(r"^\d{8}T\d{6}Z$")            # pull folder id: UTC timestamp, sortable
FOLDER_PREFIX = "acesapi-"


def unwrap(obj):
    """Strip the documented {"data": ...} envelope if present."""
    if isinstance(obj, dict) and isinstance(obj.get("data"), dict) and len(obj) <= 3:
        return obj["data"]
    return obj


def read_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return unwrap(json.load(f))


def draw_matches(doc: dict) -> list[tuple[dict, dict]]:
    """(round entry, match) for every match row in a get_tournament response, in file order.

    Accepts either the documented "rounds" nesting or a flat "matches" list.
    """
    out = []
    if isinstance(doc.get("rounds"), list):
        for rnd in doc["rounds"]:
            for m in rnd.get("matches") or []:
                out.append((rnd, m))
    elif isinstance(doc.get("matches"), list):
        out = [({}, m) for m in doc["matches"]]
    else:
        raise ValueError("get_tournament response has neither 'rounds' nor 'matches'")
    return out


def count_draw_rows(path: Path) -> int:
    """Row count for ingest_snapshot: match rows in one saved get_tournament response."""
    return len(draw_matches(read_json(path)))


# ---------------------------------------------------------------- rounds

_ORD = {"1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "5th": 5, "first": 1, "second": 2,
        "third": 3, "fourth": 4, "fifth": 5}


def _round_number(label: str) -> int | None:
    m = re.search(r"\b(\d)\b", label) or re.search(r"\b(\d)(?:st|nd|rd|th)\b", label)
    if m:
        return int(m.group(1))
    for word, n in _ORD.items():
        if re.search(rf"\b{word}\b", label):
            return n
    return None


def is_qualifying(rnd: dict, match: dict) -> bool:
    draw = str(match.get("draw") or rnd.get("draw") or "").lower()
    label = str(match.get("round") or rnd.get("round") or "").lower()
    return draw.startswith("qual") or "qual" in label


def n_main_rounds(rows: list[tuple[dict, dict]]) -> int | None:
    """Rounds in the main draw (7 for a 128 draw), from 'Round k' labels and the match count.

    'Round k' is the k-th round of the main draw; QF/SF/F follow it. A tournament fetched
    mid-event may miss late rounds, so take the larger of the two estimates.
    """
    main = [(r, m) for r, m in rows if not is_qualifying(r, m)]
    if not main:
        return None
    k_max = 0
    for r, m in main:
        label = str(m.get("round") or r.get("round") or "").lower()
        if re.search(r"quarter|semi|final|robin|group", label) or re.search(r"round of \d+", label):
            continue
        k = _round_number(label)
        if k:
            k_max = max(k_max, k)
    by_label = k_max + 3 if k_max else 0
    by_count = math.ceil(math.log2(len(main) + 1))
    return max(by_label, by_count)


def n_qual_rounds(rows: list[tuple[dict, dict]]) -> int:
    """Distinct qualifying rounds in the draw ('Qualifying Final' is the last of them)."""
    return len({str(m.get("round") or r.get("round") or "").lower()
                for r, m in rows if is_qualifying(r, m)})


def map_round(label: str | None, qualifying: bool, main_rounds: int | None,
              qual_rounds: int | None = None) -> str | None:
    """AcesAPI round label -> our round code (schema CHECK list), or None if unknown."""
    raw = (label or "").strip()
    direct = norm_round(raw)
    if direct:
        return direct
    s = raw.lower()
    if qualifying:
        if re.search(r"final", s) and not re.search(r"\d|first|second|third", s):
            # 'Qualifying Final' = the last qualifying round (Q3 at a slam, Q2 at most events)
            code = f"Q{qual_rounds}" if qual_rounds else None
            return code if code in ROUND_ORDER else None
        k = _round_number(s)
        return f"Q{k}" if k and f"Q{k}" in ROUND_ORDER else None
    if "robin" in s or "group" in s:
        return "RR"
    if "quarter" in s:
        return "QF"
    if "semi" in s:
        return "SF"
    if re.fullmatch(r"(the )?finals?", s):
        return "F"
    m = re.search(r"round of (\d+)", s)
    if m:
        code = f"R{m.group(1)}"
        return code if code in ROUND_ORDER else None
    k = _round_number(s)
    if k and main_rounds:
        code = f"R{2 ** (main_rounds - k + 1)}"
        return code if code in ROUND_ORDER else None
    return None


# ---------------------------------------------------------------- outcomes, dates

_OUTCOME = {"completed": "completed", "complete": "completed", "finished": "completed",
            "retired": "retired", "retirement": "retired", "ret": "retired",
            "walkover": "walkover", "w/o": "walkover", "wo": "walkover",
            "default": "default", "defaulted": "default", "disqualified": "default",
            "incomplete": "unfinished", "abandoned": "unfinished", "unfinished": "unfinished"}


def match_status(outcome: str | None, score: str | None) -> str | None:
    """Our matches.status from AcesAPI 'outcome' (+ score text). None = unknown value."""
    o = (outcome or "").strip().lower()
    if o:
        return _OUTCOME.get(o)
    st = score_status(score)
    return None if st == "bye" else st


def iso_day(s) -> str | None:
    """'2026-07-12' or '2026-07-12T15:05:00Z' -> '2026-07-12'; anything else -> None."""
    s = str(s or "").strip()[:10]
    try:
        return date.fromisoformat(s).isoformat()
    except ValueError:
        return None


def match_day(m: dict) -> str | None:
    return iso_day(m.get("match_date")) or iso_day(m.get("starts_at"))


def edition_start(tournament: dict, rows: list[tuple[dict, dict]]) -> str | None:
    """Edition start the way Sackmann dates it: the first main-draw day.

    AcesAPI's start_date includes qualifying (Wimbledon 2025: 06-23, main draw 06-30),
    so prefer the earliest main-draw match day and fall back to start_date.
    """
    days = [d for r, m in rows if not is_qualifying(r, m) for d in [match_day(m)] if d]
    if days:
        return min(days)
    return iso_day(tournament.get("start_date"))


def pull_folder(pull_id: str) -> str:
    return f"{FOLDER_PREFIX}{pull_id}"
