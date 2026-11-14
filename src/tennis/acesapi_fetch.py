"""AcesAPI client and pull planning (used by scripts/fetch_acesapi.py). Standard library only.

A *pull* is one run of the fetcher. It writes an immutable folder
data/raw/acesapi-<YYYYMMDDTHHMMSSZ>/ (UTC, the pull id) containing:
  tournaments/<tour>-<season>-<offset>.json   list_tournaments pages (1 credit each)
  draws/<tour>/<tournament_id>.json           get_tournament draw=all (2 credits each)
  pull.json                                   every request, its HTTP status, credits left
Responses are saved byte-for-byte. Nothing is ever rewritten: the next pull is a new folder,
and the build loads the latest pull of each tournament.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from .acesapi import FOLDER_PREFIX, LEAGUE, PULL_ID, draw_matches, iso_day, read_json, unwrap

BASE_URL = "https://mcp.acesapi.com/v1"
COST = {"list_tournaments": 1, "get_tournament": 2}
MIN_INTERVAL_S = 1.1          # docs: 60 requests/minute (10/minute under 100 credits)


class ApiError(RuntimeError):
    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body[:300]}")
        self.status = status


class Client:
    def __init__(self, key: str, base_url: str = BASE_URL):
        self.key = key
        self.base = base_url.rstrip("/")
        self.credits_remaining: int | None = None
        self._last = 0.0

    def call(self, tool: str, body: dict) -> bytes:
        """POST /v1/<tool>; returns the raw response bytes. Retries once on 429/5xx."""
        for attempt in (1, 2):
            wait = MIN_INTERVAL_S - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            req = urllib.request.Request(
                f"{self.base}/{tool}", data=json.dumps(body).encode("utf-8"), method="POST",
                headers={"Content-Type": "application/json", "x-api-key": self.key,
                         "User-Agent": "tennis-results-fetch"})
            self._last = time.monotonic()
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    raw = resp.read()
                    left = resp.headers.get("X-Credits-Remaining")
                    if left is not None and left.strip().lstrip("-").isdigit():
                        self.credits_remaining = int(left)
                    return raw
            except urllib.error.HTTPError as e:
                text = e.read().decode("utf-8", "replace")
                if attempt == 1 and (e.code == 429 or e.code >= 500):
                    time.sleep(30 if e.code == 429 else 5)
                    continue
                raise ApiError(e.code, text) from e
        raise AssertionError("unreachable")


def new_pull_id(now: datetime | None = None) -> str:
    return (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")


@dataclass
class Seen:
    pull_id: str
    status: str
    end_date: str | None


def previous_pulls(raw_root: Path) -> dict[tuple[str, str], Seen]:
    """Latest saved draw per (tour, tournament_id) across all pull folders on disk."""
    out: dict[tuple[str, str], Seen] = {}
    for folder in sorted(raw_root.glob(FOLDER_PREFIX + "*")):
        pid = folder.name[len(FOLDER_PREFIX):]
        if not PULL_ID.match(pid):
            continue
        for f in folder.glob("draws/*/*.json"):
            tour, tid = f.parent.name, f.stem
            try:
                t = read_json(f).get("tournament") or {}
            except (ValueError, OSError):
                continue
            key = (tour, tid)
            if key not in out or pid > out[key].pull_id:
                out[key] = Seen(pid, str(t.get("status") or ""), iso_day(t.get("end_date")))
    return out


def pull_day(pull_id: str) -> date:
    return date(int(pull_id[:4]), int(pull_id[4:6]), int(pull_id[6:8]))


def wanted(t: dict, prev: Seen | None, since: date, today: date, refetch_days: int) -> str | None:
    """Why tournament `t` (a list_tournaments item) needs a draw call now, or None.

    Fetch a tournament that has started and overlaps [since, today] unless an earlier pull
    was made more than `refetch_days` after it ended (AcesAPI rewrites the current season
    daily, so late corrections land in that window). The status is not trusted for this:
    some events stay "live" long after their last match.
    """
    status = str(t.get("status") or "").lower()
    start, end = iso_day(t.get("start_date")), iso_day(t.get("end_date"))
    if status not in ("final", "live") or start is None or date.fromisoformat(start) > today:
        return None
    if end is not None and date.fromisoformat(end) < since:
        return None
    if prev is None:
        return "new"
    # Settled once an earlier pull was made more than refetch_days after the end, whatever
    # the status says: AcesAPI leaves some events "live" for months after their last match.
    end_d = date.fromisoformat(prev.end_date or end or start)
    if pull_day(prev.pull_id) > end_d + timedelta(days=refetch_days):
        return None
    return "correction_window" if prev.status.lower() == "final" else "was_live"


def list_items(raw: bytes) -> tuple[list[dict], int | None]:
    doc = unwrap(json.loads(raw.decode("utf-8")))
    return list(doc.get("tournaments") or []), doc.get("next_offset")


def inspect(folder: Path) -> dict:
    """Value counts of the fields the loader maps, from saved draws (no API calls)."""
    from collections import Counter
    c = {k: Counter() for k in ("draw", "round", "status", "outcome", "envelope", "match_keys",
                                "sets_type", "tournament_status")}
    for f in sorted(folder.glob("draws/*/*.json")):
        top = json.loads(f.read_text(encoding="utf-8"))
        c["envelope"]["data" if isinstance(top, dict) and "data" in top else "bare"] += 1
        doc = unwrap(top)
        c["tournament_status"][str((doc.get("tournament") or {}).get("status"))] += 1
        for rnd, m in draw_matches(doc):
            c["draw"][str(m.get("draw") or rnd.get("draw"))] += 1
            c["round"][f"{m.get('draw') or rnd.get('draw')}|{m.get('round') or rnd.get('round')}"
                       f"|{m.get('round_order') or rnd.get('round_order')}"] += 1
            c["status"][str(m.get("status"))] += 1
            c["outcome"][str(m.get("outcome"))] += 1
            c["match_keys"][",".join(sorted(m))] += 1
            c["sets_type"][type(m.get("sets")).__name__] += 1
    return {k: dict(v.most_common()) for k, v in c.items()}


def manifest_entry(pull_id: str, folder: Path) -> dict:
    groups = []
    if any(folder.glob("draws/*/*.json")):
        groups.append({"name": "acesapi_draws", "source": "acesapi", "loader": "acesapi_draws",
                       "files": ["draws/*/*.json"]})
    ignore = [{"pattern": "pull.json", "reason": "pull metadata: requests, HTTP status, credits"}]
    if any(folder.glob("tournaments/*.json")):
        ignore.append({"pattern": "tournaments/*.json",
                       "reason": "list_tournaments pages: used to plan draw calls; draws carry the tournament"})
    return {"folder": folder.name, "repo": "acesapi", "commit": pull_id,
            "fetched_at": f"{pull_id[:4]}-{pull_id[4:6]}-{pull_id[6:8]}T{pull_id[9:11]}:"
                          f"{pull_id[11:13]}:{pull_id[13:15]}Z",
            "groups": groups, "ignore": ignore}


def pin(manifest_path: Path, entry: dict) -> bool:
    """Append a snapshot entry to the manifest, keeping the file's existing layout.

    Returns False if the folder is already pinned.
    """
    text = manifest_path.read_text(encoding="utf-8")
    m = json.loads(text)
    if any(s["folder"] == entry["folder"] for s in m["snapshots"]):
        return False
    if "acesapi" not in m["sources"]:
        raise ValueError(f"{manifest_path}: declare the 'acesapi' source before pinning a pull")
    # The file ends with the snapshots array: "...}\n  ]\n}". Insert before its "]".
    cut = text.rstrip().rstrip("}").rstrip().rstrip("]").rstrip()
    block = json.dumps(entry, ensure_ascii=False, indent=2)
    block = "\n".join("    " + line for line in block.splitlines())
    new = f"{cut},\n{block}\n  ]\n}}\n"
    try:
        check = json.loads(new)
    except ValueError:
        check = None
    if not check or check["snapshots"][:-1] != m["snapshots"] or check["snapshots"][-1] != entry \
            or {k: v for k, v in check.items() if k != "snapshots"} != \
               {k: v for k, v in m.items() if k != "snapshots"}:
        raise ValueError(f"{manifest_path}: unexpected layout; add the snapshot by hand:\n{block}")
    manifest_path.write_text(new, encoding="utf-8", newline="\n")
    return True
