"""Daily results feed for the light layer (skeleton).

Pipeline:
  1. Run a scraper actor on Apify (or replay a saved response) -> list of match items.
  2. Save the raw response as an immutable snapshot (data/raw/daily/<source>/) and
     record it in ingest_snapshot.
  3. Normalize each item, resolve both players to player_id, and write staged records
     to data/staging/<source>-<snapshot_id>.jsonl.
  Not done yet (next milestone): resolving the edition and inserting rows into
  `matches` / `match_player_stats` from the staged records. Edition resolution needs
  the backbone loaded first.

Everything about the feed's shape is UNVERIFIED. The field names in DEFAULT_FIELD_MAP
are guesses based on public descriptions of Flashscore-based actors. Run the actor once,
open its output, and adjust "field_map" in data/daily_feed.json. Missing required fields
fail loudly instead of being guessed.

Usage (from the project root, venv active):
    $env:APIFY_TOKEN = "..."                      # PowerShell
    python scripts/ingest_daily.py                # run the actor and ingest
    python scripts/ingest_daily.py --from-file data/raw/daily/flashscore_apify/<file>.json
                                                  # replay a saved snapshot (no credits spent)

Config: data/daily_feed.json (copy daily_feed.example.json and edit). Standard library only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import unicodedata
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]      # project root (archived script)
DEFAULT_CONFIG = ROOT / "data" / "daily_feed.json"
DB_PATH = ROOT / "tennis.db"
RAW_DIR = ROOT / "data" / "raw" / "daily"
STAGING_DIR = ROOT / "data" / "staging"

# Dotted paths into each actor item. GUESSES: verify against real output and override
# in the config file's "field_map".
DEFAULT_FIELD_MAP = {
    "date": "date",
    "start_time_utc": "startTimeUtc",
    "tour": "tour",
    "category": "category",
    "tournament": "tournament",
    "round": "round",
    "home_name": "homePlayer",
    "away_name": "awayPlayer",
    "score": "score",
    "status": "status",
    "winner_side": "winner",        # expected value: "home" or "away"
    "stats": "statistics.match",    # kept raw; parsed into columns in a later milestone
}
REQUIRED_FIELDS = ("date", "tour", "home_name", "away_name")


# ---------------------------------------------------------------- helpers

def get_path(item: dict, dotted: str | None):
    """Follow 'a.b.c' into nested dicts; None if any step is missing."""
    if not dotted:
        return None
    cur = item
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def normalize_name(name: str) -> str:
    """lowercase, strip accents, hyphens/periods/punctuation -> spaces, collapse spaces."""
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return s.strip()


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------- step 1: get the feed

def run_actor(actor: str, actor_input: dict, token: str) -> list[dict]:
    """Run an Apify actor synchronously and return its dataset items.

    The synchronous endpoint has a time limit (about 5 minutes), which is plenty for
    one day of results. Check the actor's API tab for its input fields and exact call.
    """
    url = f"https://api.apify.com/v2/acts/{actor}/run-sync-get-dataset-items"
    print(f"POST {url}")
    req = urllib.request.Request(
        url,
        data=json.dumps(actor_input).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "tennis-daily-ingest",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=330) as resp:
        data = json.load(resp)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list of items, got {type(data).__name__}")
    return data


# ---------------------------------------------------------------- step 2: snapshot

def register_snapshot(con: sqlite3.Connection, source: str, path: Path,
                      source_ref: str, row_count: int) -> int:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    con.execute("INSERT OR IGNORE INTO source (source, description) VALUES (?, ?)",
                (source, "daily results feed (scraped)"))
    con.execute(
        """INSERT OR IGNORE INTO ingest_snapshot
           (source, fetched_at, source_ref, sha256, raw_path, row_count)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (source, now_utc(), source_ref, digest, str(path), row_count),
    )
    return con.execute(
        "SELECT snapshot_id FROM ingest_snapshot WHERE source=? AND sha256=?",
        (source, digest),
    ).fetchone()[0]


def save_raw(source: str, items: list[dict]) -> Path:
    payload = json.dumps(items, sort_keys=True, ensure_ascii=False).encode("utf-8")
    sha = hashlib.sha256(payload).hexdigest()[:12]
    out_dir = RAW_DIR / source
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{sha}.json"
    path.write_bytes(payload)
    return path


# ---------------------------------------------------------------- step 3: normalize + resolve

class PlayerIndex:
    """In-memory lookup built once per run from the player table."""

    def __init__(self, con: sqlite3.Connection):
        self.full: dict[tuple[str, str], set[int]] = {}
        self.initial: dict[tuple[str, str, str], set[int]] = {}
        for pid, tour, first, last in con.execute(
                "SELECT player_id, tour, name_first, name_last FROM player"):
            first_n, last_n = normalize_name(first or ""), normalize_name(last or "")
            if not last_n:
                continue
            for key in (f"{first_n} {last_n}".strip(), f"{last_n} {first_n}".strip()):
                self.full.setdefault((tour, key), set()).add(pid)
            if first_n:
                self.initial.setdefault((tour, last_n, first_n[0]), set()).add(pid)
        for pid, tour, alias in con.execute(
                "SELECT a.player_id, p.tour, a.alias_normalized "
                "FROM player_alias a JOIN player p USING (player_id)"):
            self.full.setdefault((tour, alias), set()).add(pid)

    def lookup(self, tour: str, name: str) -> tuple[str, list[int]]:
        """Return (status, candidate ids). status: exact | initial | ambiguous | none.

        'exact' (a unique full-name match) is the only status that is auto-linked.
        A surname + first-initial match ("Munar J.") is only a proposal: surname+initial
        collides often, and the player table includes historical players.
        """
        n = normalize_name(name)
        ids = self.full.get((tour, n))
        if ids:
            return ("exact" if len(ids) == 1 else "ambiguous"), sorted(ids)
        m = re.fullmatch(r"(.+) ([a-z])", n)
        if m:
            ids = self.initial.get((tour, m.group(1), m.group(2)))
            if ids:
                return ("initial" if len(ids) == 1 else "ambiguous"), sorted(ids)
        return "none", []


def resolve_player(con, idx: PlayerIndex, source: str, tour: str, name: str,
                   snapshot_id: int) -> int | None:
    key = f"{tour}:{name}"
    row = con.execute("SELECT player_id FROM player_xref WHERE source=? AND source_key=?",
                      (source, key)).fetchone()
    if row:
        return row[0]
    status, ids = idx.lookup(tour, name)
    if status == "exact":
        con.execute(
            """INSERT OR IGNORE INTO player_xref
               (source, source_key, player_id, method, confidence, verified, snapshot_id)
               VALUES (?, ?, ?, 'exact_name', 1.0, 0, ?)""",
            (source, key, ids[0], snapshot_id))
        return ids[0]
    reason = {"initial": "low_confidence", "ambiguous": "ambiguous", "none": "no_candidate"}[status]
    con.execute(
        """INSERT OR IGNORE INTO review_queue
           (entity_type, source, source_key, candidate_ids, reason)
           VALUES ('player', ?, ?, ?, ?)""",
        (source, key, json.dumps(ids), reason))
    return None


def parse_item(item: dict, field_map: dict) -> dict:
    rec = {k: get_path(item, path) for k, path in field_map.items()}
    missing = [f for f in REQUIRED_FIELDS if not rec.get(f)]
    if missing:
        paths = {f: field_map.get(f) for f in missing}
        raise KeyError(
            f"item is missing required fields {paths}; "
            f"edit field_map in the config to match the actor's output. "
            f"Item keys: {sorted(item)[:15]}")
    date = str(rec["date"])[:10]
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
        raise ValueError(f"date {rec['date']!r} is not ISO (YYYY-MM-DD...)")
    rec["date"] = date
    rec["tour"] = str(rec["tour"]).strip().lower()
    return rec


def is_wanted(rec: dict, tours: set[str], category: str | None) -> bool:
    if rec["tour"] not in tours:
        return False
    return "double" not in (category or "").lower()   # singles only


# ---------------------------------------------------------------- main

def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Copy daily_feed.example.json to {path} and edit it.")
    return json.loads(path.read_text(encoding="utf-8"))


def ingest(con: sqlite3.Connection, cfg: dict, raw_path: Path, items: list[dict]) -> dict:
    source = cfg["source"]
    field_map = {**DEFAULT_FIELD_MAP, **cfg.get("field_map", {})}
    tours = {t.lower() for t in cfg.get("tours", ["atp", "wta"])}

    snapshot_id = register_snapshot(con, source, raw_path, cfg.get("actor", "replay"), len(items))
    idx = PlayerIndex(con)

    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    out_path = STAGING_DIR / f"{source}-{snapshot_id}.jsonl"
    counts = {"items": len(items), "skipped": 0, "staged": 0, "fully_resolved": 0}

    with con, open(out_path, "w", encoding="utf-8") as out:
        for item in items:
            rec = parse_item(item, field_map)
            if not is_wanted(rec, tours, rec.get("category")):
                counts["skipped"] += 1
                continue
            home_id = resolve_player(con, idx, source, rec["tour"], rec["home_name"], snapshot_id)
            away_id = resolve_player(con, idx, source, rec["tour"], rec["away_name"], snapshot_id)
            staged = {
                "source": source, "snapshot_id": snapshot_id,
                "date": rec["date"], "start_time_utc": rec["start_time_utc"],
                "tour": rec["tour"], "tournament": rec["tournament"], "round": rec["round"],
                "score": rec["score"], "status": rec["status"], "winner_side": rec["winner_side"],
                "home": {"name": rec["home_name"], "player_id": home_id},
                "away": {"name": rec["away_name"], "player_id": away_id},
                "stats": rec["stats"],
                "resolved": home_id is not None and away_id is not None,
            }
            out.write(json.dumps(staged, ensure_ascii=False) + "\n")
            counts["staged"] += 1
            counts["fully_resolved"] += staged["resolved"]

    counts["review_queue_open"] = con.execute(
        "SELECT COUNT(*) FROM review_queue WHERE status='open' AND source=?", (source,)).fetchone()[0]
    counts["staging_file"] = str(out_path)
    return counts


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--db", type=Path, default=DB_PATH)
    ap.add_argument("--from-file", type=Path,
                    help="replay a saved raw snapshot instead of calling the actor")
    args = ap.parse_args(argv)

    if not args.db.exists():
        print(f"{args.db} not found. Run scripts/init_db.py first.")
        return 2
    try:
        cfg = load_config(args.config)
        if args.from_file:
            raw_path = args.from_file.resolve()
            items = json.loads(raw_path.read_text(encoding="utf-8"))
        else:
            token = os.environ.get("APIFY_TOKEN")
            if not token:
                print("Set the APIFY_TOKEN environment variable (or use --from-file).")
                return 2
            items = run_actor(cfg["actor"], cfg.get("input", {}), token)
            raw_path = save_raw(cfg["source"], items)
            print(f"saved raw response: {raw_path}")

        con = sqlite3.connect(args.db)
        con.execute("PRAGMA foreign_keys = ON")
        counts = ingest(con, cfg, raw_path, items)
        con.close()
    except urllib.error.HTTPError as e:
        print(f"FAILED, HTTP {e.code} {e.reason} for {e.url}")
        return 1
    except (urllib.error.URLError, FileNotFoundError, KeyError, ValueError) as e:
        print(f"FAILED: {e}")
        return 1

    for k, v in counts.items():
        print(f"{k}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
