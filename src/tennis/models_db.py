"""models.db: persistent modeling results, keyed by stable source keys.

tennis.db is disposable and its surrogate IDs change on every rebuild; models.db is
never rebuilt. Typical use:

    con = models_db.connect()                    # creates/updates the schema, idempotent
    models_db.attach_data(con)                   # tennis.db as read-only schema 'data'
    mkeys = models_db.match_keys(con)            # data match_id -> stable match_key
    run_id = models_db.start_run(con, "elo_surface_v1", {"k": 32}, sweep_id="s1")
    ...
    models_db.finish_run(con, run_id)
"""
from __future__ import annotations

import json
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS_DB = ROOT / "models.db"
DATA_DB = ROOT / "tennis.db"
MODELS_SCHEMA = ROOT / "models_schema.sql"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _uri(path: Path, mode: str | None = None) -> str:
    uri = "file:" + Path(path).resolve().as_posix()
    return uri + (f"?mode={mode}" if mode else "")


def connect(path: Path = MODELS_DB) -> sqlite3.Connection:
    """Open (creating if needed) models.db and apply the additive schema."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(_uri(path), uri=True)   # uri=True so ATTACH can open read-only
    con.executescript(MODELS_SCHEMA.read_text(encoding="utf-8"))
    con.execute("PRAGMA foreign_keys = ON")
    return con


def attach_data(con: sqlite3.Connection, data_path: Path = DATA_DB, alias: str = "data") -> None:
    """Attach tennis.db read-only, so queries can join e.g. data.matches."""
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} not found; run scripts/build_db.py first")
    con.execute(f"ATTACH DATABASE ? AS {alias}", (_uri(data_path, "ro"),))


def match_keys(con: sqlite3.Connection, alias: str = "data") -> dict[int, str]:
    """tennis.db match_id -> stable match_key (lexicographically first seed xref)."""
    return dict(con.execute(
        f"""SELECT match_id, MIN(source || ':' || source_key) FROM {alias}.match_xref
            WHERE method = 'seed' GROUP BY match_id"""))


def player_keys(con: sqlite3.Connection, alias: str = "data") -> dict[int, str]:
    """tennis.db player_id -> stable player_key (lexicographically first seed xref)."""
    return dict(con.execute(
        f"""SELECT player_id, MIN(source || ':' || source_key) FROM {alias}.player_xref
            WHERE method = 'seed' GROUP BY player_id"""))


def _git_commit() -> str | None:
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
        return head + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def start_run(con: sqlite3.Connection, model_id: str, params: dict, *, sweep_id: str | None = None,
              notes: str | None = None, alias: str = "data") -> int:
    """Record a run before it starts. Provenance comes from the attached tennis.db."""
    built_at = sources = None
    attached = {r[1] for r in con.execute("PRAGMA database_list")}
    if alias in attached:
        row = con.execute(f"SELECT value FROM {alias}.schema_meta WHERE key='built_at'").fetchone()
        built_at = row[0] if row else None
        sources = json.dumps(dict(con.execute(
            f"SELECT source, MIN(source_ref) FROM {alias}.ingest_snapshot GROUP BY source")), sort_keys=True)
    cur = con.execute(
        """INSERT INTO model_run (model_id, params_json, sweep_id, data_built_at, data_sources,
               code_commit, started_at, notes) VALUES (?,?,?,?,?,?,?,?)""",
        (model_id, json.dumps(params, sort_keys=True), sweep_id, built_at, sources, _git_commit(),
         _now(), notes))
    con.commit()
    return cur.lastrowid


def finish_run(con: sqlite3.Connection, run_id: int) -> None:
    con.execute("UPDATE model_run SET finished_at=? WHERE run_id=?", (_now(), run_id))
    con.commit()


def delete_run(con: sqlite3.Connection, run_id: int) -> None:
    """Remove a run and (by cascade) its metrics, predictions and ratings."""
    con.execute("DELETE FROM model_run WHERE run_id=?", (run_id,))
    con.commit()
