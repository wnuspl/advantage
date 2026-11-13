"""Connection helpers and schema creation."""
from __future__ import annotations

import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "schema.sql"
MIN_SQLITE = (3, 37, 0)   # STRICT tables


def connect(path: Path | str, bulk: bool = False) -> sqlite3.Connection:
    con = sqlite3.connect(str(path))
    con.execute("PRAGMA foreign_keys = ON")
    if bulk:
        # Safe because builds write to a temp file that is only swapped in on success.
        con.execute("PRAGMA synchronous = OFF")
        con.execute("PRAGMA cache_size = -200000")   # ~200 MB
    return con


def remove_db(path: Path) -> None:
    for suffix in ("", "-wal", "-shm", "-journal"):
        p = Path(str(path) + suffix)
        if p.exists():
            p.unlink()


def create(path: Path, schema_path: Path = SCHEMA_PATH, bulk: bool = False) -> sqlite3.Connection:
    """Create a fresh database at `path` (deleting any existing file) and apply the schema."""
    if sqlite3.sqlite_version_info < MIN_SQLITE:
        raise RuntimeError(f"SQLite {sqlite3.sqlite_version} is too old; need 3.37+ for STRICT tables")
    remove_db(path)
    con = connect(path, bulk=bulk)
    con.executescript(schema_path.read_text(encoding="utf-8"))
    con.execute("PRAGMA foreign_keys = ON")   # executescript can reset per-connection state
    return con
