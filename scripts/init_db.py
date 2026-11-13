"""Create an empty tennis.db from schema.sql (no data). For a full build use build_db.py."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tennis import db  # noqa: E402

path = ROOT / "tennis.db"
db.create(path).close()      # rebuild from scratch; the database is disposable
print(f"Created {path}")
