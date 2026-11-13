import sqlite3
from pathlib import Path

root = Path(__file__).resolve().parent.parent
db_path = root / "tennis.db"

if db_path.exists():
    db_path.unlink()            # rebuild from scratch; the database is disposable

con = sqlite3.connect(db_path)
con.executescript((root / "schema.sql").read_text(encoding="utf-8"))
con.close()
print(f"Created {db_path}")
