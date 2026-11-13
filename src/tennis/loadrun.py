"""Per-file load accounting: every row read ends in exactly one outcome."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class LoadRun:
    """Counts outcomes for one loader over one registered file.

    Call exactly one of inserted() / linked() / skipped(reason) / queued() per row read.
    finish() writes the counts and refuses to close a run whose numbers do not add up.
    """

    MAX_EXAMPLES = 5

    def __init__(self, con: sqlite3.Connection, snapshot_id: int, loader: str):
        self.con = con
        self.snapshot_id = snapshot_id
        self.loader = loader
        self.read = self.n_inserted = self.n_linked = self.n_skipped = self.n_queued = 0
        self.skips: dict[str, list] = {}   # reason -> [count, examples]
        cur = con.execute(
            "INSERT INTO load_run (snapshot_id, loader, started_at) VALUES (?,?,?)",
            (snapshot_id, loader, _now()),
        )
        self.load_run_id = cur.lastrowid

    def row(self) -> None:
        self.read += 1

    def inserted(self) -> None:
        self.n_inserted += 1

    def linked(self) -> None:
        self.n_linked += 1

    def queued(self) -> None:
        self.n_queued += 1

    def skipped(self, reason: str, example: object = None, n: int = 1) -> None:
        self.n_skipped += n
        entry = self.skips.setdefault(reason, [0, []])
        entry[0] += n
        if example is not None and len(entry[1]) < self.MAX_EXAMPLES:
            entry[1].append(str(example))

    def finish(self) -> None:
        total = self.n_inserted + self.n_linked + self.n_skipped + self.n_queued
        if total != self.read:
            raise AssertionError(
                f"{self.loader} snapshot {self.snapshot_id}: {self.read} rows read but "
                f"{total} outcomes recorded")
        self.con.executemany(
            "INSERT INTO load_skip (load_run_id, reason, n, example) VALUES (?,?,?,?)",
            [(self.load_run_id, r, c, "; ".join(ex) or None) for r, (c, ex) in self.skips.items()],
        )
        self.con.execute(
            """UPDATE load_run SET finished_at=?, rows_read=?, rows_inserted=?, rows_linked=?,
                   rows_skipped=?, rows_queued=? WHERE load_run_id=?""",
            (_now(), self.read, self.n_inserted, self.n_linked, self.n_skipped, self.n_queued,
             self.load_run_id),
        )
