"""Manifest reading, file classification, and snapshot registration.

Registering a file (hash + row count + ingest_snapshot row) is separate from loading it.
Raw files are immutable: data/snapshot_lock.json records the sha256 of every registered
file per pinned folder, and a build stops if a file's bytes no longer match.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path

from .acesapi import count_draw_rows


class BuildError(RuntimeError):
    """A condition that must stop the build (layout drift, changed raw bytes, bad manifest)."""


@dataclass(frozen=True)
class RawFile:
    source: str
    source_ref: str
    folder: str          # snapshot folder name under data/raw
    relpath: str         # forward-slash path inside the folder
    path: Path           # absolute path on disk
    group: str
    loader: str
    fetched_at: str | None


def load_manifest(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    if m.get("manifest_version") != 1:
        raise BuildError(f"{path}: unsupported manifest_version {m.get('manifest_version')!r}")
    for snap in m["snapshots"]:
        for g in snap["groups"]:
            if g["source"] not in m["sources"]:
                raise BuildError(f"group {g['name']}: source {g['source']!r} not declared in 'sources'")
    return m


def plan_files(manifest: dict, raw_root: Path) -> tuple[list[RawFile], list[tuple[str, str, str]]]:
    """Classify every file in each pinned folder. Returns (files to register, ignored files).

    Stops on: a missing folder, a file matching two groups, a file matching nothing,
    or a group pattern that matches no file (the layout changed under us).
    """
    files: list[RawFile] = []
    ignored: list[tuple[str, str, str]] = []
    for snap in manifest["snapshots"]:
        folder = raw_root / snap["folder"]
        if not folder.is_dir():
            raise BuildError(f"pinned folder not found: {folder}")
        if snap["commit"] not in snap["folder"]:
            raise BuildError(f"{snap['folder']}: folder name does not contain commit {snap['commit']}")
        pattern_hits = {(g["name"], pat): 0 for g in snap["groups"] for pat in g["files"]}
        unclassified = []
        for p in sorted(x for x in folder.rglob("*") if x.is_file()):
            rel = p.relative_to(folder).as_posix()
            hits = [(g, pat) for g in snap["groups"] for pat in g["files"] if fnmatchcase(rel, pat)]
            groups = {g["name"] for g, _ in hits}
            if len(groups) > 1:
                raise BuildError(f"{snap['folder']}/{rel} matches several groups: {sorted(groups)}")
            if hits:
                g = hits[0][0]
                for _, pat in hits:
                    pattern_hits[(g["name"], pat)] += 1
                files.append(RawFile(g["source"], snap["commit"], snap["folder"], rel, p,
                                     g["name"], g["loader"], snap.get("fetched_at")))
                continue
            reason = next((ig["reason"] for ig in snap.get("ignore", [])
                           if fnmatchcase(rel, ig["pattern"])), None)
            if reason is None:
                unclassified.append(rel)
            else:
                ignored.append((snap["folder"], rel, reason))
        if unclassified:
            shown = "\n  ".join(unclassified[:20])
            raise BuildError(
                f"{snap['folder']}: {len(unclassified)} file(s) match no group and no ignore "
                f"pattern; add them to data/build_manifest.json:\n  {shown}")
        empty = [f"{g}: {pat}" for (g, pat), n in pattern_hits.items() if n == 0]
        if empty:
            raise BuildError(f"{snap['folder']}: patterns matched no file (layout changed?): {empty}")
    return files, ignored


# Row counters for non-CSV raw files, by loader. A JSON file's "rows" are what its loader
# reads one outcome for (e.g. match rows in an AcesAPI get_tournament response).
JSON_ROW_COUNTERS = {"acesapi_draws": count_draw_rows}


def file_stats(path: Path, loader: str | None = None) -> tuple[str, int]:
    """sha256 of the bytes and the number of data records (CSV: header excluded)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    if path.suffix.lower() == ".json":
        counter = JSON_ROW_COUNTERS.get(loader or "")
        if counter is None:
            raise BuildError(f"{path}: no row counter for JSON files of loader {loader!r}")
        try:
            return h.hexdigest(), counter(path)
        except (ValueError, KeyError, TypeError) as e:
            raise BuildError(f"{path}: cannot count rows ({e})") from e
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        rows = sum(1 for _ in reader)
    return h.hexdigest(), rows


def read_lock(lock_path: Path) -> dict:
    if not lock_path.exists():
        return {}
    with open(lock_path, encoding="utf-8") as f:
        return json.load(f)


def write_lock(lock_path: Path, lock: dict) -> None:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = lock_path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(lock, f, indent=1, sort_keys=True)
        f.write("\n")
    tmp.replace(lock_path)


def register_sources(con: sqlite3.Connection, manifest: dict) -> None:
    con.executemany(
        "INSERT INTO source (source, description, license, url) VALUES (?,?,?,?)",
        [(k, v.get("description"), v.get("license"), v.get("url")) for k, v in manifest["sources"].items()],
    )


def register_files(con: sqlite3.Connection, files: list[RawFile], project_root: Path,
                   lock_path: Path, progress=print) -> dict[str, list[tuple[int, RawFile]]]:
    """Hash, count and register every planned file. Returns {loader: [(snapshot_id, file)]}.

    Compares against the lock first and raises before writing anything if a known file
    changed or disappeared; new files are added to the lock.
    """
    lock = read_lock(lock_path)
    stats = {}
    for i, rf in enumerate(files, 1):
        stats[rf] = file_stats(rf.path, rf.loader)
        if i % 100 == 0 or i == len(files):
            progress(f"  hashed {i}/{len(files)} files")

    changed = []
    present: dict[str, set] = {}
    for rf, (sha, _) in stats.items():
        present.setdefault(rf.folder, set()).add(rf.relpath)
        known = lock.get(rf.folder, {}).get("files", {}).get(rf.relpath)
        if known is not None and known != sha:
            changed.append(f"{rf.folder}/{rf.relpath}: lock {known[:12]}, disk {sha[:12]}")
    for folder, entry in lock.items():
        if folder in present:
            missing = set(entry.get("files", {})) - present[folder]
            changed += [f"{folder}/{rel}: in lock but missing on disk (or no longer in a group)"
                        for rel in sorted(missing)]
    if changed:
        raise BuildError("raw files changed under a pinned commit (raw files are immutable; "
                         "fetch a new snapshot instead):\n  " + "\n  ".join(changed[:20]))

    by_loader: dict[str, list[tuple[int, RawFile]]] = {}
    for rf, (sha, n) in stats.items():
        raw_path = rf.path.relative_to(project_root).as_posix() if rf.path.is_relative_to(project_root) \
            else rf.path.as_posix()
        cur = con.execute(
            """INSERT INTO ingest_snapshot (source, source_ref, file_relpath, raw_path, sha256,
                   row_count, file_group, loader, fetched_at) VALUES (?,?,?,?,?,?,?,?,?)""",
            (rf.source, rf.source_ref, rf.relpath, raw_path, sha, n, rf.group, rf.loader, rf.fetched_at),
        )
        by_loader.setdefault(rf.loader, []).append((cur.lastrowid, rf))
        entry = lock.setdefault(rf.folder, {"source_ref": rf.source_ref, "files": {}})
        entry["files"][rf.relpath] = sha
    write_lock(lock_path, lock)
    return by_loader


def open_csv(path: Path):
    """Open a raw CSV for reading (UTF-8, BOM tolerated)."""
    return open(path, encoding="utf-8-sig", newline="")
