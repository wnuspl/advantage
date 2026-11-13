"""Download pinned snapshots of the tennis data repos into data/raw/.

Jeff Sackmann's tennis_atp / tennis_wta / slam point-by-point repos were removed
from GitHub (404 as of Oct 2026). This script now fetches:

  sackmann_archive  Aneeshers/tennis-sackmann-archive   third-party mirror of the ATP/WTA
                    match files (snapshot from upstream, June 2026) and slam point-by-point
                    (through 2024). An archive, not a fork: the original upstream commit
                    hashes are not recorded, so we record our own hash + checksum here.
  mcp               JeffSackmann/tennis_MatchChartingProject   still public upstream.

All data is CC BY-NC-SA 4.0 (attribution, non-commercial, share-alike).

Each repo is downloaded as a zip of one exact commit and unpacked to
data/raw/<repo>-<commit sha>/. Existing snapshot folders are never modified, so
re-running after an update adds a new folder beside the old one. Every download is
logged to data/SOURCES.md (track that file in git).

Usage (from the project root):
    python scripts/fetch_data.py                              # both
    python scripts/fetch_data.py tennis_MatchChartingProject  # just one

Standard library only. Set GITHUB_TOKEN to avoid the low unauthenticated rate limit
(optional).
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from datetime import date
from pathlib import Path

# (owner, repo, source name used in the database's `source` table)
SPECS = [
    ("Aneeshers", "tennis-sackmann-archive", "sackmann_archive"),
    ("JeffSackmann", "tennis_MatchChartingProject", "mcp"),
]

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
SOURCES_FILE = ROOT / "data" / "SOURCES.md"


def _request(url: str, accept: str | None = None) -> urllib.request.Request:
    headers = {"User-Agent": "tennis-data-fetch"}  # GitHub rejects requests without one
    if accept:
        headers["Accept"] = accept
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return urllib.request.Request(url, headers=headers)


def latest_commit_sha(owner: str, repo: str) -> str:
    """Ask GitHub for the commit hash at the tip of the repo's default branch."""
    url = f"https://api.github.com/repos/{owner}/{repo}/commits/HEAD"
    print(f"{repo}: GET {url}")
    req = _request(url, accept="application/vnd.github.sha")
    with urllib.request.urlopen(req, timeout=30) as resp:
        sha = resp.read().decode("ascii").strip()
    if len(sha) != 40 or any(c not in "0123456789abcdef" for c in sha):
        raise ValueError(f"Unexpected commit hash for {owner}/{repo}: {sha!r}")
    return sha


def download(url: str, dest: Path) -> str:
    """Stream a file to disk and return its sha256."""
    digest = hashlib.sha256()
    with urllib.request.urlopen(_request(url), timeout=300) as resp, open(dest, "wb") as out:
        while chunk := resp.read(1 << 20):
            out.write(chunk)
            digest.update(chunk)
    return digest.hexdigest()


def fetch_repo(owner: str, repo: str, source: str) -> None:
    sha = latest_commit_sha(owner, repo)
    target = RAW_DIR / f"{repo}-{sha}"
    if target.exists():
        print(f"{repo}: already have {sha[:10]}, skipping")
        report(repo, target)
        return

    zip_url = f"https://github.com/{owner}/{repo}/archive/{sha}.zip"
    print(f"{repo}: downloading {sha[:10]} from {zip_url}")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Work in a temp dir inside data/raw so the final move is on the same drive.
    # A crash mid-download then leaves no half-finished snapshot folder behind.
    with tempfile.TemporaryDirectory(dir=RAW_DIR, prefix=".tmp-") as tmp:
        tmp_path = Path(tmp)
        zip_path = tmp_path / "repo.zip"
        zip_sha256 = download(zip_url, zip_path)

        extract_dir = tmp_path / "extracted"
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)

        # GitHub zips contain a single top-level folder named <repo>-<sha>.
        top_level = [p for p in extract_dir.iterdir() if p.is_dir()]
        if len(top_level) != 1:
            raise RuntimeError(f"{repo}: expected one top-level folder, found {len(top_level)}")
        shutil.move(str(top_level[0]), str(target))

    log_source(owner, repo, source, sha, zip_sha256, zip_url)
    print(f"{repo}: saved to {target.relative_to(ROOT)}")
    report(repo, target)


MATCH_FILE = re.compile(r"^(atp|wta)_matches_(\d{4})\.csv$", re.IGNORECASE)


def report(repo: str, target: Path) -> None:
    """Print what a snapshot contains so staleness or a layout change is obvious."""
    csvs = list(target.rglob("*.csv"))
    print(f"{repo}: {len(csvs)} CSV files")
    newest: dict[str, int] = {}
    for p in csvs:
        m = MATCH_FILE.match(p.name)
        if m:
            tour, year = m.group(1).lower(), int(m.group(2))
            newest[tour] = max(newest.get(tour, 0), year)
    if newest:
        for tour, year in sorted(newest.items()):
            print(f"{repo}:   newest {tour}_matches file is {year}")
        print(f"{repo}:   (check the last match date inside it; mirrors stop on the day they were made)")
    elif "archive" in repo:
        print(f"{repo}: WARNING no atp/wta_matches_YYYY.csv files found; "
              "the mirror's layout may differ from what the loaders expect. Inspect the folder.")


def log_source(owner: str, repo: str, source: str, sha: str, zip_sha256: str, zip_url: str) -> None:
    SOURCES_FILE.parent.mkdir(parents=True, exist_ok=True)
    line = (
        f"{source}  {repo}  {sha}  zip_sha256={zip_sha256}  fetched {date.today().isoformat()}  "
        f"repo=https://github.com/{owner}/{repo}  zip={zip_url}\n"
    )
    with open(SOURCES_FILE, "a", encoding="utf-8") as f:
        f.write(line)


def main(argv: list[str]) -> int:
    by_repo = {repo: (owner, repo, source) for owner, repo, source in SPECS}
    names = argv or list(by_repo)
    unknown = [n for n in names if n not in by_repo]
    if unknown:
        print(f"Unknown repo(s): {unknown}. Choose from {list(by_repo)}")
        return 2

    failures = 0
    for name in names:
        owner, repo, source = by_repo[name]
        try:
            fetch_repo(owner, repo, source)
        except urllib.error.HTTPError as e:
            failures += 1
            print(f"{repo}: FAILED, HTTP {e.code} {e.reason} for {e.url}")
            print(f"{repo}: open that URL in a browser to inspect it")
        except (urllib.error.URLError, ValueError, RuntimeError, zipfile.BadZipFile) as e:
            failures += 1
            print(f"{repo}: FAILED ({e})")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
