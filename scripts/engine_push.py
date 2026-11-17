"""Ship the engine's replay to the server instead of replaying there (the server's 1 GB
swaps for 40+ min on a full replay; here it takes ~5 min). Uploads tennis.db, the cached
replay (models/engine/run<id>.pkl) and the saved states (models/engine/state/run<id>/),
then swaps them in and restarts the watch, which loads the cache instead of replaying.

    python scripts/engine_push.py                  # build the cache here if needed, push, restart
    python scripts/engine_push.py --no-restart     # stage and swap only

The cache only loads if its key matches what the server computes: tennis.db contents (so
tennis.db goes up with it), the model code (so the server must run the same commit: the
script checks git HEAD on both sides and refuses uncommitted model code here), the run's
params in models.db, and the calibration quarter. The server recomputes the key on the
uploaded files before anything is swapped; on a mismatch nothing changes.
Host: --host, else ADVANTAGE_SERVER in .env. Needs ssh (key auth), tar and sudo for systemctl.
"""
from __future__ import annotations

import argparse
import pickle
import shlex
import subprocess
import sys
import tarfile
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tennis.engine.alerts import env_value                          # noqa: E402
from tennis.engine.predict import CACHE_DIR, CODE, PointsPredictor  # noqa: E402
from tennis.engine.state import STATE_DIR                           # noqa: E402

STAGE = ".engine_push"          # under the remote project dir
SERVICE = "advantage-watch"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def ssh(host: str, cmd: str, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(["ssh", "-o", "BatchMode=yes", host, cmd], **kw)


def files_to_push(run_id: int, data: Path) -> list[tuple[Path, str]]:
    """(local file, path in the archive, relative to the remote project dir)."""
    out = [(data, "tennis.db"), (CACHE_DIR / f"run{run_id}.pkl", f"models/engine/run{run_id}.pkl")]
    sdir = STATE_DIR / f"run{run_id}"
    if sdir.is_dir():
        out += [(f, f"models/engine/state/run{run_id}/{f.name}") for f in sorted(sdir.glob("*.pkl"))]
    return out


def verify_snippet(run_id: int, key: str) -> str:
    """Python run on the server: the cache key of the staged tennis.db under the server's own
    code and models.db must equal the uploaded cache's key."""
    return f"""
import sqlite3, sys
from datetime import date
from pathlib import Path
sys.path.insert(0, "src")
from tennis import models_db
from tennis.engine.predict import cache_key, run_params
con = sqlite3.connect(models_db._uri(models_db.MODELS_DB, "ro"), uri=True)
models_db.attach_data(con, Path("{STAGE}/tennis.db"))
_, params = run_params(con, {run_id})
k = cache_key(con, {run_id}, params, None, date.today().isoformat())
print("key ok" if k == "{key}" else "key MISMATCH: server " + k[:12] + " vs upload {key[:12]}")
sys.exit(0 if k == "{key}" else 1)
"""


def install_script(remote_dir: str, run_id: int, key: str, restart: bool) -> str:
    """Shell run on the server after the upload: verify, then swap, then (re)start the watch.
    Nothing outside the stage changes unless the key matches."""
    d = shlex.quote(remote_dir)
    py = shlex.quote(verify_snippet(run_id, key))
    stop = f"sudo systemctl stop {SERVICE} && " if restart else ""
    start = f" && sudo systemctl start {SERVICE} && echo watch restarted" if restart else ""
    return (f"set -e; cd {d}; "
            f"if ! .venv/bin/python -c {py}; then rm -rf {STAGE}; exit 3; fi; "
            f"{stop}"
            f"mkdir -p models/engine/state/run{run_id} && "
            f"for f in $(cd {STAGE} && find . -type f); do mv -f {STAGE}/$f $f; done && "
            f"rm -f tennis.db-wal tennis.db-shm && rm -rf {STAGE} && echo installed"
            f"{start}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=env_value("ADVANTAGE_SERVER"))
    ap.add_argument("--run", type=int, default=790)
    ap.add_argument("--remote-dir", default="advantage", help="project dir, relative to the remote home")
    ap.add_argument("--data", type=Path, default=ROOT / "tennis.db")
    ap.add_argument("--no-restart", action="store_true", help="don't stop/start the watch")
    ap.add_argument("--level", type=int, default=3, help="gzip level for the upload")
    args = ap.parse_args(argv)
    if not args.host:
        raise SystemExit("no host: pass --host or set ADVANTAGE_SERVER in .env")

    # 1. same model code on both sides
    dirty = git("status", "--porcelain", "--", *[f"src/tennis/{f}" for f in CODE])
    if dirty:
        raise SystemExit(f"model code has uncommitted changes (the server would compute another key):\n{dirty}")
    head = git("rev-parse", "HEAD")
    r = ssh(args.host, f"cd {shlex.quote(args.remote_dir)} && git rev-parse HEAD", capture_output=True, text=True)
    if r.returncode != 0 or r.stdout.strip() != head:
        raise SystemExit(f"server is at {r.stdout.strip()[:10] or '?'}, here {head[:10]}: deploy the code first")
    wal = args.data.with_name(args.data.name + "-wal")
    if wal.exists() and wal.stat().st_size:
        raise SystemExit(f"{wal} is not empty: close whatever writes tennis.db (or checkpoint it) first")

    # 2. the cache, built here if missing or stale
    t = time.perf_counter()
    PointsPredictor.load(args.run, data_path=args.data, today=date.today().isoformat())
    with (CACHE_DIR / f"run{args.run}.pkl").open("rb") as f:
        key = pickle.load(f)["key"]
    print(f"cache ready ({time.perf_counter() - t:.0f}s), key {key[:12]}")

    # 3. stream everything into the remote stage
    files = files_to_push(args.run, args.data)
    total = sum(f.stat().st_size for f, _ in files)
    print(f"uploading {len(files)} files, {total / 1e9:.2f} GB before compression, to {args.host}")
    t = time.perf_counter()
    d = shlex.quote(args.remote_dir)
    p = subprocess.Popen(["ssh", "-o", "BatchMode=yes", args.host,
                          f"set -e; cd {d}; rm -rf {STAGE}; mkdir {STAGE}; tar -xzf - -C {STAGE}"],
                         stdin=subprocess.PIPE)
    with tarfile.open(fileobj=p.stdin, mode="w|gz", compresslevel=args.level) as tar:
        for f, arc in files:
            tar.add(f, arcname=arc)
    p.stdin.close()
    if p.wait() != 0:
        raise SystemExit("upload failed")
    print(f"uploaded in {time.perf_counter() - t:.0f}s")

    # 4. verify on the server, swap, restart
    r = ssh(args.host, install_script(args.remote_dir, args.run, key, not args.no_restart))
    if r.returncode == 3:
        raise SystemExit("the server computes a different cache key: nothing was changed")
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
