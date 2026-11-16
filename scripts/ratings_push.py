"""Push the ratings export to the server running scripts/web_server.py (the full replay
needs more memory than the server has, so the export runs here and only its files go up).

    python scripts/ratings_export.py && python scripts/ratings_push.py
    python scripts/ratings_push.py --host ubuntu@my-server --run 790

The host comes from --host, else ADVANTAGE_SERVER in the environment or .env. The upload
lands in run<id>.new next to the live folder and is swapped in at the end, so the server
never serves a half-copied export. Needs ssh (key auth) and tar on the server.
"""
from __future__ import annotations

import argparse
import io
import shlex
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tennis.engine.alerts import env_value                          # noqa: E402

OUT = ROOT / "models" / "ratings"


def remote_swap(remote_dir: str, run_id: int) -> str:
    d, name = shlex.quote(remote_dir), f"run{run_id}"
    return (f"set -e; mkdir -p {d} && cd {d} && rm -rf {name}.new && mkdir {name}.new && "
            f"tar -xzf - -C {name}.new && rm -rf {name}.old && "
            f"{{ [ -d {name} ] && mv {name} {name}.old || true; }} && mv {name}.new {name} && rm -rf {name}.old && "
            f"echo \"$(ls {name}/weeks | wc -l) weeks on the server\"")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=env_value("ADVANTAGE_SERVER"), help="ssh destination, e.g. ubuntu@1.2.3.4")
    ap.add_argument("--run", type=int, default=790)
    ap.add_argument("--remote-dir", default="advantage/models/ratings", help="relative to the remote home")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    if not args.host:
        raise SystemExit("no host: pass --host or set ADVANTAGE_SERVER in .env")
    src = args.out / f"run{args.run}"
    if not (src / "index.json").is_file():
        raise SystemExit(f"{src} has no export (python scripts/ratings_export.py --run {args.run})")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for f in sorted(src.rglob("*")):
            if f.is_file():
                tar.add(f, arcname=f.relative_to(src).as_posix())
    print(f"uploading {buf.tell() / 1e6:.0f} MB to {args.host}")
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", args.host, remote_swap(args.remote_dir, args.run)],
                       input=buf.getvalue())
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
