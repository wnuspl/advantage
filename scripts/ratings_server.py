"""Serve the ratings viewer: web/ratings.html plus the weekly standings written by
scripts/ratings_export.py. Standard library only, read-only.

    python scripts/ratings_server.py                         # http://127.0.0.1:8790/
    python scripts/ratings_server.py --host 0.0.0.0 --port 8790 --run 790

Routes:
    /                     web/ratings.html
    /data/index.json      the export's index (weeks, fields, run)
    /data/players.json
    /data/weeks/<monday>.json.gz   served as JSON with Content-Encoding: gzip
Files are read per request, so rerunning the export updates the page without a restart.
The export is CC BY-NC-SA data (Sackmann): bind to a private address, not the internet.
"""
from __future__ import annotations

import argparse
import sys
from email.utils import formatdate
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "web" / "ratings.html"
OUT = ROOT / "models" / "ratings"


def resolve(data_dir: Path, path: str) -> tuple[Path, str, bool] | None:
    """(file, content type, gzipped) for a request path, or None. Never leaves data_dir."""
    path = unquote(urlsplit(path).path)
    if path in ("/", "/index.html", "/ratings.html"):
        return PAGE, "text/html; charset=utf-8", False
    if not path.startswith("/data/"):
        return None
    base = data_dir.resolve()
    f = (base / path[len("/data/"):]).resolve()
    try:
        f.relative_to(base)
    except ValueError:
        return None
    if f.name.endswith(".json.gz"):
        return f, "application/json", True
    if f.suffix == ".json":
        return f, "application/json; charset=utf-8", False
    return None


def make_handler(data_dir: Path, quiet: bool = False):
    class Handler(BaseHTTPRequestHandler):
        server_version = "ratings/1"

        def do_GET(self):
            self._serve(body=True)

        def do_HEAD(self):
            self._serve(body=False)

        def _serve(self, body: bool):
            hit = resolve(data_dir, self.path)
            if hit is None or not hit[0].is_file():
                if self.path.startswith("/data/index.json"):
                    self.send_error(HTTPStatus.NOT_FOUND, "no export yet: run python scripts/ratings_export.py")
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)
                return
            f, ctype, gz = hit
            try:
                st = f.stat()
                data = f.read_bytes()
            except OSError:   # swapped out by an export mid-request
                self.send_error(HTTPStatus.SERVICE_UNAVAILABLE)
                return
            etag = f'"{st.st_mtime_ns:x}-{st.st_size:x}"'
            if self.headers.get("If-None-Match") == etag:
                self.send_response(HTTPStatus.NOT_MODIFIED)
                self.send_header("ETag", etag)
                self.end_headers()
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", ctype)
            if gz:
                self.send_header("Content-Encoding", "gzip")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("ETag", etag)
            self.send_header("Last-Modified", formatdate(st.st_mtime, usegmt=True))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            if body:
                self.wfile.write(data)

        def log_message(self, fmt, *args):
            if not quiet:
                sys.stderr.write(f"{self.address_string()} {fmt % args}\n")

    return Handler


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--run", type=int, default=790)
    ap.add_argument("--out", type=Path, default=OUT, help="the export root (ratings_export.py --out)")
    ap.add_argument("--quiet", action="store_true", help="no request log")
    args = ap.parse_args(argv)
    data_dir = args.out / f"run{args.run}"
    if not (data_dir / "index.json").is_file():
        print(f"warning: {data_dir} has no export yet (python scripts/ratings_export.py --run {args.run})")
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(data_dir, args.quiet))
    print(f"ratings viewer for run {args.run} on http://{args.host}:{args.port}/")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
