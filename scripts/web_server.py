"""Serve the web pages: the live Kalshi board (web/live.html, from the watch's live.db) and
the ratings viewer (web/ratings.html, from scripts/ratings_export.py's weekly standings).
Standard library only, read-only: nothing here talks to Kalshi or writes a database.

    python scripts/web_server.py                         # http://127.0.0.1:8790/
    python scripts/web_server.py --host 0.0.0.0 --port 8790 --run 790 --live-db live.db

Routes:
    /, /live              web/live.html
    /ratings              web/ratings.html
    /nav.js               the shared navigation bar (web/nav.js)
    /api/live             engine.livefeed.live_view(live.db) as JSON (recomputed at most every few s)
    /data/index.json      the ratings export's index (weeks, fields, run)
    /data/players.json
    /data/weeks/<monday>.json.gz   served as JSON with Content-Encoding: gzip
Files are read per request, so rerunning the export updates the page without a restart.
The data is CC BY-NC-SA (Sackmann) plus AcesAPI results: keep the address private.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from email.utils import formatdate
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tennis.engine import livefeed                                    # noqa: E402

WEB = ROOT / "web"
OUT = ROOT / "models" / "ratings"
LIVE_DB = ROOT / "live.db"
HTML = "text/html; charset=utf-8"
PAGES = {"/": ("live.html", HTML), "/index.html": ("live.html", HTML), "/live": ("live.html", HTML),
         "/live.html": ("live.html", HTML), "/ratings": ("ratings.html", HTML),
         "/ratings.html": ("ratings.html", HTML), "/nav.js": ("nav.js", "text/javascript; charset=utf-8")}


def resolve(data_dir: Path, path: str, web: Path = WEB) -> tuple[Path, str, bool] | None:
    """(file, content type, gzipped) for a request path, or None. Never leaves data_dir."""
    path = unquote(urlsplit(path).path)
    if path in PAGES:
        name, ctype = PAGES[path]
        return web / name, ctype, False
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


class LiveCache:
    """live_view() of live.db, recomputed when older than `ttl` seconds (the watch polls
    every ~60 s; several open tabs share one computation)."""

    def __init__(self, path: Path, ttl: float = 5.0):
        self.path, self.ttl = Path(path), ttl
        self.lock = threading.Lock()
        self.at, self.body = 0.0, b""

    def get(self) -> bytes:
        with self.lock:
            if time.monotonic() - self.at > self.ttl or not self.body:
                con = livefeed.connect(self.path)
                try:
                    self.body = json.dumps(livefeed.live_view(con), separators=(",", ":")).encode("utf-8")
                finally:
                    con.close()
                self.at = time.monotonic()
            return self.body


def make_handler(data_dir: Path, quiet: bool = False, live_db: Path | None = None, web: Path = WEB):
    live = LiveCache(live_db) if live_db is not None else None

    class Handler(BaseHTTPRequestHandler):
        server_version = "advantage/1"

        def do_GET(self):
            self._serve(body=True)

        def do_HEAD(self):
            self._serve(body=False)

        def _send(self, status, ctype: str, data: bytes, body: bool, headers: dict | None = None):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            if body:
                self.wfile.write(data)

        def _serve(self, body: bool):
            if urlsplit(self.path).path == "/api/live":
                return self._live(body)
            hit = resolve(data_dir, self.path, web)
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
            h = {"ETag": etag, "Last-Modified": formatdate(st.st_mtime, usegmt=True), "Cache-Control": "no-cache"}
            if gz:
                h["Content-Encoding"] = "gzip"
            self._send(HTTPStatus.OK, ctype, data, body, h)

        def _live(self, body: bool):
            if live is None or not live.path.is_file():
                msg = json.dumps({"error": "no live.db: the Kalshi watch (scripts/watch.py run) has not run here"})
                return self._send(HTTPStatus.SERVICE_UNAVAILABLE, "application/json", msg.encode(), body)
            try:
                data = live.get()
            except Exception as e:                     # locked or mid-migration: say so, the page retries
                msg = json.dumps({"error": f"live.db unreadable: {type(e).__name__}: {e}"})
                return self._send(HTTPStatus.SERVICE_UNAVAILABLE, "application/json", msg.encode(), body)
            self._send(HTTPStatus.OK, "application/json; charset=utf-8", data, body, {"Cache-Control": "no-store"})

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
    ap.add_argument("--live-db", type=Path, default=LIVE_DB, help="the watch's live.db")
    ap.add_argument("--quiet", action="store_true", help="no request log")
    args = ap.parse_args(argv)
    data_dir = args.out / f"run{args.run}"
    if not (data_dir / "index.json").is_file():
        print(f"warning: {data_dir} has no export yet (python scripts/ratings_export.py --run {args.run})")
    if not args.live_db.is_file():
        print(f"warning: {args.live_db} not found: the live page will say so until the watch runs")
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(data_dir, args.quiet, args.live_db))
    print(f"live board and ratings viewer (run {args.run}) on http://{args.host}:{args.port}/")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
