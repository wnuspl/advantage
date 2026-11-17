"""Serve the web pages: the live Kalshi board (web/live.html, from the watch's live.db), the
scorecard and paper P&L (web/scorecard.html, web/pnl.html, same live.db) and the ratings
viewer (web/ratings.html, from scripts/ratings_export.py's weekly standings).
Standard library only, read-only: nothing here talks to Kalshi or writes a database.

    python scripts/web_server.py                         # http://127.0.0.1:8790/
    python scripts/web_server.py --host 0.0.0.0 --port 8790 --run 790 --live-db live.db

Routes:
    /, /live              web/live.html
    /ratings              web/ratings.html
    /scorecard            web/scorecard.html
    /pnl                  web/pnl.html
    /nav.js               the shared navigation bar (web/nav.js)
    /api/live             engine.livefeed.live_view(live.db) as JSON (recomputed at most every few s);
                          ?bankroll=&kelly=&mw= re-size the bets for the viewer (read-only:
                          nothing is stored, the watch and its alerts keep their own policy)
    /api/match?event=<ticker>   engine.livefeed.match_view(): one match's every poll, its
                          break / set-end marks and pre-match block (400 bad ticker, 404 unseen)
    /api/scorecard?tour=ATP&tour=WTA   engine.scorecard.scorecard_view (model vs market log loss,
                          feed lag, running difference); 400 on an unknown tour
    /api/pnl?bankroll=&kelly=   engine.paper.pnl_view (paper replays of the flagged bets and
                          the sell rules; bankroll default $1000 like paper_replay.py, Kelly
                          default each watch's own); 400 if malformed
                          Both are slow on a big live.db: computed in one background thread
                          (SlowCache), refreshed when a new settlement appears or every
                          `max_age` s, never more often than `min_gap` s; until the first result
                          is ready they answer 202 {"computing": true}. The live page never waits.
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
from urllib.parse import parse_qs, unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tennis.engine import livefeed, paper, scorecard                  # noqa: E402

WEB = ROOT / "web"
OUT = ROOT / "models" / "ratings"
LIVE_DB = ROOT / "live.db"
HTML = "text/html; charset=utf-8"
PAGES = {"/": ("live.html", HTML), "/index.html": ("live.html", HTML), "/live": ("live.html", HTML),
         "/live.html": ("live.html", HTML), "/ratings": ("ratings.html", HTML),
         "/ratings.html": ("ratings.html", HTML), "/scorecard": ("scorecard.html", HTML),
         "/scorecard.html": ("scorecard.html", HTML), "/pnl": ("pnl.html", HTML), "/pnl.html": ("pnl.html", HTML),
         "/nav.js": ("nav.js", "text/javascript; charset=utf-8")}


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


QUERY = {"bankroll": "bankroll", "kelly": "kelly", "mw": "model_weight"}


def sizing_overrides(query: str) -> dict:
    """?bankroll=100&kelly=0.25&mw=0.5 -> {'bankroll': 100.0, ...}; ValueError if malformed."""
    out = {}
    for k, v in parse_qs(query).items():
        if k in QUERY:
            x = float(v[-1])
            if x != x or x in (float("inf"), float("-inf")):
                raise ValueError(k)
            out[QUERY[k]] = x
    return out


PNL_DEFAULT_BANKROLL = 1000.0           # scripts/paper_replay.py --bankroll default


def pnl_overrides(query: str) -> tuple[float, float | None]:
    """?bankroll=500&kelly=0.1 -> (500.0, 0.1), clamped as policy_with clamps them; bankroll
    default $1000, kelly default None (each watch's own). ValueError if malformed."""
    ov = {k: v for k, v in sizing_overrides(query).items() if k in ("bankroll", "kelly")}
    clamp = lambda k, v: min(max(v, livefeed.OVERRIDABLE[k][0]), livefeed.OVERRIDABLE[k][1])   # noqa: E731
    return (clamp("bankroll", ov.get("bankroll", PNL_DEFAULT_BANKROLL)),
            clamp("kelly", ov["kelly"]) if "kelly" in ov else None)


TOURS = sorted(set(livefeed.TOUR_LABEL.values()))


def tour_filter(query: str) -> tuple[str, ...]:
    """?tour=ATP&tour=WTA (or ?tour=ATP,WTA) -> ('ATP', 'WTA'); () = all. ValueError if unknown."""
    out = set()
    for v in parse_qs(query).get("tour", []):
        for t in v.split(","):
            t = t.strip()
            if t:
                if t not in TOURS:
                    raise ValueError(t)
                out.add(t)
    return tuple(sorted(out))


def settlement_signature(con) -> tuple:
    """Changes when a match settles or a new watch starts: what the slow views depend on
    (besides the unsettled-bet count, which max_age refreshes)."""
    return (tuple(con.execute("SELECT COUNT(*), MAX(seen_at) FROM settlement").fetchone())
            + tuple(con.execute("SELECT MAX(watch_id) FROM watch").fetchone()))


class SlowCache:
    """A slow read of live.db (`compute(con, key) -> dict`), computed off the request thread.

    get(key) returns at once with the newest result, possibly old, and queues a refresh when
    one is due: the result is at least `min_gap` s old and either a match settled since
    (settlement_signature) or it is older than `max_age`. One worker thread computes, one
    key at a time, so viewers with different settings queue instead of running replays in
    parallel on a small server. At most `keep` keys are kept. With no result yet, get()
    waits up to `wait` s, then returns None (the route answers 202 and the page retries)."""

    def __init__(self, path: Path, compute, min_gap: float = 300.0, max_age: float = 1800.0,
                 keep: int = 6, clock=time.monotonic):
        self.path, self.compute = Path(path), compute
        self.min_gap, self.max_age, self.keep, self.clock = min_gap, max_age, keep, clock
        self.cv = threading.Condition()
        self.cache: dict = {}          # key -> {"at", "sig", "body", "seconds", "computed_at", "error"}
        self.queue: list = []
        self.worker: threading.Thread | None = None
        self.runs = 0                  # computations done (tests)

    def _due(self, hit) -> bool:
        age = self.clock() - hit["at"]
        if age < self.min_gap:
            return False
        if age >= self.max_age or hit["error"]:
            return True
        con = livefeed.connect(self.path)
        try:
            return settlement_signature(con) != hit["sig"]
        finally:
            con.close()

    def _enqueue(self, key) -> None:
        if key not in self.queue:
            self.queue.append(key)
            self.cv.notify_all()
        if self.worker is None or not self.worker.is_alive():
            self.worker = threading.Thread(target=self._work, name="slow-cache", daemon=True)
            self.worker.start()

    def _work(self) -> None:
        while True:
            with self.cv:
                if not self.queue:
                    self.worker = None
                    return
                key = self.queue[0]
            t0 = time.perf_counter()
            entry = {"error": None, "body": None, "sig": None}
            try:
                con = livefeed.connect(self.path)
                try:
                    sig = settlement_signature(con)
                    body = json.dumps(self.compute(con, key), separators=(",", ":"), allow_nan=False)
                finally:
                    con.close()
                entry.update(sig=sig, body=body.encode("utf-8"))
            except Exception as e:     # locked, mid-migration, a bug: keep the old body, say why
                entry["error"] = f"{type(e).__name__}: {e}"
            with self.cv:
                old = self.cache.get(key)
                if entry["body"] is None and old:
                    entry["body"] = old["body"]
                entry.update(at=self.clock(), seconds=round(time.perf_counter() - t0, 3),
                             computed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
                if key not in self.cache and len(self.cache) >= self.keep:
                    self.cache.pop(min(self.cache, key=lambda k: self.cache[k]["at"]))
                self.cache[key] = entry
                self.queue.remove(key)
                self.runs += 1
                self.cv.notify_all()

    def get(self, key, wait: float = 0.0) -> dict | None:
        """The entry (body bytes, computed_at, seconds, error, refreshing) or None if not ready."""
        with self.cv:
            hit = self.cache.get(key)
        due = hit is None or self._due(hit)            # reads live.db outside the lock
        with self.cv:
            if due:
                self._enqueue(key)
            if hit is None and wait > 0:
                self.cv.wait_for(lambda: key in self.cache, timeout=wait)
                hit = self.cache.get(key)
            if hit is None:
                return None
            return {**hit, "refreshing": key in self.queue}


def wrap(entry: dict) -> bytes:
    """A SlowCache entry as the API body: {"computed_at", "seconds", "refreshing", "error", "data"}."""
    meta = json.dumps({"computed_at": entry["computed_at"], "seconds": entry["seconds"],
                       "refreshing": entry["refreshing"], "error": entry["error"]}, separators=(",", ":"))
    return meta[:-1].encode() + b',"data":' + (entry["body"] or b"null") + b"}"


def _scorecard(con, key):
    return scorecard.scorecard_view(con, set(key) or None)


def _pnl(con, key):
    return paper.pnl_view(con, bankroll=key[0], kelly=key[1])


class LiveCache:
    """live_view() of live.db per sizing, recomputed when older than `ttl` seconds (the watch
    polls every 20-60 s; open tabs with the same settings share one computation)."""

    def __init__(self, path: Path, ttl: float = 5.0, keep: int = 16):
        self.path, self.ttl, self.keep = Path(path), ttl, keep
        self.lock = threading.Lock()
        self.cache: dict[tuple, tuple[float, bytes]] = {}

    def get(self, overrides: dict | None = None) -> bytes:
        key = tuple(sorted((overrides or {}).items()))
        with self.lock:
            hit = self.cache.get(key)
            if hit and time.monotonic() - hit[0] <= self.ttl:
                return hit[1]
            con = livefeed.connect(self.path)
            try:
                body = json.dumps(livefeed.live_view(con, overrides=overrides or None),
                                  separators=(",", ":")).encode("utf-8")
            finally:
                con.close()
            if len(self.cache) >= self.keep:
                self.cache.pop(min(self.cache, key=lambda k: self.cache[k][0]))
            self.cache[key] = (time.monotonic(), body)
            return body


class MatchCache:
    """livefeed.match_view() of one event, recomputed when older than `ttl` seconds; at most
    `keep` events cached. get() returns None for an event live.db has never seen."""

    def __init__(self, path: Path, ttl: float = 5.0, keep: int = 32):
        self.path, self.ttl, self.keep = Path(path), ttl, keep
        self.lock = threading.Lock()
        self.cache: dict[str, tuple[float, bytes | None]] = {}

    def get(self, event: str) -> bytes | None:
        if not livefeed.valid_event(event):
            raise ValueError(event)
        with self.lock:
            hit = self.cache.get(event)
            if hit and time.monotonic() - hit[0] <= self.ttl:
                return hit[1]
            con = livefeed.connect(self.path)
            try:
                v = livefeed.match_view(con, event)
            finally:
                con.close()
            body = None if v is None else json.dumps(v, separators=(",", ":")).encode("utf-8")
            if len(self.cache) >= self.keep and event not in self.cache:
                self.cache.pop(min(self.cache, key=lambda k: self.cache[k][0]))
            self.cache[event] = (time.monotonic(), body)
            return body


def make_handler(data_dir: Path, quiet: bool = False, live_db: Path | None = None, web: Path = WEB,
                 slow: dict | None = None, wait: float = 2.0):
    """`slow`: {"scorecard": SlowCache, "pnl": SlowCache} to share or tune (tests); `wait`:
    how long a request waits for a first slow result before answering 202."""
    live = LiveCache(live_db) if live_db is not None else None
    match = MatchCache(live_db) if live_db is not None else None
    if slow is None and live_db is not None:
        slow = {"scorecard": SlowCache(live_db, _scorecard, min_gap=60, max_age=600),
                "pnl": SlowCache(live_db, _pnl, min_gap=300, max_age=1800)}

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
            if urlsplit(self.path).path == "/api/match":
                return self._match(body)
            if urlsplit(self.path).path in ("/api/scorecard", "/api/pnl"):
                return self._slow(body)
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
                overrides = sizing_overrides(urlsplit(self.path).query)
            except ValueError:
                msg = json.dumps({"error": "bad sizing: bankroll, kelly and mw must be numbers"})
                return self._send(HTTPStatus.BAD_REQUEST, "application/json", msg.encode(), body)
            try:
                data = live.get(overrides)
            except Exception as e:                     # locked or mid-migration: say so, the page retries
                msg = json.dumps({"error": f"live.db unreadable: {type(e).__name__}: {e}"})
                return self._send(HTTPStatus.SERVICE_UNAVAILABLE, "application/json", msg.encode(), body)
            self._send(HTTPStatus.OK, "application/json; charset=utf-8", data, body, {"Cache-Control": "no-store"})

        def _match(self, body: bool):
            def err(status, text):
                self._send(status, "application/json", json.dumps({"error": text}).encode(), body)
            if match is None or not match.path.is_file():
                return err(HTTPStatus.SERVICE_UNAVAILABLE, "no live.db: the Kalshi watch (scripts/watch.py run) has not run here")
            ev = parse_qs(urlsplit(self.path).query).get("event", [""])[-1]
            if not livefeed.valid_event(ev):
                return err(HTTPStatus.BAD_REQUEST, "bad event: a Kalshi event ticker, e.g. KXATPMATCH-26NOV16ALCTIA")
            try:
                data = match.get(ev)
            except Exception as e:
                return err(HTTPStatus.SERVICE_UNAVAILABLE, f"live.db unreadable: {type(e).__name__}: {e}")
            if data is None:
                return err(HTTPStatus.NOT_FOUND, f"{ev}: not in live.db")
            self._send(HTTPStatus.OK, "application/json; charset=utf-8", data, body, {"Cache-Control": "no-store"})

        def _slow(self, body: bool):
            def err(status, text):
                self._send(status, "application/json", json.dumps({"error": text}).encode(), body)
            if slow is None or not Path(live_db).is_file():
                return err(HTTPStatus.SERVICE_UNAVAILABLE, "no live.db: the Kalshi watch (scripts/watch.py run) has not run here")
            path, query = urlsplit(self.path).path, urlsplit(self.path).query
            try:
                key = tour_filter(query) if path == "/api/scorecard" else pnl_overrides(query)
            except ValueError:
                return err(HTTPStatus.BAD_REQUEST, f"bad tour: one of {', '.join(TOURS)}" if path == "/api/scorecard"
                           else "bad sizing: bankroll and kelly must be numbers")
            try:
                entry = slow[path[len("/api/"):]].get(key, wait)
            except Exception as e:
                return err(HTTPStatus.SERVICE_UNAVAILABLE, f"live.db unreadable: {type(e).__name__}: {e}")
            if entry is None:
                msg = json.dumps({"computing": True, "retry_s": 3}).encode()
                return self._send(HTTPStatus.ACCEPTED, "application/json", msg, body, {"Cache-Control": "no-store"})
            if entry["body"] is None:
                return err(HTTPStatus.SERVICE_UNAVAILABLE, f"live.db unreadable: {entry['error']}")
            self._send(HTTPStatus.OK, "application/json; charset=utf-8", wrap(entry), body, {"Cache-Control": "no-store"})

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
    print(f"live board, scorecard, P&L and ratings viewer (run {args.run}) on http://{args.host}:{args.port}/")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
